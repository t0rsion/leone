#!/usr/bin/env python3
"""Validate Runtime shared-prefix receipts against criteria frozen from calibration.

The module also holds the loading, scoring, and criteria derivation that
`freeze_runtime_criteria.py` shares. A validation recomputes the criteria from
the calibration receipts and requires an exact match before it scores any
evaluation receipt.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import csv
import hashlib
import json
import math
import struct
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import common_oracle
import run_llama_cached_comparator as cached
from gpu_receipt import finite, percentile95, require
from runner_support import publish_json

RESEARCH = Path(__file__).resolve().parent
ROOT = RESEARCH.parents[1]
MANIFEST = RESEARCH / "runtime_manifest.json"
RECEIPT_SCHEMA = "prefix-attention-runtime-receipt-v1"
CRITERIA_SCHEMA = "prefix-attention-runtime-criteria-v1"
REPORT_SCHEMA = "prefix-attention-runtime-report-v1"
PATHS = (
    "per_row", "fixed_tile_per_row", "shared_read_unconstrained",
    "shared_read_fixed_reduction",
)
CONTROL = "fixed_tile_per_row"
CANDIDATE = "shared_read_fixed_reduction"
SHARED_PATHS = ("shared_read_unconstrained", "shared_read_fixed_reduction")
TOOLS = (
    "freeze_runtime_criteria.py", "validate_runtime.py", "runner_support.py",
    "run_llama_cached_comparator.py", "common_oracle.py", "gpu_receipt.py", "generate_manifest.py",
)
ROW_HEADER = struct.Struct("<IIQI")
MAX_VOCAB = 1 << 20
MAX_SIDECAR_ROWS = 4096
REQUIRED_EVALUATION_CLASSES = ("short_prefix_control", "unrelated_control")
KLD_DEFINITION = (
    "mean over scored rows of KL(P_reference || P_path) in nats, "
    "full softmax over full vocabulary"
)
NUMERIC_RULE = (
    "each non-control path is scored against fixed_tile_per_row on the same rows, and the control "
    "is scored against its own other repetitions; an evaluation metric must not exceed the "
    "largest calibration value plus one representable FP64 step"
)
FASTER = "faster_than_control_beyond_review_threshold"
SLOWER = "slower_than_control_beyond_review_threshold"
MIXED = "mixed_beyond_review_threshold"
WITHIN = "within_review_threshold"
TIMING_RULE = (
    "new for the runtime study: the timed value is setup_ms plus decode_ms per repetition; "
    "the review threshold is the nearest-rank 95th percentile of abs(sample - median) / median "
    "over every path, case, and repetition, plus one representable FP64 step, where the median "
    "is the element at index n // 2 of the sorted samples, as the driver computes it; the signed "
    "effect is 1 - candidate median / control median per case. An effect at or above the "
    "threshold labels the case faster, an effect at or below its negative labels it slower, and "
    "any other effect labels it within the threshold. A run without a bound GPU sample record "
    "appends _unmonitored to every label. The spread covers repetitions inside one process. It "
    "excludes drift between processes, and each path ran in its own process, so a difference that "
    "clears it supports no speed claim. The labels are review flags"
)
TIMING_ORIGIN = (
    "runtime_manifest.json and gpu_manifest.json declare no practical effect rule; "
    "gpu_receipt.TIMING_RULE needs paired baseline, candidate, baseline samples that runtime runs lack"
)
BF16_RULE = (
    "each path and the control are scored against the BF16 oracle rows. Only the difference between "
    "a non-control path and the control on the same rows gates. The limit for each metric is the "
    "largest calibration difference, floored at zero, plus one representable FP64 step. Absolute "
    "distances are report only. Calibration holds four short-context cases, so the limits describe "
    "that sample only"
)
BF16_LIMIT = (
    "the oracle is the llama.cpp CPU run of the pinned BF16 file. A shared original checkpoint for the "
    "BF16 and Q4 files is not established, so every absolute distance includes quantization drift and "
    "the difference between two implementations. The report makes no end-to-end quality claim"
)
DIGEST_SCOPE = (
    "calibration receipts: logits_digest and raw_logit_position_digest are recomputed from the "
    "sidecar of each repetition, because every step is captured and no setup event exists. "
    "Evaluation receipts: not recomputed. Setup events write sidecar rows that the digests exclude, "
    "and the receipts do not mark which rows they are, so the sidecar hashes and header identities "
    "are the only linkage. token_digest is not recomputed in either phase"
)
PATH_ROLES = {
    "per_row": "differential reference: reduction order differs from the control by design",
    "fixed_tile_per_row": "control: scored against its own repetitions",
    "shared_read_unconstrained": "expected to differ: reduction order is unconstrained",
    "shared_read_fixed_reduction": "candidate: fixed reduction order equal to the control",
}
UNBOUND_MONITORING = {
    "bound": False,
    "limit": "no externally sampled clock or power record is bound; clocks, power, and "
             "temperature during each path are unrecorded",
}
REUSE_RULE = (
    "per_row and fixed_tile_per_row count no multi-row group; shared paths count "
    "none on unrelated cases and at least one on shared cases"
)
BUILD_FIELDS = (
    "features", "profile", "target", "host", "rustc", "rustc_sha256",
    "toolchain_sha256", "tool_versions", "native_provenance", "native_tools_sha256",
    "native_tool_versions", "build_config_sha256", "build_flags_sha256",
    "build_flags_raw_sha256", "profile_inputs_sha256", "linker_inputs_sha256",
    "source_tree_sha256", "source_tree_dirty", "provenance_unknown", "native_controls",
)
RECEIPT_FIELDS = (
    "backend", "model_name", "model_sha256", "fixture_sha256", "source_manifest_sha256",
    "source_plan_sha256", "source_subject_sha256", "source_phase", "claim_scope", "device",
)
CASE_FIELDS = (
    "name", "source_case", "topology", "input_digest", "decode_steps", "branch_count",
    "retained_branch_count", "raw_logit_vocab", "raw_logit_rows", "sidecar_logit_rows",
    "logit_bindings", "batch_schedule",
)
SAMPLE_FIELDS = (
    "setup_ms", "decode_ms", "logits_digest", "raw_logit_position_digest", "token_digest",
    "logits_sidecars", "setup_stats", "stats", "memory_before", "memory_after_setup",
    "memory_after_decode", "memory_after_retire",
)
METRICS = ("max_abs_logit_diff", "max_kld", "mean_kld")
MONITORING_COLUMNS = (
    "clocks.current.sm [MHz]", "clocks.current.memory [MHz]", "power.draw [W]", "temperature.gpu",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def tool_hashes() -> dict[str, str]:
    return {name: sha256_file(RESEARCH / name) for name in TOOLS}


@dataclass(frozen=True)
class Run:
    path: str
    file: Path
    file_sha256: str
    receipt: dict[str, Any]


@dataclass(frozen=True)
class Sidecar:
    file: Path
    sha256: str
    rows: int
    vocab: int


@dataclass(frozen=True)
class Study:
    runs: dict[str, Run]
    identity: dict[str, Any]
    cases: list[dict[str, Any]]
    sidecars: dict[str, dict[str, list[Sidecar]]]
    keys: dict[str, list[tuple]]


def load_run(file: Path, phase: str, backend: str) -> Run:
    receipt = json.loads(file.read_text())
    require(receipt.get("schema") == RECEIPT_SCHEMA, f"{file.name}: wrong receipt schema")
    require(receipt.get("phase") == phase, f"{file.name}: phase is not {phase}")
    require(receipt.get("backend") == backend, f"{file.name}: backend is not {backend}")
    require(receipt.get("quality") == "unverified", f"{file.name}: raw quality must be unverified")
    require(receipt.get("path") in PATHS, f"{file.name}: unknown path")
    require(isinstance(receipt.get("cases"), list) and receipt["cases"], f"{file.name}: no cases")
    return Run(receipt["path"], file, sha256_file(file), receipt)


def load_runs(files: list[Path], phase: str, backend: str) -> dict[str, Run]:
    runs = [load_run(file, phase, backend) for file in files]
    names = sorted(run.path for run in runs)
    require(names == sorted(PATHS), f"receipts must cover each path once, got {names}")
    return {run.path: run for run in sorted(runs, key=lambda run: PATHS.index(run.path))}


def flag_value(command: list[str], flag: str) -> int:
    require(flag in command and command.index(flag) + 1 < len(command), f"command lacks {flag}")
    return int(command[command.index(flag) + 1])


def run_identity(run: Run) -> dict[str, Any]:
    receipt, provenance = run.receipt, run.receipt["provenance"]
    build = provenance["build"]
    identity = {field: receipt[field] for field in RECEIPT_FIELDS}
    identity.update({
        "manifest_sha256": provenance["manifest_sha256"],
        "source_sha256": provenance["source_sha256"],
        "build": {field: build.get(field) for field in BUILD_FIELDS},
        "affinity": provenance["affinity"],
        "host_system": provenance["host_system"],
        "repetitions": flag_value(provenance["command"], "--repetitions"),
        "warmups": flag_value(provenance["command"], "--warmups"),
    })
    return identity


def common_identity(runs: dict[str, Run]) -> dict[str, Any]:
    identities = {path: run_identity(run) for path, run in runs.items()}
    first = identities[PATHS[0]]
    for path, identity in identities.items():
        differing = sorted(key for key in first if identity[key] != first[key])
        require(not differing, f"{path}: identity differs from {PATHS[0]} in {differing}")
    build = first["build"]
    require(build["source_tree_dirty"] is False and build["provenance_unknown"] is False,
            "receipts report a dirty tree or unknown build provenance for the tracked build inputs")
    require(first["model_sha256"] == first["source_subject_sha256"], "receipts ran a model other than the source subject")
    return first


def check_case(case: dict[str, Any], repetitions: int) -> None:
    name = case.get("name")
    for field in SAMPLE_FIELDS:
        require(isinstance(case.get(field), list) and len(case[field]) == repetitions,
                f"{name}: {field} does not hold {repetitions} repetitions")
    for field in ("setup_ms", "decode_ms"):
        for value in case[field]:
            finite(value, f"{name}.{field}", positive=True)
        require(case[f"{field[:-3]}_median_ms"] == median(case[field]), f"{name}: {field} median differs")
    require(all(stats == case["stats"][0] for stats in case["stats"]), f"{name}: counters vary across repetitions")
    require(case["sidecar_logit_rows"] >= 1 and case["logit_bindings"], f"{name}: no scored rows")


def common_cases(runs: dict[str, Run], repetitions: int) -> list[dict[str, Any]]:
    control = runs[CONTROL].receipt["cases"]
    for path, run in runs.items():
        require(len(run.receipt["cases"]) == len(control), f"{path}: case count differs")
        for case, expected in zip(run.receipt["cases"], control):
            check_case(case, repetitions)
            differing = [field for field in CASE_FIELDS if case[field] != expected[field]]
            require(not differing, f"{path}/{case['name']}: case identity differs in {differing}")
    return [{field: case[field] for field in CASE_FIELDS} for case in control]


def sidecar_file(run: Run, record: dict[str, str]) -> Path:
    relative = Path(record["file"])
    require(not relative.is_absolute() and ".." not in relative.parts, "sidecar path leaves the receipt directory")
    file = run.file.parent / relative
    require(file.is_file() and not file.is_symlink(), f"sidecar is not a regular file: {relative}")
    require(file.resolve().is_relative_to(run.file.parent.resolve()), "sidecar resolves outside the receipt directory")
    return file


def check_sidecar(file: Path, record: dict[str, str], rows: int, vocab: int) -> None:
    require(1 <= vocab <= MAX_VOCAB and 1 <= rows <= MAX_SIDECAR_ROWS, "sidecar bounds exceeded")
    size = rows * (ROW_HEADER.size + 4 * vocab)
    require(file.stat().st_size == size, f"{file.name}: size differs from {rows} declared rows")
    require(sha256_file(file) == record["sha256"], f"{file.name}: hash differs from the receipt")


def distinct_sidecars(run: Run, case: dict[str, Any]) -> list[Sidecar]:
    seen: dict[str, Sidecar] = {}
    for record in case["logits_sidecars"]:
        file = sidecar_file(run, record)
        check_sidecar(file, record, case["sidecar_logit_rows"], case["raw_logit_vocab"])
        seen.setdefault(record["sha256"], Sidecar(
            file, record["sha256"], case["sidecar_logit_rows"], case["raw_logit_vocab"]))
    return list(seen.values())


def read_headers(sidecar: Sidecar) -> Iterator[tuple[int, int, int]]:
    stride = ROW_HEADER.size + 4 * sidecar.vocab
    with sidecar.file.open("rb") as source:
        for row in range(sidecar.rows):
            source.seek(row * stride)
            sequence, index, position, vocab = ROW_HEADER.unpack(source.read(ROW_HEADER.size))
            require(vocab == sidecar.vocab, f"{sidecar.file.name}: row {row} vocabulary differs")
            yield sequence, index, position


def read_rows(sidecar: Sidecar) -> Iterator[array.array]:
    with sidecar.file.open("rb") as source:
        for _ in range(sidecar.rows):
            source.read(ROW_HEADER.size)
            yield read_values(source, sidecar.vocab)


def read_values(source: Any, vocab: int) -> array.array:
    values = array.array("f")
    values.fromfile(source, vocab)
    if sys.byteorder == "big":
        values.byteswap()
    require(all(map(math.isfinite, values)), "logit row holds a nonfinite value")
    return values


def scored_keys(sidecar: Sidecar, bindings: list[dict[str, Any]]) -> list[tuple]:
    """Bind each sidecar row to one logit binding. Rows must appear once, in binding order."""
    index = {(b["sequence"], b["token_index"], b["predicted_position"]): n for n, b in enumerate(bindings)}
    require(len(index) == len(bindings), "logit bindings repeat a row identity")
    keys, last = [], -1
    for header in read_headers(sidecar):
        found = index.get(header)
        require(found is not None and found > last,
                f"{sidecar.file.name}: row {header} is absent, repeated, or reordered against the bindings")
        last = found
        binding = bindings[found]
        keys.append((binding["event"], binding["sequence"], binding["token_index"], binding["predicted_position"]))
    return keys


def load_study(files: list[Path], phase: str, backend: str) -> Study:
    runs = load_runs(files, phase, backend)
    identity = common_identity(runs)
    cases = common_cases(runs, identity["repetitions"])
    sidecars = {
        path: {case["name"]: distinct_sidecars(run, case) for case in run.receipt["cases"]}
        for path, run in runs.items()
    }
    keys = {}
    for case in cases:
        keys[case["name"]] = scored_keys(sidecars[CONTROL][case["name"]][0], case["logit_bindings"])
        for path in PATHS:
            for sidecar in sidecars[path][case["name"]]:
                same = scored_keys(sidecar, case["logit_bindings"]) == keys[case["name"]]
                require(same, f"{path}/{case['name']}: scored rows differ from the control")
    return Study(runs, identity, cases, sidecars, keys)


def read_tokens(path: Path) -> list[int]:
    tokens = array.array("I")
    require(tokens.itemsize == 4, "platform unsigned int is not 32 bits")
    data = path.read_bytes()
    require(data and len(data) % 4 == 0, "token fixture is not a nonempty u32 stream")
    tokens.frombytes(data)
    if sys.byteorder == "big":
        tokens.byteswap()
    return list(tokens)


def window(tokens: list[int], offset: int, count: int) -> list[int]:
    require(0 <= offset and offset + count <= len(tokens), f"fixture range {offset}+{count} is out of bounds")
    return tokens[offset:offset + count]


def logical_prompt(case: dict[str, Any], branch: int, tokens: list[int]) -> list[int]:
    tail = window(tokens, case["tail_offsets"][branch], case["tail_tokens"][branch])
    if case["topology"] == "fork_of_fork":
        prefix = window(tokens, case["prefix_offsets"][0], case["partial_prefix_tokens"])
        return prefix + window(tokens, case["common_tail_offset"], case["common_tail_tokens"]) + tail
    return window(tokens, case["prefix_offsets"][branch], case["prefix_tokens"][branch]) + tail


def calibration_digest(case: dict[str, Any], tokens: list[int]) -> str:
    """Recompute the driver's input digest from the manifest and the fixture."""
    branches = range(len(case["prefix_offsets"]))
    digest = hashlib.sha256()
    for branch in branches:
        digest.update(struct.pack(f"<{len(logical_prompt(case, branch, tokens))}I", *logical_prompt(case, branch, tokens)))
    for step in range(case["decode_steps"]):
        for branch in branches:
            digest.update(struct.pack("<I", window(tokens, case["teacher_offsets"][branch][step], 1)[0]))
    return digest.hexdigest()


def calibration_bindings(case: dict[str, Any], tokens: list[int]) -> list[tuple]:
    require(not case.get("active_sequences") and not case.get("setup_events"),
            f"{case['name']}: calibration cases with setup events or membership are unsupported")
    sequences = case.get("branch_sequences") or list(range(len(case["prefix_offsets"])))
    lengths = [len(logical_prompt(case, branch, tokens)) for branch in range(len(sequences))]
    return [
        ("decode_stepwise", sequences[branch], step, lengths[branch] + step + 1)
        for step in range(case["decode_steps"]) for branch in range(len(sequences))
    ]


def check_calibration_manifest(study: Study, manifest: dict[str, Any], tokens: list[int]) -> None:
    expected = {case["name"]: case for case in manifest["calibration"]}
    require([case["name"] for case in study.cases] == list(expected), "calibration cases differ from the manifest")
    for case in study.cases:
        spec, name = expected[case["name"]], case["name"]
        require(case["input_digest"] == calibration_digest(spec, tokens), f"{name}: input digest differs from the manifest")
        rows = calibration_bindings(spec, tokens)
        bound = [(b["event"], b["sequence"], b["token_index"], b["predicted_position"]) for b in case["logit_bindings"]]
        require(bound == rows, f"{name}: bindings differ from the manifest")
        require(all(b["input_position"] + 1 == b["predicted_position"] for b in case["logit_bindings"]),
                f"{name}: input and predicted positions disagree")
        captured = [row for row in rows if row[2] in spec["capture_steps"]]
        require(study.keys[name] == captured, f"{name}: scored rows differ from the captured steps")


def source_rows(workload: cached.Workload, raw: dict[str, Any]) -> dict[str, list[tuple]]:
    """Canonical output rows per case, derived from the source workload alone."""
    rows = {}
    for index, case in enumerate(workload.cases):
        stepwise = {op["name"] for op in raw["cases"][index]["operations"] if op["kind"] == "decode_stepwise"}
        rows[case.name] = [
            canonical_key(case.events[event].name, row, stepwise)
            for event in range(len(case.events))
            for row in cached.expected_logits(index, event, case.events[event])
        ]
    return rows


def canonical_key(name: str, row: dict[str, int], stepwise: set[str]) -> tuple:
    base, _, step = name.rpartition("_")
    if base in stepwise and step.isdigit():
        return (base, row["seq"], int(step), row["predicted_position"])
    return (name, row["seq"], row["token_index"], row["predicted_position"])


def check_rows(study: Study, rows: dict[str, list[tuple]]) -> None:
    require([case["name"] for case in study.cases] == list(rows), "study cases differ from the workload cases")
    for case in study.cases:
        name = case["name"]
        require(study.keys[name] == rows[name], f"{name}: scored rows differ from the workload rows")


def check_evaluation_rows(study: Study, rows: dict[str, list[tuple]], criteria: dict[str, Any]) -> None:
    require([case["name"] for case in study.cases] == criteria["workload"]["evaluation_cases"],
            "evaluation cases differ from the frozen source workload")
    check_rows(study, rows)
    for name, keys in rows.items():
        require(sha256_json(keys) == criteria["workload"]["row_identity_sha256"][name],
                f"{name}: source workload rows differ from the frozen identity")


def log_softmax(values: array.array) -> list[float]:
    peak = max(values)
    shift = peak + math.log(sum(math.exp(value - peak) for value in values))
    return [value - shift for value in values]


def row_metrics(reference: array.array, other: array.array) -> tuple[float, float, bool]:
    log_p, log_q = log_softmax(reference), log_softmax(other)
    kld = sum(math.exp(p) * (p - q) for p, q in zip(log_p, log_q))
    kld = 0.0 if -1e-12 <= kld <= 0.0 else kld
    require(math.isfinite(kld) and kld >= 0.0, "row KLD is invalid")
    largest = max(abs(a - b) for a, b in zip(reference, other))
    return kld, largest, reference.index(max(reference)) == other.index(max(other))


def score_rows(pairs: Iterator[tuple[array.array, array.array]]) -> dict[str, Any]:
    rows, klds, largest, top1, bitwise = 0, [], 0.0, 0, True
    for reference, other in pairs:
        kld, diff, agree = row_metrics(reference, other)
        rows, largest, top1 = rows + 1, max(largest, diff), top1 + agree
        bitwise = bitwise and reference.tobytes() == other.tobytes()
        klds.append(kld)
    return {
        "rows": rows, "max_abs_logit_diff": largest, "mean_kld": sum(klds) / rows,
        "max_kld": max(klds), "top1_agreement_rows": top1, "bitwise_equal": bitwise,
    }


def identical_score(rows: int) -> dict[str, Any]:
    return {"rows": rows, "max_abs_logit_diff": 0.0, "mean_kld": 0.0, "max_kld": 0.0,
            "top1_agreement_rows": rows, "bitwise_equal": True}


def score_sidecars(reference: Sidecar, other: Sidecar) -> dict[str, Any]:
    if reference.sha256 == other.sha256:
        return identical_score(reference.rows)
    return score_rows(zip(read_rows(reference), read_rows(other)))


def worst(scores: list[dict[str, Any]]) -> dict[str, Any]:
    result = {key: max(score[key] for score in scores) for key in METRICS}
    result.update({
        "rows": scores[0]["rows"],
        "top1_agreement_rows": min(score["top1_agreement_rows"] for score in scores),
        "bitwise_equal": all(score["bitwise_equal"] for score in scores),
    })
    return result


def path_scores(study: Study, name: str, path: str) -> dict[str, Any]:
    """Score one path against the first control sidecar. The control meets its other repetitions."""
    sidecars = study.sidecars[path][name]
    reference = study.sidecars[CONTROL][name][0]
    others = sidecars[1:] if path == CONTROL else sidecars
    if not others:
        return identical_score(reference.rows)
    return worst([score_sidecars(reference, sidecar) for sidecar in others])


def control_numerics(study: Study) -> dict[str, dict[str, dict[str, Any]]]:
    """Score every path against the control on the same rows, per case.

    The control entry holds its repeatability: the worst score of its other distinct repetitions.
    """
    return {
        case["name"]: {path: path_scores(study, case["name"], path) for path in PATHS}
        for case in study.cases
    }


@dataclass(frozen=True)
class Oracle:
    file: Path
    binding: dict[str, Any]
    offsets: dict[str, list[int]]
    rows: dict[str, list[tuple]]
    vocab: int


@dataclass(frozen=True)
class Oracles:
    calibration: Oracle | None = None
    evaluation: Oracle | None = None
    q4_reference: Oracle | None = None


def oracle_rows(workload: cached.Workload, vocab: int) -> tuple[dict[str, list[dict]], dict[str, list[int]]]:
    rows, offsets, cursor = {}, {}, 0
    for index, case in enumerate(workload.cases):
        rows[case.name] = [
            (event, row) for event in range(len(case.events))
            for row in cached.expected_logits(index, event, case.events[event])
        ]
        offsets[case.name] = [cursor + n * vocab * 4 for n in range(len(rows[case.name]))]
        cursor += len(rows[case.name]) * vocab * 4
    return rows, offsets


def check_oracle_receipt(receipt: dict[str, Any], workload: cached.Workload, expected: dict[str, list]) -> None:
    for measured, case in zip(receipt["measurements"], workload.cases):
        require(measured["name"] == case.name, "oracle case order differs from the workload")
        events = measured["events"]
        require([e["event"]["name"] for e in events] == [e.name for e in case.events], f"{case.name}: oracle events differ")
        got = [row for event in events for row in event["logit_rows"]]
        want = expected[case.name]
        require(len(got) == len(want), f"{case.name}: oracle row count differs")
        for row, (_, wanted) in zip(got, want):
            require(all(row[key] == value for key, value in wanted.items()), f"{case.name}: oracle row identity differs")


def check_receipt_source(receipt: dict[str, Any], workload_file: Path, workload: cached.Workload) -> int:
    require(receipt.get("schema") == cached.RECEIPT_SCHEMA and receipt.get("quality") == "unverified",
            "oracle receipt is not a cached comparator receipt")
    require(receipt["workload"]["manifest_sha256"] == sha256_file(workload_file), "oracle workload hash differs from the workload")
    require(receipt["engine"]["requested_backend"] == "cpu" and receipt["engine"]["device"]["backend"] == "cpu",
            "the oracle must run on the llama.cpp CPU backend")
    require(receipt["provenance"]["plan_sha256"] == hashlib.sha256(cached.encode_plan(workload)).hexdigest(),
            "oracle plan hash differs from the workload")
    vocab = receipt["engine"]["model"]["vocab"]
    require(vocab == workload.model_contract["vocab"], "oracle vocabulary differs from the workload")
    return vocab


def check_metadata_pin(subject_sha256: str, gguf: dict[str, Any]) -> None:
    """Require the record's GGUF block to equal the independently read pin, with no extra field."""
    pin = common_oracle.LOGIT_ORACLE_METADATA.get(subject_sha256)
    require(pin is not None, "no independent metadata pin exists for the subject model")
    require(set(gguf) == set(common_oracle.GGUF_IDENTITY_FIELDS), "GGUF identity holds other fields than the pinned set")
    require(all(gguf[key] == value for key, value in pin.items()), "GGUF metadata differ from the independent pin")


def check_bf16_identity(identity_file: Path, receipt_file: Path, receipt: dict[str, Any], workload: cached.Workload) -> dict[str, Any]:
    """Require the identity record to bind this receipt to the pinned oracle of the workload subject."""
    record = json.loads(identity_file.read_text())
    linked = {
        "schema": cached.ORACLE_IDENTITY_SCHEMA,
        "comparator_receipt_sha256": sha256_file(receipt_file),
        "logits_sha256": receipt["artifacts"]["logits"]["sha256"],
        "workload_sha256": receipt["workload"]["manifest_sha256"],
        "plan_sha256": receipt["provenance"]["plan_sha256"],
    }
    require(all(record.get(key) == value for key, value in linked.items()), "oracle identity record is not linked to the receipt")
    subject, oracle, pair = record["subject"], record["oracle"], record["pair"]
    pinned = common_oracle.LOGIT_ORACLE_PAIRS.get(subject["sha256"])
    require(pinned is not None and pair == pinned, "oracle identity names an unpinned pair")
    require(subject["sha256"] == workload.model_contract["subject_sha256"], "oracle subject differs from the workload subject")
    require(oracle["sha256"] == pinned["oracle_sha256"] == receipt["provenance"]["model"]["sha256"], "run model is not the pinned oracle")
    gguf = oracle["gguf"]
    require(subject["gguf"] == gguf, "subject and oracle GGUF metadata differ")
    require((gguf["architecture"], gguf["tokenizer"]) == (pinned["architecture"], pinned["tokenizer"]), "GGUF differs from the pinned pair")
    check_metadata_pin(subject["sha256"], gguf)
    require(receipt["engine"]["model"]["ftype"] == pinned["oracle_ftype"], "engine model file type is not the pinned oracle type")
    return {
        "subject_model_sha256": subject["sha256"], "oracle_dtype": pinned["oracle_dtype"],
        "architecture_sha256": gguf["architecture_sha256"], "tokenizer_sha256": gguf["tokenizer_sha256"],
        "identity_sha256": sha256_file(identity_file),
    }


def oracle_origin(receipt: dict[str, Any]) -> dict[str, Any]:
    """Return the CPU origin fields compared between oracle runs.

    The binary name and the kernel release are left out. The first follows the output
    path, and the binary hash already covers its content.
    """
    provenance, device = receipt["provenance"], receipt["engine"]["device"]
    return {
        "host_system": provenance["host"]["system"], "host_machine": provenance["host"]["machine"],
        "cpu_description": device["description"], "compiler": provenance["compiler"],
        "comparator_binary_sha256": provenance["comparator_binary"]["sha256"],
        "llama_libraries": {item["name"]: item["sha256"] for item in provenance["linked_llama_libraries"]},
        "backend_module_sha256": device["backend_module_sha256"],
        "collector_sources": {item["name"]: item["sha256"] for item in provenance["source_inputs"]
                              if item["name"].startswith("research/")},
        "llama_revision": provenance["llama_revision"], "pin_file_sha256": provenance["pin_file_sha256"],
    }


def origin_label(origin: dict[str, Any], host_system: str) -> dict[str, str]:
    """Name whether the oracle ran on the receipts' host system and machine or came from another one."""
    system = {"darwin": "macos"}.get(origin["host_system"].lower(), origin["host_system"].lower())
    same = host_system.lower().startswith(system) and origin["host_machine"].lower() in host_system.lower()
    return {
        "kind": "same_host" if same else "common_bundle",
        "oracle_host": f"{origin['host_system']} {origin['host_machine']}", "receipts_host_system": host_system,
        "meaning": "one CPU oracle run bounds both phases; the receipts' backend does not affect it",
    }


def load_oracle(
    receipt_file: Path, logits_file: Path, workload_file: Path, root: Path = ROOT,
    identity_file: Path | None = None,
) -> Oracle:
    """Bind a llama.cpp CPU comparator receipt to its workload.

    With an identity record the oracle is the pinned BF16 model for the subject.
    Without one the receipt must come from the subject model itself, and the
    binding names it a same-weights comparison, never an independent oracle.
    """
    receipt = json.loads(receipt_file.read_text())
    workload = cached.load_workload(workload_file, root)
    vocab = check_receipt_source(receipt, workload_file, workload)
    expected, offsets = oracle_rows(workload, vocab)
    check_oracle_receipt(receipt, workload, expected)
    logits = receipt["artifacts"]["logits"]
    total = sum(len(v) for v in expected.values()) * vocab * 4
    require(logits["bytes"] == total == logits_file.stat().st_size, "oracle logit size differs from the workload rows")
    require(logits["sha256"] == sha256_file(logits_file), "oracle logit hash differs from the receipt")
    model = receipt["provenance"]["model"]
    binding = {
        "kind": "q4_subject_cpu_comparison", "receipt_sha256": sha256_file(receipt_file),
        "logits_sha256": logits["sha256"], "workload_sha256": receipt["workload"]["manifest_sha256"],
        "plan_sha256": receipt["provenance"]["plan_sha256"], "model_sha256": model["sha256"],
        "model_ftype_name": receipt["engine"]["model"]["ftype_name"],
        "llama_revision": receipt["provenance"]["llama_revision"], "backend": "cpu",
        "origin": oracle_origin(receipt), "host_release": receipt["provenance"]["host"]["release"],
        "rows": sum(len(v) for v in expected.values()),
    }
    if identity_file is None:
        require(model["sha256"] == workload.model_contract["subject_sha256"], "a receipt without an identity record must run the subject model")
    else:
        binding.update(check_bf16_identity(identity_file, receipt_file, receipt, workload), kind="bf16_oracle")
    raw = json.loads(workload_file.read_text())
    return Oracle(logits_file, binding, offsets, source_rows(workload, raw), vocab)


def oracle_pairs(oracle: Oracle, case: str, sidecar: Sidecar) -> Iterator[tuple[array.array, array.array]]:
    require(sidecar.vocab == oracle.vocab, f"{case}: sidecar vocabulary differs from the oracle")
    with oracle.file.open("rb") as source:
        for offset, values in zip(oracle.offsets[case], read_rows(sidecar)):
            source.seek(offset)
            yield read_values(source, oracle.vocab), values


def oracle_subject(oracle: Oracle) -> str:
    """Return the sha256 of the model that the oracle rows describe."""
    return oracle.binding.get("subject_model_sha256", oracle.binding["model_sha256"])


def oracle_numerics(study: Study, oracle: Oracle) -> dict[str, dict[str, Any]]:
    """Score every path against the oracle. Scored rows equal the oracle rows for each case."""
    require(study.identity["model_sha256"] == oracle_subject(oracle), "receipts ran a model that the oracle does not describe")
    result: dict[str, dict[str, Any]] = {}
    for case in study.cases:
        name, cache = case["name"], {}
        for path in PATHS:
            for sidecar in study.sidecars[path][name]:
                cache.setdefault(sidecar.sha256, score_rows(oracle_pairs(oracle, name, sidecar)))
        result[name] = {
            path: worst([cache[sidecar.sha256] for sidecar in study.sidecars[path][name]])
            for path in PATHS
        }
    return result


def timed_samples(run: Run, case: str) -> list[float]:
    record = case_record(run, case)
    return [setup + decode for setup, decode in zip(record["setup_ms"], record["decode_ms"])]


def timing_criteria(study: Study) -> dict[str, Any]:
    spreads = []
    for run in study.runs.values():
        for case in study.cases:
            samples = timed_samples(run, case["name"])
            spreads.extend(abs(value - median(samples)) / median(samples) for value in samples)
    require(any(spread > 0.0 for spread in spreads), "calibration has no timing variation")
    return {"minimum_reviewable_relative_effect": math.nextafter(percentile95(spreads), math.inf)}


def bounds(numerics: dict[str, dict[str, dict[str, Any]]], paths: tuple[str, ...] = PATHS) -> dict[str, dict[str, float]]:
    return {
        path: {
            metric: math.nextafter(max(numerics[case][path][metric] for case in numerics), math.inf)
            for metric in METRICS
        }
        for path in paths
    }


def bf16_deltas(numerics: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, dict[str, float]]]:
    """Return each non-control path minus the control, per case and metric, on the BF16 rows."""
    return {
        case: {
            path: {metric: paths[path][metric] - paths[CONTROL][metric] for metric in METRICS}
            for path in PATHS if path != CONTROL
        }
        for case, paths in numerics.items()
    }


def delta_bounds(deltas: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Limit each difference from above only. A path closer to BF16 than the control never fails."""
    return {
        path: {
            metric: math.nextafter(max(0.0, *(deltas[case][path][metric] for case in deltas)), math.inf)
            for metric in METRICS
        }
        for path in PATHS if path != CONTROL
    }


def build_disclosure(study: Study) -> dict[str, Any]:
    """List each path's executable hash and the run order that the receipts record."""
    stamps = {path: datetime.fromisoformat(run.receipt["provenance"]["generated_utc"]) for path, run in study.runs.items()}
    artifacts = {path: run.receipt["provenance"]["artifact_sha256"] for path, run in study.runs.items()}
    return {
        "artifact_sha256": artifacts, "receipts": len(artifacts), "distinct_executables": len(set(artifacts.values())),
        "run_order": sorted(stamps, key=stamps.__getitem__),
        "limit": "each path ran in its own process, from its own executable when distinct_executables "
                 "equals receipts; the run order is not randomized; drift between builds, processes, "
                 "and run order is not bounded",
    }


def sidecar_digests(sidecar: Sidecar, sequences: list[int]) -> tuple[str, str]:
    """Recompute the driver's logit and position digests from one sidecar with every step captured."""
    logits, positions = hashlib.sha256(), hashlib.sha256()
    with sidecar.file.open("rb") as source:
        for _ in range(sidecar.rows):
            sequence, index, position, vocab = ROW_HEADER.unpack(source.read(ROW_HEADER.size))
            require(sequence in sequences, f"{sidecar.file.name}: sequence {sequence} has no branch")
            logits.update(struct.pack("<Q", position) + source.read(4 * vocab))
            positions.update(struct.pack("<QQQ", sequences.index(sequence), index, position))
    return logits.hexdigest(), positions.hexdigest()


def check_digests(study: Study, manifest: dict[str, Any]) -> None:
    """Tie the recorded logit and position digests of every repetition to its sidecar."""
    specs = {case["name"]: case for case in manifest["calibration"]}
    for case in study.cases:
        spec, name = specs[case["name"]], case["name"]
        require(spec["capture_steps"] == list(range(spec["decode_steps"])), f"{name}: digest linkage needs every step captured")
        sequences = spec.get("branch_sequences") or list(range(len(spec["prefix_offsets"])))
        for path, run in study.runs.items():
            record = case_record(run, name)
            by_hash = {sidecar.sha256: sidecar for sidecar in study.sidecars[path][name]}
            for rep, entry in enumerate(record["logits_sidecars"]):
                got = sidecar_digests(by_hash[entry["sha256"]], sequences)
                want = (record["logits_digest"][rep], record["raw_logit_position_digest"][rep])
                require(got == want, f"{path}/{name}: repetition {rep} digests differ from its sidecar")


def source_slices(source: Path) -> set[int]:
    positions: set[int] = set()
    for item in json.loads(source.read_text())["slices"].values():
        positions.update(range(item["offset"], item["offset"] + item["count"]))
    return positions


def calibration_positions(manifest: dict[str, Any]) -> set[int]:
    positions: set[int] = set()
    for case in manifest["calibration"]:
        ranges = list(zip(case["prefix_offsets"], case["prefix_tokens"]))
        ranges += list(zip(case["tail_offsets"], case["tail_tokens"]))
        ranges += [(offset, 1) for row in case["teacher_offsets"] for offset in row]
        if case["common_tail_offset"] is not None:
            ranges += [(case["common_tail_offset"], case["common_tail_tokens"])]
        for offset, count in ranges:
            positions.update(range(offset, offset + count))
    return positions


def workload_facts(study: Study, manifest: dict[str, Any], source: Path, rows: dict[str, list[tuple]]) -> dict[str, Any]:
    raw = json.loads(source.read_text())
    classes = {case["name"]: case["class"] for case in raw["cases"]}
    missing = [name for name in REQUIRED_EVALUATION_CLASSES if name not in classes.values()]
    require(not missing, f"the source workload lacks case classes {missing}")
    require(any(c["topology"] == "unrelated" for c in study.cases), "calibration lacks an unrelated case")
    overlap = calibration_positions(manifest) & source_slices(source)
    return {
        "calibration_cases": [
            {key: case[key] for key in ("name", "topology", "input_digest", "batch_schedule", "sidecar_logit_rows")}
            for case in study.cases
        ],
        "evaluation_cases": list(rows),
        "evaluation_classes": classes,
        "row_identity_sha256": {name: sha256_json(keys) for name, keys in rows.items()},
        "fixture_positions_shared_by_phases": len(overlap),
    }


def case_record(run: Run, name: str) -> dict[str, Any]:
    return next(case for case in run.receipt["cases"] if case["name"] == name)


def reuse_broken(path: str, topology: str, groups: int) -> bool:
    if path not in SHARED_PATHS or topology == "unrelated":
        return groups != 0
    return topology == "shared" and groups == 0


def reuse_controls(study: Study) -> dict[str, list[str]]:
    """Return the multi-row group counters that break the reuse rule, by case."""
    broken: dict[str, list[str]] = {}
    for case in study.cases:
        for path, run in study.runs.items():
            groups = case_record(run, case["name"])["stats"][0]["multi_row_groups"]
            if reuse_broken(path, case["topology"], groups):
                broken.setdefault(case["name"], []).append(f"{path}:{groups}")
    return broken


def monitoring_number(text: str) -> float:
    return float(text.split()[0])


def check_collection(record: dict[str, Any], samples: Path, study: Study) -> None:
    """Require the collection record to name exactly the calibration receipts and the sample file."""
    require(record["gpu_samples_sha256"] == sha256_file(samples), "GPU samples differ from the collection record")
    entries = {item["path"]: item for item in record["records"]}
    require(sorted(entries) == sorted(run.file.name for run in study.runs.values()), "collection names other receipts")
    for path, run in study.runs.items():
        entry = entries[run.file.name]
        require(entry["sha256"] == run.file_sha256 and entry["driver_sha256"] == run.receipt["provenance"]["artifact_sha256"],
                f"{path}: collection record differs from the receipt")


def summarize_samples(samples: Path) -> dict[str, Any]:
    """Return the row count, time span, sampling interval, and clock, power, and temperature ranges."""
    with samples.open(newline="") as source:
        rows = list(csv.DictReader(source, skipinitialspace=True))
    require(rows, "GPU samples hold no rows")
    column = {name: [monitoring_number(row[name]) for row in rows] for name in MONITORING_COLUMNS}
    stamps = [datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f") for row in rows]
    gaps = sorted((later - earlier).total_seconds() for earlier, later in zip(stamps, stamps[1:]))
    return {
        "rows": len(rows), "first_timestamp": rows[0]["timestamp"], "last_timestamp": rows[-1]["timestamp"],
        "median_interval_s": gaps[len(gaps) // 2] if gaps else None,
        "sm_clock_mhz": [min(column[MONITORING_COLUMNS[0]]), max(column[MONITORING_COLUMNS[0]])],
        "memory_clock_mhz": [min(column[MONITORING_COLUMNS[1]]), max(column[MONITORING_COLUMNS[1]])],
        "power_w_max": max(column[MONITORING_COLUMNS[2]]), "temperature_c_max": max(column[MONITORING_COLUMNS[3]]),
        "pstates": sorted({row["pstate"] for row in rows}),
    }


def monitoring_binding(files: tuple[Path, Path] | None, study: Study) -> dict[str, Any]:
    """Bind an externally sampled GPU clock and power record to the receipts it names."""
    if files is None:
        return UNBOUND_MONITORING
    samples, collection = files
    record = json.loads(collection.read_text())
    check_collection(record, samples, study)
    return {
        "bound": True, "collection_sha256": sha256_file(collection), "samples_sha256": sha256_file(samples),
        "clock_policy": record["gpu_clock_policy"], "collection_scope": record["scope"],
        **summarize_samples(samples),
        "limit": "sampled outside the driver; rows are not attributed to a path, case, or repetition; "
                 "the record is not a traffic measurement",
    }


def quality_criteria(study: Study, oracles: Oracles) -> dict[str, Any]:
    """Freeze BF16 quality limits from the calibration rows, or record that none exist."""
    if oracles.calibration is None:
        return {"gate": "unverified", "bf16": None}
    check_rows(study, oracles.calibration.rows)
    numerics = oracle_numerics(study, oracles.calibration)
    deltas = bf16_deltas(numerics)
    return {
        "gate": "bf16", "rule": BF16_RULE, "limit": BF16_LIMIT, "kld_definition": KLD_DEFINITION,
        "oracles": {"calibration": oracles.calibration.binding, "evaluation": oracles.evaluation.binding},
        "calibration_absolute_report_only": numerics, "calibration_deltas": deltas,
        "delta_bounds": delta_bounds(deltas),
    }


def derive_criteria(
    study: Study, manifest: dict[str, Any], source: Path,
    oracles: Oracles | None = None, monitoring: tuple[Path, Path] | None = None,
) -> dict[str, Any]:
    """Return every frozen value. The result has no clock reading, so it recomputes exactly."""
    oracles = oracles or Oracles()
    raw = json.loads(source.read_text())
    workload = cached.load_workload(source, ROOT)
    rows = source_rows(workload, raw)
    numerics = control_numerics(study)
    return {
        "schema": CRITERIA_SCHEMA,
        "backend": study.identity["backend"],
        "identity": study.identity,
        "calibration_receipts": {p: {"sha256": r.file_sha256, "artifact_sha256": r.receipt["provenance"]["artifact_sha256"]}
                                 for p, r in study.runs.items()},
        "source_workload_sha256": sha256_file(source),
        "workload": workload_facts(study, manifest, source, rows),
        "quality": quality_criteria(study, oracles),
        "q4_reference": None if oracles.q4_reference is None else oracles.q4_reference.binding,
        "numerical_rule": NUMERIC_RULE,
        "kld_definition": KLD_DEFINITION,
        "reuse_rule": REUSE_RULE,
        "numerical": bounds(numerics),
        "calibration_observed": numerics,
        "digest_scope": DIGEST_SCOPE,
        "calibration_builds": build_disclosure(study),
        "timing_rule": TIMING_RULE,
        "timing_rule_origin": TIMING_ORIGIN,
        "timing": timing_criteria(study),
        "gpu_monitoring": monitoring_binding(monitoring, study),
        "tools": tool_hashes(),
    }


def load_calibration(files: list[Path], backend: str, manifest_file: Path = MANIFEST) -> tuple[Study, dict[str, Any], Path]:
    """Load calibration receipts and check them against the manifest and the reuse rule."""
    manifest = json.loads(manifest_file.read_text())
    source = source_path(manifest_file)
    study = load_study(files, "calibration", backend)
    require(study.identity["source_manifest_sha256"] == sha256_file(source), "receipts name a different source workload")
    require(study.identity["manifest_sha256"] == sha256_file(manifest_file), "receipts name a different runtime manifest")
    fixture = manifest_file.parent / manifest["fixture"]
    require(study.identity["fixture_sha256"] == sha256_file(fixture), "receipts name a different token fixture")
    check_calibration_manifest(study, manifest, read_tokens(fixture))
    check_digests(study, manifest)
    broken = reuse_controls(study)
    require(not broken, f"calibration breaks the reuse rule: {broken}")
    return study, manifest, source


def effect_label(effect: float, threshold: float) -> str:
    """Name the signed effect. A slower candidate has a negative effect."""
    if effect >= threshold:
        return FASTER
    return SLOWER if effect <= -threshold else WITHIN


def overall_label(labels: list[str]) -> str:
    faster, slower = FASTER in labels, SLOWER in labels
    if faster and slower:
        return MIXED
    return FASTER if faster else SLOWER if slower else WITHIN


def timing_report(study: Study, criteria: dict[str, Any], monitored: bool) -> dict[str, Any]:
    threshold = criteria["timing"]["minimum_reviewable_relative_effect"]
    suffix = "" if monitored else "_unmonitored"
    cases = {}
    for case in study.cases:
        name = case["name"]
        control = median(timed_samples(study.runs[CONTROL], name))
        candidate = median(timed_samples(study.runs[CANDIDATE], name))
        effect = 1.0 - candidate / control
        cases[name] = {"control_median_ms": control, "candidate_median_ms": candidate,
                       "relative_effect": effect, "label": effect_label(effect, threshold) + suffix}
    return {"threshold": threshold, "basis": "within-process repetition spread of the calibration receipts",
            "limit": "excludes drift between processes, builds, and run order; unpaired runs; not a speed claim",
            "monitored": monitored, "builds": build_disclosure(study), "cases": cases,
            "overall": overall_label([item["label"].removesuffix(suffix) for item in cases.values()]) + suffix}


def numerical_verdict(numerics: dict[str, Any], limits: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for name, paths in numerics.items():
        for path, bound in limits.items():
            over = [m for m in METRICS if paths[path][m] > bound[m]]
            if over:
                failures.setdefault(name, []).append(f"{path}:{','.join(over)}")
    return failures


def numerical_by_path(failures: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Give each path its own verdict and role, so an expected difference is not read as a defect."""
    failed = {entry.split(":")[0] for entries in failures.values() for entry in entries}
    return {path: {"verdict": "fail" if path in failed else "pass", "role": PATH_ROLES[path]} for path in PATHS}


def evidence_report(study: Study) -> dict[str, Any]:
    """Separate counted reads and allocated storage from traffic, which the receipts do not measure."""
    cases = {}
    for case in study.cases:
        name = case["name"]
        records = {path: case_record(run, name) for path, run in study.runs.items()}
        base = records[CONTROL]["memory_after_setup"][0]["live_bytes"]
        cases[name] = {p: {
            "counted_batch_stats": rec["stats"][0],
            "live_bytes_after_setup": rec["memory_after_setup"][0]["live_bytes"],
            "live_bytes_after_setup_minus_control": rec["memory_after_setup"][0]["live_bytes"] - base,
        } for p, rec in records.items()}
    return {
        "shared_storage": "allocator live bytes after setup, measured by the Runtime allocator",
        "shared_reads": "row groups counted by the batch path, not observed in hardware",
        "traffic": {"measured": None, "estimated": None,
                    "reason": "receipts carry no DRAM counters and no layer, head, or context geometry"},
        "cases": cases,
    }


def check_frozen(
    criteria_file: Path, calibration: list[Path], backend: str, oracles: Oracles,
    monitoring: tuple[Path, Path] | None,
) -> dict[str, Any]:
    criteria = json.loads(criteria_file.read_text())
    study, manifest, source = load_calibration(calibration, backend)
    derived = derive_criteria(study, manifest, source, oracles, monitoring)
    stored = {key: value for key, value in criteria.items() if key != "frozen_utc"}
    differing = sorted(key for key in derived if derived[key] != stored.get(key))
    require(not differing and set(stored) == set(derived), f"criteria differ from the calibration in {differing}")
    require(isinstance(criteria.get("frozen_utc"), str), "criteria lack frozen_utc")
    return criteria


def check_evaluation(study: Study, criteria: dict[str, Any], criteria_file: Path) -> None:
    require(study.identity == criteria["identity"], "evaluation identity differs from calibration")
    frozen = datetime.fromisoformat(criteria["frozen_utc"])
    for path, run in study.runs.items():
        provenance = run.receipt["provenance"]
        require(run.receipt.get("criteria_sha256") == sha256_file(criteria_file), f"{path}: receipt names other criteria")
        require(datetime.fromisoformat(provenance["generated_utc"]) >= frozen, f"{path}: receipt predates the freeze")


def source_path(manifest_file: Path = MANIFEST) -> Path:
    return manifest_file.parent / json.loads(manifest_file.read_text())["source_manifest"]


@contextlib.contextmanager
def calibration_workload_file(manifest_file: Path = MANIFEST) -> Iterator[Path]:
    """Yield the generated calibration workload, rebuilt from the runtime manifest."""
    workload = common_oracle.calibration_workload(
        json.loads(manifest_file.read_text()), json.loads(source_path(manifest_file).read_text()))
    with tempfile.TemporaryDirectory() as directory:
        file = Path(directory) / "runtime_calibration_workload.json"
        file.write_bytes(common_oracle.workload_bytes(workload))
        yield file


def load_oracles(args: argparse.Namespace) -> Oracles:
    """Load every oracle named on the command line. A BF16 gate needs both phases."""
    calibration = evaluation = reference = None
    if args.bf16_calibration:
        receipt, logits, identity = args.bf16_calibration
        with calibration_workload_file() as workload:
            calibration = load_oracle(receipt, logits, workload, identity_file=identity)
    if args.bf16_evaluation:
        receipt, logits, identity = args.bf16_evaluation
        evaluation = load_oracle(receipt, logits, source_path(), identity_file=identity)
    if args.oracle_receipt is not None:
        reference = load_oracle(args.oracle_receipt, args.oracle_logits, source_path())
    require((calibration is None) == (evaluation is None), "the BF16 gate needs a calibration and an evaluation oracle")
    if calibration is not None:
        same = {key: calibration.binding[key] for key in ("oracle_dtype", "llama_revision", "model_sha256", "origin")}
        require(same == {key: evaluation.binding[key] for key in same}, "BF16 oracles differ between phases in dtype, model, or CPU origin")
    return Oracles(calibration, evaluation, reference)


def evaluation_study(args: argparse.Namespace, criteria: dict[str, Any]) -> Study:
    study = load_study(args.evaluation, "evaluation", args.backend)
    check_evaluation(study, criteria, args.criteria)
    source = source_path()
    rows = source_rows(cached.load_workload(source, ROOT), json.loads(source.read_text()))
    check_evaluation_rows(study, rows, criteria)
    return study


def quality_report(study: Study, criteria: dict[str, Any], oracles: Oracles) -> dict[str, Any]:
    reference = None
    if oracles.q4_reference is not None:
        check_rows(study, oracles.q4_reference.rows)
        reference = {"label": "same-weights llama.cpp CPU comparison, not an independent BF16 oracle",
                     "binding": oracles.q4_reference.binding, "numerics": oracle_numerics(study, oracles.q4_reference)}
    if oracles.evaluation is None:
        return {"gate": "unverified", "bf16": None, "q4_reference": reference, "failures": {}}
    check_rows(study, oracles.evaluation.rows)
    numerics = oracle_numerics(study, oracles.evaluation)
    deltas = bf16_deltas(numerics)
    binding = oracles.evaluation.binding
    return {
        "gate": "bf16", "limit": BF16_LIMIT,
        "bf16": {"binding": binding, "origin": origin_label(binding["origin"], study.identity["host_system"]),
                 "absolute_report_only": numerics, "deltas_vs_control": deltas},
        "q4_reference": reference, "failures": numerical_verdict(deltas, criteria["quality"]["delta_bounds"]),
    }


def build_report(
    args: argparse.Namespace, study: Study, criteria: dict[str, Any], oracles: Oracles
) -> dict[str, Any]:
    numerics = control_numerics(study)
    failures, broken = numerical_verdict(numerics, criteria["numerical"]), reuse_controls(study)
    quality = quality_report(study, criteria, oracles)
    monitoring = monitoring_binding(args.evaluation_monitoring, study) if args.evaluation_monitoring else UNBOUND_MONITORING
    timing = timing_report(study, criteria, monitoring["bound"])
    bf16 = "unverified" if quality["gate"] == "unverified" else "fail" if quality["failures"] else "pass"
    return {
        "schema": REPORT_SCHEMA, "backend": args.backend, "criteria_sha256": sha256_file(args.criteria),
        "evaluation_receipts": {path: run.file_sha256 for path, run in study.runs.items()},
        "quality": quality,
        "numerical_vs_control": numerics, "numerical_failures": failures, "reuse_control_failures": broken,
        "numerical_by_path": numerical_by_path(failures),
        "timing": timing, "evidence": evidence_report(study), "gpu_monitoring": monitoring,
        "verdict": {
            "identity": "pass", "numerical_vs_control": "fail" if failures or broken else "pass",
            "bf16_quality": bf16,
            "timing": timing["overall"],
        },
        "digest_scope": DIGEST_SCOPE,
        "speed_claim": "none: the timing rule bounds repetition jitter only, not drift between processes",
        "claim": "none: no speed, novelty, or end-to-end quality claim",
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    oracles = load_oracles(args)
    criteria = check_frozen(args.criteria, args.calibration, args.backend, oracles, args.monitoring)
    return build_report(args, evaluation_study(args, criteria), criteria, oracles)


def add_frozen_inputs(parser: argparse.ArgumentParser) -> None:
    """Add the options that both the freeze and the validator use to name frozen inputs."""
    parser.add_argument("--bf16-calibration", type=Path, nargs=3, metavar=("RECEIPT", "LOGITS", "IDENTITY"),
                        help="BF16 oracle files for the generated calibration workload")
    parser.add_argument("--bf16-evaluation", type=Path, nargs=3, metavar=("RECEIPT", "LOGITS", "IDENTITY"),
                        help="BF16 oracle files for the source workload")
    parser.add_argument("--oracle-receipt", type=Path,
                        help="same-weights comparator receipt for the source workload (not BF16)")
    parser.add_argument("--oracle-logits", type=Path, help="logit rows named by --oracle-receipt")
    parser.add_argument("--gpu-samples", type=Path, help="external GPU clock and power samples for calibration")
    parser.add_argument("--collection", type=Path, help="collection record that names the calibration receipts")


def check_frozen_inputs(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if (args.oracle_receipt is None) != (args.oracle_logits is None):
        parser.error("--oracle-receipt and --oracle-logits go together")
    if (args.gpu_samples is None) != (args.collection is None):
        parser.error("--gpu-samples and --collection go together")
    if bool(args.bf16_calibration) != bool(args.bf16_evaluation):
        parser.error("--bf16-calibration and --bf16-evaluation go together")
    args.monitoring = None if args.gpu_samples is None else (args.gpu_samples, args.collection)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "metal"), required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    add_frozen_inputs(parser)
    parser.add_argument("--evaluation-gpu-samples", type=Path)
    parser.add_argument("--evaluation-collection", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-positive", action="store_true",
                        help=f"exit 1 unless the timing label is {FASTER} with GPU samples bound")
    args = parser.parse_args()
    check_frozen_inputs(parser, args)
    if (args.evaluation_gpu_samples is None) != (args.evaluation_collection is None):
        parser.error("--evaluation-gpu-samples and --evaluation-collection go together")
    args.evaluation_monitoring = (
        None if args.evaluation_gpu_samples is None else (args.evaluation_gpu_samples, args.evaluation_collection))
    return args


def exit_code(verdict: dict[str, str], require_positive: bool) -> int:
    """Return 1 on a failed numerical or BF16 verdict, or when a required positive timing label is absent.

    An unmonitored label carries a suffix, so it never equals the positive label.
    """
    failed = "fail" in (verdict["numerical_vs_control"], verdict["bf16_quality"])
    return 1 if failed or (require_positive and verdict["timing"] != FASTER) else 0


def main() -> int:
    args = parse_args()
    if args.report.exists():
        raise SystemExit(f"refusing to replace existing report: {args.report}")
    try:
        report = validate(args)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"runtime validation rejected: {error}") from error
    publish_json(args.report, report)
    print(json.dumps({"quality": report["quality"]["gate"], "verdict": report["verdict"]}))
    return exit_code(report["verdict"], args.require_positive)


if __name__ == "__main__":
    raise SystemExit(main())
