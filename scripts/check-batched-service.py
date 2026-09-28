#!/usr/bin/env python3
"""Validate one batched-service study receipt for the release evidence gate.

The checker recomputes every sample, ratio, summary, and check from the
retained run records. It does not trust a recorded boolean or summary. It
binds the receipt to the frozen workload, the gated Qwen3 model, the plan and
source manifest hashes, the Leone quality receipt named by the CUDA quality
comparison, the GPU record, and the release binary.

Limits: a run record keeps the SHA-256 of each transcript and response, not the
signed response. The checker verifies digest format, agreement between modes,
and arithmetic. It does not verify response signatures or that a digest matches
a response body. It requires the receipt to say so in its limits. Timing and
token counts are the harness's own record. Each mode's `batch_size` is the
command-line limit the harness passed, not a measured dispatch width, and the
receipt must say so. The checker rejects two byte-identical run records. It
does not detect a forgery whose runs are distinct and consistent. Appended
fields are not attested. The checker does not run the CPU
receipt validator. The comparison record validation does, and this checker
requires the quality receipt to match that record by hash.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, NamedTuple, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import source_inputs

STUDY_SCHEMA = "leone.batched-service-study.v1"
RUN_SCHEMA = "leone.server-study.v2"
BUILD_SCHEMA = "leone.build-info.v1"
SOURCE_SCHEMA = "leone.source-inputs.v2"
QUALITY_SCHEMA = "leone.quality-comparison.v2"
RELEASE_IDENTITY = ("linux-x86_64", "x86_64-unknown-linux-gnu", "cuda")
PLAN_PATH = "plans/qwen3-8b-sm89.json"
REPETITIONS = 5
WORKLOAD = {"concurrent_clients": 4, "max_tokens": 64, "temperature": 0, "seed": 0}
PROMPT = (
    "Write a detailed explanation of why deterministic scheduling matters for "
    "language model inference. Do not use a list."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
GPU_NAME = "RTX 4090"
DIGEST_LIMIT = (
    "The study keeps response and transcript digests. It does not retain "
    "signed responses or verify their signatures."
)
BATCH_LIMIT = "Batch sizes record command-line limits. Dispatch widths are unmeasured."
REQUIRED_SOURCE_FILES = (
    "scripts/check-batched-service.py",
    "scripts/render-release-evidence.sh",
    "scripts/source_inputs.py",
    "scripts/study-batched-service.sh",
    "scripts/study-live-server.sh",
)
DISCONNECT_CURL_EXIT = 28
RELATIVE_TOLERANCE = 1e-9
PERCENTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))


@dataclass(frozen=True)
class Expected:
    """Release identity that the dispatcher supplies from the trusted manifest."""

    platform: str
    target: str
    backend: str
    model_sha256: str
    binary_sha256: str
    quality_record: str
    quality_receipt: str
    source_manifest: str


class ModeResult(NamedTuple):
    """Values recomputed from the requests of one server mode."""

    wall_ms: float
    tok_s: float
    p95_ms: float
    tokens: int
    transcripts: list[str]


class RunResult(NamedTuple):
    """Values recomputed from one repetition."""

    scheduled: ModeResult
    serial: ModeResult
    throughput_ratio: float
    p95_ratio: float


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"receipt contains a non-finite number: {name}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    require(len({key for key, _ in pairs}) == len(pairs), "receipt repeats an object key")
    return dict(pairs)


def load_json(path: Path, label: str) -> dict[str, Any]:
    """Read a JSON object and reject repeated keys and non-finite constants."""
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def _relative(value: Any, label: str) -> str:
    require(isinstance(value, str) and bool(value), f"{label} is missing")
    path = PurePosixPath(value)
    require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in value,
        f"{label} is unsafe",
    )
    require(path.as_posix() == value and value != ".", f"{label} is not normalized")
    return value


def _regular_file(root: Path, relative: str, label: str) -> Path:
    current = root
    for part in PurePosixPath(_relative(relative, label)).parts:
        current = current / part
        require(not current.is_symlink(), f"{label} contains a symlink")
    require(current.is_file(), f"{label} is missing")
    return current


def _digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a SHA-256",
    )
    return value


def _commit(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{label} is not a Git object ID",
    )
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} is missing")
    return value


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} is missing")
    return value


def _positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not a number",
    )
    require(math.isfinite(value) and value > 0, f"{label} is not positive and finite")
    return float(value)


def _count(value: Any, label: str, low: int, high: int) -> int:
    require(type(value) is int and low <= value <= high, f"{label} is outside {low} to {high}")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0)


def _matches(expected: Any, actual: Any, label: str) -> None:
    """Require actual to hold every expected field. Extra fields are allowed."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), f"{label} is not an object")
        for key, value in expected.items():
            require(key in actual, f"{label}.{key} is missing")
            _matches(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, (bool, str)):
        require(type(actual) is type(expected) and actual == expected, f"{label} differs from the retained runs")
    else:
        require(
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and _close(expected, actual),
            f"{label} differs from the retained runs",
        )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * probability)]


def _median(values: Sequence[float]) -> float:
    return sorted(values)[len(values) // 2]


def check_source(root: Path, expected: Expected, offline: bool) -> tuple[str, dict[str, Any]]:
    """Return the trusted source commit and every recorded source file."""
    manifest_path = _regular_file(root, expected.source_manifest, "source input manifest")
    manifest = load_json(manifest_path, "source input manifest")
    require(manifest.get("schema_version") == SOURCE_SCHEMA, "source input manifest has an unknown schema")
    commit = _commit(manifest.get("source_commit"), "source input commit")
    if not offline:
        source_inputs.check(root, commit, Path(expected.source_manifest))
    records = source_inputs.check_packaged_files(root, manifest, REQUIRED_SOURCE_FILES)
    return commit, records


def check_identity(study: dict[str, Any], commit: str, records: dict[str, Any], expected: Expected) -> None:
    require(study.get("schema_version") == STUDY_SCHEMA, "unexpected study schema")
    require(study.get("source_commit") == commit, "study source commit differs from the source manifest")
    model = _object(study.get("model"), "study model")
    _relative(model.get("path"), "study model path")
    require(model.get("sha256") == expected.model_sha256, "study model is not the gated Qwen3 artifact")
    plan = _object(study.get("plan"), "study plan")
    require(plan.get("path") == PLAN_PATH, "study plan is not the gated Qwen3 plan")
    recorded = records.get(PLAN_PATH)
    require(recorded is not None, "source manifest does not record the study plan")
    require(plan.get("sha256") == recorded["sha256"], "study plan differs from the source manifest")


def check_quality(root: Path, study: dict[str, Any], expected: Expected) -> None:
    """Bind the study to the Leone quality receipt that the CUDA comparison embeds."""
    comparison = load_json(_regular_file(root, expected.quality_record, "quality comparison"), "quality comparison")
    require(comparison.get("schema_version") == QUALITY_SCHEMA, "quality comparison has an unknown schema")
    require(comparison.get("model_family") == "qwen3", "quality comparison is not for Qwen3")
    models = _object(comparison.get("models"), "quality comparison models")
    require(_object(models.get("subject"), "quality subject").get("sha256") == expected.model_sha256, "quality comparison subject differs from the study model")
    entry = _object(_object(comparison.get("quality"), "quality comparison rows").get("leone"), "Leone quality row")
    require(PurePosixPath(expected.quality_receipt).name == entry.get("path"), "quality receipt is not the comparison sidecar")
    sidecar_path = _regular_file(root, expected.quality_receipt, "quality receipt")
    sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    require(sidecar_sha256 == entry.get("sha256"), "quality receipt differs from the comparison record")
    sidecar = load_json(sidecar_path, "quality receipt")
    quality = _object(study.get("quality"), "study quality")
    require(quality.get("path") == expected.quality_receipt, "study quality path is not the comparison sidecar")
    require(quality.get("sha256") == sidecar_sha256, "study quality receipt hash differs from the sidecar")
    require(isinstance(sidecar.get("receipt_id"), str) and quality.get("receipt_id") == sidecar["receipt_id"], "study quality receipt id differs from the sidecar")
    require(_object(entry.get("receipt"), "Leone quality receipt").get("receipt_id") == sidecar["receipt_id"], "quality receipt id differs from the comparison record")
    artifact = _object(_object(sidecar.get("subject"), "quality subject").get("model_artifact"), "quality model artifact")
    require(artifact.get("sha256") == expected.model_sha256, "quality receipt subject differs from the study model")


def check_hardware(hardware: Any, label: str) -> None:
    hardware = _object(hardware, f"{label} hardware")
    name = hardware.get("gpu_name")
    require(isinstance(name, str) and GPU_NAME in name, f"{label} hardware is not an {GPU_NAME}")


def check_workload(workload: Any, label: str, repetitions: bool) -> None:
    workload = _object(workload, f"{label} workload")
    for key, value in WORKLOAD.items():
        require(type(workload.get(key)) is int and workload[key] == value, f"{label} workload {key} is not the frozen value")
    require(workload.get("prompt_sha256") == PROMPT_SHA256, f"{label} workload prompt is not the frozen prompt")
    if repetitions:
        require(type(workload.get("repetitions")) is int and workload["repetitions"] == REPETITIONS, f"{label} workload repetitions is not {REPETITIONS}")


def check_binary(binary: Any, label: str, commit: str, expected: Expected) -> None:
    binary = _object(binary, f"{label} binary")
    require(binary.get("sha256") == expected.binary_sha256, f"{label} binary differs from the release binary")
    build = _object(binary.get("build_info"), f"{label} build information")
    require(build.get("schema_version") == BUILD_SCHEMA, f"{label} build information has an unknown schema")
    require(build.get("source_commit") == commit, f"{label} build source differs from the final source")
    require(build.get("source_tree_dirty") is False, f"{label} build source tree is dirty")
    require(build.get("provenance_unknown") is False, f"{label} build provenance is unknown")
    require(build.get("profile") == "release", f"{label} build is not a release build")
    require(build.get("target") == expected.target, f"{label} build target differs from the release target")
    features = build.get("features")
    require(isinstance(features, str) and expected.backend in features.split(","), f"{label} build lacks the {expected.backend} feature")


def _check_request(request: Any, label: str, wall_ms: float) -> dict[str, Any]:
    request = _object(request, label)
    require(request.get("http_code") == 200 and type(request["http_code"]) is int, f"{label} did not return HTTP 200")
    ttft = _positive(request.get("ttft_ms"), f"{label} first-byte time")
    total = _positive(request.get("total_ms"), f"{label} total time")
    require(ttft <= total <= wall_ms, f"{label} times are not ordered inside the wall time")
    _count(request.get("completion_tokens"), f"{label} completion tokens", 1, WORKLOAD["max_tokens"])
    _digest(request.get("transcript_sha256"), f"{label} transcript")
    _digest(request.get("response_sha256"), f"{label} response")
    return request


def _check_requests(mode: dict[str, Any], label: str, wall_ms: float) -> list[dict[str, Any]]:
    clients = WORKLOAD["concurrent_clients"]
    raw = _list(mode.get("requests"), f"{label} requests")
    require(len(raw) == clients, f"{label} has the wrong request count")
    requests = [_check_request(item, f"{label} request {index}", wall_ms) for index, item in enumerate(raw)]
    require(sorted(request["client"] for request in requests) == list(range(1, clients + 1)), f"{label} client numbers are not 1 to {clients}")
    return requests


def _check_percentiles(mode: dict[str, Any], key: str, values: list[float], label: str) -> None:
    recorded = _object(mode.get(key), f"{label} {key}")
    for name, probability in PERCENTILES:
        _matches(_percentile(values, probability), recorded.get(name), f"{label} {key}.{name}")


def check_mode(mode: Any, name: str, batch_size: int, label: str) -> ModeResult:
    """Recompute one mode from its requests and compare the recorded summary."""
    mode = _object(mode, label)
    require(mode.get("mode") == name, f"{label} mode name differs")
    require(type(mode.get("batch_size")) is int and mode["batch_size"] == batch_size, f"{label} batch limit is not {batch_size}")
    wall_ms = _positive(mode.get("wall_ms"), f"{label} wall time")
    requests = _check_requests(mode, label, wall_ms)
    totals = [float(request["total_ms"]) for request in requests]
    tokens = sum(request["completion_tokens"] for request in requests)
    tok_s = tokens / (wall_ms / 1000)
    _matches(tok_s, mode.get("aggregate_completion_tok_s"), f"{label} aggregate_completion_tok_s")
    _matches(max(totals) / min(totals), mode.get("latency_fairness_ratio"), f"{label} latency_fairness_ratio")
    _check_percentiles(mode, "total_ms", totals, label)
    _check_percentiles(mode, "ttft_ms", [float(request["ttft_ms"]) for request in requests], label)
    _digest(mode.get("server_log_sha256"), f"{label} server log")
    return ModeResult(
        wall_ms, tok_s, _percentile(totals, 0.95), tokens,
        [request["transcript_sha256"] for request in requests],
    )


def check_disconnect(disconnect: Any, label: str, transcript: str) -> None:
    disconnect = _object(disconnect, f"{label} disconnect")
    exit_code = disconnect.get("curl_exit")
    require(type(exit_code) is int and exit_code == DISCONNECT_CURL_EXIT, f"{label} client did not time out and disconnect")
    require(disconnect.get("client_disconnected") is True, f"{label} disconnect flag differs")
    require(type(disconnect.get("recovery_http_code")) is int and disconnect["recovery_http_code"] == 200, f"{label} server did not recover")
    require(disconnect.get("recovery_transcript_sha256") == transcript, f"{label} recovery transcript differs from the batched transcript")


def _check_run_identity(run: dict[str, Any], study: dict[str, Any], label: str) -> None:
    require(run.get("schema_version") == RUN_SCHEMA, f"{label} has an unknown schema")
    for key in ("source_commit", "model", "plan", "quality", "binary", "hardware"):
        require(run.get(key) == study.get(key), f"{label} {key} differs from the study")
    check_workload(run.get("workload"), label, repetitions=False)


def check_run(run: Any, study: dict[str, Any], index: int) -> RunResult:
    """Recompute one repetition and require it to pass the frozen gate."""
    label = f"run {index + 1}"
    run = _object(run, label)
    _check_run_identity(run, study, label)
    scheduled = check_mode(run.get("scheduled"), "scheduled", WORKLOAD["concurrent_clients"], f"{label} scheduled")
    serial = check_mode(run.get("serial"), "serial", 1, f"{label} serial")
    require(scheduled.tokens == serial.tokens, f"{label} modes generated different token counts")
    transcripts = set(scheduled.transcripts)
    require(len(transcripts) == 1, f"{label} batched transcripts disagree")
    require(transcripts == set(serial.transcripts), f"{label} batched transcript differs from the baseline")
    checks = _object(run.get("checks"), f"{label} checks")
    check_disconnect(checks.get("disconnect"), label, scheduled.transcripts[0])
    result = RunResult(scheduled, serial, scheduled.tok_s / serial.tok_s, scheduled.p95_ms / serial.p95_ms)
    _matches(result.throughput_ratio, checks.get("aggregate_throughput_ratio"), f"{label} aggregate_throughput_ratio")
    _matches({"scheduled_transcripts_agree": True, "scheduled_matches_serial": True}, checks, f"{label} checks")
    require(result.throughput_ratio > 1, f"{label} aggregate throughput ratio is not greater than 1")
    require(result.p95_ratio < 1, f"{label} p95 completion latency ratio is not less than 1")
    return result


def _sample(run: RunResult) -> dict[str, Any]:
    def mode(result: ModeResult) -> dict[str, Any]:
        return {
            "wall_ms": result.wall_ms,
            "aggregate_completion_tok_s": result.tok_s,
            "p95_completion_ms": result.p95_ms,
            "transcript_sha256": result.transcripts[0],
        }

    return {
        "scheduled": mode(run.scheduled),
        "serial": mode(run.serial),
        "aggregate_throughput_ratio": run.throughput_ratio,
        "p95_completion_latency_ratio": run.p95_ratio,
        "transcript_matches": True,
        "disconnect_recovers": True,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {"minimum": min(values), "median": _median(values), "maximum": max(values)}


def check_derived(study: dict[str, Any], runs: list[RunResult]) -> None:
    """Compare the recorded samples, summary, and checks with the recomputed ones."""
    samples = _list(study.get("samples"), "study samples")
    require(len(samples) == len(runs), "study sample count differs from the retained runs")
    for index, (run, sample) in enumerate(zip(runs, samples)):
        _matches(_sample(run), sample, f"sample {index + 1}")
    _matches(
        {
            "aggregate_throughput_ratio": _summary([run.throughput_ratio for run in runs]),
            "p95_completion_latency_ratio": _summary([run.p95_ratio for run in runs]),
        },
        study.get("summary"),
        "summary",
    )
    _matches(
        {
            "every_transcript_matches": True,
            "every_disconnect_recovers": True,
            "every_throughput_sample_wins": True,
            "every_p95_completion_sample_wins": True,
        },
        study.get("checks"),
        "checks",
    )


def check_distinct_runs(raw_runs: list[Any]) -> None:
    """Reject a run record that repeats an earlier one byte for byte.

    Records compare by the hash of their canonical JSON. Timestamps are not
    ordered: fast runs can share a second and a clock can step back.
    """
    seen: set[str] = set()
    for index, run in enumerate(raw_runs):
        canonical = json.dumps(run, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = hashlib.sha256(canonical).hexdigest()
        require(key not in seen, f"study run {index + 1} repeats an earlier run record")
        seen.add(key)


def validate(root: Path, receipt: Path, expected: Expected, offline: bool) -> None:
    """Validate one study receipt. Raise ValueError at the first failure."""
    require(
        (expected.platform, expected.target, expected.backend) == RELEASE_IDENTITY,
        "batched-service evidence supports only the Linux CUDA release identity",
    )
    _digest(expected.model_sha256, "expected model SHA-256")
    _digest(expected.binary_sha256, "expected binary SHA-256")
    try:
        relative = receipt.absolute().relative_to(root.absolute()).as_posix()
    except ValueError as error:
        raise ValueError("study receipt is outside its root") from error
    study = load_json(_regular_file(root, relative, "study receipt"), "study receipt")
    commit, records = check_source(root, expected, offline)
    check_identity(study, commit, records, expected)
    check_quality(root, study, expected)
    check_workload(study.get("workload"), "study", repetitions=True)
    check_binary(study.get("binary"), "study", commit, expected)
    check_hardware(study.get("hardware"), "study")
    raw_runs = _list(study.get("runs"), "retained runs")
    require(len(raw_runs) == REPETITIONS, f"study retains {len(raw_runs)} runs, not {REPETITIONS}")
    runs = [check_run(run, study, index) for index, run in enumerate(raw_runs)]
    check_distinct_runs(raw_runs)
    check_derived(study, runs)
    limits = _list(study.get("limits"), "study limits")
    require(all(isinstance(item, str) for item in limits), "study limits are malformed")
    require(DIGEST_LIMIT in limits, "study limits omit the response digest limit")
    require(BATCH_LIMIT in limits, "study limits omit the batch size limit")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a batched-service study receipt.")
    parser.add_argument("--root", type=Path, required=True, help="data root that holds the receipt and inputs")
    parser.add_argument("--validate-receipt", type=Path, required=True)
    parser.add_argument("--source-manifest", required=True, help="source input manifest, relative to the root")
    parser.add_argument("--offline", action="store_true", help="verify packaged files without Git history")
    parser.add_argument("--platform", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--quality-record", required=True, help="CUDA quality comparison, relative to the root")
    parser.add_argument("--quality-receipt", required=True, help="Leone quality sidecar, relative to the root")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    expected = Expected(
        args.platform, args.target, args.backend, args.model_sha256,
        args.binary_sha256, args.quality_record, args.quality_receipt, args.source_manifest,
    )
    try:
        validate(args.root, args.validate_receipt, expected, args.offline)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"batched-service validation failed: {error}", file=sys.stderr)
        return 1
    print(f"validated {args.validate_receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
