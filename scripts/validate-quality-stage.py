#!/usr/bin/env python3
"""Validate quality-stage manifests, packaged comparisons, and their artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import NoReturn


ROOT = Path(__file__).resolve().parents[1]
PINNED = (ROOT / "external/PINNED").read_text().strip()
HASH_LENGTH = 64
KLD_DEFINITION = "mean over scored positions of KL(P_oracle || P_subject) in nats, full softmax over full vocab"
MIN_LONG_CONTEXT_TOKENS = 4096
PUBLIC_SAMPLE_ROWS = 128
SAMPLE_CONTRACT = "linspace-inclusive-v1:128+task-rows"
CUDA_COMPARISON_SCHEMA = "leone.quality-comparison.v2"
GENERATION_SCHEMA = "leone.quality-generation.v1"


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return digest_bytes(encoded)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read manifest {path}: {error}")
    if not isinstance(value, dict):
        fail(f"manifest is not an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def field(value: dict, name: str, kind: type):
    result = value.get(name)
    require(isinstance(result, kind), f"manifest field {name} has the wrong type")
    return result


def valid_hash(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    require(len(value) == HASH_LENGTH and all(character in "0123456789abcdef" for character in value), f"{label} is not a SHA-256")
    return value


def valid_commit(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    require(len(value) == 40 and all(character in "0123456789abcdef" for character in value), f"{label} is not a Git object ID")
    return value


def parse_trusted_options(arguments: list[str]) -> tuple[Path, Path, dict, dict | None]:
    parser = argparse.ArgumentParser(prog="validate-quality-stage.py comparison")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("receipt_verifier", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--statistics-platform")
    parser.add_argument("--target", required=True)
    parser.add_argument("--statistics-target", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--oracle-model-sha256", required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--task-manifest-sha256")
    parser.add_argument("--sample-manifest-sha256")
    parser.add_argument("--sample-contract", required=True)
    parser.add_argument("--metric-family", required=True)
    parser.add_argument("--trusted-inputs", type=Path)
    parser.add_argument("--generation-record", type=Path)
    options = parser.parse_args(arguments)
    valid_commit(options.source_commit, "trusted native source commit")
    valid_hash(options.adapter_sha256, "trusted adapter SHA-256")
    valid_hash(options.model_sha256, "trusted subject model SHA-256")
    valid_hash(options.oracle_model_sha256, "trusted oracle model SHA-256")
    valid_hash(options.corpus_sha256, "trusted corpus SHA-256")
    require(options.platform and not os.path.isabs(options.platform), "trusted platform is invalid")
    require(options.target and not os.path.isabs(options.target), "trusted target is invalid")
    require(options.statistics_target and not os.path.isabs(options.statistics_target), "trusted statistics target is invalid")
    require(options.backend in ("metal", "cuda"), "trusted comparison backend is unsupported")
    validate_sample_anchor_option(options)
    if options.backend == "cuda":
        require(options.platform == "linux-x86_64", "trusted CUDA platform differs")
        require(options.statistics_platform and not os.path.isabs(options.statistics_platform), "trusted CUDA statistics platform is invalid")
        require(options.target == "x86_64-unknown-linux-gnu", "trusted CUDA target differs")
        valid_hash(options.task_manifest_sha256, "trusted task manifest SHA-256")
        require(options.generation_record is not None, "trusted CUDA generation record is required")
    require(options.adapter_path == "research/oracle/llama_logits.cpp", "trusted adapter path differs")
    require(options.model_family, "trusted model family is missing")
    require(options.sample_contract == SAMPLE_CONTRACT, "trusted sample contract differs")
    require(options.metric_family == "kld", "trusted metric family is not KLD")
    trusted = vars(options)
    if options.backend == "cuda":
        require(options.trusted_inputs is not None, "trusted CUDA inputs are required")
        trusted = load_trusted_cuda_inputs(options.trusted_inputs)
        for name in (
            "source_commit", "platform", "statistics_platform", "target", "statistics_target", "backend",
            "adapter_path", "adapter_sha256", "model_family", "model_sha256",
            "oracle_model_sha256", "corpus_sha256", "task_manifest_sha256",
            "sample_contract", "metric_family",
        ):
            require(trusted.get(name) == getattr(options, name), f"trusted CUDA {name} differs from CLI")
    generation = load_generation_record(options.generation_record) if options.backend == "cuda" else None
    return options.manifest, options.receipt_verifier, trusted, generation


def validate_sample_anchor_option(options: argparse.Namespace) -> None:
    if options.backend == "metal":
        valid_hash(options.sample_manifest_sha256, "trusted sample manifest SHA-256")
    else:
        require(
            options.sample_manifest_sha256 is None,
            "CUDA sample anchors require --generation-record",
        )


def load_trusted_cuda_inputs(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "trusted CUDA input file is invalid")
    body = load_json(path)
    require(body.get("schema_version") == "leone.quality-trusted-cuda.v1", "trusted CUDA input schema differs")
    trusted = field(body, "trusted", dict)
    return trusted


def load_generation_record(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "CUDA generation record is invalid")
    body = load_json(path)
    require(body.get("schema_version") == GENERATION_SCHEMA, "CUDA generation record schema differs")
    stable_strings(body, "generation")
    return body


def stable_strings(value: object, label: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            stable_strings(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            stable_strings(item, f"{label}[{index}]")
    elif isinstance(value, str):
        require(not os.path.isabs(value), f"{label} contains an absolute path")
        require(not value.startswith("~"), f"{label} contains a home-relative path")


def artifact(stage: Path, record: dict, label: str) -> Path:
    path_name = field(record, "path", str)
    require(Path(path_name).name == path_name, f"{label} path is not a stable artifact name")
    path = stage / path_name
    require(path.resolve().parent == stage.resolve(), f"{label} parent escapes the stage")
    require(path.is_file(), f"{label} is missing: {path_name}")
    require(not path.is_symlink(), f"{label} must be a regular file")
    expected = valid_hash(record.get("sha256"), f"{label} SHA-256")
    require(digest(path) == expected, f"{label} SHA-256 differs")
    if "bytes" in record:
        require(path.stat().st_size == record["bytes"], f"{label} byte count differs")
    return path


def package_artifact(root: Path, record: dict, label: str) -> Path:
    path_name = field(record, "path", str)
    path_value = Path(path_name)
    require(not path_value.is_absolute(), f"{label} path is absolute")
    require(path_value.parts and ".." not in path_value.parts, f"{label} path escapes the package")
    path = root / path_value
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), f"{label} parent is a symlink")
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        fail(f"{label} parent escapes the package")
    require(path.is_file(), f"{label} is missing: {path_name}")
    require(not path.is_symlink(), f"{label} must be a regular file")
    expected = valid_hash(record.get("sha256"), f"{label} SHA-256")
    require(digest(path) == expected, f"{label} SHA-256 differs")
    if "bytes" in record:
        require(path.stat().st_size == record["bytes"], f"{label} byte count differs")
    return path


def text_artifact(stage: Path, record: dict, label: str) -> str:
    path = artifact(stage, record, label)
    body = field(record, "body", str)
    require(body == path.read_text(), f"{label} body differs")
    return body


def text_body(record: dict, label: str) -> str:
    path_name = field(record, "path", str)
    require(path_name, f"{label} path is missing")
    require(Path(path_name).name == path_name, f"{label} path is not a stable artifact name")
    body = field(record, "body", str)
    expected = valid_hash(record.get("sha256"), f"{label} SHA-256")
    require(hashlib.sha256(body.encode()).hexdigest() == expected, f"{label} SHA-256 differs")
    return body


def run_manifest(stage: Path, record: dict, label: str) -> dict:
    manifest_path = artifact(stage, record, label)
    body = field(record, "body", dict)
    require(digest(manifest_path) == valid_hash(record["sha256"], f"{label} manifest SHA-256"), f"{label} manifest changed")
    require(body == load_json(manifest_path), f"{label} manifest body differs")
    return body


def validate_run_provenance(run: dict, record: dict, label: str) -> None:
    require(run.get("schema_version") == "leone.llama-oracle.v2", f"{label} schema differs")
    require(run.get("source_commit") == record.get("source_commit"), f"{label} source commit differs")
    valid_commit(run.get("source_commit"), f"{label} source commit")
    engine = field(run, "engine", dict)
    require(engine.get("name") == "llama.cpp", f"{label} engine differs")
    require(engine.get("git_commit") == PINNED, f"{label} llama pin differs")
    require(run.get("adapter") == record.get("adapter"), f"{label} adapter differs")
    executable = field(run, "executable", dict)
    executable_path = field(executable, "path", str)
    require(executable_path, f"{label} executable path is missing")
    require(not Path(executable_path).is_absolute(), f"{label} executable path is absolute")
    require(".." not in Path(executable_path).parts, f"{label} executable path escapes the stage")
    valid_hash(executable.get("sha256"), f"{label} executable SHA-256")
    libraries = field(executable, "linked_libraries", list)
    for library in libraries:
        require(isinstance(library, dict), f"{label} linked library is not an object")
        library_path = library.get("path", library.get("name"))
        require(isinstance(library_path, str), f"{label} linked library name is missing")
        require(Path(library_path).name == library_path, f"{label} linked library path is not stable")
        valid_hash(library.get("sha256"), f"{label} linked library SHA-256")


def validate_native_identity(run: dict, record: dict) -> dict:
    require(run.get("schema_version") == "leone.native-metal-eval.v1", "native manifest schema differs")
    valid_commit(run.get("source_commit"), "native source commit")
    valid_commit(record.get("native_source_commit"), "native stage source commit")
    valid_commit(record.get("expected_native_source_commit"), "native expected source commit")
    require(run.get("source_commit") == record.get("native_source_commit"), "native source commit differs")
    require(record.get("expected_native_source_commit") == record.get("native_source_commit"), "native expected source commit differs")
    executable = field(run, "executable", dict)
    require(Path(field(executable, "name", str)).name == executable["name"], "native executable name is not stable")
    valid_hash(executable.get("sha256"), "native executable SHA-256")
    build_info = field(executable, "build_info", dict)
    require(build_info.get("schema_version") == "leone.build-info.v1", "native build info schema differs")
    valid_commit(build_info.get("source_commit"), "native build source commit")
    require(build_info.get("source_commit") == run["source_commit"], "native build source commit differs")
    require(build_info.get("source_tree_dirty") is False, "native build is dirty")
    require(build_info.get("profile") == "release", "native build is not release")
    require(build_info.get("target") == "aarch64-apple-darwin", "native build target differs")
    build_shader = field(build_info, "metal_shader", dict)
    build_shader_name = field(build_shader, "name", str)
    require(build_shader_name and Path(build_shader_name).name == build_shader_name, "native build shader name is invalid")
    valid_hash(build_shader.get("sha256"), "native build shader SHA-256")
    require(isinstance(build_shader.get("bytes"), int) and not isinstance(build_shader["bytes"], bool) and build_shader["bytes"] > 0, "native build shader byte count is invalid")
    return build_shader


def validate_native_execution(execution: dict) -> None:
    require(execution.get("engine") == "leone", "native execution engine differs")
    require(execution.get("backend") == "metal", "native execution backend differs")
    require(execution.get("backend_registry") == "metal", "native execution backend registry differs")
    require(execution.get("device_type") == "gpu", "native execution device type differs")
    require(execution.get("device") == "metal", "native execution device differs")
    require(execution.get("kv_cache_dtype") == "F16", "native KV cache dtype differs")
    prefill = field(execution, "prefill", dict)
    prefill_path = prefill.get("path")
    require(prefill_path in ("sequential", "chunked"), "native prefill path differs")
    prefill_chunk = prefill.get("chunk_tokens")
    if prefill_path == "sequential":
        require(prefill_chunk is None, "sequential native prefill has a chunk size")
    else:
        require(isinstance(prefill_chunk, int) and not isinstance(prefill_chunk, bool) and prefill_chunk > 0, "chunked native prefill lacks a positive chunk size")


def validate_native_shader(execution: dict, build_shader: dict, stage: Path | None) -> None:
    shader = field(execution, "shader", dict)
    shader_name = field(shader, "name", str)
    require(shader_name, "native shader name is missing")
    require(Path(shader_name).name == shader_name, "native shader name is not stable")
    valid_hash(shader.get("sha256"), "native shader SHA-256")
    require(isinstance(shader.get("bytes"), int) and not isinstance(shader["bytes"], bool) and shader["bytes"] > 0, "native shader byte count is invalid")
    require(shader.get("name") == build_shader["name"], "native shader name differs from executable identity")
    require(shader.get("sha256") == build_shader["sha256"], "native shader hash differs from executable identity")
    require(shader.get("bytes") == build_shader["bytes"], "native shader size differs from executable identity")
    if stage:
        artifact(stage, {"path": shader_name, "sha256": shader["sha256"], "bytes": shader["bytes"]}, "native Metal shader")


def validate_native_text(execution: dict, stage: Path | None) -> None:
    machine_record = field(execution, "machine", dict)
    machine = text_artifact(stage, machine_record, "Leone Metal machine record") if stage else text_body(machine_record, "Leone Metal machine record")
    machine_lines = machine.splitlines()
    require_unique_machine_line(machine_lines, "platform:", "platform: darwin arm64", "native machine platform")
    require_unique_machine_line(machine_lines, "selected backend:", "selected backend: metal", "native machine backend")
    require_unique_machine_line(machine_lines, "metal device:", "metal device: detected", "native machine device")
    output_record = field(execution, "eval_stdout", dict)
    output = text_artifact(stage, output_record, "Leone Metal eval output") if stage else text_body(output_record, "Leone Metal eval output")
    require(output.startswith("logits: leone-metal.f32 ("), "native eval output does not record logits")


def validate_native_provenance(run: dict, record: dict, stage: Path | None) -> None:
    build_shader = validate_native_identity(run, record)
    execution = field(run, "execution", dict)
    validate_native_execution(execution)
    validate_native_shader(execution, build_shader, stage)
    validate_native_text(execution, stage)


def require_logits_match(run: dict, record: dict, label: str) -> None:
    run_path = field(run, "path", str)
    record_path = field(record, "path", str)
    require(run_path == record_path, f"{label} path differs")
    run_hash = valid_hash(run.get("sha256"), f"{label} run SHA-256")
    record_hash = valid_hash(record.get("sha256"), f"{label} record SHA-256")
    require(run_hash == record_hash, f"{label} SHA-256 differs")
    require(isinstance(run.get("bytes"), int) and not isinstance(run["bytes"], bool) and run["bytes"] > 0, f"{label} run byte count is invalid")
    require(isinstance(record.get("bytes"), int) and not isinstance(record["bytes"], bool) and record["bytes"] > 0, f"{label} record byte count is invalid")
    require(run["bytes"] == record["bytes"], f"{label} byte count differs")
    require(run.get("encoding") == "row-major-f32-le", f"{label} run encoding differs")
    require(record.get("encoding") == "row-major-f32-le", f"{label} record encoding differs")


def validate_logits_metadata(record: dict, label: str, stable_path: bool = True) -> None:
    path_name = field(record, "path", str)
    path = Path(path_name)
    if stable_path:
        require(path.name == path_name, f"{label} path is not stable")
    else:
        require(not path.is_absolute() and ".." not in path.parts, f"{label} path escapes the package")
    valid_hash(record.get("sha256"), f"{label} SHA-256")
    require(isinstance(record.get("bytes"), int) and not isinstance(record["bytes"], bool) and record["bytes"] > 0, f"{label} byte count is invalid")
    require(record.get("encoding") == "row-major-f32-le", f"{label} encoding differs")


def require_logits_shape(run: dict, record: dict, label: str) -> None:
    input_record = field(run, "input", dict)
    rows = input_record.get("rows")
    model = field(run, "model", dict)
    vocab = input_record.get("vocab_size", model.get("vocab_size"))
    require(isinstance(rows, int) and not isinstance(rows, bool) and rows > 0, f"{label} row count is invalid")
    require(isinstance(vocab, int) and not isinstance(vocab, bool) and vocab > 0, f"{label} vocabulary size is invalid")
    require(record.get("bytes") == rows * vocab * 4, f"{label} byte count does not match its shape")


def validate_input_shape(run: dict, expected: dict, label: str) -> None:
    input_record = field(run, "input", dict)
    tokens = field(input_record, "tokens", dict)
    count = task_integer(tokens.get("count"), f"{label} token count", 2)
    require(tokens.get("bytes") == count * 4, f"{label} token byte count differs")
    window = task_integer(input_record.get("window_tokens"), f"{label} window", 2)
    stride = task_integer(input_record.get("stride_tokens"), f"{label} stride", 1)
    rows = task_integer(input_record.get("rows"), f"{label} rows", 1)
    vocab = task_integer(input_record.get("vocab_size"), f"{label} vocabulary", 1)
    require(window <= count, f"{label} window exceeds token count")
    require(window <= rows, f"{label} window exceeds scored rows")
    require(rows == count - 1, f"{label} rows do not equal token count minus one")
    require(stride == window - 1, f"{label} stride differs from window minus one")
    model = field(run, "model", dict)
    require(vocab == task_integer(model.get("vocab_size"), f"{label} model vocabulary", 1), f"{label} vocabulary differs from model")
    for name in ("tokens", "window_tokens", "stride_tokens", "rows", "vocab_size"):
        require(input_record.get(name) == expected.get(name), f"{label} input {name} differs")


def logsumexp(values: tuple[float, ...], label: str) -> float:
    require(not any(math.isnan(value) for value in values), f"{label} contains NaN")
    maximum = max(values)
    require(math.isfinite(maximum), f"{label} has no finite value")
    total = sum(math.exp(value - maximum) for value in values)
    result = maximum + math.log(total)
    require(math.isfinite(result), f"{label} logsumexp is nonfinite")
    return result


def row_stats(oracle: tuple[float, ...], subject: tuple[float, ...], label: str) -> tuple[float, bool]:
    require(len(oracle) == len(subject), f"{label} vocabulary differs")
    oracle_lse = logsumexp(oracle, f"{label} oracle")
    subject_lse = logsumexp(subject, f"{label} subject")
    value = 0.0
    for oracle_logit, subject_logit in zip(oracle, subject):
        if oracle_logit == float("-inf"):
            continue
        require(math.isfinite(oracle_logit) and math.isfinite(subject_logit), f"{label} contains a nonfinite scored logit")
        log_p = oracle_logit - oracle_lse
        log_q = subject_logit - subject_lse
        value += math.exp(log_p) * (log_p - log_q)
    if -1e-12 <= value <= 0.0:
        value = 0.0
    require(math.isfinite(value) and value >= 0.0, f"{label} KLD is invalid")
    oracle_top = canonical_argmax(oracle, f"{label} oracle")
    subject_top = canonical_argmax(subject, f"{label} subject")
    return value, oracle_top == subject_top


def canonical_argmax(values: tuple[float, ...], label: str) -> int:
    require(values, f"{label} row is empty")
    require(not any(math.isnan(value) for value in values), f"{label} contains NaN")
    best = 0
    for index, value in enumerate(values[1:], 1):
        if value > values[best] or (value == 0.0 == values[best] and math.copysign(1.0, value) > math.copysign(1.0, values[best])):
            best = index
    require(math.isfinite(values[best]), f"{label} has no finite value")
    return best


def percentile(sorted_values: list[float], fraction: float) -> float:
    rank = fraction * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (rank - lower)


def recompute_quality(oracle_path: Path, subject_path: Path, rows: int, vocab: int) -> dict:
    row_bytes = vocab * 4
    expected = rows * row_bytes
    require(oracle_path.stat().st_size == expected, "oracle quality artifact shape differs")
    require(subject_path.stat().st_size == expected, "subject quality artifact shape differs")
    values: list[float] = []
    top_matches = 0
    with oracle_path.open("rb") as oracle, subject_path.open("rb") as subject:
        for row in range(rows):
            oracle_values = struct.unpack(f"<{vocab}f", oracle.read(row_bytes))
            subject_values = struct.unpack(f"<{vocab}f", subject.read(row_bytes))
            value, top_match = row_stats(oracle_values, subject_values, f"quality row {row}")
            values.append(value)
            top_matches += int(top_match)
    values.sort()
    return {
        "mean": sum(values) / rows,
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "max": values[-1],
        "top1_agreement": top_matches / rows,
    }


def require_metric_matches(actual: object, expected: float, label: str) -> None:
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{label} is invalid")
    require(math.isfinite(actual), f"{label} is nonfinite")
    require(math.isclose(actual, expected, rel_tol=2e-8, abs_tol=2e-10), f"{label} differs from recomputed value")


def task_integer(value: object, label: str, minimum: int = 1) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, f"{label} is invalid")
    return value


def task_argmax(path: Path, row: int, vocab: int, label: str) -> int:
    with path.open("rb") as source:
        source.seek(row * vocab * 4)
        raw = source.read(vocab * 4)
    require(len(raw) == vocab * 4, f"{label} row is truncated")
    values = struct.unpack(f"<{vocab}f", raw)
    return canonical_argmax(values, label)


def task_tokens(path: Path, start: int, count: int) -> list[int]:
    with path.open("rb") as source:
        source.seek(start * 4)
        raw = source.read(count * 4)
    require(len(raw) == count * 4, "quality task answer extends beyond the token artifact")
    return list(struct.unpack(f"<{count}I", raw))


def expected_sample_indices(rows: int, task_rows: list[int]) -> tuple[int, list[int]]:
    requested = min(PUBLIC_SAMPLE_ROWS, rows)
    if requested == 1:
        base = [0]
    else:
        base = [index * (rows - 1) // (requested - 1) for index in range(requested)]
    indices = sorted(set(base + task_rows))
    require(all(0 <= row < rows for row in indices), "quality sample row is outside the source")
    return requested, indices


def validate_sample_body(
    body: dict,
    artifact_root: Path,
    source_root: Path,
    expected_input: dict,
    expected_logits: dict,
    task_rows: list[int],
    require_source_artifacts: bool = False,
) -> dict:
    require(body.get("schema_version") == "leone.quality-sample.v1", "unknown quality sample schema")
    require(body.get("provenance") == {
        "full_logits": "generation-only",
        "row_hashes": "sha256 of each packaged row in source order",
        "offline_replay": "sampled rows only",
    }, "quality sample provenance differs")
    source = field(body, "source", dict)
    rows = task_integer(source.get("rows"), "quality sample source rows", 1)
    vocab = task_integer(source.get("vocab_size"), "quality sample source vocabulary", 1)
    require(rows == expected_input.get("rows"), "quality sample source rows differ")
    require(vocab == expected_input.get("vocab_size"), "quality sample source vocabulary differs")
    source_tokens = field(source, "tokens", dict)
    require(source_tokens == expected_input["tokens"], "quality sample source tokens differ")
    package_artifact(source_root, source_tokens, "quality sample source tokens")
    source_logits = field(source, "logits", dict)
    for label, expected in expected_logits.items():
        actual = field(source_logits, label, dict)
        require(actual == expected, f"quality sample source {label} logits differ")
        validate_logits_metadata(actual, f"quality sample source {label} logits", False)
        if require_source_artifacts:
            package_artifact(source_root, actual, f"quality sample source {label} logits")
    sampling = field(body, "sampling", dict)
    require(sampling.get("algorithm") == "linspace-inclusive-v1", "quality sample algorithm differs")
    requested, indices = expected_sample_indices(rows, task_rows)
    require(sampling.get("requested_rows") == requested, "quality sample requested row count differs")
    require(sampling.get("source_rows") == rows, "quality sample source row count differs")
    require(sampling.get("sample_count") == len(indices), "quality sample count differs")
    require(sampling.get("indices") == indices, "quality sample indices differ")
    require(sampling.get("task_rows") == task_rows, "quality sample task rows differ")
    artifacts = field(body, "artifacts", dict)
    token_artifact = field(artifacts, "tokens", dict)
    token_path = package_artifact(artifact_root, token_artifact, "quality sample tokens")
    token_count = len(indices) + 1
    require(token_artifact.get("encoding") == "u32le", "quality sample token encoding differs")
    require(token_artifact.get("bytes") == token_count * 4, "quality sample token size differs")
    source_values = struct.unpack(f"<{source_tokens['count']}I", (source_root / source_tokens["path"]).read_bytes())
    sampled_values = struct.unpack(f"<{token_count}I", token_path.read_bytes())
    expected_values = (source_values[indices[0]],) + tuple(source_values[row + 1] for row in indices)
    require(sampled_values == expected_values, "quality sample tokens differ from source rows")
    sample_logits = field(artifacts, "logits", dict)
    require(set(sample_logits) == set(expected_logits), "quality sample logits labels differ")
    for label in expected_logits:
        sample = field(sample_logits, label, dict)
        path = package_artifact(artifact_root, sample, f"quality sample {label} logits")
        validate_logits_metadata(sample, f"quality sample {label} logits", False)
        row_bytes = vocab * 4
        require(sample["bytes"] == len(indices) * row_bytes, f"quality sample {label} size differs")
        row_hashes = sample.get("source_row_sha256")
        require(isinstance(row_hashes, list) and len(row_hashes) == len(indices), f"quality sample {label} row hashes differ")
        with path.open("rb") as source_file:
            for row_hash in row_hashes:
                valid_hash(row_hash, f"quality sample {label} row hash")
                data = source_file.read(row_bytes)
                require(len(data) == row_bytes, f"quality sample {label} row is truncated")
                require(digest_bytes(data) == row_hash, f"quality sample {label} row hash differs")
    return body


def validate_sample_manifest(
    sample_manifest: Path,
    artifact_root: Path,
    export_stage: Path,
    metal_stage: Path,
    task_rows: list[int],
) -> dict:
    body = load_json(sample_manifest)
    exported = load_json(export_stage / "oracle-stage.json")
    metal = load_json(metal_stage / "metal-stage.json")
    expected_logits = {
        "oracle": exported["oracle"]["logits"],
        "llama_cpp": metal["subject"]["logits"],
        "leone": metal["native"]["logits"],
    }
    return validate_sample_body(body, artifact_root, export_stage, exported["input"], expected_logits, task_rows)


def task_text_bytes(record: dict, label: str) -> bytes:
    require(record.get("encoding") == "utf-8", f"{label} encoding differs")
    text = field(record, "text", str)
    raw = text.encode("utf-8")
    require(record.get("bytes") == len(raw), f"{label} byte count differs")
    require(record.get("sha256") == digest_bytes(raw), f"{label} SHA-256 differs")
    return raw


def task_model_identity_record(tokenization: dict, expected: dict, label: str) -> bool:
    actual = field(tokenization, "model", dict)
    require(actual.get("name") == expected.get("name"), f"{label} name differs")
    valid_hash(actual.get("sha256"), f"{label} SHA-256")
    require(actual.get("sha256") == expected.get("sha256"), f"{label} SHA-256 differs")
    return True


def task_model_identity(task: dict, stage: dict, name: str) -> None:
    expected = field(stage["model"], name, dict)
    actual = field(field(task, "model", dict), name, dict)
    for key in ("name", "sha256", "storage_type"):
        require(actual.get(key) == expected.get(key), f"quality task {name} {key} differs")
    valid_hash(actual.get("sha256"), f"quality task {name} SHA-256")


def validate_task_context(task: dict, exported: dict, metal: dict) -> tuple[dict, int, int, dict, str]:
    task_spec = field(task, "task", dict)
    require(task_spec.get("name") == "needle-exact-answer", "quality task is not an exact-answer task")
    context_tokens = task_integer(task_spec.get("context_tokens"), "quality task context_tokens", MIN_LONG_CONTEXT_TOKENS)
    minimum_tokens = task_integer(task_spec.get("minimum_context_tokens"), "quality task minimum_context_tokens", MIN_LONG_CONTEXT_TOKENS)
    require(context_tokens >= minimum_tokens, "quality task context is below its declared minimum")
    input_record = field(exported, "input", dict)
    require(context_tokens == input_record.get("window_tokens"), "quality task context differs from the evaluation window")
    task_model_identity(task, exported, "oracle")
    task_model_identity(task, metal, "subject")
    model = field(task, "model", dict)
    require(isinstance(model.get("family"), str) and model["family"], "quality task model family is missing")
    tokenizer = field(task, "tokenizer", dict)
    tokenizer_name = field(tokenizer, "name", str)
    require(tokenizer_name, "quality task tokenizer name is missing")
    oracle_run = exported["oracle"]["manifest"]["body"]
    llama_run = metal["subject"]["manifest"]["body"]
    native_run = metal["native"]["manifest"]["body"]
    for run, label in ((oracle_run, "oracle"), (llama_run, "Metal comparator"), (native_run, "native Metal")):
        require(run["model"].get("tokenizer") == tokenizer_name, f"quality task {label} tokenizer differs")
    require(field(task, "corpus", dict) == exported["corpus"], "quality task corpus differs")
    task_input = field(task, "input", dict)
    require(task_input == input_record, "quality task input differs")
    return task_spec, context_tokens, minimum_tokens, input_record, tokenizer_name


def validate_task_answer(task_spec: dict, input_record: dict, context_tokens: int, exported: dict) -> tuple[dict, dict, dict, list[int], list[int], bytes]:
    task_spec_prompt = field(task_spec, "prompt", dict)
    task_spec_needle = field(task_spec, "needle", dict)
    prompt_bytes = task_text_bytes(task_spec_prompt, "quality task prompt")
    needle_bytes = task_text_bytes(task_spec_needle, "quality task needle")
    combined = prompt_bytes + needle_bytes
    require(digest_bytes(combined) == exported["corpus"]["sha256"], "quality task prompt and needle differ from corpus")
    answer = field(task_spec, "answer", dict)
    answer_row = task_integer(answer.get("row"), "quality task answer row", 0)
    require(answer.get("text") == task_spec_needle["text"], "quality task answer text differs from needle")
    require(answer.get("sha256") == task_spec_needle["sha256"], "quality task answer hash differs from needle")
    answer_tokens = answer.get("token_ids")
    require(isinstance(answer_tokens, list) and answer_tokens, "quality task answer token_ids is empty")
    require(all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < input_record["vocab_size"] for token in answer_tokens), "quality task answer token ID is invalid")
    require(answer_row >= context_tokens - 1, "quality task answer is before the declared context")
    require(answer_row + len(answer_tokens) <= input_record["rows"], "quality task answer exceeds scored rows")
    task_rows = list(range(answer_row, answer_row + len(answer_tokens)))
    return task_spec_prompt, task_spec_needle, answer, answer_tokens, task_rows, combined


def validate_task_tokenization(
    task: dict,
    exported: dict,
    export_stage: Path,
    input_record: dict,
    task_spec_prompt: dict,
    task_spec_needle: dict,
    combined: bytes,
    answer_row: int,
    answer_tokens: list[int],
) -> None:
    tokenization = field(task, "tokenization", dict)
    require(tokenization.get("method") == "llama-tokenize", "quality task tokenizer method differs")
    source = field(tokenization, "source", dict)
    require(source == {
        "name": "llama.cpp",
        "git_commit": PINNED,
        "pin_file": "external/PINNED",
        "provenance": "declared",
        "trust": "operator-attested",
    }, "quality task tokenizer source differs")
    tokenizer_executable = field(tokenization, "executable", dict)
    tokenizer_executable_name = field(tokenizer_executable, "name", str)
    require(Path(tokenizer_executable_name).name == tokenizer_executable_name, "quality task tokenizer executable name is not stable")
    valid_hash(tokenizer_executable.get("sha256"), "quality task tokenizer executable SHA-256")
    require(isinstance(tokenizer_executable.get("bytes"), int) and not isinstance(tokenizer_executable["bytes"], bool) and tokenizer_executable["bytes"] > 0, "quality task tokenizer executable byte count is invalid")
    require(task_model_identity_record(tokenization, exported["model"]["oracle"], "tokenizer model"), "quality task tokenizer model differs")
    require(tokenization.get("prompt_sha256") == task_spec_prompt["sha256"], "quality task tokenizer prompt differs")
    require(tokenization.get("needle_sha256") == task_spec_needle["sha256"], "quality task tokenizer needle differs")
    require(tokenization.get("combined_sha256") == digest_bytes(combined), "quality task tokenizer input differs")
    require(tokenization.get("input_tokens_sha256") == input_record["tokens"]["sha256"], "quality task tokenizer token input differs")
    packed_answer = struct.pack(f"<{len(answer_tokens)}I", *answer_tokens)
    require(tokenization.get("answer_tokens_sha256") == hashlib.sha256(packed_answer).hexdigest(), "quality task tokenizer answer differs")
    expected_tokens = task_tokens(export_stage / input_record["tokens"]["path"], answer_row + 1, len(answer_tokens))
    require(expected_tokens == answer_tokens, "quality task answer does not match the independently tokenized stream")


def task_logits(
    exported: dict,
    metal: dict,
    export_stage: Path,
    metal_stage: Path,
    sample_manifest: Path | None,
    task_rows: list[int],
    input_record: dict,
) -> tuple[dict | None, dict, dict]:
    vocab = input_record["vocab_size"]
    sample_body = None
    sample_root = None
    if sample_manifest is not None:
        sample_root = sample_manifest.parent.parent
        sample_body = validate_sample_manifest(sample_manifest, sample_root, export_stage, metal_stage, task_rows)
        sample_logits = sample_body["artifacts"]["logits"]
        logits = {
            "oracle": (sample_root / sample_logits["oracle"]["path"], sample_logits["oracle"]),
            "llama_cpp_metal": (sample_root / sample_logits["llama_cpp"]["path"], sample_logits["llama_cpp"]),
            "leone_metal": (sample_root / sample_logits["leone"]["path"], sample_logits["leone"]),
        }
        row_positions = {row: position for position, row in enumerate(sample_body["sampling"]["indices"])}
    else:
        logits = {
            "oracle": (export_stage / exported["oracle"]["logits"]["path"], exported["oracle"]["logits"]),
            "llama_cpp_metal": (metal_stage / metal["subject"]["logits"]["path"], metal["subject"]["logits"]),
            "leone_metal": (metal_stage / metal["native"]["logits"]["path"], metal["native"]["logits"]),
        }
        row_positions = {row: row for row in range(input_record["rows"])}
    return sample_body, logits, row_positions


def validate_task_logits(
    logits: dict,
    answer_tokens: list[int],
    answer_row: int,
    row_positions: dict[int, int],
    vocab: int,
    sample_body: dict | None,
) -> dict:
    argmax = {}
    artifacts = {}
    for label, (path, record) in logits.items():
        if sample_body is None:
            artifact(path.parent, {"path": path.name, "sha256": record["sha256"], "bytes": record["bytes"]}, f"quality task {label} logits")
        argmax[label] = [task_argmax(path, row_positions[answer_row + offset], vocab, f"quality task {label}") for offset in range(len(answer_tokens))]
        artifacts[label] = {key: record[key] for key in ("path", "sha256", "bytes", "encoding")}
    for label, values in argmax.items():
        require(values == answer_tokens, f"quality task {label} answer differs")
    return {"artifacts": artifacts, "argmax": {"expected": answer_tokens, **argmax}}


def validate_task(task_path: Path, export_stage: Path, metal_stage: Path, sample_manifest: Path | None = None) -> dict:
    task = load_json(task_path)
    require(task.get("schema_version") == "leone.quality-task.v2", "unknown quality task schema")
    stable_strings(task)
    exported, metal = validate_compare(export_stage, metal_stage, sample_manifest is None)
    task_spec, context_tokens, minimum_tokens, input_record, _tokenizer_name = validate_task_context(task, exported, metal)
    task_spec_prompt, task_spec_needle, answer, answer_tokens, task_rows, combined = validate_task_answer(task_spec, input_record, context_tokens, exported)
    validate_task_tokenization(task, exported, export_stage, input_record, task_spec_prompt, task_spec_needle, combined, answer["row"], answer_tokens)
    sample_body, logits, row_positions = task_logits(exported, metal, export_stage, metal_stage, sample_manifest, task_rows, input_record)
    answer_row = answer["row"]
    result = validate_task_logits(logits, answer_tokens, answer_row, row_positions, input_record["vocab_size"], sample_body)
    return {
        "schema_version": "leone.quality-task-result.v2",
        "task": {"name": task_spec["name"], "context_tokens": context_tokens, "minimum_context_tokens": minimum_tokens, "prompt_sha256": task_spec_prompt["sha256"], "needle_sha256": task_spec_needle["sha256"], "answer": answer},
        "model": {"oracle": exported["model"]["oracle"], "subject": metal["model"]["subject"]},
        "corpus": exported["corpus"],
        "input": input_record,
        **result,
        "passed": True,
    }


def common_checks(record: dict, stage: Path, expected_stage: str) -> None:
    require(record.get("schema_version") == "leone.quality-stage.v1", "unknown quality-stage schema")
    require(record.get("stage") == expected_stage, f"expected {expected_stage} stage")
    valid_commit(record.get("source_commit"), "source commit")
    stable_strings(record)
    engine = field(record, "engine", dict)
    require(engine.get("name") == "llama.cpp", "quality stage uses the wrong engine")
    require(engine.get("git_commit") == PINNED, "quality stage does not use external/PINNED")
    adapter = field(record, "adapter", dict)
    require(adapter.get("path") == "research/oracle/llama_logits.cpp", "unexpected oracle adapter")
    adapter_sha256 = valid_hash(adapter.get("sha256"), "adapter SHA-256")
    adapter_artifact = field(adapter, "artifact", dict)
    require(adapter_artifact.get("path") == "llama_logits.cpp", "oracle adapter artifact path differs")
    require(adapter_artifact.get("sha256") == adapter_sha256, "oracle adapter artifact hash differs")
    require(isinstance(adapter_artifact.get("bytes"), int) and not isinstance(adapter_artifact["bytes"], bool) and adapter_artifact["bytes"] > 0, "oracle adapter artifact byte count is invalid")
    artifact(stage, adapter_artifact, "oracle adapter source")
    input_record = field(record, "input", dict)
    tokens = field(input_record, "tokens", dict)
    artifact(stage, tokens, "token artifact")
    require(tokens.get("encoding") == "u32le", "token artifact is not u32le")
    token_count = task_integer(tokens.get("count"), "stage token count", 2)
    require(tokens.get("bytes") == token_count * 4, "stage token byte count differs")
    window = task_integer(input_record.get("window_tokens"), "stage window", 2)
    stride = task_integer(input_record.get("stride_tokens"), "stage stride", 1)
    rows = task_integer(input_record.get("rows"), "stage rows", 1)
    vocab = task_integer(input_record.get("vocab_size"), "stage vocabulary", 1)
    require(rows == token_count - 1, "stage rows do not equal token count minus one")
    require(stride == window - 1, "stage stride differs from window minus one")
    require(input_record.get("tokens", {}).get("count") == token_count, "stage token count differs")
    require(input_record.get("tokens", {}).get("bytes") == token_count * 4, "stage token bytes differ")
    model_records = field(record, "model", dict)
    oracle_model = model_records.get("oracle")
    require(isinstance(oracle_model, dict), "stage oracle model record is missing")
    require(vocab == task_integer(oracle_model.get("vocab_size"), "stage model vocabulary", 1), "stage vocabulary differs from oracle model")


def validate_export(stage: Path, require_logits: bool = True) -> dict:
    record = load_json(stage / "oracle-stage.json")
    common_checks(record, stage, "oracle-export")
    oracle_model = field(field(record, "model", dict), "oracle", dict)
    require(oracle_model.get("storage_type") in ("BF16", "F16"), "oracle model is not full precision")
    valid_hash(oracle_model.get("sha256"), "oracle model SHA-256")
    subject_model = field(field(record, "model", dict), "subject", dict)
    valid_hash(subject_model.get("sha256"), "subject model SHA-256")
    require(isinstance(subject_model.get("storage_type"), str) and subject_model["storage_type"], "subject model storage type is missing")
    corpus = field(record, "corpus", dict)
    valid_hash(corpus.get("sha256"), "corpus SHA-256")
    oracle = field(record, "oracle", dict)
    run_manifest(stage, field(oracle, "manifest", dict), "oracle run manifest")
    oracle_logits = field(oracle, "logits", dict)
    if require_logits:
        artifact(stage, oracle_logits, "oracle logits")
    else:
        validate_logits_metadata(oracle_logits, "oracle logits")
    run = field(oracle, "manifest", dict)["body"]
    validate_run_provenance(run, record, "oracle run")
    require(run.get("corpus") == record["corpus"], "oracle corpus provenance differs")
    execution = field(run, "execution", dict)
    require(execution.get("device") == "cpu", "export oracle must run on CPU")
    require(isinstance(execution.get("backend_registry"), str) and execution["backend_registry"].lower() == "cpu", "export oracle backend is not CPU")
    require(isinstance(execution.get("device_type"), str) and execution["device_type"], "export oracle device type is missing")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], "export oracle device name is missing")
    require(isinstance(execution.get("device_description"), str), "export oracle device description is missing")
    require(run["model"]["sha256"] == oracle_model["sha256"], "oracle model and run manifest differ")
    for name in ("vocab_size", "architecture", "tokenizer"):
        require(oracle_model.get(name) == run["model"].get(name), f"oracle model {name} differs")
    require(run["input"]["tokens"]["sha256"] == record["input"]["tokens"]["sha256"], "oracle token input differs")
    require(run["input"]["tokens"].get("encoding") == "u32le", "oracle token encoding differs")
    validate_input_shape(run, record["input"], "oracle")
    require(run["model"].get("storage_type") == oracle_model["storage_type"], "oracle storage type differs")
    require_logits_match(run["logits"], oracle["logits"], "oracle logits record")
    require_logits_shape(run, oracle["logits"], "oracle logits")
    return record


def validate_metal(stage: Path, require_logits: bool = True) -> dict:
    record = load_json(stage / "metal-stage.json")
    common_checks(record, stage, "metal-subject")
    parent = field(record, "parent", dict)
    parent_path = field(parent, "path", str)
    require(Path(parent_path).name == parent_path, "parent manifest path is not stable")
    parent_file = stage / parent_path
    require(parent_file.is_file(), "parent manifest snapshot is missing")
    valid_hash(parent.get("manifest_sha256"), "parent manifest SHA-256")
    require(digest(parent_file) == parent["manifest_sha256"], "parent manifest snapshot changed")
    require(field(parent, "manifest", dict) == load_json(parent_file), "parent manifest body differs")
    subject_model = field(field(record, "model", dict), "subject", dict)
    valid_hash(subject_model.get("sha256"), "subject model SHA-256")
    oracle = field(record, "oracle", dict)
    oracle_logits = field(oracle, "logits", dict)
    if require_logits:
        artifact(stage, oracle_logits, "copied oracle logits")
    else:
        validate_logits_metadata(oracle_logits, "copied oracle logits")
    subject = field(record, "subject", dict)
    subject_logits = field(subject, "logits", dict)
    if require_logits:
        artifact(stage, subject_logits, "Metal logits")
    else:
        validate_logits_metadata(subject_logits, "Metal logits")
    run = run_manifest(stage, field(subject, "manifest", dict), "Metal run manifest")
    validate_run_provenance(run, record, "Metal run")
    execution = field(run, "execution", dict)
    require(execution.get("device") == "metal", "subject run is not Metal")
    require(isinstance(execution.get("backend_registry"), str) and execution["backend_registry"].lower() in ("mtl", "metal"), "subject backend is not Metal")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], "Metal device name is missing")
    require(isinstance(execution.get("device_description"), str), "Metal device description is missing")
    require(run["model"]["sha256"] == subject_model["sha256"], "subject model and run manifest differ")
    require(run["model"].get("storage_type") == "Q4_K - Medium", "Metal subject is not Q4_K - Medium")
    for name in ("vocab_size", "architecture", "tokenizer"):
        require(name in run["model"], f"Metal model metadata is missing: {name}")
        require(subject_model.get(name) == run["model"][name], f"Metal subject {name} differs")
    require(run["input"]["tokens"]["sha256"] == record["input"]["tokens"]["sha256"], "Metal token input differs")
    require(run["input"]["tokens"].get("encoding") == "u32le", "Metal token encoding differs")
    validate_input_shape(run, record["input"], "Metal")
    require_logits_match(run["logits"], subject["logits"], "Metal logits record")
    require_logits_shape(run, subject["logits"], "Metal logits")
    parent_record = parent["manifest"]
    require(record["model"]["oracle"] == parent_record["model"]["oracle"], "Metal oracle model differs from parent")
    require(record["model"]["subject"]["name"] == parent_record["model"]["subject"]["name"], "Metal subject model name differs from parent")
    require(record["model"]["subject"]["sha256"] == parent_record["model"]["subject"]["sha256"], "Metal subject model hash differs from parent")
    require(record["model"]["subject"].get("storage_type") == "Q4_K - Medium", "Metal subject storage type is missing")
    require(record["input"] == parent_record["input"], "Metal inputs differ from parent")
    require(record["corpus"] == parent_record["corpus"], "Metal corpus differs from parent")
    require(record["oracle"]["logits"]["sha256"] == parent_record["oracle"]["logits"]["sha256"], "Metal oracle differs from parent")
    native = field(record, "native", dict)
    native_logits = field(native, "logits", dict)
    if require_logits:
        artifact(stage, native_logits, "native Leone Metal logits")
    else:
        validate_logits_metadata(native_logits, "native Leone Metal logits")
    native_run = run_manifest(stage, field(native, "manifest", dict), "native Leone Metal manifest")
    validate_native_provenance(native_run, record, stage)
    require(native_run["model"]["sha256"] == subject_model["sha256"], "native model and stage differ")
    require(native_run["model"].get("storage_type") == "Q4_K - Medium", "native subject is not Q4_K - Medium")
    for name in ("vocab_size", "architecture", "tokenizer"):
        require(native_run["model"].get(name) == subject_model[name], f"native subject {name} differs")
    require(native_run["input"]["tokens"]["sha256"] == record["input"]["tokens"]["sha256"], "native token input differs")
    require(native_run["input"]["tokens"].get("encoding") == "u32le", "native token encoding differs")
    validate_input_shape(native_run, record["input"], "native")
    require_logits_match(native_run["logits"], native_logits, "native logits record")
    require_logits_shape(native_run, native_logits, "native Leone Metal logits")
    require(record.get("native_execution") == native_run["execution"], "native execution record differs")
    return record


def validate_compare(export_stage: Path, metal_stage: Path, require_logits: bool = True) -> tuple[dict, dict]:
    exported = validate_export(export_stage, require_logits)
    metal = validate_metal(metal_stage, require_logits)
    require(metal["parent"]["manifest_sha256"] == digest(export_stage / "oracle-stage.json"), "Metal parent stage differs")
    require(metal["engine"] == exported["engine"], "Metal and oracle llama pins differ")
    require(metal["adapter"] == exported["adapter"], "Metal and oracle adapters differ")
    require(metal["input"] == exported["input"], "Metal and oracle inputs differ")
    require(metal["model"]["oracle"] == exported["model"]["oracle"], "Metal and oracle model records differ")
    require(metal["model"]["subject"]["name"] == exported["model"]["subject"]["name"], "Metal subject model name differs")
    require(metal["model"]["subject"]["sha256"] == exported["model"]["subject"]["sha256"], "Metal subject model hash differs")
    require(metal["model"]["subject"].get("storage_type") == "Q4_K - Medium", "Metal subject storage type is missing")
    require(metal["oracle"]["logits"]["sha256"] == exported["oracle"]["logits"]["sha256"], "Metal copied oracle differs")
    require(metal["oracle"]["logits"]["path"] == exported["oracle"]["logits"]["path"], "Metal oracle artifact name differs")
    return exported, metal


def verify_quality_receipt(path: Path, quality: dict, receipt_verifier: Path | None, label: str) -> Path:
    receipt = artifact(path.parent, quality, f"{label} quality receipt")
    if receipt_verifier is None:
        return receipt
    try:
        subprocess.run(
            [str(receipt_verifier), "quality", str(receipt)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", "") or str(error)
        fail(f"{label} canonical receipt parser rejected the receipt: {detail.strip()}")
    return receipt


def validate_quality_metric_values(metrics: dict, label: str) -> None:
    kld = field(metrics, "kld", dict)
    require(kld.get("definition") == KLD_DEFINITION, f"{label} quality KLD definition differs")
    for name in ("mean", "p50", "p99"):
        value = kld.get(name)
        require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{label} quality metric {name} is invalid")
        require(value >= 0.0, f"{label} quality metric {name} is negative")
    value = kld.get("max")
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{label} quality metric max is invalid")
    require(value >= 0.0, f"{label} quality metric max is negative")
    value = metrics.get("top1_agreement")
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value), f"{label} quality metric top1_agreement is invalid")
    require(0.0 <= value <= 1.0, f"{label} quality metric top1_agreement is outside [0, 1]")


def validate_quality_link(
    path: Path,
    quality: dict,
    export_body: dict,
    subject: dict,
    subject_engine: dict,
    oracle_logits: Path,
    subject_logits: Path,
    sample_body: dict,
    sample_oracle: dict,
    sample_subject: dict,
    receipt_verifier: Path | None,
    label: str,
) -> None:
    receipt = verify_quality_receipt(path, quality, receipt_verifier, label)
    body = field(quality, "receipt", dict)
    require(body == load_json(receipt), f"{label} quality receipt body differs")
    require(body.get("schema_version") == 3, f"{label} quality receipt has the wrong schema")
    require(body["corpus"]["sha256"] == export_body["corpus"]["sha256"], f"{label} quality corpus differs")
    sample_count = sample_body["sampling"]["sample_count"]
    require(body["corpus"]["n_tokens_scored"] == sample_count, f"{label} quality row count differs")
    require(body["sample_count"] == body["corpus"]["n_tokens_scored"], f"{label} quality sample count differs")
    require(body["oracle"]["engine"] == {"name": "llama.cpp", "git_commit": PINNED}, f"{label} quality oracle pin differs")
    require(body["oracle"]["dtype"] == export_body["model"]["oracle"]["storage_type"].lower(), f"{label} quality oracle is not full precision")
    require(body["subject"]["engine"] == subject_engine, f"{label} quality subject engine differs")
    metrics = field(body, "metrics", dict)
    validate_quality_metric_values(metrics, label)
    require(body["oracle"]["artifact_sha256"] == sample_oracle["sha256"], f"{label} quality oracle differs")
    require(body["subject"]["model_artifact"]["sha256"] == export_body["model"]["subject"]["sha256"], f"{label} quality subject model differs")
    require(body["subject"]["logits_artifact"]["sha256"] == sample_subject["sha256"], f"{label} quality logits differ")
    recomputed = recompute_quality(
        oracle_logits,
        subject_logits,
        sample_count,
        export_body["input"]["vocab_size"],
    )
    kld = metrics["kld"]
    for name in ("mean", "p50", "p99", "max"):
        require_metric_matches(kld[name], recomputed[name], f"{label} quality KLD {name}")
    require_metric_matches(metrics["top1_agreement"], recomputed["top1_agreement"], f"{label} quality top1 agreement")


def native_platform(execution: dict) -> str:
    machine = field(execution, "machine", dict)
    body = field(machine, "body", str)
    require_unique_machine_line(body.splitlines(), "platform:", "platform: darwin arm64", "native machine platform")
    return "darwin-arm64"


def require_unique_machine_line(lines: list[str], prefix: str, expected: str, label: str) -> None:
    matches = [line for line in lines if line.startswith(prefix)]
    require(matches == [expected], f"{label} field is missing, duplicated, or noncanonical")


def require_trusted_inputs(record: dict, trusted: dict, export_body: dict, metal_body: dict, task_body: dict, sample_body: dict) -> None:
    actual_sources = field(record, "source_identities", dict)
    native_identity = field(actual_sources, "leone_metal", dict)
    require(native_identity.get("source_commit") == trusted["source_commit"], "native source differs from trusted input")
    require(native_identity.get("expected_source_commit") == trusted["source_commit"], "native expected source differs from trusted input")
    build_info = field(record, "build_info", dict)
    require(record.get("source_commit") == trusted["source_commit"], "statistics source differs from trusted input")
    require(actual_sources["statistics"].get("source_commit") == trusted["source_commit"], "statistics identity differs from trusted input")
    require(build_info.get("target") == trusted["statistics_target"], "statistics target differs from trusted input")
    native_run = metal_body["native"]["manifest"]["body"]
    native_build = native_run["executable"]["build_info"]
    require(native_build.get("source_commit") == trusted["source_commit"], "native build source differs from trusted input")
    require(native_build.get("target") == trusted["target"], "native target differs from trusted input")
    require(native_platform(native_run["execution"]) == trusted["platform"], "native platform differs from trusted input")
    require(export_body["model"]["oracle"]["sha256"] == trusted["oracle_model_sha256"], "oracle model differs from trusted input")
    require(metal_body["model"]["subject"]["sha256"] == trusted["model_sha256"], "subject model differs from trusted input")
    require(export_body["corpus"]["sha256"] == trusted["corpus_sha256"], "corpus differs from trusted input")
    require(field(task_body, "model", dict).get("family") == trusted["model_family"], "model family differs from trusted input")
    require(export_body["adapter"]["path"] == trusted["adapter_path"], "export adapter path differs from trusted input")
    adapter = {"path": trusted["adapter_path"], "sha256": trusted["adapter_sha256"]}
    for body, label in ((export_body, "export adapter"), (metal_body, "Metal adapter")):
        require(body["adapter"]["path"] == adapter["path"], f"{label} path differs from trusted expectation")
        require(body["adapter"]["sha256"] == adapter["sha256"], f"{label} hash differs from trusted expectation")
    require(metal_body["native_execution"].get("backend") == trusted["backend"], "native backend differs from trusted input")
    sampling = field(sample_body, "sampling", dict)
    require(trusted["sample_contract"] == SAMPLE_CONTRACT, "sample contract differs from trusted input")
    require(sampling.get("algorithm") == "linspace-inclusive-v1", "sample algorithm differs from trusted input")
    require(sampling.get("requested_rows") == PUBLIC_SAMPLE_ROWS, "sample row count differs from trusted input")


def validate_cuda_identity_record(
    identity: dict,
    label: str,
    require_build: bool = False,
    require_target: bool = False,
) -> None:
    require(isinstance(identity, dict), f"trusted CUDA {label} identity is missing")
    valid_commit(identity.get("source_commit"), f"trusted CUDA {label} source commit")
    if "manifest_sha256" in identity:
        valid_hash(identity.get("manifest_sha256"), f"trusted CUDA {label} manifest SHA-256")
    executable = field(identity, "executable", dict)
    name = field(executable, "name", str)
    require(Path(name).name == name and name not in ("", ".", ".."), f"trusted CUDA {label} executable name is invalid")
    valid_hash(executable.get("sha256"), f"trusted CUDA {label} executable SHA-256")
    if require_build:
        valid_hash(identity.get("build_info_sha256"), f"trusted CUDA {label} build info SHA-256")
    if require_target:
        target = identity.get("target")
        require(isinstance(target, str) and target and not os.path.isabs(target), f"trusted CUDA {label} target is invalid")


def validate_cuda_trusted_base(trusted: dict) -> None:
    required = (
        "source_commit", "platform", "statistics_platform", "target", "statistics_target", "backend",
        "adapter_path", "adapter_sha256", "model_family", "model_sha256",
        "oracle_model_sha256", "corpus_sha256", "task_manifest_sha256",
        "sample_contract", "metric_family", "producer_identities", "statistics_identity",
        "native_workflow",
    )
    for name in required:
        require(name in trusted, f"trusted CUDA input {name} is missing")
    valid_commit(trusted["source_commit"], "trusted CUDA source commit")
    require(trusted["platform"] == "linux-x86_64", "trusted CUDA platform differs")
    require(isinstance(trusted["statistics_platform"], str) and trusted["statistics_platform"] and not os.path.isabs(trusted["statistics_platform"]), "trusted CUDA statistics platform is invalid")
    require(trusted["target"] == "x86_64-unknown-linux-gnu", "trusted CUDA target differs")
    require(isinstance(trusted["statistics_target"], str) and trusted["statistics_target"] and not os.path.isabs(trusted["statistics_target"]), "trusted CUDA statistics target is invalid")
    require(trusted["backend"] == "cuda", "trusted CUDA backend is not CUDA")
    require(trusted["adapter_path"] == "research/oracle/llama_logits.cpp", "trusted CUDA adapter path differs")
    require(isinstance(trusted["model_family"], str) and trusted["model_family"], "trusted CUDA model family is missing")
    for name in ("adapter_sha256", "model_sha256", "oracle_model_sha256", "corpus_sha256", "task_manifest_sha256"):
        valid_hash(trusted[name], f"trusted CUDA {name}")
    require(trusted["sample_contract"] == SAMPLE_CONTRACT, "trusted CUDA sample contract differs")
    require(trusted["metric_family"] == "kld", "trusted CUDA metric family differs")


def validate_cuda_trusted_workflow(workflow: dict) -> None:
    require(isinstance(workflow.get("kv_cache_dtype"), str) and workflow["kv_cache_dtype"], "trusted CUDA KV cache dtype is invalid")
    prefill = field(workflow, "prefill", dict)
    require(prefill.get("path") in ("sequential", "chunked"), "trusted CUDA prefill path is invalid")
    if prefill["path"] == "chunked":
        require(isinstance(prefill.get("chunk_tokens"), int) and not isinstance(prefill["chunk_tokens"], bool) and prefill["chunk_tokens"] > 0, "trusted CUDA prefill chunk is invalid")
    else:
        require(prefill.get("chunk_tokens") is None, "trusted CUDA sequential prefill has a chunk")
    require(isinstance(workflow.get("token_count"), int) and not isinstance(workflow["token_count"], bool) and workflow["token_count"] >= 2, "trusted CUDA token count is invalid")


def validate_cuda_trusted_contract(trusted: dict) -> None:
    validate_cuda_trusted_base(trusted)
    identities = field(trusted, "producer_identities", dict)
    validate_cuda_identity_record(field(identities, "oracle_stage", dict), "oracle stage")
    validate_cuda_identity_record(field(identities, "llama_cpp_cuda", dict), "llama.cpp CUDA")
    validate_cuda_identity_record(field(identities, "leone_cuda", dict), "Leone CUDA", True, True)
    require(identities["leone_cuda"]["target"] == trusted["target"], "trusted native target differs")
    statistics = field(trusted, "statistics_identity", dict)
    validate_cuda_identity_record(statistics, "statistics", True, True)
    require(statistics["source_commit"] == trusted["source_commit"], "trusted statistics source differs")
    require(statistics["target"] == trusted["statistics_target"], "trusted statistics target differs")
    validate_cuda_trusted_workflow(field(trusted, "native_workflow", dict))


def validate_cuda_identity(record: dict, trusted: dict, receipt_verifier: Path, root: Path) -> None:
    require(record.get("schema_version") == CUDA_COMPARISON_SCHEMA, "unknown CUDA comparison schema")
    validate_cuda_trusted_contract(trusted)
    valid_commit(record.get("source_commit"), "CUDA statistics source commit")
    require(record["source_commit"] == trusted["source_commit"], "CUDA statistics source differs from trusted input")
    require(record.get("statistics_platform") == trusted["statistics_platform"], "CUDA statistics platform differs")
    require(record.get("statistics_target") == trusted["statistics_target"], "CUDA statistics target differs")
    executable = field(record, "executable", dict)
    executable_name = field(executable, "name", str)
    require(Path(executable_name).name == executable_name, "CUDA statistics executable name is not stable")
    valid_hash(executable.get("sha256"), "CUDA statistics executable SHA-256")
    require(executable == trusted["statistics_identity"]["executable"], "CUDA statistics executable differs from the external tuple")
    build_info = field(record, "build_info", dict)
    require(build_info.get("schema_version") == "leone.build-info.v1", "CUDA build info schema differs")
    statistics_identity = trusted["statistics_identity"]
    require(record["source_commit"] == statistics_identity["source_commit"], "CUDA statistics source differs from its identity")
    require(build_info.get("source_commit") == statistics_identity["source_commit"], "CUDA build source differs from its identity")
    require(build_info.get("source_tree_dirty") is False, "CUDA build is dirty")
    require(build_info.get("profile") == "release", "CUDA build is not release")
    require(build_info.get("target") == statistics_identity["target"], "CUDA build target differs from its identity")
    models = field(record, "models", dict)
    oracle_model = field(models, "oracle", dict)
    subject_model = field(models, "subject", dict)
    require(oracle_model.get("sha256") == trusted["oracle_model_sha256"], "CUDA oracle model differs from trusted input")
    require(subject_model.get("sha256") == trusted["model_sha256"], "CUDA subject model differs from trusted input")
    valid_hash(oracle_model.get("sha256"), "CUDA oracle model SHA-256")
    valid_hash(subject_model.get("sha256"), "CUDA subject model SHA-256")
    require(oracle_model.get("storage_type") in ("BF16", "F16"), "CUDA oracle is not full precision")
    require(isinstance(subject_model.get("storage_type"), str) and subject_model["storage_type"], "CUDA subject storage type is missing")
    require(field(record, "model_family", str) == trusted["model_family"], "CUDA model family differs from trusted input")
    corpus = field(record, "corpus", dict)
    require(corpus.get("sha256") == trusted["corpus_sha256"], "CUDA corpus differs from trusted input")
    valid_hash(corpus.get("sha256"), "CUDA corpus SHA-256")
    adapter = field(record, "adapter", dict)
    require(adapter.get("path") == trusted["adapter_path"], "CUDA adapter path differs from trusted input")
    require(adapter.get("sha256") == trusted["adapter_sha256"], "CUDA adapter differs from trusted input")
    valid_hash(adapter.get("sha256"), "CUDA adapter SHA-256")
    adapter_artifact = field(adapter, "artifact", dict)
    require(adapter_artifact.get("sha256") == trusted["adapter_sha256"], "CUDA adapter artifact differs from trusted input")
    package_artifact(root, adapter_artifact, "CUDA adapter source")
    identities = field(record, "source_identities", dict)
    for name in ("oracle", "llama_cpp_cuda", "leone_cuda", "statistics"):
        valid_commit(field(identities, name, dict).get("source_commit"), f"CUDA {name} source commit")
    trusted_identities = trusted["producer_identities"]
    require(identities["oracle"] == trusted_identities["oracle_stage"], "CUDA oracle identity differs from the external tuple")
    require(identities["llama_cpp_cuda"] == trusted_identities["llama_cpp_cuda"], "CUDA llama.cpp identity differs from the external tuple")
    require(identities["leone_cuda"] == trusted_identities["leone_cuda"], "CUDA Leone identity differs from the external tuple")
    require(identities["statistics"] == trusted["statistics_identity"], "CUDA statistics identity differs from the external tuple")
    validation = field(record, "validation", dict)
    require(validation.get("mode") == "offline", "CUDA validation mode differs")
    require(validation.get("packaged_logits") is False, "CUDA full logits must remain generation-only")
    require(validation.get("generation_rehashed_original_artifacts") is True, "CUDA generation did not hash original artifacts")
    trusted_record = field(validation, "trusted", dict)
    require(trusted_record == trusted, "CUDA trusted inputs differ from the external tuple")
    statistics_executable = field(validation, "statistics_executable", dict)
    require(statistics_executable == trusted["statistics_identity"]["executable"], "CUDA statistics executable identity differs")
    require(json_digest(build_info) == statistics_identity["build_info_sha256"], "CUDA statistics build info identity differs")
    parser_record = field(validation, "receipt_parser", dict)
    require(parser_record.get("name") == "leone-receipt-verify", "CUDA receipt parser is not canonical")
    require(parser_record.get("interface") == "quality", "CUDA receipt parser interface differs")
    parser_sha256 = valid_hash(parser_record.get("sha256"), "CUDA receipt parser SHA-256")
    require(receipt_verifier.is_file() and not receipt_verifier.is_symlink(), "CUDA receipt parser executable is invalid")
    require(receipt_verifier.resolve().name == parser_record["name"], "CUDA receipt parser executable differs")
    require(digest(receipt_verifier) == parser_sha256, "CUDA receipt parser executable SHA-256 differs")


def validate_cuda_input(record: dict, root: Path) -> dict:
    input_record = field(record, "input", dict)
    tokens = field(input_record, "tokens", dict)
    token_count = task_integer(tokens.get("count"), "CUDA token count", 2)
    require(tokens.get("encoding") == "u32le", "CUDA token encoding differs")
    require(tokens.get("bytes") == token_count * 4, "CUDA token byte count differs")
    package_artifact(root, tokens, "CUDA token artifact")
    window = task_integer(input_record.get("window_tokens"), "CUDA window", 2)
    stride = task_integer(input_record.get("stride_tokens"), "CUDA stride", 1)
    rows = task_integer(input_record.get("rows"), "CUDA rows", 1)
    vocab = task_integer(input_record.get("vocab_size"), "CUDA vocabulary", 1)
    require(window <= rows, "CUDA window exceeds scored rows")
    require(rows == token_count - 1, "CUDA rows do not equal token count minus one")
    require(stride == window - 1, "CUDA stride differs from window minus one")
    models = field(record, "models", dict)
    require(vocab == task_integer(field(models, "oracle", dict).get("vocab_size"), "CUDA oracle vocabulary", 1), "CUDA vocabulary differs from oracle model")
    require(vocab == task_integer(field(models, "subject", dict).get("vocab_size"), "CUDA subject vocabulary", 1), "CUDA vocabulary differs from subject model")
    return input_record


def validate_cuda_logits(record: dict, root: Path, input_record: dict, label: str) -> dict:
    logits = field(record, "logits", dict)
    validate_logits_metadata(logits, f"CUDA {label} logits", False)
    require(logits["bytes"] == input_record["rows"] * input_record["vocab_size"] * 4, f"CUDA {label} logits shape differs")
    return logits


def validate_cuda_producer_snapshot(root: Path, record: dict, label: str) -> dict:
    path = package_artifact(root, record, f"CUDA {label} producer manifest")
    body = field(record, "body", dict)
    require(body == load_json(path), f"CUDA {label} producer manifest changed")
    return body


def validate_cuda_artifact_link(actual: dict, expected: dict, label: str) -> None:
    for name in ("sha256", "bytes", "encoding"):
        require(actual.get(name) == expected.get(name), f"{label} {name} differs")


def validate_cuda_source_run(
    run: dict,
    label: str,
    expected_engine: dict,
    expected_device: str,
    model: dict,
    input_record: dict,
    logits: dict,
    adapter_sha256: str,
    identity: dict,
    manifest_record: dict,
) -> None:
    require(run.get("schema_version") == "leone.llama-oracle.v2", f"{label} producer schema differs")
    require(run.get("engine") == expected_engine, f"{label} producer engine differs")
    require(run.get("source_commit") == identity["source_commit"], f"{label} producer source commit differs from the external tuple")
    if "manifest_sha256" in identity:
        require(manifest_record.get("sha256") == identity["manifest_sha256"], f"{label} producer manifest differs from the external tuple")
    adapter = field(run, "adapter", dict)
    require(adapter.get("path") == "research/oracle/llama_logits.cpp", f"{label} producer adapter differs")
    require(adapter.get("sha256") == adapter_sha256, f"{label} producer adapter hash differs")
    valid_hash(adapter.get("sha256"), f"{label} producer adapter hash")
    executable = field(run, "executable", dict)
    executable_path = field(executable, "path", str)
    require(not Path(executable_path).is_absolute() and ".." not in Path(executable_path).parts, f"{label} producer executable path is not stable")
    require(Path(executable_path).name == identity["executable"]["name"], f"{label} producer executable differs from the external tuple")
    valid_hash(executable.get("sha256"), f"{label} producer executable hash")
    require(executable.get("sha256") == identity["executable"]["sha256"], f"{label} producer executable hash differs from the external tuple")
    require(run.get("model", {}).get("sha256") == model.get("sha256"), f"{label} producer model differs")
    require(run.get("model", {}).get("vocab_size") == model.get("vocab_size"), f"{label} producer vocabulary differs")
    require(run.get("model", {}).get("architecture") == model.get("architecture"), f"{label} producer architecture differs")
    require(run.get("model", {}).get("storage_type") == model.get("storage_type"), f"{label} producer storage differs")
    actual_input = field(run, "input", dict)
    for name in ("window_tokens", "stride_tokens", "rows", "vocab_size"):
        require(actual_input.get(name) == input_record.get(name), f"{label} producer input {name} differs")
    validate_cuda_artifact_link(actual_input["tokens"], input_record["tokens"], f"{label} producer tokens")
    validate_cuda_artifact_link(run["logits"], logits, f"{label} producer logits")
    execution = field(run, "execution", dict)
    require(execution.get("device") == expected_device, f"{label} producer device differs")
    require(str(execution.get("backend_registry", "")).lower() == expected_device, f"{label} producer backend differs")
    expected_type = "cpu" if expected_device == "cpu" else "gpu"
    require(execution.get("device_type") == expected_type, f"{label} producer device type differs")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], f"{label} producer device name is missing")
    require(isinstance(execution.get("device_description"), str), f"{label} producer device description is missing")
    require(execution.get("logits_dtype") == "f32", f"{label} producer logits dtype differs")


def validate_cuda_native_run(
    run: dict,
    record: dict,
    trusted: dict,
    input_record: dict,
    model: dict,
    logits: dict,
    identity: dict,
    manifest_record: dict,
) -> None:
    require(run.get("schema_version") == "leone.native-eval.v1", "CUDA native producer schema differs")
    require(run.get("source_commit") == identity["source_commit"], "CUDA native producer source differs from the external tuple")
    if "manifest_sha256" in identity:
        require(manifest_record.get("sha256") == identity["manifest_sha256"], "CUDA native producer manifest differs from the external tuple")
    executable = field(run, "executable", dict)
    require(executable.get("name") == identity["executable"]["name"], "CUDA native executable differs from the external tuple")
    require(executable.get("sha256") == identity["executable"]["sha256"], "CUDA native executable differs")
    native_build = field(executable, "build_info", dict)
    require(json_digest(native_build) == identity["build_info_sha256"], "CUDA native build info differs from the external tuple")
    require(native_build.get("schema_version") == "leone.build-info.v1", "CUDA native build info schema differs")
    require(native_build.get("source_commit") == identity["source_commit"], "CUDA native build source differs from its identity")
    require(native_build.get("source_tree_dirty") is False, "CUDA native build is dirty")
    require(native_build.get("profile") == "release", "CUDA native build is not release")
    require(native_build.get("target") == identity["target"], "CUDA native build target differs from its identity")
    require(run.get("model", {}).get("sha256") == model.get("sha256"), "CUDA native model differs")
    require(run.get("model", {}).get("vocab_size") == model.get("vocab_size"), "CUDA native vocabulary differs")
    require(run.get("model", {}).get("architecture") == model.get("architecture"), "CUDA native architecture differs")
    actual_input = field(run, "input", dict)
    for name in ("window_tokens", "stride_tokens", "rows", "vocab_size"):
        require(actual_input.get(name) == input_record.get(name), f"CUDA native input {name} differs")
    validate_cuda_artifact_link(actual_input["tokens"], input_record["tokens"], "CUDA native tokens")
    validate_cuda_artifact_link(run["logits"], logits, "CUDA native logits")
    execution = field(run, "execution", dict)
    require(execution == field(field(record, "subjects", dict)["leone"], "execution", dict), "CUDA native execution differs")
    require(execution.get("engine") == "leone", "CUDA native engine differs")
    require(execution.get("device") == "cuda", "CUDA native device differs")
    require(execution.get("backend") == "cuda", "CUDA native backend differs")
    workflow = trusted["native_workflow"]
    require(execution.get("kv_cache_dtype") == workflow["kv_cache_dtype"], "CUDA native KV cache dtype differs")
    require(execution.get("prefill") == workflow["prefill"], "CUDA native prefill differs")
    require(actual_input["tokens"].get("count") == workflow["token_count"], "CUDA native token count differs")


def validate_cuda_producers(
    record: dict,
    root: Path,
    trusted: dict,
    input_record: dict,
    models: dict,
    logits: dict,
) -> None:
    producers = field(record, "producers", dict)
    trusted_adapter_sha256 = field(record, "adapter", dict)["sha256"]
    oracle_record = field(producers, "oracle", dict)
    llama_record = field(producers, "llama_cpp", dict)
    native_record = field(producers, "leone", dict)
    oracle_run = validate_cuda_producer_snapshot(root, oracle_record, "oracle")
    llama_run = validate_cuda_producer_snapshot(root, llama_record, "llama.cpp")
    native_run = validate_cuda_producer_snapshot(root, native_record, "Leone")
    identities = trusted["producer_identities"]
    validate_cuda_source_run(
        oracle_run,
        "CUDA oracle",
        {"name": "llama.cpp", "git_commit": PINNED},
        "cpu",
        models["oracle"],
        input_record,
        logits["oracle"],
        trusted_adapter_sha256,
        identities["oracle_stage"],
        oracle_record,
    )
    validate_cuda_source_run(
        llama_run,
        "CUDA llama.cpp",
        {"name": "llama.cpp", "git_commit": PINNED},
        "cuda",
        models["subject"],
        input_record,
        logits["llama_cpp"],
        trusted_adapter_sha256,
        identities["llama_cpp_cuda"],
        llama_record,
    )
    validate_cuda_native_run(
        native_run,
        record,
        trusted,
        input_record,
        models["subject"],
        logits["leone"],
        identities["leone_cuda"],
        native_record,
    )
    require(oracle_run["execution"] == record["oracle"]["execution"], "CUDA oracle execution differs from producer")
    require(llama_run["execution"] == record["subjects"]["llama_cpp"]["execution"], "CUDA llama.cpp execution differs from producer")


def validate_cuda_task(
    record: dict,
    root: Path,
    input_record: dict,
    models: dict,
    trusted: dict,
    label: str = "CUDA long-context task",
) -> list[int]:
    tasks = field(record, "tasks", dict)
    task = field(tasks, "long_context", dict)
    task_record = field(task, "manifest", dict)
    require(task_record.get("sha256") == trusted["task_manifest_sha256"], f"{label} manifest differs from trusted input")
    task_path = package_artifact(root, task_record, f"{label} manifest")
    body = field(task_record, "body", dict)
    require(body == load_json(task_path), f"{label} manifest changed")
    require(body.get("schema_version") == "leone.quality-task.v2", f"{label} schema differs")
    require(body.get("corpus", {}).get("sha256") == record["corpus"]["sha256"], f"{label} corpus differs")
    task_models = field(body, "model", dict)
    require(task_models.get("family") == record["model_family"], f"{label} model family differs")
    require(task_models.get("oracle", {}).get("sha256") == models["oracle"]["sha256"], f"{label} oracle differs")
    require(task_models.get("subject", {}).get("sha256") == models["subject"]["sha256"], f"{label} subject differs")
    task_input = field(body, "input", dict)
    for name in ("window_tokens", "stride_tokens", "rows", "vocab_size"):
        require(task_input.get(name) == input_record.get(name), f"{label} input {name} differs")
    validate_cuda_artifact_link(task_input["tokens"], input_record["tokens"], f"{label} tokens")
    task_spec = field(body, "task", dict)
    require(task_spec.get("name") == "needle-exact-answer", f"{label} name differs")
    context_tokens = task_integer(task_spec.get("context_tokens"), f"{label} context", MIN_LONG_CONTEXT_TOKENS)
    minimum_tokens = task_integer(task_spec.get("minimum_context_tokens"), f"{label} minimum context", MIN_LONG_CONTEXT_TOKENS)
    require(context_tokens >= minimum_tokens, f"{label} context is below its declared minimum")
    require(context_tokens == input_record["window_tokens"], f"{label} context differs from the evaluation window")
    prompt = field(task_spec, "prompt", dict)
    needle = field(task_spec, "needle", dict)
    require(digest_bytes(task_text_bytes(prompt, f"{label} prompt") + task_text_bytes(needle, f"{label} needle")) == record["corpus"]["sha256"], f"{label} prompt and needle differ from corpus")
    require(body.get("tokenizer", {}).get("name") == models["oracle"].get("tokenizer"), f"{label} tokenizer differs")
    tokenization = field(body, "tokenization", dict)
    require(tokenization.get("method") == "llama-tokenize", f"{label} tokenizer method differs")
    require(tokenization.get("source", {}).get("name") == "llama.cpp", f"{label} tokenizer source differs")
    require(tokenization.get("source", {}).get("git_commit") == PINNED, f"{label} tokenizer pin differs")
    require(tokenization.get("model", {}).get("sha256") == models["oracle"].get("sha256"), f"{label} tokenizer model differs")
    valid_hash(tokenization.get("combined_sha256"), f"{label} combined input SHA-256")
    require(tokenization.get("combined_sha256") == record["corpus"]["sha256"], f"{label} corpus tokenization differs")
    valid_hash(tokenization.get("input_tokens_sha256"), f"{label} tokenization input SHA-256")
    require(tokenization.get("input_tokens_sha256") == input_record["tokens"]["sha256"], f"{label} tokenization input differs")
    answer = field(task_spec, "answer", dict)
    answer_row = answer.get("row")
    answer_tokens = answer.get("token_ids")
    require(isinstance(answer_row, int) and not isinstance(answer_row, bool) and answer_row >= 0, f"{label} answer row is invalid")
    require(isinstance(answer_tokens, list) and answer_tokens, f"{label} answer tokens are missing")
    require(answer_row >= context_tokens - 1, f"{label} answer is before the declared context")
    require(all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < input_record["vocab_size"] for token in answer_tokens), f"{label} answer token ID is invalid")
    require(answer.get("text") == needle.get("text"), f"{label} answer text differs")
    require(answer.get("sha256") == needle.get("sha256"), f"{label} answer needle differs")
    packed_answer = struct.pack(f"<{len(answer_tokens)}I", *answer_tokens)
    require(tokenization.get("answer_tokens_sha256") == digest_bytes(packed_answer), f"{label} answer tokenization differs")
    rows = list(range(answer_row, answer_row + len(answer_tokens)))
    require(rows[-1] < input_record["rows"], f"{label} answer exceeds scored rows")
    token_path = root / input_record["tokens"]["path"]
    require(task_tokens(token_path, answer_row + 1, len(answer_tokens)) == answer_tokens, f"{label} answer does not match the token stream")
    declared = task.get("rows")
    require(declared == rows, f"{label} rows differ")
    return rows


def validate_cuda_task_logits(
    sample_body: dict,
    root: Path,
    task_body: dict,
    task_rows: list[int],
    vocab: int,
    label: str = "CUDA long-context task",
) -> None:
    answer = field(field(task_body, "task", dict), "answer", dict)
    answer_tokens = answer["token_ids"]
    indices = field(field(sample_body, "sampling", dict), "indices", list)
    positions = {row: position for position, row in enumerate(indices)}
    sample_logits = field(field(sample_body, "artifacts", dict), "logits", dict)
    for subject, record in sample_logits.items():
        path = root / record["path"]
        observed = []
        for row in task_rows:
            require(row in positions, f"{label} row is absent from the sample")
            observed.append(task_argmax(path, positions[row], vocab, f"{label} {subject}"))
        require(observed == answer_tokens, f"{label} {subject} answer differs")


def validate_cuda_execution(record: dict, trusted: dict, label: str, model_sha256: str, token_sha256: str) -> None:
    engine = field(record, "engine", dict)
    execution = field(record, "execution", dict)
    require(record.get("model_sha256") == model_sha256, f"{label} model differs")
    require(record.get("input_tokens_sha256") == token_sha256, f"{label} token input differs")
    if label == "oracle":
        require(engine == {"name": "llama.cpp", "git_commit": PINNED}, "CUDA oracle engine differs")
        require(execution.get("device") == "cpu", "CUDA oracle is not CPU")
        require(str(execution.get("backend_registry", "")).lower() == "cpu", "CUDA oracle backend is not CPU")
    elif label == "llama.cpp CUDA":
        require(engine == {"name": "llama.cpp-cuda", "git_commit": PINNED}, "CUDA llama.cpp engine differs")
        require(execution.get("device") == "cuda", "CUDA llama.cpp subject is not CUDA")
        require(str(execution.get("backend_registry", "")).lower() == "cuda", "CUDA llama.cpp backend differs")
    else:
        native_source = trusted["producer_identities"]["leone_cuda"]["source_commit"]
        require(engine == {"name": "leone-cuda-eval", "git_commit": native_source}, "CUDA Leone engine differs")
        require(execution.get("device") == "cuda", "CUDA Leone subject is not CUDA")
        require(execution.get("backend") == "cuda", "CUDA Leone backend differs")
        require(str(execution.get("backend_registry", "")).lower() == "cuda", "CUDA Leone backend registry differs")
    expected_device_type = "cpu" if label == "oracle" else "gpu"
    require(execution.get("device_type") == expected_device_type, f"{label} device type differs")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], f"{label} device name is missing")
    require(isinstance(execution.get("device_description"), str), f"{label} device description is missing")
    require(execution.get("logits_dtype") == "f32", f"{label} logits dtype differs")


def validate_cuda_sample(
    record: dict,
    root: Path,
    input_record: dict,
    full_logits: dict,
    task_rows: list[int],
) -> dict:
    samples = field(record, "samples", dict)
    sample_record = field(samples, "manifest", dict)
    sample_path = package_artifact(root, sample_record, "packaged CUDA quality samples")
    sample_body = field(sample_record, "body", dict)
    require(sample_body == load_json(sample_path), "packaged CUDA quality samples body differs")
    originals = field(sample_body, "originals", dict)
    require(originals.get("mode") == "generation-only", "CUDA full logits are not generation-only")
    require(originals.get("retention") == "caller-owned immutable cache", "CUDA original logits retention differs")
    require(originals.get("logits") == sample_body["source"]["logits"], "CUDA original logits metadata differs")
    return validate_sample_body(sample_body, root, root, input_record, full_logits, task_rows, False)


def validate_cuda_generation_inputs(
    generation: dict,
    record: dict,
    input_record: dict,
    task_rows: list[int],
) -> None:
    require(generation.get("schema_version") == GENERATION_SCHEMA, "CUDA generation record schema differs")
    models = field(record, "models", dict)
    inputs = field(generation, "inputs", dict)
    expected = {
        "model_sha256": models["subject"]["sha256"],
        "oracle_model_sha256": models["oracle"]["sha256"],
        "corpus_sha256": record["corpus"]["sha256"],
        "task_manifest_sha256": record["tasks"]["long_context"]["manifest"]["sha256"],
        "tokens_sha256": input_record["tokens"]["sha256"],
        "token_count": input_record["tokens"]["count"],
        "rows": input_record["rows"],
        "vocab_size": input_record["vocab_size"],
        "sample_contract": SAMPLE_CONTRACT,
    }
    for name, value in expected.items():
        require(inputs.get(name) == value, f"CUDA generation input {name} differs")
    sampling = field(generation, "sampling", dict)
    requested, indices = expected_sample_indices(input_record["rows"], task_rows)
    require(sampling.get("algorithm") == "linspace-inclusive-v1", "CUDA generation sampling algorithm differs")
    require(sampling.get("requested_rows") == requested, "CUDA generation sample count differs")
    require(sampling.get("indices") == indices, "CUDA generation sample indices differ")
    require(sampling.get("task_rows") == task_rows, "CUDA generation task rows differ")


def validate_cuda_generation_producers(generation: dict, record: dict) -> None:
    manifests = field(generation, "producer_manifests", dict)
    producers = field(record, "producers", dict)
    for label, producer_label in (("oracle", "oracle"), ("llama_cpp", "llama_cpp"), ("leone", "leone")):
        expected = valid_hash(manifests.get(label), f"CUDA generation {label} manifest SHA-256")
        actual = field(producers, producer_label, dict).get("sha256")
        require(actual == expected, f"CUDA generation {label} manifest differs")


def generation_metadata(value: dict, label: str) -> dict:
    valid_hash(value.get("sha256"), f"CUDA generation {label} SHA-256")
    require(isinstance(value.get("bytes"), int) and not isinstance(value["bytes"], bool) and value["bytes"] > 0, f"CUDA generation {label} byte count is invalid")
    require(value.get("encoding") == "row-major-f32-le", f"CUDA generation {label} encoding differs")
    return {name: value[name] for name in ("sha256", "bytes", "encoding")}


def validate_cuda_generation_logits(generation: dict, source_logits: dict, sample_body: dict) -> None:
    expected_full = field(generation, "full_logits", dict)
    for label in ("oracle", "llama_cpp", "leone"):
        expected = generation_metadata(field(expected_full, label, dict), f"{label} full logits")
        actual = generation_metadata(source_logits[label], f"{label} source logits")
        require(actual == expected, f"CUDA generation {label} full logits differ")
    expected_samples = field(generation, "sampled_logits", dict)
    actual_samples = field(sample_body["artifacts"], "logits", dict)
    for label in ("oracle", "llama_cpp", "leone"):
        expected = field(expected_samples, label, dict)
        actual = field(actual_samples, label, dict)
        for name in ("sha256", "bytes", "encoding", "source_row_sha256"):
            require(actual.get(name) == expected.get(name), f"CUDA generation {label} sampled {name} differs")


def validate_cuda_generation(
    record: dict,
    root: Path,
    generation: dict,
    input_record: dict,
    task_rows: list[int],
    source_logits: dict,
    sample_body: dict,
) -> None:
    validate_cuda_generation_inputs(generation, record, input_record, task_rows)
    validation = field(record, "validation", dict)
    generation_record = field(validation, "generation_record", dict)
    generation_path = package_artifact(root, generation_record, "CUDA generation record")
    require(field(generation_record, "body", dict) == load_json(generation_path), "CUDA generation record body differs")
    require(generation_record["body"] == generation, "CUDA generation record differs from external input")
    validate_cuda_generation_producers(generation, record)
    validate_cuda_generation_logits(generation, source_logits, sample_body)


def validate_cuda_quality(record: dict, root: Path, receipt_verifier: Path, sample_body: dict) -> None:
    quality = field(record, "quality", dict)
    models = field(record, "models", dict)
    pseudo_export = {
        "corpus": record["corpus"],
        "input": record["input"],
        "model": models,
    }
    sample_logits = sample_body["artifacts"]["logits"]
    oracle_path = root / sample_logits["oracle"]["path"]
    for label, subject_key, engine in (
        ("llama.cpp CUDA", "llama_cpp", {"name": "llama.cpp-cuda", "git_commit": PINNED}),
        ("Leone CUDA", "leone", {"name": "leone-cuda-eval", "git_commit": record["source_identities"]["leone_cuda"]["source_commit"]}),
    ):
        subject = field(field(record, "subjects", dict), subject_key, dict)
        subject_sample = sample_logits[subject_key]
        subject_path = root / subject_sample["path"]
        validate_quality_link(
            root / "quality-comparison.json",
            field(quality, subject_key, dict),
            pseudo_export,
            subject,
            engine,
            oracle_path,
            subject_path,
            sample_body,
            sample_logits["oracle"],
            subject_sample,
            receipt_verifier,
            label,
        )


def validate_cuda_comparison(path: Path, receipt_verifier: Path, trusted: dict, generation: dict) -> dict:
    record = load_json(path)
    require(record.get("schema_version") == CUDA_COMPARISON_SCHEMA, "unknown CUDA comparison schema")
    stable_strings(record)
    require(generation is not None, "CUDA generation record is required")
    stable_strings(generation, "generation")
    validate_cuda_identity(record, trusted, receipt_verifier, path.parent)
    input_record = validate_cuda_input(record, path.parent)
    full_logits = {}
    oracle = field(record, "oracle", dict)
    subjects = field(record, "subjects", dict)
    subject_model_sha256 = field(record, "models", dict)["subject"]["sha256"]
    oracle_model_sha256 = field(record, "models", dict)["oracle"]["sha256"]
    token_sha256 = input_record["tokens"]["sha256"]
    for label, value, model_sha256 in (
        ("oracle", oracle, oracle_model_sha256),
        ("llama.cpp CUDA", subjects.get("llama_cpp"), subject_model_sha256),
        ("Leone CUDA", subjects.get("leone"), subject_model_sha256),
    ):
        require(isinstance(value, dict), f"CUDA {label} record is missing")
        validate_cuda_execution(value, trusted, label, model_sha256, token_sha256)
        full_logits[label] = field(value, "logits", dict)
        validate_cuda_logits(value, path.parent, input_record, label)
    require(oracle["logits"] == full_logits["oracle"], "CUDA oracle logits record differs")
    require(subjects["llama_cpp"]["logits"] == full_logits["llama.cpp CUDA"], "CUDA llama.cpp logits record differs")
    require(subjects["leone"]["logits"] == full_logits["Leone CUDA"], "CUDA Leone logits record differs")
    source_logits = {
        "oracle": full_logits["oracle"],
        "llama_cpp": full_logits["llama.cpp CUDA"],
        "leone": full_logits["Leone CUDA"],
    }
    validate_cuda_producers(
        record,
        path.parent,
        trusted,
        input_record,
        field(record, "models", dict),
        source_logits,
    )
    task_rows = validate_cuda_task(
        record,
        path.parent,
        input_record,
        field(record, "models", dict),
        trusted,
    )
    sample_body = validate_cuda_sample(record, path.parent, input_record, source_logits, task_rows)
    validate_cuda_generation(record, path.parent, generation, input_record, task_rows, source_logits, sample_body)
    validate_cuda_task_logits(
        sample_body,
        path.parent,
        field(field(record, "tasks", dict)["long_context"], "manifest", dict)["body"],
        task_rows,
        input_record["vocab_size"],
    )
    validate_cuda_quality(record, path.parent, receipt_verifier, sample_body)
    return record


def validate_metal_sample_anchor(sample_record: dict, expected: str | None) -> None:
    if expected is not None:
        valid_hash(expected, "trusted sample manifest SHA-256")
        require(
            sample_record.get("sha256") == expected,
            "quality sample manifest differs from the trusted anchor",
        )


def validate_embedded_sample_anchor(
    trusted_record: dict, sample_record: dict, expected: str | None,
) -> None:
    embedded_sample = trusted_record.get("sample_manifest_sha256")
    if embedded_sample is not None:
        embedded_hash = valid_hash(embedded_sample, "comparison trusted sample manifest SHA-256")
        require(
            embedded_hash == sample_record.get("sha256"),
            "comparison embedded sample manifest differs",
        )
        if expected is not None:
            require(embedded_hash == expected, "comparison trusted sample manifest differs")


def validate_comparison(
    path: Path,
    receipt_verifier: Path | None = None,
    trusted: dict | None = None,
    generation: dict | None = None,
) -> dict:
    record = load_json(path)
    if record.get("schema_version") == CUDA_COMPARISON_SCHEMA:
        require(receipt_verifier is not None, "canonical CUDA comparison requires a receipt verifier")
        require(trusted is not None, "CUDA comparison trusted inputs are required")
        return validate_cuda_comparison(path, receipt_verifier, trusted, generation)
    require(record.get("schema_version") == "leone.quality-cross-device.v2", "unknown cross-device schema")
    require(receipt_verifier is not None, "canonical comparison requires a receipt verifier")
    require(trusted is not None, "comparison trusted inputs are required")
    stable_strings(record)
    valid_commit(record.get("source_commit"), "statistics source commit")
    executable = field(record, "executable", dict)
    require(Path(field(executable, "name", str)).name == executable["name"], "statistics executable name is not stable")
    valid_hash(executable.get("sha256"), "statistics executable SHA-256")
    build_info = field(record, "build_info", dict)
    require(build_info.get("schema_version") == "leone.build-info.v1", "statistics build info schema differs")
    require(build_info.get("source_commit") == record["source_commit"], "statistics build source differs")
    require(build_info.get("source_tree_dirty") is False, "statistics build is dirty")
    require(build_info.get("profile") == "release", "statistics build is not release")
    stages = field(record, "stages", dict)
    exported = field(stages, "oracle_export", dict)
    metal = field(stages, "metal_subject", dict)
    exported_file = package_artifact(path.parent, exported, "packaged export stage")
    metal_file = package_artifact(path.parent, metal, "packaged Metal stage")
    export_body = field(exported, "body", dict)
    metal_body = field(metal, "body", dict)
    require(export_body == load_json(exported_file), "packaged export stage body differs")
    require(metal_body == load_json(metal_file), "packaged Metal stage body differs")
    require(validate_export(exported_file.parent, False) == export_body, "packaged export stage failed semantic validation")
    require(validate_metal(metal_file.parent, False) == metal_body, "packaged Metal stage failed semantic validation")
    require(export_body.get("stage") == "oracle-export", "comparison embeds the wrong export stage")
    require(metal_body.get("stage") == "metal-subject", "comparison embeds the wrong Metal stage")
    require(metal_body["model"]["oracle"] == export_body["model"]["oracle"], "comparison oracle model records differ")
    require(metal_body["model"]["subject"]["name"] == export_body["model"]["subject"]["name"], "comparison subject model name differs")
    require(metal_body["model"]["subject"]["sha256"] == export_body["model"]["subject"]["sha256"], "comparison subject model hash differs")
    require(metal_body["model"]["subject"].get("storage_type") == "Q4_K - Medium", "comparison subject storage type is missing")
    require(metal_body.get("input") == export_body.get("input"), "comparison input records differ")
    models = field(record, "models", dict)
    require(models.get("oracle") == export_body["model"]["oracle"], "comparison oracle model differs")
    require(models.get("subject") == metal_body["model"]["subject"], "comparison subject model differs")
    require(record.get("corpus") == export_body.get("corpus"), "comparison corpus differs")
    require(record.get("input") == export_body.get("input"), "comparison input differs")
    require(metal_body["parent"]["manifest_sha256"] == exported["sha256"], "comparison parent stage hash differs")
    require(metal_body["parent"]["manifest"] == export_body, "comparison parent stage body differs")
    tasks = field(record, "tasks", dict)
    long_context = field(tasks, "long_context", dict)
    task_manifest_record = field(long_context, "manifest", dict)
    task_manifest_file = package_artifact(path.parent, task_manifest_record, "packaged long-context task")
    task_body = field(task_manifest_record, "body", dict)
    require(task_body == load_json(task_manifest_file), "packaged long-context task body differs")
    task_answer = field(field(task_body, "task", dict), "answer", dict)
    task_answer_row = task_integer(task_answer.get("row"), "packaged task answer row", 0)
    task_rows = list(range(task_answer_row, task_answer_row + len(field(task_answer, "token_ids", list))))
    samples = field(record, "samples", dict)
    sample_manifest_record = field(samples, "manifest", dict)
    expected_sample = trusted.get("sample_manifest_sha256")
    validate_metal_sample_anchor(sample_manifest_record, expected_sample)
    sample_manifest_file = package_artifact(path.parent, sample_manifest_record, "packaged quality samples")
    sample_body = field(sample_manifest_record, "body", dict)
    require(sample_body == load_json(sample_manifest_file), "packaged quality samples body differs")
    require(validate_sample_manifest(sample_manifest_file, path.parent, exported_file.parent, metal_file.parent, task_rows) == sample_body, "packaged quality samples failed semantic validation")
    task_result = field(long_context, "result", dict)
    require(task_result == validate_task(task_manifest_file, exported_file.parent, metal_file.parent, sample_manifest_file), "long-context task result differs")
    require(export_body["model"]["oracle"]["storage_type"] in ("BF16", "F16"), "comparison oracle is not full precision")
    adapter = field(export_body, "adapter", dict)
    require(adapter.get("path") == "research/oracle/llama_logits.cpp", "comparison adapter path differs")
    valid_hash(adapter.get("sha256"), "comparison adapter SHA-256")
    oracle_run = export_body["oracle"]["manifest"]["body"]
    validate_run_provenance(oracle_run, export_body, "comparison oracle run")
    require(oracle_run.get("corpus") == export_body["corpus"], "comparison oracle corpus differs")
    require(oracle_run["execution"]["device"] == "cpu", "comparison oracle is not CPU")
    require(isinstance(oracle_run["execution"].get("backend_registry"), str) and oracle_run["execution"]["backend_registry"].lower() == "cpu", "comparison oracle backend is not CPU")
    require(oracle_run["model"]["sha256"] == export_body["model"]["oracle"]["sha256"], "comparison oracle model differs")
    require(oracle_run["input"]["tokens"]["sha256"] == export_body["input"]["tokens"]["sha256"], "comparison oracle tokens differ")
    require(oracle_run["input"]["tokens"].get("count") == export_body["input"]["tokens"].get("count"), "comparison oracle token count differs")
    require(oracle_run["input"]["tokens"].get("encoding") == "u32le", "comparison oracle token encoding differs")
    require(oracle_run["input"].get("window_tokens") == export_body["input"].get("window_tokens"), "comparison oracle window differs")
    require(oracle_run["input"].get("rows") == export_body["input"].get("rows"), "comparison oracle rows differ")
    require(oracle_run["model"].get("storage_type") == export_body["model"]["oracle"]["storage_type"], "comparison oracle storage type differs")
    require_logits_match(oracle_run["logits"], export_body["oracle"]["logits"], "comparison oracle logits")
    llama_run = metal_body["subject"]["manifest"]["body"]
    validate_run_provenance(llama_run, metal_body, "comparison llama Metal run")
    llama_execution = field(llama_run, "execution", dict)
    require(llama_execution.get("device") == "metal", "comparison llama subject is not Metal")
    require(isinstance(llama_execution.get("backend_registry"), str) and llama_execution["backend_registry"].lower() in ("mtl", "metal"), "comparison llama backend is not Metal")
    require(isinstance(llama_execution.get("device_name"), str) and llama_execution["device_name"], "comparison llama device name is missing")
    require(isinstance(llama_execution.get("device_description"), str), "comparison llama device description is missing")
    require(llama_run["model"]["sha256"] == metal_body["model"]["subject"]["sha256"], "comparison llama model differs")
    for name in ("vocab_size", "architecture", "tokenizer"):
        require(llama_run["model"].get(name) == metal_body["model"]["subject"].get(name), f"comparison llama {name} differs")
    require_logits_match(llama_run["logits"], metal_body["subject"]["logits"], "comparison llama logits")
    native_run = metal_body["native"]["manifest"]["body"]
    validate_native_provenance(native_run, metal_body, None)
    require(native_run["model"]["sha256"] == metal_body["model"]["subject"]["sha256"], "comparison native model differs")
    require(native_run["model"].get("storage_type") == "Q4_K - Medium", "comparison native storage type is missing")
    for name in ("vocab_size", "architecture", "tokenizer"):
        require(native_run["model"].get(name) == metal_body["model"]["subject"].get(name), f"comparison native {name} differs")
    require(native_run["input"]["tokens"]["sha256"] == metal_body["input"]["tokens"]["sha256"], "comparison native tokens differ")
    require(native_run["input"]["tokens"].get("count") == metal_body["input"]["tokens"].get("count"), "comparison native token count differs")
    require(native_run["input"].get("window_tokens") == metal_body["input"].get("window_tokens"), "comparison native window differs")
    require(native_run["input"].get("rows") == metal_body["input"].get("rows"), "comparison native rows differ")
    require(native_run["input"].get("vocab_size") == metal_body["input"].get("vocab_size"), "comparison native vocabulary differs")
    require_logits_match(native_run["logits"], metal_body["native"]["logits"], "comparison native logits")
    validation = field(record, "validation", dict)
    require(validation.get("mode") == "offline", "comparison validation mode differs")
    require(validation.get("packaged_stage_manifests") is True, "comparison stage manifests are not packaged")
    require(validation.get("generation_rehashed_original_artifacts") is True, "comparison generation did not hash original artifacts")
    trusted_record = field(validation, "trusted", dict)
    for name in ("source_commit", "platform", "target", "statistics_target", "backend", "adapter_path", "adapter_sha256", "model_family", "model_sha256", "oracle_model_sha256", "corpus_sha256", "sample_contract", "metric_family"):
        require(trusted_record.get(name) == trusted[name], f"comparison trusted {name} differs")
    validate_embedded_sample_anchor(trusted_record, sample_manifest_record, expected_sample)
    statistics_executable = field(validation, "statistics_executable", dict)
    require(statistics_executable == executable, "comparison statistics executable identity differs")
    parser_record = field(validation, "receipt_parser", dict)
    parser_name = field(parser_record, "name", str)
    require(Path(parser_name).name == parser_name, "comparison receipt parser name is not stable")
    require(parser_name == "leone-receipt-verify", "comparison receipt parser is not canonical")
    require(parser_record.get("interface") == "quality", "comparison receipt parser interface differs")
    parser_sha256 = valid_hash(parser_record.get("sha256"), "comparison receipt parser SHA-256")
    if receipt_verifier is not None:
        require(receipt_verifier.is_file(), "comparison receipt parser executable is missing")
        require(not receipt_verifier.is_symlink(), "comparison receipt parser executable must be a regular file")
        require(receipt_verifier.resolve().name == parser_name, "comparison receipt parser executable differs")
        require(digest(receipt_verifier) == parser_sha256, "comparison receipt parser executable SHA-256 differs")
    identities = field(record, "source_identities", dict)
    for name in ("oracle", "llama_cpp_metal", "leone_metal", "statistics"):
        identity = field(identities, name, dict)
        valid_commit(identity.get("source_commit"), f"{name} source commit")
    require(identities["statistics"]["source_commit"] == record["source_commit"], "statistics identity differs")
    require(identities["oracle"]["source_commit"] == export_body["oracle"]["manifest"]["body"]["source_commit"], "oracle identity differs")
    require(identities["llama_cpp_metal"]["source_commit"] == metal_body["subject"]["manifest"]["body"]["source_commit"], "llama Metal identity differs")
    require(identities["leone_metal"]["source_commit"] == metal_body["native_source_commit"], "native identity differs")
    require(identities["leone_metal"].get("expected_source_commit") == identities["leone_metal"]["source_commit"], "native expected source identity differs")
    require(metal_body["native_execution"]["device"] == "metal", "native comparison subject is not Metal")
    require(metal_body["execution"]["device"] == "metal", "llama comparison subject is not Metal")
    require(field(record, "oracle", dict)["execution"]["device"] == "cpu", "comparison oracle is not CPU")
    subjects = field(record, "subjects", dict)
    llama_subject = field(subjects, "llama_cpp", dict)
    leone_subject = field(subjects, "leone", dict)
    comparison_oracle = field(record, "oracle", dict)
    require(comparison_oracle["engine"] == export_body["engine"], "comparison oracle engine differs")
    require(comparison_oracle["execution"] == oracle_run["execution"], "comparison oracle execution differs")
    require(comparison_oracle["logits"] == export_body["oracle"]["logits"], "comparison oracle logits differ")
    require(llama_subject["execution"]["device"] == "metal", "llama comparison subject is not Metal")
    require(leone_subject["execution"]["backend"] == "metal", "Leone comparison subject is not Metal")
    require(llama_subject["engine"] == {"name": "llama.cpp-metal", "git_commit": PINNED}, "llama subject engine differs")
    require(leone_subject["engine"] == {"name": "leone-metal", "git_commit": native_run["source_commit"]}, "Leone subject engine differs")
    require(llama_subject["execution"] == metal_body["execution"], "llama subject execution differs")
    require(llama_subject["logits"] == metal_body["subject"]["logits"], "llama subject logits differ")
    require(leone_subject["execution"] == metal_body["native_execution"], "Leone subject execution differs")
    require(leone_subject["logits"] == metal_body["native"]["logits"], "Leone subject logits differ")
    require_trusted_inputs(record, trusted, export_body, metal_body, task_body, sample_body)
    quality = field(record, "quality", dict)
    llama_quality = field(quality, "llama_cpp", dict)
    leone_quality = field(quality, "leone", dict)
    sample_logits = sample_body["artifacts"]["logits"]
    oracle_logits = path.parent / sample_logits["oracle"]["path"]
    llama_logits = path.parent / sample_logits["llama_cpp"]["path"]
    leone_logits = path.parent / sample_logits["leone"]["path"]
    validate_quality_link(
        path,
        llama_quality,
        export_body,
        llama_subject,
        {"name": "llama.cpp-metal", "git_commit": PINNED},
        oracle_logits,
        llama_logits,
        sample_body,
        sample_logits["oracle"],
        sample_logits["llama_cpp"],
        receipt_verifier,
        "llama.cpp Metal",
    )
    validate_quality_link(
        path,
        leone_quality,
        export_body,
        leone_subject,
        {"name": "leone-metal", "git_commit": metal_body["native_source_commit"]},
        oracle_logits,
        leone_logits,
        sample_body,
        sample_logits["oracle"],
        sample_logits["leone"],
        receipt_verifier,
        "Leone Metal",
    )
    return record


USAGE = "usage: validate-quality-stage.py export|metal|compare|comparison|task PATH [PATH] [PATH]"


def run_task_command(paths: list[str]) -> int:
    if len(paths) not in (3, 4):
        raise ValueError(USAGE)
    sample_manifest = Path(paths[3]) if len(paths) == 4 else None
    result = validate_task(Path(paths[0]), Path(paths[1]), Path(paths[2]), sample_manifest)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def run_stage_command(command: str, paths: list[str]) -> None:
    if command == "export":
        if len(paths) != 1:
            raise ValueError(USAGE)
        validate_export(Path(paths[0]))
    elif command == "metal":
        if len(paths) != 1:
            raise ValueError(USAGE)
        validate_metal(Path(paths[0]))
    elif command == "compare":
        if len(paths) != 2:
            raise ValueError(USAGE)
        validate_compare(Path(paths[0]), Path(paths[1]))
    elif command == "comparison":
        manifest, verifier, trusted, generation = parse_trusted_options(paths)
        validate_comparison(manifest, verifier, trusted, generation)
    else:
        raise ValueError(USAGE)


def main(arguments: list[str]) -> int:
    if not arguments:
        raise ValueError(USAGE)
    command, *paths = arguments
    if command == "task":
        return run_task_command(paths)
    run_stage_command(command, paths)
    print("quality stage passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
