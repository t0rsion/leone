#!/usr/bin/env python3
"""Package a reproducible CUDA quality comparison from retained logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PINNED = (ROOT / "external/PINNED").read_text().strip()
SAMPLE_ROWS = 128
SAMPLE_CONTRACT = "linspace-inclusive-v1:128+task-rows"
MIN_LONG_CONTEXT_TOKENS = 4096
GENERATION_SCHEMA = "leone.quality-generation.v1"


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read JSON {path}: {error}")
    require(isinstance(value, dict), f"JSON value is not an object: {path}")
    return value


def load_trusted_inputs(path: Path) -> dict:
    regular_file(path, "trusted CUDA inputs")
    body = load(path)
    require(body.get("schema_version") == "leone.quality-trusted-cuda.v1", "trusted CUDA input schema differs")
    trusted = body.get("trusted")
    require(isinstance(trusted, dict), "trusted CUDA inputs are missing")
    return trusted


def load_generation_record(path: Path) -> dict:
    regular_file(path, "CUDA generation record")
    body = load(path)
    require(body.get("schema_version") == GENERATION_SCHEMA, "CUDA generation record schema differs")
    stable_strings(body, "generation")
    return body


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_digest(value: object) -> str:
    encoded = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def publish_temp(path: Path, writer) -> None:
    require(not os.path.lexists(path), f"refusing to replace generated file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            writer(target)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            fail(f"refusing to replace generated file: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def write_no_replace(path: Path, body: str) -> None:
    publish_temp(path, lambda target: target.write(body.encode()))


def copy_file(source: Path, target) -> None:
    with source.open("rb") as source_file:
        shutil.copyfileobj(source_file, target)


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file: {path}")


def stable_strings(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            stable_strings(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            stable_strings(item, f"{label}[{index}]")
    elif isinstance(value, str):
        require(not os.path.isabs(value), f"{label} contains an absolute path")
        require(not value.startswith("~"), f"{label} contains a home-relative path")


def public_name(value: object, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} is missing")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"{label} path is private or escapes")
    name = path.name
    require(name not in ("", ".", ".."), f"{label} has no public name")
    return name


def valid_hash(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    require(len(value) == 64 and all(character in "0123456789abcdef" for character in value), f"{label} is not a SHA-256")
    return value


def valid_commit(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    require(len(value) == 40 and all(character in "0123456789abcdef" for character in value), f"{label} is not a Git object ID")
    return value


def task_text_bytes(record: dict, label: str) -> bytes:
    require(record.get("encoding") == "utf-8", f"{label} encoding differs")
    text = record.get("text")
    require(isinstance(text, str), f"{label} text is missing")
    raw = text.encode("utf-8")
    require(record.get("bytes") == len(raw), f"{label} byte count differs")
    require(valid_hash(record.get("sha256"), f"{label} SHA-256") == hashlib.sha256(raw).hexdigest(), f"{label} SHA-256 differs")
    return raw


def copy_artifact(source: Path, destination: Path, relative: str, encoding: str) -> dict:
    regular_file(source, "artifact")
    destination.parent.mkdir(parents=True, exist_ok=True)
    publish_temp(destination, lambda target: copy_file(source, target))
    return {
        "path": relative,
        "sha256": digest(destination),
        "bytes": destination.stat().st_size,
        "encoding": encoding,
    }


def snapshot(source: Path, destination: Path, relative: str, body: dict) -> dict:
    record = copy_artifact(source, destination, relative, "json")
    record["body"] = body
    return record


def sample_indices(rows: int) -> list[int]:
    requested = min(SAMPLE_ROWS, rows)
    if requested == 1:
        return [0]
    return [index * (rows - 1) // (requested - 1) for index in range(requested)]


def source_logits(path: Path, manifest: dict, logical: str, rows: int, vocab: int, label: str) -> dict:
    regular_file(path, label)
    record = manifest.get("logits")
    require(isinstance(record, dict), f"{label} manifest logits are missing")
    require(record.get("encoding") == "row-major-f32-le", f"{label} logits encoding differs")
    expected_bytes = rows * vocab * 4
    require(record.get("bytes") == expected_bytes, f"{label} manifest logits shape differs")
    require(path.stat().st_size == expected_bytes, f"{label} logits file shape differs")
    require(digest(path) == record.get("sha256"), f"{label} logits hash differs from its manifest")
    return {
        "path": logical,
        "sha256": record["sha256"],
        "bytes": expected_bytes,
        "encoding": "row-major-f32-le",
    }


def sample_file(
    source: Path,
    destination: Path,
    package_root: Path,
    indices: list[int],
    vocab: int,
    expected_hash: str,
) -> dict:
    require(digest(source) == expected_hash, f"source logits changed before sampling: {source}")
    row_bytes = vocab * 4
    row_hashes = []

    def write_rows(target) -> None:
        with source.open("rb") as source_file:
            for row in indices:
                source_file.seek(row * row_bytes)
                data = source_file.read(row_bytes)
                require(len(data) == row_bytes, f"logits source row is truncated: {source}")
                target.write(data)
                row_hashes.append(hashlib.sha256(data).hexdigest())

    publish_temp(destination, write_rows)
    require(digest(source) == expected_hash, f"source logits changed during sampling: {source}")
    return {
        "path": destination.relative_to(package_root).as_posix(),
        "sha256": digest(destination),
        "bytes": destination.stat().st_size,
        "encoding": "row-major-f32-le",
        "source_row_sha256": row_hashes,
    }


def verify_originals(options: argparse.Namespace, full_records: dict) -> None:
    for label, source in (
        ("oracle", options.oracle_logits),
        ("llama_cpp", options.llama_logits),
        ("leone", options.leone_logits),
    ):
        require(digest(source) == full_records[label]["sha256"], f"generation-only {label} logits changed before publication")


def receipt_from_quality(
    binary: Path,
    package_root: Path,
    oracle: Path,
    subject: Path,
    corpus: Path,
    tokens: Path,
    oracle_model: Path,
    subject_model: Path,
    oracle_dtype: str,
    subject_engine: str,
    subject_commit: str,
) -> dict:
    with tempfile.TemporaryDirectory(dir=package_root) as temporary:
        working = Path(temporary)
        links = {
            "oracle.f32": oracle,
            "subject.f32": subject,
            "corpus.txt": corpus,
            "tokens.u32le": tokens,
            "oracle-model.gguf": oracle_model,
            "subject-model.gguf": subject_model,
        }
        for name, source in links.items():
            (working / name).symlink_to(source)
        result = subprocess.run(
            [
                str(binary),
                "quality",
                "--oracle",
                "oracle.f32",
                "--subject",
                "subject.f32",
                "--corpus",
                "corpus.txt",
                "--tokens",
                "tokens.u32le",
                "--oracle-model",
                "oracle-model.gguf",
                "--subject-model",
                "subject-model.gguf",
                "--oracle-engine",
                "llama.cpp",
                "--oracle-commit",
                PINNED,
                "--oracle-dtype",
                oracle_dtype.lower(),
                "--subject-engine",
                subject_engine,
                "--subject-commit",
                subject_commit,
                "--receipt",
            ],
            cwd=working,
            check=True,
            capture_output=True,
            text=True,
        )
        reported_path = next(
            (
                Path(line.split(": ", 1)[1])
                for line in result.stdout.splitlines()
                if line.startswith("receipt: ")
            ),
            None,
        )
        require(reported_path is not None, "quality did not report a receipt")
        receipt_path = reported_path if reported_path.is_absolute() else working / reported_path
        require(receipt_path.is_file(), "quality did not produce a receipt")
        return load(receipt_path)


def parse_options(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--receipt-verifier", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--build-info", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--statistics-platform", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--statistics-target", required=True)
    parser.add_argument("--trusted-inputs", type=Path, required=True)
    parser.add_argument("--generation-record", type=Path, required=True)
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--subject-model", type=Path, required=True)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--oracle-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--leone-logits", type=Path, required=True)
    parser.add_argument("--oracle-manifest", type=Path, required=True)
    parser.add_argument("--llama-manifest", type=Path, required=True)
    parser.add_argument("--leone-manifest", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--leone-backend", required=True)
    return parser.parse_args(arguments)


def validate_native_manifest(
    native: dict,
    options: argparse.Namespace,
    trusted: dict,
    build_info: dict,
    input_record: dict,
    llama_model: dict,
    rows: int,
    vocab: int,
) -> None:
    require(native.get("schema_version") == "leone.native-eval.v1", "Leone eval metadata schema differs")
    identity = trusted["producer_identities"]["leone_cuda"]
    require(native.get("source_commit") == identity["source_commit"], "Leone metadata source differs")
    executable = native.get("executable", {})
    require(executable.get("name") == identity["executable"]["name"], "Leone metadata executable differs")
    require(executable.get("sha256") == identity["executable"]["sha256"], "Leone metadata executable hash differs")
    native_build = executable.get("build_info")
    require(isinstance(native_build, dict), "Leone metadata build info is missing")
    require(json_digest(native_build) == identity["build_info_sha256"], "Leone metadata build info differs")
    require(native_build.get("schema_version") == "leone.build-info.v1", "Leone metadata build info schema differs")
    require(native_build.get("source_commit") == identity["source_commit"], "Leone metadata build source differs")
    require(native_build.get("source_tree_dirty") is False, "Leone metadata build is dirty")
    require(native_build.get("profile") == "release", "Leone metadata build is not release")
    require(native_build.get("target") == identity["target"], "Leone metadata build target differs")
    model = native.get("model", {})
    require(model.get("sha256") == llama_model.get("sha256"), "Leone metadata model differs")
    require(model.get("architecture") == llama_model.get("architecture"), "Leone metadata architecture differs")
    require(model.get("vocab_size") == vocab, "Leone metadata vocabulary differs")
    actual_input = native.get("input", {})
    require(actual_input == input_record, "Leone metadata input differs")
    execution = native.get("execution", {})
    require(execution.get("engine") == "leone", "Leone metadata engine differs")
    require(execution.get("device") == "cuda", "Leone metadata device differs")
    require(execution.get("backend") == options.leone_backend == "cuda", "Leone metadata backend differs")
    require(str(execution.get("backend_registry", "")).lower() == "cuda", "Leone metadata registry differs")
    require(execution.get("device_type") == "gpu", "Leone metadata device type differs")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], "Leone device name is missing")
    require(isinstance(execution.get("device_description"), str) and execution["device_description"], "Leone device description is missing")
    require(execution.get("logits_dtype") == "f32", "Leone metadata logits dtype differs")
    prefill = execution.get("prefill", {})
    workflow = trusted["native_workflow"]
    require(execution.get("kv_cache_dtype") == workflow["kv_cache_dtype"], "Leone metadata KV cache dtype differs")
    require(prefill == workflow["prefill"], "Leone metadata prefill differs")
    require(input_record["tokens"]["count"] == workflow["token_count"], "Leone metadata token count differs")
    source = native.get("logits", {})
    require(source.get("bytes") == rows * vocab * 4, "Leone metadata logits shape differs")
    require(source.get("encoding") == "row-major-f32-le", "Leone metadata logits encoding differs")
    require(source.get("sha256") == digest(options.leone_logits), "Leone metadata logits hash differs")
    require(source.get("bytes") == options.leone_logits.stat().st_size, "Leone metadata logits bytes differ")


def validate_identity_record(identity: dict, label: str, require_build: bool = False, require_target: bool = False) -> None:
    require(isinstance(identity, dict), f"trusted {label} identity is missing")
    if "manifest_sha256" in identity:
        valid_hash(identity.get("manifest_sha256"), f"trusted {label} manifest SHA-256")
    valid_commit(identity.get("source_commit"), f"trusted {label} source commit")
    executable = identity.get("executable")
    require(isinstance(executable, dict), f"trusted {label} executable identity is missing")
    public_name(executable.get("name"), f"trusted {label} executable name")
    valid_hash(executable.get("sha256"), f"trusted {label} executable SHA-256")
    if require_build:
        valid_hash(identity.get("build_info_sha256"), f"trusted {label} build info SHA-256")
    if require_target:
        target = identity.get("target")
        require(isinstance(target, str) and target and not Path(target).is_absolute(), f"trusted {label} target is invalid")


def validate_trusted_inputs(
    trusted: dict,
    options: argparse.Namespace,
    build_info: dict,
    oracle_run: dict,
    llama_run: dict,
    native_run: dict,
) -> None:
    required = (
        "source_commit", "platform", "statistics_platform", "target", "statistics_target", "backend",
        "adapter_path", "adapter_sha256", "model_family", "model_sha256",
        "oracle_model_sha256", "corpus_sha256", "task_manifest_sha256",
        "sample_contract", "metric_family", "producer_identities", "statistics_identity", "native_workflow",
    )
    for name in required:
        require(name in trusted, f"trusted CUDA input {name} is missing")
    expected = {
        "source_commit": options.source_commit,
        "platform": options.platform,
        "statistics_platform": options.statistics_platform,
        "target": options.target,
        "statistics_target": options.statistics_target,
        "backend": "cuda",
        "adapter_path": "research/oracle/llama_logits.cpp",
        "model_family": options.model_family,
        "sample_contract": SAMPLE_CONTRACT,
        "metric_family": "kld",
    }
    for name, value in expected.items():
        require(trusted.get(name) == value, f"trusted CUDA input {name} differs from invocation")
    for name in ("adapter_sha256", "model_sha256", "oracle_model_sha256", "corpus_sha256", "task_manifest_sha256"):
        valid_hash(trusted.get(name), f"trusted CUDA input {name}")
    require(trusted["backend"] == "cuda", "trusted CUDA backend differs")
    identities = trusted["producer_identities"]
    require(isinstance(identities, dict), "trusted CUDA producer identities are invalid")
    validate_identity_record(identities.get("oracle_stage"), "oracle stage")
    validate_identity_record(identities.get("llama_cpp_cuda"), "llama.cpp CUDA")
    validate_identity_record(identities.get("leone_cuda"), "Leone CUDA", True, True)
    validate_identity_record(trusted.get("statistics_identity"), "statistics", True, True)
    require(identities["leone_cuda"]["target"] == trusted["target"], "trusted native target differs")
    workflow = trusted["native_workflow"]
    require(isinstance(workflow, dict), "trusted CUDA native workflow is invalid")
    require(isinstance(workflow.get("kv_cache_dtype"), str) and workflow["kv_cache_dtype"], "trusted CUDA KV cache dtype is invalid")
    require(isinstance(workflow.get("prefill"), dict), "trusted CUDA prefill contract is invalid")
    require(workflow["prefill"].get("path") in ("sequential", "chunked"), "trusted CUDA prefill path is invalid")
    if workflow["prefill"].get("path") == "chunked":
        require(isinstance(workflow["prefill"].get("chunk_tokens"), int) and workflow["prefill"]["chunk_tokens"] > 0, "trusted CUDA prefill chunk is invalid")
    else:
        require(workflow["prefill"].get("chunk_tokens") is None, "trusted CUDA sequential prefill has a chunk")
    require(isinstance(workflow.get("token_count"), int) and workflow["token_count"] >= 2, "trusted CUDA token count is invalid")
    observed = {
        "oracle_stage": (oracle_run, options.oracle_manifest),
        "llama_cpp_cuda": (llama_run, options.llama_manifest),
        "leone_cuda": (native_run, options.leone_manifest),
    }
    for label, (run, manifest_path) in observed.items():
        validate_observed_identity(label, run, manifest_path, identities[label])
    statistics = trusted["statistics_identity"]
    require(statistics["source_commit"] == options.source_commit, "trusted statistics source differs")
    require(statistics.get("target") == options.statistics_target, "trusted statistics target differs")
    require(statistics["executable"]["name"] == options.binary.name, "trusted statistics executable differs")
    require(statistics["executable"]["sha256"] == digest(options.binary), "trusted statistics executable hash differs")
    require(statistics["target"] == options.statistics_target, "trusted statistics target differs")
    require(build_info["target"] == statistics["target"], "statistics build target differs from trusted identity")
    require(statistics["build_info_sha256"] == json_digest(build_info), "trusted statistics build info differs")


def generation_metadata(record: dict, label: str) -> dict:
    value = record.get(label)
    require(isinstance(value, dict), f"CUDA generation {label} record is missing")
    valid_hash(value.get("sha256"), f"CUDA generation {label} SHA-256")
    require(isinstance(value.get("bytes"), int) and value["bytes"] > 0, f"CUDA generation {label} byte count is invalid")
    require(value.get("encoding") == "row-major-f32-le", f"CUDA generation {label} encoding differs")
    return {name: value[name] for name in ("sha256", "bytes", "encoding")}


def validate_generation_record(
    generation: dict,
    options: argparse.Namespace,
    trusted: dict,
    input_record: dict,
    rows: int,
    vocab: int,
    task_rows: list[int],
    full_records: dict,
) -> None:
    require(generation.get("schema_version") == GENERATION_SCHEMA, "CUDA generation record schema differs")
    inputs = generation.get("inputs")
    require(isinstance(inputs, dict), "CUDA generation inputs are missing")
    expected_inputs = {
        "model_sha256": trusted["model_sha256"],
        "oracle_model_sha256": trusted["oracle_model_sha256"],
        "corpus_sha256": trusted["corpus_sha256"],
        "task_manifest_sha256": trusted["task_manifest_sha256"],
        "tokens_sha256": input_record["tokens"]["sha256"],
        "token_count": input_record["tokens"]["count"],
        "rows": rows,
        "vocab_size": vocab,
        "sample_contract": SAMPLE_CONTRACT,
    }
    for name, expected in expected_inputs.items():
        require(inputs.get(name) == expected, f"CUDA generation input {name} differs")
    sampling = generation.get("sampling")
    require(isinstance(sampling, dict), "CUDA generation sampling is missing")
    expected_indices = sorted(set(sample_indices(rows) + task_rows))
    require(sampling.get("algorithm") == "linspace-inclusive-v1", "CUDA generation sampling algorithm differs")
    require(sampling.get("requested_rows") == min(SAMPLE_ROWS, rows), "CUDA generation sample count differs")
    require(sampling.get("indices") == expected_indices, "CUDA generation sample indices differ")
    require(sampling.get("task_rows") == task_rows, "CUDA generation task rows differ")
    manifests = generation.get("producer_manifests")
    require(isinstance(manifests, dict), "CUDA generation producer manifests are missing")
    for label, path in (
        ("oracle", options.oracle_manifest),
        ("llama_cpp", options.llama_manifest),
        ("leone", options.leone_manifest),
    ):
        valid_hash(manifests.get(label), f"CUDA generation {label} manifest SHA-256")
        require(digest(path) == manifests[label], f"CUDA generation {label} manifest differs")
    full_logits = generation.get("full_logits")
    require(isinstance(full_logits, dict), "CUDA generation full logits are missing")
    for label in ("oracle", "llama_cpp", "leone"):
        require(generation_metadata(full_logits, label) == {name: full_records[label][name] for name in ("sha256", "bytes", "encoding")}, f"CUDA generation {label} full logits differ")


def validate_generation_samples(generation: dict, samples: dict) -> None:
    expected = generation.get("sampled_logits")
    require(isinstance(expected, dict), "CUDA generation sampled logits are missing")
    for label in ("oracle", "llama_cpp", "leone"):
        expected_record = expected.get(label)
        require(isinstance(expected_record, dict), f"CUDA generation {label} sampled record is missing")
        actual_record = samples[label]
        for name in ("sha256", "bytes", "encoding", "source_row_sha256"):
            require(actual_record.get(name) == expected_record.get(name), f"CUDA generation {label} sampled {name} differs")


def validate_source_execution(run: dict, label: str, device: str) -> None:
    execution = run.get("execution", {})
    require(execution.get("device") == device, f"{label} producer device differs")
    require(str(execution.get("backend_registry", "")).lower() == device, f"{label} producer backend differs")
    device_type = "cpu" if device == "cpu" else "gpu"
    require(execution.get("device_type") == device_type, f"{label} producer device type differs")
    require(isinstance(execution.get("device_name"), str) and execution["device_name"], f"{label} producer device name is missing")
    require(isinstance(execution.get("device_description"), str), f"{label} producer device description differs")
    require(execution.get("logits_dtype") == "f32", f"{label} producer logits dtype differs")


def validate_observed_identity(label: str, run: dict, manifest_path: Path, identity: dict) -> None:
    if "manifest_sha256" in identity:
        require(digest(manifest_path) == identity["manifest_sha256"], f"trusted {label} manifest differs")
    require(run.get("source_commit") == identity["source_commit"], f"trusted {label} source differs")
    executable = run.get("executable", {})
    name = executable.get("name", Path(executable.get("path", "")).name)
    require(name == identity["executable"]["name"], f"trusted {label} executable differs")
    require(executable.get("sha256") == identity["executable"]["sha256"], f"trusted {label} executable hash differs")
    if label == "leone_cuda":
        require(json_digest(executable.get("build_info")) == identity["build_info_sha256"], f"trusted {label} build info differs")


def validate_options(options: argparse.Namespace) -> tuple[dict, dict, dict, dict, dict, list[int], dict, int, int, dict, dict]:
    validate_option_paths(options)
    build_info = load(options.build_info)
    require(build_info.get("schema_version") == "leone.build-info.v1", "build info schema differs")
    require(build_info.get("source_commit") == options.source_commit, "build info source differs")
    require(build_info.get("source_tree_dirty") is False, "build is dirty")
    require(build_info.get("profile") == "release", "build is not release")
    require(build_info.get("target") == options.statistics_target, "statistics build target differs")
    oracle_run = load(options.oracle_manifest)
    llama_run = load(options.llama_manifest)
    native_run = load(options.leone_manifest)
    task_body = load(options.task_manifest)
    trusted = load_trusted_inputs(options.trusted_inputs)
    generation = load_generation_record(options.generation_record)
    require(oracle_run.get("engine", {}).get("name") == "llama.cpp", "oracle manifest engine differs")
    require(oracle_run.get("engine", {}).get("git_commit") == PINNED, "oracle manifest pin differs")
    require(llama_run.get("engine", {}).get("name") == "llama.cpp", "llama manifest engine differs")
    require(llama_run.get("engine", {}).get("git_commit") == PINNED, "llama manifest pin differs")
    require(oracle_run.get("input") == llama_run.get("input"), "oracle and llama inputs differ")
    oracle_model = oracle_run.get("model", {})
    llama_model = llama_run.get("model", {})
    require(oracle_model.get("storage_type") in ("BF16", "F16"), "oracle model is not full precision")
    require(llama_model.get("storage_type") == "Q4_K - Medium", "llama subject model is not Q4_K - Medium")
    require(digest(options.oracle_model) == oracle_model.get("sha256"), "oracle model digest differs")
    require(digest(options.subject_model) == llama_model.get("sha256"), "subject model digest differs")
    validate_source_adapters(oracle_run, llama_run)
    input_record = oracle_run["input"]
    token_record = input_record.get("tokens", {})
    require(digest(options.tokens) == token_record.get("sha256"), "token artifact digest differs")
    require(token_record.get("bytes") == options.tokens.stat().st_size, "token artifact byte count differs")
    require(token_record.get("encoding") == "u32le", "token artifact encoding differs")
    require(options.tokens.stat().st_size == token_record.get("count", 0) * 4, "token artifact shape differs")
    rows = input_record["rows"]
    vocab = input_record["vocab_size"]
    require(isinstance(rows, int) and rows > 0 and isinstance(vocab, int) and vocab > 0, "quality input shape is invalid")
    full_records = {
        "oracle": source_logits(options.oracle_logits, oracle_run, "generated/oracle.f32", rows, vocab, "oracle logits"),
        "llama_cpp": source_logits(options.llama_logits, llama_run, "generated/llama-cpp-cuda.f32", rows, vocab, "llama logits"),
        "leone": source_logits(options.leone_logits, native_run, "generated/leone-cuda.f32", rows, vocab, "Leone logits"),
    }
    validate_source_execution(oracle_run, "oracle", "cpu")
    validate_source_execution(llama_run, "llama.cpp", "cuda")
    validate_trusted_inputs(trusted, options, build_info, oracle_run, llama_run, native_run)
    require(trusted["adapter_sha256"] == digest(ROOT / "research/oracle/llama_logits.cpp"), "trusted adapter hash differs from the source")
    require(trusted["model_sha256"] == llama_model.get("sha256"), "trusted subject model differs from the manifest")
    require(trusted["oracle_model_sha256"] == oracle_model.get("sha256"), "trusted oracle model differs from the manifest")
    require(trusted["corpus_sha256"] == digest(options.corpus), "trusted corpus differs from the input")
    require(trusted["task_manifest_sha256"] == digest(options.task_manifest), "trusted task manifest differs from the input")
    validate_native_manifest(native_run, options, trusted, build_info, input_record, llama_model, rows, vocab)
    task_rows = validate_task_manifest(
        task_body,
        input_record,
        oracle_model,
        llama_model,
        digest(options.corpus),
        options.model_family,
        options.tokens,
        rows,
    )
    validate_generation_record(generation, options, trusted, input_record, rows, vocab, task_rows, full_records)
    return build_info, oracle_run, llama_run, native_run, task_body, task_rows, input_record, rows, vocab, trusted, generation


def validate_option_paths(options: argparse.Namespace) -> None:
    require(not os.path.lexists(options.output), f"refusing to replace comparison manifest: {options.output}")
    path_names = (
        "build_info",
        "subject_model",
        "oracle_model",
        "corpus",
        "tokens",
        "oracle_logits",
        "llama_logits",
        "leone_logits",
        "oracle_manifest",
        "llama_manifest",
        "leone_manifest",
        "task_manifest",
        "binary",
        "receipt_verifier",
        "trusted_inputs",
        "generation_record",
    )
    for name in path_names:
        path = getattr(options, name)
        regular_file(path, name)
        setattr(options, name, path.resolve())
    require(options.leone_backend == "cuda", "CUDA comparison requires the CUDA Leone backend")
    require(options.receipt_verifier.name == "leone-receipt-verify", "receipt verifier name is not canonical")
    require(options.platform == "linux-x86_64", "CUDA comparison requires linux-x86_64")
    require(options.statistics_platform and not Path(options.statistics_platform).is_absolute(), "CUDA statistics platform is invalid")
    require(options.target == "x86_64-unknown-linux-gnu", "CUDA comparison requires the Linux CUDA target")
    require(options.statistics_target and not Path(options.statistics_target).is_absolute(), "CUDA statistics target is invalid")
    require(
        len(options.source_commit) == 40
        and all(c in "0123456789abcdef" for c in options.source_commit),
        "source commit is invalid",
    )
    require(
        options.output.parent.exists() and not options.output.parent.is_symlink(),
        "comparison output parent is missing or is a symlink",
    )
    for suffix in ("llama-cpp-quality.json", "leone-quality.json"):
        quality_path = options.output.parent / f"{options.output.stem}-{suffix}"
        require(not os.path.lexists(quality_path), f"refusing to replace quality receipt: {quality_path}")


def validate_source_adapters(oracle_run: dict, llama_run: dict) -> None:
    adapter_sha = digest(ROOT / "research/oracle/llama_logits.cpp")
    for label, run in (("oracle", oracle_run), ("llama", llama_run)):
        adapter = run.get("adapter", {})
        require(adapter.get("path") == "research/oracle/llama_logits.cpp", f"{label} adapter path differs")
        require(adapter.get("sha256") == adapter_sha, f"{label} adapter hash differs")


def validate_task_manifest(
    task: dict,
    input_record: dict,
    oracle_model: dict,
    subject_model: dict,
    corpus_sha256: str,
    model_family: str,
    tokens_path: Path,
    rows: int,
) -> list[int]:
    require(task.get("schema_version") == "leone.quality-task.v2", "quality task schema differs")
    require(task.get("input") == input_record, "quality task input differs")
    require(task.get("corpus", {}).get("sha256") == corpus_sha256, "quality task corpus differs")
    task_models = task.get("model", {})
    require(task_models.get("family") == model_family, "quality task model family differs")
    require(task_models.get("oracle", {}).get("sha256") == oracle_model.get("sha256"), "quality task oracle differs")
    require(task_models.get("subject", {}).get("sha256") == subject_model.get("sha256"), "quality task subject differs")
    task_spec = validate_task_metadata(task, oracle_model, corpus_sha256, input_record)
    return validate_task_answer(task_spec, task["tokenization"], input_record, tokens_path, rows)


def validate_task_metadata(task: dict, oracle_model: dict, corpus_sha256: str, input_record: dict) -> dict:
    task_spec = task.get("task", {})
    require(task_spec.get("name") == "needle-exact-answer", "quality task name differs")
    context_tokens = task_spec.get("context_tokens")
    minimum_tokens = task_spec.get("minimum_context_tokens")
    require(isinstance(context_tokens, int) and not isinstance(context_tokens, bool) and context_tokens >= MIN_LONG_CONTEXT_TOKENS, "quality task context is invalid")
    require(isinstance(minimum_tokens, int) and not isinstance(minimum_tokens, bool) and minimum_tokens >= MIN_LONG_CONTEXT_TOKENS, "quality task minimum context is invalid")
    require(context_tokens >= minimum_tokens, "quality task context is below its declared minimum")
    require(context_tokens == input_record["window_tokens"], "quality task context differs from the evaluation window")
    prompt = task_spec.get("prompt", {})
    needle = task_spec.get("needle", {})
    require(isinstance(prompt, dict) and isinstance(needle, dict), "quality task text records are missing")
    prompt_bytes = task_text_bytes(prompt, "quality task prompt")
    needle_bytes = task_text_bytes(needle, "quality task needle")
    require(hashlib.sha256(prompt_bytes + needle_bytes).hexdigest() == corpus_sha256, "quality task prompt and needle differ from corpus")
    require(task.get("tokenizer", {}).get("name") == oracle_model.get("tokenizer"), "quality task tokenizer differs")
    tokenization = task.get("tokenization", {})
    require(tokenization.get("method") == "llama-tokenize", "quality task tokenizer method differs")
    require(tokenization.get("source", {}).get("name") == "llama.cpp", "quality task tokenizer source differs")
    require(tokenization.get("source", {}).get("git_commit") == PINNED, "quality task tokenizer pin differs")
    require(tokenization.get("model", {}).get("sha256") == oracle_model.get("sha256"), "quality task tokenizer model differs")
    valid_hash(tokenization.get("combined_sha256"), "quality task combined input SHA-256")
    require(tokenization.get("combined_sha256") == corpus_sha256, "quality task corpus tokenization differs")
    valid_hash(tokenization.get("input_tokens_sha256"), "quality task tokenization input SHA-256")
    require(tokenization.get("input_tokens_sha256") == input_record["tokens"]["sha256"], "quality task tokenization input differs")
    return task_spec


def validate_task_answer(task_spec: dict, tokenization: dict, input_record: dict, tokens_path: Path, rows: int) -> list[int]:
    answer = task_spec.get("answer", {})
    context_tokens = task_spec["context_tokens"]
    needle = task_spec["needle"]
    answer_row = answer.get("row")
    answer_tokens = answer.get("token_ids")
    require(isinstance(answer_row, int) and not isinstance(answer_row, bool) and answer_row >= 0, "quality task answer row is invalid")
    require(isinstance(answer_tokens, list) and answer_tokens, "quality task answer tokens are missing")
    require(answer_row >= context_tokens - 1, "quality task answer is before the declared context")
    require(all(isinstance(token, int) and not isinstance(token, bool) and 0 <= token < input_record["vocab_size"] for token in answer_tokens), "quality task answer token ID is invalid")
    require(answer.get("text") == needle.get("text"), "quality task answer text differs")
    require(answer.get("sha256") == needle.get("sha256"), "quality task answer needle differs")
    packed_answer = struct.pack(f"<{len(answer_tokens)}I", *answer_tokens)
    require(tokenization.get("answer_tokens_sha256") == hashlib.sha256(packed_answer).hexdigest(), "quality task answer tokenization differs")
    task_rows = list(range(answer_row, answer_row + len(answer_tokens)))
    require(task_rows[-1] < rows, "quality task answer exceeds scored rows")
    token_values = struct.unpack(f"<{input_record['tokens']['count']}I", tokens_path.read_bytes())
    require(list(token_values[answer_row + 1 : answer_row + 1 + len(answer_tokens)]) == answer_tokens, "quality task answer does not match the token stream")
    return task_rows


def package_quality_artifacts(
    options: argparse.Namespace,
    oracle_run: dict,
    llama_run: dict,
    native_run: dict,
    task_body: dict,
    task_rows: list[int],
    input_record: dict,
    rows: int,
    vocab: int,
    generation: dict,
) -> dict:
    package_root = options.output.parent
    base = options.output.stem
    artifact_dir = package_root / f"{base}-cuda-artifacts"
    sample_dir = package_root / f"{base}-cuda-samples"
    require(not os.path.lexists(artifact_dir), "comparison artifact directory already exists")
    require(not os.path.lexists(sample_dir), "comparison sample directory already exists")
    artifact_dir.mkdir()
    sample_dir.mkdir()
    token_record = copy_artifact(
        options.tokens,
        artifact_dir / "tokens.u32le",
        f"{artifact_dir.name}/tokens.u32le",
        "u32le",
    )
    token_record["count"] = input_record["tokens"]["count"]
    manifests = {}
    for label, source, body in (
        ("oracle", options.oracle_manifest, oracle_run),
        ("llama_cpp", options.llama_manifest, llama_run),
        ("leone", options.leone_manifest, native_run),
    ):
        target = artifact_dir / f"{label}-manifest.json"
        manifests[label] = snapshot(source, target, f"{artifact_dir.name}/{target.name}", body)
    task_path = artifact_dir / "long-context-task.json"
    task_record = snapshot(
        options.task_manifest,
        task_path,
        f"{artifact_dir.name}/{task_path.name}",
        task_body,
    )
    generation_path = artifact_dir / "generation-record.json"
    generation_record = snapshot(
        options.generation_record,
        generation_path,
        f"{artifact_dir.name}/{generation_path.name}",
        generation,
    )
    adapter_source = ROOT / "research/oracle/llama_logits.cpp"
    adapter_record = copy_artifact(
        adapter_source,
        artifact_dir / "llama_logits.cpp",
        f"{artifact_dir.name}/llama_logits.cpp",
        "source",
    )
    indices = sorted(set(sample_indices(rows) + task_rows))
    source_values = struct.unpack(f"<{input_record['tokens']['count']}I", options.tokens.read_bytes())
    sample_tokens_path = sample_dir / "sample.tokens.u32le"
    sample_token_bytes = struct.pack(
        f"<{len(indices) + 1}I",
        source_values[0],
        *[source_values[row + 1] for row in indices],
    )
    publish_temp(sample_tokens_path, lambda target: target.write(sample_token_bytes))
    sample_token_record = {
        "path": f"{sample_dir.name}/sample.tokens.u32le",
        "sha256": digest(sample_tokens_path),
        "bytes": sample_tokens_path.stat().st_size,
        "count": len(indices) + 1,
        "encoding": "u32le",
    }
    full_records = {
        "oracle": source_logits(
            options.oracle_logits,
            oracle_run,
            "generated/oracle.f32",
            rows,
            vocab,
            "oracle logits",
        ),
        "llama_cpp": source_logits(
            options.llama_logits,
            llama_run,
            "generated/llama-cpp-cuda.f32",
            rows,
            vocab,
            "llama logits",
        ),
        "leone": source_logits(
            options.leone_logits,
            native_run,
            "generated/leone-cuda.f32",
            rows,
            vocab,
            "Leone logits",
        ),
    }
    samples = {}
    for label, source_path in (
        ("oracle", options.oracle_logits),
        ("llama_cpp", options.llama_logits),
        ("leone", options.leone_logits),
    ):
        samples[label] = sample_file(
            source_path,
            sample_dir / f"{label}.f32",
            package_root,
            indices,
            vocab,
            full_records[label]["sha256"],
        )
    validate_generation_samples(generation, samples)
    verify_originals(options, full_records)
    sample_body = {
        "schema_version": "leone.quality-sample.v1",
        "provenance": {
            "full_logits": "generation-only",
            "row_hashes": "sha256 of each packaged row in source order",
            "offline_replay": "sampled rows only",
        },
        "source": {
            "rows": rows,
            "vocab_size": vocab,
            "tokens": token_record,
            "logits": full_records,
        },
        "originals": {
            "mode": "generation-only",
            "retention": "caller-owned immutable cache",
            "logits": full_records,
        },
        "sampling": {
            "algorithm": "linspace-inclusive-v1",
            "requested_rows": min(SAMPLE_ROWS, rows),
            "source_rows": rows,
            "sample_count": len(indices),
            "indices": indices,
            "task_rows": task_rows,
        },
        "artifacts": {"tokens": sample_token_record, "logits": samples},
    }
    sample_manifest_path = sample_dir / "sampling.json"
    write_no_replace(sample_manifest_path, json.dumps(sample_body, indent=2, sort_keys=True) + "\n")
    sample_manifest_record = {
        "path": f"{sample_dir.name}/sampling.json",
        "sha256": digest(sample_manifest_path),
        "bytes": sample_manifest_path.stat().st_size,
        "body": sample_body,
    }
    return {
        "package_root": package_root,
        "base": base,
        "artifact_records": {"tokens": token_record},
        "producer_records": manifests,
        "native_source_commit": native_run["source_commit"],
        "task_record": task_record,
        "generation_record": generation_record,
        "task_rows": task_rows,
        "adapter_record": adapter_record,
        "sample_tokens_path": sample_tokens_path,
        "sample_manifest_record": sample_manifest_record,
        "sample_body": sample_body,
        "sample_dir": sample_dir,
        "full_records": full_records,
    }


def write_quality_files(options: argparse.Namespace, package: dict, oracle_model: dict) -> dict:
    sample_dir = package["sample_dir"]
    quality_bodies = {
        "llama_cpp": receipt_from_quality(
            options.binary,
            package["package_root"],
            sample_dir / "oracle.f32",
            sample_dir / "llama_cpp.f32",
            options.corpus,
            package["sample_tokens_path"],
            options.oracle_model,
            options.subject_model,
            oracle_model["storage_type"],
            "llama.cpp-cuda",
            PINNED,
        ),
        "leone": receipt_from_quality(
            options.binary,
            package["package_root"],
            sample_dir / "oracle.f32",
            sample_dir / "leone.f32",
            options.corpus,
            package["sample_tokens_path"],
            options.oracle_model,
            options.subject_model,
            oracle_model["storage_type"],
            "leone-cuda-eval",
            package["native_source_commit"],
        ),
    }
    quality_paths = {
        label: package["package_root"] / f"{package['base']}-{label.replace('_', '-')}-quality.json"
        for label in quality_bodies
    }
    for label, path in quality_paths.items():
        write_no_replace(path, json.dumps(quality_bodies[label], indent=2, sort_keys=True) + "\n")
    return {"bodies": quality_bodies, "paths": quality_paths}


def build_comparison_body(
    options: argparse.Namespace,
    package: dict,
    quality: dict,
    build_info: dict,
    oracle_run: dict,
    llama_run: dict,
    native_run: dict,
    task_body: dict,
    task_rows: list[int],
    input_record: dict,
    oracle_model: dict,
    llama_model: dict,
    vocab: int,
    trusted: dict,
) -> dict:
    artifact_records = package["artifact_records"]
    source_model = {
        "oracle": {
            "name": public_name(oracle_model.get("path"), "oracle model"),
            "sha256": oracle_model["sha256"],
            "storage_type": oracle_model["storage_type"],
            "vocab_size": vocab,
            "architecture": oracle_model.get("architecture"),
            "tokenizer": oracle_model.get("tokenizer"),
        },
        "subject": {
            "name": public_name(llama_model.get("path"), "subject model"),
            "sha256": llama_model["sha256"],
            "storage_type": llama_model["storage_type"],
            "vocab_size": vocab,
            "architecture": llama_model.get("architecture"),
            "tokenizer": llama_model.get("tokenizer"),
        },
    }
    executable = {"name": options.binary.name, "sha256": digest(options.binary)}
    adapter_record = package["adapter_record"]
    quality_records = {
        label: {"path": path.name, "sha256": digest(path), "receipt": quality["bodies"][label]}
        for label, path in quality["paths"].items()
    }
    return {
        "schema_version": "leone.quality-comparison.v2",
        "source_commit": options.source_commit,
        "statistics_platform": options.statistics_platform,
        "statistics_target": options.statistics_target,
        "executable": executable,
        "build_info": build_info,
        "source_identities": {
            "oracle": trusted["producer_identities"]["oracle_stage"],
            "llama_cpp_cuda": trusted["producer_identities"]["llama_cpp_cuda"],
            "leone_cuda": trusted["producer_identities"]["leone_cuda"],
            "statistics": trusted["statistics_identity"],
        },
        "model_family": options.model_family,
        "models": source_model,
        "corpus": {
            "name": public_name(options.corpus.name, "corpus"),
            "sha256": digest(options.corpus),
        },
        "input": {**input_record, "tokens": artifact_records["tokens"]},
        "adapter": {
            "path": "research/oracle/llama_logits.cpp",
            "sha256": adapter_record["sha256"],
            "artifact": adapter_record,
        },
        "producers": package["producer_records"],
        "tasks": {
            "long_context": {
                "manifest": package["task_record"],
                "rows": task_rows,
            }
        },
        "oracle": {
            "engine": {"name": "llama.cpp", "git_commit": PINNED},
            "model_sha256": source_model["oracle"]["sha256"],
            "input_tokens_sha256": artifact_records["tokens"]["sha256"],
            "execution": oracle_run["execution"],
            "logits": package["full_records"]["oracle"],
        },
        "subjects": {
            "llama_cpp": {
                "engine": {"name": "llama.cpp-cuda", "git_commit": PINNED},
                "model_sha256": source_model["subject"]["sha256"],
                "input_tokens_sha256": artifact_records["tokens"]["sha256"],
                "execution": llama_run["execution"],
                "logits": package["full_records"]["llama_cpp"],
            },
            "leone": {
                "engine": {"name": "leone-cuda-eval", "git_commit": native_run["source_commit"]},
                "model_sha256": source_model["subject"]["sha256"],
                "input_tokens_sha256": artifact_records["tokens"]["sha256"],
                "execution": native_run["execution"],
                "logits": package["full_records"]["leone"],
            },
        },
        "samples": {"manifest": package["sample_manifest_record"]},
        "quality": quality_records,
        "validation": {
            "mode": "offline",
            "packaged_logits": False,
            "generation_rehashed_original_artifacts": True,
            "generation_record": package["generation_record"],
            "statistics_executable": executable,
            "receipt_parser": {
                "name": options.receipt_verifier.name,
                "sha256": digest(options.receipt_verifier),
                "interface": "quality",
            },
            "trusted": trusted,
        },
    }


def main(arguments: list[str]) -> int:
    options = parse_options(arguments)
    (
        build_info,
        oracle_run,
        llama_run,
        native_run,
        task_body,
        task_rows,
        input_record,
        rows,
        vocab,
        trusted,
        generation,
    ) = validate_options(options)
    package = package_quality_artifacts(
        options,
        oracle_run,
        llama_run,
        native_run,
        task_body,
        task_rows,
        input_record,
        rows,
        vocab,
        generation,
    )
    quality = write_quality_files(options, package, oracle_run["model"])
    body = build_comparison_body(
        options,
        package,
        quality,
        build_info,
        oracle_run,
        llama_run,
        native_run,
        task_body,
        task_rows,
        input_record,
        oracle_run["model"],
        llama_run["model"],
        vocab,
        trusted,
    )
    write_no_replace(options.output, json.dumps(body, indent=2, sort_keys=True) + "\n")
    print(options.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, struct.error, subprocess.CalledProcessError) as error:
        raise SystemExit(f"error: {error}")
