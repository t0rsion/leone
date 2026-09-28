#!/usr/bin/env python3
"""Dispatch release evidence to trusted owner supplied validators."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


VALIDATOR_SCRIPTS = {
    "quality-stage-v1": "scripts/validate-quality-stage.py",
    "concurrent-service-v1": "scripts/study-concurrent-service.py",
    "service-metrics-v1": "scripts/study-concurrent-service.py",
    "openai-client-v1": "scripts/check-release-evidence.py",
    "batched-service-v1": "scripts/check-batched-service.py",
    "branching-service-v1": "scripts/study-branching-service.py",
}
RECEIPT_VALIDATORS = {
    "runtime-receipt-v1": "runtime",
    "quality-receipt-v1": "quality",
}
STAGE_MODES = {"export", "metal", "comparison"}
PUBLIC_CHECKERS = {"scripts/check-public-tree.sh", "scripts/check-public-tree.py"}
ORACLE_ADAPTER_PATH = "research/oracle/llama_logits.cpp"
BRANCHING_SCHEMA = "leone.branching-service.v1"
HISTORY_RESULT_SCHEMA = "leone.history-reexecution.v1"
HISTORY_EXPECTED_SCHEMA = "leone.history-reexecution-expected.v1"
HISTORY_TEMPLATE = "fixtures/qwen3-legacy-chatml.jinja"
HISTORY_PRODUCER = "scripts/produce-history-tokenization.py"
HISTORY_GENERATOR = "scripts/generate-openai-chat-template-fixtures.py"
LLAMA_PIN = "external/PINNED"
LLAMA_KINDS = ("llama.cpp", "llama_cpp")
HISTORY_NOT_COVERED = (
    "gpu_kv_contents",
    "server_execution_timestamps",
    "peer_server_execution",
    "offline_recompute_consistency",
    "loaded_library_reobservation",
)
HISTORY_IDENTITY_PINS = (
    "source_commit",
    "executable_sha256",
    "model_sha256",
    "template_config_sha256",
    "template_mode",
    "producer_sha256",
    "template_generator_sha256",
    "loaded_library_sha256",
    "special_tokens_policy_sha256",
    "vocab_size",
)
HISTORY_EXPECTED_DIGESTS = (
    "input_sha256",
    "producer_sha256",
    "template_generator_sha256",
    "executable_sha256",
    "loaded_library_sha256",
    "model_sha256",
    "template_config_sha256",
    "template_bytes_sha256",
    "special_tokens_policy_sha256",
)
HISTORY_EXPECTED_FIELDS = frozenset(
    ("schema_version", "source_commit", "template_mode", "vocab_size", "prompt_prefix",
     "special_tokens_policy", *HISTORY_EXPECTED_DIGESTS)
)
HISTORY_ORACLE_FIELDS = (
    "engine", "source_commit", "executable_sha256", "loaded_library_sha256", "gguf_sha256",
    "tokenizer_metadata_sha256", "tokenizer_metadata_hash_scheme", "template_config_sha256",
    "template_config_hash_scheme", "template_bytes_sha256", "special_tokens_policy_sha256",
    "vocab_size", "proof_adapter",
)
HISTORY_ORACLE_JOINS = (
    ("source_commit", "source_commit"),
    ("executable_sha256", "executable_sha256"),
    ("loaded_library_sha256", "loaded_library_sha256"),
    ("gguf_sha256", "model_sha256"),
    ("template_config_sha256", "template_config_sha256"),
    ("template_bytes_sha256", "template_bytes_sha256"),
    ("special_tokens_policy_sha256", "special_tokens_policy_sha256"),
    ("vocab_size", "vocab_size"),
)
HISTORY_PROOF_ADAPTER = "llama.cpp.verbose"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
HISTORY_CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "LLAMA_ARG_DEVICE": "none"}
STUDY_PATH_KEYS = ("path", "manifest_path", "receipt")
QUALITY_POLICY = {
    "name": "canonical_v2_common_oracle",
    "sample_contract": "linspace-inclusive-v1:128+task-rows",
    "release_validation": "quality-stage-v1",
}
QUALITY_PRODUCERS = {"leone": "same_executable", "llama_cpp": "peer_adapter"}
QUALITY_PATH_STATUS = "eval_path_not_served_path"
QUALITY_RECOMPUTATION_STATUS = "not_run_by_harness"
LOADED_LIBRARY_STATUS = "resolved_linkage_not_process_map"
RETIRED_QUALITY_FIELDS = frozenset(
    {"quality_tolerances", "quality_receipt", "quality_calibration_receipt"}
)
BRANCHING_ARCHIVE_INPUTS = (
    "scripts/study-branching-service.py",
    "scripts/produce-history-tokenization.py",
    "scripts/check-history-tokenization.py",
    "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/linked_libraries.py",
    "scripts/source_inputs.py",
    "fixtures/qwen3-legacy-chatml.jinja",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _relative(value: Any, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} is missing")
    path = PurePosixPath(value)
    _require(
        not path.is_absolute() and ".." not in path.parts and "\\" not in value,
        f"{label} is unsafe",
    )
    _require(path.as_posix() == value and value != ".", f"{label} is not normalized")
    return path.as_posix()


def _load(root: Path, relative: str) -> dict[str, Any]:
    path = _regular_file(root, relative, f"packaged file {relative}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"cannot read {relative}: {error}")
    _require(isinstance(value, dict), f"{relative} is not an object")
    return value


def _regular_file(root: Path, relative: str, label: str) -> Path:
    path = root / relative
    if not path.exists():
        raise ValueError(f"{label} is missing")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{label} escapes its root") from error
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        _require(not current.is_symlink(), f"{label} contains a symlink parent")
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path


def _hash(value: Any, label: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower()),
        f"{label} is not a SHA-256",
    )


def _source_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    relative = _relative(manifest.get("source_manifest"), "source manifest")
    record = _load(root, relative)
    source_commit = record.get("source_commit")
    _require(
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit.lower()),
        "source input commit is invalid",
    )
    _source_records(record)
    return record


def _source_records(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for field in ("files", "workload_files", "evidence_files"):
        values = record.get(field)
        if values is None:
            if field == "files":
                _fail("source input files are empty")
            continue
        _require(isinstance(values, dict), f"source input {field} are malformed")
        if field == "files":
            _require(values, "source input files are empty")
        for name, value in values.items():
            _validate_source_record(name, value)
            _require(name not in records, f"source input path is repeated: {name}")
            records[name] = value
    return records


def _validate_source_record(name: Any, value: Any) -> None:
    relative = _relative(name, "source input path")
    _require(isinstance(value, dict), f"source input entry is invalid: {relative}")
    _hash(value.get("sha256"), f"source input SHA-256: {relative}")
    _require(isinstance(value.get("executable"), bool), f"source input mode is invalid: {relative}")


def _check_source_file(root: Path, name: str, value: dict[str, Any]) -> None:
    _validate_source_record(name, value)
    relative = _relative(name, "source input path")
    path = _regular_file(root, relative, f"packaged source input {relative}")
    _require(
        bool(path.stat().st_mode & 0o111) == value["executable"],
        f"packaged source input mode differs: {relative}",
    )
    _require(_file_hash(path) == value["sha256"], f"packaged source input changed: {relative}")


def _check_source_file_if_present(root: Path, name: str, value: dict[str, Any]) -> bool:
    _validate_source_record(name, value)
    relative = _relative(name, "source input path")
    candidate = root / relative
    if not candidate.exists() and not candidate.is_symlink():
        return False
    _check_source_file(root, relative, value)
    return True


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_source_hashes(
    manifest: dict[str, Any], records: list[dict[str, Any]]
) -> set[str]:
    source_manifest = manifest["source_manifest"]
    record_paths = {record["path"] for record in records}
    required = set(manifest["trusted_validators"]) | PUBLIC_CHECKERS
    for record in records:
        required.update(
            dependency
            for dependency in record["dependencies"]
            if dependency != source_manifest and dependency not in record_paths
        )
    return required


def _check_dependencies(root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    destinations = {item["destination"] for item in manifest["files"]}
    for path in manifest["trusted_validators"]:
        _require(path in destinations, f"trusted validator is not declared: {path}")
        _regular_file(root, path, f"trusted validator {path}")
    for record in records:
        for dependency in record["dependencies"]:
            _require(dependency in destinations, f"record dependency is not declared: {dependency}")
            _regular_file(root, dependency, f"record dependency {dependency}")
        for artifact in record.get("artifact_files", []):
            _require(artifact in destinations, f"quality artifact is not declared: {artifact}")
            _regular_file(root, artifact, f"quality artifact {artifact}")
        _check_quality_artifact_references(root, record)


def _json_paths(value: Any, keys: tuple[str, ...] = ("path",)) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and isinstance(child, str):
                paths.append(child)
            paths.extend(_json_paths(child, keys))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_json_paths(child, keys))
    return paths


def _shader_names(value: Any) -> list[str]:
    """Return shader artifact names, which the native Metal manifest keys by `name`."""
    names: list[str] = []
    if isinstance(value, dict):
        shader = value.get("shader")
        if isinstance(shader, dict) and isinstance(shader.get("name"), str):
            names.append(shader["name"])
        for child in value.values():
            names.extend(_shader_names(child))
    elif isinstance(value, list):
        for child in value:
            names.extend(_shader_names(child))
    return names


def _artifact_candidates(value: str, bases: list[PurePosixPath]) -> set[str]:
    if PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
        return set()
    return {(base / value).as_posix() for base in bases}


def _record_artifact_references(
    root: Path,
    artifact_files: set[str],
    artifact_roots: list[PurePosixPath],
    comparison_path: PurePosixPath,
) -> set[str]:
    references: set[str] = set()
    pending = [comparison_path]
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative.as_posix() in visited:
            continue
        visited.add(relative.as_posix())
        body = _load(root, relative.as_posix())
        for value in _json_paths(body) + _shader_names(body):
            value_path = PurePosixPath(value)
            value_bases = [relative.parent]
            if len(value_path.parts) == 1:
                value_bases.extend(artifact_roots)
            for candidate in _artifact_candidates(value, value_bases):
                if candidate not in artifact_files:
                    continue
                references.add(candidate)
                if candidate.endswith(".json"):
                    pending.append(PurePosixPath(candidate))
    return references


def _check_quality_artifact_references(root: Path, record: dict[str, Any]) -> None:
    if record.get("validator") != "quality-stage-v1" or record.get("validator_mode") != "comparison":
        return
    artifact_files = set(record.get("artifact_files", []))
    artifact_roots = [PurePosixPath(item) for item in record.get("artifact_roots", [])]
    comparison_path = PurePosixPath(record["path"])
    if not _json_paths(_load(root, comparison_path.as_posix())):
        return
    references = _record_artifact_references(
        root,
        artifact_files,
        artifact_roots,
        comparison_path,
    )
    missing = sorted(artifact_files - references)
    _require(not missing, "quality artifact is unreferenced: " + ", ".join(missing))


def _environment() -> dict[str, str]:
    python_dir = str(Path(sys.executable).parent)
    return {
        "PATH": os.pathsep.join((python_dir, os.defpath)),
        "PYTHONNOUSERSITE": "1",
    }


def _trusted_receipt_tool(path: Path | None) -> Path | None:
    if path is None:
        return None
    _require(path.is_file() and not path.is_symlink(), "trusted receipt validator is not a regular file")
    for parent in path.parents:
        _require(not parent.is_symlink(), "trusted receipt validator has a symlink parent")
    return path.resolve()


def _run(command: list[str], checker_root: Path) -> None:
    try:
        subprocess.run(
            command,
            cwd=checker_root,
            env=_environment(),
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        _fail(f"trusted evidence validator failed: {command[1]}: {error}")


def _stage_expectations(
    record: dict[str, Any], source_commit: str, adapter_sha256: str, root: Path
) -> list[str]:
    if record.get("backend") == "cuda":
        _require(
            isinstance(record.get("task_manifest_sha256"), str),
            "CUDA comparison requires a trusted task manifest SHA-256",
        )
    options = [
        ("--source-commit", source_commit),
        ("--platform", record["platform"]),
        ("--target", record["target"]),
        ("--statistics-target", record["statistics_target"]),
        ("--backend", record["backend"]),
        ("--adapter-path", ORACLE_ADAPTER_PATH),
        ("--adapter-sha256", adapter_sha256),
    ]
    if record.get("native_source_commit") is not None:
        options.append(("--native-source-commit", record["native_source_commit"]))
    if record.get("backend") == "cuda" and record.get("statistics_platform") is not None:
        options.append(("--statistics-platform", record["statistics_platform"]))
    if record.get("backend") == "cuda":
        _require(
            isinstance(record.get("generation_record_sha256"), str),
            "CUDA comparison requires a trusted generation record SHA-256",
        )
        trusted_inputs = _relative(record["trusted_inputs"], "trusted CUDA inputs")
        options.append(("--trusted-inputs", str(root / trusted_inputs)))
        generation_record = _relative(record["generation_record"], "CUDA generation record")
        generation_path = _regular_file(root, generation_record, "CUDA generation record")
        _require(
            _file_hash(generation_path) == record["generation_record_sha256"],
            "CUDA generation record differs from the trusted release identity",
        )
        options.append(("--generation-record", str(generation_path)))
    for key, option in (
        ("model_family", "--model-family"),
        ("model_sha256", "--model-sha256"),
        ("metric_family", "--metric-family"),
        ("corpus_sha256", "--corpus-sha256"),
        ("oracle_model_sha256", "--oracle-model-sha256"),
        ("task_manifest_sha256", "--task-manifest-sha256"),
        ("sample_contract", "--sample-contract"),
    ):
        if key in record:
            options.append((option, record[key]))
    if record.get("backend") == "metal":
        options.append(("--sample-manifest-sha256", record["sample_manifest_sha256"]))
    arguments: list[str] = []
    for option, value in options:
        arguments.extend((option, str(value)))
    return arguments


def _stage_command(
    script: Path,
    root: Path,
    record: dict[str, Any],
    receipt_validator: Path | None,
    source_commit: str,
    adapter_sha256: str,
) -> list[str]:
    mode = record.get("validator_mode")
    _require(mode in STAGE_MODES, "quality stage validator mode is invalid")
    if mode == "comparison":
        _require(receipt_validator is not None, "comparison validation requires a receipt validator")
        comparison = _relative(record["path"], "quality comparison manifest")
        _regular_file(root, comparison, "quality comparison manifest")
        return [
            sys.executable,
            str(script),
            mode,
            str(root / comparison),
            str(receipt_validator),
            *_stage_expectations(record, source_commit, adapter_sha256, root),
        ]
    artifact_root = _relative(record.get("artifact_root"), "quality stage artifact root")
    stage = root / artifact_root
    _require(stage.is_dir(), f"quality stage artifact root is missing: {artifact_root}")
    return [sys.executable, str(script), mode, str(stage)]


def _service_command(script: Path, root: Path, path: str, source_manifest: str) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--validate-receipt",
        str(root / path),
        "--source-manifest",
        source_manifest,
        "--offline",
    ]


def _service_identity(root: Path, record: dict[str, Any], source_commit: str) -> None:
    receipt = _load(root, record["path"])
    model = receipt.get("model")
    _require(isinstance(model, dict), "service receipt model is missing")
    _require(
        model.get("sha256") == record["model_sha256"],
        f"service receipt model differs: {record['path']}",
    )
    source = receipt.get("source")
    _require(isinstance(source, dict), "service receipt source is missing")
    _require(source.get("commit") == source_commit, "service receipt source commit differs")
    _require(source.get("tracked_tree_clean") is True, "service receipt source tree is dirty")
    build = receipt.get("binaries", {}).get("leone", {}).get("build_info", {})
    value = build.get("value") if isinstance(build, dict) else None
    _require(isinstance(value, dict), "service receipt build information is missing")
    _require(value.get("source_commit") == source_commit, "service build source differs")
    _require(value.get("target") == record["target"], "service build target differs")
    _require(value.get("source_tree_dirty") is False, "service build source tree is dirty")
    _require(value.get("profile") == "release", "service build is not a release build")
    _service_runs_match_backend(receipt, record["backend"])


def _service_runs_match_backend(receipt: dict[str, Any], backend: str) -> None:
    runs = receipt.get("runs")
    _require(isinstance(runs, list) and runs, "service receipt has no runs")
    for run in runs:
        _require(isinstance(run, dict), "service receipt run is malformed")
        argv = run.get("launch_argv")
        _require(isinstance(argv, list), "service receipt launch command is missing")
        _require("--backend" in argv, "service receipt launch backend is missing")
        index = argv.index("--backend")
        _require(index + 1 < len(argv), "service receipt launch backend is incomplete")
        _require(argv[index + 1] == backend, "service receipt backend differs")


def _batching_command(
    script: Path, root: Path, record: dict[str, Any], source_manifest: str
) -> list[str]:
    for key in ("model_sha256", "binary_sha256", "quality_record", "quality_receipt"):
        _require(key in record, f"batching record has no {key}: {record['path']}")
    return [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--validate-receipt",
        str(root / record["path"]),
        "--source-manifest",
        source_manifest,
        "--offline",
        "--platform",
        record["platform"],
        "--target",
        record["target"],
        "--backend",
        record["backend"],
        "--model-sha256",
        record["model_sha256"],
        "--binary-sha256",
        record["binary_sha256"],
        "--quality-record",
        _relative(record["quality_record"], "batching quality record"),
        "--quality-receipt",
        _relative(record["quality_receipt"], "batching quality receipt"),
    ]


def _client_command(
    script: Path,
    root: Path,
    record: dict[str, Any],
    source_manifest: str,
    receipt_validator: Path | None,
) -> list[str]:
    _require(receipt_validator is not None, "client validation requires a receipt validator")
    command = [
        sys.executable,
        str(script),
        "--root",
        str(root),
        "--validate-client",
        str(root / record["path"]),
        "--source-manifest",
        source_manifest,
        "--offline",
        "--platform",
        record["platform"],
        "--target",
        record["target"],
        "--backend",
        record["backend"],
        "--trusted-receipt-validator",
        str(receipt_validator),
    ]
    for key, option, value in (
        ("binary_sha256", "--binary-sha256", lambda item: item),
        ("model_family", "--model-family", lambda item: item),
        ("model_sha256", "--model-sha256", lambda item: item),
    ):
        if key in record:
            command.extend((option, value(record[key])))
    return command


def _receipt_command(
    root: Path,
    record: dict[str, Any],
    receipt_validator: Path | None,
    source_commit: str,
    source_manifest: str,
) -> list[str]:
    kind = RECEIPT_VALIDATORS[record["validator"]]
    _require(receipt_validator is not None, f"canonical receipt validator is required: {kind}")
    command = [
        str(receipt_validator),
        "--offline",
        "--root",
        str(root),
        "--manifest",
        str(root / "release-evidence.json"),
        "--source-manifest",
        str(root / source_manifest),
        "--source-commit",
        source_commit,
        "--record",
        str(root / record["path"]),
        "--kind",
        kind,
        "--platform",
        record["platform"],
        "--target",
        record["target"],
        "--backend",
        record["backend"],
    ]
    if "model_family" in record:
        command.extend(("--model-family", record["model_family"]))
    if "model_sha256" in record:
        command.extend(("--model-sha256", record["model_sha256"]))
    if "metric_family" in record:
        command.extend(("--metric-family", record["metric_family"]))
    return command


def _command(
    root: Path,
    checker_root: Path,
    record: dict[str, Any],
    receipt_validator: Path | None,
    source_commit: str,
    source_manifest: str,
    adapter_sha256: str,
) -> list[str]:
    validator = record["validator"]
    if validator in RECEIPT_VALIDATORS:
        return _receipt_command(
            root, record, receipt_validator, source_commit, source_manifest
        )
    relative_script = VALIDATOR_SCRIPTS.get(validator)
    _require(relative_script is not None, f"canonical validator is unavailable: {validator}")
    script = checker_root / relative_script
    _require(script.is_file(), f"canonical validator is unavailable: {relative_script}")
    if validator == "quality-stage-v1":
        return _stage_command(
            script, root, record, receipt_validator, source_commit, adapter_sha256
        )
    if validator in {"concurrent-service-v1", "service-metrics-v1"}:
        return _service_command(script, root, record["path"], source_manifest)
    if validator == "batched-service-v1":
        return _batching_command(script, root, record, source_manifest)
    if validator == "branching-service-v1":
        return _branching_command(script, root, record, source_manifest)
    if validator == "openai-client-v1":
        return _client_command(script, root, record, source_manifest, receipt_validator)
    _fail(f"canonical validator is unavailable: {validator}")


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} is missing")
    return value


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _source_sha256(source_files: dict[str, dict[str, Any]], relative: str) -> str:
    _require(relative in source_files, f"source input hash is missing: {relative}")
    return source_files[relative]["sha256"]


def _engine_row(receipt: dict[str, Any], kinds: tuple[str, ...], label: str) -> dict[str, Any]:
    rows = receipt.get("engines")
    _require(isinstance(rows, list), "branching receipt engines are missing")
    chosen = [row for row in rows if isinstance(row, dict) and row.get("kind") in kinds]
    _require(len(chosen) == 1, f"branching receipt needs exactly one {label} engine")
    return chosen[0]


def _engine_provenance(row: dict[str, Any]) -> dict[str, Any]:
    provenance = _mapping(row.get("provenance"), f"branching engine provenance: {row.get('id')}")
    _require(provenance.get("status") == "observed", f"branching engine is not observed: {row.get('id')}")
    return provenance


def _engine_executable(row: dict[str, Any]) -> str:
    digest = _engine_provenance(row).get("executable_sha256")
    _hash(digest, f"branching engine executable SHA-256: {row.get('id')}")
    return digest


def _source_id_commit(source_id: Any) -> str | None:
    parts = source_id.split(":") if isinstance(source_id, str) else []
    commits = [part for part in parts if len(part) == 40 and set(part) <= set("0123456789abcdef")]
    return commits[0] if len(commits) == 1 and "true" not in parts else None


def _branching_manifest(root: Path, receipt: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    reference = _mapping(receipt.get("manifest"), "branching receipt manifest")
    path = _relative(reference.get("path"), "branching study manifest")
    _require(path in record["study_files"], "branching study manifest is not a listed study file")
    manifest = _load(root, path)
    _require(
        manifest.get("phase") == "frozen" and manifest.get("freeze_status") == "frozen",
        "branching study manifest is not frozen",
    )
    criteria = manifest.get("freeze_criteria")
    _require(
        isinstance(criteria, dict) and bool(criteria) and all(item is True for item in criteria.values()),
        "branching study manifest does not meet every freeze criterion",
    )
    return manifest


def _branching_receipt(root: Path, record: dict[str, Any], source_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _load(root, record["path"])
    _require(receipt.get("schema_version") == BRANCHING_SCHEMA, "branching receipt schema differs")
    _require(
        receipt.get("phase") == "frozen" and receipt.get("freeze_status") == "frozen",
        "branching receipt is not a frozen study",
    )
    source = _mapping(receipt.get("source"), "branching receipt source")
    _require(source.get("status") == "observed", "branching receipt source is not observed")
    _require(source.get("commit") == source_commit, "branching receipt source commit differs")
    _require(source.get("tracked_tree_clean") is True, "branching receipt source tree is dirty")
    artifacts = _mapping(receipt.get("artifacts"), "branching receipt artifacts")
    model = _mapping(artifacts.get("model"), "branching receipt model")
    _require(
        model.get("sha256") == record["model_sha256"] and model.get("status") == "observed",
        "branching receipt model differs",
    )
    return receipt, _branching_manifest(root, receipt, record)


def _branching_engines(
    receipt: dict[str, Any], record: dict[str, Any], source_commit: str
) -> dict[str, dict[str, Any]]:
    leone = _engine_row(receipt, ("leone",), "Leone")
    peer = _engine_row(receipt, LLAMA_KINDS, "llama.cpp")
    for row in (leone, peer):
        _require(row.get("backend") == record["backend"], f"branching receipt backend differs: {row.get('id')}")
    method = _mapping(leone.get("branch_method"), "Leone branch method")
    _require(
        method.get("name") == "leone_fork_session" and method.get("status") != "unsupported",
        "Leone branch method is not supported",
    )
    _require(
        _engine_executable(leone) == record["binary_sha256"],
        "branching receipt Leone binary differs from the release binary",
    )
    build = _mapping(_engine_provenance(leone).get("build_info"), "Leone build information")
    _require(
        _source_id_commit(build.get("source_id")) == source_commit,
        "branching receipt Leone build source differs from the final source",
    )
    _engine_executable(peer)
    return {"leone": leone, "peer": peer}


def _run_identity(run: Any) -> dict[str, Any]:
    run = _mapping(run, "branching run")
    start = _mapping(_mapping(run.get("running_identity"), "branching run identity").get("start"), "branching run start")
    return _mapping(start.get("identity"), "branching run process identity")


def _branching_runs(receipt: dict[str, Any], engines: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    executables = {row["id"]: _engine_executable(row) for row in engines.values()}
    runs = receipt.get("runs")
    _require(isinstance(runs, list) and bool(runs), "branching receipt has no runs")
    for run in runs:
        identity = _run_identity(run)
        _require(
            identity.get("executable_sha256") == executables.get(run.get("engine")),
            "branching run executable differs from its engine",
        )
        _require(identity.get("model_sha256") == record["model_sha256"], "branching run model differs")


def _branching_gates(receipt: dict[str, Any], leone: dict[str, Any]) -> None:
    evaluation = _mapping(receipt.get("evaluation"), "branching receipt evaluation")
    results = evaluation.get("threshold_results")
    _require(isinstance(results, list), "branching receipt threshold results are missing")
    gates = [item for item in results if isinstance(item, dict) and item.get("engine") == leone["id"]]
    _require(bool(gates), "branching receipt has no Leone gate")
    _require(
        all(item.get("status") == "pass" and item.get("scope_matches") is True for item in gates),
        "a Leone branching gate did not pass",
    )


def _branching_reference(manifest: dict[str, Any], key: str) -> dict[str, Any]:
    evaluation = _mapping(manifest.get("evaluation"), "branching study evaluation")
    return _mapping(evaluation.get(key), f"branching study {key}")


def _branching_policy(manifest: dict[str, Any]) -> None:
    evaluation = _mapping(manifest.get("evaluation"), "branching study evaluation")
    _require(
        not RETIRED_QUALITY_FIELDS & set(evaluation),
        "branching study carries a retired quality calibration field",
    )
    _require(
        _branching_reference(manifest, "quality_policy") == QUALITY_POLICY,
        "branching study quality policy is not canonical_v2_common_oracle",
    )
    labels = {}
    for engine in manifest.get("engines") or ():
        labels[engine.get("quality_label")] = engine.get("quality_producer")
    _require(labels == QUALITY_PRODUCERS, "branching study quality producers differ from the policy")


def _branching_quality_link(
    root: Path, manifest: dict[str, Any], record: dict[str, Any], linked: dict[str, Any]
) -> None:
    """Bind the frozen study to the exact quality-stage-v1 record the release validates."""
    _require(
        linked.get("validator") == "quality-stage-v1" and linked.get("validator_mode") == "comparison",
        "branching quality record is not validated by quality-stage-v1 comparison",
    )
    _require(
        linked.get("backend") == record["backend"]
        and linked.get("model_family") == record["model_family"]
        and linked.get("model_sha256") == record["model_sha256"],
        "branching quality record backend or model differs from the study",
    )
    reference = _branching_reference(manifest, "quality_record")
    _require(reference.get("path") == linked["path"], "branching study names another quality record")
    packaged = _file_hash(_regular_file(root, linked["path"], "branching quality record"))
    _require(reference.get("sha256") == packaged, "branching study quality record digest differs from the package")
    _require(
        linked.get("sha256", packaged) == packaged,
        "branching quality record digest differs from the release record",
    )
    _branching_sidecars(root, linked, record)


def _branching_sidecars(root: Path, linked: dict[str, Any], record: dict[str, Any]) -> None:
    """Require both quality sidecars of the linked record to be packaged dependencies."""
    body = _load(root, linked["path"])
    rows = _mapping(body.get("quality"), "branching quality record rows")
    for label in QUALITY_PRODUCERS:
        entry = _mapping(rows.get(label), f"branching quality row: {label}")
        sidecar = PurePosixPath(linked["path"]).parent / _relative(entry.get("path"), "quality sidecar")
        _require(
            sidecar.as_posix() in record["dependencies"],
            f"branching quality sidecar is not a dependency: {sidecar.as_posix()}",
        )


def _branching_binding(receipt: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Require the receipt to state the policy, its scope limits, and both quality rows."""
    binding = _mapping(
        _mapping(receipt.get("evaluation"), "branching receipt evaluation").get("quality_binding"),
        "branching receipt quality binding",
    )
    engines = {
        engine["quality_label"]: {
            "quality_label": engine["quality_label"],
            "quality_producer": engine["quality_producer"],
            **(
                {"loaded_library_status": LOADED_LIBRARY_STATUS}
                if engine["quality_producer"] == "peer_adapter"
                else {}
            ),
        }
        for engine in manifest["engines"]
    }
    expected = {
        "quality_policy": QUALITY_POLICY["name"],
        "quality_path_status": QUALITY_PATH_STATUS,
        "quality_recomputation_status": QUALITY_RECOMPUTATION_STATUS,
        "engines": engines,
    }
    _require(binding == expected, "branching receipt quality binding differs from the policy")



def _study_references(root: Path, record: dict[str, Any]) -> set[str]:
    allowed, seen, pending = set(record["study_files"]), set(), [record["path"]]
    while pending:
        relative = pending.pop()
        try:
            body = _load(root, relative) if relative.endswith(".json") else {}
        except ValueError:
            continue
        for value in _json_paths(body, STUDY_PATH_KEYS):
            if value in allowed and value not in seen:
                seen.add(value)
                pending.append(value)
    return seen


def _branching_closure(root: Path, record: dict[str, Any]) -> None:
    study = record["study_files"]
    _require(len(set(study)) == len(study), "branching study files repeat")
    _require(set(study) <= set(record["dependencies"]), "branching study file is not a dependency")
    unreferenced = sorted(set(study) - _study_references(root, record))
    _require(not unreferenced, "branching study file is unreferenced: " + ", ".join(unreferenced))


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and DIGEST_PATTERN.fullmatch(value) is not None


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _history_expected_shape(expected: dict[str, Any]) -> None:
    """Apply the standalone checker's expected-file contract: exact fields, typed values."""
    _require(set(expected) == HISTORY_EXPECTED_FIELDS, "history expected fields differ from the checker contract")
    commit = expected["source_commit"]
    checks = {
        "source_commit": isinstance(commit, str) and COMMIT_PATTERN.fullmatch(commit) is not None,
        "prompt_prefix": isinstance(expected["prompt_prefix"], str),
        "special_tokens_policy": isinstance(expected["special_tokens_policy"], dict),
        "vocab_size": _is_count(expected["vocab_size"]),
        **{name: _is_digest(expected[name]) for name in HISTORY_EXPECTED_DIGESTS},
    }
    invalid = [name for name, valid in checks.items() if not valid]
    _require(not invalid, f"history expected {invalid[0] if invalid else ''} is invalid")


def _history_expected(expected: dict[str, Any], pins: dict[str, str]) -> None:
    _require(expected.get("schema_version") == HISTORY_EXPECTED_SCHEMA, "history expected schema differs")
    _require(expected.get("template_mode") == "legacy", "history expected template mode is not legacy")
    _history_expected_shape(expected)
    for name, value in pins.items():
        _require(expected.get(name) == value, f"history expected {name} differs from the packaged pin")
    _require(
        expected.get("special_tokens_policy_sha256") == _canonical_sha256(expected.get("special_tokens_policy")),
        "history expected special-token policy hash differs",
    )


def _history_declaration(manifest: dict[str, Any], expected: dict[str, Any], pins: dict[str, str]) -> None:
    engines = manifest.get("engines")
    _require(isinstance(engines, list), "branching study engines are missing")
    peers = [item for item in engines if isinstance(item, dict) and item.get("kind") in LLAMA_KINDS]
    _require(len(peers) == 1, "branching study needs exactly one llama.cpp engine")
    declared = _mapping(peers[0].get("history_tokenization"), "branching study history declaration")
    _require(declared.get("template_file") == HISTORY_TEMPLATE, "branching study template file differs")
    template = pins["template_bytes_sha256"]
    _require(
        declared.get("template_bytes_sha256", template) == template,
        "branching study template bytes differ from the packaged template",
    )
    policy = expected["special_tokens_policy_sha256"]
    _require(
        declared.get("special_tokens_policy_sha256", policy) == policy,
        "branching study special-token policy differs from the expected pins",
    )


def _history_pins(
    root: Path,
    record: dict[str, Any],
    receipt_sha256: str,
    peer: dict[str, Any],
    source_files: dict[str, dict[str, Any]],
) -> dict[str, str]:
    template = _source_sha256(source_files, HISTORY_TEMPLATE)
    pin = _regular_file(root, LLAMA_PIN, "llama.cpp pin").read_text(encoding="utf-8").strip()
    return {
        "input_sha256": receipt_sha256,
        "source_commit": pin,
        "producer_sha256": _source_sha256(source_files, HISTORY_PRODUCER),
        "template_generator_sha256": _source_sha256(source_files, HISTORY_GENERATOR),
        "template_config_sha256": template,
        "template_bytes_sha256": template,
        "model_sha256": record["model_sha256"],
        "executable_sha256": _engine_executable(peer),
    }


def _history_result(
    result: dict[str, Any], expected: dict[str, Any], receipt: bytes, expected_bytes: bytes
) -> None:
    _require(
        result.get("schema_version") == HISTORY_RESULT_SCHEMA
        and result.get("check") == "fresh_tokenizer_reexecution",
        "history result is not a fresh tokenizer reexecution",
    )
    _require(
        result.get("status") == "verified" and result.get("reason") is None and result.get("failure") is None,
        "history reexecution did not verify",
    )
    _require(result.get("offline_recompute") == "not_evaluated", "history result claims an offline recompute")
    _require(
        set(HISTORY_NOT_COVERED) <= set(result.get("not_covered") or ()),
        "history result omits its stated limits",
    )
    source = _mapping(result.get("input"), "history result input")
    _require(
        source.get("format") == "receipt"
        and source.get("sha256") == hashlib.sha256(receipt).hexdigest()
        and source.get("bytes") == len(receipt),
        "history result input is not the packaged branching receipt",
    )
    _require(
        result.get("expected_sha256") == hashlib.sha256(expected_bytes).hexdigest(),
        "history result does not bind the packaged expected pins",
    )
    _history_identity(_mapping(result.get("identity"), "history result identity"), expected)


def _history_identity(identity: dict[str, Any], expected: dict[str, Any]) -> None:
    for name in HISTORY_IDENTITY_PINS:
        _require(identity.get(name) == expected.get(name), f"history result {name} differs from the expected pins")
    _require(
        identity.get("loaded_library_status") == "pinned_by_caller_not_reobserved",
        "history result loaded library status differs",
    )
    prefix = hashlib.sha256(expected["prompt_prefix"].encode("utf-8")).hexdigest()
    _require(identity.get("prompt_prefix_sha256") == prefix, "history result prompt prefix differs from the expected pins")


def _retained_records(receipt: dict[str, Any], peer: dict[str, Any]) -> list[tuple[int, int, dict[str, Any]]]:
    rows = []
    for run_index, run in enumerate(receipt["runs"]):
        if run.get("engine") != peer["id"]:
            continue
        for branch_index, branch in enumerate(run.get("branches") or ()):
            reuse = _mapping(_mapping(branch, "branching branch").get("history_reuse"), "branch history reuse")
            rows.append((run_index, branch_index, _mapping(reuse.get("tokenization"), "branch history tokenization")))
    return rows


def _retained_branches(receipt: dict[str, Any], peer: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"run_index": run, "branch_index": branch, "retained_record_sha256": _canonical_sha256(record)}
        for run, branch, record in _retained_records(receipt, peer)
    ]


def _history_oracles(
    result_oracle: Any, records: list[tuple[int, int, dict[str, Any]]], expected: dict[str, Any]
) -> None:
    """Join the result's oracle identity and the expected pins to every retained oracle."""
    identity = _mapping(result_oracle, "history result oracle identity")
    _require(set(identity) == set(HISTORY_ORACLE_FIELDS), "history result oracle identity fields differ")
    for run, branch, record in records:
        where = f"run {run} branch {branch}"
        oracle = _mapping(record.get("oracle"), f"retained oracle: {where}")
        _require(
            record.get("status") == "observed" and oracle.get("proof_adapter") == HISTORY_PROOF_ADAPTER,
            f"retained oracle is not a llama.cpp verbose proof: {where}",
        )
        _require(
            {name: oracle.get(name) for name in HISTORY_ORACLE_FIELDS} == identity,
            f"history result oracle identity differs from the retained oracle: {where}",
        )
        _require(
            oracle.get("special_tokens_policy") == expected["special_tokens_policy"],
            f"retained oracle special-token policy differs from the expected pins: {where}",
        )
        for name, pin in HISTORY_ORACLE_JOINS:
            _require(oracle.get(name) == expected[pin], f"retained oracle {name} differs from the expected {pin}: {where}")


def _history_branches(
    result: dict[str, Any], receipt: dict[str, Any], peer: dict[str, Any], expected: dict[str, Any]
) -> None:
    retained = _retained_branches(receipt, peer)
    checked = result.get("checked_branches")
    _require(bool(retained), "branching receipt retains no llama.cpp branch")
    _require(isinstance(checked, list), "history result checked branches are missing")
    _require(
        result.get("checked_branch_count") == len(retained),
        "history result checked branch count differs from the receipt",
    )
    fields = ("run_index", "branch_index", "retained_record_sha256")
    rows = [{name: item.get(name) for name in fields} for item in checked if isinstance(item, dict)]
    _require(rows == retained, "history result branches differ from the retained receipt records")
    identity = _mapping(result.get("identity"), "history result identity")
    _history_oracles(identity.get("oracle"), _retained_records(receipt, peer), expected)
    _history_launches(result.get("fresh_server_launches"), len(retained))


def _history_launches(launches: Any, count: int) -> None:
    _require(
        isinstance(launches, list) and len(launches) == count,
        "history result has no fresh server launch for each branch",
    )
    for launch in launches:
        launch = _mapping(launch, "history fresh server launch")
        argv = launch.get("argv")
        _require(isinstance(argv, list), "history fresh server launch has no command")
        _require(("--n-gpu-layers", "0") in zip(argv, argv[1:]), "history fresh server launch may use a GPU")
        _require(launch.get("env") == HISTORY_CPU_ENV, "history fresh server launch does not disable the GPU")


def _branching_history(
    root: Path,
    record: dict[str, Any],
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    peer: dict[str, Any],
    source_files: dict[str, dict[str, Any]],
) -> None:
    receipt_bytes = _regular_file(root, record["path"], "branching receipt").read_bytes()
    expected_bytes = _regular_file(root, record["history_expected"], "history expected pins").read_bytes()
    expected = _load(root, record["history_expected"])
    pins = _history_pins(root, record, hashlib.sha256(receipt_bytes).hexdigest(), peer, source_files)
    _history_expected(expected, pins)
    _history_declaration(manifest, expected, pins)
    result = _load(root, record["history_result"])
    _history_result(result, expected, receipt_bytes, expected_bytes)
    _history_branches(result, receipt, peer, expected)


def _branching_identity(
    root: Path,
    record: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    source_commit: str,
    source_files: dict[str, dict[str, Any]],
) -> None:
    """Bind one frozen branching study to its release identity and history reexecution.

    The checks read the archived receipt, the frozen manifest, the linked quality
    record, and the history reexecution result. An offline archive cannot rerun
    a model, a server, or the tokenizer. The history result is an unsigned record
    of a run at collection. These checks show that it names this receipt and
    these pins, not that the run happened. The quality record is validated by
    its own quality-stage-v1 record. The study harness only binds it.
    """
    _hash(record.get("binary_sha256"), f"branching release binary SHA-256: {record['path']}")
    _require(record.get("quality_record") in by_path, f"branching quality record is missing: {record['path']}")
    receipt, manifest = _branching_receipt(root, record, source_commit)
    engines = _branching_engines(receipt, record, source_commit)
    _branching_runs(receipt, engines, record)
    _branching_gates(receipt, engines["leone"])
    _branching_policy(manifest)
    _branching_binding(receipt, manifest)
    _branching_quality_link(root, manifest, record, by_path[record["quality_record"]])
    _branching_closure(root, record)
    _branching_history(root, record, receipt, manifest, engines["peer"], source_files)


def _branching_command(script: Path, root: Path, record: dict[str, Any], source_manifest: str) -> list[str]:
    """Run the harness offline. It always checks the source manifest, in archive scope."""
    return [
        *_service_command(script, root, record["path"], source_manifest),
        "--source-scope",
        "archive",
    ]


def _require_quality_ran(records: list[dict[str, Any]], executed: set[str]) -> None:
    for record in records:
        if record.get("validator") == "branching-service-v1":
            _require(
                record["quality_record"] in executed,
                f"canonical quality validation did not run on {record['quality_record']}",
            )



def _record_identity(
    root: Path,
    record: dict[str, Any],
    by_path: dict[str, dict[str, Any]],
    source_commit: str,
    source_files: dict[str, dict[str, Any]],
) -> None:
    if record["validator"] in {"concurrent-service-v1", "service-metrics-v1"}:
        _service_identity(root, record, source_commit)
    if record["validator"] == "branching-service-v1":
        _branching_identity(root, record, by_path, source_commit, source_files)


def validate_records(
    root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    checker_root: Path,
    receipt_validator: Path | None = None,
) -> None:
    """Run canonical validators for every complete evidence record."""
    receipt_validator = _trusted_receipt_tool(receipt_validator)
    source = _source_manifest(root, manifest)
    source_files = _source_records(source)
    source_manifest = _relative(manifest["source_manifest"], "source manifest")
    adapter = source_files.get(ORACLE_ADAPTER_PATH)
    _require(isinstance(adapter, dict), f"source input hash is missing: {ORACLE_ADAPTER_PATH}")
    _check_source_file(root, ORACLE_ADAPTER_PATH, adapter)
    adapter_sha256 = adapter["sha256"]
    checked = {ORACLE_ADAPTER_PATH}
    for path, value in source_files.items():
        if path not in checked:
            if _check_source_file_if_present(root, path, value):
                checked.add(path)
    for path in _required_source_hashes(manifest, records):
        _require(path in source_files, f"source input hash is missing: {path}")
        if path not in checked:
            _check_source_file(root, path, source_files[path])
    _check_dependencies(root, manifest, records)
    by_path = {record["path"]: record for record in records}
    executed: set[str] = set()
    for record in records:
        _record_identity(root, record, by_path, source["source_commit"], source_files)
        _run(
            _command(
                root,
                checker_root,
                record,
                receipt_validator,
                source["source_commit"],
                source_manifest,
                adapter_sha256,
            ),
            checker_root,
        )
        executed.add(record["path"])
    _require_quality_ran(records, executed)
