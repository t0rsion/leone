"""Load and stage an exact release evidence manifest."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any


SCHEMA = "leone.release-evidence.v1"
HASH_LENGTH = 64
RELEASE_LINE = re.compile(r"^v[0-9]+\.[0-9]+$")
REQUIRED_BACKENDS = (
    ("linux-x86_64", "x86_64-unknown-linux-gnu", "cuda"),
    ("darwin-arm64", "aarch64-apple-darwin", "metal"),
)
ALLOWED_IDENTITIES = frozenset(REQUIRED_BACKENDS)
V04_REQUIRED_MODEL_FAMILIES = ("qwen3", "llama")
V04_REQUIRED_QUALITY_METRIC = "kld"
V04_STATISTICS_TARGET = "x86_64-unknown-linux-gnu"
V04_STATISTICS_PLATFORM = "linux-x86_64"
V04_CUDA_GENERATION_ARTIFACT = "generation-record.json"
V04_METAL_STAGE_SUFFIX = "-metal-stage"
V04_METAL_NATIVE_ARTIFACTS = ("leone.metal", "leone-doctor.txt", "leone-eval.stdout")
V04_SUBJECT_MODEL_SHA256 = {
    "qwen3": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    "llama": "1d0e9419ec4e12aef73ccf4ffd122703e94c48344a96bc7c5f0f2772c2152ce3",
}
V04_QUALITY_ADAPTER_PATH = "research/oracle/llama_logits.cpp"
V04_CORPUS_PATH = "corpus/quality-long-v3.txt"
V04_CORPUS_SHA256 = "a4f81b97182b0fb2f48c1e58c14e9ffee8f8e4472aa2742ecf39dd84ac8d95b8"
V04_ORACLE_MODEL_SHA256 = {
    "qwen3": "5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30",
    "llama": "1f33ad43d2b85b908ff06fe7002b69806a57359b9b2617ca27d7bdea428ae146",
}
V04_SAMPLE_CONTRACT = "linspace-inclusive-v1:128+task-rows"
V04_QUALITY_SAMPLE_F32_NAMES = frozenset({"oracle.f32", "llama_cpp.f32", "leone.f32"})
V04_VALIDATORS = {
    "runtime-receipt-v1",
    "quality-receipt-v1",
    "quality-stage-v1",
    "concurrent-service-v1",
    "service-metrics-v1",
    "openai-client-v1",
    "batched-service-v1",
    "branching-service-v1",
}
V04_EXTERNAL_VALIDATORS = {"runtime-receipt-v1", "quality-receipt-v1"}
V04_VALIDATOR_SCRIPTS = {
    "quality-stage-v1": "scripts/validate-quality-stage.py",
    "concurrent-service-v1": "scripts/study-concurrent-service.py",
    "service-metrics-v1": "scripts/study-concurrent-service.py",
    "openai-client-v1": "scripts/check-release-evidence.py",
    "batched-service-v1": "scripts/check-batched-service.py",
    "branching-service-v1": "scripts/study-branching-service.py",
}
V04_ROLES = {
    "runtime", "quality", "oracle", "service", "metrics", "client", "batching", "branching",
}
V04_ROLE_VALIDATORS = {
    "runtime": {"runtime-receipt-v1"},
    "quality": {"quality-receipt-v1", "quality-stage-v1"},
    "oracle": {"quality-stage-v1"},
    "service": {"concurrent-service-v1"},
    "metrics": {"service-metrics-v1"},
    "client": {"openai-client-v1"},
    "batching": {"batched-service-v1"},
    "branching": {"branching-service-v1"},
}
V04_BACKEND_ROLES = {
    "cuda": {"runtime", "quality", "service", "batching", "branching"},
    "metal": {"runtime", "quality", "service", "branching"},
}
V04_BATCHING_PATH = "receipts/v04-linux-cuda-batched-service.json"
V04_BATCHING_SCHEMA = "leone.batched-service-study.v1"
V04_BATCHING_QUALITY_RECORD = "receipts/v04-linux-cuda-quality-comparison-qwen3.json"
V04_BATCHING_QUALITY_RECEIPT = (
    "receipts/v04-linux-cuda-quality-comparison-qwen3-leone-quality.json"
)
V04_BATCHING_DEPENDENCIES = frozenset({
    "receipts/source-inputs-v04.json",
    "scripts/check-batched-service.py",
    "scripts/render-release-evidence.sh",
    "scripts/source_inputs.py",
    "scripts/study-batched-service.sh",
    "scripts/study-live-server.sh",
    "plans/qwen3-8b-sm89.json",
    V04_BATCHING_QUALITY_RECORD,
    V04_BATCHING_QUALITY_RECEIPT,
})
V04_BRANCHING_SCHEMA = "leone.branching-service.v1"
V04_BRANCHING = {
    "receipts/v04-linux-cuda-branching-service.json": (
        "cuda",
        "receipts/v04-linux-cuda-quality-comparison-qwen3.json",
    ),
    "receipts/v04-darwin-metal-branching-service.json": (
        "metal",
        "receipts/v04-darwin-metal-quality-comparison-qwen3.json",
    ),
}
V04_QUALITY_SIDECARS = {
    "cuda": ("leone-quality", "llama-cpp-quality"),
    "metal": ("leone-quality", "llama-quality"),
}
V04_BRANCHING_DEPENDENCIES = frozenset({
    "receipts/source-inputs-v04.json",
    "external/PINNED",
    "fixtures/qwen3-legacy-chatml.jinja",
    "research/oracle/llama_logits.cpp",
    "scripts/check-history-tokenization.py",
    "scripts/freeze-branching-manifest.py",
    "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/linked_libraries.py",
    "scripts/produce-history-tokenization.py",
    "scripts/source_inputs.py",
    "scripts/study-branching-service.py",
    "scripts/fetch-llama-cpp.sh",
    "scripts/study-branching-service.sh",
})
V04_TRUSTED_FILES = {
    "scripts/study-branching-service.py",
    "scripts/check-batched-service.py",
    "scripts/check-release-evidence.py",
    "scripts/release_evidence_manifest.py",
    "scripts/release_evidence_validators.py",
    "scripts/source_inputs.py",
    "scripts/study-concurrent-service.py",
    "scripts/validate-quality-stage.py",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _relative(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        _fail(f"{label} is unsafe: {value}")
    if path.as_posix() != value or value == ".":
        _fail(f"{label} is not normalized: {value}")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != HASH_LENGTH:
        _fail(f"{label} is not a SHA-256")
    if any(character not in "0123456789abcdef" for character in value.lower()):
        _fail(f"{label} is not a SHA-256")
    return value.lower()


def load(path: Path) -> dict[str, Any]:
    """Read one release evidence manifest as a JSON object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evidence manifest {path}: {error}") from error
    if not isinstance(value, dict):
        _fail("evidence manifest must be an object")
    return value


def _file_entries(manifest: dict[str, Any]) -> list[dict[str, str]]:
    raw = manifest.get("files")
    if not isinstance(raw, list) or not raw:
        _fail("evidence manifest files must be a non-empty list")
    entries: list[dict[str, str]] = []
    destinations: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            _fail(f"evidence file {index} is not an object")
        source = _relative(item.get("source"), f"evidence file {index} source")
        destination = _relative(item.get("destination"), f"evidence file {index} destination")
        if destination in destinations:
            _fail(f"evidence manifest repeats destination {destination}")
        destinations.add(destination)
        entries.append({"source": source, "destination": destination})
    return entries


def _backend_records(backend: Any, index: int) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(backend, dict):
        _fail(f"backend requirement {index} is not an object")
    backend_id = backend.get("id")
    if not isinstance(backend_id, str) or not backend_id:
        _fail(f"backend requirement {index} has no id")
    _backend_identity(backend, index)
    raw_records = backend.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        _fail(f"backend {backend_id} has no records")
    return backend_id, [_record(backend, record, record_index) for record_index, record in enumerate(raw_records)]


def _record_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_backends = manifest.get("backend_requirements")
    if not isinstance(raw_backends, list) or not raw_backends:
        _fail("evidence manifest backend_requirements must be a non-empty list")
    records: list[dict[str, Any]] = []
    backend_ids: set[str] = set()
    for index, backend in enumerate(raw_backends):
        backend_id, backend_records = _backend_records(backend, index)
        if backend_id in backend_ids:
            _fail(f"evidence manifest repeats backend {backend_id}")
        backend_ids.add(backend_id)
        records.extend(backend_records)
    if manifest.get("release_line") == "v0.4":
        _require_v04_backends(manifest, backend_ids)
        records.extend(_shared_records(manifest))
    return records


def _backend_identity(backend: dict[str, Any], index: int) -> tuple[str, str, str]:
    values = tuple(backend.get(key) for key in ("platform", "target", "backend"))
    if not all(isinstance(value, str) and value for value in values):
        _fail(f"backend requirement {index} has an incomplete identity")
    return values  # type: ignore[return-value]


def _record(
    backend: dict[str, Any], record: Any, record_index: int
) -> dict[str, Any]:
    if not isinstance(record, dict):
        _fail(f"backend {backend['id']} record {record_index} is not an object")
    path = _relative(record.get("path"), f"backend {backend['id']} record {record_index}")
    schema = record.get("schema_version")
    if not isinstance(schema, (str, int)) or isinstance(schema, bool):
        _fail(f"backend {backend['id']} record {record_index} has no schema_version")
    result = {
        "path": path,
        "schema_version": schema,
        "backend": backend["backend"],
        "platform": backend["platform"],
        "target": backend["target"],
        "_scope": "backend",
    }
    if "sha256" in record:
        result["sha256"] = _digest(record["sha256"], f"backend {backend['id']} record {record_index}")
    _record_contract(record, result, f"backend {backend['id']} record {record_index}")
    return result


def _record_contract(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    for key in ("role", "validator"):
        if key in record:
            value = _optional_text(record[key])
            if value is not None:
                result[key] = value
    _copy_dependencies(record, result, label)
    _copy_model_family(record, result, label)
    _copy_model_sha256(record, result, label)
    _copy_source_commit(record, result, label)
    _copy_statistics_platform(record, result, label)
    _copy_statistics_target(record, result, label)
    _copy_metric_family(record, result, label)
    _copy_quality_contract(record, result, label)
    _copy_validator_artifacts(record, result, label)
    _copy_binary_contract(record, result, label)
    _copy_quality_record(record, result, label)
    _copy_quality_receipt(record, result, label)
    _copy_link_contract(record, result, label)


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _copy_dependencies(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "dependencies" not in record:
        return
    dependencies = record["dependencies"]
    if not isinstance(dependencies, list) or not dependencies:
        _fail(f"{label} dependencies must be a non-empty list")
    result["dependencies"] = [_relative(item, f"{label} dependency") for item in dependencies]


def _copy_model_family(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "model_family" not in record:
        return
    family = record["model_family"]
    if not isinstance(family, str) or not family:
        _fail(f"{label} model_family must be a non-empty string")
    result["model_family"] = family


def _copy_model_sha256(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "model_sha256" not in record:
        return
    result["model_sha256"] = _digest(record["model_sha256"], f"{label} model SHA-256")


def _copy_source_commit(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "native_source_commit" not in record:
        return
    value = record["native_source_commit"]
    if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value.lower()):
        _fail(f"{label} native_source_commit is not a Git object ID")
    result["native_source_commit"] = value.lower()


def _copy_statistics_target(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "statistics_target" not in record:
        return
    value = record["statistics_target"]
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        _fail(f"{label} statistics_target must be a non-empty target")
    result["statistics_target"] = value


def _copy_statistics_platform(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "statistics_platform" not in record:
        return
    value = record["statistics_platform"]
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        _fail(f"{label} statistics_platform must be a non-empty platform")
    result["statistics_platform"] = value


def _copy_metric_family(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "metric_family" not in record:
        return
    metric = record["metric_family"]
    if not isinstance(metric, str) or not metric:
        _fail(f"{label} metric_family must be a non-empty string")
    result["metric_family"] = metric


def _copy_quality_contract(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    for key in (
        "corpus_path",
        "corpus_sha256",
        "oracle_model_sha256",
        "task_manifest_sha256",
        "sample_manifest_sha256",
        "generation_record_sha256",
        "sample_contract",
    ):
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, str) or not value:
            _fail(f"{label} {key} must be a non-empty string")
        if key.endswith("sha256"):
            value = _digest(value, f"{label} {key}")
        result[key] = value
    if "sample_requested_rows" in record:
        value = record["sample_requested_rows"]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            _fail(f"{label} sample_requested_rows must be positive")
        result["sample_requested_rows"] = value


def _copy_validator_artifacts(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    for key in ("validator_mode", "artifact_root", "trusted_inputs", "generation_record"):
        if key not in record:
            continue
        result[key] = _relative(record[key], f"{label} {key}")
    _copy_artifact_list(record, result, label, "artifact_roots", "artifact root")
    _copy_artifact_list(record, result, label, "artifact_files", "artifact file")


def _copy_artifact_list(
    record: dict[str, Any],
    result: dict[str, Any],
    label: str,
    key: str,
    item_label: str,
) -> None:
    if key not in record:
        return
    values = record[key]
    if not isinstance(values, list) or not values:
        _fail(f"{label} {key} must be a non-empty list")
    result[key] = [_relative(item, f"{label} {item_label}") for item in values]


def _copy_binary_contract(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "binary_path" in record:
        result["binary_path"] = _relative(record["binary_path"], f"{label} binary_path")
    if "binary_sha256" in record:
        result["binary_sha256"] = _digest(record["binary_sha256"], f"{label} binary SHA-256")


def _copy_link_contract(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    for key in ("history_expected", "history_result"):
        if key in record:
            result[key] = _relative(record[key], f"{label} {key}")
    if "study_files" in record:
        files = record["study_files"]
        if not isinstance(files, list) or not files:
            _fail(f"{label} study_files must be a non-empty list")
        result["study_files"] = [_relative(item, f"{label} study file") for item in files]


def _copy_quality_record(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "quality_record" in record:
        result["quality_record"] = _relative(record["quality_record"], f"{label} quality_record")


def _copy_quality_receipt(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    if "quality_receipt" in record:
        result["quality_receipt"] = _relative(record["quality_receipt"], f"{label} quality_receipt")


def _require_v04_backends(manifest: dict[str, Any], backend_ids: set[str]) -> None:
    expected = {f"{platform}-{backend}" for platform, _, backend in REQUIRED_BACKENDS}
    identities = {
        (backend.get("platform"), backend.get("target"), backend.get("backend"))
        for backend in manifest["backend_requirements"]
    }
    if backend_ids != expected or identities != set(REQUIRED_BACKENDS):
        _fail("v0.4 evidence must declare exactly Linux CUDA and Darwin Metal backends")


def _shared_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_records = manifest.get("shared_records")
    if not isinstance(raw_records, list) or not raw_records:
        _fail("v0.4 evidence must declare shared records")
    records: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, record in enumerate(raw_records):
        result = _shared_record(record, index)
        if result["path"] in paths:
            _fail(f"evidence manifest repeats shared record {result['path']}")
        paths.add(result["path"])
        records.append(result)
    return records


def _shared_record(record: Any, index: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        _fail(f"shared record {index} is not an object")
    label = f"shared record {index}"
    path = _relative(record.get("path"), label)
    schema = record.get("schema_version")
    if not isinstance(schema, (str, int)) or isinstance(schema, bool):
        _fail(f"{label} has no schema_version")
    result = {"path": path, "schema_version": schema, "_scope": "shared"}
    _copy_shared_identity(record, result, label)
    if "sha256" in record:
        result["sha256"] = _digest(record["sha256"], label)
    _record_contract(record, result, label)
    return result


def _copy_shared_identity(record: dict[str, Any], result: dict[str, Any], label: str) -> None:
    for key in ("platform", "target", "backend"):
        if key not in record:
            continue
        value = record[key]
        if not isinstance(value, str) or not value:
            _fail(f"{label} has an invalid {key}")
        result[key] = value


def _validate_header(manifest: dict[str, Any]) -> str:
    if manifest.get("schema_version") != SCHEMA:
        _fail("evidence manifest has an unknown schema")
    release_line = manifest.get("release_line")
    if not isinstance(release_line, str) or not RELEASE_LINE.fullmatch(release_line):
        _fail("evidence manifest has an invalid release_line")
    if manifest.get("package_kind") != "evidence":
        _fail("evidence manifest has the wrong package kind")
    return release_line


def _validate_status(manifest: dict[str, Any], require_complete: bool) -> None:
    status = manifest.get("status")
    if status not in {"planned", "complete"}:
        _fail("evidence manifest has an invalid status")
    if require_complete and status != "complete":
        _fail(f"evidence manifest is not complete: {status}")


def _validate_record_destinations(files: list[dict[str, str]], records: list[dict[str, Any]]) -> None:
    destinations = {entry["destination"] for entry in files}
    record_paths = [record["path"] for record in records]
    if len(record_paths) != len(set(record_paths)):
        _fail("evidence manifest repeats a backend record path")
    missing = [path for path in record_paths if path not in destinations]
    if missing:
        _fail("backend records are absent from evidence files: " + ", ".join(missing))
    if "release-evidence.json" not in destinations:
        _fail("evidence files must include release-evidence.json")


def _validate_v04_contract(
    manifest: dict[str, Any], files: list[dict[str, str]], records: list[dict[str, Any]]
) -> None:
    if manifest.get("release_line") != "v0.4":
        return
    destinations = {entry["destination"] for entry in files}
    trusted_paths = _validate_source_contract(manifest, destinations)
    families = _required_families(manifest)
    _validate_record_identities(records)
    for record in records:
        _validate_record_contract(
            record,
            destinations,
            families,
            trusted_paths,
            manifest.get("status") == "complete",
        )
    _validate_backend_contract(records, families)
    _validate_shared_contract(records)
    _validate_f32_files(files, records)


def _cuda_sample_root(record_path: str) -> str:
    return f"{record_path.removesuffix('.json')}-cuda-samples"


def _validate_f32_files(files: list[dict[str, str]], records: list[dict[str, Any]]) -> None:
    allowed = set()
    for record in records:
        if record.get("role") == "quality":
            allowed.update(_validate_quality_sample_files(record))
    undeclared = sorted(
        entry["destination"]
        for entry in files
        if entry["destination"].lower().endswith(".f32")
        and entry["destination"] not in allowed
    )
    if undeclared:
        _fail(
            "evidence f32 file is outside a declared quality sample artifact: "
            + ", ".join(undeclared)
        )


def _validate_quality_sample_files(record: dict[str, Any]) -> set[str]:
    declared = {
        path
        for path in record.get("artifact_files", [])
        if path.lower().endswith(".f32")
    }
    expected = set()
    rooted = True
    if record.get("backend") == "cuda":
        root = _cuda_sample_root(record["path"])
        expected = {f"{root}/{name}" for name in V04_QUALITY_SAMPLE_F32_NAMES}
        rooted = root in record.get("artifact_roots", [])
    if declared != expected or not rooted:
        _fail(
            "quality sample f32 artifacts differ from the bounded contract: "
            + record["path"]
        )
    return expected


def _validate_record_identities(records: list[dict[str, Any]]) -> None:
    for record in records:
        identity = tuple(record.get(key) for key in ("platform", "target", "backend"))
        if identity not in ALLOWED_IDENTITIES:
            _fail(f"v0.4 record has an unsupported platform/backend pair: {record['path']}")


def _validate_source_contract(manifest: dict[str, Any], destinations: set[str]) -> set[str]:
    _require_source_manifest(manifest, destinations)
    trusted_paths = _trusted_paths(manifest)
    _validate_trusted_paths(trusted_paths, destinations)
    return trusted_paths


def _require_source_manifest(manifest: dict[str, Any], destinations: set[str]) -> None:
    source_manifest = manifest.get("source_manifest")
    if not isinstance(source_manifest, str) or source_manifest not in destinations:
        _fail("v0.4 evidence must package its source input manifest")


def _trusted_paths(manifest: dict[str, Any]) -> set[str]:
    trusted = manifest.get("trusted_validators")
    if not isinstance(trusted, list) or not trusted:
        _fail("v0.4 evidence must declare trusted validators")
    trusted_paths_list = [_relative(item, "trusted validator") for item in trusted]
    if len(set(trusted_paths_list)) != len(trusted_paths_list):
        _fail("v0.4 evidence repeats a trusted validator")
    return set(trusted_paths_list)


def _validate_trusted_paths(trusted_paths: set[str], destinations: set[str]) -> None:
    missing = sorted(V04_TRUSTED_FILES - trusted_paths)
    if missing:
        _fail("v0.4 evidence omits trusted validators: " + ", ".join(missing))
    extra = sorted(trusted_paths - V04_TRUSTED_FILES)
    if extra:
        _fail("v0.4 evidence declares unknown trusted validators: " + ", ".join(extra))
    absent = sorted(path for path in trusted_paths if path not in destinations)
    if absent:
        _fail("trusted validator is absent from evidence files: " + ", ".join(absent))


def _required_families(manifest: dict[str, Any]) -> list[str]:
    families = manifest.get("required_model_families")
    if not isinstance(families, list) or not families or any(
        not isinstance(family, str) or not family for family in families
    ):
        _fail("v0.4 evidence must declare required model families")
    if set(families) != set(V04_REQUIRED_MODEL_FAMILIES) or len(families) != len(
        V04_REQUIRED_MODEL_FAMILIES
    ):
        _fail(
            "v0.4 evidence must declare the reviewed model families: "
            + ", ".join(V04_REQUIRED_MODEL_FAMILIES)
        )
    return list(V04_REQUIRED_MODEL_FAMILIES)


def _validate_record_contract(
    record: dict[str, Any],
    destinations: set[str],
    families: list[str],
    trusted_paths: set[str],
    complete: bool,
) -> None:
    if not {"role", "validator", "dependencies"}.issubset(record):
        _fail(f"v0.4 record lacks a semantic validator: {record['path']}")
    role = record["role"]
    validator = record["validator"]
    if role not in V04_ROLES:
        _fail(f"v0.4 record has an unknown role: {record['path']}")
    if validator not in V04_VALIDATORS:
        _fail(f"v0.4 record has an unknown validator: {record['path']}")
    if validator not in V04_ROLE_VALIDATORS[role]:
        _fail(f"v0.4 record validator does not match its role: {record['path']}")
    _validate_record_binding(record, destinations, families, complete)
    _validate_validator_command(record, validator, trusted_paths, destinations, complete)


def _validate_record_binding(
    record: dict[str, Any], destinations: set[str], families: list[str], complete: bool
) -> None:
    if any(dependency not in destinations for dependency in record["dependencies"]):
        _fail(f"v0.4 record has an undeclared dependency: {record['path']}")
    _validate_role_binding(record, destinations, complete)
    _validate_quality_record(record, families)
    _validate_complete_model_binding(record, complete)


def _validate_role_binding(
    record: dict[str, Any], destinations: set[str], complete: bool
) -> None:
    role = record["role"]
    if role in {"runtime", "service", "metrics", "batching", "branching"}:
        _require_model_binding(record, "qwen3")
    if role == "batching":
        _validate_batching_record(record, complete)
    if role == "branching":
        _validate_branching_record(record, complete)
    if role == "client":
        if record.get("schema_version") != "leone.openai-client-check.v2":
            _fail(f"v0.4 client record must use the v2 API evidence schema: {record['path']}")
        _require_model_binding(record, "llama")
        if complete and "binary_sha256" not in record:
            _fail(f"complete v0.4 client record has no native binary SHA-256: {record['path']}")


def _validate_batching_record(record: dict[str, Any], complete: bool) -> None:
    path = record["path"]
    if path != V04_BATCHING_PATH or record.get("schema_version") != V04_BATCHING_SCHEMA:
        _fail(f"v0.4 batching record must be {V04_BATCHING_PATH} with schema {V04_BATCHING_SCHEMA}: {path}")
    if record.get("quality_record") != V04_BATCHING_QUALITY_RECORD:
        _fail(f"v0.4 batching record must bind {V04_BATCHING_QUALITY_RECORD}: {path}")
    if record.get("quality_receipt") != V04_BATCHING_QUALITY_RECEIPT:
        _fail(f"v0.4 batching record must bind {V04_BATCHING_QUALITY_RECEIPT}: {path}")
    missing = sorted(V04_BATCHING_DEPENDENCIES - set(record["dependencies"]))
    if missing:
        _fail(f"v0.4 batching record omits dependencies: {', '.join(missing)}")
    if complete and "binary_sha256" not in record:
        _fail(f"complete v0.4 batching record has no native binary SHA-256: {path}")


def _validate_branching_record(record: dict[str, Any], complete: bool) -> None:
    path = record["path"]
    if path not in V04_BRANCHING or record.get("schema_version") != V04_BRANCHING_SCHEMA:
        _fail(f"v0.4 branching record has an unreviewed path or schema: {path}")
    if record.get("quality_record") != V04_BRANCHING[path][1]:
        _fail(f"v0.4 branching record must bind {V04_BRANCHING[path][1]}: {path}")
    backend, quality = V04_BRANCHING[path]
    base = quality.removesuffix(".json")
    sidecars = {f"{base}-{name}.json" for name in V04_QUALITY_SIDECARS[backend]}
    missing = sorted((V04_BRANCHING_DEPENDENCIES | sidecars) - set(record["dependencies"]))
    if missing:
        _fail(f"v0.4 branching record omits dependencies: {', '.join(missing)}")
    _validate_branching_files(record)
    if complete and "binary_sha256" not in record:
        _fail(f"complete v0.4 branching record has no native binary SHA-256: {path}")


def _validate_branching_files(record: dict[str, Any]) -> None:
    path = record["path"]
    history = [record.get("history_expected"), record.get("history_result")]
    study = record.get("study_files")
    if not isinstance(study, list) or not study or None in history:
        _fail(f"v0.4 branching record has no study or history files: {path}")
    files = [*study, *history, record["quality_record"], path]
    if len(set(files)) != len(files):
        _fail(f"v0.4 branching record repeats a study or history file: {path}")
    if not set(study) | set(history) <= set(record["dependencies"]):
        _fail(f"v0.4 branching record does not depend on its study files: {path}")


def _require_model_binding(record: dict[str, Any], family: str) -> None:
    if record.get("model_family") != family:
        _fail(f"v0.4 {record['role']} record has an unexpected model family: {record['path']}")
    if "model_sha256" not in record:
        _fail(f"v0.4 {record['role']} record has no model SHA-256: {record['path']}")
    if record.get("model_sha256") != V04_SUBJECT_MODEL_SHA256[family]:
        _fail(f"v0.4 {record['role']} record has an unreviewed model SHA-256: {record['path']}")


def _validate_quality_record(record: dict[str, Any], families: list[str]) -> None:
    if record["role"] != "quality":
        return
    if record.get("model_family") not in families:
        _fail(f"v0.4 quality record has no required model family: {record['path']}")
    if record.get("metric_family") != V04_REQUIRED_QUALITY_METRIC:
        _fail(f"v0.4 quality record must bind {V04_REQUIRED_QUALITY_METRIC}: {record['path']}")
    _validate_quality_binding(record)


def _validate_complete_model_binding(record: dict[str, Any], complete: bool) -> None:
    if complete and record["role"] in {"runtime", "quality"} and "model_sha256" not in record:
        _fail(f"v0.4 {record['role']} record has no model SHA-256: {record['path']}")


def _validate_validator_command(
    record: dict[str, Any],
    validator: str,
    trusted_paths: set[str],
    destinations: set[str],
    complete: bool,
) -> None:
    script = V04_VALIDATOR_SCRIPTS.get(validator)
    if script is None:
        if validator in V04_EXTERNAL_VALIDATORS:
            return
        if complete:
            _fail(f"canonical validator is unavailable: {validator}")
        return
    if script not in trusted_paths or script not in record["dependencies"]:
        _fail(f"v0.4 record does not package its canonical validator: {record['path']}")
    if validator == "quality-stage-v1":
        _validate_quality_stage_record(record, destinations, complete)


def _validate_quality_stage_record(
    record: dict[str, Any], destinations: set[str], complete: bool
) -> None:
    mode = record.get("validator_mode")
    _validate_quality_stage_header(record, mode)
    if mode != "comparison":
        _validate_artifact_contract(record, destinations)
        return
    _validate_comparison_stage(record, destinations, complete)


def _validate_quality_stage_header(record: dict[str, Any], mode: Any) -> None:
    if mode not in {"export", "metal", "comparison"}:
        _fail(f"quality stage record has no canonical validator mode: {record['path']}")
    if "artifact_root" not in record:
        _fail(f"quality stage record has no artifact root: {record['path']}")


def _validate_comparison_stage(
    record: dict[str, Any], destinations: set[str], complete: bool
) -> None:
    _validate_comparison_identity(record)
    if record["backend"] == "cuda":
        _validate_cuda_trusted_inputs(record, destinations)
    elif complete:
        _validate_metal_native_artifacts(record)
    _validate_comparison_fields(record, complete)
    if V04_QUALITY_ADAPTER_PATH not in record["dependencies"]:
        _fail(
            "comparison quality stage does not depend on the trusted adapter: "
            + record["path"]
        )
    _validate_artifact_contract(record, destinations)


def _validate_comparison_identity(record: dict[str, Any]) -> None:
    if record.get("backend") not in {"metal", "cuda"}:
        _fail("quality stage backend is unsupported: " + record["path"])
    if record.get("validator") != "quality-stage-v1":
        _fail("quality stage validator does not match its backend: " + record["path"])
    expected_schema = (
        "leone.quality-comparison.v2"
        if record["backend"] == "cuda"
        else "leone.quality-cross-device.v2"
    )
    if record.get("schema_version") != expected_schema:
        _fail("quality stage schema does not match its backend: " + record["path"])


def _validate_cuda_trusted_inputs(record: dict[str, Any], destinations: set[str]) -> None:
    trusted_inputs = record.get("trusted_inputs")
    if not isinstance(trusted_inputs, str):
        _fail("CUDA quality stage has no external trusted inputs: " + record["path"])
    if trusted_inputs not in destinations:
        _fail("CUDA trusted inputs are absent from evidence files: " + record["path"])
    if trusted_inputs not in record["dependencies"]:
        _fail("CUDA quality stage does not depend on its trusted inputs: " + record["path"])
    generation_record = record.get("generation_record")
    if generation_record is not None:
        _validate_cuda_generation_record(record, generation_record, destinations)


def _validate_cuda_generation_record(
    record: dict[str, Any], generation_record: str, destinations: set[str]
) -> None:
    if generation_record not in destinations:
        _fail("CUDA generation record is absent from evidence files: " + record["path"])
    if generation_record not in record["dependencies"]:
        _fail("CUDA quality stage does not depend on its generation record: " + record["path"])
    snapshot = record["artifact_root"] + "/" + V04_CUDA_GENERATION_ARTIFACT
    if snapshot not in record.get("artifact_files", []):
        _fail("CUDA quality stage does not package its generation record: " + record["path"])


def _validate_metal_native_artifacts(record: dict[str, Any]) -> None:
    stages = [
        root for root in record.get("artifact_roots", []) if root.endswith(V04_METAL_STAGE_SUFFIX)
    ]
    files = set(record.get("artifact_files", []))
    missing = sorted(
        f"{stage}/{name}"
        for stage in stages
        for name in V04_METAL_NATIVE_ARTIFACTS
        if f"{stage}/{name}" not in files
    )
    if not stages or missing:
        _fail(
            "Metal quality stage does not package its native artifacts: "
            + (", ".join(missing) or record["path"])
        )


def _validate_comparison_fields(record: dict[str, Any], complete: bool) -> None:
    required = {
        "model_family",
        "model_sha256",
        "statistics_target",
        "metric_family",
        "corpus_path",
        "corpus_sha256",
        "oracle_model_sha256",
        "sample_contract",
    }
    if complete:
        if record.get("backend") == "metal":
            required.add("sample_manifest_sha256")
            required.add("native_source_commit")
    if complete and record.get("backend") == "cuda":
        required.add("statistics_platform")
        required.add("task_manifest_sha256")
        required.add("generation_record")
        required.add("generation_record_sha256")
    missing = sorted(required - record.keys())
    if missing:
        _fail(
            "comparison quality stage lacks trusted identity: "
            + ", ".join(missing)
        )


def _validate_artifact_contract(record: dict[str, Any], destinations: set[str]) -> None:
    root = record["artifact_root"]
    roots = record.get("artifact_roots", [root])
    files = record.get("artifact_files")
    _require_artifact_lists(record, roots, files)
    _require_artifact_paths(record, root, roots, files)
    _require_artifact_destinations(destinations, files)
    _require_artifact_sidecars(roots, files)


def _require_artifact_lists(
    record: dict[str, Any], roots: Any, files: Any
) -> None:
    if not isinstance(roots, list) or not roots:
        _fail(f"quality stage record has no artifact roots: {record['path']}")
    if not isinstance(files, list) or not files:
        _fail(f"quality stage record has no artifact sidecars: {record['path']}")


def _require_artifact_paths(
    record: dict[str, Any], root: str, roots: list[str], files: list[str]
) -> None:
    if len(set(roots)) != len(roots) or len(set(files)) != len(files):
        _fail(f"quality stage record repeats an artifact path: {record['path']}")
    if root not in roots:
        _fail(f"quality stage primary artifact root is undeclared: {record['path']}")


def _require_artifact_destinations(destinations: set[str], files: list[str]) -> None:
    missing = sorted(path for path in files if path not in destinations)
    if missing:
        _fail(
            "quality stage artifact sidecar is absent from evidence files: "
            + ", ".join(missing)
        )


def _require_artifact_sidecars(roots: list[str], files: list[str]) -> None:
    for artifact_root in roots:
        prefix = artifact_root + "/"
        if not any(path.startswith(prefix) for path in files):
            _fail(f"quality stage artifact root has no sidecar: {artifact_root}")


def _validate_quality_binding(record: dict[str, Any]) -> None:
    family = record.get("model_family")
    expected_subject = V04_SUBJECT_MODEL_SHA256.get(family)
    if record.get("model_sha256") != expected_subject:
        _fail(f"v0.4 quality record has an unreviewed model SHA-256: {record['path']}")
    expected = {
        "corpus_path": V04_CORPUS_PATH,
        "corpus_sha256": V04_CORPUS_SHA256,
        "oracle_model_sha256": V04_ORACLE_MODEL_SHA256[family],
        "sample_contract": V04_SAMPLE_CONTRACT,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            _fail(f"v0.4 quality record has an unreviewed {key}: {record['path']}")
    if record.get("statistics_target") != V04_STATISTICS_TARGET:
        _fail(f"v0.4 quality record has an unreviewed statistics_target: {record['path']}")
    if "statistics_platform" in record and record["statistics_platform"] != V04_STATISTICS_PLATFORM:
        _fail(f"v0.4 quality record has an unreviewed statistics_platform: {record['path']}")


def _validate_backend_contract(records: list[dict[str, Any]], families: list[str]) -> None:
    backend_names = {item[2] for item in REQUIRED_BACKENDS}
    backend_records = [
        record
        for record in records
        if record.get("_scope") == "backend" and record.get("backend") in backend_names
    ]
    for backend in REQUIRED_BACKENDS:
        _validate_one_backend(backend[2], backend_records, families)


def _validate_one_backend(
    backend_name: str, records: list[dict[str, Any]], families: list[str]
) -> None:
    scoped = [record for record in records if record["backend"] == backend_name]
    _require_backend_roles(backend_name, scoped)
    _require_quality_families(backend_name, scoped, families)
    for record in scoped:
        if record.get("role") == "quality" and not _is_comparison_quality(record):
            _fail(f"v0.4 {backend_name} quality is not linked comparison evidence")


def _require_backend_roles(backend_name: str, records: list[dict[str, Any]]) -> None:
    required = V04_BACKEND_ROLES[backend_name]
    roles = [record.get("role") for record in records]
    if set(roles) != required or any(
        roles.count(role) != 1 for role in required - {"quality"}
    ):
        _fail(
            f"v0.4 {backend_name} evidence must contain records for exactly these roles: "
            + ", ".join(sorted(required))
        )


def _require_quality_families(
    backend_name: str, records: list[dict[str, Any]], families: list[str]
) -> None:
    quality_families = [
        record.get("model_family") for record in records if record.get("role") == "quality"
    ]
    if sorted(quality_families) != sorted(families):
        _fail(
            f"v0.4 {backend_name} evidence must contain one quality record for every gated model"
        )


def _is_comparison_quality(record: dict[str, Any]) -> bool:
    return (
        record.get("validator") == "quality-stage-v1"
        and record.get("validator_mode") == "comparison"
    )


def _validate_shared_contract(records: list[dict[str, Any]]) -> None:
    shared = [
        record for record in records if record.get("_scope") == "shared"
    ]
    roles = [record.get("role") for record in shared]
    if roles.count("client") != len(REQUIRED_BACKENDS) or roles.count("metrics") != 1 or len(shared) != 3:
        _fail("v0.4 evidence must contain one CUDA client, one Metal client, and one metrics record")
    _validate_shared_identities(shared)
    _validate_batching_binary(records)
    _validate_branching_binary(records)
    _validate_branching_links(records)


def _validate_batching_binary(records: list[dict[str, Any]]) -> None:
    cuda_identity = REQUIRED_BACKENDS[0]

    def binaries(role: str) -> set[Any]:
        return {
            record.get("binary_sha256")
            for record in records
            if record.get("role") == role
            and tuple(record.get(key) for key in ("platform", "target", "backend")) == cuda_identity
        }

    if binaries("batching") != binaries("client"):
        _fail("v0.4 batching record binary differs from the CUDA client binary")


def _validate_branching_binary(records: list[dict[str, Any]]) -> None:
    clients = {
        record.get("backend"): record.get("binary_sha256")
        for record in records
        if record.get("role") == "client"
    }
    for record in records:
        if record.get("role") == "branching" and record.get("binary_sha256") != clients.get(
            record.get("backend")
        ):
            _fail(f"v0.4 branching record binary differs from the {record.get('backend')} client binary")


def _validate_branching_links(records: list[dict[str, Any]]) -> None:
    by_path = {record["path"]: record for record in records}
    for record in records:
        if record.get("role") != "branching":
            continue
        linked = by_path.get(record.get("quality_record"))
        if (
            linked is None
            or not _is_comparison_quality(linked)
            or linked.get("backend") != record.get("backend")
            or linked.get("model_family") != record.get("model_family")
            or linked.get("model_sha256") != record.get("model_sha256")
        ):
            _fail(f"v0.4 branching record has no matching quality comparison: {record['path']}")


def _validate_shared_identities(shared: list[dict[str, Any]]) -> None:
    client_identities = {
        tuple(record.get(key) for key in ("platform", "target", "backend"))
        for record in shared
        if record.get("role") == "client"
    }
    if client_identities != set(REQUIRED_BACKENDS):
        _fail("v0.4 evidence must contain one client record for each native backend")
    metrics = [record for record in shared if record.get("role") == "metrics"]
    if tuple(metrics[0].get(key) for key in ("platform", "target", "backend")) != REQUIRED_BACKENDS[0]:
        _fail("v0.4 metrics record has an unsupported release identity")


def _validate_record_hashes(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    if manifest.get("release_line") == "v0.4" and manifest.get("status") == "complete":
        missing = [record["path"] for record in records if "sha256" not in record]
        if missing:
            _fail("complete v0.4 evidence records must declare SHA-256: " + ", ".join(missing))


def validate(manifest: dict[str, Any], *, require_complete: bool = False) -> list[dict[str, Any]]:
    """Validate structure and return backend-specific record requirements."""
    _validate_header(manifest)
    _validate_status(manifest, require_complete)
    files = _file_entries(manifest)
    records = _record_entries(manifest)
    _validate_record_destinations(files, records)
    _validate_v04_contract(manifest, files, records)
    _validate_record_hashes(manifest, records)
    return records


def source_files(
    manifest: dict[str, Any],
    source_root: Path,
    source_overrides: dict[str, Path] | None = None,
) -> list[dict[str, str]]:
    """Validate and return manifest files that exist below one source root."""
    entries = _file_entries(manifest)
    for entry in entries:
        override = (source_overrides or {}).get(entry["source"])
        if override is None:
            _source_path(source_root, entry["source"])
        else:
            _source_override(override, entry["source"])
    return entries


def _source_path(source_root: Path, relative: str) -> Path:
    source = source_root / relative
    current = source_root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            _fail(f"evidence input contains a symlink: {relative}")
    try:
        resolved = source.resolve(strict=True)
        resolved.relative_to(source_root.resolve())
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"evidence input is outside the source tree: {relative}") from error
    if not source.is_file() or source.is_symlink():
        _fail(f"evidence input is not a regular file: {relative}")
    return source


def _source_override(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        _fail(f"evidence binary override is not a regular file: {label}")
    current = path.parent
    while current != current.parent:
        if current.is_symlink():
            _fail(f"evidence binary override has a symlink parent: {label}")
        current = current.parent
    return path


def stage(
    manifest: dict[str, Any],
    source_root: Path,
    destination: Path,
    source_overrides: dict[str, Path] | None = None,
) -> None:
    """Copy every declared evidence file into a fresh staging directory."""
    validate(manifest, require_complete=True)
    _prepare_destination(destination)
    overrides = source_overrides or {}
    entries = source_files(manifest, source_root, overrides)
    for entry in entries:
        source = overrides.get(entry["source"])
        if source is None:
            source = _source_path(source_root, entry["source"])
        else:
            source = _source_override(source, entry["source"])
        target = _destination_path(destination, entry["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(source.stat().st_mode & 0o777)


def _prepare_destination(destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            _fail("evidence staging destination is not a directory")
        if any(destination.iterdir()):
            _fail("evidence staging destination is not empty")
        return
    existing = destination.parent
    while not existing.exists():
        if existing.is_symlink():
            _fail("evidence staging destination parent contains a symlink")
        existing = existing.parent
    if existing.is_symlink() or not existing.is_dir():
        _fail("evidence staging destination parent is not a directory")
    destination.mkdir(parents=True)


def _destination_path(destination: Path, relative: str) -> Path:
    target = destination / relative
    current = destination
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            _fail(f"evidence staging destination contains a symlink: {relative}")
    try:
        target.resolve(strict=False).relative_to(destination.resolve(strict=True))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"evidence staging destination escapes its root: {relative}") from error
    if target.exists() and not target.is_file():
        _fail(f"evidence staging destination is not a regular file: {relative}")
    return target


def select(source_root: Path) -> Path:
    """Select the manifest matching the workspace major and minor version."""
    cargo = source_root / "Cargo.toml"
    version = ""
    for line in cargo.read_text(encoding="utf-8").splitlines():
        if line.startswith('version = "'):
            version = line.split('"', 2)[1]
            break
    if not version:
        _fail("workspace version is missing")
    parts = version.split(".")
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        _fail(f"workspace version is invalid: {version}")
    path = source_root / "packaging" / f"release-evidence.v{parts[0]}.{parts[1]}.json"
    if not path.is_file():
        _fail(f"release evidence manifest is missing for {version}: {path}")
    return path
