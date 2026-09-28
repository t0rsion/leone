"""Validate receipts from the opt-in production prefill differential."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from common_oracle import PROVENANCE_SCOPE, f32, output_digest
from generate_manifest import (
    MANIFEST_INPUT_DIGEST,
    MANIFEST_INPUT_GENERATOR,
    MANIFEST_ORACLE_ID,
    MANIFEST_STORAGE,
    validate_manifest,
)
from gpu_receipt import finite_value, manifest, require
from runner_support import LEGACY_SOURCE_LAYOUT, SOURCE_LAYOUT
from production_oracle import (
    case_digests,
    case_oracle_values,
    max_absolute_error,
    max_relative_error,
)

SCHEMA = "prefix-attention-production-receipt-v1"
CRITERIA_SCHEMA = "prefix-attention-production-criteria-v1"
OPERATOR = "leone-production-prefill-attention-f16"
QUALITY_RULE = "the production FP16 attention error must not exceed the largest calibration error plus one representable FP64 step"
DIGEST16 = re.compile(r"^[0-9a-f]{16}$")
HASH64 = re.compile(r"^[0-9a-f]{64}$")
LEGACY_SOURCE_NAMES = {
    "research/prefix_attention/production_driver/Cargo.toml",
    "research/prefix_attention/production_driver/Cargo.lock",
    "research/prefix_attention/production_driver/src/main.rs",
    "research/prefix_attention/gpu_manifest.json",
    "research/prefix_attention/ORACLE_CONTRACT.md",
    "research/prefix_attention/common_oracle.py",
    "research/prefix_attention/production_oracle.py",
    "research/prefix_attention/run_production.py",
    "research/prefix_attention/production_receipt.py",
    "research/prefix_attention/freeze_production_criteria.py",
    "research/prefix_attention/validate_production_receipt.py",
}
SOURCE_NAMES = LEGACY_SOURCE_NAMES | {"research/prefix_attention/runner_support.py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value: Any, field: str, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is not finite")
    require(result > 0 if positive else result >= 0, f"{field} is out of range")
    return result


def expected_cases(phase: str) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in manifest()[phase]}


def validate_provenance_identity(provenance: dict[str, Any]) -> None:
    require(provenance.get("scope") == PROVENANCE_SCOPE, "unsupported production provenance scope")
    require(provenance.get("manifest_sha256") == sha256(Path(__file__).with_name("gpu_manifest.json")), "manifest hash differs")
    require(isinstance(provenance.get("git_revision"), str) and provenance["git_revision"], "missing revision")
    require(isinstance(provenance.get("working_tree_dirty"), bool), "missing dirty state")


def validate_provenance_sources(provenance: dict[str, Any]) -> None:
    sources = provenance.get("source_sha256")
    layout = provenance.get("source_layout", LEGACY_SOURCE_LAYOUT)
    if layout == LEGACY_SOURCE_LAYOUT:
        expected = LEGACY_SOURCE_NAMES
    elif layout == SOURCE_LAYOUT:
        expected = SOURCE_NAMES
    else:
        raise ValueError("unsupported source layout")
    require(isinstance(sources, dict) and set(sources) == expected, "invalid source set")
    require(all(isinstance(value, str) and HASH64.fullmatch(value) for value in sources.values()), "invalid source hashes")


def validate_provenance_build(provenance: dict[str, Any]) -> None:
    require(isinstance(provenance.get("artifact_sha256"), str) and HASH64.fullmatch(provenance["artifact_sha256"]), "invalid driver hash")
    build = provenance.get("build")
    require(
        isinstance(build, dict)
        and isinstance(build.get("command"), list)
        and build["command"]
        and isinstance(build.get("rustc_version"), str)
        and build["rustc_version"],
        "missing build provenance",
    )
    require(isinstance(provenance.get("host_system"), str) and provenance["host_system"], "missing host provenance")
    require(isinstance(provenance.get("command"), list) and provenance["command"], "missing run command")


def validate_provenance_device(receipt: dict[str, Any]) -> None:
    device = receipt.get("device")
    require(
        isinstance(device, dict)
        and isinstance(device.get("name"), str)
        and device["name"]
        and isinstance(device.get("compute_capability"), str)
        and device["compute_capability"],
        "missing device",
    )


def validate_provenance(receipt: dict[str, Any]) -> None:
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "missing production provenance")
    validate_provenance_identity(provenance)
    validate_provenance_sources(provenance)
    validate_provenance_build(provenance)
    validate_provenance_device(receipt)


def validate_output_values(values: Any, expected_length: int, field: str) -> list[float]:
    require(isinstance(values, list) and len(values) == expected_length, f"{field}: wrong output length")
    result = []
    for index, value in enumerate(values):
        converted = finite_value(value, f"{field}[{index}]")
        try:
            require(f32(converted) == converted, f"{field}[{index}] is not an FP32 value")
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{field}[{index}] is not an FP32 value") from error
        result.append(converted)
    return result


def validate_case(case: dict[str, Any], expected: dict[str, Any], repetitions: int) -> None:
    require(case.get("spec") == {key: expected[key] for key in (
        "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
        "tile_tokens", "group_rows", "seed", "shared_prefix",
    )}, f"{case.get('name')}: spec differs")
    require(case.get("production_query_rows") == expected["query_rows"], f"{case['name']}: query rows differ")
    require(case.get("layout_scope") == "causal_single_cache", f"{case['name']}: layout scope differs")
    for field in ("input_digest", "oracle_digest", "backend_oracle_digest", "output_digest"):
        require(isinstance(case.get(field), str) and DIGEST16.fullmatch(case[field]), f"{case['name']}: invalid {field}")
    expected_input, expected_oracle = case_digests(expected)
    require(case["input_digest"] == expected_input, f"{case['name']}: input digest differs from common oracle")
    require(case["oracle_digest"] == expected_oracle, f"{case['name']}: oracle digest differs from common oracle")
    samples = case.get("samples_ms")
    require(isinstance(samples, list) and len(samples) == repetitions, f"{case['name']}: wrong sample count")
    for index, value in enumerate(samples):
        finite(value, f"{case['name']}.samples_ms[{index}]", positive=True)
    median = finite(case.get("median_ms"), f"{case['name']}.median_ms", positive=True)
    require(median == sorted(samples)[len(samples) // 2], f"{case['name']}: invalid median")
    output_length = expected["query_rows"] * expected["query_heads"] * expected["head_dim"]
    output = validate_output_values(case.get("output_values"), output_length, f"{case['name']}.output_values")
    require(case["output_digest"] == output_digest(output), f"{case['name']}: output digest differs from values")
    oracle = case_oracle_values(expected)
    require(case["quality_max_abs"] == max_absolute_error(output, oracle), f"{case['name']}: quality absolute error was not recomputed")
    require(case["quality_max_rel"] == max_relative_error(output, oracle), f"{case['name']}: quality relative error was not recomputed")
    finite(case.get("backend_quality_max_abs"), f"{case['name']}.backend_quality_max_abs")
    finite(case.get("backend_quality_max_rel"), f"{case['name']}.backend_quality_max_rel")


def validate_receipt(path: Path, phase: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    data = manifest()
    validate_manifest(data)
    require(receipt.get("schema") == SCHEMA, "wrong production schema")
    require(receipt.get("operator") == OPERATOR, "wrong production operator")
    require(receipt.get("backend") == "cuda", "wrong production backend")
    require(receipt.get("phase") == phase, "wrong production phase")
    require(receipt.get("storage") == MANIFEST_STORAGE, "wrong production storage")
    require(receipt.get("input_generator") == MANIFEST_INPUT_GENERATOR, "wrong production input generator")
    require(receipt.get("input_digest_algorithm") == MANIFEST_INPUT_DIGEST, "wrong production input digest")
    require(receipt.get("oracle_id") == MANIFEST_ORACLE_ID, "wrong production oracle")
    require(receipt.get("warmups") == data["warmup_runs"], "production warmups differ")
    require(receipt.get("repetitions") == data["measured_runs"], "production repetitions differ")
    records = receipt.get("cases")
    expected = expected_cases(phase)
    require(isinstance(records, list) and len(records) == len(expected), "production cases differ")
    require(all(isinstance(case, dict) and case.get("name") in expected for case in records), "production case is unknown")
    require(len({case["name"] for case in records}) == len(expected), "production cases repeat")
    for case in records:
        validate_case(case, expected[case["name"]], data["measured_runs"])
    validate_provenance(receipt)
    return receipt


def quality_bounds(receipt: dict[str, Any]) -> dict[str, float]:
    absolute = max(case["quality_max_abs"] for case in receipt["cases"])
    relative = max(case["quality_max_rel"] for case in receipt["cases"])
    return {
        "max_absolute_error": math.nextafter(absolute, math.inf),
        "max_relative_error": math.nextafter(relative, math.inf),
    }


def validate_criteria(path: Path, calibration_path: Path) -> dict[str, Any]:
    criteria = json.loads(path.read_text())
    calibration = validate_receipt(calibration_path, "calibration")
    require(criteria.get("schema") == CRITERIA_SCHEMA, "wrong production criteria schema")
    require(criteria.get("operator") == OPERATOR, "wrong production criteria operator")
    require(criteria.get("backend") == "cuda", "wrong production criteria backend")
    require(criteria.get("calibration_receipt_sha256") == sha256(calibration_path), "production calibration hash differs")
    require(criteria.get("manifest_sha256") == sha256(Path(__file__).with_name("gpu_manifest.json")), "production manifest hash differs")
    require(criteria.get("repetitions") == manifest()["measured_runs"], "production criteria repetitions differ")
    require(criteria.get("device") == calibration["device"], "production criteria device differs")
    require(
        criteria.get("source_layout", LEGACY_SOURCE_LAYOUT)
        == calibration["provenance"].get("source_layout", LEGACY_SOURCE_LAYOUT),
        "production criteria source layout differs",
    )
    require(criteria.get("artifact_sha256") == calibration["provenance"]["artifact_sha256"], "production artifact differs")
    require(criteria.get("source_sha256") == calibration["provenance"]["source_sha256"], "production sources differ")
    require(criteria.get("build") == calibration["provenance"]["build"], "production build differs")
    require(criteria.get("git_revision") == calibration["provenance"]["git_revision"], "production revision differs")
    require(criteria.get("working_tree_dirty") == calibration["provenance"]["working_tree_dirty"], "production dirty state differs")
    require(criteria.get("quality_rule") == QUALITY_RULE, "production quality rule differs")
    require(criteria.get("quality") == quality_bounds(calibration), "production quality bounds differ from calibration")
    return criteria
