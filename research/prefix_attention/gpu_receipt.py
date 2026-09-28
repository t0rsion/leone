"""Shared validation for generated prefix attention GPU records."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from common_oracle import (
    PROVENANCE_SCOPE,
    case_digests,
    case_oracle_values,
    f32,
    max_absolute_error,
    max_relative_error,
    output_digest,
)
from generate_manifest import (
    MANIFEST_INPUT_DIGEST,
    MANIFEST_INPUT_GENERATOR,
    MANIFEST_OPERATOR,
    MANIFEST_ORACLE_ID,
    MANIFEST_STORAGE,
    validate_manifest,
)
from runner_support import LEGACY_SOURCE_LAYOUT, SOURCE_LAYOUT

ROOT = Path(__file__).parent
MANIFEST_PATH = ROOT / "gpu_manifest.json"
SCHEMA = "prefix-attention-gpu-receipt-v1"
CRITERIA_SCHEMA = "prefix-attention-gpu-criteria-v1"
OPERATOR = MANIFEST_OPERATOR
DIGEST16 = re.compile(r"^[0-9a-f]{16}$")
HASH64 = re.compile(r"^[0-9a-f]{64}$")
UNAVAILABLE_EVIDENCE = ["host_launch_time", "dram_traffic", "power", "clocks"]
QUALITY_RULE = "the evaluation error must not exceed the largest calibration error plus one representable FP64 step"
TIMING_RULE = "paired baseline-candidate-baseline samples use the nearest-rank 95th percentile of abs(after-before)/mean(before,after)"
FIXED_SCHEDULE_PATHS = {
    "per_row",
    "fixed_tile_per_row",
    "shared_read_fixed_reduction",
}
LEGACY_SOURCE_NAMES = {
    "cuda": {
        "research/prefix_attention/cuda_fixed_reduction/operator.cuh",
        "research/prefix_attention/cuda_fixed_reduction/operator.cu",
        "research/prefix_attention/cuda_fixed_reduction/main.cu",
        "research/prefix_attention/cuda_fixed_reduction/manifest_cases.h",
        "research/prefix_attention/cuda_fixed_reduction/build_cuda.sh",
        "research/prefix_attention/generate_manifest.py",
        "research/prefix_attention/common_oracle.py",
        "research/prefix_attention/ORACLE_CONTRACT.md",
        "research/prefix_attention/gpu_manifest.json",
        "research/prefix_attention/run_cuda.py",
    },
    "metal": {
        "research/prefix_attention/metal_fixed_reduction/PrefixAttention.metal",
        "research/prefix_attention/metal_fixed_reduction/RunPrefixAttention.swift",
        "research/prefix_attention/metal_fixed_reduction/build_metal.sh",
        "research/prefix_attention/generate_manifest.py",
        "research/prefix_attention/common_oracle.py",
        "research/prefix_attention/ORACLE_CONTRACT.md",
        "research/prefix_attention/gpu_manifest.json",
        "research/prefix_attention/run_metal.py",
    },
}
SOURCE_NAMES = {
    backend: names | {"research/prefix_attention/runner_support.py"}
    for backend, names in LEGACY_SOURCE_NAMES.items()
}


def manifest() -> dict[str, Any]:
    data = json.loads(MANIFEST_PATH.read_text())
    validate_manifest(data)
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value: Any, field: str, *, positive: bool = False) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is not finite")
    require(result > 0 if positive else result >= 0, f"{field} is out of range")
    return result


def finite_value(value: Any, field: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{field} is not numeric")
    result = float(value)
    require(math.isfinite(result), f"{field} is not finite")
    return result


def exact_integer(value: Any, field: str, *, positive: bool = False) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), f"{field} is not an integer")
    require(value > 0 if positive else value >= 0, f"{field} is out of range")
    return value


def expected_cases(phase: str) -> dict[str, dict[str, Any]]:
    data = manifest()
    return {case["name"]: case for case in data[phase]}


def validate_identity(receipt: dict[str, Any], phase: str, backend: str) -> None:
    data = manifest()
    require(receipt.get("schema") == SCHEMA, "wrong receipt schema")
    require(receipt.get("operator") == MANIFEST_OPERATOR, "wrong receipt operator")
    require(receipt.get("backend") == backend, "wrong receipt backend")
    require(receipt.get("phase") == phase, "wrong receipt phase")
    require(receipt.get("storage") == MANIFEST_STORAGE, "wrong storage")
    require(receipt.get("input_generator") == MANIFEST_INPUT_GENERATOR, "wrong input generator")
    require(receipt.get("input_digest_algorithm") == MANIFEST_INPUT_DIGEST, "wrong input digest")
    require(receipt.get("oracle_id") == MANIFEST_ORACLE_ID, "wrong oracle id")
    require(receipt.get("claims") == [], "receipt contains an unreviewed claim")
    require(receipt.get("unavailable_evidence") == UNAVAILABLE_EVIDENCE, "evidence limits differ")
    require(exact_integer(receipt.get("warmups"), "warmups") == data["warmup_runs"], "warmups differ from manifest")
    require(exact_integer(receipt.get("repetitions"), "repetitions", positive=True) == data["measured_runs"], "repetitions differ from manifest")


def validate_provenance_identity(provenance: dict[str, Any]) -> None:
    require(provenance.get("scope") == PROVENANCE_SCOPE, "unsupported provenance scope")
    require(provenance.get("manifest_sha256") == sha256(MANIFEST_PATH), "manifest hash differs")
    require(isinstance(provenance.get("git_revision"), str) and provenance["git_revision"], "missing git revision")
    require(isinstance(provenance.get("working_tree_dirty"), bool), "missing dirty state")


def validate_artifact(provenance: dict[str, Any], backend: str) -> None:
    artifact = provenance.get("artifact_sha256")
    if backend == "cuda":
        require(isinstance(artifact, str) and HASH64.fullmatch(artifact), "missing CUDA artifact hash")
    else:
        require(isinstance(artifact, dict), "missing Metal artifact hashes")
        require(set(artifact) == {"runner", "metallib"}, "invalid Metal artifact set")
        require(all(isinstance(value, str) and HASH64.fullmatch(value) for value in artifact.values()), "invalid Metal artifact hash")


def validate_sources(provenance: dict[str, Any], backend: str) -> None:
    sources = provenance.get("source_sha256")
    layout = provenance.get("source_layout", LEGACY_SOURCE_LAYOUT)
    if layout == LEGACY_SOURCE_LAYOUT:
        expected = LEGACY_SOURCE_NAMES[backend]
    elif layout == SOURCE_LAYOUT:
        expected = SOURCE_NAMES[backend]
    else:
        raise ValueError("unsupported source layout")
    require(isinstance(sources, dict) and set(sources) == expected, "source set differs")
    require(
        all(isinstance(value, str) and HASH64.fullmatch(value) for value in sources.values()),
        "invalid source hash",
    )


def validate_build(provenance: dict[str, Any], backend: str) -> None:
    build = provenance.get("build")
    require(isinstance(build, dict) and isinstance(build.get("command"), list) and build["command"], "missing build provenance")
    required_build = {"nvcc", "cuda_arch", "nvcc_version"} if backend == "cuda" else {"xcrun_version", "swiftc_version"}
    require(required_build.issubset(build), "compiler provenance is incomplete")
    require(isinstance(provenance.get("command"), list) and provenance["command"], "missing run command")
    require(isinstance(provenance.get("host_system"), str) and provenance["host_system"], "missing host provenance")


def validate_device(receipt: dict[str, Any], backend: str) -> None:
    device = receipt.get("device")
    require(isinstance(device, dict) and isinstance(device.get("name"), str) and device["name"], "missing device")
    if backend == "cuda":
        require(isinstance(device.get("compute_major"), int) and not isinstance(device.get("compute_major"), bool), "missing CUDA capability")
    else:
        require(isinstance(device.get("registry_id"), int) and not isinstance(device.get("registry_id"), bool), "missing Metal registry id")


def validate_provenance(receipt: dict[str, Any], backend: str) -> None:
    provenance = receipt.get("provenance")
    require(isinstance(provenance, dict), "missing provenance")
    validate_provenance_identity(provenance)
    validate_artifact(provenance, backend)
    validate_sources(provenance, backend)
    validate_build(provenance, backend)
    validate_device(receipt, backend)


def validate_spec(case: dict[str, Any], expected: dict[str, Any]) -> None:
    require(case.get("spec") == {key: expected[key] for key in (
        "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
        "tile_tokens", "group_rows", "seed", "shared_prefix"
    )}, f"{case.get('name')}: spec differs from manifest")


def validate_samples(path: dict[str, Any], repetitions: int, prefix: str) -> None:
    samples = path.get("samples_ms")
    require(isinstance(samples, list) and len(samples) == repetitions, f"{prefix}: wrong sample count")
    for index, value in enumerate(samples):
        finite(value, f"{prefix}.samples_ms[{index}]", positive=True)
    median = finite(path.get("median_ms"), f"{prefix}.median_ms", positive=True)
    require(median == sorted(samples)[len(samples) // 2], f"{prefix}: median is not a sample median")


def expected_path_estimates(spec: dict[str, Any], path: str) -> dict[str, int]:
    tile_count = (spec["tokens"] + spec["tile_tokens"] - 1) // spec["tile_tokens"]
    if path in {"shared_read_unconstrained", "shared_read_fixed_reduction"}:
        rows_per_group = spec["group_rows"] if spec["shared_prefix"] else 1
        groups = ((spec["query_rows"] + rows_per_group - 1) // rows_per_group) * spec["query_heads"]
        shared_bytes = 2 * spec["tile_tokens"] * spec["head_dim"] * 2
    else:
        groups = spec["query_rows"] * spec["query_heads"]
        shared_bytes = 0
    return {
        "estimated_kv_elements_read": groups * spec["tokens"] * spec["head_dim"] * 2,
        "estimated_tile_loads": groups * tile_count,
        "output_elements_written": spec["query_rows"] * spec["query_heads"] * spec["head_dim"],
        "per_block_shared_bytes": shared_bytes,
    }


def validate_output_values(
    values: Any, expected_length: int, field: str
) -> list[float]:
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


def bitwise_equal(left: list[float], right: list[float]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        struct.pack("<f", value) == struct.pack("<f", other)
        for value, other in zip(left, right)
    )


def validate_path(
    path: dict[str, Any], expected_name: str, repetitions: int,
    prefix: str, spec: dict[str, Any], expected: list[float]
) -> None:
    require(path.get("path") == expected_name, f"{prefix}: unexpected path")
    validate_samples(path, repetitions, prefix)
    for field in (
        "quality_max_abs", "quality_max_rel", "schedule_b_quality_max_abs",
        "schedule_b_quality_max_rel", "backend_quality_max_abs",
        "backend_quality_max_rel", "backend_schedule_b_quality_max_abs",
        "backend_schedule_b_quality_max_rel",
    ):
        finite(path.get(field), f"{prefix}.{field}")
    for field in ("estimated_kv_elements_read", "estimated_tile_loads", "output_elements_written", "per_block_shared_bytes"):
        exact_integer(path.get(field), f"{prefix}.{field}")
    estimates = expected_path_estimates(spec, expected_name)
    for field, expected_value in estimates.items():
        require(path[field] == expected_value, f"{prefix}: invalid {field}")
    output_length = spec["query_rows"] * spec["query_heads"] * spec["head_dim"]
    output = validate_output_values(path.get("output_values"), output_length, f"{prefix}.output_values")
    schedule_output = validate_output_values(
        path.get("schedule_b_output_values"), output_length,
        f"{prefix}.schedule_b_output_values",
    )
    require(path["quality_max_abs"] == max_absolute_error(output, expected), f"{prefix}: quality absolute error was not recomputed")
    require(path["quality_max_rel"] == max_relative_error(output, expected), f"{prefix}: quality relative error was not recomputed")
    require(path["schedule_b_quality_max_abs"] == max_absolute_error(schedule_output, expected), f"{prefix}: schedule quality absolute error was not recomputed")
    require(path["schedule_b_quality_max_rel"] == max_relative_error(schedule_output, expected), f"{prefix}: schedule quality relative error was not recomputed")
    require(isinstance(path.get("digest"), str) and DIGEST16.fullmatch(path["digest"]), f"{prefix}: invalid digest")
    require(isinstance(path.get("schedule_b_digest"), str) and DIGEST16.fullmatch(path["schedule_b_digest"]), f"{prefix}: invalid schedule digest")
    require(path["digest"] == output_digest(output), f"{prefix}: output digest differs from values")
    require(path["schedule_b_digest"] == output_digest(schedule_output), f"{prefix}: schedule digest differs from values")
    require(isinstance(path.get("schedule_b_bitwise_equal"), bool), f"{prefix}: missing schedule equality")
    if expected_name in FIXED_SCHEDULE_PATHS:
        require(path["schedule_b_bitwise_equal"], f"{prefix}: fixed schedule changed output")
    if path["schedule_b_bitwise_equal"]:
        require(bitwise_equal(output, schedule_output), f"{prefix}: schedule equality is false")
        require(path["digest"] == path["schedule_b_digest"], f"{prefix}: equal schedules have different digests")


def validate_timing_pair(pair: dict[str, Any], repetitions: int, prefix: str) -> None:
    require(isinstance(pair, dict), f"{prefix}: missing timing pair")
    require(pair.get("baseline_path") == "fixed_tile_per_row", f"{prefix}: wrong baseline")
    require(pair.get("candidate_path") == "shared_read_fixed_reduction", f"{prefix}: wrong candidate")
    require(pair.get("acquisition_order") == ["baseline_before", "candidate", "baseline_after"], f"{prefix}: wrong acquisition order")
    for field in ("baseline_before_samples_ms", "candidate_samples_ms", "baseline_after_samples_ms"):
        values = pair.get(field)
        require(isinstance(values, list) and len(values) == repetitions, f"{prefix}.{field}: wrong samples")
        for index, value in enumerate(values):
            finite(value, f"{prefix}.{field}[{index}]", positive=True)
    median_fields = {
        "baseline_before_median_ms": "baseline_before_samples_ms",
        "candidate_median_ms": "candidate_samples_ms",
        "baseline_after_median_ms": "baseline_after_samples_ms",
    }
    for field, samples_field in median_fields.items():
        median = finite(pair.get(field), f"{prefix}.{field}", positive=True)
        require(median == sorted(pair[samples_field])[len(pair[samples_field]) // 2], f"{prefix}: invalid {field}")


def validate_cases(receipt: dict[str, Any], phase: str) -> None:
    data = manifest()
    expected = expected_cases(phase)
    records = receipt.get("cases")
    require(isinstance(records, list) and records, "receipt has no cases")
    require(all(isinstance(case, dict) for case in records), "receipt case is not an object")
    require({case.get("name") for case in records} == set(expected), "receipt cases differ from manifest")
    require(len(records) == len(expected), "receipt has duplicate cases")
    repetitions = data["measured_runs"]
    for case in records:
        validate_case(case, expected, data["paths"], repetitions)


def validate_case(case: dict[str, Any], expected: dict[str, dict[str, Any]], paths: list[str], repetitions: int) -> None:
    name = case.get("name")
    require(isinstance(name, str) and name in expected, "receipt case is not in manifest")
    validate_spec(case, expected[name])
    require(isinstance(case.get("input_digest"), str) and DIGEST16.fullmatch(case["input_digest"]), f"{name}: invalid input digest")
    require(isinstance(case.get("oracle_digest"), str) and DIGEST16.fullmatch(case["oracle_digest"]), f"{name}: invalid oracle digest")
    require(isinstance(case.get("backend_oracle_digest"), str) and DIGEST16.fullmatch(case["backend_oracle_digest"]), f"{name}: invalid backend oracle digest")
    expected_input, expected_oracle = case_digests(expected[name])
    require(case["input_digest"] == expected_input, f"{name}: input digest differs from common oracle")
    require(case["oracle_digest"] == expected_oracle, f"{name}: oracle digest differs from common oracle")
    for field, message in (
        ("partial_final_tile", "missing partial tile control"),
        ("missing_tile_rejected", "missing tile control failed"),
        ("interior_partial_tile_rejected", "partial tile control failed"),
        ("scheduled_tile_rejected", "scheduled tile control failed"),
    ):
        require(case.get(field) is True, f"{name}: {message}")
    validate_case_paths(case.get("paths"), paths, repetitions, name, expected[name])
    validate_timing_pair(case.get("timing_pair"), repetitions, f"{name}/timing_pair")


def validate_case_paths(
    records: Any, expected_paths: list[str], repetitions: int, name: str,
    spec: dict[str, Any]
) -> None:
    require(isinstance(records, list), f"{name}: paths are not a list")
    require(all(isinstance(path, dict) for path in records), f"{name}: path is not an object")
    require([path.get("path") for path in records] == expected_paths, f"{name}: paths differ from manifest")
    require(len(records) == len(expected_paths), f"{name}: duplicate paths")
    expected = case_oracle_values(spec)
    for path in records:
        validate_path(path, path["path"], repetitions, f"{name}/{path['path']}", spec, expected)


def validate_receipt(path: Path, phase: str, backend: str) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    validate_identity(receipt, phase, backend)
    validate_provenance(receipt, backend)
    validate_cases(receipt, phase)
    return receipt


def percentile95(values: list[float]) -> float:
    require(bool(values), "cannot calculate a percentile from no samples")
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def quality_bounds(receipt: dict[str, Any]) -> dict[str, dict[str, float]]:
    bounds: dict[str, dict[str, float]] = {}
    for case in receipt["cases"]:
        for path in case["paths"]:
            current = bounds.setdefault(path["path"], {"absolute": 0.0, "relative": 0.0})
            current["absolute"] = max(current["absolute"], path["quality_max_abs"])
            current["relative"] = max(current["relative"], path["quality_max_rel"])
    return {
        name: {
            "max_absolute_error": math.nextafter(values["absolute"], math.inf),
            "max_relative_error": math.nextafter(values["relative"], math.inf),
        }
        for name, values in sorted(bounds.items())
    }


def timing_case(case: dict[str, Any]) -> tuple[dict[str, Any], list[float]]:
    pair = case["timing_pair"]
    split_noise = []
    candidate_ratios = []
    for before, candidate, after in zip(
        pair["baseline_before_samples_ms"],
        pair["candidate_samples_ms"],
        pair["baseline_after_samples_ms"],
    ):
        baseline = (before + after) / 2.0
        split_noise.append(abs(after - before) / baseline)
        candidate_ratios.append(candidate / baseline)
    baseline_median = (
        pair["baseline_before_median_ms"] + pair["baseline_after_median_ms"]
    ) / 2.0
    return (
        {
            "baseline_median_ms": baseline_median,
            "candidate_median_ms": pair["candidate_median_ms"],
            "candidate_to_baseline_ratio": pair["candidate_median_ms"] / baseline_median,
            "candidate_relative_effect": 1.0 - pair["candidate_median_ms"] / baseline_median,
            "split_half_noise_95": percentile95(split_noise),
            "candidate_ratio_samples": candidate_ratios,
            "split_half_noise_samples": split_noise,
        },
        split_noise,
    )


def timing_criteria(receipt: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float]:
    comparisons: dict[str, dict[str, Any]] = {}
    all_noise: list[float] = []
    for case in receipt["cases"]:
        comparison, noise = timing_case(case)
        comparisons[case["name"]] = comparison
        all_noise.extend(noise)
    require(any(value > 0.0 for value in all_noise), "calibration has no timing variation")
    return comparisons, math.nextafter(percentile95(all_noise), math.inf)


def validate_criteria(path: Path, calibration_path: Path, backend: str) -> dict[str, Any]:
    calibration = validate_receipt(calibration_path, "calibration", backend)
    criteria = json.loads(path.read_text())
    require(criteria.get("schema") == CRITERIA_SCHEMA, "wrong criteria schema")
    require(criteria.get("operator") == OPERATOR, "wrong criteria operator")
    require(criteria.get("backend") == backend, "wrong criteria backend")
    require(criteria.get("repetitions") == manifest()["measured_runs"], "criteria repetitions differ")
    require(
        criteria.get("calibration_receipt_sha256") == sha256(calibration_path),
        "criteria calibration hash differs",
    )
    require(criteria.get("quality_rule") == QUALITY_RULE, "quality rule differs")
    require(criteria.get("timing_rule") == TIMING_RULE, "timing rule differs")
    expected_quality = quality_bounds(calibration)
    require(criteria.get("quality") == expected_quality, "quality bounds differ from calibration")
    expected_timing, expected_threshold = timing_criteria(calibration)
    require(criteria.get("paired_comparison") == expected_timing, "timing comparisons differ from calibration")
    require(criteria.get("minimum_reviewable_relative_effect") == expected_threshold, "timing threshold differs from calibration")
    require(criteria.get("manifest_sha256") == sha256(MANIFEST_PATH), "criteria manifest hash differs")
    require(criteria.get("device") == calibration.get("device"), "criteria device differs")
    calibration_provenance = calibration.get("provenance", {})
    require(
        criteria.get("source_layout", LEGACY_SOURCE_LAYOUT)
        == calibration_provenance.get("source_layout", LEGACY_SOURCE_LAYOUT),
        "criteria source layout differs",
    )
    require(criteria.get("artifact_sha256") == calibration_provenance.get("artifact_sha256"), "criteria artifact differs")
    require(criteria.get("source_sha256") == calibration_provenance.get("source_sha256"), "criteria sources differ")
    require(criteria.get("build") == calibration_provenance.get("build"), "criteria build differs")
    require(criteria.get("git_revision") == calibration_provenance.get("git_revision"), "criteria revision differs")
    require(criteria.get("working_tree_dirty") == calibration_provenance.get("working_tree_dirty"), "criteria dirty state differs")
    return criteria
