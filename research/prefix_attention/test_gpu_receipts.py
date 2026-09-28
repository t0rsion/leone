"""Check that malformed GPU records cannot pass the evidence validators."""

from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_oracle import (
    canonicalize_receipt,
    case_digests,
    case_oracle_values,
    f32,
    max_absolute_error,
    max_relative_error,
    output_digest,
)
from freeze_gpu_criteria import quality_bounds, timing_criteria
from gpu_receipt import (
    CRITERIA_SCHEMA,
    MANIFEST_PATH,
    OPERATOR,
    PROVENANCE_SCOPE,
    QUALITY_RULE,
    SCHEMA,
    TIMING_RULE,
    UNAVAILABLE_EVIDENCE,
    expected_path_estimates,
    manifest,
    sha256,
    validate_criteria,
    validate_receipt,
)
from validate_gpu_receipt import apply_quality, report_timing
from runner_support import SOURCE_LAYOUT


def valid_provenance(backend: str) -> dict:
    names = {
        "cuda": [
            "research/prefix_attention/cuda_fixed_reduction/operator.cuh",
            "research/prefix_attention/cuda_fixed_reduction/operator.cu",
            "research/prefix_attention/cuda_fixed_reduction/main.cu",
            "research/prefix_attention/cuda_fixed_reduction/manifest_cases.h",
            "research/prefix_attention/cuda_fixed_reduction/build_cuda.sh",
            "research/prefix_attention/generate_manifest.py",
            "research/prefix_attention/common_oracle.py",
            "research/prefix_attention/runner_support.py",
            "research/prefix_attention/ORACLE_CONTRACT.md",
            "research/prefix_attention/gpu_manifest.json",
            "research/prefix_attention/run_cuda.py",
        ],
        "metal": [
            "research/prefix_attention/metal_fixed_reduction/PrefixAttention.metal",
            "research/prefix_attention/metal_fixed_reduction/RunPrefixAttention.swift",
            "research/prefix_attention/metal_fixed_reduction/build_metal.sh",
            "research/prefix_attention/generate_manifest.py",
            "research/prefix_attention/common_oracle.py",
            "research/prefix_attention/runner_support.py",
            "research/prefix_attention/ORACLE_CONTRACT.md",
            "research/prefix_attention/gpu_manifest.json",
            "research/prefix_attention/run_metal.py",
        ],
    }[backend]
    artifact = "b" * 64 if backend == "cuda" else {"runner": "b" * 64, "metallib": "c" * 64}
    build = {"command": ["fixture-build"]}
    if backend == "cuda":
        build.update({"nvcc": "nvcc", "cuda_arch": "sm_89", "nvcc_version": "fixture"})
    else:
        build.update({"xcrun_version": "fixture", "swiftc_version": "fixture"})
    return {
        "source_layout": SOURCE_LAYOUT,
        "scope": PROVENANCE_SCOPE,
        "git_revision": "fixture-revision",
        "working_tree_dirty": False,
        "manifest_sha256": sha256(MANIFEST_PATH),
        "source_sha256": {name: "a" * 64 for name in names},
        "artifact_sha256": artifact,
        "build": build,
        "host_system": "fixture-host",
        "command": ["fixture-run"],
    }


def timing_pair() -> dict:
    before = [1.0 + index * 0.01 for index in range(9)]
    candidate = [0.8 + index * 0.01 for index in range(9)]
    after = [1.02 + index * 0.01 for index in range(9)]
    return {
        "baseline_path": "fixed_tile_per_row",
        "candidate_path": "shared_read_fixed_reduction",
        "acquisition_order": ["baseline_before", "candidate", "baseline_after"],
        "baseline_before_samples_ms": before,
        "candidate_samples_ms": candidate,
        "baseline_after_samples_ms": after,
        "baseline_before_median_ms": before[4],
        "candidate_median_ms": candidate[4],
        "baseline_after_median_ms": after[4],
    }


def path_record(name: str, case: dict) -> dict:
    samples = [1.0 + index * 0.1 for index in range(9)]
    oracle = case_oracle_values(case)
    output = [f32(value) for value in oracle]
    estimates = expected_path_estimates(case, name)
    absolute = max_absolute_error(output, oracle)
    relative = max_relative_error(output, oracle)
    schedule_output = list(output)
    if name == "shared_read_unconstrained":
        schedule_output[0] = f32(schedule_output[0] + 0.125)
    schedule_absolute = max_absolute_error(schedule_output, oracle)
    schedule_relative = max_relative_error(schedule_output, oracle)
    return {
        "path": name,
        "samples_ms": samples,
        "median_ms": samples[4],
        "quality_max_abs": absolute,
        "quality_max_rel": relative,
        "schedule_b_quality_max_abs": schedule_absolute,
        "schedule_b_quality_max_rel": schedule_relative,
        "backend_quality_max_abs": 0.001,
        "backend_quality_max_rel": 0.002,
        "backend_schedule_b_quality_max_abs": 0.001,
        "backend_schedule_b_quality_max_rel": 0.002,
        "output_values": output,
        "schedule_b_output_values": schedule_output,
        "digest": output_digest(output),
        "schedule_b_digest": output_digest(schedule_output),
        "schedule_b_bitwise_equal": name != "shared_read_unconstrained",
        **estimates,
    }


def case_record(case: dict) -> dict:
    fields = (
        "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
        "tile_tokens", "group_rows", "seed", "shared_prefix",
    )
    input_digest, oracle_digest = case_digests(case)
    return {
        "name": case["name"],
        "spec": {field: case[field] for field in fields},
        "input_digest": input_digest,
        "oracle_digest": oracle_digest,
        "backend_oracle_digest": "5" * 16,
        "partial_final_tile": True,
        "missing_tile_rejected": True,
        "interior_partial_tile_rejected": True,
        "scheduled_tile_rejected": True,
        "paths": [path_record(name, case) for name in manifest()["paths"]],
        "timing_pair": timing_pair(),
    }


def receipt(phase: str = "calibration", backend: str = "cuda") -> dict:
    data = manifest()
    cases = [case_record(case) for case in data[phase]]
    return {
        "schema": SCHEMA,
        "operator": OPERATOR,
        "storage": "fp16-kv-fp32-accumulation",
        "backend": backend,
        "phase": phase,
        "device": {"name": "fixture-device", "compute_major": 8}
        if backend == "cuda" else {"name": "fixture-device", "registry_id": 1},
        "input_generator": data["input_generator"],
        "input_digest_algorithm": data["input_digest"],
        "oracle_id": data["oracle_id"],
        "unavailable_evidence": UNAVAILABLE_EVIDENCE,
        "claims": [],
        "warmups": data["warmup_runs"],
        "repetitions": data["measured_runs"],
        "cases": cases,
        "provenance": valid_provenance(backend),
    }


def write_json(directory: Path, name: str, value: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(value, allow_nan=True))
    return path


class ReceiptValidationTests(unittest.TestCase):
    def test_wrapper_keeps_backend_quality_and_recomputes_canonical_quality(self) -> None:
        value = receipt()
        path = value["cases"][0]["paths"][0]
        path["quality_max_abs"] = 0.75
        path["quality_max_rel"] = 0.5
        canonicalize_receipt(value)
        self.assertEqual(path["backend_quality_max_abs"], 0.75)
        self.assertEqual(path["backend_quality_max_rel"], 0.5)
        expected = case_oracle_values(value["cases"][0]["spec"])
        self.assertEqual(path["quality_max_abs"], max_absolute_error(path["output_values"], expected))
        self.assertEqual(path["quality_max_rel"], max_relative_error(path["output_values"], expected))

    def test_valid_calibration_and_criteria_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = write_json(root, "calibration.json", receipt())
            calibration = validate_receipt(calibration_path, "calibration", "cuda")
            comparisons, threshold = timing_criteria(calibration)
            criteria = {
                "schema": CRITERIA_SCHEMA,
                "operator": OPERATOR,
                "backend": "cuda",
                "calibration_receipt_sha256": sha256(calibration_path),
                "manifest_sha256": sha256(MANIFEST_PATH),
                "source_layout": calibration["provenance"]["source_layout"],
                "device": calibration["device"],
                "artifact_sha256": calibration["provenance"]["artifact_sha256"],
                "source_sha256": calibration["provenance"]["source_sha256"],
                "build": calibration["provenance"]["build"],
                "git_revision": calibration["provenance"]["git_revision"],
                "working_tree_dirty": calibration["provenance"]["working_tree_dirty"],
                "repetitions": manifest()["measured_runs"],
                "quality_rule": QUALITY_RULE,
                "quality": quality_bounds(calibration),
                "timing_rule": TIMING_RULE,
                "paired_comparison": comparisons,
                "minimum_reviewable_relative_effect": threshold,
            }
            criteria_path = write_json(root, "criteria.json", criteria)
            self.assertEqual(validate_criteria(criteria_path, calibration_path, "cuda"), criteria)

    def test_estimates_match_independent_frozen_cases(self) -> None:
        data = manifest()
        shared = data["calibration"][0]
        unrelated = data["calibration"][2]
        for path_name in ("per_row", "fixed_tile_per_row"):
            self.assertEqual(
                expected_path_estimates(shared, path_name),
                {
                    "estimated_kv_elements_read": 17408,
                    "estimated_tile_loads": 48,
                    "output_elements_written": 512,
                    "per_block_shared_bytes": 0,
                },
            )
        self.assertEqual(
            expected_path_estimates(shared, "shared_read_fixed_reduction"),
            {
                "estimated_kv_elements_read": 8704,
                "estimated_tile_loads": 24,
                "output_elements_written": 512,
                "per_block_shared_bytes": 1024,
            },
        )
        self.assertEqual(
            expected_path_estimates(unrelated, "shared_read_unconstrained"),
            {
                "estimated_kv_elements_read": 66560,
                "estimated_tile_loads": 80,
                "output_elements_written": 512,
                "per_block_shared_bytes": 2048,
            },
        )

    def test_unconstrained_schedule_difference_is_allowed(self) -> None:
        value = receipt()
        path = value["cases"][0]["paths"][2]
        self.assertFalse(path["schedule_b_bitwise_equal"])
        with tempfile.TemporaryDirectory() as directory:
            path_file = write_json(Path(directory), "receipt.json", value)
            validate_receipt(path_file, "calibration", "cuda")

    def test_fixed_schedule_difference_is_rejected_after_digest_update(self) -> None:
        value = receipt()
        path = value["cases"][0]["paths"][3]
        path["schedule_b_output_values"][0] = f32(path["schedule_b_output_values"][0] + 0.125)
        path["schedule_b_digest"] = output_digest(path["schedule_b_output_values"])
        path["schedule_b_quality_max_abs"] = max_absolute_error(
            path["schedule_b_output_values"], case_oracle_values(value["cases"][0]["spec"])
        )
        path["schedule_b_quality_max_rel"] = max_relative_error(
            path["schedule_b_output_values"], case_oracle_values(value["cases"][0]["spec"])
        )
        with tempfile.TemporaryDirectory() as directory:
            path_file = write_json(Path(directory), "receipt.json", value)
            with self.assertRaises(ValueError):
                validate_receipt(path_file, "calibration", "cuda")

    def test_fixed_schedule_false_flag_is_rejected(self) -> None:
        value = receipt()
        value["cases"][0]["paths"][3]["schedule_b_bitwise_equal"] = False
        with tempfile.TemporaryDirectory() as directory:
            path_file = write_json(Path(directory), "receipt.json", value)
            with self.assertRaises(ValueError):
                validate_receipt(path_file, "calibration", "cuda")

    def test_non_fp32_output_is_rejected(self) -> None:
        value = receipt()
        path = value["cases"][0]["paths"][0]
        path["output_values"][0] = 0.1
        path["digest"] = output_digest(path["output_values"])
        with tempfile.TemporaryDirectory() as directory:
            path_file = write_json(Path(directory), "receipt.json", value)
            with self.assertRaises(ValueError):
                validate_receipt(path_file, "calibration", "cuda")

    def test_metal_uses_the_same_common_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(Path(directory), "metal.json", receipt(backend="metal"))
            self.assertTrue(validate_receipt(path, "calibration", "metal")["cases"])

    def test_source_layout_preserves_legacy_and_rejects_incomplete_new_sets(self) -> None:
        legacy = receipt()
        legacy["provenance"].pop("source_layout")
        legacy["provenance"]["source_sha256"].pop("research/prefix_attention/runner_support.py")
        with tempfile.TemporaryDirectory() as directory:
            path = write_json(Path(directory), "legacy.json", legacy)
            self.assertTrue(validate_receipt(path, "calibration", "cuda")["cases"])

        mutations = (
            lambda value: value["provenance"]["source_sha256"].pop("research/prefix_attention/runner_support.py"),
            lambda value: value["provenance"]["source_sha256"].update({
                "research/prefix_attention/foreign.py": "a" * 64,
            }),
            lambda value: value["provenance"].update(source_layout="future"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejects(mutation)

    def test_evaluation_quality_and_schedule_controls_pass(self) -> None:
        calibration = receipt()
        evaluation = receipt("evaluation")
        quality = quality_bounds(calibration)
        for case in evaluation["cases"]:
            for path in case["paths"]:
                bound = quality[path["path"]]
                bound["max_absolute_error"] = max(
                    bound["max_absolute_error"],
                    path["quality_max_abs"],
                )
                bound["max_relative_error"] = max(
                    bound["max_relative_error"],
                    path["quality_max_rel"],
                )
        criteria = {
            "quality": quality,
            "minimum_reviewable_relative_effect": timing_criteria(calibration)[1],
        }
        apply_quality(evaluation, criteria)
        with redirect_stdout(io.StringIO()):
            self.assertIsInstance(report_timing(evaluation, criteria), bool)
        evaluation["cases"][0]["paths"][3]["schedule_b_bitwise_equal"] = False
        with self.assertRaises(ValueError):
            apply_quality(evaluation, criteria)

    def test_slower_candidate_is_a_negative_timing_result(self) -> None:
        evaluation = {"cases": [{
            "name": "slower",
            "timing_pair": {
                "baseline_before_median_ms": 1.0,
                "baseline_after_median_ms": 1.0,
                "candidate_median_ms": 2.0,
            },
        }]}
        output = io.StringIO()
        with redirect_stdout(output):
            positive = report_timing(
                evaluation, {"minimum_reviewable_relative_effect": 0.01}
            )
        report = json.loads(output.getvalue())
        self.assertFalse(positive)
        self.assertFalse(report["positive_result"])
        self.assertEqual(report["timing_effects"][0]["relative_effect"], -1.0)
        self.assertFalse(report["timing_effects"][0]["clears_threshold"])

    def test_rejects_empty_cases(self) -> None:
        self.assert_rejects(lambda value: value.update(cases=[]))

    def test_rejects_identity_and_case_mutations(self) -> None:
        mutations = [
            lambda value: value.update(schema="wrong"),
            lambda value: value["provenance"].update(scope="unattested"),
            lambda value: value["cases"][0].update(name="fake"),
            lambda value: value["cases"][0].update(scheduled_tile_rejected=False),
            lambda value: value["cases"][0]["paths"].pop(),
            lambda value: value["cases"][0]["paths"][0].update(quality_max_abs=-1),
            lambda value: value["cases"][0]["paths"][0].update(digest="0" * 16),
            lambda value: value["cases"][0]["paths"][0].update(output_values=[]),
            lambda value: value["cases"][0]["paths"][0].update(estimated_tile_loads=1),
            lambda value: value["cases"][0]["paths"][0].update(samples_ms=[float("nan")] * 9),
            lambda value: value["cases"][0]["timing_pair"].update(candidate_samples_ms=[]),
            lambda value: value["cases"][0]["timing_pair"].update(acquisition_order=["candidate"]),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejects(mutation)

    def test_rejects_criteria_mutations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = write_json(root, "calibration.json", receipt())
            calibration = validate_receipt(calibration_path, "calibration", "cuda")
            comparisons, threshold = timing_criteria(calibration)
            criteria = {
                "schema": CRITERIA_SCHEMA,
                "operator": OPERATOR,
                "backend": "cuda",
                "calibration_receipt_sha256": sha256(calibration_path),
                "manifest_sha256": sha256(MANIFEST_PATH),
                "source_layout": calibration["provenance"]["source_layout"],
                "device": calibration["device"],
                "artifact_sha256": calibration["provenance"]["artifact_sha256"],
                "source_sha256": calibration["provenance"]["source_sha256"],
                "build": calibration["provenance"]["build"],
                "git_revision": calibration["provenance"]["git_revision"],
                "working_tree_dirty": calibration["provenance"]["working_tree_dirty"],
                "repetitions": manifest()["measured_runs"],
                "quality_rule": QUALITY_RULE,
                "quality": quality_bounds(calibration),
                "timing_rule": TIMING_RULE,
                "paired_comparison": comparisons,
                "minimum_reviewable_relative_effect": threshold,
            }
            mutations = [
                ("repetitions", lambda value: value.update(repetitions=0)),
                ("manifest_sha256", lambda value: value.update(manifest_sha256="0" * 64)),
                ("source_layout", lambda value: value.update(source_layout="legacy")),
                ("quality", lambda value: value["quality"]["per_row"].update(max_absolute_error=1e99)),
                ("timing_threshold", lambda value: value.update(minimum_reviewable_relative_effect=1e-99)),
            ]
            for field, mutate in mutations:
                mutated = copy.deepcopy(criteria)
                mutate(mutated)
                path = write_json(root, f"criteria-{field}.json", mutated)
                with self.assertRaises(ValueError):
                    validate_criteria(path, calibration_path, "cuda")

    def assert_rejects(self, mutation) -> None:
        with tempfile.TemporaryDirectory() as directory:
            value = receipt()
            mutation(value)
            path = write_json(Path(directory), "receipt.json", value)
            with self.assertRaises((ValueError, KeyError, IndexError)):
                validate_receipt(path, "calibration", "cuda")


if __name__ == "__main__":
    unittest.main()
