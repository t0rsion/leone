"""Check the opt-in production differential receipt contract."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_oracle import f32, output_digest
from gpu_receipt import manifest, sha256
from production_oracle import (
    canonicalize_case,
    case_digests,
    case_oracle_values,
    max_absolute_error,
    max_relative_error,
)
from production_receipt import (
    PROVENANCE_SCOPE,
    QUALITY_RULE,
    SOURCE_NAMES,
    quality_bounds,
    validate_criteria,
    validate_receipt,
)
from runner_support import SOURCE_LAYOUT


def production_receipt(phase: str = "calibration") -> dict:
    data = manifest()
    records = []
    for case in data[phase]:
        fields = (
            "tokens", "query_rows", "query_heads", "kv_heads", "head_dim",
            "tile_tokens", "group_rows", "seed", "shared_prefix",
        )
        samples = [1.0 + index * 0.1 for index in range(9)]
        input_digest, oracle_digest = case_digests(case)
        oracle = case_oracle_values(case)
        output = [f32(value) for value in oracle]
        records.append({
            "name": case["name"],
            "spec": {field: case[field] for field in fields},
            "production_query_rows": case["query_rows"],
            "layout_scope": "causal_single_cache",
            "input_digest": input_digest,
            "oracle_digest": oracle_digest,
            "backend_oracle_digest": "4" * 16,
            "output_digest": output_digest(output),
            "output_values": output,
            "samples_ms": samples,
            "median_ms": samples[4],
            "quality_max_abs": max_absolute_error(output, oracle),
            "quality_max_rel": max_relative_error(output, oracle),
            "backend_quality_max_abs": 0.001,
            "backend_quality_max_rel": 0.002,
        })
    sources = {name: "a" * 64 for name in SOURCE_NAMES}
    return {
        "schema": "prefix-attention-production-receipt-v1",
        "operator": "leone-production-prefill-attention-f16",
        "storage": "fp16-kv-fp32-accumulation",
        "backend": "cuda",
        "phase": phase,
        "device": {"name": "fixture", "compute_capability": "8.9"},
        "input_generator": data["input_generator"],
        "input_digest_algorithm": data["input_digest"],
        "oracle_id": data["oracle_id"],
        "warmups": data["warmup_runs"],
        "repetitions": data["measured_runs"],
        "cases": records,
        "provenance": {
            "source_layout": SOURCE_LAYOUT,
            "scope": PROVENANCE_SCOPE,
            "manifest_sha256": sha256(Path(__file__).with_name("gpu_manifest.json")),
            "git_revision": "fixture",
            "working_tree_dirty": False,
            "source_sha256": sources,
            "artifact_sha256": "c" * 64,
            "build": {"command": ["fixture"], "rustc_version": "fixture"},
            "host_system": "fixture-host",
            "command": ["fixture"],
        },
    }


def production_criteria(calibration_path: Path, calibration: dict) -> dict:
    return {
        "schema": "prefix-attention-production-criteria-v1",
        "operator": "leone-production-prefill-attention-f16",
        "backend": "cuda",
        "calibration_receipt_sha256": sha256(calibration_path),
        "manifest_sha256": calibration["provenance"]["manifest_sha256"],
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
    }


class ProductionReceiptTests(unittest.TestCase):
    def test_wrapper_keeps_backend_quality_and_recomputes_canonical_quality(self) -> None:
        value = production_receipt()
        case = value["cases"][0]
        case["quality_max_abs"] = 0.75
        case["quality_max_rel"] = 0.5
        canonicalize_case(case)
        self.assertEqual(case["backend_quality_max_abs"], 0.75)
        self.assertEqual(case["backend_quality_max_rel"], 0.5)
        expected = case_oracle_values(case["spec"])
        self.assertEqual(case["quality_max_abs"], max_absolute_error(case["output_values"], expected))
        self.assertEqual(case["quality_max_rel"], max_relative_error(case["output_values"], expected))

    def test_valid_receipt_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(production_receipt()))
            self.assertTrue(validate_receipt(path, "calibration")["cases"])

    def test_source_layout_preserves_legacy_and_rejects_incomplete_new_sets(self) -> None:
        legacy = production_receipt()
        legacy["provenance"].pop("source_layout")
        legacy["provenance"]["source_sha256"].pop("research/prefix_attention/runner_support.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps(legacy))
            self.assertTrue(validate_receipt(path, "calibration")["cases"])

        mutations = (
            lambda value: value["provenance"]["source_sha256"].pop("research/prefix_attention/runner_support.py"),
            lambda value: value["provenance"]["source_sha256"].update({
                "research/prefix_attention/foreign.py": "a" * 64,
            }),
            lambda value: value["provenance"].update(source_layout="future"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value = production_receipt()
                mutation(value)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "invalid.json"
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        validate_receipt(path, "calibration")

    def test_output_values_are_bound_to_canonical_quality(self) -> None:
        value = production_receipt()
        case = value["cases"][0]
        case["output_values"][0] = f32(case["output_values"][0] + 0.125)
        case["output_digest"] = output_digest(case["output_values"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_receipt(path, "calibration")

    def test_valid_criteria_and_identity_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(production_receipt()))
            calibration = validate_receipt(calibration_path, "calibration")
            criteria_path = root / "criteria.json"
            criteria_path.write_text(json.dumps(production_criteria(calibration_path, calibration)))
            self.assertTrue(validate_criteria(criteria_path, calibration_path)["quality"])
            value = json.loads(criteria_path.read_text())
            value["repetitions"] = 0
            criteria_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_criteria(criteria_path, calibration_path)
            value["repetitions"] = manifest()["measured_runs"]
            value["source_layout"] = "legacy"
            with self.assertRaises(ValueError):
                validate_criteria(criteria_path, calibration_path)
            value["source_layout"] = SOURCE_LAYOUT
            value["backend"] = "metal"
            criteria_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_criteria(criteria_path, calibration_path)
            value["backend"] = "cuda"
            value["quality"]["max_absolute_error"] = 1e99
            criteria_path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                validate_criteria(criteria_path, calibration_path)

    def test_empty_cases_and_negative_quality_fail(self) -> None:
        for mutate in (
            self.empty_cases,
            self.negative_quality,
            self.wrong_input_digest,
            self.wrong_output_digest,
            self.wrong_provenance_scope,
        ):
            with self.subTest(mutate=mutate):
                value = production_receipt()
                mutate(value)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "receipt.json"
                    path.write_text(json.dumps(value))
                    with self.assertRaises(ValueError):
                        validate_receipt(path, "calibration")

    @staticmethod
    def empty_cases(value: dict) -> None:
        value["cases"] = []

    @staticmethod
    def negative_quality(value: dict) -> None:
        value["cases"][0]["quality_max_abs"] = -1.0

    @staticmethod
    def wrong_input_digest(value: dict) -> None:
        value["cases"][0]["input_digest"] = "f" * 16

    @staticmethod
    def wrong_output_digest(value: dict) -> None:
        value["cases"][0]["output_digest"] = "f" * 16

    @staticmethod
    def wrong_provenance_scope(value: dict) -> None:
        value["provenance"]["scope"] = "unattested"


if __name__ == "__main__":
    unittest.main()
