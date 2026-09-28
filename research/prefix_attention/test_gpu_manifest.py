"""Check the frozen standalone GPU study manifest without a device."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from generate_manifest import validate_manifest


class GpuManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "gpu_manifest.json").read_text())

    def test_paths_and_case_names_are_frozen(self):
        self.assertEqual(
            self.manifest["paths"],
            [
                "per_row",
                "fixed_tile_per_row",
                "shared_read_unconstrained",
                "shared_read_fixed_reduction",
            ],
        )
        calibration = {case["name"] for case in self.manifest["calibration"]}
        evaluation = {case["name"] for case in self.manifest["evaluation"]}
        self.assertTrue(calibration.isdisjoint(evaluation))

    def test_cases_use_checked_dimensions_and_partial_tiles(self):
        cases = self.manifest["calibration"] + self.manifest["evaluation"]
        self.assertTrue(all(case["query_heads"] % case["kv_heads"] == 0 for case in cases))
        self.assertTrue(all(case["head_dim"] <= self.manifest["maximum_head_dim"] for case in cases))
        self.assertTrue(any(case["tokens"] % case["tile_tokens"] for case in cases))
        self.assertEqual(self.manifest["warmup_runs"], 3)
        self.assertGreater(self.manifest["measured_runs"], 0)

    def test_source_isolation(self):
        self.assertTrue((ROOT / "cuda_fixed_reduction/operator.cu").exists())
        self.assertTrue((ROOT / "metal_fixed_reduction/PrefixAttention.metal").exists())
        self.assertTrue((ROOT / "cuda_fixed_reduction/operator.cu").is_relative_to(ROOT))
        self.assertTrue((ROOT / "metal_fixed_reduction/PrefixAttention.metal").is_relative_to(ROOT))

    def test_manifest_bounds_reject_oversized_case(self):
        value = copy.deepcopy(self.manifest)
        value["calibration"][0]["head_dim"] = value["maximum_head_dim"] + 1
        with self.assertRaises(ValueError):
            validate_manifest(value)

    def test_manifest_identity_rejects_unsupported_protocols(self):
        for field, replacement in (
            ("storage", "f32"),
            ("input_generator", "other-generator"),
            ("input_digest", "other-digest"),
            ("oracle_id", "other-oracle"),
        ):
            with self.subTest(field=field):
                value = copy.deepcopy(self.manifest)
                value[field] = replacement
                with self.assertRaises(ValueError):
                    validate_manifest(value)


if __name__ == "__main__":
    unittest.main()
