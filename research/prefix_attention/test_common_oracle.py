"""Check that the FP64 oracle stays portable across Python runtimes."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_oracle
import production_oracle

EXPECTED_COMMON_DIGESTS = {
    "calibration_short_gqa": "c33cae80d62ee260",
    "calibration_long_gqa": "f819969f515b2831",
    "calibration_unrelated": "0da5d5a1d8f59a61",
    "evaluation_partial_gqa": "b62b03f787ce0121",
    "evaluation_long_gqa": "a0d662963c0a0b90",
    "evaluation_short_shared": "d07b8c54d4c3258d",
    "evaluation_unrelated": "a7d1df7b35252839",
}

EXPECTED_PRODUCTION_DIGESTS = {
    "calibration_short_gqa": "aac13998809f7c46",
    "calibration_long_gqa": "6819051e57e401a9",
    "calibration_unrelated": "6bad78f7bf35c62c",
    "evaluation_partial_gqa": "8d2682839190f6c9",
    "evaluation_long_gqa": "387f419bc7396743",
    "evaluation_short_shared": "e2cbc19a84ceb84d",
    "evaluation_unrelated": "15d470c4f476f6e6",
}


class PortableOracleTests(unittest.TestCase):
    """Checks fixed arithmetic and manifest oracle digests."""

    def test_accumulation_order_is_explicit(self) -> None:
        values = (1.0e16, 1.0, -1.0e16)
        self.assertEqual(common_oracle.f64_sum(values), 0.0)

    def test_transcendental_vectors_are_fixed(self) -> None:
        self.assertEqual(common_oracle.oracle_exp(-1.0).hex(), "0x1.78b56362cef39p-2")
        self.assertEqual(common_oracle.oracle_exp(0.0), 1.0)
        self.assertEqual(common_oracle.oracle_sqrt(2.0).hex(), "0x1.6a09e667f3bcdp+0")

    def test_transcendental_domains_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            common_oracle.oracle_exp(0.5)
        with self.assertRaises(ValueError):
            common_oracle.oracle_exp(-16.001)
        with self.assertRaises(ValueError):
            common_oracle.oracle_sqrt(0.5)
        with self.assertRaises(ValueError):
            common_oracle.oracle_sqrt(65.0)

    def test_common_manifest_digests(self) -> None:
        manifest = json.loads(Path(__file__).with_name("gpu_manifest.json").read_text())
        self.assertEqual(common_oracle.ORACLE_ID, manifest["oracle_id"])
        self.assertEqual(
            common_oracle.ORACLE_ALGORITHM,
            "explicit-f64-left-to-right-taylor-transcendentals-v1",
        )
        for phase in ("calibration", "evaluation"):
            for case in manifest[phase]:
                self.assertEqual(
                    common_oracle.case_digests(case)[1],
                    EXPECTED_COMMON_DIGESTS[case["name"]],
                )

    def test_production_manifest_digests(self) -> None:
        manifest = json.loads(Path(__file__).with_name("gpu_manifest.json").read_text())
        for phase in ("calibration", "evaluation"):
            for case in manifest[phase]:
                self.assertEqual(
                    production_oracle.case_digests(case)[1],
                    EXPECTED_PRODUCTION_DIGESTS[case["name"]],
                )


if __name__ == "__main__":
    unittest.main()
