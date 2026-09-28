#!/usr/bin/env python3
"""Validate one opt-in production prefill receipt against frozen quality."""

from __future__ import annotations

import argparse
from pathlib import Path

from gpu_receipt import require
from production_receipt import sha256, validate_criteria, validate_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("criteria", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    criteria = validate_criteria(args.criteria, args.calibration)
    receipt = validate_receipt(args.receipt, "evaluation")
    require(receipt.get("criteria_sha256") == sha256(args.criteria), "production criteria hash differs")
    require(receipt["device"] == criteria["device"], "production evaluation device differs")
    require(
        receipt["provenance"].get("source_layout", "legacy")
        == criteria.get("source_layout", "legacy"),
        "production evaluation source layout differs",
    )
    require(receipt["provenance"]["artifact_sha256"] == criteria["artifact_sha256"], "production evaluation artifact differs")
    require(receipt["provenance"]["source_sha256"] == criteria["source_sha256"], "production evaluation sources differ")
    require(receipt["provenance"]["build"] == criteria["build"], "production evaluation build differs")
    require(receipt["provenance"]["git_revision"] == criteria["git_revision"], "production evaluation revision differs")
    require(receipt["provenance"]["working_tree_dirty"] == criteria["working_tree_dirty"], "production evaluation dirty state differs")
    for case in receipt["cases"]:
        require(case["quality_max_abs"] <= criteria["quality"]["max_absolute_error"], f"{case['name']}: absolute error exceeded")
        require(case["quality_max_rel"] <= criteria["quality"]["max_relative_error"], f"{case['name']}: relative error exceeded")
    print("production prefill receipt passed frozen quality and identity checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
