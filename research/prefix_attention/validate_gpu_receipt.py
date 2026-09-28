#!/usr/bin/env python3
"""Validate an evaluation receipt against one frozen calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_receipt import (
    FIXED_SCHEDULE_PATHS,
    finite,
    finite_value,
    require,
    sha256,
    validate_criteria,
    validate_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("criteria", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "metal"), required=True)
    parser.add_argument(
        "--require-positive",
        action="store_true",
        help="fail unless every paired timing comparison clears the frozen threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_receipt(args.calibration, "calibration", args.backend)
    criteria = validate_criteria(args.criteria, args.calibration, args.backend)
    evaluation = validate_receipt(args.receipt, "evaluation", args.backend)
    if evaluation.get("criteria_sha256") != sha256(args.criteria):
        raise ValueError("evaluation criteria hash differs from frozen criteria")
    require(evaluation["device"] == criteria["device"], "evaluation device differs from calibration")
    require(evaluation["provenance"]["artifact_sha256"] == criteria["artifact_sha256"], "evaluation artifact differs")
    require(
        evaluation["provenance"].get("source_layout", "legacy")
        == criteria.get("source_layout", "legacy"),
        "evaluation source layout differs",
    )
    require(evaluation["provenance"]["source_sha256"] == criteria["source_sha256"], "evaluation sources differ")
    require(evaluation["provenance"]["build"] == criteria["build"], "evaluation build differs")
    require(evaluation["provenance"]["git_revision"] == criteria["git_revision"], "evaluation revision differs")
    require(evaluation["provenance"]["working_tree_dirty"] == criteria["working_tree_dirty"], "evaluation dirty state differs")
    apply_quality(evaluation, criteria)
    positive = report_timing(evaluation, criteria)
    if positive:
        print("evaluation receipt passed frozen identity and quality controls; timing cleared the review threshold")
    else:
        print("evaluation receipt passed frozen identity and quality controls; timing did not clear the review threshold")
    if args.require_positive and not positive:
        return 1
    return 0


def apply_quality(receipt: dict, criteria: dict) -> None:
    bounds = criteria["quality"]
    for case in receipt["cases"]:
        for path in case["paths"]:
            name = path["path"]
            limit = bounds[name]
            finite(path["quality_max_abs"], f"{case['name']}/{name} quality")
            require(path["quality_max_abs"] <= limit["max_absolute_error"], f"{case['name']}/{name}: absolute error exceeded")
            require(path["quality_max_rel"] <= limit["max_relative_error"], f"{case['name']}/{name}: relative error exceeded")
            if name in FIXED_SCHEDULE_PATHS:
                require(path["schedule_b_bitwise_equal"], f"{case['name']}/{name}: fixed schedule changed output")
                finite(path["schedule_b_quality_max_abs"], f"{case['name']}/{name} schedule quality")
                finite(path["schedule_b_quality_max_rel"], f"{case['name']}/{name} schedule relative quality")
                require(path["schedule_b_quality_max_abs"] <= limit["max_absolute_error"], f"{case['name']}/{name}: schedule absolute error exceeded")
                require(path["schedule_b_quality_max_rel"] <= limit["max_relative_error"], f"{case['name']}/{name}: schedule relative error exceeded")


def report_timing(receipt: dict, criteria: dict) -> bool:
    threshold = criteria["minimum_reviewable_relative_effect"]
    results = []
    for case in receipt["cases"]:
        pair = case["timing_pair"]
        baseline = (pair["baseline_before_median_ms"] + pair["baseline_after_median_ms"]) / 2.0
        effect = 1.0 - pair["candidate_median_ms"] / baseline
        finite_value(effect, f"{case['name']} timing effect")
        results.append({
            "case": case["name"],
            "relative_effect": effect,
            "threshold": threshold,
            "clears_threshold": effect >= threshold,
        })
    positive = all(item["clears_threshold"] for item in results)
    print(json.dumps({"timing_effects": results, "positive_result": positive}))
    return positive


if __name__ == "__main__":
    raise SystemExit(main())
