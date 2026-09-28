#!/usr/bin/env python3
"""Freeze quality and paired timing criteria from one calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_receipt import (
    QUALITY_RULE,
    TIMING_RULE,
    manifest,
    quality_bounds,
    sha256,
    timing_criteria,
    validate_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--backend", choices=("cuda", "metal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = validate_receipt(args.calibration, "calibration", args.backend)
    comparisons, effect_threshold = timing_criteria(receipt)
    data = manifest()
    criteria = {
        "schema": "prefix-attention-gpu-criteria-v1",
        "backend": args.backend,
        "operator": receipt["operator"],
        "calibration_receipt_sha256": sha256(args.calibration),
        "manifest_sha256": receipt["provenance"]["manifest_sha256"],
        "source_layout": receipt["provenance"].get("source_layout", "legacy"),
        "device": receipt["device"],
        "artifact_sha256": receipt["provenance"]["artifact_sha256"],
        "source_sha256": receipt["provenance"]["source_sha256"],
        "build": receipt["provenance"]["build"],
        "git_revision": receipt["provenance"]["git_revision"],
        "working_tree_dirty": receipt["provenance"]["working_tree_dirty"],
        "repetitions": data["measured_runs"],
        "quality_rule": QUALITY_RULE,
        "quality": quality_bounds(receipt),
        "timing_rule": TIMING_RULE,
        "paired_comparison": comparisons,
        "minimum_reviewable_relative_effect": effect_threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(criteria, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
