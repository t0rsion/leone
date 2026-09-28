#!/usr/bin/env python3
"""Freeze production prefill quality bounds from one calibration receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_receipt import manifest
from production_receipt import (
    OPERATOR,
    QUALITY_RULE,
    quality_bounds,
    sha256,
    validate_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calibration = validate_receipt(args.calibration, "calibration")
    provenance = calibration["provenance"]
    criteria = {
        "schema": "prefix-attention-production-criteria-v1",
        "operator": OPERATOR,
        "backend": "cuda",
        "calibration_receipt_sha256": sha256(args.calibration),
        "manifest_sha256": provenance["manifest_sha256"],
        "source_layout": provenance.get("source_layout", "legacy"),
        "device": calibration["device"],
        "artifact_sha256": provenance["artifact_sha256"],
        "source_sha256": provenance["source_sha256"],
        "build": provenance["build"],
        "git_revision": provenance["git_revision"],
        "working_tree_dirty": provenance["working_tree_dirty"],
        "repetitions": manifest()["measured_runs"],
        "quality_rule": QUALITY_RULE,
        "quality": quality_bounds(calibration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(criteria, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
