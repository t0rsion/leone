#!/usr/bin/env python3
"""Freeze Runtime shared-prefix criteria from four calibration receipts.

The output records content hashes of the receipts, the source workload, the
tools, the BF16 oracles, and an optional external GPU sample record. Quality
limits come from the BF16 calibration rows. It carries no evaluation result.
The command refuses to replace an existing criteria file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from runner_support import publish_json
from validate_runtime import (
    add_frozen_inputs,
    check_frozen_inputs,
    derive_criteria,
    load_calibration,
    load_oracles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "metal"), required=True)
    parser.add_argument("--calibration", type=Path, nargs="+", required=True,
                        help="one calibration receipt for each of the four paths")
    add_frozen_inputs(parser)
    parser.add_argument("--quality-unverified", action="store_true",
                        help="freeze without BF16 oracles. The report then prints quality: unverified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_frozen_inputs(parser, args)
    if not args.bf16_calibration and not args.quality_unverified:
        parser.error("pass the BF16 oracle files, or pass --quality-unverified to freeze without them")
    if args.bf16_calibration and args.quality_unverified:
        parser.error("--quality-unverified contradicts the BF16 oracle files")
    return args


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to replace existing criteria: {args.output}")
    try:
        study, manifest, source = load_calibration(args.calibration, args.backend)
        criteria = derive_criteria(study, manifest, source, load_oracles(args), args.monitoring)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"criteria not frozen: {error}") from error
    criteria["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_json(args.output, criteria)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
