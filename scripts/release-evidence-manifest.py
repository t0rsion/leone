#!/usr/bin/env python3
"""Validate and stage an exact release evidence manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from release_evidence_manifest import load, select, stage, validate


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage a Leone release evidence manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="select the workspace manifest")
    select_parser.add_argument("--root", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="validate manifest structure")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--complete", action="store_true")

    stage_parser = subparsers.add_parser("stage", help="stage declared evidence files")
    stage_parser.add_argument("--manifest", type=Path, required=True)
    stage_parser.add_argument("--source-root", type=Path, required=True)
    stage_parser.add_argument("--destination", type=Path, required=True)
    stage_parser.add_argument("--binary", type=Path)

    args = parser.parse_args(arguments)
    try:
        if args.command == "select":
            print(select(args.root.resolve()))
        elif args.command == "validate":
            validate(load(args.manifest.resolve()), require_complete=args.complete)
            print(f"validated {args.manifest}")
        else:
            source_root = args.source_root.resolve()
            manifest = load(args.manifest.resolve())
            overrides = {}
            if args.binary is not None:
                overrides = {
                    entry["source"]: args.binary.resolve()
                    for entry in manifest.get("files", [])
                    if isinstance(entry, dict) and entry.get("destination") == "bin/leone"
                }
            stage(
                manifest,
                source_root,
                args.destination.resolve(),
                overrides,
            )
            print(f"staged {args.manifest}")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
