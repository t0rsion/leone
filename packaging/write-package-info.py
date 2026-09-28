#!/usr/bin/env python3
"""Write deterministic package and build environment records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_records(args: argparse.Namespace) -> None:
    """Write package metadata with only declared reproducibility inputs."""
    features = [args.backend]
    package = {
        "schema_version": "leone.package.v1",
        "package": args.package_kind,
        "version": args.version,
        "platform": args.platform,
        "target": args.target,
        "backend": args.backend,
        "features": features,
        "archive_format": "tar.gz",
        "source_date_epoch": args.epoch,
        "signature": None,
        "notarization": None,
    }
    environment = {
        "schema_version": "leone.package-environment.v1",
        "package": args.package_kind,
        "target": args.target,
        "platform": args.platform,
        "backend": args.backend,
        "rust_toolchain": "1.92",
        "profile": "release",
        "source_date_epoch": args.epoch,
        "path_remap_prefixes": {
            "source": "/source/leone",
            "cargo": "/cargo",
            "rustup": "/rustup",
            "c": "/source/c",
            "objc": "/source/objc",
        },
        "archive": {"format": "tar.gz", "gzip_level": 9},
        "signing": {"status": "unsigned", "notarization": "notarization not performed"},
    }
    for path, record in ((args.package, package), (args.environment, environment)):
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Leone package metadata.")
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument(
        "--package-kind", choices=("runtime", "evidence"), default="runtime"
    )
    args = parser.parse_args(arguments)
    write_records(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
