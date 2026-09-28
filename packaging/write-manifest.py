#!/usr/bin/env python3
"""Write a sorted SHA-256 manifest for one staged artifact."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def regular_files(root: Path, output: Path) -> list[Path]:
    """Return staged regular files without following symlink components."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"manifest input contains a symlink: {path.relative_to(root)}")
        if path.is_file() and path != output:
            files.append(path)
    return files


def digest(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_manifest(root: Path, output: Path) -> None:
    """Write manifest rows in locale-independent relative path order."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"manifest input is not a directory: {root}")
    files = sorted(regular_files(root, output))
    rows = [
        f"{digest(path)}  ./{path.relative_to(root).as_posix()}"
        for path in files
    ]
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create one SHA-256 file manifest.")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(arguments)
    write_manifest(args.root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
