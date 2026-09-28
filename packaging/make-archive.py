#!/usr/bin/env python3
"""Write a deterministic tar.gz archive with standard-library tools."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import tarfile


def add_file(bundle: tarfile.TarFile, path: Path, relative: str, epoch: int) -> None:
    """Add one regular file with normalized archive metadata."""
    info = bundle.gettarinfo(str(path), arcname=relative)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    with path.open("rb") as source:
        bundle.addfile(info, source)


def write_archive(stage: Path, artifact: str, output: Path, epoch: int) -> None:
    """Write one top-level artifact directory in sorted order."""
    root = stage / artifact
    if stage.is_symlink() or root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as bundle:
                for path in sorted(root.rglob("*")):
                    relative = path.relative_to(stage).as_posix()
                    _add_path(bundle, path, relative, epoch)


def _add_path(bundle: tarfile.TarFile, path: Path, relative: str, epoch: int) -> None:
    if path.is_symlink():
        raise ValueError(f"artifact contains a symlink: {relative}")
    if path.is_dir():
        _add_directory(bundle, path, relative, epoch)
        return
    if path.is_file():
        add_file(bundle, path, relative, epoch)
        return
    raise ValueError(f"artifact contains a non-regular path: {path}")


def _add_directory(bundle: tarfile.TarFile, path: Path, relative: str, epoch: int) -> None:
    info = bundle.gettarinfo(str(path), arcname=relative)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = epoch
    bundle.addfile(info)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create one reproducible Leone archive.")
    parser.add_argument("stage", type=Path)
    parser.add_argument("artifact")
    parser.add_argument("output", type=Path)
    parser.add_argument("epoch", type=int)
    args = parser.parse_args(arguments)
    write_archive(args.stage, args.artifact, args.output, args.epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
