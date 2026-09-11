#!/usr/bin/env python3
"""Verify measured source inputs without requiring their development history."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("receipts/source-inputs.json")
INPUTS = (
    "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo",
    "crates", "fixtures", "plans", "corpus",
    "benchmarks", "research/oracle", "external/PINNED",
    "scripts/quality-concurrent-service.sh", "scripts/run-llama-oracle.sh",
    "scripts/build-llama-oracle.sh", "scripts/check-openai-client.py",
    "scripts/check-openai-client.sh", "scripts/client-requirements.txt",
    "scripts/study-concurrent-service.py", "scripts/study-concurrent-service.sh",
    "scripts/study-live-server.sh", "scripts/study-batched-service.sh",
)


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.PIPE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root, revision):
    entries = git(root, "ls-tree", "-r", "-z", revision, "--", *INPUTS)
    files = {}
    for entry in entries.decode().split("\0"):
        if entry:
            metadata, name = entry.split("\t", 1)
            files[name] = {
                "sha256": digest(git(root, "show", f"{revision}:{name}")),
                "executable": metadata.split()[0] == "100755",
            }
    return {
        "source_commit": git(root, "rev-parse", revision).decode().strip(),
        "files": files,
    }


def current_files(root):
    names = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *INPUTS)
    return {name for name in names.decode().split("\0") if name}


def check_files(root, files):
    if not isinstance(files, dict) or not files:
        raise ValueError("source input manifest has no files")
    if set(files) != current_files(root):
        raise ValueError("measured source file set differs")
    for name, expected in files.items():
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source input is not a regular file: {name}")
        if bool(path.stat().st_mode & 0o111) != expected["executable"]:
            raise ValueError(f"source executable mode changed: {name}")
        if digest(path.read_bytes()) != expected["sha256"]:
            raise ValueError(f"measured source input changed: {name}")


def check(root, revision):
    path = root / MANIFEST
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest["source_commit"] == revision:
            check_files(root, manifest["files"])
            return
    git(root, "merge-base", "--is-ancestor", revision, "HEAD")
    git(root, "diff", "--exit-code", revision, "--", *INPUTS)
    if git(root, "ls-files", "--others", "--exclude-standard", "--", *INPUTS):
        raise ValueError("source inputs include unrecorded files")


def main(arguments):
    if len(arguments) == 3 and arguments[0] == "record":
        record = snapshot(ROOT, arguments[1])
        with Path(arguments[2]).open("x") as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
        return
    if len(arguments) == 2 and arguments[0] == "check":
        check(ROOT, arguments[1])
        return
    raise ValueError("usage: source_inputs.py check SOURCE | record SOURCE OUTPUT")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"source verification failed: {error}", file=sys.stderr)
        sys.exit(1)
