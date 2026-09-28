#!/usr/bin/env python3
"""Write the external anchor produced by a Metal quality comparison."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import NoReturn


SCHEMA = "leone.quality-metal-generation.v1"
HASH_LENGTH = 64


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), f"comparison is not a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"cannot read comparison {path}: {error}")
    require(isinstance(value, dict), "comparison is not an object")
    return value


def field(value: dict, name: str, kind: type) -> object:
    result = value.get(name)
    require(isinstance(result, kind), f"comparison field {name} has the wrong type")
    return result


def valid_hash(value: object, label: str) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    require(len(value) == HASH_LENGTH and all(character in "0123456789abcdef" for character in value), f"{label} is not a SHA-256")
    return value


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_strings(value: object, label: str = "generation") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            stable_strings(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            stable_strings(item, f"{label}[{index}]")
    elif isinstance(value, str):
        require(not os.path.isabs(value), f"{label} contains an absolute path")
        require(not value.startswith("~"), f"{label} contains a home-relative path")


def logits_record(stage: dict, section: str, label: str) -> dict:
    record = field(stage, section, dict)
    logits = field(record, "logits", dict)
    valid_hash(logits.get("sha256"), f"{label} logits SHA-256")
    require(isinstance(logits.get("bytes"), int) and logits["bytes"] > 0, f"{label} logits byte count is invalid")
    require(logits.get("encoding") == "row-major-f32-le", f"{label} logits encoding differs")
    return {key: logits[key] for key in ("path", "sha256", "bytes", "encoding")}


def generation_body(comparison: dict, root: Path) -> dict:
    require(comparison.get("schema_version") == "leone.quality-cross-device.v2", "comparison schema differs")
    validation = field(comparison, "validation", dict)
    trusted = field(validation, "trusted", dict)
    sample = field(field(comparison, "samples", dict), "manifest", dict)
    sample_hash = valid_hash(sample.get("sha256"), "sample manifest SHA-256")
    sample_name = field(sample, "path", str)
    sample_relative = Path(sample_name)
    require(not sample_relative.is_absolute() and ".." not in sample_relative.parts, "sample manifest path is not stable")
    sample_path = root / sample_relative
    require(sample_path.is_file() and not sample_path.is_symlink(), "sample manifest is missing")
    require(digest(sample_path) == sample_hash, "sample manifest SHA-256 differs")
    require(trusted.get("sample_manifest_sha256") == sample_hash, "comparison does not contain its sample anchor")
    stages = field(comparison, "stages", dict)
    oracle_stage = field(stages, "oracle_export", dict)
    metal_stage = field(stages, "metal_subject", dict)
    identities = field(comparison, "source_identities", dict)
    task = field(field(comparison, "tasks", dict), "long_context", dict)
    body = {
        "schema_version": SCHEMA,
        "backend": "metal",
        "source_commit": comparison["source_commit"],
        "native_source_commit": field(identities, "leone_metal", dict)["source_commit"],
        "models": comparison["models"],
        "corpus": comparison["corpus"],
        "input": comparison["input"],
        "task_manifest_sha256": valid_hash(field(task, "manifest", dict).get("sha256"), "task manifest SHA-256"),
        "sample_contract": trusted["sample_contract"],
        "sample_manifest": {"path": sample["path"], "sha256": sample_hash},
        "producer_manifests": {
            "oracle_export": valid_hash(oracle_stage.get("sha256"), "oracle stage SHA-256"),
            "metal_subject": valid_hash(metal_stage.get("sha256"), "Metal stage SHA-256"),
        },
        "full_logits": {
            "oracle": logits_record(oracle_stage["body"], "oracle", "oracle"),
            "llama_cpp_metal": logits_record(metal_stage["body"], "subject", "llama Metal"),
            "leone_metal": logits_record(metal_stage["body"], "native", "Leone Metal"),
        },
    }
    stable_strings(body)
    return body


def write_no_replace(path: Path, body: dict) -> tuple[int, int]:
    require(not path.exists() and not path.is_symlink(), f"generation record already exists: {path}")
    require(path.parent.is_dir() and not path.parent.is_symlink(), "generation record parent is not a regular directory")
    encoded = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
            identity = os.fstat(target.fileno())
        os.link(temporary, path)
    except FileExistsError:
        fail(f"generation record already exists: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return identity.st_dev, identity.st_ino


def main(arguments: list[str]) -> int:
    if len(arguments) not in (2, 3) or len(arguments) == 3 and arguments[2] != "--identity":
        raise ValueError("usage: write-metal-quality-generation.py COMPARISON OUTPUT [--identity]")
    comparison = Path(arguments[0])
    output = Path(arguments[1])
    identity = write_no_replace(output, generation_body(load(comparison), comparison.parent))
    if len(arguments) == 3:
        print(f"{identity[0]}:{identity[1]}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
