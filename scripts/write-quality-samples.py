#!/usr/bin/env python3
"""Write deterministic sampled logits for a portable quality record."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import NoReturn


SAMPLE_ROWS = 128


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest is not an object: {path}")
    return value


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_prefix(value: str) -> str:
    path = Path(value)
    require(not path.is_absolute() and len(path.parts) == 1 and path.parts[0] not in (".", ".."), "sample path prefix is not a stable directory name")
    return path.as_posix().rstrip("/")


def sampled_indices(rows: int, task_rows: list[int]) -> tuple[int, list[int]]:
    requested = min(SAMPLE_ROWS, rows)
    if requested == 1:
        base = [0]
    else:
        base = [index * (rows - 1) // (requested - 1) for index in range(requested)]
    indices = sorted(set(base + task_rows))
    require(all(0 <= row < rows for row in indices), "sample row is outside the source")
    return requested, indices


def stage_record(stage: dict, section: str, name: str) -> dict:
    record = stage.get(section, {}).get(name)
    require(isinstance(record, dict), f"stage {section}.{name} record is missing")
    return record


def write_atomic(path: Path, data: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def sample_logits(source: Path, destination: Path, indices: list[int], vocab: int, expected_bytes: int, expected_hash: str) -> list[str]:
    row_bytes = vocab * 4
    require(source.is_file(), f"logits source is missing: {source}")
    require(source.stat().st_size == expected_bytes, "logits source size differs from its stage record")
    require(digest_file(source) == expected_hash, "logits source hash differs from its stage record")
    require(source.stat().st_size >= (max(indices) + 1) * row_bytes, "logits source is truncated")
    hashes = []
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with source.open("rb") as source_file, os.fdopen(descriptor, "wb") as target:
            for row in indices:
                source_file.seek(row * row_bytes)
                data = source_file.read(row_bytes)
                require(len(data) == row_bytes, "logits source row is truncated")
                target.write(data)
                hashes.append(digest_bytes(data))
        require(digest_file(source) == expected_hash, "logits source changed during sampling")
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return hashes


def artifact_record(path: Path, relative: str, encoding: str, row_hashes: list[str] | None = None) -> dict:
    record = {
        "path": relative,
        "sha256": digest_file(path),
        "bytes": path.stat().st_size,
        "encoding": encoding,
    }
    if row_hashes is not None:
        record["source_row_sha256"] = row_hashes
    return record


def sample_context(arguments: argparse.Namespace) -> tuple[Path, Path, dict, dict, int, int, list[int], list[int], int, list[int], dict]:
    export_stage_path = Path(arguments.oracle_stage).resolve()
    metal_stage_path = Path(arguments.metal_stage).resolve()
    task_path = Path(arguments.task_manifest).resolve()
    export_stage = load(export_stage_path / "oracle-stage.json")
    metal_stage = load(metal_stage_path / "metal-stage.json")
    task = load(task_path)
    input_record = export_stage.get("input", {})
    rows = input_record.get("rows")
    vocab = input_record.get("vocab_size")
    require(isinstance(rows, int) and rows > 0, "stage row count is invalid")
    require(isinstance(vocab, int) and vocab > 0, "stage vocabulary is invalid")
    task_spec = task.get("task", {})
    answer = task_spec.get("answer", {})
    answer_row = answer.get("row")
    answer_tokens = answer.get("token_ids")
    require(isinstance(answer_row, int) and answer_row >= 0, "task answer row is invalid")
    require(isinstance(answer_tokens, list) and answer_tokens, "task answer token IDs are missing")
    task_rows = list(range(answer_row, answer_row + len(answer_tokens)))
    require(all(row < rows for row in task_rows), "task answer row is outside the source")
    requested, indices = sampled_indices(rows, task_rows)
    tokens_record = input_record.get("tokens")
    require(isinstance(tokens_record, dict), "stage token record is missing")
    tokens_source = export_stage_path / tokens_record["path"]
    require(tokens_source.is_file(), "stage token source is missing")
    token_bytes = tokens_source.read_bytes()
    require(digest_bytes(token_bytes) == tokens_record.get("sha256"), "stage token source hash differs")
    token_count = tokens_record.get("count")
    require(isinstance(token_count, int) and len(token_bytes) == token_count * 4, "stage token source shape differs")
    source_tokens = struct.unpack(f"<{token_count}I", token_bytes)
    return export_stage_path, metal_stage_path, export_stage, metal_stage, rows, vocab, task_rows, indices, requested, source_tokens, tokens_record


def write_sample_logits(
    export_stage_path: Path,
    metal_stage_path: Path,
    export_stage: dict,
    metal_stage: dict,
    output: Path,
    prefix: str,
    indices: list[int],
    vocab: int,
) -> dict:

    source_logits = {
        "oracle": (export_stage_path, stage_record(export_stage, "oracle", "logits")),
        "llama_cpp": (metal_stage_path, stage_record(metal_stage, "subject", "logits")),
        "leone": (metal_stage_path, stage_record(metal_stage, "native", "logits")),
    }
    names = {"oracle": "oracle.logits.bin", "llama_cpp": "llama-cpp-metal.logits.bin", "leone": "leone-metal.logits.bin"}
    sampled_logits = {}
    for label, (stage_path, record) in source_logits.items():
        source = stage_path / record["path"]
        destination = output / names[label]
        row_hashes = sample_logits(source, destination, indices, vocab, record["bytes"], record["sha256"])
        sampled_logits[label] = {
            "source": record,
            "artifact": artifact_record(destination, f"{prefix}/{names[label]}", "row-major-f32-le", row_hashes),
        }
    return sampled_logits


def write_samples(arguments: argparse.Namespace) -> None:
    output = Path(arguments.output_dir).resolve()
    require(not output.exists(), f"refusing to replace sample directory: {output}")
    output.mkdir(parents=True)
    prefix = stable_prefix(arguments.path_prefix)
    export_stage_path, metal_stage_path, export_stage, metal_stage, rows, vocab, task_rows, indices, requested, source_tokens, tokens_record = sample_context(arguments)
    sampled_tokens = [source_tokens[indices[0]]]
    sampled_tokens.extend(source_tokens[row + 1] for row in indices)
    tokens_path = output / "sample.tokens.u32le"
    write_atomic(tokens_path, struct.pack(f"<{len(sampled_tokens)}I", *sampled_tokens))
    sampled_logits = write_sample_logits(export_stage_path, metal_stage_path, export_stage, metal_stage, output, prefix, indices, vocab)
    body = {
        "schema_version": "leone.quality-sample.v1",
        "provenance": {
            "full_logits": "generation-only",
            "row_hashes": "sha256 of each packaged row in source order",
            "offline_replay": "sampled rows only",
        },
        "source": {
            "rows": rows,
            "vocab_size": vocab,
            "tokens": tokens_record,
            "logits": {label: value["source"] for label, value in sampled_logits.items()},
        },
        "sampling": {
            "algorithm": "linspace-inclusive-v1",
            "requested_rows": requested,
            "source_rows": rows,
            "sample_count": len(indices),
            "indices": indices,
            "task_rows": task_rows,
        },
        "artifacts": {
            "tokens": artifact_record(tokens_path, f"{prefix}/sample.tokens.u32le", "u32le"),
            "logits": {label: value["artifact"] for label, value in sampled_logits.items()},
        },
    }
    sampling_path = output / "sampling.json"
    write_atomic(sampling_path, (json.dumps(body, indent=2, sort_keys=True) + "\n").encode())
    print(sampling_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-stage", required=True)
    parser.add_argument("--metal-stage", required=True)
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--path-prefix", required=True)
    write_samples(parser.parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, struct.error) as error:
        raise SystemExit(f"error: {error}")
