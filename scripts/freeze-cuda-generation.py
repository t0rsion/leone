#!/usr/bin/env python3
"""Freeze sampled CUDA quality rows from a retained producer collection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
GENERATION_SCHEMA = "leone.quality-generation.v1"
COLLECTION_SCHEMA = "leone.quality-collection.v1"
WRITER_PATH = ROOT / "scripts/write-cuda-quality-comparison.py"


def load_writer():
    spec = importlib.util.spec_from_file_location("write_cuda_quality_comparison", WRITER_PATH)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load CUDA quality writer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fail(message: str) -> "NoReturn":
    raise ValueError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def parse_options(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--trusted-inputs", type=Path, required=True)
    parser.add_argument("--subject-model", type=Path, required=True)
    parser.add_argument("--oracle-model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(arguments)


def load_collection(options: argparse.Namespace, writer) -> tuple[dict, Path, dict]:
    writer.regular_file(options.collection, "CUDA collection manifest")
    body = writer.load(options.collection)
    require(body.get("schema_version") == COLLECTION_SCHEMA, "CUDA collection schema differs")
    writer.stable_strings(body, "collection")
    root = options.collection.parent
    artifacts = body.get("artifacts")
    require(isinstance(artifacts, dict), "CUDA collection artifacts are missing")
    for label, record in artifacts.items():
        require(isinstance(record, dict), f"CUDA collection {label} record is invalid")
        path_name = writer.public_name(record.get("path"), f"CUDA collection {label} path")
        require(path_name == record["path"], f"CUDA collection {label} path is not stable")
        path = root / path_name
        writer.regular_file(path, f"CUDA collection {label}")
        if "sha256" in record:
            require(writer.digest(path) == writer.valid_hash(record["sha256"], f"CUDA collection {label} SHA-256"), f"CUDA collection {label} changed")
        if "bytes" in record:
            require(path.stat().st_size == record["bytes"], f"CUDA collection {label} byte count differs")
    trusted = writer.load_trusted_inputs(options.trusted_inputs)
    return body, root, trusted


def validate_models_and_inputs(options: argparse.Namespace, collection: dict, trusted: dict, writer) -> tuple[dict, dict, dict]:
    for path, expected, label in (
        (options.subject_model, trusted["model_sha256"], "subject model"),
        (options.oracle_model, trusted["oracle_model_sha256"], "oracle model"),
        (options.corpus, trusted["corpus_sha256"], "corpus"),
    ):
        writer.regular_file(path, label)
        require(writer.digest(path) == expected, f"{label} differs from trusted inputs")
    models = collection.get("models")
    require(isinstance(models, dict), "CUDA collection models are missing")
    require(models.get("subject_sha256") == trusted["model_sha256"], "collection subject model differs")
    require(models.get("oracle_sha256") == trusted["oracle_model_sha256"], "collection oracle model differs")
    require(models.get("corpus_sha256") == trusted["corpus_sha256"], "collection corpus differs")
    input_record = collection.get("input")
    require(isinstance(input_record, dict), "CUDA collection input is missing")
    token_record = collection["artifacts"].get("tokens")
    require(isinstance(token_record, dict), "CUDA collection token artifact is missing")
    input_tokens = input_record.get("tokens")
    require(isinstance(input_tokens, dict), "CUDA collection input tokens are missing")
    token_count = input_tokens.get("count")
    require(isinstance(token_count, int) and not isinstance(token_count, bool) and token_count > 0, "CUDA collection token count is invalid")
    require(token_record.get("encoding") == "u32le", "CUDA collection token encoding differs")
    require(token_record.get("count") == token_count, "CUDA collection token count differs")
    require(token_record.get("bytes") == token_count * 4, "CUDA collection token shape differs")
    return models, input_record, collection["artifacts"]


def producer_files(root: Path, artifacts: dict) -> dict:
    return {
        "oracle": root / artifacts["oracle_manifest"]["path"],
        "llama_cpp": root / artifacts["llama_manifest"]["path"],
        "leone": root / artifacts["leone_manifest"]["path"],
    }


def validate_producers(
    options: argparse.Namespace,
    root: Path,
    artifacts: dict,
    collection: dict,
    trusted: dict,
    writer,
) -> tuple[dict, dict, dict, list[int], dict]:
    paths = producer_files(root, artifacts)
    runs = {label: writer.load(path) for label, path in paths.items()}
    oracle = runs["oracle"]
    llama = runs["llama_cpp"]
    native = runs["leone"]
    require(oracle.get("input") == llama.get("input") == native.get("input"), "CUDA collection producer inputs differ")
    input_record = oracle.get("input")
    require(isinstance(input_record, dict), "CUDA collection producer input is missing")
    collection_input = collection.get("input")
    require(isinstance(collection_input, dict), "CUDA collection input is missing")
    normalized_input = dict(input_record)
    normalized_tokens = dict(input_record.get("tokens", {}))
    normalized_tokens["path"] = "tokens.u32le"
    normalized_input["tokens"] = normalized_tokens
    require(collection_input == normalized_input, "CUDA collection input differs from producer manifests")
    rows = input_record.get("rows")
    vocab = input_record.get("vocab_size")
    require(isinstance(rows, int) and rows > 0 and isinstance(vocab, int) and vocab > 0, "CUDA collection shape is invalid")
    full = {
        "oracle": writer.source_logits(root / artifacts["oracle_logits"]["path"], oracle, "generated/oracle.f32", rows, vocab, "oracle logits"),
        "llama_cpp": writer.source_logits(root / artifacts["llama_logits"]["path"], llama, "generated/llama-cpp-cuda.f32", rows, vocab, "llama logits"),
        "leone": writer.source_logits(root / artifacts["leone_logits"]["path"], native, "generated/leone-cuda.f32", rows, vocab, "Leone logits"),
    }
    for label, key in (("oracle", "oracle_logits"), ("llama.cpp", "llama_logits"), ("Leone", "leone_logits")):
        record = artifacts[key]
        require(record.get("encoding") == "row-major-f32-le", f"CUDA collection {label} logits encoding differs")
        require(record.get("bytes") == rows * vocab * 4, f"CUDA collection {label} logits shape differs")
    manifests = collection.get("producer_manifests")
    require(isinstance(manifests, dict), "CUDA collection producer manifests are missing")
    for label, key in (("oracle", "oracle_manifest"), ("llama_cpp", "llama_manifest"), ("leone", "leone_manifest")):
        require(writer.digest(root / artifacts[key]["path"]) == manifests.get(label), f"CUDA collection {label} manifest differs")
    native_build_info = writer.load(root / artifacts["native_build_info"]["path"])
    native_identity = trusted["producer_identities"]["leone_cuda"]
    require(writer.json_digest(native_build_info) == native_identity["build_info_sha256"], "CUDA collection native build info differs")
    require(native_build_info.get("source_commit") == native_identity["source_commit"], "CUDA collection native build source differs")
    require(native_build_info.get("target") == native_identity["target"], "CUDA collection native build target differs")
    identities = trusted["producer_identities"]
    for label, run, identity in (
        ("oracle_stage", oracle, identities["oracle_stage"]),
        ("llama_cpp_cuda", llama, identities["llama_cpp_cuda"]),
        ("leone_cuda", native, identities["leone_cuda"]),
    ):
        writer.validate_observed_identity(label, run, paths["oracle" if label == "oracle_stage" else "llama_cpp" if label == "llama_cpp_cuda" else "leone"], identity)
    task = writer.load(options.task_manifest)
    task_rows = writer.validate_task_manifest(
        task,
        input_record,
        oracle["model"],
        llama["model"],
        writer.digest(options.corpus),
        trusted["model_family"],
        root / artifacts["tokens"]["path"],
        rows,
    )
    return oracle, llama, native, task_rows, full


def sampled_metadata(path: Path, indices: list[int], vocab: int) -> dict:
    row_bytes = vocab * 4
    hasher = hashlib.sha256()
    row_hashes = []
    with path.open("rb") as source:
        for row in indices:
            source.seek(row * row_bytes)
            data = source.read(row_bytes)
            require(len(data) == row_bytes, f"sample source row is truncated: {path}")
            hasher.update(data)
            row_hashes.append(hashlib.sha256(data).hexdigest())
    return {
        "sha256": hasher.hexdigest(),
        "bytes": len(indices) * row_bytes,
        "encoding": "row-major-f32-le",
        "source_row_sha256": row_hashes,
    }


def build_generation(options: argparse.Namespace, collection: dict, root: Path, trusted: dict, runs: tuple[dict, dict, dict, list[int], dict], writer) -> dict:
    oracle, llama, native, task_rows, full = runs
    input_record = oracle["input"]
    rows = input_record["rows"]
    vocab = input_record["vocab_size"]
    requested = min(writer.SAMPLE_ROWS, rows)
    base = writer.sample_indices(rows)
    indices = sorted(set(base + task_rows))
    artifacts = collection["artifacts"]
    logits_paths = {
        "oracle": root / artifacts["oracle_logits"]["path"],
        "llama_cpp": root / artifacts["llama_logits"]["path"],
        "leone": root / artifacts["leone_logits"]["path"],
    }
    generation = {
        "schema_version": GENERATION_SCHEMA,
        "inputs": {
            "model_sha256": trusted["model_sha256"],
            "oracle_model_sha256": trusted["oracle_model_sha256"],
            "corpus_sha256": trusted["corpus_sha256"],
            "task_manifest_sha256": writer.digest(options.task_manifest),
            "tokens_sha256": input_record["tokens"]["sha256"],
            "token_count": input_record["tokens"]["count"],
            "rows": rows,
            "vocab_size": vocab,
            "sample_contract": writer.SAMPLE_CONTRACT,
        },
        "sampling": {
            "algorithm": "linspace-inclusive-v1",
            "requested_rows": requested,
            "indices": indices,
            "task_rows": task_rows,
        },
        "producer_manifests": {
            "oracle": writer.digest(root / artifacts["oracle_manifest"]["path"]),
            "llama_cpp": writer.digest(root / artifacts["llama_manifest"]["path"]),
            "leone": writer.digest(root / artifacts["leone_manifest"]["path"]),
        },
        "full_logits": {label: {name: full[label][name] for name in ("sha256", "bytes", "encoding")} for label in full},
        "sampled_logits": {label: sampled_metadata(path, indices, vocab) for label, path in logits_paths.items()},
    }
    writer.validate_generation_record(generation, options_namespace(trusted, root, artifacts), trusted, input_record, rows, vocab, task_rows, full)
    return generation


def options_namespace(trusted: dict, root: Path, artifacts: dict) -> argparse.Namespace:
    return argparse.Namespace(
        oracle_manifest=root / artifacts["oracle_manifest"]["path"],
        llama_manifest=root / artifacts["llama_manifest"]["path"],
        leone_manifest=root / artifacts["leone_manifest"]["path"],
        source_commit=trusted["source_commit"],
        platform=trusted["platform"],
        target=trusted["target"],
        statistics_target=trusted["statistics_target"],
        model_family=trusted["model_family"],
    )


def main(arguments: list[str]) -> int:
    options = parse_options(arguments)
    writer = load_writer()
    collection, root, trusted = load_collection(options, writer)
    _models, _input_record, artifacts = validate_models_and_inputs(options, collection, trusted, writer)
    runs = validate_producers(options, root, artifacts, collection, trusted, writer)
    generation = build_generation(options, collection, root, trusted, runs, writer)
    writer.write_no_replace(options.output, json.dumps(generation, indent=2, sort_keys=True) + "\n")
    print(options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
