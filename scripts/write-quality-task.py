#!/usr/bin/env python3
"""Write a long-context task from bytes and an independent token record."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import os
import struct
import subprocess
import tempfile


MIN_CONTEXT_TOKENS = 4096
PINNED_PATH = Path(__file__).resolve().parents[1] / "external/PINNED"


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
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


def tokenizer_artifacts(executable: Path, model: Path) -> dict:
    return {
        "executable": {
            "name": executable.name,
            "sha256": digest_file(executable),
            "bytes": executable.stat().st_size,
        },
        "model": {"name": model.name, "sha256": digest_file(model)},
    }


def publish_task(output: Path, task: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as temporary:
            temporary.write(json.dumps(task, indent=2, sort_keys=True) + "\n")
        os.link(temporary_name, output)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def tokenize(executable: Path, model: Path, data: bytes) -> list[int]:
    command = [
        str(executable),
        "-m",
        str(model),
        "--stdin",
        "--ids",
        "--no-bos",
        "--no-parse-special",
        "--device",
        "none",
        "--gpu-layers",
        "0",
        "--threads",
        "1",
    ]
    try:
        result = subprocess.run(command, input=data, capture_output=True, check=True)
        values = ast.literal_eval(result.stdout.decode("utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError, subprocess.CalledProcessError) as error:
        raise ValueError(f"pinned tokenizer failed: {error}") from error
    if not isinstance(values, list) or not values or not all(isinstance(value, int) and value >= 0 for value in values):
        raise ValueError("pinned tokenizer did not return token IDs")
    return values


def text_record(path: Path, label: str) -> dict:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not UTF-8: {error}") from error
    return {"encoding": "utf-8", "text": text, "bytes": len(raw), "sha256": digest_bytes(raw)}


def write_task(arguments: argparse.Namespace) -> None:
    oracle_stage = Path(arguments.oracle_stage).resolve()
    output = Path(arguments.output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError(f"refusing to replace existing task: {output}")
    exported = load(oracle_stage / "oracle-stage.json")
    context_tokens = exported["input"]["window_tokens"]
    if context_tokens < MIN_CONTEXT_TOKENS:
        raise ValueError(f"the evaluation window must be at least {MIN_CONTEXT_TOKENS} tokens")
    prompt = text_record(Path(arguments.prompt_file).resolve(), "prompt")
    needle = text_record(Path(arguments.needle_file).resolve(), "needle")
    combined = prompt["text"].encode() + needle["text"].encode()
    if digest_bytes(combined) != exported["corpus"]["sha256"]:
        raise ValueError("prompt and needle bytes do not reconstruct the staged corpus")
    tokenizer_executable = Path(arguments.tokenizer_executable).resolve()
    tokenizer_model = Path(arguments.tokenizer_model).resolve()
    combined_tokens = tokenize(tokenizer_executable, tokenizer_model, combined)
    answer_tokens = tokenize(tokenizer_executable, tokenizer_model, needle["text"].encode())
    answer_token_bytes = struct.pack(f"<{len(answer_tokens)}I", *answer_tokens)
    token_path = oracle_stage / exported["input"]["tokens"]["path"]
    staged_tokens = list(struct.unpack(f"<{token_path.stat().st_size // 4}I", token_path.read_bytes()))
    if staged_tokens != combined_tokens:
        raise ValueError("pinned tokenizer output differs from the staged token stream")
    answer_row = arguments.answer_row
    if answer_row < context_tokens - 1:
        raise ValueError("the answer row is before the declared context")
    if combined_tokens[answer_row + 1:answer_row + 1 + len(answer_tokens)] != answer_tokens:
        raise ValueError("pinned answer tokens do not match the staged token stream")
    tokenizer = {
        "source": {
            "name": "llama.cpp",
            "git_commit": PINNED_PATH.read_text().strip(),
            "pin_file": "external/PINNED",
            "provenance": "declared",
            "trust": "operator-attested",
        },
        **tokenizer_artifacts(tokenizer_executable, tokenizer_model),
        "method": "llama-tokenize",
        "prompt_sha256": prompt["sha256"],
        "needle_sha256": needle["sha256"],
        "combined_sha256": digest_bytes(combined),
        "answer_tokens_sha256": digest_bytes(answer_token_bytes),
        "input_tokens_sha256": exported["input"]["tokens"]["sha256"],
    }
    oracle_run = exported["oracle"]["manifest"]["body"]
    task = {
        "schema_version": "leone.quality-task.v2",
        "task": {
            "name": "needle-exact-answer",
            "context_tokens": context_tokens,
            "minimum_context_tokens": MIN_CONTEXT_TOKENS,
            "prompt": prompt,
            "needle": needle,
            "answer": {
                "row": answer_row,
                "text": needle["text"],
                "sha256": needle["sha256"],
                "token_ids": answer_tokens,
            },
        },
        "model": {
            "family": arguments.family,
            "oracle": exported["model"]["oracle"],
            "subject": exported["model"]["subject"],
        },
        "tokenizer": {"name": oracle_run["model"]["tokenizer"]},
        "tokenization": tokenizer,
        "corpus": exported["corpus"],
        "input": exported["input"],
    }
    publish_task(output, task)
    print(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle_stage")
    parser.add_argument("family")
    parser.add_argument("answer_row", type=int)
    parser.add_argument("output")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--needle-file", required=True)
    parser.add_argument("--tokenizer-executable", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    write_task(parser.parse_args())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}")
