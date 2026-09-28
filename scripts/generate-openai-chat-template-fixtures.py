#!/usr/bin/env python3
"""Regenerates official chat-template fixtures through pinned llama.cpp."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import socket
import struct
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


MAX_CASES = 32
HEALTH_TIMEOUT_SECONDS = 120
TEMPLATE_DATE = "26 Jul 2024"
PINNED_COMMIT = (Path(__file__).resolve().parents[1] / "external/PINNED").read_text().strip()
GGUF_MAGIC = b"GGUF"
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_SCALAR_SIZES = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_u64(source):
    return struct.unpack("<Q", source.read(8))[0]


def read_string(source):
    length = read_u64(source)
    value = source.read(length)
    if len(value) != length:
        raise ValueError("GGUF string ended early")
    return value


def skip_value(source, value_type):
    if value_type == GGUF_STRING:
        read_string(source)
        return
    if value_type == GGUF_ARRAY:
        element_type = struct.unpack("<I", source.read(4))[0]
        length = read_u64(source)
        for _ in range(length):
            skip_value(source, element_type)
        return
    try:
        size = GGUF_SCALAR_SIZES[value_type]
    except KeyError as error:
        raise ValueError(f"unsupported GGUF metadata type {value_type}") from error
    source.seek(size, 1)


def embedded_template(path):
    with path.open("rb") as source:
        if source.read(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        source.seek(4 + 4 + 8, 0)
        metadata_count = read_u64(source)
        for _ in range(metadata_count):
            key = read_string(source).decode("utf-8")
            value_type = struct.unpack("<I", source.read(4))[0]
            if key == "tokenizer.chat_template":
                if value_type != GGUF_STRING:
                    raise ValueError("GGUF chat template is not a string")
                return read_string(source).decode("utf-8")
            skip_value(source, value_type)
    raise ValueError(f"{path} has no tokenizer.chat_template metadata")


def verify_template_source(source, expected):
    request = urllib.request.Request(source, headers={"User-Agent": "leone-fixture/1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot fetch pinned template source {source}") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"template source hash differs for {source}: {actual}")
    try:
        document = json.loads(payload)
        template = document["chat_template"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"template source has no chat_template: {source}") from error
    if not isinstance(template, str):
        raise ValueError(f"template source chat_template is not a string: {source}")
    return template


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def post_json(url, value):
    body = json.dumps(value).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=HEALTH_TIMEOUT_SECONDS) as response:
        return json.load(response)


def start_server(binary, model, template_file):
    port = free_port()
    process = subprocess.Popen(
        [
            str(binary),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--n-gpu-layers",
            "0",
            "--device",
            "none",
            "--no-op-offload",
            "--threads",
            "2",
            "--threads-batch",
            "2",
            "--parallel",
            "1",
            "--ctx-size",
            "8192",
            "--jinja",
            "--chat-template-file",
            str(template_file),
            "--no-warmup",
            "--log-disable",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    endpoint = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{endpoint}/health", timeout=1) as response:
                if response.status == 200:
                    return process, endpoint
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)
    process.terminate()
    process.wait(timeout=10)
    raise RuntimeError(f"llama.cpp server did not become healthy for {model}")


def verify_engine(binary):
    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=True
    )
    version = result.stdout + result.stderr
    match = re.search(r"\bcommit\s+([0-9a-f]{7,40})\b", version.lower())
    if match is None or not PINNED_COMMIT.startswith(match.group(1)):
        raise ValueError(
            f"llama.cpp binary does not report pinned commit {PINNED_COMMIT}"
        )
    return PINNED_COMMIT


def stop_server(process):
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def apply_template(endpoint, case):
    body = {
        "messages": case["messages"],
        "chat_template_kwargs": {"date_string": TEMPLATE_DATE},
    }
    if case.get("tools") is not None:
        body["tools"] = case["tools"]
    if case.get("tool_choice") is not None:
        body["tool_choice"] = case["tool_choice"]
    rendered = post_json(f"{endpoint}/apply-template", body)["prompt"]
    tokenized = post_json(
        f"{endpoint}/tokenize",
        {"content": rendered, "add_special": False, "parse_special": True},
    )["tokens"]
    return rendered, tokenized


def prepare_template_files(fixture, models, directory):
    template_files = {}
    for architecture, model in models.items():
        source = fixture["architectures"][architecture]
        template = verify_template_source(
            source["template_source"], source["template_sha256"]
        )
        embedded = embedded_template(model)
        embedded_hash = hashlib.sha256(embedded.encode("utf-8")).hexdigest()
        if embedded_hash != source["embedded_template_sha256"]:
            raise ValueError(f"embedded template hash differs for {architecture}")
        template_file = Path(directory) / f"{architecture}.jinja"
        template_file.write_text(template)
        template_files[architecture] = template_file
        model_hash = digest(model)
        if model_hash != source["model_sha256"]:
            raise ValueError(f"model hash differs for {architecture}: {model_hash}")
    return template_files


def validate_fixture_architectures(cases, models):
    for case in cases:
        if case["architecture"] not in models:
            raise ValueError(f"unknown fixture architecture {case['architecture']}")


def refresh_fixture_cases(cases, models, binary, template_files):
    for architecture, model in models.items():
        process, endpoint = start_server(binary, model, template_files[architecture])
        try:
            for case in cases:
                if case["architecture"] != architecture:
                    continue
                rendered, tokens = apply_template(endpoint, case)
                case["rendered_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
                case["expected_tokens"] = tokens
                case["token_count"] = len(tokens)
        finally:
            stop_server(process)


def update_fixture(path, binary, models):
    fixture = json.loads(path.read_text())
    cases = fixture["cases"]
    if not cases or len(cases) > MAX_CASES:
        raise ValueError("fixture case count is outside the supported bound")
    validate_fixture_architectures(cases, models)
    engine_commit = verify_engine(binary)
    with tempfile.TemporaryDirectory(prefix="leone-chat-template-") as directory:
        template_files = prepare_template_files(fixture, models, directory)
        refresh_fixture_cases(cases, models, binary, template_files)
    fixture["schema_version"] = "leone.openai-chat-template-fixtures.v3"
    fixture["generator"] = {
        "path": "scripts/generate-openai-chat-template-fixtures.py",
        "engine": "pinned llama.cpp /apply-template and /tokenize with pinned template files",
        "llama_cpp_commit": engine_commit,
        "source": "independently fetched tokenizer_config.json chat_template and pinned GGUF tokenizer",
        "date_string": TEMPLATE_DATE,
    }
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/openai-chat-template-tokens.json"),
    )
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--llama-model", type=Path, required=True)
    arguments = parser.parse_args()
    update_fixture(
        arguments.fixture,
        arguments.llama_server,
        {"qwen3": arguments.qwen_model, "llama3": arguments.llama_model},
    )


if __name__ == "__main__":
    main()
