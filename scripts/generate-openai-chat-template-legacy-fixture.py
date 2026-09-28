#!/usr/bin/env python3
"""Generate independent Qwen legacy history tokens through pinned llama.cpp."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


MAX_CASES = 32
SOURCE_NAME = "generate-openai-chat-template-fixtures.py"


def load_source():
    source = Path(__file__).with_name(SOURCE_NAME)
    spec = importlib.util.spec_from_file_location("legacy_fixture_generator", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the legacy fixture helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_case(source, endpoint, case):
    body = {
        "messages": case["messages"],
        "add_generation_prompt": True,
        "chat_template_kwargs": {},
    }
    rendered = source.post_json(f"{endpoint}/apply-template", body)["prompt"]
    tokens = source.post_json(
        f"{endpoint}/tokenize",
        {"content": rendered, "add_special": True, "parse_special": True},
    )["tokens"]
    return rendered, tokens


def generate_fixture(path, source, llama_server, model, template, model_artifact):
    fixture = json.loads(path.read_text())
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise ValueError("legacy fixture case count is outside the supported bound")
    if any(case.get("architecture") != "qwen3" for case in cases):
        raise ValueError("legacy fixture cases must use the qwen3 architecture")
    commit = source.verify_engine(llama_server)
    process, endpoint = source.start_server(llama_server, model, template)
    try:
        for case in cases:
            rendered, tokens = tokenize_case(source, endpoint, case)
            case["rendered_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
            case["expected_tokens"] = tokens
            case["token_count"] = len(tokens)
    finally:
        source.stop_server(process)
    fixture["schema_version"] = "leone.openai-chat-template-legacy-reasoning.v1"
    fixture["contract"] = "legacy-chatml-independent-fixture"
    fixture["generator"] = {
        "path": "scripts/generate-openai-chat-template-legacy-fixture.py",
        "engine": "pinned llama.cpp /apply-template and /tokenize",
        "llama_cpp_commit": commit,
        "source": "checked-in qwen3-legacy-chatml.jinja and pinned Qwen GGUF tokenizer",
        "template_file": "fixtures/qwen3-legacy-chatml.jinja",
        "template_sha256": file_digest(template),
        "special_tokens": {"add_special": True, "parse_special": True},
        "model_artifact": model_artifact,
        "model_sha256": file_digest(model),
    }
    path.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("fixtures/openai-chat-template-legacy-reasoning.json"),
    )
    parser.add_argument("--llama-server", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument(
        "--model-artifact", default="models/Qwen3-8B-Q4_K_M.gguf"
    )
    arguments = parser.parse_args()
    source = load_source()
    generate_fixture(
        arguments.fixture,
        source,
        arguments.llama_server,
        arguments.qwen_model,
        Path("fixtures/qwen3-legacy-chatml.jinja"),
        arguments.model_artifact,
    )


if __name__ == "__main__":
    main()
