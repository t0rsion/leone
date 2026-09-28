#!/usr/bin/env python3
"""Build and run the pinned llama.cpp cached-prefix comparator."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import common_oracle

MANIFEST_SCHEMA = "leone.llama-cached-workload.v2"
RECEIPT_SCHEMA = "leone.llama-cached-comparator.v2"
ORACLE_IDENTITY_SCHEMA = "leone.llama-cached-oracle-identity.v1"
ENGINE_PROTOCOL = "leone.llama-cached-engine.v2"
PLAN_MAGIC = b"LCMPPLN1"
PLAN_VERSION = 2
MAX_CASES = 64
MAX_EVENTS = 1024
MAX_STREAMS = 64
MAX_NAME_BYTES = 128
MAX_TOKENS = 1 << 20
SHA256_LENGTH = 64


class ValidationError(ValueError):
    """Reports an input or comparator output that violates the public schema."""


@dataclasses.dataclass(frozen=True)
class Context:
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_seq_max: int
    threads: int
    kv_type: str
    flash_attention: str
    kv_unified: bool
    swa_full: bool


@dataclasses.dataclass(frozen=True)
class Stream:
    seq: int
    position: int
    tokens: tuple[int, ...]
    outputs: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Event:
    kind: str
    name: str
    streams: tuple[Stream, ...] = ()
    sequence: int = -1
    source: int = -1
    target: int = -1
    p0: int = -1
    p1: int = -1


@dataclasses.dataclass(frozen=True)
class Case:
    name: str
    case_class: str
    events: tuple[Event, ...]
    expected_ranges: tuple[tuple[tuple[int, int], ...], ...]


@dataclasses.dataclass(frozen=True)
class Workload:
    phase: str
    model_contract: dict[str, Any]
    context: Context
    token_fixture: dict[str, Any]
    cases: tuple[Case, ...]
    manifest_sha256: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def exact_keys(value: dict[str, Any], expected: set[str], field: str) -> None:
    actual = set(value)
    require(
        actual == expected,
        f"{field} keys differ: expected {sorted(expected)}, got {sorted(actual)}",
    )


def integer(value: Any, field: str, minimum: int, maximum: int) -> int:
    require(type(value) is int, f"{field} must be an integer")
    require(minimum <= value <= maximum, f"{field} is outside [{minimum}, {maximum}]")
    return value


def text_value(value: Any, field: str) -> str:
    require(isinstance(value, str) and value, f"{field} must be a nonempty string")
    encoded = value.encode("utf-8")
    require(len(encoded) <= MAX_NAME_BYTES, f"{field} exceeds {MAX_NAME_BYTES} bytes")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON object {path}: {error}") from error
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def source_path(root: Path, value: Any, field: str) -> Path:
    relative = Path(text_value(value, field))
    require(not relative.is_absolute(), f"{field} must be relative to the source root")
    require(".." not in relative.parts, f"{field} must not contain '..'")
    resolved = (root / relative).resolve()
    require(resolved.is_relative_to(root.resolve()), f"{field} leaves the source root")
    return resolved


def read_u32le_tokens(path: Path) -> tuple[int, ...]:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValidationError(f"cannot read token fixture {path}: {error}") from error
    require(
        data and len(data) % 4 == 0,
        "token fixture size must be a positive multiple of four",
    )
    count = len(data) // 4
    require(count <= MAX_TOKENS, f"token fixture exceeds {MAX_TOKENS} tokens")
    return struct.unpack(f"<{count}I", data)


def parse_context(value: Any) -> Context:
    require(isinstance(value, dict), "context must be an object")
    keys = {
        "n_ctx",
        "n_batch",
        "n_ubatch",
        "n_seq_max",
        "threads",
        "kv_type",
        "flash_attention",
        "kv_unified",
        "swa_full",
    }
    exact_keys(value, keys, "context")
    context = Context(
        integer(value["n_ctx"], "context.n_ctx", 2, MAX_TOKENS),
        integer(value["n_batch"], "context.n_batch", 1, MAX_TOKENS),
        integer(value["n_ubatch"], "context.n_ubatch", 1, MAX_TOKENS),
        integer(value["n_seq_max"], "context.n_seq_max", 1, MAX_STREAMS),
        integer(value["threads"], "context.threads", 1, 1024),
        text_value(value["kv_type"], "context.kv_type"),
        text_value(value["flash_attention"], "context.flash_attention"),
        value["kv_unified"],
        value["swa_full"],
    )
    require(context.kv_type in {"f16", "q8_0"}, "context.kv_type must be f16 or q8_0")
    require(
        context.flash_attention in {"auto", "on", "off"},
        "context.flash_attention must be auto, on, or off",
    )
    require(type(context.kv_unified) is bool, "context.kv_unified must be boolean")
    require(type(context.swa_full) is bool, "context.swa_full must be boolean")
    require(
        context.kv_unified,
        "context.kv_unified must be true for shared-prefix comparison",
    )
    require(
        context.swa_full,
        "context.swa_full must be true for complete observed position ranges",
    )
    require(
        context.n_ubatch <= context.n_batch, "context.n_ubatch exceeds context.n_batch"
    )
    return context


def parse_fixture(value: Any, root: Path) -> tuple[dict[str, Any], tuple[int, ...]]:
    require(isinstance(value, dict), "token_fixture must be an object")
    keys = {
        "path",
        "encoding",
        "bytes",
        "sha256",
        "tokenization",
        "source_corpus",
    }
    exact_keys(value, keys, "token_fixture")
    require(value["encoding"] == "u32le", "token_fixture.encoding must be u32le")
    path = source_path(root, value["path"], "token_fixture.path")
    require(path.is_file(), f"token fixture is missing: {value['path']}")
    byte_count = integer(value["bytes"], "token_fixture.bytes", 4, MAX_TOKENS * 4)
    digest = text_value(value["sha256"], "token_fixture.sha256")
    require(
        len(digest) == SHA256_LENGTH and digest == digest.lower(),
        "token_fixture.sha256 must be lowercase SHA-256",
    )
    require(
        path.stat().st_size == byte_count,
        "token fixture byte count differs from the manifest",
    )
    require(
        sha256_file(path) == digest, "token fixture digest differs from the manifest"
    )
    tokens = read_u32le_tokens(path)
    parse_tokenization(value["tokenization"], len(tokens))
    parse_source_corpus(value["source_corpus"], root)
    return dict(value), tokens


def parse_tokenization(value: Any, token_count: int) -> None:
    require(isinstance(value, dict), "token_fixture.tokenization must be an object")
    exact_keys(
        value,
        {
            "token_limit",
            "add_bos",
            "parse_special",
            "llama_revision",
            "reference_tokenizer_model_sha256",
        },
        "token_fixture.tokenization",
    )
    limit = integer(
        value["token_limit"],
        "token_fixture.tokenization.token_limit",
        1,
        MAX_TOKENS,
    )
    require(limit == token_count, "token fixture count differs from its token limit")
    require(
        type(value["add_bos"]) is bool,
        "token_fixture.tokenization.add_bos must be boolean",
    )
    require(
        type(value["parse_special"]) is bool,
        "token_fixture.tokenization.parse_special must be boolean",
    )
    parse_git_commit(
        text_value(
            value["llama_revision"], "token_fixture.tokenization.llama_revision"
        ),
        "token_fixture.tokenization.llama_revision",
    )
    parse_digest(
        text_value(
            value["reference_tokenizer_model_sha256"],
            "token_fixture.tokenization.reference_tokenizer_model_sha256",
        ),
        "token_fixture.tokenization.reference_tokenizer_model_sha256",
    )


def parse_model_contract(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "model_contract must be an object")
    exact_keys(
        value,
        {"architecture", "tokenizer", "vocab", "subject_sha256"},
        "model_contract",
    )
    text_value(value["architecture"], "model_contract.architecture")
    text_value(value["tokenizer"], "model_contract.tokenizer")
    integer(value["vocab"], "model_contract.vocab", 1, MAX_TOKENS)
    parse_digest(
        text_value(value["subject_sha256"], "model_contract.subject_sha256"),
        "model_contract.subject_sha256",
    )
    return dict(value)


def parse_source_corpus(value: Any, root: Path) -> None:
    require(isinstance(value, dict), "token_fixture.source_corpus must be an object")
    exact_keys(value, {"path", "bytes", "sha256"}, "token_fixture.source_corpus")
    path = source_path(root, value["path"], "token_fixture.source_corpus.path")
    require(path.is_file(), "token fixture source corpus is missing")
    size = integer(
        value["bytes"], "token_fixture.source_corpus.bytes", 1, MAX_TOKENS * 64
    )
    digest = text_value(value["sha256"], "token_fixture.source_corpus.sha256")
    parse_digest(digest, "token_fixture.source_corpus.sha256")
    require(
        path.stat().st_size == size, "token fixture source corpus byte count differs"
    )
    require(sha256_file(path) == digest, "token fixture source corpus digest differs")


def parse_slices(value: Any, tokens: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    require(isinstance(value, dict) and value, "slices must be a nonempty object")
    result: dict[str, tuple[int, ...]] = {}
    for name, spec in value.items():
        parse_slice(name, spec, tokens, result)
    return result


def parse_slice(
    name: Any, spec: Any, tokens: tuple[int, ...], result: dict[str, tuple[int, ...]]
) -> None:
    key = text_value(name, "slice name")
    require(key not in result, f"slice repeats: {key}")
    require(isinstance(spec, dict), f"slice {key} must be an object")
    exact_keys(spec, {"offset", "count"}, f"slice {key}")
    offset = integer(spec["offset"], f"slice {key}.offset", 0, len(tokens))
    count = integer(spec["count"], f"slice {key}.count", 1, len(tokens))
    require(offset + count <= len(tokens), f"slice {key} leaves the token fixture")
    selected = tokens[offset : offset + count]
    require(
        all(token <= 0x7FFFFFFF for token in selected),
        f"slice {key} contains a token outside llama_token",
    )
    result[key] = selected


def resolve_tokens(
    value: Any, slices: dict[str, tuple[int, ...]], field: str
) -> tuple[int, ...]:
    require(
        isinstance(value, list) and value,
        f"{field} must be a nonempty slice-name array",
    )
    names = [text_value(item, field) for item in value]
    require(all(name in slices for name in names), f"{field} names an unknown slice")
    return tuple(token for name in names for token in slices[name])


def output_indices(value: Any, token_count: int, field: str) -> tuple[int, ...]:
    if value == "none":
        return ()
    if value == "last":
        return (token_count - 1,)
    if value == "all":
        return tuple(range(token_count))
    require(
        isinstance(value, list) and value,
        f"{field} must be none, last, all, or an index array",
    )
    indices = tuple(integer(item, field, 0, token_count - 1) for item in value)
    require(
        tuple(sorted(set(indices))) == indices,
        f"{field} must contain unique sorted indices",
    )
    return indices


def parse_stream(
    value: Any,
    slices: dict[str, tuple[int, ...]],
    context: Context,
    field: str,
    outputs: bool,
) -> Stream:
    require(isinstance(value, dict), f"{field} must be an object")
    expected = (
        {"seq", "position", "tokens", "outputs"}
        if outputs
        else {"seq", "position", "tokens"}
    )
    exact_keys(value, expected, field)
    tokens = resolve_tokens(value["tokens"], slices, f"{field}.tokens")
    seq = integer(value["seq"], f"{field}.seq", 0, context.n_seq_max - 1)
    position = integer(value["position"], f"{field}.position", 0, context.n_ctx - 1)
    selected = (
        output_indices(value["outputs"], len(tokens), f"{field}.outputs")
        if outputs
        else ()
    )
    return Stream(seq, position, tokens, selected)


def parse_decode(
    value: dict[str, Any],
    slices: dict[str, tuple[int, ...]],
    context: Context,
    field: str,
) -> Event:
    exact_keys(value, {"kind", "name", "streams"}, field)
    streams = value["streams"]
    require(
        isinstance(streams, list) and 1 <= len(streams) <= MAX_STREAMS,
        f"{field}.streams has invalid length",
    )
    parsed = tuple(
        parse_stream(item, slices, context, f"{field}.streams[{index}]", True)
        for index, item in enumerate(streams)
    )
    require(
        len({stream.seq for stream in parsed}) == len(parsed),
        f"{field} repeats a sequence",
    )
    return Event("decode", text_value(value["name"], f"{field}.name"), parsed)


def parse_copy(value: dict[str, Any], context: Context, field: str) -> Event:
    exact_keys(value, {"kind", "name", "source", "target", "p0", "p1"}, field)
    source = integer(value["source"], f"{field}.source", 0, context.n_seq_max - 1)
    target = integer(value["target"], f"{field}.target", 0, context.n_seq_max - 1)
    p0 = integer(value["p0"], f"{field}.p0", 0, context.n_ctx - 1)
    p1 = integer(value["p1"], f"{field}.p1", -1, context.n_ctx)
    require(source != target, f"{field} source and target must differ")
    require(p1 == -1 or p1 > p0, f"{field}.p1 must exceed p0 or equal -1")
    return Event(
        "copy",
        text_value(value["name"], f"{field}.name"),
        source=source,
        target=target,
        p0=p0,
        p1=p1,
    )


def parse_remove(value: dict[str, Any], context: Context, field: str) -> Event:
    exact_keys(value, {"kind", "name", "sequence", "p0", "p1"}, field)
    sequence = integer(value["sequence"], f"{field}.sequence", 0, context.n_seq_max - 1)
    p0 = integer(value["p0"], f"{field}.p0", 0, context.n_ctx - 1)
    p1 = integer(value["p1"], f"{field}.p1", -1, context.n_ctx)
    require(p1 == -1 or p1 > p0, f"{field}.p1 must exceed p0 or equal -1")
    return Event(
        "remove",
        text_value(value["name"], f"{field}.name"),
        sequence=sequence,
        p0=p0,
        p1=p1,
    )


def parse_stepwise(
    value: dict[str, Any],
    slices: dict[str, tuple[int, ...]],
    context: Context,
    field: str,
) -> tuple[Event, ...]:
    exact_keys(value, {"kind", "name", "streams", "capture_steps"}, field)
    raw_streams = value["streams"]
    require(
        isinstance(raw_streams, list) and 1 <= len(raw_streams) <= MAX_STREAMS,
        f"{field}.streams has invalid length",
    )
    streams = tuple(
        parse_stream(item, slices, context, f"{field}.streams[{index}]", False)
        for index, item in enumerate(raw_streams)
    )
    require(
        len({stream.seq for stream in streams}) == len(streams),
        f"{field} repeats a sequence",
    )
    counts = {len(stream.tokens) for stream in streams}
    require(len(counts) == 1, f"{field} streams must have equal token counts")
    step_count = next(iter(counts))
    raw_capture = value["capture_steps"]
    require(
        isinstance(raw_capture, list) and raw_capture,
        f"{field}.capture_steps must be nonempty",
    )
    capture = tuple(
        integer(item, f"{field}.capture_steps", 0, step_count - 1)
        for item in raw_capture
    )
    require(
        tuple(sorted(set(capture))) == capture,
        f"{field}.capture_steps must be unique and sorted",
    )
    name = text_value(value["name"], f"{field}.name")
    return tuple(
        step_event(name, streams, step, step in capture) for step in range(step_count)
    )


def step_event(
    name: str, streams: tuple[Stream, ...], step: int, capture: bool
) -> Event:
    selected = (0,) if capture else ()
    items = tuple(
        Stream(stream.seq, stream.position + step, (stream.tokens[step],), selected)
        for stream in streams
    )
    return Event("decode", f"{name}_{step}", items)


def parse_operation(
    value: Any, slices: dict[str, tuple[int, ...]], context: Context, field: str
) -> tuple[Event, ...]:
    require(isinstance(value, dict), f"{field} must be an object")
    kind = value.get("kind")
    if kind == "decode":
        return (parse_decode(value, slices, context, field),)
    if kind == "copy":
        return (parse_copy(value, context, field),)
    if kind == "remove":
        return (parse_remove(value, context, field),)
    require(kind == "decode_stepwise", f"{field}.kind is unknown")
    return parse_stepwise(value, slices, context, field)


def memory_ranges(state: list[set[int]]) -> tuple[tuple[int, int], ...]:
    ranges = []
    for positions in state:
        ranges.append((-1, -1) if not positions else (min(positions), max(positions)))
    return tuple(ranges)


def replay_decode(
    event: Event, state: list[set[int]], context: Context, field: str
) -> int:
    submitted = sum(len(stream.tokens) for stream in event.streams)
    require(submitted <= context.n_batch, f"{field} exceeds context.n_batch")
    for stream in event.streams:
        positions = set(range(stream.position, stream.position + len(stream.tokens)))
        require(max(positions) < context.n_ctx, f"{field} exceeds context.n_ctx")
        require(
            not state[stream.seq].intersection(positions),
            f"{field} overwrites sequence {stream.seq}",
        )
        expected = 0 if not state[stream.seq] else max(state[stream.seq]) + 1
        require(
            stream.position == expected,
            f"{field} starts at {stream.position}, expected {expected}",
        )
        state[stream.seq].update(positions)
    return submitted


def replay_copy(event: Event, state: list[set[int]], field: str) -> None:
    upper = event.p1 if event.p1 >= 0 else sys.maxsize
    copied = {
        position for position in state[event.source] if event.p0 <= position < upper
    }
    require(copied, f"{field} copies no positions")
    require(not state[event.target], f"{field} target sequence is not empty")
    require(
        copied == set(range(min(copied), max(copied) + 1)),
        f"{field} source range is not contiguous",
    )
    state[event.target].update(copied)


def replay_remove(event: Event, state: list[set[int]], field: str) -> None:
    upper = event.p1 if event.p1 >= 0 else sys.maxsize
    removed = {
        position for position in state[event.sequence] if event.p0 <= position < upper
    }
    require(removed, f"{field} removes no positions")
    state[event.sequence].difference_update(removed)


def replay_case(
    events: tuple[Event, ...], context: Context, field: str
) -> tuple[tuple[tuple[int, int], ...], ...]:
    state = [set() for _ in range(context.n_seq_max)]
    ranges = []
    submitted = 0
    for index, event in enumerate(events):
        event_field = f"{field}.events[{index}]"
        if event.kind == "decode":
            submitted += replay_decode(event, state, context, event_field)
        elif event.kind == "copy":
            replay_copy(event, state, event_field)
        else:
            replay_remove(event, state, event_field)
        ranges.append(memory_ranges(state))
    require(
        submitted <= context.n_ctx,
        f"{field} submits {submitted} cache entries, exceeding context.n_ctx",
    )
    return tuple(ranges)


def parse_case(
    value: Any, slices: dict[str, tuple[int, ...]], context: Context, index: int
) -> Case:
    field = f"cases[{index}]"
    require(isinstance(value, dict), f"{field} must be an object")
    exact_keys(value, {"name", "class", "operations"}, field)
    operations = value["operations"]
    require(
        isinstance(operations, list) and operations,
        f"{field}.operations must be nonempty",
    )
    events = tuple(
        event
        for op_index, operation in enumerate(operations)
        for event in parse_operation(
            operation, slices, context, f"{field}.operations[{op_index}]"
        )
    )
    require(len(events) <= MAX_EVENTS, f"{field} exceeds {MAX_EVENTS} expanded events")
    names = [event.name for event in events]
    require(len(set(names)) == len(names), f"{field} expanded event names repeat")
    ranges = replay_case(events, context, field)
    return Case(
        text_value(value["name"], f"{field}.name"),
        text_value(value["class"], f"{field}.class"),
        events,
        ranges,
    )


def load_workload(manifest: Path, root: Path) -> Workload:
    data = load_json(manifest)
    keys = {
        "schema",
        "phase",
        "model_contract",
        "token_fixture",
        "context",
        "slices",
        "cases",
    }
    exact_keys(data, keys, "manifest")
    require(
        data["schema"] == MANIFEST_SCHEMA, f"manifest.schema must be {MANIFEST_SCHEMA}"
    )
    phase = text_value(data["phase"], "manifest.phase")
    model_contract = parse_model_contract(data["model_contract"])
    context = parse_context(data["context"])
    fixture, tokens = parse_fixture(data["token_fixture"], root)
    slices = parse_slices(data["slices"], tokens)
    raw_cases = data["cases"]
    require(
        isinstance(raw_cases, list) and 1 <= len(raw_cases) <= MAX_CASES,
        "cases has invalid length",
    )
    cases = tuple(
        parse_case(value, slices, context, index)
        for index, value in enumerate(raw_cases)
    )
    names = [case.name for case in cases]
    require(len(set(names)) == len(names), "case names repeat")
    return Workload(
        phase,
        model_contract,
        context,
        fixture,
        cases,
        sha256_file(manifest),
    )


def write_u32(output: bytearray, value: int) -> None:
    output.extend(struct.pack("<I", value))


def write_i32(output: bytearray, value: int) -> None:
    output.extend(struct.pack("<i", value))


def write_string(output: bytearray, value: str) -> None:
    encoded = value.encode("utf-8")
    write_u32(output, len(encoded))
    output.extend(encoded)


def write_stream(output: bytearray, stream: Stream) -> None:
    write_i32(output, stream.seq)
    write_i32(output, stream.position)
    write_u32(output, len(stream.tokens))
    selected = set(stream.outputs)
    for index, token in enumerate(stream.tokens):
        write_i32(output, token)
        output.append(1 if index in selected else 0)


def write_event(output: bytearray, event: Event) -> None:
    output.append({"decode": 1, "copy": 2, "remove": 3}[event.kind])
    write_string(output, event.name)
    if event.kind == "decode":
        write_u32(output, len(event.streams))
        for stream in event.streams:
            write_stream(output, stream)
    elif event.kind == "copy":
        for value in (event.source, event.target, event.p0, event.p1):
            write_i32(output, value)
    else:
        for value in (event.sequence, event.p0, event.p1):
            write_i32(output, value)


def encode_plan(workload: Workload) -> bytes:
    output = bytearray(PLAN_MAGIC)
    write_u32(output, PLAN_VERSION)
    write_u32(output, len(workload.cases))
    for case in workload.cases:
        write_string(output, case.name)
        write_u32(output, len(case.events))
        for event in case.events:
            write_event(output, event)
    return bytes(output)


def stream_json(stream: Stream) -> dict[str, Any]:
    return {
        "seq": stream.seq,
        "position": stream.position,
        "tokens": list(stream.tokens),
        "output_indices": list(stream.outputs),
    }


def event_json(event: Event) -> dict[str, Any]:
    if event.kind == "decode":
        return {
            "kind": "decode",
            "name": event.name,
            "streams": [stream_json(stream) for stream in event.streams],
        }
    if event.kind == "copy":
        return {
            "kind": "copy",
            "name": event.name,
            "source": event.source,
            "target": event.target,
            "p0": event.p0,
            "p1": event.p1,
        }
    return {
        "kind": "remove",
        "name": event.name,
        "sequence": event.sequence,
        "p0": event.p0,
        "p1": event.p1,
    }


def context_json(context: Context) -> dict[str, Any]:
    return dataclasses.asdict(context)


def workload_json(workload: Workload) -> dict[str, Any]:
    return {
        "manifest_sha256": workload.manifest_sha256,
        "model_contract": workload.model_contract,
        "token_fixture": workload.token_fixture,
        "context": context_json(workload.context),
        "cases": [case_json(case) for case in workload.cases],
    }


def case_json(case: Case) -> dict[str, Any]:
    decode_tokens = sum(
        len(stream.tokens) for event in case.events for stream in event.streams
    )
    logit_rows = sum(
        len(stream.outputs) for event in case.events for stream in event.streams
    )
    return {
        "name": case.name,
        "class": case.case_class,
        "decode_tokens_submitted": decode_tokens,
        "seq_copy_calls": sum(event.kind == "copy" for event in case.events),
        "seq_remove_calls": sum(event.kind == "remove" for event in case.events),
        "logit_rows": logit_rows,
        "events": [event_json(event) for event in case.events],
    }


def parse_digest(value: str, field: str) -> str:
    require(
        len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value),
        f"{field} is not lowercase SHA-256",
    )
    return value


def parse_git_commit(value: str, field: str) -> str:
    require(
        len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{field} is not a lowercase Git commit",
    )
    return value


def git_output(arguments: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *arguments], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValidationError(f"git command failed: {' '.join(arguments)}") from error


def validate_llama_checkout(llama_dir: Path, pin_file: Path) -> str:
    require((llama_dir / ".git").exists(), "pinned llama.cpp checkout is missing")
    pin = pin_file.read_text().strip()
    parse_git_commit(pin, "external/PINNED")
    revision = git_output(["-C", str(llama_dir), "rev-parse", "HEAD"])
    require(
        revision == pin,
        f"llama.cpp revision {revision} differs from external/PINNED {pin}",
    )
    status = git_output(
        ["-C", str(llama_dir), "status", "--porcelain", "--untracked-files=all"]
    )
    require(not status, "pinned llama.cpp checkout has local changes")
    return revision


def compile_prefix(cpuset: str) -> list[str]:
    if cpuset and shutil.which("taskset"):
        return ["taskset", "-c", cpuset]
    return []


def library_inputs(llama_dir: Path, backend: str) -> list[Path]:
    library_dir = llama_dir / "build/bin"
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    names = [f"libllama{suffix}", f"libggml{suffix}", f"libggml-base{suffix}"]
    paths = [library_dir / name for name in names]
    require(
        all(path.exists() for path in paths), "llama.cpp shared libraries are missing"
    )
    return paths + backend_libraries(llama_dir, backend)


def backend_libraries(llama_dir: Path, backend: str) -> list[Path]:
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    library_dir = llama_dir / "build/bin"
    candidates = sorted(library_dir.glob(f"libggml-{backend}*{suffix}"))
    require(candidates, f"llama.cpp {backend} backend library is missing")
    return candidates


def compiler_identity(compiler: str) -> dict[str, str]:
    resolved = shutil.which(compiler)
    require(resolved is not None, f"C++ compiler is missing: {compiler}")
    version = subprocess.check_output([resolved, "--version"], text=True).splitlines()[
        0
    ]
    return {"name": Path(resolved).name, "version": version}


def build_comparator(
    root: Path, llama_dir: Path, binary: Path, cpuset: str
) -> tuple[list[str], dict[str, str]]:
    source = root / "research/prefix_attention/llama_cached_comparator.cpp"
    require(source.is_file(), "comparator source is missing")
    compiler = os.environ.get("CXX", "c++")
    identity = compiler_identity(compiler)
    library_dir = llama_dir / "build/bin"
    prefix = compile_prefix(cpuset)
    logical = prefix + [
        identity["name"],
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I$LLAMA_CPP/include",
        "-I$LLAMA_CPP/ggml/include",
        "$SOURCE",
        "-L$LLAMA_CPP/build/bin",
        "-Wl,-rpath,$LLAMA_CPP/build/bin",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-o",
        "$BINARY",
    ]
    command = prefix + [
        compiler,
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{llama_dir / 'include'}",
        f"-I{llama_dir / 'ggml/include'}",
        str(source),
        f"-L{library_dir}",
        f"-Wl,-rpath,{library_dir.resolve()}",
        "-lllama",
        "-lggml",
        "-lggml-base",
        "-o",
        str(binary),
    ]
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    return logical, identity


def resolve_linked_library(
    candidate: str, binary: Path, llama_dir: Path
) -> Path | None:
    library_dir = llama_dir / "build/bin"
    prefixes = {
        "@rpath/": library_dir,
        "@loader_path/": binary.parent,
        "@executable_path/": binary.parent,
    }
    path = Path(candidate)
    for prefix, directory in prefixes.items():
        if candidate.startswith(prefix):
            path = directory / candidate.removeprefix(prefix)
            break
    if not path.is_file():
        return None
    try:
        path.resolve().relative_to(llama_dir.resolve())
    except ValueError:
        return None
    return path


def linked_libraries(binary: Path, llama_dir: Path) -> list[dict[str, Any]]:
    if platform.system() == "Darwin":
        output = subprocess.check_output(["otool", "-L", str(binary)], text=True)
        candidates = [line.strip().split(" ", 1)[0] for line in output.splitlines()[1:]]
    else:
        output = subprocess.check_output(["ldd", str(binary)], text=True)
        candidates = [
            line.split("=>", 1)[1].strip().split(" ", 1)[0]
            for line in output.splitlines()
            if "=>" in line
        ]
    result = []
    for candidate in candidates:
        path = resolve_linked_library(candidate, binary, llama_dir)
        if path is not None:
            result.append(
                {
                    "name": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    require(
        any(item["name"].startswith("libllama") for item in result),
        "linked llama.cpp library was not resolved",
    )
    return sorted(result, key=lambda item: item["name"])


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "research/prefix_attention/llama_cached_workload.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "metal"), required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--build-cpuset", default="16-31")
    parser.add_argument("--run-cpuset", default="0-3,12-15")
    parser.add_argument(
        "--subject-model",
        type=Path,
        help="oracle mode: --model is the pinned independent oracle for this subject model",
    )
    parser.add_argument(
        "--oracle-identity",
        type=Path,
        help="oracle mode: output path of the hash-linked oracle identity record",
    )
    args = parser.parse_args()
    if (args.subject_model is None) != (args.oracle_identity is None):
        parser.error("--subject-model and --oracle-identity go together")
    return args


def parse_workload_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Write the comparator workload for the runtime calibration cases."
    )
    parser.add_argument("--write-calibration-workload", type=Path, required=True)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=root / "research/prefix_attention/runtime_manifest.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "research/prefix_attention/llama_cached_workload.json",
        help="source workload that supplies the model contract, fixture, and context",
    )
    return parser.parse_args()


def write_calibration_workload(args: argparse.Namespace) -> Path:
    """Write the generated calibration workload. An existing output is never replaced."""
    workload = common_oracle.calibration_workload(
        load_json(args.runtime_manifest), load_json(args.manifest)
    )
    with args.write_calibration_workload.open("xb") as output:
        output.write(common_oracle.workload_bytes(workload))
    return args.write_calibration_workload


def comparator_binary(args: argparse.Namespace) -> Path:
    if args.binary:
        return args.binary.resolve()
    return args.output.parent.resolve() / "llama-cached-comparator"


def command_for_run(
    binary: Path,
    model: Path,
    plan: Path,
    logits: Path,
    backend: str,
    context: Context,
    cpuset: str,
) -> tuple[list[str], list[str]]:
    values = [
        backend,
        str(context.n_ctx),
        str(context.n_batch),
        str(context.n_ubatch),
        str(context.n_seq_max),
        str(context.threads),
        context.kv_type,
        context.flash_attention,
        str(context.kv_unified).lower(),
        str(context.swa_full).lower(),
    ]
    prefix = compile_prefix(cpuset)
    command = prefix + [str(binary), str(model), str(plan), str(logits), *values]
    logical = prefix + ["$BINARY", "$MODEL", "$PLAN", "$LOGITS", *values]
    return command, logical


def read_transcript(stdout: str) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(stdout.splitlines(), 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(
                f"engine line {line_number} is not JSON: {error}"
            ) from error
        require(isinstance(record, dict), f"engine line {line_number} is not an object")
        records.append(record)
    require(records, "engine returned no records")
    return records


def range_json(ranges: tuple[tuple[int, int], ...]) -> list[dict[str, int]]:
    return [
        {"seq": seq, "min": pair[0], "max": pair[1]} for seq, pair in enumerate(ranges)
    ]


def expected_logits(
    case_index: int, event_index: int, event: Event
) -> list[dict[str, int]]:
    rows = []
    batch_index = 0
    for stream_index, stream in enumerate(event.streams):
        for token_index in range(len(stream.tokens)):
            if token_index in stream.outputs:
                position = stream.position + token_index
                rows.append(
                    {
                        "case": case_index,
                        "event": event_index,
                        "stream": stream_index,
                        "token_index": token_index,
                        "batch_index": batch_index,
                        "seq": stream.seq,
                        "input_position": position,
                        "predicted_position": position + 1,
                    }
                )
            batch_index += 1
    return rows


def validate_header(record: dict[str, Any], workload: Workload, backend: str) -> None:
    keys = {
        "kind",
        "protocol",
        "requested_backend",
        "device",
        "model",
        "context",
        "model_load_ns",
        "context_create_ns",
    }
    exact_keys(record, keys, "engine header")
    require(
        record["kind"] == "header" and record["protocol"] == ENGINE_PROTOCOL,
        "engine protocol differs",
    )
    require(record["requested_backend"] == backend, "engine requested backend differs")
    validate_device(record["device"], backend)
    validate_model(record["model"])
    actual_contract = {
        field: record["model"][field]
        for field in ("architecture", "tokenizer", "vocab")
    }
    expected_contract = {
        field: workload.model_contract[field]
        for field in ("architecture", "tokenizer", "vocab")
    }
    require(
        actual_contract == expected_contract,
        "engine model differs from the workload model contract",
    )
    validate_effective_context(record["context"], workload.context)
    requested = record["context"]["requested"]
    require(
        requested == context_json(workload.context), "engine requested context differs"
    )
    integer(record["model_load_ns"], "engine model_load_ns", 0, 2**63 - 1)
    integer(record["context_create_ns"], "engine context_create_ns", 0, 2**63 - 1)


def validate_device(value: Any, backend: str) -> None:
    require(isinstance(value, dict), "engine device metadata is missing")
    keys = {
        "backend",
        "registry",
        "name",
        "description",
        "type",
        "memory_free_before_model",
        "memory_total_before_model",
        "memory_free_after_context",
        "memory_total_after_context",
        "backend_module_path",
        "backend_module_dynamic",
    }
    exact_keys(value, keys, "engine device")
    require(value["backend"] == backend, "selected engine backend differs")
    for field in ("registry", "name", "description"):
        text_value(value[field], f"engine device {field}")
    require(
        isinstance(value["backend_module_path"], str),
        "engine device backend_module_path must be a string",
    )
    require(
        type(value["backend_module_dynamic"]) is bool,
        "engine device backend_module_dynamic must be boolean",
    )
    integer(value["type"], "engine device type", 0, 3)
    for field in (
        "memory_free_before_model",
        "memory_total_before_model",
        "memory_free_after_context",
        "memory_total_after_context",
    ):
        integer(value[field], f"engine device {field}", 0, 2**64 - 1)
    require(
        value["memory_free_before_model"] <= value["memory_total_before_model"],
        "engine pre-load device memory is invalid",
    )
    require(
        value["memory_free_after_context"] <= value["memory_total_after_context"],
        "engine post-context device memory is invalid",
    )


def validate_model(value: Any) -> None:
    require(isinstance(value, dict), "engine model metadata is missing")
    keys = {
        "vocab",
        "context_train",
        "size_bytes",
        "ftype",
        "ftype_name",
        "architecture",
        "tokenizer",
    }
    exact_keys(value, keys, "engine model")
    integer(value["vocab"], "engine model vocab", 1, MAX_TOKENS)
    integer(value["context_train"], "engine model context_train", 1, 2**31 - 1)
    integer(value["size_bytes"], "engine model size_bytes", 1, 2**64 - 1)
    integer(value["ftype"], "engine model ftype", 0, 2**31 - 1)
    for field in ("ftype_name", "architecture", "tokenizer"):
        text_value(value[field], f"engine model {field}")


def validate_effective_context(value: Any, requested: Context) -> None:
    require(isinstance(value, dict), "engine context metadata is missing")
    exact_keys(value, {"requested", "effective"}, "engine context")
    effective = value["effective"]
    require(isinstance(effective, dict), "engine effective context is missing")
    exact_keys(
        effective,
        {"n_ctx", "n_ctx_seq", "n_batch", "n_ubatch", "n_seq_max"},
        "engine effective context",
    )
    for field in effective:
        integer(effective[field], f"engine effective context {field}", 1, MAX_TOKENS)
    require(
        effective["n_ctx"] >= requested.n_ctx,
        "engine effective n_ctx is smaller than requested",
    )
    require(
        effective["n_batch"] == requested.n_batch, "engine effective n_batch differs"
    )
    require(
        effective["n_ubatch"] == requested.n_ubatch, "engine effective n_ubatch differs"
    )
    require(
        effective["n_seq_max"] == requested.n_seq_max,
        "engine effective n_seq_max differs",
    )


def validate_event_record(
    record: dict[str, Any], case_index: int, event_index: int, case: Case
) -> None:
    event = case.events[event_index]
    keys = {
        "kind",
        "case",
        "event",
        "name",
        "operation",
        "duration_ns",
        "submitted_tokens",
        "output_rows",
        "ranges",
    }
    exact_keys(record, keys, f"engine event {case_index}/{event_index}")
    require(record["kind"] == "event", "engine event kind differs")
    integer(record["case"], "engine event case", 0, MAX_CASES - 1)
    integer(record["event"], "engine event event", 0, MAX_EVENTS - 1)
    require(
        record["case"] == case_index and record["event"] == event_index,
        "engine event index differs",
    )
    require(
        record["name"] == event.name and record["operation"] == event.kind,
        "engine event identity differs",
    )
    integer(record["duration_ns"], "engine event duration_ns", 0, 2**63 - 1)
    submitted = (
        sum(len(stream.tokens) for stream in event.streams)
        if event.kind == "decode"
        else 0
    )
    outputs = (
        sum(len(stream.outputs) for stream in event.streams)
        if event.kind == "decode"
        else 0
    )
    integer(record["submitted_tokens"], "engine submitted_tokens", 0, MAX_TOKENS)
    integer(record["output_rows"], "engine output_rows", 0, MAX_TOKENS)
    require(
        record["submitted_tokens"] == submitted, "engine submitted-token count differs"
    )
    require(record["output_rows"] == outputs, "engine output-row count differs")
    validate_ranges(record["ranges"], case.expected_ranges[event_index])
    require(
        record["ranges"] == range_json(case.expected_ranges[event_index]),
        "engine memory ranges differ",
    )


def validate_ranges(value: Any, expected: tuple[tuple[int, int], ...]) -> None:
    require(
        isinstance(value, list) and len(value) == len(expected),
        "engine memory range count differs",
    )
    for sequence, item in enumerate(value):
        require(isinstance(item, dict), "engine memory range is not an object")
        exact_keys(item, {"seq", "min", "max"}, "engine memory range")
        integer(item["seq"], "engine memory range seq", 0, MAX_STREAMS - 1)
        integer(item["min"], "engine memory range min", -1, MAX_TOKENS)
        integer(item["max"], "engine memory range max", -1, MAX_TOKENS)
        require(item["seq"] == sequence, "engine memory range sequence differs")


def validate_logit_record(
    record: dict[str, Any], expected: dict[str, int], offset: int, vocab: int
) -> int:
    keys = {
        "kind",
        "case",
        "event",
        "stream",
        "token_index",
        "batch_index",
        "seq",
        "input_position",
        "predicted_position",
        "offset_bytes",
        "float_count",
    }
    exact_keys(record, keys, "engine logit row")
    require(record["kind"] == "logit", "engine logit kind differs")
    for field, value in expected.items():
        integer(record[field], f"engine logit {field}", 0, MAX_TOKENS)
        require(record[field] == value, f"engine logit {field} differs")
    integer(record["offset_bytes"], "engine logit offset_bytes", 0, 2**64 - 1)
    integer(record["float_count"], "engine logit float_count", 1, MAX_TOKENS)
    require(record["offset_bytes"] == offset, "engine logit offsets are not contiguous")
    require(record["float_count"] == vocab, "engine logit vocabulary differs")
    return offset + vocab * 4


def validate_engine_records(
    records: list[dict[str, Any]], workload: Workload, backend: str, logits_bytes: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = records[0]
    validate_header(header, workload, backend)
    vocab = header["model"]["vocab"]
    cursor = 1
    offset = 0
    measurements = []
    for case_index, case in enumerate(workload.cases):
        case_events = []
        for event_index, event in enumerate(case.events):
            require(cursor < len(records), "engine omitted an event record")
            validate_event_record(records[cursor], case_index, event_index, case)
            event_record = records[cursor]
            cursor += 1
            rows = []
            for expected in expected_logits(case_index, event_index, event):
                require(cursor < len(records), "engine omitted a logit record")
                offset = validate_logit_record(records[cursor], expected, offset, vocab)
                rows.append(records[cursor])
                cursor += 1
            case_events.append({"event": event_record, "logit_rows": rows})
        measurements.append({"name": case.name, "events": case_events})
    require(cursor == len(records), "engine returned unexpected records")
    require(offset == logits_bytes, "logit sidecar size differs from engine rows")
    return header, measurements


def file_identity(path: Path, name: str | None = None) -> dict[str, Any]:
    return {
        "name": name or path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_subject_model(
    model: dict[str, Any], model_contract: dict[str, Any]
) -> None:
    require(
        model["sha256"] == model_contract["subject_sha256"],
        "selected model differs from the workload subject model",
    )


def resolve_backend_module(path_text: str, llama_dir: Path) -> Path:
    candidate = Path(path_text)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    candidates = [llama_dir / candidate, llama_dir / "build/bin" / candidate]
    for path in candidates:
        if path.is_file():
            return path
    raise ValidationError("resolved backend module is missing")


def backend_module_identity(
    device: dict[str, Any],
    llama_dir: Path,
    library_paths: list[Path],
    before_libraries: list[dict[str, Any]],
) -> dict[str, Any]:
    path_text = device["backend_module_path"]
    require(path_text, "resolved backend module path is unavailable")
    path = resolve_backend_module(path_text, llama_dir)
    trusted = {
        candidate.resolve(): identity
        for candidate, identity in zip(library_paths, before_libraries, strict=True)
    }
    resolved = path.resolve()
    require(
        resolved in trusted,
        "resolved backend module is outside the trusted llama.cpp artifacts",
    )
    identity = file_identity(path)
    require(
        all(
            identity[field] == trusted[resolved][field]
            for field in ("bytes", "sha256")
        ),
        "resolved backend module changed while the comparator ran",
    )
    return identity


def normalize_engine_header(
    header: dict[str, Any], backend_library: dict[str, Any]
) -> dict[str, Any]:
    normalized = dict(header)
    device = dict(header["device"])
    device.pop("backend_module_path")
    device["backend_module_name"] = backend_library["name"]
    device["backend_module_bytes"] = backend_library["bytes"]
    device["backend_module_sha256"] = backend_library["sha256"]
    normalized["device"] = device
    return normalized


def source_inputs(root: Path, llama_dir: Path) -> list[dict[str, Any]]:
    paths = [
        root / "research/prefix_attention/llama_cached_comparator.cpp",
        root / "research/prefix_attention/run_llama_cached_comparator.py",
        root / "research/prefix_attention/llama_cached_comparator.schema.json",
        llama_dir / "include/llama.h",
        llama_dir / "ggml/include/ggml-backend.h",
    ]
    names = [
        str(paths[0].relative_to(root)),
        str(paths[1].relative_to(root)),
        str(paths[2].relative_to(root)),
        "$LLAMA_CPP/include/llama.h",
        "$LLAMA_CPP/ggml/include/ggml-backend.h",
    ]
    return [file_identity(path, name) for path, name in zip(paths, names, strict=True)]


def run_comparator(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        sys.stderr.write(error.stderr)
        raise ValidationError(
            f"cached comparator failed with status {error.returncode}"
        ) from error
    require(
        not result.stderr, f"cached comparator wrote stderr: {result.stderr.strip()}"
    )
    return result.stdout


def refuse_output(path: Path, field: str) -> None:
    require(not os.path.lexists(path), f"{field} already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def make_temporary_sidecar(path: Path) -> Path:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        return Path(handle.name)


def verify_stable_inputs(
    root: Path,
    llama_dir: Path,
    model: Path,
    manifest: Path,
    before_model: dict[str, Any],
    before_sources: list[dict[str, Any]],
    before_libraries: list[dict[str, Any]],
    library_paths: list[Path],
    manifest_digest: str,
    revision: str,
    token_fixture: dict[str, Any],
) -> None:
    require(
        file_identity(model) == before_model, "model changed while the comparator ran"
    )
    require(
        source_inputs(root, llama_dir) == before_sources,
        "comparator source inputs changed while the comparator ran",
    )
    require(
        [file_identity(path) for path in library_paths] == before_libraries,
        "llama.cpp libraries changed while the comparator ran",
    )
    require(
        sha256_file(manifest) == manifest_digest,
        "workload manifest changed while the comparator ran",
    )
    parse_fixture(token_fixture, root)
    require(
        validate_llama_checkout(llama_dir, root / "external/PINNED") == revision,
        "llama.cpp revision changed while the comparator ran",
    )


def check_oracle_models(
    args: argparse.Namespace, workload: Workload, oracle: dict[str, Any]
) -> dict[str, Any]:
    """Bind the run model to its pinned oracle role for the workload subject.

    The oracle must be the pinned file for the subject. The two GGUF files must
    also agree on architecture and tokenizer metadata, so an unrelated model with
    the same vocabulary cannot pass.
    """
    subject = file_identity(args.subject_model)
    validate_subject_model(subject, workload.model_contract)
    pair = common_oracle.LOGIT_ORACLE_PAIRS.get(subject["sha256"])
    require(pair is not None, "no pinned oracle model exists for the subject model")
    require(oracle["sha256"] == pair["oracle_sha256"], "model is not the pinned oracle for the subject")
    contract = workload.model_contract
    require(
        (contract["architecture"], contract["tokenizer"]) == (pair["architecture"], pair["tokenizer"]),
        "workload contract differs from the pinned oracle pair",
    )
    subject_gguf = common_oracle.gguf_identity(args.subject_model)
    oracle_gguf = common_oracle.gguf_identity(args.model)
    require(subject_gguf == oracle_gguf, "subject and oracle GGUF metadata differ")
    require(oracle_gguf["architecture"] == pair["architecture"], "GGUF architecture differs from the pin")
    pin = common_oracle.LOGIT_ORACLE_METADATA.get(subject["sha256"])
    require(pin is not None, "no independent metadata pin exists for the subject model")
    require(all(oracle_gguf[key] == value for key, value in pin.items()), "GGUF metadata differ from the independent pin")
    return {
        "subject": {**subject, "gguf": subject_gguf},
        "oracle": {**oracle, "gguf": oracle_gguf},
        "pair": pair,
    }


def check_oracle_ftype(header: dict[str, Any], record: dict[str, Any]) -> None:
    expected = record["pair"]["oracle_ftype"]
    require(header["model"]["ftype"] == expected, "engine model file type differs from the pinned oracle")


def identity_text(record: dict[str, Any], receipt: dict[str, Any], receipt_text: str) -> str:
    linked = {
        "schema": ORACLE_IDENTITY_SCHEMA,
        "comparator_receipt_sha256": hashlib.sha256(receipt_text.encode()).hexdigest(),
        "logits_sha256": receipt["artifacts"]["logits"]["sha256"],
        "workload_sha256": receipt["workload"]["manifest_sha256"],
        "plan_sha256": receipt["provenance"]["plan_sha256"],
        **record,
    }
    return json.dumps(linked, indent=2, sort_keys=True) + "\n"


def create_receipt(
    args: argparse.Namespace, root: Path, oracle_sink: dict[str, Any] | None = None
) -> tuple[dict[str, Any], Path]:
    manifest = args.manifest.resolve()
    llama_dir = args.llama_dir.resolve()
    model = args.model.resolve()
    require(model.is_file(), "model file is missing")
    pin_file = root / "external/PINNED"
    revision = validate_llama_checkout(llama_dir, pin_file)
    workload = load_workload(manifest, root)
    require(
        workload.token_fixture["tokenization"]["llama_revision"] == revision,
        "token fixture llama.cpp revision differs from the active pin",
    )
    before_model = file_identity(model)
    if oracle_sink is None:
        validate_subject_model(before_model, workload.model_contract)
    else:
        oracle_sink.update(check_oracle_models(args, workload, before_model))
    before_sources = source_inputs(root, llama_dir)
    library_paths = library_inputs(llama_dir, args.backend)
    before_libraries = [file_identity(path) for path in library_paths]
    binary = comparator_binary(args)
    build_command, compiler = build_comparator(
        root, llama_dir, binary, args.build_cpuset
    )
    after_sources = source_inputs(root, llama_dir)
    require(
        before_sources == after_sources,
        "comparator build inputs changed during compilation",
    )
    libraries = linked_libraries(binary, llama_dir)
    plan_data = encode_plan(workload)
    with tempfile.NamedTemporaryFile(
        prefix="llama-cached-plan-", suffix=".bin", delete=False
    ) as temporary:
        temporary.write(plan_data)
        plan_path = Path(temporary.name)
    temporary_logits = make_temporary_sidecar(args.logits)
    try:
        command, logical_command = command_for_run(
            binary,
            model,
            plan_path,
            temporary_logits,
            args.backend,
            workload.context,
            args.run_cpuset,
        )
        transcript = read_transcript(run_comparator(command))
        require(
            temporary_logits.is_file(), "comparator did not create the logit sidecar"
        )
        header, measurements = validate_engine_records(
            transcript, workload, args.backend, temporary_logits.stat().st_size
        )
        if oracle_sink is not None:
            check_oracle_ftype(header, oracle_sink)
        actual_backend_library = backend_module_identity(
            header["device"], llama_dir, library_paths, before_libraries
        )
    except Exception:
        temporary_logits.unlink(missing_ok=True)
        raise
    finally:
        plan_path.unlink(missing_ok=True)
    try:
        verify_stable_inputs(
            root,
            llama_dir,
            model,
            manifest,
            before_model,
            before_sources,
            before_libraries,
            library_paths,
            workload.manifest_sha256,
            revision,
            workload.token_fixture,
        )
    except Exception:
        temporary_logits.unlink(missing_ok=True)
        raise
    normalized_header = normalize_engine_header(header, actual_backend_library)
    return {
        "schema": RECEIPT_SCHEMA,
        "phase": workload.phase,
        "quality": "unverified",
        "claim_scope": (
            "llama.cpp sequence-copy calls, submitted tokens, observed positions, "
            "timings, and raw logits. The receipt sets no hardware cache, performance, "
            "or quality threshold."
        ),
        "workload": workload_json(workload),
        "engine": normalized_header,
        "measurements": measurements,
        "artifacts": {
            "logits": file_identity(temporary_logits, args.logits.name),
            "encoding": "contiguous little-endian IEEE-754 float32 rows",
        },
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "host": {
                "system": platform.system(),
                "machine": platform.machine(),
                "release": platform.release(),
            },
            "llama_revision": revision,
            "pin_file_sha256": sha256_file(pin_file),
            "model": before_model,
            "comparator_binary": file_identity(binary),
            "linked_llama_libraries": libraries,
            "active_backend_library": actual_backend_library,
            "source_inputs": before_sources,
            "compiler": compiler,
            "build_command": build_command,
            "run_command": logical_command,
            "plan_sha256": hashlib.sha256(plan_data).hexdigest(),
        },
    }, temporary_logits


def write_outputs(
    receipt: dict[str, Any], temporary_logits: Path, output: Path, logits: Path
) -> None:
    temporary_receipt = make_temporary_sidecar(output)
    moved_logits = False
    try:
        temporary_receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        )
        temporary_logits.replace(logits)
        moved_logits = True
        temporary_receipt.replace(output)
    except OSError:
        temporary_receipt.unlink(missing_ok=True)
        temporary_logits.unlink(missing_ok=True)
        if moved_logits:
            logits.unlink(missing_ok=True)
        raise


def publish_new(temporary: Path, target: Path, published: list[Path]) -> None:
    """Move `temporary` to `target` only if `target` is absent, and record the publication."""
    os.link(temporary, target)
    published.append(target)
    temporary.unlink()


def write_oracle_outputs(
    receipt: dict[str, Any],
    temporary_logits: Path,
    output: Path,
    logits: Path,
    record: dict[str, Any],
    identity_path: Path,
) -> None:
    """Publish the logits and the identity record first and the receipt last, each only if absent.

    A reader that finds the receipt finds the files it names. A failure removes only
    the files this call published, and every temporary file.
    """
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    temporaries, published = [temporary_logits], []
    try:
        temporary_identity = make_temporary_sidecar(identity_path)
        temporaries.append(temporary_identity)
        temporary_receipt = make_temporary_sidecar(output)
        temporaries.append(temporary_receipt)
        temporary_identity.write_text(identity_text(record, receipt, text))
        temporary_receipt.write_text(text)
        publish_new(temporary_logits, logits, published)
        publish_new(temporary_identity, identity_path, published)
        publish_new(temporary_receipt, output, published)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporaries:
            path.unlink(missing_ok=True)


def refuse_outputs(args: argparse.Namespace, binary: Path) -> None:
    paths = [args.output, args.logits, binary]
    label = "receipt, logit, and binary"
    if args.oracle_identity is not None:
        paths.append(args.oracle_identity)
        label = "receipt, logit, binary, and identity"
    require(len({path.resolve() for path in paths}) == len(paths), f"{label} outputs must differ")
    for path, field in zip(paths, ("receipt output", "logit output", "comparator binary", "oracle identity output")):
        refuse_output(path, field)


def write_workload_main() -> int:
    try:
        print(write_calibration_workload(parse_workload_args()))
    except (OSError, KeyError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    if "--write-calibration-workload" in sys.argv[1:]:
        return write_workload_main()
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        refuse_outputs(args, comparator_binary(args))
        sink = None if args.subject_model is None else {}
        receipt, temporary_logits = create_receipt(args, root, sink)
        if sink is None:
            write_outputs(receipt, temporary_logits, args.output, args.logits)
        else:
            write_oracle_outputs(receipt, temporary_logits, args.output, args.logits, sink, args.oracle_identity)
    except (OSError, ValueError, subprocess.CalledProcessError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
