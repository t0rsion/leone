#!/usr/bin/env python3
"""Produce independently checked retained-history tokenization evidence."""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "leone.history-tokenization.v1"
SCOPE = "plain_length_explicit_fork"
HARNESS_EVIDENCE_FIELDS = (
    "schema_version",
    "status",
    "scope",
    "parent_service_request_id",
    "service_request_id",
    "process_instance_id",
    "workload_epoch",
    "parent_receipt_sha256",
    "branch_receipt_sha256",
    "parent_request_sha256",
    "request_sha256",
    "canonical_prefix_sha256",
    "canonical_request_sha256",
    "parent_prompt_token_ids",
    "parent_generated_token_ids",
    "parent_evaluated_token_ids",
    "request_token_ids",
    "parent_evaluated_token_count",
    "expected_reused_token_count",
    "observed_reused_token_count",
    "oracle",
)
U32_MAX = (1 << 32) - 1
MAX_TOKENIZER_RESPONSE_BYTES = 1024 * 1024
DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
RESPONSE_FIELDS = (
    "schema_version",
    "receipt_id",
    "created_utc",
    "engine_version",
    "model_sha256",
    "request_sha256",
    "prompt_tokens_sha256",
    "response_tokens_sha256",
    "transcript_sha256",
    "seed",
    "prompt_tokens",
    "generated_tokens",
    "finish_reason",
    "cancelled",
    "session",
)
SESSION_FIELDS = (
    "session_id",
    "reuse_class",
    "cached_tokens",
    "reused_tokens",
    "replayed_tokens",
    "computed_tokens",
)
APPLY_FIELDS = (
    "chat_template",
    "chat_template_kwargs",
    "continue_final_message",
    "add_generation_prompt",
    "enable_thinking",
    "reasoning_format",
    "date_string",
    "tools",
    "tool_choice",
)
UNSUPPORTED_REQUEST_FIELDS = (
    "stop",
    "stop_sequences",
    "draft_tokens",
    "adaptive_speculation",
    "speculation",
    "response_format",
)


class _Unavailable(Exception):
    """Carries one typed producer rejection."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def canonical_json(value: Any) -> bytes:
    """Encode a value with the study's stable JSON hash convention."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def uint32_le_sha256(tokens: Sequence[int]) -> str:
    """Hash token IDs as concatenated little-endian uint32 values."""

    output = bytearray()
    for token in tokens:
        if not isinstance(token, int) or isinstance(token, bool) or not 0 <= token <= U32_MAX:
            raise _Unavailable("token_id_out_of_range")
        output.extend(token.to_bytes(4, "little"))
    return sha256_bytes(bytes(output))


def produce_history_tokenization(
    engine: Mapping[str, Any],
    parent_request_bytes: bytes,
    parent_raw_events: Any,
    branch_request_bytes: bytes,
    branch_raw_events: Any,
) -> dict[str, Any]:
    """Produce a checked plain length-stop explicit-fork history record."""

    try:
        producer = _produce_llama_cpp if _is_llama_cpp(engine) else _produce
        return producer(
            engine,
            parent_request_bytes,
            parent_raw_events,
            branch_request_bytes,
            branch_raw_events,
        )
    except _Unavailable as error:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": error.reason}
    except Exception:  # noqa: BLE001
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "producer_error",
        }


def _is_llama_cpp(engine: Mapping[str, Any]) -> bool:
    if not isinstance(engine, Mapping):
        return False
    return engine.get("protocol", engine.get("kind")) in {"llama.cpp", "llama_cpp"}


def _produce(
    engine: Mapping[str, Any],
    parent_request_bytes: bytes,
    parent_raw_events: Any,
    branch_request_bytes: bytes,
    branch_raw_events: Any,
) -> dict[str, Any]:
    prepared = _prepare_engine(engine)
    parent_request = _parse_request(parent_request_bytes)
    branch_request = _parse_request(branch_request_bytes)
    _check_request_scope(parent_request)
    _check_request_scope(branch_request)
    parent_wire = _parse_wire_events(parent_raw_events)
    branch_wire = _parse_wire_events(branch_raw_events)
    parent_claim = _verify_receipt(parent_wire["receipt"], prepared)
    branch_claim = _verify_receipt(branch_wire["receipt"], prepared)
    _check_response_scope(parent_claim)
    _check_response_scope(branch_claim)
    _check_request_hash(parent_claim, parent_request_bytes)
    _check_request_hash(branch_claim, branch_request_bytes)
    _check_identity(parent_wire, branch_wire, parent_claim, branch_claim, prepared)
    _check_fork_controls(branch_request, parent_claim, branch_claim)
    parent_content = _response_content(parent_wire)
    branch_messages = _messages(branch_request)
    prefix_messages = _messages(parent_request) + [{"role": "assistant", "content": parent_content}]
    if branch_messages[: len(prefix_messages)] != prefix_messages:
        raise _Unavailable("branch_history_differs_from_parent_response")
    if len(branch_messages) <= len(prefix_messages):
        raise _Unavailable("branch_has_no_new_message")
    independent = _run_tokenizer(prepared, parent_request, branch_request, parent_content)
    parent_tokens = independent["parent_tokens"]
    generated_tokens = independent["generated_tokens"]
    branch_tokens = independent["branch_tokens"]
    _check_signed_tokens(parent_claim, parent_tokens, generated_tokens)
    _check_branch_prompt(branch_claim, branch_tokens)
    evaluated, expected, observed, lcp = _checked_reuse(
        parent_claim, branch_claim, parent_tokens, generated_tokens, branch_tokens
    )
    return _evidence(
        prepared,
        parent_wire,
        branch_wire,
        parent_request_bytes,
        branch_request_bytes,
        prefix_messages,
        branch_messages,
        parent_tokens,
        generated_tokens,
        evaluated,
        branch_tokens,
        expected,
        observed,
        lcp,
        independent["records"],
    )


def _produce_llama_cpp(
    engine: Mapping[str, Any],
    parent_request_bytes: bytes,
    parent_raw_events: Any,
    branch_request_bytes: bytes,
    branch_raw_events: Any,
) -> dict[str, Any]:
    prepared = _prepare_engine(engine, require_trusted_key=False)
    parsed = _parse_llama_inputs(
        engine, prepared, parent_request_bytes, parent_raw_events, branch_request_bytes, branch_raw_events
    )
    independent = _run_tokenizer(
        prepared, parsed["parent_request"], parsed["branch_request"], parsed["parent_content"]
    )
    return _prove_llama(prepared, parsed, independent)


def recompute_llama_history(
    engine: Mapping[str, Any],
    oracle: Mapping[str, Any],
    parent_request_bytes: bytes,
    parent_raw_events: Any,
    branch_request_bytes: bytes,
    branch_raw_events: Any,
) -> dict[str, Any]:
    """Rebuild llama.cpp evidence from retained raw exchanges without a tokenizer server.

    The tokenizer exchanges come from the retained oracle records. Each retained
    request must match the request this module derives from the raw request, and
    each retained response supplies the rendered prompt and token IDs. The result
    equals the original evidence only when every retained byte agrees.
    """

    try:
        prepared = _prepared_from_oracle(engine, oracle)
        parsed = _parse_llama_inputs(
            engine, prepared, parent_request_bytes, parent_raw_events,
            branch_request_bytes, branch_raw_events,
        )
        independent = _replay_tokenizer(prepared, parsed, oracle.get("apply_template_tokenize"))
        return _prove_llama(prepared, parsed, independent)
    except _Unavailable as error:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": error.reason}
    except Exception:  # noqa: BLE001
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": "producer_error"}


def _prepared_from_oracle(engine: Mapping[str, Any], oracle: Mapping[str, Any]) -> dict[str, Any]:
    policy = oracle.get("special_tokens_policy")
    if not isinstance(policy, Mapping) or sha256_bytes(canonical_json(policy)) != oracle.get(
        "special_tokens_policy_sha256"
    ):
        raise _Unavailable("special_tokens_policy_missing")
    vocabulary = oracle.get("vocab_size")
    if not isinstance(vocabulary, int) or isinstance(vocabulary, bool) or vocabulary <= 0:
        raise _Unavailable("vocabulary_size_missing")
    return {
        "engine": oracle.get("engine"),
        "source_commit": oracle.get("source_commit"),
        "executable_sha256": oracle.get("executable_sha256"),
        "loaded_library_sha256": oracle.get("loaded_library_sha256"),
        "model_sha256": oracle.get("gguf_sha256"),
        "tokenizer_metadata_sha256": oracle.get("tokenizer_metadata_sha256"),
        "tokenizer_metadata_hash_scheme": oracle.get("tokenizer_metadata_hash_scheme"),
        "template_config_sha256": oracle.get("template_config_sha256"),
        "template_config_hash_scheme": oracle.get("template_config_hash_scheme"),
        "template_bytes_sha256": oracle.get("template_bytes_sha256"),
        "special_tokens_policy": dict(policy),
        "special_tokens_policy_sha256": oracle["special_tokens_policy_sha256"],
        "vocab_size": vocabulary,
        "running_identity": engine.get("_running_identity"),
        "prompt_prefix": engine.get("prompt_prefix", ""),
        "model": engine.get("served_model_alias"),
    }


def _parse_llama_inputs(
    engine: Mapping[str, Any], prepared: dict[str, Any],
    parent_request_bytes: bytes, parent_raw_events: Any,
    branch_request_bytes: bytes, branch_raw_events: Any,
) -> dict[str, Any]:
    """Check every llama.cpp fact that needs no tokenizer and return the parsed inputs."""

    parent_request = _parse_request(parent_request_bytes)
    branch_request = _parse_request(branch_request_bytes)
    _check_request_scope(parent_request)
    _check_request_scope(branch_request)
    _check_llama_requests(parent_request, branch_request, engine)
    parent_wire = _parse_llama_wire(parent_raw_events)
    branch_wire = _parse_llama_wire(branch_raw_events)
    slot_copy = _llama_slot_copy_claim(parent_request, branch_request, branch_wire)
    if prepared.get("model") is None:
        prepared["model"] = _last_value(parent_wire["events"], "model")
    _check_llama_identity(parent_wire, branch_wire, prepared)
    parent_content = _response_content(parent_wire)
    branch_messages = _messages(branch_request)
    prefix_messages = _messages(parent_request) + [{"role": "assistant", "content": parent_content}]
    if branch_messages[: len(prefix_messages)] != prefix_messages:
        raise _Unavailable("branch_history_differs_from_parent_response")
    if len(branch_messages) <= len(prefix_messages):
        raise _Unavailable("branch_has_no_new_message")
    return {
        "parent_request": parent_request, "branch_request": branch_request,
        "parent_request_bytes": parent_request_bytes, "branch_request_bytes": branch_request_bytes,
        "parent_wire": parent_wire, "branch_wire": branch_wire, "slot_copy": slot_copy,
        "parent_content": parent_content, "prefix_messages": prefix_messages,
        "branch_messages": branch_messages,
        "generated_tokens": _llama_generated_tokens(parent_wire["verbose"], prepared["vocab_size"]),
    }


def _prove_llama(
    prepared: Mapping[str, Any], parsed: Mapping[str, Any], independent: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare tokenizer output with the verbose data and build the evidence record."""

    parent_verbose = parsed["parent_wire"]["verbose"]
    parent_tokens = independent["parent_tokens"]
    branch_tokens = independent["branch_tokens"]
    generated_tokens = parsed["generated_tokens"]
    if independent["generated_tokens"] != generated_tokens:
        raise _Unavailable("generated_token_ids_differ_from_fresh_tokenizer")
    _check_generation_prompt(prepared, independent)
    _check_llama_parent(
        parent_verbose, independent, parent_tokens, generated_tokens,
        parsed["parent_request"]["id_slot"], prepared,
    )
    if parsed["slot_copy"] is not None:
        _check_slot_copy_boundary(parsed["slot_copy"], parent_verbose)
    observed = _check_llama_branch(
        parsed["branch_wire"], parsed["branch_request"]["id_slot"], independent, branch_tokens, prepared
    )
    evaluated, expected, lcp = _checked_llama_reuse(
        parent_verbose, observed, parent_tokens, generated_tokens, branch_tokens
    )
    return _evidence(
        prepared, parsed["parent_wire"], parsed["branch_wire"],
        parsed["parent_request_bytes"], parsed["branch_request_bytes"],
        parsed["prefix_messages"], parsed["branch_messages"],
        parent_tokens, generated_tokens, evaluated, branch_tokens, expected, observed, lcp,
        independent["records"], proof_adapter="llama.cpp.verbose", slot_copy=parsed["slot_copy"],
        apply_kwargs=_apply_kwargs(prepared, parsed),
    )


def _check_generation_prompt(prepared: Mapping[str, Any], independent: Mapping[str, Any]) -> None:
    """Require the rendered prompts to end with the declared generation prompt bytes.

    The suffix pins the thinking policy: a template that adds a reasoning block
    after the assistant header renders different bytes than the other engine.
    """

    suffix = prepared["special_tokens_policy"].get("generation_prompt_suffix")
    if suffix is None:
        return
    if not isinstance(suffix, str) or not suffix:
        raise _Unavailable("generation_prompt_suffix_invalid")
    if not all(independent[name].endswith(suffix) for name in ("parent_rendered", "branch_rendered")):
        raise _Unavailable("llama_generation_prompt_incompatible")


def _apply_kwargs(prepared: Mapping[str, Any], parsed: Mapping[str, Any]) -> dict[str, Any]:
    """Return the template kwargs each retained apply-template request carries."""

    return {
        name: _apply_body(prepared, parsed[f"{name}_request"]).get("chat_template_kwargs")
        for name in ("parent", "branch")
    }


def _replay_tokenizer(
    prepared: Mapping[str, Any], parsed: Mapping[str, Any], records: Any
) -> dict[str, Any]:
    if not isinstance(records, Mapping):
        raise _Unavailable("tokenizer_exchanges_missing")
    parent = _replay_prompt(prepared, records.get("parent"), parsed["parent_request"])
    branch = _replay_prompt(prepared, records.get("branch"), parsed["branch_request"])
    generated_record = records.get("generated")
    generated = _replay_text(
        prepared, generated_record.get("tokenize") if isinstance(generated_record, Mapping) else None,
        parsed["parent_content"], "generated",
    )
    return {
        "parent_tokens": parent["tokens"], "branch_tokens": branch["tokens"],
        "generated_tokens": generated["tokens"], "parent_rendered": parent["rendered"],
        "branch_rendered": branch["rendered"], "records": dict(records),
    }


def _replay_prompt(
    prepared: Mapping[str, Any], record: Any, request: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise _Unavailable("tokenizer_exchanges_missing")
    applied = _replay_exchange(record.get("apply_template"), _apply_body(prepared, request))
    rendered = applied.get("prompt")
    if not isinstance(rendered, str):
        raise _Unavailable("apply_template_response_invalid")
    tokenized = _replay_text(prepared, record.get("tokenize"), rendered, "prompt")
    return {"tokens": tokenized["tokens"], "rendered": rendered}


def _replay_text(
    prepared: Mapping[str, Any], record: Any, text: str, kind: str
) -> dict[str, Any]:
    response = _replay_exchange(record, _tokenize_body(prepared, text, kind))
    tokens = response.get("tokens")
    if not _valid_tokens(tokens, prepared["vocab_size"]):
        raise _Unavailable("tokenize_response_invalid")
    return {"tokens": tokens}


def _replay_exchange(record: Any, expected_request: Mapping[str, Any]) -> dict[str, Any]:
    """Return the retained response after checking its request and digests."""

    if not isinstance(record, Mapping):
        raise _Unavailable("tokenizer_exchanges_missing")
    request, response = _hex_bytes(record.get("request_hex")), _hex_bytes(record.get("response_hex"))
    if sha256_bytes(request) != record.get("request_sha256") or sha256_bytes(response) != record.get(
        "response_sha256"
    ):
        raise _Unavailable("tokenizer_exchange_digest_mismatch")
    if _load_json(request) != json.loads(json.dumps(expected_request)):
        raise _Unavailable("tokenizer_exchange_request_differs")
    parsed = _load_json(response)
    if not isinstance(parsed, dict):
        raise _Unavailable("tokenizer_response_invalid")
    return parsed


def _prepare_engine(
    engine: Mapping[str, Any], *, require_trusted_key: bool = True
) -> dict[str, Any]:
    """Validate pinned model, template, binary, and token policy inputs."""

    if not isinstance(engine, Mapping):
        raise _Unavailable("engine_declaration_missing")
    generator = _template_generator()
    artifacts = _prepare_artifacts(engine, generator)
    binary, model = artifacts["binary"], artifacts["model"]
    template = _template_declaration(engine, model, generator)
    policy = _policy_declaration(engine)
    trusted_key = _prepare_trusted_key(engine, require_trusted_key)
    vocabulary = _prepare_vocabulary(engine, model)
    return {
        "binary": binary,
        "model": model,
        **artifacts,
        "trusted_public_key": trusted_key,
        "engine": engine.get("engine", "leone" if require_trusted_key else "llama.cpp"),
        "running_identity": engine.get("_running_identity"),
        "prompt_prefix": engine.get("prompt_prefix", ""),
        "vocab_size": vocabulary,
        **template,
        **policy,
    }


def _prepare_artifacts(engine: Mapping[str, Any], generator: Any) -> dict[str, Any]:
    binary = _required_path(engine, ("llama_server", "binary", "llama_server_path"))
    model = _required_path(engine, ("model", "model_path"))
    if not binary.is_file() or not model.is_file():
        raise _Unavailable("pinned_artifact_missing")
    source_commit = generator.verify_engine(binary)
    declared_commit = engine.get("llama_cpp_commit", engine.get("source_commit"))
    if not isinstance(declared_commit, str) or source_commit != declared_commit:
        raise _Unavailable("pinned_source_commit_mismatch")
    loaded_library_sha256 = _loaded_library_digest(engine, binary)
    return {
        "binary": binary,
        "model": model,
        "source_commit": source_commit,
        "executable_sha256": _checked_file_digest(binary, engine.get("executable_sha256")),
        "model_sha256": _checked_file_digest(model, engine.get("model_sha256")),
        "loaded_library_sha256": loaded_library_sha256,
    }


def _loaded_library_digest(engine: Mapping[str, Any], binary: Path) -> str:
    """Hash the resolved tokenizer library closure and check its optional pin."""

    source = Path(__file__).resolve().parent / "linked_libraries.py"
    resolver, linkage_error = _load_linked_library_helper(source)
    try:
        records = resolver(binary, binary.parent, engine.get("backend"))
    except (OSError, ValueError, TypeError, subprocess.SubprocessError, linkage_error) as error:
        raise _Unavailable("loaded_library_unavailable") from error
    actual = sha256_bytes(canonical_json(records))
    declared = engine.get("loaded_library_sha256")
    if declared is not None and _required_digest(declared) != actual:
        raise _Unavailable("loaded_library_hash_mismatch")
    return actual


def _load_linked_library_helper(source: Path) -> tuple[Any, type[Exception]]:
    """Load the linkage resolver and validate its failure type before use."""

    spec = importlib.util.spec_from_file_location("leone_history_linked_libraries", source)
    if spec is None or spec.loader is None:
        raise _Unavailable("loaded_library_helper_missing")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise _Unavailable("loaded_library_unavailable") from error
    resolver = getattr(module, "resolve", None)
    linkage_error = getattr(module, "LinkageError", None)
    if not callable(resolver) or not isinstance(linkage_error, type) or not issubclass(linkage_error, Exception):
        raise _Unavailable("loaded_library_unavailable")
    return resolver, linkage_error


def _prepare_trusted_key(engine: Mapping[str, Any], required: bool) -> str | None:
    value = engine.get("trusted_public_key_ed25519")
    if required or value is not None:
        return _required_digest(value, 64)
    return None


def _prepare_vocabulary(engine: Mapping[str, Any], model: Path) -> int:
    vocabulary = engine.get("vocab_size")
    if vocabulary is None:
        vocabulary = _gguf_vocabulary(model)
    if not isinstance(vocabulary, int) or isinstance(vocabulary, bool) or vocabulary <= 0:
        raise _Unavailable("vocabulary_size_missing")
    return vocabulary


def _gguf_vocabulary(model: Path) -> int:
    """Read the tokenizer vocabulary length from the pinned GGUF metadata."""

    try:
        with model.open("rb") as source:
            if source.read(4) != b"GGUF":
                raise _Unavailable("vocabulary_metadata_invalid")
            _read_u32(source)
            _read_u64(source)
            metadata_count = _read_u64(source)
            for _ in range(metadata_count):
                key = _read_gguf_string(source)
                value_type = _read_u32(source)
                if key == "tokenizer.ggml.tokens" and value_type == 9:
                    _read_u32(source)
                    count = _read_u64(source)
                    if count <= 0 or count > 10_000_000:
                        raise _Unavailable("vocabulary_metadata_invalid")
                    return count
                _skip_gguf_value(source, value_type)
    except (OSError, struct.error, UnicodeDecodeError) as error:
        raise _Unavailable("vocabulary_metadata_invalid") from error
    raise _Unavailable("vocabulary_size_missing")


def _read_u32(source: Any) -> int:
    return struct.unpack("<I", source.read(4))[0]


def _read_u64(source: Any) -> int:
    return struct.unpack("<Q", source.read(8))[0]


def _read_gguf_string(source: Any) -> str:
    length = _read_u64(source)
    if length > 1_000_000:
        raise _Unavailable("vocabulary_metadata_invalid")
    value = source.read(length)
    if len(value) != length:
        raise _Unavailable("vocabulary_metadata_invalid")
    return value.decode("utf-8")


def _skip_gguf_value(source: Any, value_type: int) -> None:
    if value_type == 8:
        length = _read_u64(source)
        source.seek(length, 1)
        return
    if value_type == 9:
        element_type = _read_u32(source)
        count = _read_u64(source)
        if count > 10_000_000:
            raise _Unavailable("vocabulary_metadata_invalid")
        for _ in range(count):
            _skip_gguf_value(source, element_type)
        return
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    size = sizes.get(value_type)
    if size is None:
        raise _Unavailable("vocabulary_metadata_invalid")
    source.seek(size, 1)


def _required_path(engine: Mapping[str, Any], names: Sequence[str]) -> Path:
    for name in names:
        value = engine.get(name)
        if isinstance(value, Mapping):
            value = next(
                (
                    value.get(child)
                    for child in ("path", "binary", "model_path", "llama_server_path")
                    if value.get(child)
                ),
                None,
            )
        if isinstance(value, (str, os.PathLike)) and str(value):
            return Path(value)
    raise _Unavailable("engine_artifact_path_missing")


def _checked_file_digest(path: Path, declared: Any) -> str:
    actual = _file_digest(path)
    if declared is not None and declared != actual:
        raise _Unavailable("pinned_artifact_hash_mismatch")
    return actual


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_digest(value: Any, length: int = 64) -> str:
    if not isinstance(value, str) or len(value) != length or not re.fullmatch(r"[0-9a-fA-F]+", value):
        raise _Unavailable("pinned_digest_missing")
    return value.lower()


def _template_declaration(
    engine: Mapping[str, Any], model: Path, generator: Any
) -> dict[str, Any]:
    mode = engine.get("template_mode")
    if mode not in {"official", "legacy"}:
        raise _Unavailable("template_mode_unsupported")
    if mode == "official":
        config_path = _required_path(engine, ("tokenizer_config", "template_config"))
        config_bytes = config_path.read_bytes()
        config_sha256 = sha256_bytes(config_bytes)
        document = _load_json(config_bytes)
        template_text = document.get("chat_template") if isinstance(document, Mapping) else None
        if not isinstance(template_text, str):
            raise _Unavailable("official_template_missing")
        embedded = generator.embedded_template(model)
        expected_embedded = engine.get("embedded_template_sha256")
        if expected_embedded is not None and sha256_bytes(embedded.encode()) != expected_embedded:
            raise _Unavailable("embedded_template_hash_mismatch")
        config_scheme = "tokenizer_config_json_bytes"
    else:
        template_path = _required_path(engine, ("template_file", "legacy_template"))
        template_bytes = template_path.read_bytes()
        config_bytes = template_bytes
        config_sha256 = sha256_bytes(config_bytes)
        try:
            template_text = template_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _Unavailable("template_not_utf8") from error
        config_scheme = "legacy_template_file_bytes"
    template_bytes = template_text.encode("utf-8")
    template_bytes_sha256 = sha256_bytes(template_bytes)
    _match_declared(engine, "template_config_sha256", config_sha256, "template_config_hash_mismatch")
    _match_declared(engine, "template_bytes_sha256", template_bytes_sha256, "template_bytes_hash_mismatch")
    metadata_sha256 = engine.get("tokenizer_metadata_sha256", config_sha256)
    if metadata_sha256 != config_sha256:
        raise _Unavailable("tokenizer_metadata_hash_mismatch")
    return {
        "template_mode": mode,
        "template_text": template_text,
        "template_bytes": template_bytes,
        "template_config_sha256": config_sha256,
        "template_bytes_sha256": template_bytes_sha256,
        "template_config_hash_scheme": config_scheme,
        "tokenizer_metadata_sha256": metadata_sha256,
        "tokenizer_metadata_hash_scheme": "tokenizer_config_json_bytes" if mode == "official" else config_scheme,
    }


def _match_declared(engine: Mapping[str, Any], field: str, actual: str, reason: str) -> None:
    declared = engine.get(field)
    if declared is not None and declared != actual:
        raise _Unavailable(reason)


def _policy_declaration(engine: Mapping[str, Any]) -> dict[str, Any]:
    policy = engine.get("special_tokens_policy")
    if not isinstance(policy, Mapping):
        raise _Unavailable("special_tokens_policy_missing")
    policy = json.loads(json.dumps(policy, ensure_ascii=False))
    policy_sha256 = sha256_bytes(canonical_json(policy))
    _match_declared(engine, "special_tokens_policy_sha256", policy_sha256, "special_tokens_policy_hash_mismatch")
    prompt = policy.get("prompt", policy.get("tokenize", policy))
    generated = policy.get("generated", {})
    _token_flags(prompt, "prompt")
    if not isinstance(generated, Mapping):
        raise _Unavailable("generated_token_policy_missing")
    _token_flags(generated or {"add_special": False, "parse_special": prompt["parse_special"]}, "generated")
    return {"special_tokens_policy": policy, "special_tokens_policy_sha256": policy_sha256}


def _token_flags(policy: Mapping[str, Any], name: str) -> dict[str, bool]:
    if not isinstance(policy.get("add_special"), bool) or not isinstance(policy.get("parse_special"), bool):
        raise _Unavailable(f"{name}_token_policy_incomplete")
    return {"add_special": policy["add_special"], "parse_special": policy["parse_special"]}


def _parse_request(raw: bytes) -> dict[str, Any]:
    if not isinstance(raw, bytes):
        raise _Unavailable("request_bytes_required")
    value = _load_json(raw)
    if not isinstance(value, dict):
        raise _Unavailable("request_object_required")
    _messages(value)
    return value


def _load_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise _Unavailable("invalid_json_input") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _messages(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _Unavailable("messages_missing")
    output = []
    for message in messages:
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str):
            raise _Unavailable("message_shape_unsupported")
        content = message.get("content")
        if not isinstance(content, str):
            raise _Unavailable("message_content_unsupported")
        output.append(dict(message))
    return output


def _check_request_scope(request: Mapping[str, Any]) -> None:
    for field in UNSUPPORTED_REQUEST_FIELDS:
        value = request.get(field)
        if value not in (None, False, [], ""):
            raise _Unavailable(f"unsupported_{field}")
    for message in _messages(request):
        if message.get("role") == "tool" or "tool_calls" in message:
            raise _Unavailable("unsupported_tools")
    if request.get("tools") not in (None, []):
        raise _Unavailable("unsupported_tools")
    if request.get("tool_choice") not in (None, "none"):
        raise _Unavailable("unsupported_tools")


def _parse_wire_events(raw: Any) -> dict[str, Any]:
    raw_bytes = _raw_bytes(raw)
    events = _wire_objects(raw)
    receipt = _find_receipt(events)
    if receipt is None:
        raise _Unavailable("signed_receipt_missing")
    return {
        "events": events,
        "raw_sha256": sha256_bytes(raw_bytes),
        "receipt": receipt,
        "receipt_sha256": sha256_bytes(canonical_json(receipt)),
        "metadata": _metadata(events),
    }


def _parse_llama_wire(raw: Any) -> dict[str, Any]:
    raw_bytes = _raw_bytes(raw)
    events = _wire_objects(raw)
    verbose = _last_value(events, "__verbose")
    if not isinstance(verbose, Mapping):
        raise _Unavailable("llama_verbose_missing")
    usage = _last_value(events, "usage")
    return {
        "events": events,
        "raw_sha256": sha256_bytes(raw_bytes),
        "verbose": dict(verbose),
        "usage": dict(usage) if isinstance(usage, Mapping) else None,
        "metadata": _metadata(events),
    }


def _last_value(events: Sequence[Mapping[str, Any]], field: str) -> Any:
    found = None
    for event in events:
        value = _find_value(event, field)
        if value is not None:
            found = value
    return found


def _check_llama_requests(
    parent: Mapping[str, Any], branch: Mapping[str, Any], engine: Mapping[str, Any]
) -> None:
    _check_llama_stream_controls(parent, branch)
    _check_llama_usage_controls(branch)
    _check_llama_slots(parent, branch, engine)


def _check_llama_stream_controls(parent: Mapping[str, Any], branch: Mapping[str, Any]) -> None:
    if parent.get("stream") is not False or branch.get("stream") is not True:
        raise _Unavailable("llama_stream_protocol_mismatch")
    for request in (parent, branch):
        if request.get("verbose") is not True or request.get("cache_prompt") is not True:
            raise _Unavailable("llama_verbose_cache_controls_missing")


def _check_llama_usage_controls(branch: Mapping[str, Any]) -> None:
    stream_options = branch.get("stream_options")
    if not isinstance(stream_options, Mapping) or stream_options.get("include_usage") is not True:
        raise _Unavailable("llama_branch_usage_controls_missing")


def _check_llama_slots(
    parent: Mapping[str, Any], branch: Mapping[str, Any], engine: Mapping[str, Any]
) -> None:
    if parent.get("return_tokens") is not True:
        raise _Unavailable("llama_parent_token_controls_missing")
    parent_slot = parent.get("id_slot")
    declared_slot = engine.get("id_slot")
    if not _valid_slot(parent_slot) or not _valid_slot(branch.get("id_slot")):
        raise _Unavailable("llama_slot_identity_missing")
    if declared_slot is not None and parent_slot != declared_slot:
        raise _Unavailable("llama_slot_declaration_mismatch")


def _llama_slot_copy_claim(
    parent: Mapping[str, Any], branch: Mapping[str, Any], branch_wire: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return the verified parent-to-branch slot copy, or None for one shared slot.

    A branch in another slot starts cold unless `/slots/{id}?action=restore`
    loaded the parent's saved slot first. The retained save and restore
    exchanges must name one file, one token count, and the two slots.
    """

    parent_slot, branch_slot = parent["id_slot"], branch["id_slot"]
    claim = _last_value(branch_wire["events"], "slot_copy")
    if parent_slot == branch_slot:
        if claim is not None:
            raise _Unavailable("llama_slot_copy_unexpected")
        return None
    if not isinstance(claim, Mapping):
        raise _Unavailable("llama_slot_copy_missing")
    save = _slot_exchange(claim, "save", parent_slot, ("n_saved", "n_written"))
    restore = _slot_exchange(claim, "restore", branch_slot, ("n_restored", "n_read"))
    branch_start = claim.get("branch_request_start_ns")
    if save["filename"] != restore["filename"]:
        raise _Unavailable("llama_slot_copy_file_mismatch")
    if save["tokens"] != restore["tokens"] or save["bytes"] != restore["bytes"] or save["bytes"] <= 0:
        raise _Unavailable("llama_slot_copy_count_mismatch")
    if not _is_count(branch_start) or not save["end_ns"] < restore["start_ns"] <= restore["end_ns"] < branch_start:
        raise _Unavailable("llama_slot_copy_order_invalid")
    return {
        "parent_slot": parent_slot,
        "branch_slot": branch_slot,
        "filename": save["filename"],
        "saved_token_count": save["tokens"],
        "restored_token_count": restore["tokens"],
        "byte_count": save["bytes"],
        "save_request_sha256": save["request_sha256"],
        "save_response_sha256": save["response_sha256"],
        "restore_request_sha256": restore["request_sha256"],
        "restore_response_sha256": restore["response_sha256"],
        "restore_elapsed_ns": restore["end_ns"] - restore["start_ns"],
    }


def _slot_exchange(
    claim: Mapping[str, Any], action: str, slot: int, count_fields: tuple[str, str]
) -> dict[str, Any]:
    record = claim.get(action)
    if not isinstance(record, Mapping) or record.get("action") != action or record.get("slot") != slot:
        raise _Unavailable("llama_slot_copy_action_mismatch")
    if record.get("http_status") != 200:
        raise _Unavailable("llama_slot_copy_failed")
    request, response = _hex_bytes(record.get("request_hex")), _hex_bytes(record.get("response_hex"))
    filename, reply = _slot_reply(request, response, slot)
    tokens, size = (reply.get(field) for field in count_fields)
    start, end = record.get("request_start_ns"), record.get("request_end_ns")
    if not all(_is_count(value) for value in (tokens, size, start, end)):
        raise _Unavailable("llama_slot_copy_response_invalid")
    return {
        "filename": filename, "tokens": tokens, "bytes": size, "start_ns": start, "end_ns": end,
        "request_sha256": sha256_bytes(request), "response_sha256": sha256_bytes(response),
    }


def _slot_reply(request: bytes, response: bytes, slot: int) -> tuple[str, Mapping[str, Any]]:
    body, reply = _load_json(request), _load_json(response)
    filename = body.get("filename") if isinstance(body, Mapping) else None
    if not isinstance(reply, Mapping) or not isinstance(filename, str) or not filename:
        raise _Unavailable("llama_slot_copy_response_invalid")
    if reply.get("id_slot") != slot or reply.get("filename") != filename:
        raise _Unavailable("llama_slot_copy_response_mismatch")
    return filename, reply


def _hex_bytes(value: Any) -> bytes:
    try:
        decoded = bytes.fromhex(value) if isinstance(value, str) else b""
    except ValueError:
        decoded = b""
    if not decoded:
        raise _Unavailable("llama_slot_copy_bytes_missing")
    return decoded


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_slot_copy_boundary(slot_copy: Mapping[str, Any], parent_verbose: Mapping[str, Any]) -> None:
    if slot_copy["saved_token_count"] != parent_verbose.get("tokens_cached"):
        raise _Unavailable("llama_slot_copy_boundary_mismatch")


def _valid_slot(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _check_llama_identity(
    parent: Mapping[str, Any], branch: Mapping[str, Any], prepared: Mapping[str, Any]
) -> None:
    required = ("service_request_id", "process_instance_id", "workload_epoch")
    parent_meta = parent["metadata"]
    branch_meta = branch["metadata"]
    if any(field not in parent_meta or field not in branch_meta for field in required):
        raise _Unavailable("service_identity_missing")
    for field in ("process_instance_id", "workload_epoch"):
        if parent_meta[field] != branch_meta[field]:
            raise _Unavailable("service_identity_changed")
    expected_process = _running_process_identity(prepared.get("running_identity"))
    if expected_process is not None:
        for metadata in (parent_meta, branch_meta):
            if str(metadata.get("process_instance_id")) != expected_process:
                raise _Unavailable("service_process_identity_mismatch")
    _check_llama_response_model(parent, prepared)
    _check_llama_response_model(branch, prepared)
    parent_id = parent_meta["service_request_id"]
    branch_parent = branch_meta.get("parent_service_request_id")
    if branch_parent != parent_id:
        raise _Unavailable("parent_service_request_identity_missing")


def _check_llama_response_model(wire: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    observed = _last_value(wire["events"], "model")
    if observed is None:
        raise _Unavailable("llama_response_model_missing")
    expected = str(prepared["model"])
    if str(observed) not in {expected, Path(expected).name}:
        raise _Unavailable("llama_response_model_mismatch")


def _running_process_identity(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    start = value.get("start") if isinstance(value.get("start"), Mapping) else value
    identity = start.get("identity") if isinstance(start, Mapping) else start
    if not isinstance(identity, Mapping):
        return None
    for field in ("process_instance_id", "process_start_ns"):
        candidate = identity.get(field)
        if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
            return str(candidate)
    return None


def _llama_generated_tokens(verbose: Mapping[str, Any], vocabulary: int) -> list[int]:
    tokens = verbose.get("tokens")
    if not _valid_tokens(tokens, vocabulary) or not tokens:
        raise _Unavailable("llama_parent_generated_tokens_missing")
    return list(tokens)


def _check_llama_parent(
    verbose: Mapping[str, Any], independent: Mapping[str, Any],
    parent_tokens: Sequence[int], generated_tokens: Sequence[int], parent_slot: int,
    prepared: Mapping[str, Any],
) -> None:
    generation_settings = verbose.get("generation_settings")
    if not isinstance(generation_settings, Mapping):
        raise _Unavailable("llama_parent_generation_settings_missing")
    if generation_settings.get("speculative.types") != "none":
        raise _Unavailable("unsupported_llama_parent_speculation")
    if verbose.get("prompt") != _llama_rendered_prompt(prepared, independent["parent_rendered"]):
        raise _Unavailable("llama_parent_prompt_mismatch")
    if verbose.get("id_slot") != parent_slot:
        raise _Unavailable("llama_parent_slot_mismatch")
    if verbose.get("tokens_evaluated") != len(parent_tokens):
        raise _Unavailable("llama_parent_prompt_count_mismatch")
    if verbose.get("tokens_predicted") != len(generated_tokens):
        raise _Unavailable("llama_parent_generated_count_mismatch")
    if verbose.get("tokens_cached") != len(parent_tokens) + len(generated_tokens) - 1:
        raise _Unavailable("llama_parent_cached_count_mismatch")
    if verbose.get("stop_type") != "limit" or verbose.get("truncated") is not False:
        raise _Unavailable("unsupported_llama_parent_stop")


def _check_llama_branch(
    branch: Mapping[str, Any], branch_slot: int,
    independent: Mapping[str, Any], branch_tokens: Sequence[int], prepared: Mapping[str, Any],
) -> int:
    verbose = branch["verbose"]
    _check_llama_branch_prompt(verbose, branch_slot, independent, prepared)
    _check_llama_branch_stop(verbose, branch["events"])
    return _llama_branch_cached_tokens(branch, len(branch_tokens))


def _check_llama_branch_prompt(
    verbose: Mapping[str, Any], branch_slot: int,
    independent: Mapping[str, Any], prepared: Mapping[str, Any],
) -> None:
    if verbose.get("prompt") != _llama_rendered_prompt(prepared, independent["branch_rendered"]):
        raise _Unavailable("llama_branch_prompt_mismatch")
    if verbose.get("id_slot") != branch_slot:
        raise _Unavailable("llama_branch_slot_mismatch")


def _check_llama_branch_stop(
    verbose: Mapping[str, Any], events: Sequence[Mapping[str, Any]]
) -> None:
    if "stop_type" in verbose and verbose["stop_type"] != "limit":
        raise _Unavailable("unsupported_llama_branch_stop")
    if "truncated" in verbose and verbose["truncated"] is not False:
        raise _Unavailable("unsupported_llama_branch_stop")
    if not _finish_reason_length(events):
        raise _Unavailable("unsupported_llama_branch_stop")


def _llama_branch_cached_tokens(branch: Mapping[str, Any], token_count: int) -> int:
    usage = branch.get("usage")
    if not isinstance(usage, Mapping) or usage.get("prompt_tokens") != token_count:
        raise _Unavailable("llama_branch_prompt_count_mismatch")
    details = usage.get("prompt_tokens_details")
    observed = details.get("cached_tokens") if isinstance(details, Mapping) else None
    if not isinstance(observed, int) or isinstance(observed, bool) or observed < 0:
        raise _Unavailable("llama_branch_cached_count_missing")
    if observed > token_count:
        raise _Unavailable("llama_branch_cached_count_invalid")
    return observed


def _llama_rendered_prompt(prepared: Mapping[str, Any], rendered: str) -> str:
    prefix = prepared.get("prompt_prefix", "")
    if not isinstance(prefix, str):
        raise _Unavailable("llama_prompt_prefix_invalid")
    return rendered if rendered.startswith(prefix) else prefix + rendered


def _finish_reason_length(events: Sequence[Mapping[str, Any]]) -> bool:
    reasons = []
    for event in events:
        for payload in _nested_mappings(event):
            choices = payload.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if isinstance(choice, Mapping) and choice.get("finish_reason") is not None:
                    reasons.append(choice.get("finish_reason"))
    return bool(reasons) and reasons[-1] == "length"


def _checked_llama_reuse(
    parent_verbose: Mapping[str, Any], observed: int,
    parent_tokens: Sequence[int], generated_tokens: Sequence[int],
    branch_tokens: Sequence[int],
) -> tuple[list[int], int, int]:
    expected = len(parent_tokens) + len(generated_tokens) - 1
    n = parent_verbose.get("tokens_cached")
    if len(generated_tokens) < 2 or len(parent_tokens) == 0:
        raise _Unavailable("plain_history_boundary_too_short")
    if n != expected:
        raise _Unavailable("cached_count_does_not_match_evaluated_boundary")
    if n <= len(parent_tokens) or n > len(parent_tokens) + len(generated_tokens):
        raise _Unavailable("evaluated_boundary_invalid")
    evaluated = [*parent_tokens, *generated_tokens][:n]
    lcp = _longest_common_prefix(evaluated, branch_tokens)
    if len(branch_tokens) <= len(evaluated) or lcp != len(evaluated):
        raise _Unavailable("retained_history_is_not_full_branch_prefix")
    if observed != len(evaluated):
        raise _Unavailable("reused_count_does_not_match_evaluated_boundary")
    return evaluated, expected, lcp


def _raw_bytes(raw: Any) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, str):
        return raw.encode("utf-8")
    if isinstance(raw, (list, Mapping)):
        return canonical_json(raw)
    raise _Unavailable("raw_events_unsupported")


def _wire_objects(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, Mapping):
        return _flatten_events([dict(raw)])
    if isinstance(raw, list):
        return _wire_list(raw)
    if isinstance(raw, str):
        return _wire_objects(raw.encode("utf-8"))
    if not isinstance(raw, bytes):
        raise _Unavailable("raw_events_unsupported")
    return _wire_bytes(raw)


def _wire_list(raw: Sequence[Any]) -> list[dict[str, Any]]:
    objects = []
    for item in raw:
        objects.extend(_wire_objects(item))
    return objects


def _wire_bytes(raw: bytes) -> list[dict[str, Any]]:
    try:
        parsed = _load_json(raw)
        return _wire_objects(parsed)
    except _Unavailable:
        return _wire_sse(raw)


def _wire_sse(raw: bytes) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise _Unavailable("raw_events_not_utf8") from error
    objects = []
    for line in text.splitlines():
        payload = line.strip()
        if payload.startswith("data:"):
            payload = payload[5:].strip()
        if payload and payload != "[DONE]":
            objects.extend(_wire_sse_payload(payload))
    if not objects:
        raise _Unavailable("raw_events_invalid")
    return objects


def _wire_sse_payload(payload: str) -> list[dict[str, Any]]:
    try:
        return _wire_objects(payload.encode("utf-8"))
    except _Unavailable as error:
        raise _Unavailable("raw_events_invalid") from error


def _flatten_events(events: Sequence[Any]) -> list[dict[str, Any]]:
    output = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        nested = event.get("events")
        if isinstance(nested, list):
            wrapper = {key: value for key, value in event.items() if key != "events"}
            if wrapper:
                output.append(wrapper)
            output.extend(_flatten_events(nested))
        else:
            output.append(dict(event))
    if not output:
        raise _Unavailable("raw_events_empty")
    return output


def _find_receipt(events: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for event in events:
        receipt = _find_value(event, "leone_receipt")
        if isinstance(receipt, Mapping):
            return dict(receipt)
        if _is_receipt(event):
            return dict(event)
    return None


def _is_receipt(value: Mapping[str, Any]) -> bool:
    return all(field in value for field in ("claim", "public_key_ed25519", "signature_ed25519"))


def _find_value(value: Any, field: str) -> Any:
    if isinstance(value, Mapping):
        if field in value:
            return value[field]
        for child in value.values():
            found = _find_value(child, field)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_value(child, field)
            if found is not None:
                return found
    return None


def _metadata(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fields = (
        "service_request_id",
        "parent_service_request_id",
        "process_instance_id",
        "workload_epoch",
        "source_id",
        "receipt_id",
        "parent_session_id",
    )
    for event in events:
        candidates = _nested_mappings(event)
        for candidate in candidates:
            for field in fields:
                value = candidate.get(field)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    output.setdefault(field, value)
    return output


def _nested_mappings(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    output: list[Mapping[str, Any]] = [value]
    for field in ("identity", "response", "body", "event", "data", "json"):
        child = value.get(field)
        if isinstance(child, Mapping):
            output.extend(_nested_mappings(child))
    return output


def _verify_receipt(receipt: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    _check_receipt_identity(receipt, prepared)
    claim = receipt.get("claim")
    session = claim.get("session") if isinstance(claim, Mapping) else None
    _check_receipt_shape(claim, session)
    payload = _receipt_payload(claim, session)
    _verify_receipt_signature(receipt, prepared, payload)
    _validate_claim_values(claim, prepared)
    return dict(claim)


def _check_receipt_identity(receipt: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    if not isinstance(receipt, Mapping) or not _is_receipt(receipt):
        raise _Unavailable("signed_receipt_invalid")
    if receipt.get("public_key_ed25519") != prepared["trusted_public_key"]:
        raise _Unavailable("signed_receipt_trusted_key_mismatch")


def _check_receipt_shape(claim: Any, session: Any) -> None:
    if not isinstance(claim, Mapping) or set(claim) != set(RESPONSE_FIELDS):
        raise _Unavailable("signed_claim_fields_invalid")
    if not isinstance(session, Mapping) or set(session) != set(SESSION_FIELDS):
        raise _Unavailable("signed_session_fields_invalid")


def _receipt_payload(claim: Mapping[str, Any], session: Mapping[str, Any]) -> bytes:
    ordered_claim = {field: claim[field] for field in RESPONSE_FIELDS if field != "session"}
    ordered_claim["session"] = {field: session[field] for field in SESSION_FIELDS}
    return json.dumps(ordered_claim, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _verify_receipt_signature(
    receipt: Mapping[str, Any], prepared: Mapping[str, Any], payload: bytes
) -> None:
    try:
        key_bytes = bytes.fromhex(receipt["public_key_ed25519"])
        signature_bytes = bytes.fromhex(receipt["signature_ed25519"])
    except (TypeError, ValueError) as error:
        raise _Unavailable("signed_receipt_encoding_invalid") from error
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature_bytes, payload)
    except Exception as error:
        raise _Unavailable("signed_receipt_signature_invalid") from error


def _validate_claim_values(claim: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    _validate_claim_identity(claim, prepared)
    _validate_claim_digests(claim)
    _validate_claim_counts(claim)
    _validate_claim_session(claim["session"])


def _validate_claim_identity(claim: Mapping[str, Any], prepared: Mapping[str, Any]) -> None:
    if claim.get("schema_version") != 1:
        raise _Unavailable("signed_claim_schema_unsupported")
    if claim.get("model_sha256") != prepared["model_sha256"]:
        raise _Unavailable("signed_claim_model_mismatch")


def _validate_claim_digests(claim: Mapping[str, Any]) -> None:
    for field in (
        "request_sha256",
        "prompt_tokens_sha256",
        "response_tokens_sha256",
        "transcript_sha256",
    ):
        _required_digest(claim.get(field))


def _validate_claim_counts(claim: Mapping[str, Any]) -> None:
    for field in ("prompt_tokens", "generated_tokens", "seed"):
        if not isinstance(claim.get(field), int) or isinstance(claim[field], bool) or claim[field] < 0:
            raise _Unavailable("signed_claim_count_invalid")


def _validate_claim_session(session: Mapping[str, Any]) -> None:
    if not isinstance(session.get("session_id"), str) or not session["session_id"]:
        raise _Unavailable("signed_session_id_missing")
    for field in SESSION_FIELDS[2:]:
        if not isinstance(session.get(field), int) or isinstance(session[field], bool) or session[field] < 0:
            raise _Unavailable("signed_session_count_invalid")


def _check_response_scope(claim: Mapping[str, Any]) -> None:
    if claim.get("finish_reason") != "length":
        raise _Unavailable("unsupported_finish_reason")
    if claim.get("cancelled") is not False:
        raise _Unavailable("unsupported_cancelled_response")
    if claim["session"]["reuse_class"] in {"restore-replay", "host-wake"}:
        raise _Unavailable("unsupported_restored_response")


def _check_request_hash(claim: Mapping[str, Any], request_bytes: bytes) -> None:
    if claim.get("request_sha256") != sha256_bytes(request_bytes):
        raise _Unavailable("signed_request_hash_mismatch")


def _check_identity(
    parent: Mapping[str, Any], branch: Mapping[str, Any],
    parent_claim: Mapping[str, Any], branch_claim: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> None:
    parent_meta = parent["metadata"]
    branch_meta = branch["metadata"]
    _check_service_identity(parent_meta, branch_meta)
    expected_process = _running_process_identity(prepared.get("running_identity"))
    if expected_process is not None and any(
        str(metadata.get("process_instance_id")) != expected_process
        for metadata in (parent_meta, branch_meta)
    ):
        raise _Unavailable("service_process_identity_mismatch")
    _check_receipt_identities(parent, branch, parent_claim, branch_claim)
    _check_parent_identity(parent_meta, branch_meta)
    _check_fork_session_identity(parent_claim["session"], branch_claim["session"])


def _check_service_identity(parent: Mapping[str, Any], branch: Mapping[str, Any]) -> None:
    required = ("service_request_id", "process_instance_id", "workload_epoch")
    if any(field not in parent or field not in branch for field in required):
        raise _Unavailable("service_identity_missing")
    for field in ("process_instance_id", "workload_epoch"):
        if parent[field] != branch[field]:
            raise _Unavailable("service_identity_changed")


def _check_receipt_identities(
    parent: Mapping[str, Any], branch: Mapping[str, Any],
    parent_claim: Mapping[str, Any], branch_claim: Mapping[str, Any],
) -> None:
    for wire, claim in ((parent, parent_claim), (branch, branch_claim)):
        receipt_id = wire["metadata"].get("receipt_id")
        if receipt_id is not None and receipt_id != claim["receipt_id"]:
            raise _Unavailable("receipt_identity_mismatch")


def _check_parent_identity(parent: Mapping[str, Any], branch: Mapping[str, Any]) -> None:
    if branch.get("parent_service_request_id") != parent["service_request_id"]:
        raise _Unavailable("parent_service_request_identity_missing")


def _check_fork_session_identity(
    parent: Mapping[str, Any], branch: Mapping[str, Any]
) -> None:
    if parent["session_id"] == branch["session_id"]:
        raise _Unavailable("fresh_fork_session_missing")


def _check_fork_controls(
    branch_request: Mapping[str, Any],
    parent_claim: Mapping[str, Any], branch_claim: Mapping[str, Any],
) -> None:
    parent_field = branch_request.get("leone_fork_session")
    session_field = branch_request.get("leone_session")
    if not isinstance(parent_field, str) or not isinstance(session_field, str):
        raise _Unavailable("explicit_fork_controls_missing")
    if parent_field != parent_claim["session"]["session_id"]:
        raise _Unavailable("fork_parent_identity_mismatch")
    if session_field != branch_claim["session"]["session_id"]:
        raise _Unavailable("fork_session_identity_mismatch")
    if branch_claim["session"]["reuse_class"] != "device-fork":
        raise _Unavailable("explicit_fork_replay_class_missing")
    parent_session_id = branch_request.get("parent_session_id")
    if parent_session_id is not None and parent_session_id != parent_field:
        raise _Unavailable("fork_parent_identity_mismatch")


def _response_content(wire: Mapping[str, Any]) -> str:
    full, deltas, direct = _response_texts(wire["events"])
    candidates = full or deltas or direct
    if not candidates:
        raise _Unavailable("assistant_text_missing")
    if full and len(set(full)) != 1:
        raise _Unavailable("assistant_text_ambiguous")
    return candidates[-1] if full else "".join(candidates)


def _response_texts(events: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str], list[str]]:
    full: list[str] = []
    deltas: list[str] = []
    direct: list[str] = []
    for event in events:
        event_full, event_deltas, event_direct = _response_event_texts(event)
        full.extend(event_full)
        deltas.extend(event_deltas)
        direct.extend(event_direct)
    return full, deltas, direct


def _response_event_texts(event: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    full: list[str] = []
    deltas: list[str] = []
    direct: list[str] = []
    for payload in _nested_mappings(event):
        _append_direct_text(payload, direct)
        choices = payload.get("choices")
        if isinstance(choices, list):
            _append_choice_text(choices, full, deltas)
    return full, deltas, direct


def _append_direct_text(payload: Mapping[str, Any], direct: list[str]) -> None:
    for field in ("_content_text", "content"):
        value = payload.get(field)
        if isinstance(value, str):
            direct.append(value)


def _append_choice_text(
    choices: Sequence[Any], full: list[str], deltas: list[str]
) -> None:
    for choice in choices:
        if not isinstance(choice, Mapping):
            continue
        message = choice.get("message")
        if isinstance(message, Mapping) and isinstance(message.get("content"), str):
            full.append(message["content"])
        delta = choice.get("delta")
        if isinstance(delta, Mapping) and isinstance(delta.get("content"), str):
            deltas.append(delta["content"])


def _run_tokenizer(
    prepared: Mapping[str, Any], parent_request: Mapping[str, Any],
    branch_request: Mapping[str, Any], assistant_text: str,
) -> dict[str, Any]:
    """Run the pinned CPU llama server for both prompts and candidate text."""

    generator = _template_generator()
    with tempfile.TemporaryDirectory(prefix="leone-history-tokenization-") as directory:
        template_file = Path(directory) / "chat-template.jinja"
        template_file.write_bytes(prepared["template_bytes"])
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            process, endpoint = generator.start_server(
                prepared["binary"], prepared["model"], template_file
            )
        finally:
            _restore_env("CUDA_VISIBLE_DEVICES", old_cuda)
        try:
            parent = _apply_and_tokenize(prepared, endpoint, parent_request)
            branch = _apply_and_tokenize(prepared, endpoint, branch_request)
            generated = _tokenize_text(prepared, endpoint, assistant_text, "generated")
            return {
                "parent_tokens": parent["tokens"],
                "branch_tokens": branch["tokens"],
                "generated_tokens": generated["tokens"],
                "parent_rendered": parent["rendered"],
                "branch_rendered": branch["rendered"],
                "records": {
                    "parent": parent["records"],
                    "branch": branch["records"],
                    "generated": generated["records"],
                },
            }
        finally:
            generator.stop_server(process)


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def _apply_and_tokenize(
    prepared: Mapping[str, Any], endpoint: str, request: Mapping[str, Any]
) -> dict[str, Any]:
    body = _apply_body(prepared, request)
    applied, apply_record = _post_json(endpoint, "/apply-template", body)
    rendered = applied.get("prompt") if isinstance(applied, Mapping) else None
    if not isinstance(rendered, str):
        raise _Unavailable("apply_template_response_invalid")
    tokenized = _tokenize_text(prepared, endpoint, rendered, "prompt")
    tokenized["records"]["apply_template"] = apply_record
    tokenized["rendered"] = rendered
    return tokenized


def _apply_body(prepared: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    body = {"messages": _messages(request)}
    for field in APPLY_FIELDS:
        if field in request:
            body[field] = request[field]
    body.setdefault("add_generation_prompt", True)
    policy = prepared["special_tokens_policy"]
    kwargs = policy.get("chat_template_kwargs")
    if "chat_template_kwargs" not in body and isinstance(kwargs, Mapping):
        body["chat_template_kwargs"] = dict(kwargs)
    return body


def _tokenize_body(prepared: Mapping[str, Any], text: str, kind: str) -> dict[str, Any]:
    policy = prepared["special_tokens_policy"]
    flags = policy.get(kind, policy.get("prompt", policy.get("tokenize", policy)))
    if kind == "generated":
        flags = policy.get("generated", {"add_special": False, "parse_special": flags["parse_special"]})
    return {"content": text, **_token_flags(flags, kind)}


def _tokenize_text(
    prepared: Mapping[str, Any], endpoint: str, text: str, kind: str
) -> dict[str, Any]:
    response, record = _post_json(endpoint, "/tokenize", _tokenize_body(prepared, text, kind))
    tokens = response.get("tokens") if isinstance(response, Mapping) else None
    if not _valid_tokens(tokens, prepared["vocab_size"]):
        raise _Unavailable("tokenize_response_invalid")
    return {"tokens": tokens, "records": {"tokenize": record}}


def _valid_tokens(tokens: Any, vocabulary: int) -> bool:
    return isinstance(tokens, list) and all(
        isinstance(token, int) and not isinstance(token, bool) and 0 <= token < vocabulary
        for token in tokens
    )


def _post_json(endpoint: str, route: str, body: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    request_bytes = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}{route}", data=request_bytes, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            response_bytes = response.read(MAX_TOKENIZER_RESPONSE_BYTES + 1)
    except Exception as error:
        raise _Unavailable("tokenizer_request_failed") from error
    if len(response_bytes) > MAX_TOKENIZER_RESPONSE_BYTES:
        raise _Unavailable("tokenizer_response_too_large")
    parsed = _load_json(response_bytes)
    if not isinstance(parsed, dict):
        raise _Unavailable("tokenizer_response_invalid")
    return parsed, {
        "request_sha256": sha256_bytes(request_bytes),
        "response_sha256": sha256_bytes(response_bytes),
        "request_hex": request_bytes.hex(),
        "response_hex": response_bytes.hex(),
    }


def _check_signed_tokens(
    claim: Mapping[str, Any], parent_tokens: Sequence[int], generated_tokens: Sequence[int]
) -> None:
    transcript = [*parent_tokens, *generated_tokens]
    checks = (
        ("prompt_tokens", len(parent_tokens)),
        ("generated_tokens", len(generated_tokens)),
    )
    for field, expected in checks:
        if claim[field] != expected:
            raise _Unavailable("signed_token_count_mismatch")
    if claim["prompt_tokens_sha256"] != uint32_le_sha256(parent_tokens):
        raise _Unavailable("signed_prompt_token_hash_mismatch")
    if claim["response_tokens_sha256"] != uint32_le_sha256(generated_tokens):
        raise _Unavailable("signed_response_token_hash_mismatch")
    if claim["transcript_sha256"] != uint32_le_sha256(transcript):
        raise _Unavailable("signed_transcript_hash_mismatch")


def _check_branch_prompt(claim: Mapping[str, Any], branch_tokens: Sequence[int]) -> None:
    if claim["prompt_tokens"] != len(branch_tokens):
        raise _Unavailable("branch_prompt_token_count_mismatch")
    if claim["prompt_tokens_sha256"] != uint32_le_sha256(branch_tokens):
        raise _Unavailable("branch_prompt_token_hash_mismatch")


def _checked_reuse(
    parent_claim: Mapping[str, Any], branch_claim: Mapping[str, Any],
    parent_tokens: Sequence[int], generated_tokens: Sequence[int], branch_tokens: Sequence[int],
) -> tuple[list[int], int, int, int]:
    n = branch_claim["session"]["cached_tokens"]
    expected = len(parent_tokens) + len(generated_tokens) - 1
    if len(generated_tokens) < 2 or len(parent_tokens) == 0:
        raise _Unavailable("plain_history_boundary_too_short")
    if n != expected:
        raise _Unavailable("cached_count_does_not_match_evaluated_boundary")
    if n <= len(parent_tokens) or n > len(parent_tokens) + len(generated_tokens):
        raise _Unavailable("evaluated_boundary_invalid")
    evaluated = [*parent_tokens, *generated_tokens][:n]
    lcp = _longest_common_prefix(evaluated, branch_tokens)
    if len(branch_tokens) <= len(evaluated) or lcp != len(evaluated):
        raise _Unavailable("retained_history_is_not_full_branch_prefix")
    observed = branch_claim["session"]["reused_tokens"]
    if observed != len(evaluated):
        raise _Unavailable("reused_count_does_not_match_evaluated_boundary")
    del parent_claim
    return evaluated, expected, observed, lcp


def _longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    count = 0
    for one, two in zip(left, right):
        if one != two:
            break
        count += 1
    return count


def _evidence(
    prepared: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any],
    parent_request_bytes: bytes, branch_request_bytes: bytes,
    prefix_messages: Sequence[Mapping[str, Any]], branch_messages: Sequence[Mapping[str, Any]],
    parent_tokens: Sequence[int], generated_tokens: Sequence[int], evaluated: Sequence[int],
    branch_tokens: Sequence[int], expected: int, observed: int, lcp: int,
    records: Mapping[str, Any], proof_adapter: str = "leone.signed-receipt",
    slot_copy: Mapping[str, Any] | None = None,
    apply_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    oracle = {
        "engine": prepared.get("engine", "leone"),
        "source_commit": prepared["source_commit"],
        "executable_sha256": prepared["executable_sha256"],
        "loaded_library_sha256": prepared["loaded_library_sha256"],
        "gguf_sha256": prepared["model_sha256"],
        "tokenizer_metadata_sha256": prepared["tokenizer_metadata_sha256"],
        "tokenizer_hash_scheme": prepared["tokenizer_metadata_hash_scheme"],
        "tokenizer_metadata_hash_scheme": prepared["tokenizer_metadata_hash_scheme"],
        "template_config_sha256": prepared["template_config_sha256"],
        "template_config_hash_scheme": prepared["template_config_hash_scheme"],
        "template_bytes_sha256": prepared["template_bytes_sha256"],
        "special_tokens_policy_sha256": prepared["special_tokens_policy_sha256"],
        "special_tokens_policy": prepared["special_tokens_policy"],
        "vocab_size": prepared["vocab_size"],
        "apply_template_request_sha256": _record_digest(records, "apply_template", "request_sha256"),
        "apply_template_response_sha256": _record_digest(records, "apply_template", "response_sha256"),
        "tokenize_request_sha256": _record_digest(records, "tokenize", "request_sha256"),
        "tokenize_response_sha256": _record_digest(records, "tokenize", "response_sha256"),
        "raw_parent_events_sha256": parent["raw_sha256"],
        "raw_branch_events_sha256": branch["raw_sha256"],
        "parent_receipt_sha256": parent.get("receipt_sha256"),
        "branch_receipt_sha256": branch.get("receipt_sha256"),
        "proof_adapter": proof_adapter,
        "apply_template_tokenize": records,
    }
    if slot_copy is not None:
        oracle["slot_copy"] = dict(slot_copy)
    if apply_kwargs is not None:
        oracle["apply_template_kwargs"] = dict(apply_kwargs)
        oracle["evidence_class"] = "retained_and_recomputed"
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "status": "observed",
        "scope": SCOPE,
        "parent_service_request_id": parent["metadata"]["service_request_id"],
        "service_request_id": branch["metadata"]["service_request_id"],
        "process_instance_id": parent["metadata"]["process_instance_id"],
        "workload_epoch": parent["metadata"]["workload_epoch"],
        "parent_receipt_sha256": parent.get("receipt_sha256"),
        "branch_receipt_sha256": branch.get("receipt_sha256"),
        "parent_request_sha256": sha256_bytes(parent_request_bytes),
        "request_sha256": sha256_bytes(branch_request_bytes),
        "canonical_prefix_sha256": sha256_bytes(canonical_json(prefix_messages)),
        "canonical_request_sha256": sha256_bytes(canonical_json(branch_messages)),
        "parent_prompt_token_ids": list(parent_tokens),
        "parent_generated_token_ids": list(generated_tokens),
        "parent_evaluated_token_ids": list(evaluated),
        "request_token_ids": list(branch_tokens),
        "parent_evaluated_token_count": len(evaluated),
        "expected_reused_token_count": expected,
        "observed_reused_token_count": observed,
        "oracle": oracle,
    }
    return evidence


def _record_digest(records: Mapping[str, Any], operation: str, field: str) -> str:
    values = []
    for name in ("parent", "branch", "generated"):
        record = records.get(name)
        value = record.get(operation, {}).get(field) if isinstance(record, Mapping) else None
        if isinstance(value, str):
            values.append(value)
    return sha256_bytes(canonical_json(values))


def harness_tokenization_record(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt observed evidence to the branching study tokenization record."""

    if evidence.get("status") != "observed":
        return {"status": "unavailable", "reason": evidence.get("reason", "producer_unavailable")}
    prefix = evidence.get("parent_evaluated_token_ids")
    request = evidence.get("request_token_ids")
    oracle = evidence.get("oracle")
    if not isinstance(prefix, list) or not isinstance(request, list) or not isinstance(oracle, Mapping):
        return {"status": "unavailable", "reason": "producer_evidence_incomplete"}
    service_id = evidence.get("service_request_id")
    vocabulary = oracle.get("vocab_size")
    if not isinstance(service_id, str) or not isinstance(vocabulary, int):
        return {"status": "unavailable", "reason": "producer_identity_incomplete"}
    return {
        "status": "observed",
        "prefix_token_ids": list(prefix),
        "request_token_ids": list(request),
        "prefix_token_count": len(prefix),
        "request_token_count": len(request),
        "prefix_token_ids_sha256": sha256_bytes(canonical_json(prefix)),
        "request_token_ids_sha256": sha256_bytes(canonical_json(request)),
        "tokenizer_sha256": oracle.get("tokenizer_metadata_sha256"),
        "chat_template_sha256": oracle.get("template_config_sha256"),
        "special_tokens_policy_sha256": oracle.get("special_tokens_policy_sha256"),
        "canonical_prefix_sha256": evidence.get("canonical_prefix_sha256"),
        "canonical_request_sha256": evidence.get("canonical_request_sha256"),
        "service_request_id": service_id,
        "vocab_size": vocabulary,
    }


def harness_tokenization_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Project producer evidence onto the frozen harness evidence shape."""

    if evidence.get("status") != "observed":
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": evidence.get("reason")}
    oracle = evidence.get("oracle")
    if isinstance(oracle, Mapping) and oracle.get("proof_adapter") not in {
        "leone.signed-receipt",
        "llama.cpp.verbose",
    }:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": "unsupported_proof_adapter"}
    if any(field not in evidence for field in HARNESS_EVIDENCE_FIELDS):
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": "producer_evidence_incomplete"}
    return {field: evidence[field] for field in HARNESS_EVIDENCE_FIELDS}


@functools.lru_cache(maxsize=1)
def _template_generator() -> Any:
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts" / "generate-openai-chat-template-fixtures.py"
    spec = importlib.util.spec_from_file_location("leone_template_fixture_generator", source)
    if spec is None or spec.loader is None:
        raise _Unavailable("template_generator_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit("import produce_history_tokenization from this helper")
