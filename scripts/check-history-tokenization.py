#!/usr/bin/env python3
"""Re-run the retained llama.cpp history tokenizer requests on a fresh pinned CPU server.

The command reads one retained branching study receipt (or one run record), a
caller-owned expected-identity file, and explicit local paths for the pinned
llama-server binary, the model, and the template. For every retained branch it
calls `produce_history_tokenization` from the producer beside this script,
which starts a fresh CPU llama-server and replays `/apply-template` and
`/tokenize` for the parent prompt, the generated text, and the branch prompt.
Each fresh response and token ID list is compared with the retained record.

It writes one result file with status `verified`, `rejected`, or `incomplete`.
The result shows that the retained tokenizer responses equal a fresh run of the
pinned binary. It does not show GPU KV contents, server timestamps, or peer
server execution. It does not run the offline recompute of the harness.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "leone.history-reexecution.v1"
EXPECTED_SCHEMA = "leone.history-reexecution-expected.v1"
RECEIPT_SCHEMA = "leone.branching-service.v1"
LLAMA_KINDS = ("llama.cpp", "llama_cpp")
PROOF_ADAPTER = "llama.cpp.verbose"
TEMPLATE_MODES = ("official", "legacy")
MAX_INPUT_BYTES = 64 << 20
MAX_EXPECTED_BYTES = 1 << 20
MAX_SCRIPT_BYTES = 4 << 20
MAX_REQUEST_BYTES = 1 << 20
MAX_RAW_EVENTS = 4096
MAX_RUNS = 64
MAX_BRANCHES_PER_RUN = 16
MAX_CHECKED_BRANCHES = 64
CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "LLAMA_ARG_DEVICE": "none"}
EXIT_CODES = {"verified": 0, "rejected": 1, "incomplete": 2}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
NAME_RE = re.compile(r"^[a-z0-9_]{1,80}$")
EXPECTED_DIGESTS = (
    "input_sha256",
    "producer_sha256",
    "template_generator_sha256",
    "executable_sha256",
    "loaded_library_sha256",
    "model_sha256",
    "template_config_sha256",
    "template_bytes_sha256",
    "special_tokens_policy_sha256",
)
EXPECTED_FIELDS = frozenset(
    ("schema_version", "source_commit", "template_mode", "vocab_size", "prompt_prefix",
     "special_tokens_policy", *EXPECTED_DIGESTS)
)
ORACLE_IDENTITY_FIELDS = (
    "engine", "source_commit", "executable_sha256", "loaded_library_sha256", "gguf_sha256",
    "tokenizer_metadata_sha256", "tokenizer_metadata_hash_scheme", "template_config_sha256",
    "template_config_hash_scheme", "template_bytes_sha256", "special_tokens_policy_sha256",
    "vocab_size", "proof_adapter",
)
BRANCH_HASH_FIELDS = (
    "parent_request_sha256", "request_sha256", "canonical_prefix_sha256", "canonical_request_sha256",
)
NOT_COVERED = (
    "gpu_kv_contents",
    "server_execution_timestamps",
    "peer_server_execution",
    "offline_recompute_consistency",
    "loaded_library_reobservation",
)
INCOMPLETE_PRODUCER_REASONS = frozenset((
    "engine_declaration_missing", "engine_artifact_path_missing", "pinned_artifact_missing",
    "pinned_digest_missing", "template_mode_unsupported", "template_generator_missing",
    "official_template_missing", "template_not_utf8", "vocabulary_size_missing",
    "special_tokens_policy_missing", "prompt_token_policy_incomplete",
    "generated_token_policy_incomplete", "generated_token_policy_missing",
    "tokenizer_request_failed", "tokenizer_response_invalid", "tokenizer_response_too_large",
    "tokenize_response_invalid",
    "apply_template_response_invalid", "producer_error", "request_bytes_required",
    "request_object_required", "messages_missing", "message_shape_unsupported",
    "message_content_unsupported", "raw_events_unsupported", "raw_events_empty",
    "raw_events_invalid", "raw_events_not_utf8", "invalid_json_input", "token_id_out_of_range",
    "producer_result_invalid",
))


class Stop(Exception):
    """Carries one typed result: `rejected` for a disagreement, `incomplete` for a gap."""

    def __init__(self, status: str, reason: str, detail: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.status = status
        self.reason = reason
        self.detail = dict(detail or {})


def canonical_json(value: Any) -> bytes:
    """Encode a value with the hash convention of the study."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def safe_name(value: Any, default: str = "invalid_name") -> str:
    """Return a lowercase identifier, or the default when the value is not one."""

    return value if isinstance(value, str) and NAME_RE.fullmatch(value) else default


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON object key")
        output[key] = value
    return output


def _reject_constant(name: str) -> Any:
    raise ValueError(f"non-finite JSON constant {name}")


def parse_json(data: bytes, reason: str) -> Any:
    """Parse strict UTF-8 JSON with unique keys and finite numbers."""

    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise Stop("incomplete", reason) from error


def read_bounded(path: Path, limit: int, name: str) -> bytes:
    """Read a file of at most `limit` bytes. A larger file is incomplete, never truncated."""

    try:
        with path.open("rb") as source:
            data = source.read(limit + 1)
    except OSError as error:
        raise Stop("incomplete", f"{name}_unreadable") from error
    if len(data) > limit:
        raise Stop("incomplete", f"{name}_too_large")
    return data


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and DIGEST_RE.fullmatch(value) is not None


def load_expected(data: bytes) -> dict[str, Any]:
    """Validate the caller-owned expected-identity file. Every field is required."""

    value = parse_json(data, "expected_invalid_json")
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_SCHEMA:
        raise Stop("incomplete", "expected_schema_unsupported")
    if value.get("template_mode") not in TEMPLATE_MODES:
        raise Stop("incomplete", "expected_template_mode_unsupported")
    required = set(EXPECTED_FIELDS)
    if value["template_mode"] == "official":
        required.add("embedded_template_sha256")
    for field in sorted(required - set(value)):
        raise Stop("incomplete", "expected_field_missing", {"field": safe_name(field)})
    for field in sorted(set(value) - required):
        raise Stop("incomplete", "expected_field_unsupported", {"field": safe_name(field)})
    _check_expected_values(value)
    return value


def _invalid_expected_fields(value: Mapping[str, Any]) -> list[str]:
    vocabulary = value["vocab_size"]
    checks = {
        "source_commit": isinstance(value["source_commit"], str)
        and COMMIT_RE.fullmatch(value["source_commit"]) is not None,
        "prompt_prefix": isinstance(value["prompt_prefix"], str),
        "special_tokens_policy": isinstance(value["special_tokens_policy"], dict),
        "vocab_size": _is_count(vocabulary) and vocabulary > 0,
    }
    for name in (*EXPECTED_DIGESTS, "embedded_template_sha256"):
        if name in value:
            checks[name] = _valid_digest(value[name])
    return [name for name, valid in checks.items() if not valid]


def _check_expected_values(value: Mapping[str, Any]) -> None:
    invalid = _invalid_expected_fields(value)
    if invalid:
        raise Stop("incomplete", "expected_value_invalid", {"field": safe_name(invalid[0])})


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def fingerprint(path: Path, name: str) -> dict[str, Any]:
    """Hash a regular file and record its stat identity from one open handle."""

    try:
        with path.open("rb") as source:
            first = os.fstat(source.fileno())
            if not stat.S_ISREG(first.st_mode):
                raise Stop("incomplete", "artifact_not_a_file", {"artifact": name})
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1 << 20), b""):
                digest.update(block)
            second = os.fstat(source.fileno())
    except OSError as error:
        raise Stop("incomplete", "artifact_unreadable", {"artifact": name}) from error
    if _stat_identity(first) != _stat_identity(second):
        raise Stop("rejected", "artifact_changed_during_read", {"artifact": name})
    return {"sha256": digest.hexdigest(), "stat": _stat_identity(first)}


def _script_paths() -> dict[str, Path]:
    directory = Path(__file__).resolve().parent
    return {
        "producer": directory / "produce-history-tokenization.py",
        "template_generator": directory / "generate-openai-chat-template-fixtures.py",
    }


def take_snapshot(paths: Mapping[str, Path], producer_source: bytes) -> dict[str, dict[str, Any]]:
    """Fingerprint every artifact that the fresh run trusts."""

    snapshot = {
        name: fingerprint(path, name) for name, path in paths.items() if name != "producer"
    }
    snapshot["producer"] = {"sha256": sha256_bytes(producer_source), "stat": None}
    return snapshot


def _artifact_pins(expected: Mapping[str, Any]) -> dict[str, str]:
    return {
        "producer": expected["producer_sha256"],
        "template_generator": expected["template_generator_sha256"],
        "llama_server": expected["executable_sha256"],
        "model": expected["model_sha256"],
        "template": expected["template_config_sha256"],
    }


def check_pins(snapshot: Mapping[str, Mapping[str, Any]], expected: Mapping[str, Any]) -> None:
    """Reject the first artifact whose actual digest differs from the caller pin."""

    for name, pinned in _artifact_pins(expected).items():
        if snapshot[name]["sha256"] != pinned:
            raise Stop("rejected", "artifact_digest_mismatch", {"artifact": name})


def check_unchanged(
    before: Mapping[str, Mapping[str, Any]], after: Mapping[str, Mapping[str, Any]]
) -> None:
    """Reject any artifact whose digest or stat identity moved during the run."""

    for name, first in before.items():
        second = after[name]
        moved = first["sha256"] != second["sha256"]
        if first["stat"] is not None and first["stat"] != second["stat"]:
            moved = True
        if moved:
            raise Stop("rejected", "artifact_mutated_during_run", {"artifact": name})


def load_producer(path: Path, source: bytes) -> Any:
    """Compile the producer from the bytes that were hashed, not from a second read."""

    spec = importlib.util.spec_from_file_location("leone_history_tokenization_checked", path)
    if spec is None:
        raise Stop("incomplete", "producer_unloadable")
    module = importlib.util.module_from_spec(spec)
    try:
        exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    except Exception as error:  # noqa: BLE001
        raise Stop("incomplete", "producer_unloadable") from error
    for name in ("produce_history_tokenization", "harness_tokenization_evidence"):
        if not callable(getattr(module, name, None)):
            raise Stop("incomplete", "producer_entrypoint_missing")
    return module


def producer_engine(
    expected: Mapping[str, Any], paths: Mapping[str, Path], running_identity: Any
) -> dict[str, Any]:
    """Build the producer engine mapping from caller pins and explicit paths only."""

    engine: dict[str, Any] = {
        "protocol": "llama.cpp",
        "engine": "llama.cpp",
        "llama_server_path": str(paths["llama_server"]),
        "model_path": str(paths["model"]),
        "llama_cpp_commit": expected["source_commit"],
        "executable_sha256": expected["executable_sha256"],
        "model_sha256": expected["model_sha256"],
        "loaded_library_sha256": expected["loaded_library_sha256"],
        "template_mode": expected["template_mode"],
        "template_config_sha256": expected["template_config_sha256"],
        "template_bytes_sha256": expected["template_bytes_sha256"],
        "special_tokens_policy": json.loads(json.dumps(expected["special_tokens_policy"])),
        "special_tokens_policy_sha256": expected["special_tokens_policy_sha256"],
        "vocab_size": expected["vocab_size"],
        "prompt_prefix": expected["prompt_prefix"],
        "_running_identity": running_identity,
    }
    if expected["template_mode"] == "official":
        engine["tokenizer_config"] = str(paths["template"])
        engine["embedded_template_sha256"] = expected["embedded_template_sha256"]
    else:
        engine["template_file"] = str(paths["template"])
    return engine


def check_date_pinned(expected: Mapping[str, Any], template_bytes: bytes) -> None:
    """Refuse a template that reads the clock unless the policy pins `date_string`."""

    if b"strftime_now" not in template_bytes:
        return
    kwargs = expected["special_tokens_policy"].get("chat_template_kwargs")
    if not isinstance(kwargs, Mapping) or not isinstance(kwargs.get("date_string"), str):
        raise Stop("incomplete", "template_date_not_pinned")


def _launch_problem(argv: Sequence[str], env: Mapping[str, str]) -> str | None:
    flags = list(zip(argv, argv[1:]))
    if ("--n-gpu-layers", "0") not in flags:
        return "fresh_server_gpu_layers_not_zero"
    if any(env.get(name) != value for name, value in CPU_ENV.items()):
        return "fresh_server_gpu_environment_not_disabled"
    return None


def _redacted_launch(argv: Sequence[str], env: Mapping[str, str]) -> dict[str, Any]:
    roles = {"-m": "<model>", "--chat-template-file": "<template>", "--port": "<port>"}
    flags = ["<llama-server>"] + [roles.get(prev, token) for prev, token in zip(argv, argv[1:])]
    return {"argv": flags, "env": {name: env.get(name) for name in CPU_ENV}}


class LaunchGuard:
    """Wraps `subprocess.Popen` and refuses server launches that could use a GPU."""

    def __init__(self, real: Callable[..., Any]):
        self.real = real
        self.launches: list[dict[str, Any]] = []
        self.refusals: list[str] = []

    def __call__(self, args: Any, *positional: Any, **keywords: Any) -> Any:
        argv = [str(item) for item in args] if isinstance(args, (list, tuple)) else []
        if "--port" in argv:
            env = keywords.get("env") or os.environ
            problem = _launch_problem(argv, env)
            if problem is not None:
                self.refusals.append(problem)
                raise RuntimeError(problem)
            self.launches.append(_redacted_launch(argv, env))
        return self.real(args, *positional, **keywords)


@contextlib.contextmanager
def cpu_only_launches() -> Iterator[LaunchGuard]:
    """Set the CPU-only environment and guard every server launch inside the block."""

    guard = LaunchGuard(subprocess.Popen)
    saved = {name: os.environ.get(name) for name in CPU_ENV}
    os.environ.update(CPU_ENV)
    subprocess.Popen = guard  # type: ignore[misc,assignment]
    try:
        yield guard
    finally:
        subprocess.Popen = guard.real  # type: ignore[misc]
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _request_bytes(record: Mapping[str, Any], label: Mapping[str, Any]) -> bytes:
    encoded = record.get("_request_bytes_hex")
    try:
        raw = bytes.fromhex(encoded) if isinstance(encoded, str) else b""
    except ValueError:
        raw = b""
    if not raw or len(raw) > MAX_REQUEST_BYTES:
        raise Stop("incomplete", "request_bytes_invalid", label)
    return raw


def _request_slot(raw: bytes) -> int | None:
    try:
        slot = parse_json(raw, "request_invalid_json").get("id_slot")
    except (Stop, AttributeError):
        return None
    return slot if _is_count(slot) else None


def _identity_body(run: Mapping[str, Any]) -> Mapping[str, Any]:
    running = run.get("running_identity")
    start = running.get("start") if isinstance(running, Mapping) else None
    start = start if isinstance(start, Mapping) else running
    identity = start.get("identity") if isinstance(start, Mapping) else None
    return identity if isinstance(identity, Mapping) else {}


def _end_metrics_body(run: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = run.get("service_metrics")
    end = metrics.get("end") if isinstance(metrics, Mapping) else None
    body = end.get("body") if isinstance(end, Mapping) else None
    return body if isinstance(body, Mapping) else {}


def _wrapper_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for field in ("process_instance_id", "workload_epoch", "source_id"):
        for body in (_identity_body(run), _end_metrics_body(run)):
            value = body.get(field)
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                found[field] = value
                break
    return found


def _slot_copy(run: Mapping[str, Any], slot: int | None, start_ns: Any) -> dict[str, Any] | None:
    plan = run.get("slot_copy")
    if not isinstance(plan, Mapping) or slot is None or slot == plan.get("parent_slot"):
        return None
    restore = plan.get("restore")
    return {
        "save": plan.get("save"),
        "restore": restore.get(str(slot)) if isinstance(restore, Mapping) else None,
        "branch_request_start_ns": start_ns,
    }


def _raw_wrapper(
    run: Mapping[str, Any], record: Mapping[str, Any], label: Mapping[str, Any],
    parent_id: Any = None, slot: int | None = None,
) -> dict[str, Any]:
    events = record.get("_raw_events")
    if not isinstance(events, list) or not 0 < len(events) <= MAX_RAW_EVENTS:
        raise Stop("incomplete", "raw_events_missing", label)
    wrapper: dict[str, Any] = {
        "events": events,
        "service_request_id": record.get("service_request_id") or record.get("response_id"),
    }
    if parent_id is not None:
        wrapper["parent_service_request_id"] = parent_id
        copy = _slot_copy(run, slot, record.get("request_start_ns"))
        if copy is not None:
            wrapper["slot_copy"] = copy
    wrapper.update(_wrapper_identity(run))
    return wrapper


def _retained_tokenization(branch: Any, label: Mapping[str, Any]) -> Mapping[str, Any]:
    history = branch.get("history_reuse") if isinstance(branch, Mapping) else None
    record = history.get("tokenization") if isinstance(history, Mapping) else None
    if not isinstance(record, Mapping) or record.get("status") != "observed":
        raise Stop("incomplete", "retained_tokenization_missing", label)
    oracle = record.get("oracle")
    if not isinstance(oracle, Mapping) or oracle.get("proof_adapter") != PROOF_ADAPTER:
        raise Stop("incomplete", "proof_adapter_unsupported", label)
    return record


def _branch_item(
    run: Mapping[str, Any], run_index: int, index: int, branch: Any,
    parent_request: bytes, parent_wrapper: Mapping[str, Any],
) -> dict[str, Any]:
    label = {"run_index": run_index, "branch_index": index}
    retained = _retained_tokenization(branch, label)
    request = _request_bytes(branch, label)
    parent_id = parent_wrapper["service_request_id"]
    wrapper = _raw_wrapper(run, branch, label, parent_id, _request_slot(request))
    return {
        "label": label,
        "retained": retained,
        "running_identity": run.get("running_identity"),
        "inputs": (parent_request, dict(parent_wrapper), request, wrapper),
    }


def _run_parts(run: Any, label: Mapping[str, Any]) -> tuple[Mapping[str, Any], list[Any]]:
    parent = run.get("parent") if isinstance(run, Mapping) else None
    branches = run.get("branches") if isinstance(run, Mapping) else None
    if not isinstance(parent, Mapping) or not isinstance(branches, list) or not branches:
        raise Stop("incomplete", "run_record_unsupported", label)
    if len(branches) > MAX_BRANCHES_PER_RUN:
        raise Stop("incomplete", "too_many_branches_in_run", label)
    return parent, branches


def run_items(run_index: int, run: Any) -> list[dict[str, Any]]:
    """Return one item per retained branch of one run record."""

    label = {"run_index": run_index}
    parent, branches = _run_parts(run, label)
    parent_request = _request_bytes(parent, label)
    wrapper = _raw_wrapper(run, parent, label)
    if not wrapper["service_request_id"]:
        raise Stop("incomplete", "parent_identity_missing", label)
    return [
        _branch_item(run, run_index, index, branch, parent_request, wrapper)
        for index, branch in enumerate(branches)
    ]


def _engine_kinds(document: Mapping[str, Any]) -> dict[Any, Any]:
    rows = document.get("engines")
    if not isinstance(rows, list):
        raise Stop("incomplete", "input_schema_unsupported")
    return {row.get("id"): row.get("kind") for row in rows if isinstance(row, Mapping)}


def _receipt_run_list(document: Any) -> list[Any]:
    if not isinstance(document, dict) or document.get("schema_version") != RECEIPT_SCHEMA:
        raise Stop("incomplete", "input_schema_unsupported")
    runs = document.get("runs")
    if not isinstance(runs, list) or not runs or len(runs) > MAX_RUNS:
        raise Stop("incomplete", "input_run_count_unsupported")
    return runs


def receipt_runs(document: Any) -> tuple[list[tuple[int, Any]], int]:
    """Select the llama.cpp runs of a study receipt and count the runs left out."""

    runs = _receipt_run_list(document)
    kinds = _engine_kinds(document)
    chosen = []
    for index, run in enumerate(runs):
        engine = run.get("engine") if isinstance(run, Mapping) else None
        if engine not in kinds:
            raise Stop("incomplete", "engine_row_missing", {"run_index": index})
        if kinds[engine] in LLAMA_KINDS:
            chosen.append((index, run))
    return chosen, len(runs) - len(chosen)


def collect_inputs(document: Any, input_format: str) -> tuple[list[dict[str, Any]], int]:
    """Turn the retained input into bounded branch items and a skipped-run count."""

    if input_format == "receipt":
        runs, skipped = receipt_runs(document)
    else:
        runs, skipped = [(0, document)], 0
    items = [item for index, run in runs for item in run_items(index, run)]
    if not items:
        raise Stop("incomplete", "no_llama_history_records")
    if len(items) > MAX_CHECKED_BRANCHES:
        raise Stop("incomplete", "too_many_branches")
    return items, skipped


def _ids_equal(left: Any, right: Any) -> bool:
    """Compare token ID lists exactly. `True` is not 1 here."""

    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    return all(type(a) is int and type(b) is int and a == b for a, b in zip(left, right))


def _response_tokens(exchange: Any) -> list[int]:
    body = parse_json(bytes.fromhex(exchange["response_hex"]), "fresh_record_invalid")
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(type(item) is int for item in tokens):
        raise Stop("incomplete", "fresh_record_invalid")
    return tokens


def fresh_token_ids(fresh: Mapping[str, Any]) -> dict[str, list[int]]:
    """Read the token IDs straight from the fresh `/tokenize` response bytes."""

    try:
        records = fresh["oracle"]["apply_template_tokenize"]
        return {
            "parent": _response_tokens(records["parent"]["tokenize"]),
            "generated": _response_tokens(records["generated"]["tokenize"]),
            "branch": _response_tokens(records["branch"]["tokenize"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise Stop("incomplete", "fresh_record_invalid") from error


def _check_fresh_status(fresh: Any) -> None:
    if isinstance(fresh, Mapping) and fresh.get("status") == "observed":
        return
    reason = fresh.get("reason") if isinstance(fresh, Mapping) else None
    code = safe_name(reason, "producer_result_invalid")
    raise Stop("incomplete" if code in INCOMPLETE_PRODUCER_REASONS else "rejected", code)


def _compare_token_ids(retained: Mapping[str, Any], ids: Mapping[str, list[int]]) -> None:
    pairs = (
        ("parent_prompt_token_ids", ids["parent"]),
        ("parent_generated_token_ids", ids["generated"]),
        ("request_token_ids", ids["branch"]),
    )
    for field, expected in pairs:
        if not _ids_equal(retained.get(field), expected):
            raise Stop("rejected", "token_ids_differ", {"field": field})
    count = retained.get("parent_evaluated_token_count")
    evaluated = [*ids["parent"], *ids["generated"]][: count if _is_count(count) else 0]
    if not _ids_equal(retained.get("parent_evaluated_token_ids"), evaluated):
        raise Stop("rejected", "token_ids_differ", {"field": "parent_evaluated_token_ids"})


def _oracle_difference(retained: Any, fresh: Mapping[str, Any]) -> str:
    if not isinstance(retained, Mapping):
        return "oracle"
    for key in sorted(set(retained) | set(fresh)):
        if canonical_json(retained.get(key)) != canonical_json(fresh.get(key)):
            return safe_name(key)
    return "oracle"


def _compare_fields(
    retained: Mapping[str, Any], projected: Mapping[str, Any], fields: Sequence[str]
) -> None:
    for field in fields:
        if canonical_json(retained[field]) == canonical_json(projected[field]):
            continue
        detail = {"field": field}
        if field == "oracle":
            detail["oracle_key"] = _oracle_difference(retained[field], projected[field])
        raise Stop("rejected", "retained_record_differs_from_fresh_run", detail)


def compare_reexecution(
    retained: Any, fresh: Any, project: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    """Compare one retained tokenization record with its fresh evidence.

    Token IDs are compared first, against the fresh `/tokenize` response bytes.
    The fresh IDs of the generated text must also equal the retained `verbose`
    IDs that the fresh evidence carries. Then every frozen evidence field is
    compared, including the retained tokenizer request and response bytes.
    """

    _check_fresh_status(fresh)
    projected = project(fresh)
    if projected.get("status") != "observed":
        raise Stop("incomplete", "fresh_evidence_incomplete")
    if not isinstance(retained, Mapping) or set(retained) != set(fields):
        raise Stop("incomplete", "retained_record_partial")
    ids = fresh_token_ids(fresh)
    if not _ids_equal(fresh.get("parent_generated_token_ids"), ids["generated"]):
        raise Stop("rejected", "generated_token_ids_differ_from_fresh_tokenizer")
    _compare_token_ids(retained, ids)
    _compare_fields(retained, projected, fields)
    return dict(projected)


def branch_summary(label: Mapping[str, Any], retained: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one checked branch by hashes only."""

    oracle = retained["oracle"]
    summary = {**label, **{field: retained[field] for field in BRANCH_HASH_FIELDS}}
    for field in ("parent_prompt_token_ids", "parent_generated_token_ids", "request_token_ids"):
        summary[f"{field}_sha256"] = sha256_bytes(canonical_json(retained[field]))
    summary["retained_record_sha256"] = sha256_bytes(canonical_json(retained))
    for field in ("apply_template", "tokenize"):
        summary[f"{field}_response_sha256"] = oracle[f"{field}_response_sha256"]
    return summary


def _check_launch(fresh: Any, guard: LaunchGuard, launched: int) -> None:
    if guard.refusals:
        raise Stop("incomplete", guard.refusals[-1])
    observed = isinstance(fresh, Mapping) and fresh.get("status") == "observed"
    if observed and len(guard.launches) != launched + 1:
        raise Stop("incomplete", "fresh_server_not_launched")


def _reexecute_branch(
    module: Any, engine: Mapping[str, Any], item: Mapping[str, Any],
    guard: LaunchGuard, report: dict[str, Any],
) -> None:
    launched = len(guard.launches)
    try:
        fresh = module.produce_history_tokenization(engine, *item["inputs"])
        _check_launch(fresh, guard, launched)
        compare_reexecution(
            item["retained"], fresh, module.harness_tokenization_evidence,
            module.HARNESS_EVIDENCE_FIELDS,
        )
    except Stop as stop:
        stop.detail.update(item["label"])
        raise
    if report["identity"]["oracle"] is None:
        oracle = fresh["oracle"]
        report["identity"]["oracle"] = {name: oracle[name] for name in ORACLE_IDENTITY_FIELDS}
    report["checked_branches"].append(branch_summary(item["label"], item["retained"]))


def _identity_record(
    expected: Mapping[str, Any], snapshot: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    return {
        "source_commit": expected["source_commit"],
        "executable_sha256": snapshot["llama_server"]["sha256"],
        "model_sha256": snapshot["model"]["sha256"],
        "template_config_sha256": snapshot["template"]["sha256"],
        "template_mode": expected["template_mode"],
        "producer_sha256": snapshot["producer"]["sha256"],
        "template_generator_sha256": snapshot["template_generator"]["sha256"],
        "loaded_library_sha256": expected["loaded_library_sha256"],
        "loaded_library_status": "pinned_by_caller_not_reobserved",
        "special_tokens_policy_sha256": sha256_bytes(canonical_json(expected["special_tokens_policy"])),
        "prompt_prefix_sha256": sha256_bytes(expected["prompt_prefix"].encode("utf-8")),
        "vocab_size": expected["vocab_size"],
        "oracle": None,
    }


def canonical_path(value: str) -> Path:
    """Resolve symlinks and relative parts, so every spelling of one file is one path.

    The fresh server reports the model path it was given. A spelling such as
    `/tmp/m.gguf` on a host where `/tmp` links to `/private/tmp` would differ
    from the retained canonical spelling. A missing file resolves without error
    and fails later as a typed `artifact_unreadable`.
    """

    return Path(os.path.realpath(value))


def _explicit_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        **_script_paths(),
        "llama_server": canonical_path(args.llama_server),
        "model": canonical_path(args.model),
        "template": canonical_path(args.template),
    }


def _prepare(
    args: argparse.Namespace, expected: Mapping[str, Any], report: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], Any]:
    paths = _explicit_paths(args)
    source = read_bounded(paths["producer"], MAX_SCRIPT_BYTES, "producer")
    snapshot = take_snapshot(paths, source)
    report["identity"] = _identity_record(expected, snapshot)
    check_pins(snapshot, expected)
    check_date_pinned(expected, read_bounded(paths["template"], MAX_SCRIPT_BYTES, "template"))
    return paths, snapshot, load_producer(paths["producer"], source)


def _load_input(
    args: argparse.Namespace, expected: Mapping[str, Any], report: dict[str, Any]
) -> list[dict[str, Any]]:
    data = read_bounded(Path(args.input), MAX_INPUT_BYTES, "input")
    report["input"] = {
        "format": args.input_format, "sha256": sha256_bytes(data), "bytes": len(data),
    }
    if report["input"]["sha256"] != expected["input_sha256"]:
        raise Stop("rejected", "input_digest_mismatch")
    items, skipped = collect_inputs(parse_json(data, "input_invalid_json"), args.input_format)
    report["input"]["runs_not_llama_cpp"] = skipped
    return items


def _check_mutation(paths: Mapping[str, Path], before: Mapping[str, Mapping[str, Any]]) -> None:
    source = read_bounded(paths["producer"], MAX_SCRIPT_BYTES, "producer")
    check_unchanged(before, take_snapshot(paths, source))


def _verify(args: argparse.Namespace, report: dict[str, Any]) -> None:
    raw = read_bounded(Path(args.expected), MAX_EXPECTED_BYTES, "expected")
    report["expected_sha256"] = sha256_bytes(raw)
    expected = load_expected(raw)
    items = _load_input(args, expected, report)
    paths, before, module = _prepare(args, expected, report)
    with cpu_only_launches() as guard:
        try:
            for item in items:
                engine = producer_engine(expected, paths, item["running_identity"])
                _reexecute_branch(module, engine, item, guard, report)
        finally:
            report["fresh_server_launches"] = guard.launches
    _check_mutation(paths, before)


def new_report() -> dict[str, Any]:
    """Return the report with every observed value unset."""

    return {
        "input": None, "expected_sha256": None, "identity": None,
        "fresh_server_launches": [], "checked_branches": [],
    }


def finish(
    report: Mapping[str, Any], status: str, reason: str | None, detail: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Build the result. Unset values stay null. A count is never inferred."""

    return {
        "schema_version": SCHEMA_VERSION,
        "check": "fresh_tokenizer_reexecution",
        "status": status,
        "reason": reason,
        "failure": dict(detail) if detail else None,
        "basis": "retained tokenizer responses compared with a fresh pinned CPU llama-server run",
        "offline_recompute": "not_evaluated",
        "not_covered": list(NOT_COVERED),
        **report,
        "checked_branch_count": len(report["checked_branches"]),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    """Run the check and return a typed result. It does not raise on a bad input."""

    report = new_report()
    try:
        _verify(args, report)
    except Stop as stop:
        return finish(report, stop.status, stop.reason, stop.detail)
    except Exception as error:  # noqa: BLE001
        detail = {"error_type": safe_name(type(error).__name__.lower(), "unknown")}
        return finish(report, "incomplete", "unexpected_error", detail)
    return finish(report, "verified", None, None)


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write the result with a hard link, so an existing file is never replaced."""

    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".reexecution-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(data)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line. Every option is required."""

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="retained study receipt or run record")
    parser.add_argument("--input-format", required=True, choices=("receipt", "run"))
    parser.add_argument("--expected", required=True, help="caller-owned expected-identity JSON")
    parser.add_argument("--llama-server", required=True, help="pinned llama-server binary")
    parser.add_argument("--model", required=True, help="pinned GGUF model")
    parser.add_argument("--template", required=True, help="tokenizer config or legacy template file")
    parser.add_argument("--output", required=True, help="new result file; never overwritten")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Check, write the result, and exit 0 (verified), 1 (rejected), or 2 (incomplete)."""

    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        print("history reexecution: incomplete output_exists", file=sys.stderr)
        return 2
    result = execute(args)
    try:
        write_result(output, result)
    except OSError:
        print("history reexecution: incomplete output_unwritable", file=sys.stderr)
        return 2
    print(f"history reexecution: {result['status']} {result['reason'] or ''}".rstrip())
    return EXIT_CODES[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
