#!/usr/bin/env python3
"""Run or validate a reproducible cross-engine session branching study.

The harness records parent and branch outcomes for each configured engine. It
uses only the request fields declared by that engine's manifest entry.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import contextvars
import difflib
import hashlib
import http.client
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = "leone.branching-service.v1"
QUANTILE_DEFINITION = "linear interpolation between adjacent sorted observations at h=(n-1)p"
DEFAULT_READ_BYTES = 1024 * 1024
STREAM_READ_BYTES = 4096
MAX_SCHEDULE_WORKERS = 64
MAX_STREAM_EVENTS = 4096
MAX_SSE_LINE_BYTES = 1024 * 1024
MAX_CONTENT_BYTES = 4 * 1024 * 1024
MAX_RETAINED_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
THRESHOLD_METRICS = {
    "ttft_p95_ms",
    "inter_token_latency_p95_ms",
    "fork_latency_p95_ms",
    "cancel_latency_p95_ms",
    "request_success_rate",
    "history_reuse_min_tokens",
    "backpressure_observed",
    "schedule_overlap_observed",
    "physical_memory_peak_bytes",
}
THRESHOLD_OPERATORS = {"<", "<=", "==", ">=", ">"}
THRESHOLD_UNITS = {"ms", "ratio", "tokens", "boolean", "bytes"}
THRESHOLD_SCENARIOS = {"parent", "branch", "new_prompt", "cancel", "slow_reader", "schedule"}
THRESHOLD_METRIC_UNITS = {
    "ttft_p95_ms": "ms",
    "inter_token_latency_p95_ms": "ms",
    "fork_latency_p95_ms": "ms",
    "cancel_latency_p95_ms": "ms",
    "request_success_rate": "ratio",
    "history_reuse_min_tokens": "tokens",
    "backpressure_observed": "boolean",
    "schedule_overlap_observed": "boolean",
    "physical_memory_peak_bytes": "bytes",
}
REQUIRED_SCHEDULE_ROLES = {"new_prompt", "cancel", "slow_reader"}
KNOWN_OUTCOMES = {
    "completed",
    "cancel_requested",
    "cancelled",
    "cancel_acknowledgement_unavailable",
    "connection_error",
    "http_error",
    "incomplete_stream",
    "stream_limit_exceeded",
    "probe_error",
    "unsupported",
    "unavailable",
    "slow_client",
    "rejected",
    "deadline_expired",
    "client_disconnected",
    "execution_error",
    "queued_expired",
}
BACKPRESSURE_REASONS = {"output_queue_high_water", "write_blocked"}
REQUIRED_THRESHOLD_SCOPES = {
    ("branch", "branch"),
    ("new_prompt", "new_prompt"),
    ("cancel", "cancel"),
    ("slow_reader", "slow_reader"),
    ("schedule", "schedule"),
}
REQUIRED_THRESHOLD_METRICS = {
    ("branch", "branch"): {
        "inter_token_latency_p95_ms",
        "fork_latency_p95_ms",
        "history_reuse_min_tokens",
    },
    ("new_prompt", "new_prompt"): {"ttft_p95_ms", "inter_token_latency_p95_ms"},
    ("cancel", "cancel"): {"cancel_latency_p95_ms"},
    ("slow_reader", "slow_reader"): {"inter_token_latency_p95_ms", "backpressure_observed"},
    ("schedule", "schedule"): {"schedule_overlap_observed", "physical_memory_peak_bytes"},
}


def _measured(source: str) -> Dict[str, str]:
    return {"status": "measured", "source": source}


def _unsupported(reason: str) -> Dict[str, str]:
    return {"status": "unsupported", "reason": reason}


# Each engine kind carries a fixed evidence contract. A manifest can restate it
# but cannot change it, so no declaration can waive a Leone gate. For a
# `measured` llama.cpp metric the receipt must hold the retained wire bytes.
# Token counts compare only at a limit stop, where both engines report the
# number of sampled tokens: llama.cpp `n_decoded` and Leone `generated.tokens`.
TOKEN_COUNT_DEFINITION = "usage_completion_tokens_at_limit_stop"
EVIDENCE_CAPABILITIES = {
    "leone": {
        "ttft_p95_ms": _measured("client_stream_receive_times"),
        "completion_token_count": _measured("usage_completion_tokens"),
        "history_reuse_min_tokens": _measured("signed_response_receipt"),
        "schedule_overlap_observed": _measured("client_request_intervals"),
        "branch_setup_ms": _measured("fork_inside_branch_interval"),
        "inter_token_latency_p95_ms": _measured("engine_token_boundary"),
        "fork_latency_p95_ms": _measured("service_metrics_snapshot"),
        "cancel_latency_p95_ms": _measured("service_metrics_snapshot"),
        "backpressure_observed": _measured("service_metrics_snapshot"),
        "physical_memory_peak_bytes": _measured("service_metrics_snapshot"),
    },
    "llama.cpp": {
        "ttft_p95_ms": _measured("client_stream_receive_times"),
        "completion_token_count": _measured("usage_completion_tokens"),
        "history_reuse_min_tokens": _measured("cached_tokens_and_producer_recompute"),
        "schedule_overlap_observed": _measured("client_request_intervals"),
        "branch_setup_ms": _measured("slot_save_restore_exchanges"),
        "inter_token_latency_p95_ms": _unsupported("sse_has_no_per_token_boundary"),
        "fork_latency_p95_ms": _unsupported("no_fork_method"),
        "cancel_latency_p95_ms": _unsupported("no_cancel_acknowledgement"),
        "backpressure_observed": _unsupported("no_server_backpressure_event"),
        "physical_memory_peak_bytes": _unsupported("no_physical_memory_tracker"),
    },
}
PHYSICAL_PEAK_DEFINITION = (
    "physical_tracker_peak_bytes is the selected parent ledger peak_live_bytes; "
    "backend class samples are separate"
)
HISTORY_TOKEN_SCHEMA = "leone.history-tokenization.v1"
HISTORY_TOKEN_SCOPE = "plain_length_explicit_fork"
HISTORY_TOKEN_FIELDS = {
    "schema_version", "status", "scope", "parent_service_request_id", "service_request_id",
    "process_instance_id", "workload_epoch", "parent_receipt_sha256", "branch_receipt_sha256",
    "parent_request_sha256", "request_sha256", "canonical_prefix_sha256",
    "canonical_request_sha256", "parent_prompt_token_ids", "parent_generated_token_ids",
    "parent_evaluated_token_ids", "request_token_ids", "parent_evaluated_token_count",
    "expected_reused_token_count", "observed_reused_token_count", "oracle",
}
HISTORY_ORACLE_REQUIRED = {
    "engine", "source_commit", "executable_sha256", "loaded_library_sha256", "gguf_sha256",
    "template_config_sha256", "template_bytes_sha256", "special_tokens_policy_sha256",
    "vocab_size", "apply_template_request_sha256", "apply_template_response_sha256",
    "tokenize_request_sha256", "tokenize_response_sha256",
}
CALIBRATION_DECISION_FIELDS = {
    "receipt_sha256",
    "observation",
    "rule",
    "derived_value",
    "decision_sha256",
}
CALIBRATION_RULES = {"identity", "upper_10_percent", "lower_10_percent", "boolean_true"}
CALIBRATION_RULES_BY_METRIC = {
    "ttft_p95_ms": {"identity", "upper_10_percent"},
    "inter_token_latency_p95_ms": {"identity", "upper_10_percent"},
    "fork_latency_p95_ms": {"identity", "upper_10_percent"},
    "cancel_latency_p95_ms": {"identity", "upper_10_percent"},
    "physical_memory_peak_bytes": {"identity", "upper_10_percent"},
    "history_reuse_min_tokens": {"identity", "lower_10_percent"},
    "request_success_rate": {"identity", "lower_10_percent"},
}
MEMORY_COLLECTION_LOSS_FIELDS = {
    "request_started", "request_finished", "token_boundaries", "outcomes",
    "cancellation", "slow_client", "fork_latency", "physical_bytes",
    "scheduler_reservations", "process_memory", "system_memory", "total", "overflowed",
}
MINIMUM_FROZEN_SAMPLES = 12
QUALITY_RECORD_SCHEMAS = {"cuda": "leone.quality-comparison.v2", "metal": "leone.quality-cross-device.v2"}
QUALITY_RECEIPT_SCHEMA = 3
QUALITY_PRODUCERS = {"leone": "same_executable", "llama_cpp": "peer_adapter"}
QUALITY_PATH_STATUS = "eval_path_not_served_path"
LOADED_LIBRARY_STATUS = "resolved_linkage_not_process_map"
QUALITY_RECOMPUTATION_STATUS = "not_run_by_harness"
QUALITY_POLICY = {
    "name": "canonical_v2_common_oracle",
    "sample_contract": "linspace-inclusive-v1:128+task-rows",
    "release_validation": "quality-stage-v1",
}
TASK_RESULT_SCHEMA = "leone.quality-task-result.v2"
SOURCE_SCOPES = ("repository", "archive")
ARCHIVE_REQUIRED_SOURCES = (
    "scripts/study-branching-service.py", "scripts/produce-history-tokenization.py",
    "scripts/check-history-tokenization.py", "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/linked_libraries.py", "scripts/source_inputs.py", "fixtures/qwen3-legacy-chatml.jinja",
)
MINIMUM_CALIBRATION_SAMPLES = 2
PROMPT_DISJOINT_MAX_SEQUENCE_RATIO = 0.8
SERVICE_OUTCOMES = {
    "completed": "finished",
    "cancelled": "cancelled",
    "deadline_expired": "deadline_expired",
    "rejected": "rejected",
    "client_disconnected": "client_disconnected",
    "execution_error": "execution_error",
}


class StudyError(RuntimeError):
    """Reports an invalid manifest or unrecoverable study setup."""


def canonical_json(value: Any) -> bytes:
    """Encode JSON with stable key ordering for provenance hashes."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: Any) -> bool:
    """Return whether a value is a lowercase SHA-256 digest."""

    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _valid_commit(value: Any) -> bool:
    """Return whether a source commit uses a complete hexadecimal identity."""

    return isinstance(value, str) and len(value) in {40, 64} and all(
        character in "0123456789abcdef" for character in value
    )


def root_path(root: Path, value: str) -> Path:
    """Resolve one relative manifest path while rejecting escapes."""

    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("manifest paths must be relative and contained by root")
    root_resolved = root.resolve()
    resolved = (root_resolved / path).resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise ValueError("manifest path escapes root")
    return resolved


def _manifest_path_error(value: Any, label: str) -> Optional[str]:
    """Reject absolute and parent paths before a manifest is executed."""

    if not isinstance(value, str) or not value:
        return f"{label} must be a nonempty relative path"
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return f"{label} must stay within the study root"
    return None


def _relative_root_path(root: Path, path: Path) -> str:
    """Return one canonical root-relative path for a receipt reference."""

    root_resolved = root.resolve()
    path_resolved = path.resolve()
    if path_resolved != root_resolved and root_resolved not in path_resolved.parents:
        raise ValueError("receipt path escapes root")
    return path_resolved.relative_to(root_resolved).as_posix()


def linear_quantile(values: Sequence[float], probability: float) -> Optional[float]:
    """Return a linear-interpolation quantile, or `None` for no values."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bounded_history(values: Sequence[Any], capacity: int) -> Dict[str, Any]:
    """Retain a bounded suffix and report every evicted value."""

    if capacity <= 0:
        raise ValueError("history capacity must be positive")
    dropped = max(0, len(values) - capacity)
    return {
        "capacity": capacity,
        "values": list(values[-capacity:]),
        "dropped_count": dropped,
        "sample_count": len(values),
        "truncated": dropped > 0,
    }


def empirical_quantiles(
    values: Sequence[float], minimum_samples: int, truncated: bool = False
) -> Dict[str, Any]:
    """Return p50, p95, and qualification without hiding truncation."""

    count = len(values)
    qualified = count >= minimum_samples and not truncated
    result: Dict[str, Any] = {
        "definition": QUANTILE_DEFINITION,
        "sample_count": count,
        "minimum_samples": minimum_samples,
        "qualified": qualified,
        "truncated": truncated,
        "values": {},
    }
    if truncated:
        result["reason"] = "sample_history_truncated"
    elif count < minimum_samples:
        result["reason"] = "insufficient_samples"
    else:
        result["values"] = {
            "p50": linear_quantile(values, 0.50),
            "p95": linear_quantile(values, 0.95),
        }
    return result


def latency_metrics(
    started_ns: int,
    observed_ns: Sequence[int],
    token_boundaries_verified: bool,
    token_boundary_ns: Optional[Sequence[int]] = None,
    token_boundary_indexes: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Calculate client TTFT and ITL from receive-time observations."""

    content_timestamps = list(observed_ns)
    boundary_timestamps = _boundary_timestamps(content_timestamps, token_boundary_ns, token_boundaries_verified)
    _check_timestamps(started_ns, content_timestamps)
    _check_timestamps(started_ns, boundary_timestamps)
    ttft = _ttft_ms(started_ns, content_timestamps)
    intervals = _intervals_ms(boundary_timestamps)
    itl = _itl_measurement(intervals, token_boundaries_verified)
    content_intervals = _intervals_ms(content_timestamps)
    return {
        "clock_domain": "client_monotonic_ns",
        "ttft_definition": "first_content_receive_after_request_start",
        "itl_definition": "adjacent_explicit_token_boundary_receive_times",
        "request_start_ns": started_ns,
        "content_receive_ns": content_timestamps,
        "token_boundary_receive_ns": boundary_timestamps,
        "ttft_ms": _measurement(ttft, "no_token_observed"),
        "inter_token_latency_ms": itl,
        "inter_token_latency_p95_ms": _itl_p95(itl),
        "inter_token_latency_samples_ms": intervals if token_boundaries_verified else [],
        "content_event_count": len(content_timestamps),
        "content_event_elapsed_ms": [(value - started_ns) / 1_000_000 for value in content_timestamps],
        "content_event_intervals_ms": [None] + content_intervals if content_timestamps else [],
        "token_boundary_count": len(boundary_timestamps),
        "token_boundaries_verified": token_boundaries_verified,
        "token_boundary_indexes": (
            list(token_boundary_indexes) if token_boundary_indexes is not None else []
        ),
    }


def _boundary_timestamps(
    content_timestamps: Sequence[int],
    token_boundary_ns: Optional[Sequence[int]],
    verified: bool,
) -> List[int]:
    """Select client boundary timestamps without using engine clocks."""

    if token_boundary_ns is not None:
        return list(token_boundary_ns)
    return list(content_timestamps) if verified else []


def _ttft_ms(started_ns: int, timestamps: Sequence[int]) -> Optional[float]:
    """Return the first content receive latency."""

    return None if not timestamps else (timestamps[0] - started_ns) / 1_000_000


def _intervals_ms(timestamps: Sequence[int]) -> List[float]:
    """Return adjacent receive-time intervals in milliseconds."""

    return [(right - left) / 1_000_000 for left, right in zip(timestamps, timestamps[1:])]


def _itl_p95(itl: Mapping[str, Any]) -> Dict[str, Any]:
    """Return p95 with the same explicit absence state as ITL."""

    if itl["status"] != "observed":
        return _measurement(None, itl["reason"] or "no_inter_token_latency")
    return _measurement(itl["value"]["values"]["p95"], "no_inter_token_latency")


def _check_timestamps(started_ns: int, timestamps: Sequence[int]) -> None:
    """Reject timestamps that cannot define a latency."""

    if any(value < started_ns for value in timestamps):
        raise ValueError("an observed timestamp precedes request start")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("observed timestamps are not monotonic")


def _itl_measurement(intervals: Sequence[float], verified: bool) -> Dict[str, Any]:
    """Return ITL only when the protocol identifies token boundaries."""

    if not verified:
        return {"value": None, "status": "unavailable", "reason": "token_boundaries_unverified"}
    if not intervals:
        return {"value": None, "status": "unavailable", "reason": "fewer_than_two_tokens"}
    return {"value": empirical_quantiles(intervals, 1), "status": "observed", "reason": None}


def _measurement(value: Any, reason: str) -> Dict[str, Any]:
    """Represent an optional server metric without converting absence to zero."""

    if value is None:
        return {"value": None, "status": "unavailable", "reason": reason}
    return {"value": value, "status": "observed", "reason": None}


def outcome_summary(
    records: Sequence[Mapping[str, Any]],
    minimum_samples: int,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Count every outcome and summarize completed request latencies."""

    outcomes = Counter(str(record.get("status", "missing_status")) for record in records)
    ttft = _completed_ttft(records)
    itl = _completed_itl(records)
    ttft_history = bounded_history(ttft, max_samples) if max_samples is not None else None
    itl_history = bounded_history(itl, max_samples) if max_samples is not None else None
    ttft_summary = empirical_quantiles(
        ttft,
        minimum_samples,
        ttft_history["truncated"] if ttft_history else False,
    )
    itl_summary = empirical_quantiles(
        itl,
        minimum_samples,
        itl_history["truncated"] if itl_history else False,
    )
    return {
        "request_count": len(records),
        "outcomes": dict(sorted(outcomes.items())),
        "ttft_ms": ttft_summary,
        "inter_token_latency_p95_ms": itl_summary,
        "history": {"ttft_ms": ttft_history, "inter_token_latency_p95_ms": itl_history},
    }


def _completed_ttft(records: Sequence[Mapping[str, Any]]) -> List[float]:
    """Collect observed TTFT values from completed records."""

    values = []
    for record in records:
        value = _record_ttft(record)
        if value is not None:
            values.append(value)
    return values


def _completed_itl(records: Sequence[Mapping[str, Any]]) -> List[float]:
    """Collect observed token intervals from completed records."""

    values = []
    for record in records:
        if record.get("status") != "completed":
            continue
        metrics = record.get("metrics")
        metric = metrics.get("inter_token_latency_ms", {}) if isinstance(metrics, dict) else {}
        if isinstance(metric, dict) and metric.get("status") == "observed":
            values.extend(_record_itl(record))
    return values


def _record_ttft(record: Mapping[str, Any]) -> Optional[float]:
    """Return TTFT from raw receive times, with a legacy metric fallback."""

    return _record_ttft_value(record.get("metrics", {})) if record.get("status") == "completed" else None


def _record_ttft_value(metric: Any) -> Optional[float]:
    """Read one TTFT metric from raw timestamps or a legacy measurement."""

    if isinstance(metric, dict) and ("request_start_ns" in metric or "content_receive_ns" in metric):
        return _raw_ttft(metric)
    measurement = metric.get("ttft_ms") if isinstance(metric, dict) else None
    value = measurement.get("value") if isinstance(measurement, dict) else None
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
        else None
    )


def _record_itl(record: Mapping[str, Any]) -> List[float]:
    """Return ITL samples from raw boundaries, with a legacy fallback."""

    return _record_itl_value(record.get("metrics", {}))


def _record_itl_value(metric: Any) -> List[float]:
    """Read ITL samples from raw boundaries or a legacy metric."""

    if isinstance(metric, dict) and ("token_boundary_receive_ns" in metric or metric.get("token_boundaries_verified") is True):
        raw = _raw_intervals(metric)
        return raw if raw is not None else []
    samples = metric.get("inter_token_latency_samples_ms", []) if isinstance(metric, dict) else []
    return _finite_samples(samples)


def _finite_samples(samples: Any) -> List[float]:
    """Return finite nonnegative numeric samples."""

    if not isinstance(samples, list):
        return []
    return [float(sample) for sample in samples if _finite_nonnegative(sample)]


def _raw_ttft(metric: Any) -> Optional[float]:
    """Calculate TTFT from integer client receive timestamps."""

    if not isinstance(metric, dict):
        return None
    started = metric.get("request_start_ns")
    received = _raw_ns_list(metric.get("content_receive_ns"))
    if isinstance(started, bool) or not isinstance(started, int) or not received:
        return None
    try:
        _check_timestamps(started, received)
    except ValueError:
        return None
    return (received[0] - started) / 1_000_000


def _raw_intervals(metric: Any) -> Optional[List[float]]:
    """Calculate ITL samples from integer client token boundaries."""

    if not isinstance(metric, dict) or metric.get("token_boundaries_verified") is not True:
        return None
    boundaries = _raw_ns_list(metric.get("token_boundary_receive_ns"))
    if boundaries is not None:
        started = metric.get("request_start_ns")
        if isinstance(started, int) and not isinstance(started, bool):
            try:
                _check_timestamps(started, boundaries)
            except ValueError:
                return None
        elif "request_start_ns" in metric:
            return None
    return _intervals_ms(boundaries) if boundaries is not None else None


def _raw_ns_list(value: Any) -> Optional[List[int]]:
    """Validate one raw timestamp list before deriving a latency."""

    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return value


class SseParser:
    """Parse data-only SSE events and retain receive timestamps."""

    def __init__(self, max_line_bytes: int = MAX_SSE_LINE_BYTES) -> None:
        self._buffer = b""
        self._data: List[str] = []
        self._max_line_bytes = max_line_bytes

    def feed(self, chunk: bytes, received_ns: int) -> List[Tuple[str, int]]:
        """Consume bytes and return complete event payloads."""

        self._buffer += chunk
        if len(self._buffer) > self._max_line_bytes:
            raise StudyError("SSE line exceeds the configured byte bound")
        events: List[Tuple[str, int]] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            if len(line) > self._max_line_bytes:
                raise StudyError("SSE line exceeds the configured byte bound")
            line = line[:-1] if line.endswith(b"\r") else line
            if line == b"":
                if self._data:
                    events.append(("\n".join(self._data), received_ns))
                    self._data = []
            elif line.startswith(b":"):
                continue
            elif line.startswith(b"data:"):
                self._data.append(line[5:].lstrip().decode("utf-8"))
            else:
                raise StudyError("unsupported SSE field")
        return events

    def finish(self) -> None:
        """Reject an incomplete SSE event."""

        if self._buffer or self._data:
            raise StudyError("SSE stream ended before an event boundary")


def _connection(url: str) -> Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise StudyError(f"only HTTP URLs are supported: {url}")
    return parsed.hostname, parsed.port or 80, parsed.path.rstrip("/")


def _request_json(url: str, method: str, payload: Optional[bytes], timeout_s: float) -> Tuple[int, Dict[str, str], bytes]:
    host, port, path = _connection(url)
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers.update({"Content-Type": "application/json", "Content-Length": str(len(payload))})
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse()
        body = response.read(DEFAULT_READ_BYTES + 1)
        if len(body) > DEFAULT_READ_BYTES:
            raise StudyError("JSON response exceeds the configured byte bound")
        return response.status, dict(response.getheaders()), body
    finally:
        connection.close()


def inspect_branch_method(engine: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Inspect a declared branch capability without assuming support."""

    method = engine["branch_method"]
    if method.get("status") == "unsupported":
        return {"status": "unsupported", "reason": method.get("reason", "manifest_declares_unsupported")}
    endpoint = method.get("inspect_endpoint")
    if not isinstance(endpoint, str):
        return {"status": "unverified", "reason": "branch_inspection_endpoint_missing"}
    base = str(engine["base_url"]).rstrip("/")
    try:
        status, headers, body = _request_json(base + "/" + endpoint.lstrip("/"), "GET", None, timeout_s)
    except (OSError, http.client.HTTPException) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return _inspection_response(method, status, headers, body)


def fetch_metrics_snapshot(engine: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Fetch an optional structured metrics endpoint without inventing fields."""

    endpoint = engine.get("metrics_endpoint")
    if not isinstance(endpoint, str):
        return {"status": "unavailable", "reason": "metrics_endpoint_missing"}
    base = str(engine["base_url"]).rstrip("/")
    try:
        status, _, body = _request_json(base + "/" + endpoint.lstrip("/"), "GET", None, timeout_s)
    except (OSError, http.client.HTTPException) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    if status < 200 or status >= 300:
        return {"status": "unavailable", "http_status": status, "body_sha256": sha256_bytes(body)}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "unavailable", "reason": "metrics_response_not_json", "body_sha256": sha256_bytes(body)}
    return {"status": "observed", "body_sha256": sha256_bytes(body), "body": value}


def fetch_service_trace_snapshot(engine: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Fetch the optional server-clock trace used for sibling progress."""

    endpoint = engine.get("trace_endpoint")
    if not isinstance(endpoint, str):
        return {"status": "unavailable", "reason": "trace_endpoint_missing"}
    base = str(engine["base_url"]).rstrip("/")
    try:
        status, _, body = _request_json(base + "/" + endpoint.lstrip("/"), "GET", None, timeout_s)
    except (OSError, http.client.HTTPException) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    if status < 200 or status >= 300:
        return {"status": "unavailable", "http_status": status, "body_sha256": sha256_bytes(body)}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "unavailable", "reason": "trace_response_not_json", "body_sha256": sha256_bytes(body)}
    return {"status": "observed", "body_sha256": sha256_bytes(body), "body": value}


FILE_HASH_CACHE_LIMIT = 16
_FILE_HASHES: "collections.OrderedDict[Tuple[int, ...], str]" = collections.OrderedDict()
PATH_FLAG_ROLES = {
    "-m": "model", "--model": "model", "--slot-save-path": "slot_save_path",
    "--chat-template-file": "template_file", "--port": "port",
}


def _stat_key(path: Path) -> Tuple[int, ...]:
    """Identify file content by device, inode, size, and both change times.

    Any write, truncate, or `utime` changes the ctime, so a mutated file never
    reuses an earlier hash.
    """

    status = os.stat(path)
    return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)


def cached_file_sha256(path: Path) -> str:
    """Hash one file once per file identity, keeping at most `FILE_HASH_CACHE_LIMIT` entries.

    A file that changes while it is read raises `OSError` and is not cached.
    """

    before = _stat_key(path)
    cached = _FILE_HASHES.get(before)
    if cached is not None:
        _FILE_HASHES.move_to_end(before)
        return cached
    digest = sha256_file(path)
    if _stat_key(path) != before:
        raise OSError("file changed while it was hashed")
    _FILE_HASHES[before] = digest
    while len(_FILE_HASHES) > FILE_HASH_CACHE_LIMIT:
        _FILE_HASHES.popitem(last=False)
    return digest


def _proc_available() -> bool:
    """Return whether this host exposes the Linux process filesystem."""

    return Path("/proc/self/stat").is_file() and Path("/proc/net/tcp").is_file()


def _listening_inodes(port: int) -> set:
    """Return the socket inodes that listen on one local TCP port."""

    inodes = set()
    for name in ("tcp", "tcp6"):
        try:
            rows = Path(f"/proc/net/{name}").read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def _socket_owner_pid(inodes: set) -> Optional[int]:
    """Find the process that holds one of the socket inodes, or None."""

    targets = {f"socket:[{inode}]" for inode in inodes}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            if any(os.readlink(fd) in targets for fd in (entry / "fd").iterdir()):
                return int(entry.name)
        except OSError:
            continue
    return None


def _lsof_port_owners(port: int) -> Optional[set]:
    """List listening pids for one TCP port with `lsof`, or None when it cannot answer."""

    try:
        done = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"],
            capture_output=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = done.stdout.decode("utf-8", "replace").splitlines()
    owners = {int(line[1:]) for line in lines if line.startswith("p") and line[1:].isdigit()}
    return owners if done.returncode in (0, 1) else None


def port_owner_pids(port: int) -> Optional[set]:
    """Return the pids that listen on a port, or None when the host cannot tell."""

    if _proc_available():
        pid = _socket_owner_pid(_listening_inodes(port))
        return {pid} if pid is not None else set()
    return _lsof_port_owners(port)


def boot_session_id() -> Optional[str]:
    """Read the boot session id that scopes a pid: Linux boot id or macOS boot session uuid."""

    try:
        text = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        text = ""
    if text:
        return text
    try:
        done = subprocess.run(["sysctl", "-n", "kern.bootsessionuuid"], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    value = done.stdout.decode("utf-8", "replace").strip()
    return value if done.returncode == 0 and value else None


def _read_proc_argv(pid: int) -> Optional[List[str]]:
    """Read one process argv from the process filesystem."""

    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    return [item.decode("utf-8", "replace") for item in raw.split(b"\0") if item] or None


def _flag_values(argv: Sequence[str]) -> Dict[str, str]:
    """Map each `--flag value` and `--flag=value` pair to its value, keeping the last."""

    values: Dict[str, str] = {}
    for index, item in enumerate(argv):
        if not item.startswith("-"):
            continue
        name, equals, value = item.partition("=")
        if equals:
            values[name] = value
        elif index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            values[name] = argv[index + 1]
        else:
            values.setdefault(name, "")
    return values


def _int_flag(values: Mapping[str, str], *names: str) -> Optional[int]:
    """Read one integer flag under any of its names, or None when absent or not an integer."""

    for name in names:
        text = values.get(name)
        if text is not None and text.lstrip("-").isdigit():
            return int(text)
    return None


def argv_flags(argv: Sequence[str]) -> Dict[str, Any]:
    """Return the public, non-path server flags the comparison depends on."""

    values = _flag_values(argv)
    return {
        "parallel": _int_flag(values, "--parallel", "-np"),
        "ctx_size": _int_flag(values, "--ctx-size", "-c"),
        "n_gpu_layers": _int_flag(values, "--n-gpu-layers", "-ngl", "--gpu-layers"),
        "threads": _int_flag(values, "--threads", "-t"),
        "slot_save_path": "--slot-save-path" in values,
        "alias": values.get("--alias", values.get("-a")),
        "jinja": "--jinja" in values,
        "metrics": "--metrics" in values,
    }


def argv_hashes(argv: Sequence[str]) -> Dict[str, str]:
    """Hash the exact argv and its role form, in which path and port values become role names."""

    roles = ["<executable>"] + _role_values(argv[1:])
    return {
        "argv_sha256": sha256_bytes(canonical_json(list(argv))),
        "argv_roles_sha256": sha256_bytes(canonical_json(roles)),
    }


def _role_values(argv: Sequence[str]) -> List[str]:
    """Replace the value that follows a path flag with the flag's role name."""

    output: List[str] = []
    previous_role: Optional[str] = None
    for item in argv:
        name, equals, _ = item.partition("=")
        if previous_role is not None:
            output.append(previous_role)
            previous_role = None
        elif equals and name in PATH_FLAG_ROLES:
            output.append(f"{name}=<{PATH_FLAG_ROLES[name]}>")
        else:
            output.append(item)
            if name in PATH_FLAG_ROLES and not equals:
                previous_role = f"<{PATH_FLAG_ROLES[name]}>"
    return output


def _model_from_argv(argv: Sequence[str], cwd: Path) -> Optional[Path]:
    """Resolve the `-m` or `--model` value against the process working directory."""

    values = _flag_values(argv)
    value = values.get("--model") or values.get("-m")
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else cwd / path


def _process_cwd(pid: int) -> Optional[Path]:
    """Read the working directory of one process."""

    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def _process_start_ticks(pid: int) -> int:
    """Read the process start time in clock ticks since boot."""

    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def _process_identity_body(
    argv: Sequence[str], executable: str, model: str, source_id: str, start_ns: int, instance: str,
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the identity fields shared by the local and spawned providers.

    `template_sha256` names the bytes of the served `--chat-template-file`. The
    argv hashes bind only the path.
    """

    body = {
        "source_id": source_id, "executable_sha256": executable, "model_sha256": model,
        "process_start_ns": start_ns, "process_instance_id": instance, "workload_epoch": instance,
        **argv_hashes(argv), "flags": argv_flags(argv),
    }
    if template is not None:
        body["template_sha256"] = template
    return body


def _observed_identity(identity: Mapping[str, Any]) -> Dict[str, Any]:
    """Wrap one identity body with its digest."""

    return {"status": "observed", "body_sha256": sha256_bytes(canonical_json(identity)), "identity": dict(identity)}


def _listening_process(engine: Mapping[str, Any]) -> Optional[Tuple[int, List[str], Path]]:
    """Find the listening process for the engine port: its pid, argv, and resolved model path."""

    port = urllib.parse.urlparse(str(engine["base_url"])).port
    pid = _socket_owner_pid(_listening_inodes(port)) if port else None
    argv = _read_proc_argv(pid) if pid is not None else None
    cwd = _process_cwd(pid) if pid is not None else None
    model = _model_from_argv(argv, cwd) if argv and cwd else None
    return (pid, argv, model) if pid is not None and argv and model is not None else None


def _read_local_identity(found: Tuple[int, List[str], Path], source_id: str) -> Dict[str, Any]:
    """Hash the process files and build the identity, or return the typed reason it cannot be read."""

    pid, argv, model = found
    try:
        start_ticks = _process_start_ticks(pid)
        boot_id = boot_session_id()
        executable = cached_file_sha256(Path(f"/proc/{pid}/exe"))
        model_digest = cached_file_sha256(model)
    except (OSError, ValueError, IndexError):
        return {"status": "unavailable", "reason": "local_process_unreadable"}
    if boot_id is None:
        return {"status": "unavailable", "reason": "boot_session_unavailable"}
    start_ns = start_ticks * 1_000_000_000 // os.sysconf("SC_CLK_TCK")
    instance = f"{boot_id}:{pid}:{start_ticks}"
    return _observed_identity(_process_identity_body(argv, executable, model_digest, source_id, start_ns, instance))


def _local_process_identity(engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Identify a llama.cpp server that was already running, from the Linux process filesystem.

    llama.cpp serves no identity endpoint. The process is the epoch: it has no
    workload reset, so `workload_epoch` equals `process_instance_id`. Any read
    that fails yields a typed `unavailable` state, never a partial identity.
    """

    if not _proc_available():
        return {"status": "unavailable", "reason": "proc_filesystem_unavailable"}
    source_id = engine.get("_build_source_id")
    if not isinstance(source_id, str) or not source_id:
        return {"status": "unavailable", "reason": "build_source_unavailable"}
    found = _listening_process(engine)
    if found is None:
        return {"status": "unavailable", "reason": "local_process_not_found"}
    return _read_local_identity(found, source_id)


def _require_port_free(base_url: str) -> None:
    """Reject a study port that another process already answers on."""

    parsed = urllib.parse.urlparse(base_url)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.1)
        if probe.connect_ex((parsed.hostname or "127.0.0.1", parsed.port or 80)) == 0:
            raise StudyError(f"study port is already in use: {parsed.hostname}:{parsed.port}")
    finally:
        probe.close()


def _spawn_argv(
    engine: Mapping[str, Any], binary: Path, model: Path, slot_dir: Path
) -> List[str]:
    """Fill the declared launch argv. Only the named placeholders are accepted."""

    parsed = urllib.parse.urlparse(str(engine["base_url"]))
    values = {
        "executable": str(binary), "model": str(model), "host": parsed.hostname or "127.0.0.1",
        "port": str(parsed.port), "slot_save_path": str(slot_dir) + os.sep,
        "slot_count": str(engine.get("slot_count")),
    }
    try:
        return [item.format_map(values) for item in engine["spawn"]["argv"]]
    except (KeyError, IndexError, ValueError) as error:
        raise StudyError(f"spawn argv uses an unknown placeholder: {error}") from error


class SpawnedServer:
    """One llama.cpp server that the harness starts, identifies first-hand, and stops.

    The harness knows the pid, the argv, and the files it launched, so nothing
    is read back from the process table except the port owner, which must be
    the spawned pid. The server writes slot files only into a private
    directory that this object creates and removes.
    """

    def __init__(self, engine: Mapping[str, Any], root: Path) -> None:
        self.engine = engine
        self.root = root
        self.process: Optional[subprocess.Popen] = None
        self.slot_dir: Optional[Path] = None
        self._launch: Dict[str, Any] = {}

    def __enter__(self) -> "SpawnedServer":
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Hash the launch files, start the process, and wait for its health endpoint."""

        _require_port_free(str(self.engine["base_url"]))
        binary = root_path(self.root, str(self.engine["executable_path"]))
        model = root_path(self.root, str(self.engine["model_artifact"]))
        self.slot_dir = Path(tempfile.mkdtemp(prefix="leone-branching-slots-"))
        argv = _spawn_argv(self.engine, binary, model, self.slot_dir)
        template = self._template_path(argv)
        self._launch = {
            "argv": argv, "binary": binary, "model": model, "template": template,
            "executable_sha256": cached_file_sha256(binary), "model_sha256": cached_file_sha256(model),
            "template_sha256": cached_file_sha256(template) if template is not None else None,
            "boot_id": boot_session_id(),
        }
        self.process = subprocess.Popen(
            argv, cwd=self.root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True
        )
        self._launch["start_ns"] = time.monotonic_ns()
        self._wait_ready(float(self.engine["spawn"]["startup_timeout_s"]))

    def _template_path(self, argv: Sequence[str]) -> Optional[Path]:
        """Resolve the served template file against the study root, like the model path."""

        values = _flag_values(argv)
        value = values.get("--chat-template-file")
        return root_path(self.root, value) if value else None

    def _wait_ready(self, startup_timeout_s: float) -> None:
        """Poll `/health` until it answers, failing at once if the process exits."""

        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self.process is None or self.process.poll() is not None:
                raise StudyError("spawned server exited before it became healthy")
            try:
                status, _, _ = _request_json(str(self.engine["base_url"]).rstrip("/") + "/health", "GET", None, 2.0)
            except (OSError, http.client.HTTPException):
                status = 0
            if status == 200:
                return
            time.sleep(0.1)
        raise StudyError("spawned server did not become healthy in time")

    def identity(self) -> Dict[str, Any]:
        """Return the first-hand identity, or the typed reason it cannot be vouched for."""

        source_id = self.engine.get("_build_source_id")
        launch = self._launch
        if self.process is None or self.process.poll() is not None:
            return {"status": "unavailable", "reason": "spawned_process_not_running"}
        if not isinstance(source_id, str) or not source_id:
            return {"status": "unavailable", "reason": "build_source_unavailable"}
        if launch.get("boot_id") is None:
            return {"status": "unavailable", "reason": "boot_session_unavailable"}
        reason = self._binding_error()
        if reason is not None:
            return {"status": "unavailable", "reason": reason}
        instance = f"{launch['boot_id']}:{self.process.pid}:{launch['start_ns']}"
        return _observed_identity(_process_identity_body(
            launch["argv"], launch["executable_sha256"], launch["model_sha256"], source_id,
            launch["start_ns"], instance, launch["template_sha256"],
        ))

    def _binding_error(self) -> Optional[str]:
        """Check that the launched files are unchanged and the spawned pid owns the port."""

        launch = self._launch
        try:
            if cached_file_sha256(launch["binary"]) != launch["executable_sha256"]:
                return "spawned_executable_changed"
            if cached_file_sha256(launch["model"]) != launch["model_sha256"]:
                return "spawned_model_changed"
            if launch["template"] is not None and cached_file_sha256(launch["template"]) != launch["template_sha256"]:
                return "spawned_template_changed"
        except OSError:
            return "spawned_file_unreadable"
        port = urllib.parse.urlparse(str(self.engine["base_url"])).port
        owners = port_owner_pids(port) if port else None
        if owners is None:
            return "port_owner_unavailable"
        return None if owners == {self.process.pid} else "port_owner_is_not_the_spawned_process"

    def stop(self) -> None:
        """Stop the process and remove the private slot directory this object created."""

        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if self.slot_dir is not None:
            shutil.rmtree(self.slot_dir, ignore_errors=True)
            self.slot_dir = None


def _remove_owned_slot_file(directory: Path, filename: Any) -> str:
    """Delete one saved slot file from a directory this harness created, never elsewhere."""

    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".slot"):
        return "refused"
    target = directory / filename
    if target.is_symlink() or not target.is_file():
        return "missing"
    target.unlink()
    return "removed"


def _release_slot_files(engine: Mapping[str, Any], plan: Any) -> Any:
    """Record whether the slot file was removed. llama.cpp has no delete endpoint.

    `action=erase` clears slot memory and leaves the saved file. Only a
    spawned server has a directory the harness owns, so only there is a file removed.
    """

    if not isinstance(plan, Mapping):
        return plan
    directory = engine.get("_slot_dir")
    status = _remove_owned_slot_file(directory, plan.get("filename")) if isinstance(directory, Path) else "not_owned"
    return {**plan, "cleanup": {"status": status}}


def fetch_server_identity(engine: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Read build and process identity from the running server when declared."""

    provider = engine.get("identity_provider")
    if provider == "local_process":
        return _local_process_identity(engine)
    if provider == "spawned_process":
        server = engine.get("_spawned")
        if isinstance(server, SpawnedServer):
            return server.identity()
        return {"status": "unavailable", "reason": "spawned_process_not_started"}
    return _endpoint_identity(engine, timeout_s)


def _endpoint_identity(engine: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
    """Read the identity object a declared identity endpoint serves."""

    endpoint = engine.get("identity_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return {"status": "unavailable", "reason": "running_identity_endpoint_missing"}
    base = str(engine["base_url"]).rstrip("/")
    try:
        status, _, body = _request_json(base + "/" + endpoint.lstrip("/"), "GET", None, timeout_s)
    except (OSError, http.client.HTTPException) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    if status < 200 or status >= 300:
        return {"status": "unavailable", "http_status": status, "body_sha256": sha256_bytes(body)}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "unavailable", "reason": "running_identity_not_json", "body_sha256": sha256_bytes(body)}
    if not isinstance(value, dict):
        return {"status": "unavailable", "reason": "running_identity_not_object", "body_sha256": sha256_bytes(body)}
    if not _identity_fields_valid(value):
        return {"status": "unavailable", "reason": "running_identity_fields_missing", "body_sha256": sha256_bytes(body)}
    return {"status": "observed", "body_sha256": sha256_bytes(body), "identity": value}


def _identity_fields_valid(value: Mapping[str, Any]) -> bool:
    """Check the fields required to bind a response to one running process."""

    if not isinstance(value.get("source_id"), str) or not value["source_id"]:
        return False
    for field in ("executable_sha256", "model_sha256"):
        if not _valid_digest(value.get(field)):
            return False
    process_start = value.get("process_start_ns")
    return isinstance(process_start, int) and not isinstance(process_start, bool) and process_start >= 0


def _inspection_response(
    method: Mapping[str, Any], status: int, headers: Mapping[str, str], body: bytes
) -> Dict[str, Any]:
    """Interpret one capability response without inferring missing methods."""

    if status < 200 or status >= 300:
        return {"status": "unavailable", "http_status": status, "body_sha256": sha256_bytes(body)}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "unavailable", "reason": "inspection_response_not_json", "body_sha256": sha256_bytes(body)}
    methods = value.get("branch_methods") if isinstance(value, dict) else None
    name = method.get("name")
    supported = isinstance(methods, list) and name in methods
    return {
        "status": "supported" if supported else "unsupported",
        "method": name,
        "advertised_methods": methods if isinstance(methods, list) else None,
        "body_sha256": sha256_bytes(body),
        "response_headers": {key.lower(): value for key, value in headers.items() if key.lower() == "etag"},
    }


def _content_and_timestamps(events: Sequence[Tuple[Mapping[str, Any], int]]) -> Tuple[str, List[int], Optional[Dict[str, int]], Optional[str]]:
    content: List[str] = []
    timestamps: List[int] = []
    usage: Optional[Dict[str, int]] = None
    response_id: Optional[str] = None
    for event, timestamp in events:
        response_id = response_id or event.get("id")
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or len(choices) > 1:
            continue
        if choices:
            delta = choices[0].get("delta", {})
            piece = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(piece, str) and piece:
                content.append(piece)
                timestamps.append(timestamp)
    return "".join(content), timestamps, usage, response_id


def _event_has_content(event: Mapping[str, Any]) -> bool:
    """Return whether one event contains a nonempty assistant content delta."""

    return bool(_event_content(event))


def _event_content(event: Mapping[str, Any]) -> str:
    """Return one assistant content delta, or an empty string."""

    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta", {})
    piece = delta.get("content") if isinstance(delta, dict) else None
    return piece if isinstance(piece, str) else ""


def _token_timing(
    events: Sequence[Tuple[Mapping[str, Any], int]],
    method: Mapping[str, Any],
    usage: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[int], bool]:
    """Return receive times only for a complete declared token sequence."""

    boundary = method.get("token_boundary")
    field, event_field, event_value = _boundary_spec(method, boundary)
    if not isinstance(field, str):
        return [], False
    timestamps: List[int] = []
    indexes: List[int] = []
    for event, received_ns in events:
        marker = _token_boundary_value(event, field, event_field, event_value)
        if marker is not None:
            indexes.append(marker)
            timestamps.append(received_ns)
    if not timestamps or not _valid_token_indexes(indexes, usage):
        return [], False
    return timestamps, True


def _token_indexes(
    events: Sequence[Tuple[Mapping[str, Any], int]],
    method: Mapping[str, Any],
    usage: Optional[Mapping[str, Any]],
) -> List[int]:
    """Return the verified token indexes represented by one response."""

    boundary = method.get("token_boundary")
    field, event_field, event_value = _boundary_spec(method, boundary)
    if not isinstance(field, str):
        return []
    indexes = [
        marker
        for event, _ in events
        for marker in [_token_boundary_value(event, field, event_field, event_value)]
        if marker is not None
    ]
    return indexes if _valid_token_indexes(indexes, usage) else []


def _boundary_spec(
    method: Mapping[str, Any], boundary: Any
) -> Tuple[Any, Any, Any]:
    """Read the declared token boundary marker."""

    if isinstance(boundary, dict):
        return boundary.get("field"), boundary.get("event_field"), boundary.get("event_value")
    return method.get("token_boundary_field"), None, None


def _token_boundary_value(
    event: Mapping[str, Any], field: str, event_field: Any, event_value: Any
) -> Optional[int]:
    """Read one integer boundary marker while ignoring server timestamps."""

    if event_field is not None and event.get(event_field) != event_value:
        return None
    telemetry = event.get("leone_telemetry")
    source = telemetry if isinstance(telemetry, dict) else event
    if field not in source:
        return None
    value = source[field]
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else -1


def _valid_token_indexes(indexes: Sequence[int], usage: Optional[Mapping[str, Any]]) -> bool:
    """Require a zero-based contiguous token sequence and matching usage."""

    if not indexes:
        return False
    if indexes[0] != 0 or any(value < 0 for value in indexes):
        return False
    if any(right != left + 1 for left, right in zip(indexes, indexes[1:])):
        return False
    if not isinstance(usage, Mapping):
        return False
    count = usage.get("completion_tokens")
    return isinstance(count, int) and not isinstance(count, bool) and count == len(indexes)


def _engine_timing(
    events: Sequence[Tuple[Mapping[str, Any], int]], method: Mapping[str, Any]
) -> Dict[str, Any]:
    """Record engine timestamps with their declared clock without mixing domains."""

    timing = method.get("engine_timing")
    if not isinstance(timing, dict):
        return {"status": "unavailable", "reason": "engine_clock_undeclared"}
    field, clock = timing.get("field"), timing.get("clock")
    if not isinstance(field, str) or not isinstance(clock, str) or not clock:
        return {"status": "unavailable", "reason": "engine_clock_declaration_incomplete"}
    values = _engine_timing_values(events, field)
    if not values:
        return {"status": "unavailable", "reason": "engine_timestamps_unreported", "clock": clock}
    if not _monotonic(values):
        return {"status": "unavailable", "reason": "engine_timestamps_not_monotonic", "clock": clock}
    return {"status": "observed", "clock": clock, "field": field, "values": values}


def _engine_timing_values(
    events: Sequence[Tuple[Mapping[str, Any], int]], field: str
) -> List[float]:
    """Collect finite values from one declared telemetry field."""

    return [
        float(value)
        for event, _ in events
        for telemetry in [event.get("leone_telemetry")]
        for value in [telemetry.get(field) if isinstance(telemetry, Mapping) else None]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]


def _monotonic(values: Sequence[float]) -> bool:
    """Return whether values are nondecreasing."""

    return all(right >= left for left, right in zip(values, values[1:]))


def _cache_reuse(
    events: Sequence[Tuple[Mapping[str, Any], int]], engine: Mapping[str, Any]
) -> Dict[str, Any]:
    """Record a declared prompt-reuse count without inferring it from latency."""

    method = engine.get("cache_method")
    branch_method = engine.get("branch_method")
    field = method.get("reuse_count_field") if isinstance(method, dict) else None
    paths = _reuse_count_paths(method, branch_method)
    if not isinstance(field, str) and not paths:
        return {"status": "unavailable", "reason": "reuse_count_undeclared"}
    values = []
    for event, _ in events:
        telemetry = event.get("leone_telemetry")
        value = _reuse_value(event, telemetry, field, paths)
        if value is not None:
            values.append(value)
    path = paths[0] if paths else None
    if not values:
        return {
            "status": "unavailable",
            "reason": "reuse_count_unreported",
            "field": field,
            "path": path,
            "paths": paths,
        }
    return {"status": "observed", "field": field, "path": path, "paths": paths, "values": values}


def _reuse_count_paths(method: Any, branch_method: Any) -> List[List[str]]:
    """Collect declared nested reuse paths from cache and fork methods."""

    paths = []
    for declaration in (method, branch_method):
        path = declaration.get("reuse_count_path") if isinstance(declaration, dict) else None
        if isinstance(path, list) and all(isinstance(part, str) for part in path):
            paths.append(path)
    return paths


def _reuse_value(event: Mapping[str, Any], telemetry: Any, field: Any, paths: Sequence[Sequence[str]]) -> Optional[int]:
    """Read one declared nonnegative reuse count from an event."""

    candidates = []
    if isinstance(field, str):
        candidates.extend(
            source.get(field)
            for source in (telemetry, event)
            if isinstance(source, dict) and field in source
        )
    candidates.extend(_path_value(event, path) for path in paths)
    return next(
        (value for value in candidates if isinstance(value, int) and not isinstance(value, bool) and value >= 0),
        None,
    )


def _path_value(value: Any, path: Any) -> Any:
    """Read one declared nested response field."""

    current = value
    if not isinstance(path, list):
        return None
    for part in path:
        if not isinstance(part, str) or not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _service_metrics(events: Sequence[Tuple[Mapping[str, Any], int]]) -> Dict[str, Any]:
    """Retain optional server metrics with an explicit missing state."""

    telemetry = [
        event.get("leone_telemetry")
        for event, _ in events
        if isinstance(event.get("leone_telemetry"), dict)
    ]
    physical_samples = [
        _measurement(item.get("physical_bytes_by_class"), "physical_bytes_unreported")
        for item in telemetry
        if "physical_bytes_by_class" in item
    ]
    scheduler_samples = [
        _measurement(item.get("scheduler_reservation_bytes"), "scheduler_reservation_unreported")
        for item in telemetry
        if "scheduler_reservation_bytes" in item
    ]
    return {
        "fork_latency_ns": _latest_measurement(telemetry, "fork_latency_ns", "fork_latency_unreported"),
        "cancel_latency_ns": _latest_measurement(telemetry, "cancel_latency_ns", "cancel_latency_unreported"),
        "slow_client": _latest_measurement(telemetry, "slow_client", "slow_client_unreported"),
        "physical_bytes_by_class": _measurement(
            _latest_value(telemetry, "physical_bytes_by_class"), "physical_bytes_unreported"
        ),
        "scheduler_reservation_bytes": _measurement(
            _latest_value(telemetry, "scheduler_reservation_bytes"), "scheduler_reservation_unreported"
        ),
        "process_memory": _latest_measurement(telemetry, "process_memory", "process_memory_unreported"),
        "system_memory": _latest_measurement(telemetry, "system_memory", "system_memory_unreported"),
        "physical_bytes_samples": physical_samples,
        "scheduler_reservation_samples": scheduler_samples,
    }


def _latest_measurement(telemetry: Sequence[Mapping[str, Any]], field: str, reason: str) -> Dict[str, Any]:
    """Return the latest declared telemetry field as an explicit measurement."""

    return _measurement(_latest_value(telemetry, field), reason)


def _latest_value(telemetry: Sequence[Mapping[str, Any]], field: str) -> Any:
    """Find the latest event that declares one telemetry field."""

    for item in reversed(telemetry):
        if field in item:
            return item[field]
    return None


def _read_stream_events(
    response: http.client.HTTPResponse,
    read_delay_s: float = 0.0,
    cancel_after_first_content: bool = False,
    max_events: int = MAX_STREAM_EVENTS,
    max_content_bytes: int = MAX_CONTENT_BYTES,
    read_bytes: int = STREAM_READ_BYTES,
    deadline_ns: Optional[int] = None,
    max_retained_bytes: int = MAX_RETAINED_BYTES,
) -> Tuple[List[Tuple[Dict[str, Any], int]], bool]:
    """Read SSE events with available-byte reads and client receive times."""

    if max_events <= 0 or max_content_bytes <= 0 or read_bytes <= 0 or max_retained_bytes <= 0:
        raise StudyError("stream bounds must be positive")
    read_available = getattr(response, "read1", None)
    read_chunk = read_available if callable(read_available) else response.read
    return _read_stream_loop(
        read_chunk, read_delay_s, cancel_after_first_content, max_events,
        max_content_bytes, read_bytes, deadline_ns, max_retained_bytes,
    )


def _read_stream_loop(
    read_chunk: Any, read_delay_s: float, cancel_after_first_content: bool,
    max_events: int, max_content_bytes: int, read_bytes: int, deadline_ns: Optional[int],
    max_retained_bytes: int,
) -> Tuple[List[Tuple[Dict[str, Any], int]], bool]:
    """Read and bound one SSE stream from an incremental reader."""

    parser = SseParser()
    events: List[Tuple[Dict[str, Any], int]] = []
    terminal_seen = False
    content_total = 0
    retained_total = 0
    while True:
        _check_stream_deadline(deadline_ns)
        chunk = read_chunk(read_bytes)
        if not chunk:
            break
        received_ns = time.monotonic_ns()
        decoded, saw_done = _decode_sse_events(parser.feed(chunk, received_ns))
        content_total += _content_size(decoded)
        retained_total += _retained_size(decoded)
        _check_stream_limits(
            events, decoded, max_events, max_content_bytes, max_retained_bytes,
            content_total, retained_total,
        )
        events.extend(decoded)
        terminal_seen = terminal_seen or saw_done
        if cancel_after_first_content and _has_content(decoded):
            return events, False
        _delay_stream_read(read_delay_s)
    parser.finish()
    return events, terminal_seen


def _check_stream_deadline(deadline_ns: Optional[int]) -> None:
    """Reject a stream that exceeds its monotonic wall-time bound."""

    if deadline_ns is not None and time.monotonic_ns() >= deadline_ns:
        raise StudyError("study wall-time bound exceeded")


def _check_stream_limits(
    events: Sequence[Tuple[Mapping[str, Any], int]], decoded: Sequence[Tuple[Mapping[str, Any], int]],
    max_events: int, max_content_bytes: int, max_retained_bytes: int,
    content_total: int, retained_total: int,
) -> None:
    """Reject event or content histories beyond their configured bounds."""

    if len(events) + len(decoded) > max_events:
        raise StudyError("SSE event history exceeds the configured bound")
    if content_total > max_content_bytes:
        raise StudyError("SSE content history exceeds the configured bound")
    if retained_total > max_retained_bytes:
        raise StudyError("SSE retained event history exceeds the configured byte bound")


def _has_content(events: Sequence[Tuple[Mapping[str, Any], int]]) -> bool:
    """Return whether decoded events contain assistant content."""

    return any(_event_has_content(value) for value, _ in events)


def _delay_stream_read(delay_s: float) -> None:
    """Apply the configured slow-reader delay."""

    if delay_s > 0:
        time.sleep(delay_s)


def _content_size(events: Sequence[Tuple[Mapping[str, Any], int]]) -> int:
    """Return UTF-8 bytes retained in content events."""

    return sum(len(_event_content(event).encode("utf-8")) for event, _ in events)


def _retained_size(events: Sequence[Tuple[Mapping[str, Any], int]]) -> int:
    """Return canonical JSON bytes retained for decoded stream events."""

    return sum(len(canonical_json(event)) for event, _ in events)


def _decode_sse_events(payloads: Sequence[Tuple[str, int]]) -> Tuple[List[Tuple[Dict[str, Any], int]], bool]:
    """Decode one available SSE chunk and retain its client receipt time."""

    events: List[Tuple[Dict[str, Any], int]] = []
    terminal_seen = False
    for payload, event_ns in payloads:
        if payload == "[DONE]":
            terminal_seen = True
            continue
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise StudyError("SSE data is not JSON") from error
        if not isinstance(value, dict):
            raise StudyError("SSE data must be an object")
        events.append((value, event_ns))
    return events, terminal_seen


def _cancel_acknowledgement(
    engine: Mapping[str, Any], request_id: Optional[str], timeout_s: float
) -> Dict[str, Any]:
    """Read explicit cancellation acknowledgement from a declared endpoint."""

    url, error = _cancel_ack_url(engine, request_id)
    if url is None:
        return error
    try:
        status, _, body = _request_json(url, "GET", None, timeout_s)
    except (OSError, http.client.HTTPException) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    if status < 200 or status >= 300:
        return {"status": "unavailable", "reason": "cancel_acknowledgement_http_error", "http_status": status}
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {"status": "unavailable", "reason": "cancel_acknowledgement_not_json"}
    return _cancel_ack_response(value, request_id)


def _cancel_ack_url(
    engine: Mapping[str, Any], request_id: Optional[str]
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Resolve the declared cancellation observation endpoint."""

    endpoint = engine.get("cancel_observation_endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        return None, {"status": "unavailable", "reason": "cancel_acknowledgement_endpoint_missing"}
    if not isinstance(request_id, str) or not request_id:
        return None, {"status": "unavailable", "reason": "cancel_request_id_missing"}
    path = endpoint.replace("{request_id}", urllib.parse.quote(request_id, safe=""))
    return str(engine["base_url"]).rstrip("/") + "/" + path.lstrip("/"), {}


def _cancel_ack_response(value: Any, request_id: str) -> Dict[str, Any]:
    """Interpret one cancellation observation response."""

    if not isinstance(value, dict):
        return {"status": "unavailable", "reason": "cancel_acknowledgement_not_object"}
    acknowledged = (
        value.get("request_id") == request_id
        and value.get("outcome") == "cancelled"
        and value.get("reclaimed") is True
    )
    return {
        "status": "observed" if acknowledged else "unavailable",
        "outcome": value.get("outcome"),
        "reclaimed": value.get("reclaimed"),
        "request_id": value.get("request_id"),
        "reason": None if acknowledged else "server_did_not_report_matching_cancelled_and_reclaimed",
    }


def stream_request(
    engine: Mapping[str, Any],
    request: Mapping[str, Any],
    branch_fields: Mapping[str, Any],
    timeout_s: float,
    *,
    cancel_after_first_content: bool = False,
    read_delay_s: float = 0.0,
    max_events: int = MAX_STREAM_EVENTS,
    max_content_bytes: int = MAX_CONTENT_BYTES,
    read_bytes: int = STREAM_READ_BYTES,
    deadline_ns: Optional[int] = None,
    max_retained_bytes: int = MAX_RETAINED_BYTES,
) -> Dict[str, Any]:
    """Send one streaming request and retain every terminal outcome."""

    started_ns = time.monotonic_ns()
    encoded = canonical_json({**request, **branch_fields})
    result = _stream_initial_result(started_ns, encoded)
    host, port, path = _connection(str(engine["base_url"]))
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request(
            "POST",
            path + "/v1/chat/completions",
            body=encoded,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        result["http_status"] = response.status
        result["response_headers"] = {
            key.lower(): value for key, value in response.getheaders() if key.lower() in {"x-leone-session", "x-request-id"}
        }
        header_request_id = _service_request_header_id(result["response_headers"])
        if response.status != 200:
            return _stream_http_error(result, response)
        return _stream_response(
            result, response, engine, timeout_s, started_ns, header_request_id,
            cancel_after_first_content, read_delay_s, max_events, max_content_bytes,
            read_bytes, deadline_ns, max_retained_bytes,
        )
    except (OSError, http.client.HTTPException, UnicodeError) as error:
        result["status"] = "connection_error"
        result["error"] = f"{type(error).__name__}: {error}"
        result["metrics"] = latency_metrics(started_ns, [], False)
        return result
    except StudyError as error:
        result["status"] = _stream_error_status(error)
        result["error"] = str(error)
        result["metrics"] = latency_metrics(started_ns, [], False)
        result["history_complete"] = False
        return result
    finally:
        result["request_end_ns"] = time.monotonic_ns()
        connection.close()


def _nonstream_initial_result(started_ns: int, encoded: bytes) -> Dict[str, Any]:
    """Create the explicit missing-observation state for the non-stream parent."""

    unmeasured = {"status": "unavailable", "reason": "nonstream_parent_unmeasured"}
    return {
        "status": "connection_error",
        "request_body_sha256": sha256_bytes(encoded),
        "_request_bytes_hex": encoded.hex(),
        "metrics": dict(unmeasured),
        "engine_timing": dict(unmeasured),
        "cache_reuse": dict(unmeasured),
        "cancel_ack": {"status": "unavailable", "reason": "not_a_cancel_request"},
        "history_complete": True,
        "request_start_ns": started_ns,
        "service_request_id": None,
        "first_content_ns": None,
        "token_boundary_complete": False,
    }


def _nonstream_event(result: Dict[str, Any], body: bytes) -> Optional[Mapping[str, Any]]:
    """Parse one non-stream body, or record why it is not a response object."""

    try:
        event = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        result.update({"status": "execution_error", "error": f"invalid non-stream response: {error}"})
        return None
    if not isinstance(event, Mapping):
        result.update({"status": "execution_error", "error": "non-stream response is not an object"})
        return None
    return dict(event)


def _nonstream_completed(result: Dict[str, Any], event: Mapping[str, Any], content: str) -> Dict[str, Any]:
    """Record one completed non-stream response with its exact raw event."""

    response_id = event.get("id") if isinstance(event.get("id"), str) else None
    header_id = _service_request_header_id(result["response_headers"])
    receipt = event.get("leone_receipt") if isinstance(event.get("leone_receipt"), Mapping) else None
    result.update(
        {
            "status": "completed",
            "_raw_events": [{"event": dict(event), "received_ns": time.monotonic_ns()}],
            "_content_text": content,
            "content_sha256": sha256_bytes(content.encode()),
            "content_bytes": len(content.encode()),
            "usage": dict(event["usage"]) if isinstance(event.get("usage"), Mapping) else None,
            "finish_reason": _finish_reason([event]),
            "response_id": response_id,
            "service_request_id": header_id or response_id,
            "response_receipt": receipt,
            "_response_receipt_sha256": sha256_bytes(canonical_json(receipt)) if receipt is not None else None,
        }
    )
    return result


def _nonstream_body_result(
    result: Dict[str, Any], response: http.client.HTTPResponse, max_content_bytes: int
) -> Dict[str, Any]:
    """Read one bounded non-stream response into the retained result."""

    result["http_status"] = response.status
    result["response_headers"] = {
        key.lower(): value
        for key, value in response.getheaders()
        if key.lower() in {"x-leone-session", "x-request-id"}
    }
    body = response.read(max_content_bytes + 1)
    if len(body) > max_content_bytes:
        result.update({"status": "stream_limit_exceeded", "error": "non-stream response exceeds byte bound"})
        return result
    if response.status != 200:
        result.update({"status": "http_error", "error": body.decode("utf-8", "replace")})
        return result
    event = _nonstream_event(result, body)
    if event is None:
        return result
    content = _nonstream_content(event)
    if content is None:
        result.update({"status": "execution_error", "error": "non-stream response has no assistant content"})
        return result
    return _nonstream_completed(result, event, content)


def nonstream_request(
    engine: Mapping[str, Any],
    request: Mapping[str, Any],
    branch_fields: Mapping[str, Any],
    timeout_s: float,
    *,
    max_content_bytes: int = MAX_CONTENT_BYTES,
) -> Dict[str, Any]:
    """Send one non-streaming parent request without token timing claims."""

    started_ns = time.monotonic_ns()
    encoded = canonical_json({**request, **branch_fields})
    result = _nonstream_initial_result(started_ns, encoded)
    host, port, path = _connection(str(engine["base_url"]))
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request(
            "POST",
            path + "/v1/chat/completions",
            body=encoded,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        return _nonstream_body_result(result, connection.getresponse(), max_content_bytes)
    except (OSError, http.client.HTTPException) as error:
        result.update({"status": "connection_error", "error": f"{type(error).__name__}: {error}"})
        return result
    finally:
        result["request_end_ns"] = time.monotonic_ns()
        connection.close()


def _finish_reason(events: Sequence[Mapping[str, Any]]) -> Optional[str]:
    """Return the last finish reason any chat completion event carries."""

    reason = None
    for event in events:
        choices = event.get("choices") if isinstance(event, Mapping) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        value = first.get("finish_reason") if isinstance(first, Mapping) else None
        reason = value if isinstance(value, str) else reason
    return reason


def _nonstream_content(event: Mapping[str, Any]) -> Optional[str]:
    """Read the assistant message from one non-stream response."""

    choices = event.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    return content if isinstance(content, str) and content else None


def _service_request_header_id(headers: Mapping[str, Any]) -> Optional[str]:
    """Return the server request identity carried by response headers."""

    for field in ("x-request-id", "x-leone-session"):
        value = headers.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _stream_initial_result(started_ns: int, encoded: bytes) -> Dict[str, Any]:
    """Create the explicit missing-observation state for one request."""

    return {
        "status": "connection_error", "request_body_sha256": sha256_bytes(encoded),
        "_request_bytes_hex": encoded.hex(),
        "metrics": latency_metrics(started_ns, [], False), "service_metrics": _service_metrics([]),
        "engine_timing": {"status": "unavailable", "reason": "no_stream_events"},
        "cache_reuse": {"status": "unavailable", "reason": "no_stream_events"},
        "cancel_ack": {"status": "unavailable", "reason": "not_a_cancel_request"},
        "history_complete": True, "request_start_ns": started_ns, "service_request_id": None,
    }


def _stream_http_error(result: Dict[str, Any], response: http.client.HTTPResponse) -> Dict[str, Any]:
    """Retain an explicit non-success HTTP response."""

    result["status"] = "http_error"
    result["error"] = response.read(DEFAULT_READ_BYTES).decode("utf-8", "replace")
    return result


def _stream_response(
    result: Dict[str, Any], response: http.client.HTTPResponse, engine: Mapping[str, Any],
    timeout_s: float, started_ns: int, header_request_id: Optional[str], cancel_after_first_content: bool,
    read_delay_s: float, max_events: int, max_content_bytes: int, read_bytes: int, deadline_ns: Optional[int],
    max_retained_bytes: int,
) -> Dict[str, Any]:
    """Read one successful response and derive all request evidence."""

    events, terminal_seen = _read_stream_events(
        response, read_delay_s, cancel_after_first_content, max_events,
        max_content_bytes, read_bytes, deadline_ns, max_retained_bytes,
    )
    result["_raw_events"] = [
        {"event": event, "received_ns": received_ns} for event, received_ns in events
    ]
    content, event_times, usage, response_id = _content_and_timestamps(events)
    timing = _stream_timing(engine, started_ns, events, event_times, usage)
    result.update(_stream_observations(engine, events, event_times, content, usage, response_id, timing, header_request_id))
    if cancel_after_first_content and event_times:
        return _stream_cancel_result(result, response, engine, timeout_s, response_id)
    if not terminal_seen:
        result.update({"status": "incomplete_stream", "error": "SSE stream has no [DONE] event"})
    else:
        result["status"] = "completed"
    return result


def _stream_timing(
    engine: Mapping[str, Any], started_ns: int, events: Sequence[Tuple[Mapping[str, Any], int]],
    event_times: Sequence[int], usage: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Derive client token timing with verified index provenance."""

    token_times, verified = _token_timing(events, engine["branch_method"], usage)
    return latency_metrics(
        started_ns, event_times, verified, token_boundary_ns=token_times,
        token_boundary_indexes=_token_indexes(events, engine["branch_method"], usage),
    )


def _stream_observations(
    engine: Mapping[str, Any], events: Sequence[Tuple[Mapping[str, Any], int]], event_times: Sequence[int], content: str,
    usage: Optional[Mapping[str, Any]], response_id: Optional[str], timing: Mapping[str, Any],
    header_request_id: Optional[str],
) -> Dict[str, Any]:
    """Build response observations from parsed stream events."""

    receipt = _response_receipt(events)
    return {
        "metrics": timing, "first_content_ns": event_times[0] if event_times else None,
        "token_boundary_complete": timing.get("token_boundaries_verified"),
        "engine_timing": _engine_timing(events, engine["branch_method"]),
        "cache_reuse": _cache_reuse(events, engine), "service_metrics": _service_metrics(events),
        "content_sha256": sha256_bytes(content.encode()), "content_bytes": len(content.encode()),
        "usage": usage, "finish_reason": _finish_reason([event for event, _ in events]),
        "response_id": response_id, "service_request_id": header_request_id or response_id,
        "response_receipt": receipt,
        "_response_receipt_sha256": (
            sha256_bytes(canonical_json(receipt)) if isinstance(receipt, Mapping) else None
        ),
        "_content_text": content,
    }


def _response_receipt(events: Sequence[Tuple[Mapping[str, Any], int]]) -> Optional[Mapping[str, Any]]:
    """Return the final signed response receipt retained in one stream."""

    for event, _ in reversed(events):
        receipt = event.get("leone_receipt") if isinstance(event, Mapping) else None
        if isinstance(receipt, Mapping):
            return dict(receipt)
    return None


def _stream_cancel_result(
    result: Dict[str, Any], response: http.client.HTTPResponse, engine: Mapping[str, Any],
    timeout_s: float, response_id: Optional[str],
) -> Dict[str, Any]:
    """Close a first-content stream and record explicit cancellation."""

    result["cancel_requested_ns"] = time.monotonic_ns()
    response_headers = result.get("response_headers", {})
    request_id = _service_request_header_id(response_headers) or response_id
    response.close()
    result["cancel_ack"] = _cancel_acknowledgement(engine, request_id, timeout_s)
    result["status"] = "cancelled" if result["cancel_ack"].get("status") == "observed" else "cancel_acknowledgement_unavailable"
    result["cancel_reason"] = "client_cancelled_after_first_content"
    return result


def _stream_error_status(error: StudyError) -> str:
    """Map bounded stream failures to explicit terminal outcomes."""

    message = str(error).lower()
    if "wall-time" in message:
        return "deadline_expired"
    if "bound" in message or "history" in message:
        return "stream_limit_exceeded"
    return "connection_error"


_DEFAULT_SESSION_FIELD = object()


def _prompt_messages(prompt: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return the declared message sequence for one prompt."""

    messages = prompt.get("messages")
    if isinstance(messages, list):
        return [dict(message) for message in messages]
    text = prompt.get("text")
    if isinstance(text, str):
        return [{"role": "user", "content": text}]
    raise StudyError(f"prompt {prompt.get('id', '<unknown>')} has no messages")


def request_body(
    manifest: Mapping[str, Any],
    prompt_or_messages: Any,
    session_id: str,
    session_field: Any = _DEFAULT_SESSION_FIELD,
    *,
    stream: bool = True,
) -> Dict[str, Any]:
    """Build common request fields and the exact declared message history."""

    request = manifest["request"]
    messages = (
        [{"role": "user", "content": prompt_or_messages}]
        if isinstance(prompt_or_messages, str)
        else [dict(message) for message in prompt_or_messages]
    )
    body: Dict[str, Any] = {
        "model": request["model"],
        "messages": messages,
        "max_tokens": request["max_tokens"],
        "temperature": request["temperature"],
        "seed": request["seed"],
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    if session_field is _DEFAULT_SESSION_FIELD:
        session_field = request.get("session_field")
    if isinstance(session_field, str) and session_field:
        body[session_field] = session_id
    return body


def _engine_request_body(
    engine: Mapping[str, Any], body: Mapping[str, Any], slot: Optional[int] = None, proof: bool = False
) -> Dict[str, Any]:
    """Add protocol controls required by the declared endpoint.

    Only the history proof requests (`proof=True`: parent and branches) ask for
    verbose output. It adds payload and server work that the timed probes do not need.
    """

    result = dict(body)
    if not _is_llama_cpp(engine):
        return result
    result.update({"cache_prompt": True, "id_slot": _checked_llama_slot(engine, slot)})
    if proof:
        result["verbose"] = True
    result.update(_thinking_request_fields(engine))
    if result.get("stream") is False:
        result["return_tokens"] = True
    return result


def _checked_llama_slot(engine: Mapping[str, Any], slot: Optional[int]) -> int:
    """Select the request slot and require it inside the declared slot count."""

    selected = engine.get("id_slot") if slot is None else slot
    if not _valid_slot(selected):
        raise StudyError("llama.cpp engine requires a nonnegative id_slot")
    slot_count = engine.get("slot_count")
    if not isinstance(slot_count, int) or isinstance(slot_count, bool) or not 0 <= selected < slot_count:
        raise StudyError("llama.cpp request slot exceeds declared slot_count")
    return selected


def _thinking_request_fields(engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the request fields that pin how the server treats reasoning text."""

    policy = engine.get("thinking_policy")
    if not isinstance(policy, Mapping):
        return {}
    fields: Dict[str, Any] = {"reasoning_format": policy["reasoning_format"]}
    if policy.get("chat_template_kwargs"):
        fields["chat_template_kwargs"] = dict(policy["chat_template_kwargs"])
    return fields


def _engine_schedule_slot(
    engine: Mapping[str, Any], role: str, ordinal: int = 0, branch_count: int = 0
) -> Optional[int]:
    """Select a slot that keeps timed independent requests concurrent."""

    base = engine.get("id_slot")
    if engine.get("kind", engine.get("protocol")) not in {"llama.cpp", "llama_cpp"}:
        return None
    if not _valid_slot(base):
        raise StudyError("llama.cpp engine requires a nonnegative id_slot")
    if not isinstance(ordinal, int) or ordinal < 0:
        raise StudyError("llama.cpp slot ordinal must be nonnegative")
    if not isinstance(branch_count, int) or branch_count < 0:
        raise StudyError("llama.cpp branch count must be nonnegative")
    if role == "branch":
        return base + 1 + ordinal
    offsets = {
        "parent": 0,
        "new_prompt": branch_count + 1,
        "cancel": branch_count + 2,
        "slow_reader": branch_count + 3,
    }
    return base + offsets.get(role, 0)


def _valid_slot(value: Any) -> bool:
    """Return whether one llama.cpp slot identifier is valid."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def engine_kind(engine: Mapping[str, Any]) -> Optional[str]:
    """Return the canonical engine kind, folding the `llama_cpp` spelling."""

    kind = engine.get("kind", engine.get("protocol"))
    if kind in {"llama.cpp", "llama_cpp"}:
        return "llama.cpp"
    return kind if isinstance(kind, str) else None


def _is_llama_cpp(engine: Mapping[str, Any]) -> bool:
    """Return whether the engine declares the llama.cpp protocol."""

    return engine_kind(engine) == "llama.cpp"


def capability_record(kind: Optional[str]) -> Optional[Dict[str, Dict[str, str]]]:
    """Return a copy of the fixed evidence contract for one engine kind."""

    contract = EVIDENCE_CAPABILITIES.get(kind) if isinstance(kind, str) else None
    return {metric: dict(entry) for metric, entry in contract.items()} if contract else None


def _unsupported_metrics(kind: Optional[str]) -> set:
    """List the metrics the fixed contract marks unsupported for one engine kind."""

    contract = EVIDENCE_CAPABILITIES.get(kind, {}) if isinstance(kind, str) else {}
    return {metric for metric, entry in contract.items() if entry["status"] == "unsupported"}


def _run_engine_kind(run: Mapping[str, Any], manifest: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Resolve a run's engine kind from the manifest, defaulting to Leone without one."""

    engines = manifest.get("engines") if isinstance(manifest, Mapping) else None
    if not isinstance(engines, list):
        return "leone"
    declared = next((e for e in engines if isinstance(e, Mapping) and e.get("id") == run.get("engine")), None)
    return engine_kind(declared) if isinstance(declared, Mapping) else None


def _slot_action(
    engine: Mapping[str, Any], slot: int, action: str, filename: str, timeout_s: float
) -> Dict[str, Any]:
    """Send one llama.cpp slot save or restore and retain the exchanged bytes."""

    payload = canonical_json({"filename": filename})
    host, port, path = _connection(str(engine["base_url"]))
    record: Dict[str, Any] = {
        "action": action,
        "slot": slot,
        "request_hex": payload.hex(),
        "http_status": None,
        "response_hex": "",
        "request_start_ns": time.monotonic_ns(),
    }
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request(
            "POST", f"{path}/slots/{slot}?action={action}", body=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        response = connection.getresponse()
        record["http_status"] = response.status
        record["response_hex"] = response.read(DEFAULT_READ_BYTES).hex()
    except (OSError, http.client.HTTPException) as error:
        record["error"] = type(error).__name__
    finally:
        connection.close()
        record["request_end_ns"] = time.monotonic_ns()
    return record


def _prepare_slot_copies(
    engine: Mapping[str, Any], parent_id: str, parent: Mapping[str, Any],
    tasks: Sequence[Tuple[Any, ...]], timeout_s: float,
) -> Optional[Dict[str, Any]]:
    """Copy the completed parent slot into every history request slot.

    The copies run in sequence before the start barrier, so the timed requests
    stay concurrent and start warm. A slot outside the parent slot is cold
    otherwise: llama.cpp reuses only the prompt cached in the slot it serves
    from. The save and restore exchanges stay outside every request interval.
    """

    if not _is_llama_cpp(engine) or parent.get("status") != "completed":
        return None
    parent_slot = _engine_schedule_slot(engine, "parent")
    targets = sorted({
        body["id_slot"] for role, _, body, *_ in tasks
        if role in {"branch", "cancel"} and body.get("id_slot") != parent_slot
    })
    if not targets:
        return None
    filename = f"{parent_id}.slot"
    save = _slot_action(engine, parent_slot, "save", filename, timeout_s)
    restores: Dict[str, Any] = {}
    if save.get("http_status") == 200:
        for slot in targets:
            restores[str(slot)] = _slot_action(engine, slot, "restore", filename, timeout_s)
    return {"parent_slot": parent_slot, "filename": filename, "save": save, "restore": restores}


def _record_request_slot(record: Mapping[str, Any]) -> Optional[int]:
    """Read the llama.cpp slot from the exact request bytes of one record."""

    raw = _record_request_bytes(record)
    try:
        body = json.loads(raw.decode("utf-8")) if raw is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    slot = body.get("id_slot") if isinstance(body, dict) else None
    return slot if _valid_slot(slot) else None


def _branch_slot_copy(engine: Mapping[str, Any], record: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Select the retained slot copy that preceded one branch request."""

    plan = engine.get("_slot_copy")
    slot = _record_request_slot(record)
    if not isinstance(plan, Mapping) or slot is None or slot == plan.get("parent_slot"):
        return None
    return {
        "save": plan.get("save"),
        "restore": plan.get("restore", {}).get(str(slot)),
        "branch_request_start_ns": record.get("request_start_ns"),
    }


def _history_messages(parent_messages: Sequence[Mapping[str, Any]], parent: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Append the observed parent assistant turn before a branch message."""

    messages = [dict(message) for message in parent_messages]
    content = parent.get("_content_text")
    if parent.get("status") == "completed" and isinstance(content, str) and content:
        messages.append({"role": "assistant", "content": content})
    return messages


def _history_material(
    prefix: Sequence[Mapping[str, Any]],
    branch_messages: Sequence[Mapping[str, Any]],
    body: Optional[Mapping[str, Any]],
    branch_prompt_id: Optional[str],
) -> Dict[str, Any]:
    """Retain the canonical request material used to claim prefix reuse."""

    request_messages = [dict(message) for message in prefix]
    request_messages.extend(dict(message) for message in branch_messages)
    material: Dict[str, Any] = {
        "branch_prompt_id": branch_prompt_id,
        "prefix_messages": [dict(message) for message in prefix],
        "request_messages": request_messages,
    }
    if isinstance(body, Mapping):
        material["request_body"] = dict(body)
    return material


def _reuse_count_verified(reuse_count: Any) -> bool:
    """Return whether the response reports a positive reuse count."""

    values = reuse_count.get("values", []) if isinstance(reuse_count, dict) else []
    return (
        isinstance(reuse_count, dict)
        and reuse_count.get("status") == "observed"
        and isinstance(values, list)
        and any(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values)
    )


def _history_check(
    parent_messages: Sequence[Mapping[str, Any]],
    branch_messages: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    response: Mapping[str, Any],
    body: Optional[Mapping[str, Any]] = None,
    branch_prompt_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Record structural history reuse and leave engine reuse explicit."""

    prefix = _history_messages(parent_messages, parent)
    material = _history_material(prefix, branch_messages, body, branch_prompt_id)
    reuse_count = response.get("cache_reuse", {"status": "unavailable", "reason": "reuse_count_unreported"})
    parent_available = parent.get("status") == "completed" and bool(parent.get("content_sha256"))
    reuse_verified = _reuse_count_verified(reuse_count)
    material_verified = isinstance(body, Mapping)
    return {
        "status": "observed" if parent_available and reuse_verified and material_verified else "unavailable",
        "reason": None if parent_available and reuse_verified and material_verified else "history_reuse_unverified",
        "prefix_verified": parent_available,
        "reuse_verified": reuse_verified,
        "prefix_message_count": len(prefix),
        "request_message_count": len(prefix) + len(branch_messages),
        "prefix_sha256": sha256_bytes(canonical_json(prefix)),
        "request_sha256": sha256_bytes(canonical_json(prefix + [dict(item) for item in branch_messages])),
        "parent_content_sha256": parent.get("content_sha256"),
        "request_body_sha256": response.get("request_body_sha256"),
        "reuse_count": reuse_count,
        "required_reuse_tokens": _history_required_reuse_tokens(parent, None),
        "request_material": material,
        "tokenization": response.get("history_tokenization", {
            "status": "unavailable", "reason": "canonical_prefix_tokenization_missing"
        }),
    }


_HISTORY_PRODUCER_MODULE: Dict[str, Any] = {}
_SIBLING_MODULES: Dict[str, Any] = {}


def _load_history_producer_module() -> Any:
    """Load the pinned CPU history producer module beside this harness."""

    if "module" in _HISTORY_PRODUCER_MODULE:
        return _HISTORY_PRODUCER_MODULE["module"]
    directory = Path(__file__).resolve().parent
    path = directory / "produce-history-tokenization.py"
    if not path.is_file():
        raise StudyError(f"history tokenization producer is missing: {path}")
    spec = importlib.util.spec_from_file_location("leone_history_tokenizer", path)
    if spec is None or spec.loader is None:
        raise StudyError("history tokenization producer cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HISTORY_PRODUCER_MODULE["module"] = module
    return module


def _load_history_tokenization_producer() -> Callable[..., Mapping[str, Any]]:
    """Load the pinned CPU history producer entrypoint."""

    module = _load_history_producer_module()
    producer = getattr(module, "produce_history_tokenization", None)
    if not callable(producer):
        raise StudyError("history tokenization producer entrypoint is missing")
    return producer


def _producer_engine(engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge the declared `history_tokenization` inputs into the engine mapping."""

    declaration = engine.get("history_tokenization")
    merged = dict(engine)
    if isinstance(declaration, Mapping):
        merged.update(declaration)
    merged.update(engine.get("_producer_paths") or {})
    policy = engine.get("thinking_policy")
    if isinstance(policy, Mapping):
        merged["special_tokens_policy"] = {**(merged.get("special_tokens_policy") or {}), **_thinking_policy_fields(policy)}
    return merged


def _producer_paths(root: Path, engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve the independent tokenizer inputs against the study root."""

    declaration = engine.get("history_tokenization")
    if not isinstance(declaration, Mapping):
        return {}
    paths: Dict[str, str] = {}
    for destination, field in (
        ("llama_server", "producer_executable_path"),
        ("model", "producer_model_artifact"),
        ("template_file", "producer_template_file"),
    ):
        value = declaration.get(field)
        if isinstance(value, str):
            paths[destination] = str(root_path(root, value))
    metadata = {
        "engine": declaration.get("producer_engine"),
        "backend": declaration.get("producer_backend", engine.get("backend")),
        "source_commit": declaration.get("producer_source_commit"),
        "llama_cpp_commit": declaration.get("producer_source_commit"),
        "executable_sha256": declaration.get("producer_executable_sha256"),
        "model_sha256": declaration.get("producer_model_sha256"),
        "loaded_library_sha256": declaration.get("producer_loaded_library_sha256"),
        "vocab_size": declaration.get("vocab_size"),
    }
    paths.update({name: value for name, value in metadata.items() if value is not None})
    if not _is_llama_cpp(engine):
        source = declaration.get("trusted_public_key_source")
        if not isinstance(source, str) or not source:
            raise StudyError(f"{engine.get('id')}: history tokenizer trusted key source is missing")
        paths["trusted_public_key_ed25519"] = _trusted_public_key(root, source)
    return paths


def _study_producer_paths(root: Path, engines: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Resolve and check tokenizer inputs before starting a serving process."""

    resolved = {}
    for engine in engines:
        paths = _producer_paths(root, engine)
        missing = [
            name for name in ("llama_server", "model", "template_file")
            if not isinstance(paths.get(name), str) or not Path(paths[name]).is_file()
        ]
        if missing:
            raise StudyError(
                f"{engine.get('id')}: history tokenizer inputs are missing: {', '.join(missing)}"
            )
        resolved[engine["id"]] = paths
    return resolved


def _trusted_public_key(root: Path, environment_name: str) -> str:
    """Derive the public key from the explicit run signing-key path."""

    key_value = os.environ.get(environment_name)
    if not isinstance(key_value, str) or not key_value:
        raise StudyError(f"history tokenizer signing key environment is missing: {environment_name}")
    path = Path(key_value)
    if not path.is_absolute():
        path = root / path
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise StudyError(f"history tokenizer signing key cannot be read: {error}") from error
    if len(raw) != 32:
        raise StudyError("history tokenizer signing key must contain exactly 32 bytes")
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        public = Ed25519PrivateKey.from_private_bytes(raw).public_key()
        derived = public.public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
        configured = os.environ.get("LEONE_BRANCHING_TRUSTED_PUBLIC_KEY")
        if configured is not None and configured.lower() != derived:
            raise StudyError("history tokenizer trusted public key differs from signing key")
        return derived
    except ImportError as error:
        raise StudyError("history tokenizer signing-key derivation needs cryptography") from error


def _thinking_policy_fields(policy: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the thinking policy fields that enter the special tokens policy."""

    return {
        "chat_template_kwargs": dict(policy.get("chat_template_kwargs") or {}),
        "generation_prompt_suffix": policy.get("generation_prompt_suffix"),
        "reasoning_format": policy.get("reasoning_format"),
    }


def _record_request_bytes(record: Mapping[str, Any]) -> Optional[bytes]:
    """Decode the exact request bytes retained by the streaming runner."""

    encoded = record.get("_request_bytes_hex")
    if not isinstance(encoded, str) or not encoded:
        return None
    try:
        value = bytes.fromhex(encoded)
    except ValueError:
        return None
    return value if value else None


def _record_raw_events(record: Mapping[str, Any]) -> Optional[List[Mapping[str, Any]]]:
    """Read the bounded raw event rows retained by the streaming runner."""

    events = record.get("_raw_events")
    return events if isinstance(events, list) and events else None


def _history_raw_events(
    record: Mapping[str, Any], engine: Mapping[str, Any], parent_request_id: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    """Attach measured request identity to retained wire events."""

    events = _record_raw_events(record)
    if events is None:
        return None
    wrapper: Dict[str, Any] = {
        "events": events,
        "service_request_id": record.get("service_request_id") or record.get("response_id"),
    }
    if parent_request_id is not None:
        wrapper["parent_service_request_id"] = parent_request_id
        slot_copy = _branch_slot_copy(engine, record)
        if slot_copy is not None:
            wrapper["slot_copy"] = slot_copy
    wrapper.update(_wrapper_identity(engine))
    return wrapper


def _wrapper_identity(engine: Mapping[str, Any]) -> Dict[str, Any]:
    """Read process, epoch, and source identity from the start identity or the end metrics."""

    identity_body = _history_identity_body(engine.get("_running_identity"))
    metrics = engine.get("_service_metrics")
    metrics_body = metrics.get("body") if isinstance(metrics, Mapping) else None
    found = {}
    for field in ("process_instance_id", "workload_epoch", "source_id"):
        value = next(
            (body.get(field) for body in (identity_body, metrics_body)
             if isinstance(body, Mapping) and body.get(field) is not None),
            None,
        )
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            found[field] = value
    return found


def _history_identity_body(value: Any) -> Optional[Mapping[str, Any]]:
    """Read the start identity retained for producer joins."""

    if not isinstance(value, Mapping):
        return None
    start = value.get("start") if isinstance(value.get("start"), Mapping) else value
    identity = start.get("identity") if isinstance(start, Mapping) else None
    return identity if isinstance(identity, Mapping) else None


def _history_evidence_projection(evidence: Any) -> Dict[str, Any]:
    """Require the producer result to match the frozen evidence shape."""

    if not isinstance(evidence, Mapping):
        return _unavailable_history_tokenization("history_tokenization_producer_invalid")
    if evidence.get("status") != "observed":
        return _unavailable_history_tokenization(str(evidence.get("reason", "producer_unavailable")))
    if any(field not in evidence for field in HISTORY_TOKEN_FIELDS):
        return _unavailable_history_tokenization("history_tokenization_evidence_incomplete")
    return {field: evidence[field] for field in sorted(HISTORY_TOKEN_FIELDS)}


def _unavailable_history_tokenization(reason: str) -> Dict[str, str]:
    """Represent missing independent history evidence explicitly."""

    return {"status": "unavailable", "reason": reason}


def _history_scope_explicit(engine: Mapping[str, Any], branch: Mapping[str, Any]) -> bool:
    """Return whether the branch has an explicit lineage the producer can check.

    A llama.cpp branch has no fork field. Its lineage is the retained slot copy
    from the parent slot, so a cache-only branch without one stays out of scope.
    """

    if branch.get("branch_mode") == "fork":
        return True
    return (
        branch.get("branch_mode") == "cached_history"
        and _is_llama_cpp(engine)
        and _branch_slot_copy(engine, branch) is not None
    )


def _attach_history_tokenization(
    engine: Mapping[str, Any], parent: Mapping[str, Any], branch: Dict[str, Any],
    producer: Optional[Callable[..., Mapping[str, Any]]],
) -> None:
    """Attach independently produced history evidence to one explicit fork."""

    history = branch.get("history_reuse")
    if not isinstance(history, dict):
        return
    if not _history_scope_explicit(engine, branch):
        history["tokenization"] = _unavailable_history_tokenization(
            "history_tokenization_scope_requires_explicit_fork"
        )
        return
    if producer is None:
        history["tokenization"] = _unavailable_history_tokenization(
            "history_tokenization_producer_missing"
        )
        return
    parent_id = parent.get("service_request_id") or parent.get("response_id") or parent.get("request_id")
    inputs = (
        _record_request_bytes(parent), _history_raw_events(parent, engine),
        _record_request_bytes(branch), _history_raw_events(branch, engine, parent_id),
    )
    if any(value is None for value in inputs):
        history["tokenization"] = _unavailable_history_tokenization(
            "history_tokenization_raw_inputs_missing"
        )
        return
    try:
        evidence = producer(engine, *inputs)
    except (OSError, StudyError, ValueError) as error:
        history["tokenization"] = _unavailable_history_tokenization(
            f"history_tokenization_unavailable:{type(error).__name__}"
        )
        return
    history["tokenization"] = _history_evidence_projection(evidence)


def _cache_method(engine: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return a declared same-history cache method."""

    method = engine.get("cache_method")
    return method if isinstance(method, dict) and method.get("status") == "supported" else None


def _branch_fields(
    engine: Mapping[str, Any],
    inspection: Mapping[str, Any],
    parent_id: str,
    branch_id: str,
) -> Tuple[str, Dict[str, Any]]:
    """Select a declared fork or cached-history request shape."""

    method = engine["branch_method"]
    if inspection.get("status") == "supported":
        return "fork", {
            str(method["parent_field"]): parent_id,
            str(method["session_field"]): branch_id,
        }
    cache = _cache_method(engine)
    if cache is not None:
        field = cache.get("request_field")
        fields = {str(field): cache.get("request_value", True)} if isinstance(field, str) else {}
        return "cached_history", fields
    return "unavailable", {}


def _failed_probe(status: str, reason: Any, role: str, request_id: str) -> Dict[str, Any]:
    """Represent a probe that could not be sent."""

    return {
        "role": role,
        "request_id": request_id,
        "status": status,
        "error": reason,
        "metrics": latency_metrics(0, [], False),
        "history_complete": False,
        "request_start_ns": 0,
        "request_end_ns": 0,
        "service_request_id": None,
    }


def _run_probe(
    engine: Mapping[str, Any],
    body: Mapping[str, Any],
    fields: Mapping[str, Any],
    timeout_s: float,
    role: str,
    request_id: str,
    cancel: bool = False,
    delay_s: float = 0.0,
    start_barrier: Optional[threading.Barrier] = None,
    max_events: int = MAX_STREAM_EVENTS,
    max_content_bytes: int = MAX_CONTENT_BYTES,
    read_bytes: int = STREAM_READ_BYTES,
    deadline_ns: Optional[int] = None,
    max_retained_bytes: int = MAX_RETAINED_BYTES,
) -> Dict[str, Any]:
    """Run one bounded schedule probe and attach its role."""

    try:
        if start_barrier is not None:
            barrier_timeout = None
            if deadline_ns is not None:
                barrier_timeout = max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)
            start_barrier.wait(timeout=barrier_timeout)
        result = stream_request(
            engine,
            body,
            fields,
            timeout_s,
            cancel_after_first_content=cancel,
            read_delay_s=delay_s,
            max_events=max_events,
            max_content_bytes=max_content_bytes,
            read_bytes=read_bytes,
            deadline_ns=deadline_ns,
            max_retained_bytes=max_retained_bytes,
        )
    except (OSError, StudyError, ValueError, threading.BrokenBarrierError) as error:
        result = _failed_probe("probe_error", f"{type(error).__name__}: {error}", role, request_id)
    result["role"] = role
    result["request_id"] = request_id
    if role == "slow_reader":
        result["backpressure"] = _backpressure_evidence(result)
    result.pop("_content_text", None)
    return result


def _schedule_overlap(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Report whether recorded request intervals overlap in client time."""

    valid = _schedule_intervals(records)
    if len(valid) < 2:
        return {"status": "unavailable", "reason": "request_interval_missing", "overlap": False}
    overlap = _intervals_overlap(valid)
    required = _required_schedule_overlap(valid, _declared_schedule_roles(records))
    reason, observed = _schedule_overlap_state(overlap, required)
    result = {
        "status": "observed" if observed else "unavailable",
        "reason": reason,
        "overlap": overlap,
        "roles": [item[2] for item in valid],
    }
    if required is not None:
        result["required_overlap"] = required
    return result


def _declared_schedule_roles(records: Sequence[Mapping[str, Any]]) -> set:
    """Collect schedule roles before invalid intervals are discarded."""

    return {item.get("role") for item in records if isinstance(item, Mapping)}


def _schedule_overlap_state(
    overlap: bool, required: Optional[bool]
) -> Tuple[Optional[str], bool]:
    """Choose the explicit status for one schedule overlap result."""

    if not overlap:
        return "request_intervals_do_not_overlap", False
    if required is False:
        return "required_schedule_overlap_missing", False
    return None, True


def _required_schedule_overlap(
    intervals: Sequence[Tuple[int, int, Any]], declared_roles: Optional[set] = None
) -> Optional[bool]:
    """Require every concurrent probe role to overlap a branch interval."""

    roles = {item[2] for item in intervals}
    required_roles = {"branch", "new_prompt", "cancel", "slow_reader"}
    if not required_roles.issubset(declared_roles or roles):
        return None
    if not required_roles.issubset(roles):
        return False
    branch_intervals = [item for item in intervals if item[2] == "branch"]
    return all(
        any(_interval_pair_overlaps(probe, branch) for branch in branch_intervals)
        for probe in intervals
        if probe[2] in required_roles - {"branch"}
    )


def _interval_pair_overlaps(
    left: Tuple[int, int, Any], right: Tuple[int, int, Any]
) -> bool:
    """Return whether two half-open schedule intervals overlap."""

    return left[0] < right[1] and right[0] < left[1]


def _schedule_intervals(records: Sequence[Mapping[str, Any]]) -> List[Tuple[int, int, Any]]:
    """Collect ordered client intervals from schedule records."""

    return [
        (item.get("request_start_ns"), item.get("request_end_ns"), item.get("role"))
        for item in records
        if _valid_request_bounds(item.get("request_start_ns"), item.get("request_end_ns"))
    ]


def _intervals_overlap(intervals: Sequence[Tuple[int, int, Any]]) -> bool:
    """Return whether any pair of half-open intervals overlaps."""

    return any(
        _interval_pair_overlaps(left, right)
        for index, left in enumerate(intervals)
        for right in intervals[index + 1 :]
    )


def _pressure_interval(
    value: Any, default_clock_domain: Optional[str] = None
) -> Optional[Tuple[int, int, str]]:
    """Read the measured pressure interval and its declared clock domain."""

    if not isinstance(value, Mapping):
        return None
    start = _pressure_boundary(value, "interval_start_ns", "start_ns")
    end = _pressure_boundary(value, "interval_end_ns", "end_ns")
    clock_domain = value.get("clock_domain") or default_clock_domain
    if not _valid_pressure_bounds(start, end):
        return None
    if not isinstance(clock_domain, str) or not clock_domain:
        return None
    return start, end, clock_domain


def _pressure_boundary(value: Mapping[str, Any], primary: str, alternate: str) -> Any:
    """Read a pressure boundary from either harness or transport spelling."""

    boundary = value.get(primary)
    return value.get(alternate) if boundary is None else boundary


def _valid_pressure_bounds(start: Any, end: Any) -> bool:
    """Check ordered nonnegative pressure boundaries."""

    return (
        isinstance(start, int)
        and not isinstance(start, bool)
        and isinstance(end, int)
        and not isinstance(end, bool)
        and start >= 0
        and end > start
    )


def _record_clock_domain(record: Mapping[str, Any]) -> Optional[str]:
    """Read the clock domain retained with one client record."""

    metrics = record.get("metrics")
    value = metrics.get("clock_domain") if isinstance(metrics, Mapping) else None
    return value if isinstance(value, str) and value else None


def _pressure_request_id(value: Mapping[str, Any]) -> Optional[str]:
    """Read the stable wire or numeric request identity from a pressure row."""

    for field in ("request_id", "wire_request_id", "numeric_request_id"):
        candidate = value.get(field)
        if isinstance(candidate, str) and candidate:
            return candidate
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
            return str(candidate)
    return None


def _pressure_matches_request(value: Mapping[str, Any], request_id: Optional[str]) -> bool:
    """Match either the wire or numeric identity emitted by the service."""

    if request_id is None:
        return True
    candidates = {
        str(value[field])
        for field in ("request_id", "wire_request_id", "numeric_request_id")
        if field in value and isinstance(value[field], (str, int)) and not isinstance(value[field], bool)
    }
    return request_id in candidates


def _sibling_progress(
    records: Sequence[Mapping[str, Any]], pressure: Optional[Mapping[str, Any]] = None,
    service_trace: Optional[Mapping[str, Any]] = None,
    service_metrics: Optional[Mapping[str, Any]] = None,
    running_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Report sibling content received during the measured pressure interval."""

    slow = next((item for item in records if item.get("role") == "slow_reader"), None)
    if not isinstance(slow, dict):
        return {"status": "unavailable", "reason": "slow_reader_record_missing"}
    interval = _sibling_interval(pressure)
    if interval is None:
        return {"status": "unavailable", "reason": "server_backpressure_interval_missing"}
    if service_trace is not None:
        return _trace_sibling_progress(records, pressure, service_trace, service_metrics, running_identity)
    return _record_sibling_progress(records, pressure, interval)


def _sibling_interval(pressure: Any) -> Optional[Tuple[int, int, str]]:
    """Read an observed pressure interval before selecting sibling records."""

    if not isinstance(pressure, Mapping) or pressure.get("status") != "observed":
        return None
    return _pressure_interval(pressure)


def _record_sibling_progress(
    records: Sequence[Mapping[str, Any]], pressure: Mapping[str, Any],
    interval: Tuple[int, int, str],
) -> Dict[str, Any]:
    """Select client records whose declared server timestamps fall in pressure."""

    siblings = _sibling_records(
        records, interval[0], interval[1], interval[2], pressure.get("workload_epoch")
    )
    return {
        "status": "observed" if siblings else "unavailable",
        "reason": None if siblings else "no_sibling_content_during_slow_reader",
        "roles": [item.get("role") for item in siblings],
    }


def _trace_sibling_progress(
    records: Sequence[Mapping[str, Any]], pressure: Mapping[str, Any],
    service_trace: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]],
    running_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Select sibling decode events from the server-clock trace."""

    body, events, error = _trace_payload(service_trace)
    if error is not None:
        return {"status": "unavailable", "reason": error}
    interval, error = _trace_pressure_context(pressure, service_metrics, body, running_identity)
    if error is not None or interval is None:
        return {"status": "unavailable", "reason": error or "server_trace_epoch_missing"}
    identity_body = _trace_identity_body(body, service_metrics)
    slow_id = _pressure_request_id(pressure)
    slow_numeric = _trace_numeric_request_id(identity_body, slow_id)
    if slow_numeric is None:
        return {"status": "unavailable", "reason": "server_trace_slow_request_missing"}
    return _trace_sibling_result(events, records, identity_body, slow_numeric, interval)


def _trace_payload(
    service_trace: Mapping[str, Any],
) -> Tuple[Optional[Mapping[str, Any]], Optional[Sequence[Mapping[str, Any]]], Optional[str]]:
    """Validate the trace wrapper and return its complete event list."""

    body = service_trace.get("body") if isinstance(service_trace, Mapping) else None
    if service_trace.get("status") != "observed" or not isinstance(body, Mapping):
        return None, None, "server_trace_missing"
    events = body.get("events")
    if body.get("schema_version") != "leone.service-trace.v1" or not isinstance(events, list):
        return None, None, "server_trace_invalid"
    if not _trace_events_valid(events) or not _trace_history_complete(body):
        return None, None, "server_trace_partial"
    return body, events, None


def _trace_pressure_context(
    pressure: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]],
    body: Mapping[str, Any], running_identity: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optional[Tuple[int, int, str]], Optional[str]]:
    """Require trace and pressure to share the selected server epoch and clock."""

    interval, error = _trace_pressure_interval(pressure, service_metrics)
    if error is not None:
        return None, error
    error = _trace_pressure_identity_error(body, service_metrics, running_identity)
    if error is not None:
        return None, error
    if body.get("clock_domain") != interval[2]:
        return None, "server_trace_clock_mismatch"
    return interval, None


def _trace_pressure_interval(
    pressure: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]]
) -> Tuple[Optional[Tuple[int, int, str]], Optional[str]]:
    """Require the pressure record to use the selected metrics epoch."""

    interval = _sibling_interval(pressure)
    end = service_metrics.get("end") if isinstance(service_metrics, Mapping) else None
    epoch = _snapshot_epoch(end) if isinstance(end, Mapping) else None
    if interval is None or epoch is None:
        return None, "server_trace_epoch_missing"
    if pressure.get("workload_epoch") != epoch:
        return None, "server_trace_epoch_missing"
    return interval, None


def _trace_pressure_identity_error(
    body: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]],
    running_identity: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Require trace, metrics, and run identity to name one process."""

    end = service_metrics.get("end") if isinstance(service_metrics, Mapping) else None
    metrics_body = end.get("body") if isinstance(end, Mapping) else None
    error = _trace_process_identity_error(body, metrics_body)
    if error is not None:
        return error
    error = _trace_epoch_identity_error(body, end)
    if error is not None:
        return error
    error = _trace_running_epoch_error(body, running_identity)
    if error is not None:
        return error
    error = _trace_source_identity_error(body, metrics_body)
    if error is not None:
        return error
    trace_process = _trace_process_identity(body)
    trace_source = body.get("source_id")
    error = _trace_running_identity_error(trace_process, trace_source, running_identity)
    if error is not None:
        return error
    return None


def _trace_process_identity_error(
    body: Mapping[str, Any], metrics_body: Any
) -> Optional[str]:
    """Require trace and metrics to name one process."""

    trace_process = _trace_process_identity(body)
    metrics_process = _trace_process_identity(metrics_body)
    if trace_process is None or metrics_process is None or trace_process != metrics_process:
        return "server_trace_process_identity_missing"
    return None


def _trace_epoch_identity_error(body: Mapping[str, Any], end: Any) -> Optional[str]:
    """Require trace and metrics to name one workload epoch."""

    epoch = _snapshot_epoch(end if isinstance(end, Mapping) else {})
    return None if body.get("workload_epoch") == epoch else "server_trace_epoch_mismatch"


def _trace_source_identity_error(
    body: Mapping[str, Any], metrics_body: Any
) -> Optional[str]:
    """Require trace and metrics to name one source build."""

    trace_source = body.get("source_id") if isinstance(body.get("source_id"), str) else None
    metrics_source = metrics_body.get("source_id") if isinstance(metrics_body, Mapping) else None
    if trace_source is None or trace_source != metrics_source:
        return "server_trace_source_identity_mismatch"
    return None


def _trace_running_epoch_error(
    body: Mapping[str, Any], running_identity: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Compare trace workload epoch with an identity endpoint epoch when exposed."""

    expected = _running_workload_epoch(running_identity)
    if expected is None:
        return None
    return None if body.get("workload_epoch") == expected else "server_trace_epoch_identity_mismatch"


def _trace_running_identity_error(
    trace_process: Optional[str], trace_source: Any, running_identity: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Require trace identity to match the run start identity."""

    expected_process = _running_process_identity(running_identity)
    if expected_process is not None and trace_process != expected_process:
        return "server_trace_process_identity_mismatch"
    expected_source = _running_source_identity(running_identity)
    if expected_source is not None and trace_source != expected_source:
        return "server_trace_source_identity_mismatch"
    return None


def _trace_process_identity(value: Any) -> Optional[str]:
    """Read the producer process identity required for trace joins."""

    if not isinstance(value, Mapping):
        return None
    for field in ("process_instance_id", "process_start_ns"):
        candidate = value.get(field)
        if isinstance(candidate, (str, int)) and not isinstance(candidate, bool):
            return str(candidate)
    return None


def _running_process_identity(value: Any) -> Optional[str]:
    """Read the serving process identity retained for one frozen run."""

    if not isinstance(value, Mapping):
        return None
    start = value.get("start") if isinstance(value.get("start"), Mapping) else {}
    identity = start.get("identity") if isinstance(start, Mapping) else {}
    return _trace_process_identity(identity)


def _running_source_identity(value: Any) -> Optional[str]:
    """Read the source identity retained for one frozen run."""

    if not isinstance(value, Mapping):
        return None
    start = value.get("start") if isinstance(value.get("start"), Mapping) else {}
    identity = start.get("identity") if isinstance(start, Mapping) else {}
    source = identity.get("source_id") if isinstance(identity, Mapping) else None
    return source if isinstance(source, str) and source else None


def _trace_sibling_result(
    events: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
    body: Mapping[str, Any], slow_numeric: str, interval: Tuple[int, int, str],
) -> Dict[str, Any]:
    """Build one sibling progress claim from complete trace events."""

    roles = _trace_sibling_roles(events, records, body, slow_numeric, interval)
    return {
        "status": "observed" if roles else "unavailable",
        "reason": None if roles else "no_sibling_content_during_slow_reader",
        "roles": roles,
    }


def _trace_identity_body(
    trace_body: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    """Combine trace events with terminal wire-to-scheduler request joins."""

    result = dict(trace_body)
    end = service_metrics.get("end") if isinstance(service_metrics, Mapping) else None
    metrics_body = end.get("body") if isinstance(end, Mapping) else None
    if isinstance(metrics_body, Mapping):
        result.update(metrics_body)
    return result


def _trace_events_valid(events: Sequence[Any]) -> bool:
    """Require complete server trace event identity and timestamps."""

    return all(_trace_event_valid(event) for event in events)


def _trace_history_complete(body: Mapping[str, Any]) -> bool:
    """Reject traces with missing or discarded event and memory samples."""

    return all(
        _nonnegative_int(body.get(field)) and body[field] == 0
        for field in ("dropped_events", "dropped_memory_samples")
    )


def _trace_event_valid(event: Any) -> bool:
    """Validate one trace event before it can support a timing claim."""

    if not isinstance(event, Mapping) or event.get("kind") not in {"prefill_chunk", "resident_decode_progress"}:
        return False
    if not _nonnegative_int(event.get("request_id")) or not _nonnegative_int(event.get("at_ns")):
        return False
    if event["kind"] == "prefill_chunk":
        return _nonnegative_int(event.get("processed_tokens"))
    return _positive_integer(event.get("emitted_tokens"))


def _trace_numeric_request_id(body: Mapping[str, Any], request_id: Optional[str]) -> Optional[str]:
    """Resolve a wire request identity to the server trace's numeric identity."""

    if request_id is None:
        return None
    if request_id.isdigit():
        return request_id
    for row in _trace_terminal_rows(body):
        if request_id in {str(row.get("request_id")), str(row.get("wire_request_id"))}:
            numeric = row.get("numeric_request_id")
            if _nonnegative_int(numeric):
                return str(numeric)
    return None


def _running_workload_epoch(value: Any) -> Optional[Union[str, int]]:
    """Read an optional workload epoch from the retained process identity."""

    if not isinstance(value, Mapping):
        return None
    start = value.get("start") if isinstance(value.get("start"), Mapping) else value
    identity = start.get("identity") if isinstance(start, Mapping) else start
    if not isinstance(identity, Mapping):
        return None
    candidate = identity.get("workload_epoch")
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
        return candidate
    return candidate if isinstance(candidate, str) and candidate else None


def _trace_terminal_rows(body: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Read bounded terminal rows used to join wire and scheduler identities."""

    history = body.get("terminal_requests")
    if not isinstance(history, Mapping):
        history = body.get("requests")
    values = history.get("values") if isinstance(history, Mapping) else None
    return [row for row in values if isinstance(row, Mapping)] if isinstance(values, list) else []


def _trace_sibling_roles(
    events: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
    body: Mapping[str, Any], slow_numeric: str, interval: Tuple[int, int, str],
) -> List[str]:
    """Map server progress events to non-slow harness roles."""

    roles = []
    for event in events:
        if not _trace_progress_in_interval(event, slow_numeric, interval):
            continue
        role = _trace_event_role(event, records, body)
        if role is not None and role not in roles and role != "slow_reader":
            roles.append(role)
    return roles


def _trace_progress_in_interval(
    event: Mapping[str, Any], slow_numeric: str, interval: Tuple[int, int, str]
) -> bool:
    """Check one positive sibling decode event against the selected interval."""

    if event.get("kind") != "resident_decode_progress":
        return False
    at_ns = event.get("at_ns")
    during = event.get("during_prefill_request_ids", [])
    residents = event.get("resident_request_ids", [])
    if event.get("during_prefill_request_id") is not None:
        during = [*during, event.get("during_prefill_request_id")]
    return (
        isinstance(during, list)
        and isinstance(residents, list)
        and slow_numeric in {
            str(value) for value in [*during, *residents] if _nonnegative_int(value)
        }
        and interval[0] <= at_ns <= interval[1]
    )


def _trace_event_role(
    event: Mapping[str, Any], records: Sequence[Mapping[str, Any]], body: Mapping[str, Any]
) -> Optional[str]:
    """Find the harness role for one numeric server request identity."""

    event_id = str(event.get("request_id"))
    return next(
        (
            record.get("role")
            for record in records
            if isinstance(record.get("role"), str)
            and _trace_record_matches(event_id, record, body)
        ),
        None,
    )


def _trace_record_matches(
    event_id: str, record: Mapping[str, Any], body: Mapping[str, Any]
) -> bool:
    """Match a server numeric request identity to one harness record."""

    for field in ("service_request_id", "request_id"):
        value = record.get(field)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            if event_id in _trace_request_aliases(body, str(value)):
                return True
    return False


def _trace_request_aliases(body: Mapping[str, Any], request_id: str) -> set[str]:
    """Return wire and numeric aliases for one terminal request identity."""

    aliases = {request_id}
    for row in _trace_terminal_rows(body):
        values = {str(row.get("request_id")), str(row.get("wire_request_id"))}
        if request_id in values and _nonnegative_int(row.get("numeric_request_id")):
            aliases.add(str(row["numeric_request_id"]))
    return aliases


def _sibling_records(
    records: Sequence[Mapping[str, Any]], start: int, end: int, clock_domain: str,
    workload_epoch: Optional[Union[str, int]] = None,
) -> List[Mapping[str, Any]]:
    """Select non-slow requests that produced content during pressure."""

    return [
        item
        for item in records
        if item.get("role") != "slow_reader"
        and _record_clock_domain(item) == clock_domain
        and (workload_epoch is None or _record_workload_epoch(item) == workload_epoch)
        and isinstance(item.get("first_content_ns"), int)
        and start <= item["first_content_ns"] <= end
    ]


def _record_workload_epoch(record: Mapping[str, Any]) -> Optional[Union[str, int]]:
    """Read one request event's service workload epoch."""

    metrics = record.get("metrics")
    value = metrics.get("workload_epoch") if isinstance(metrics, Mapping) else None
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    return record.get("workload_epoch") if isinstance(record.get("workload_epoch"), (str, int)) else None


def _backpressure_value(result: Mapping[str, Any]) -> Any:
    """Read the server slow-client measurement from one probe."""

    metrics = result.get("service_metrics")
    measurement = metrics.get("slow_client") if isinstance(metrics, dict) else None
    return measurement.get("value") if isinstance(measurement, dict) else None


def _backpressure_value_observed(value: Any) -> bool:
    """Return whether a slow-client measurement reports pressure."""

    return value is True or (
        isinstance(value, dict) and value.get("status") == "observed" and value.get("value") is True
    )


def _backpressure_evidence(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Accept backpressure only when the server reports the condition."""

    value = _backpressure_value(result)
    observed = _backpressure_value_observed(value)
    interval = _pressure_interval(value)
    reason = value.get("reason") if isinstance(value, Mapping) else None
    observed = observed and interval is not None and reason in BACKPRESSURE_REASONS
    evidence = {
        "status": "observed" if observed else "unavailable",
        "reason": reason if observed else "server_backpressure_interval_missing",
    }
    request_id = _pressure_request_id(value) if isinstance(value, Mapping) else None
    if request_id is not None:
        evidence["request_id"] = request_id
    if interval is not None:
        evidence["interval_start_ns"], evidence["interval_end_ns"], evidence["clock_domain"] = interval
    epoch = _result_workload_epoch(result)
    if epoch is not None:
        evidence["workload_epoch"] = epoch
    return evidence


def _result_workload_epoch(result: Mapping[str, Any]) -> Optional[Union[str, int]]:
    """Read the workload epoch retained beside one direct pressure event."""

    metrics = result.get("service_metrics")
    end = metrics.get("end") if isinstance(metrics, Mapping) else None
    if isinstance(end, Mapping):
        return _snapshot_epoch(end)
    return result.get("workload_epoch") if isinstance(result.get("workload_epoch"), (str, int)) else None


def _snapshot_backpressure(snapshot: Mapping[str, Any], request_id: Optional[str] = None) -> Dict[str, Any]:
    """Read measured slow-reader pressure from a bounded service snapshot."""

    body = snapshot.get("body") if isinstance(snapshot, dict) else None
    epoch = _snapshot_epoch(snapshot)
    if epoch is None:
        return {"status": "unavailable", "reason": "server_workload_epoch_missing"}
    history = body.get("slow_client_intervals") if isinstance(body, dict) else None
    if not isinstance(history, dict):
        history = body.get("slow_client") if isinstance(body, dict) else None
    values = history.get("values") if isinstance(history, dict) else None
    if not isinstance(values, list):
        return {"status": "unavailable", "reason": "server_backpressure_history_missing"}
    for item in values:
        evidence = _snapshot_pressure_item(item, request_id, body.get("clock_domain"), epoch)
        if evidence is not None:
            return evidence
    return {"status": "unavailable", "reason": "measured_backpressure_event_missing"}


def _snapshot_request_matches(
    value: Mapping[str, Any], request_id: Optional[str], body: Mapping[str, Any]
) -> bool:
    """Match a service row through its wire or numeric terminal identity."""

    if _pressure_matches_request(value, request_id):
        return True
    history = body.get("terminal_requests")
    if not isinstance(history, Mapping):
        history = body.get("requests")
    values = history.get("values") if isinstance(history, Mapping) else None
    if not isinstance(request_id, str) or not isinstance(values, list):
        return False
    for terminal in values:
        if not isinstance(terminal, Mapping):
            continue
        terminal_ids = {terminal.get("request_id"), terminal.get("wire_request_id")}
        if request_id in terminal_ids:
            numeric_id = terminal.get("numeric_request_id")
            return numeric_id is not None and _pressure_matches_request(
                value, str(numeric_id)
            )
    return False


def _snapshot_cancellation(
    snapshot: Mapping[str, Any], request_id: Optional[str]
) -> Dict[str, Any]:
    """Read terminal cancellation latency from the service snapshot."""

    body = snapshot.get("body") if isinstance(snapshot, Mapping) else None
    history = body.get("cancellation") if isinstance(body, Mapping) else None
    values = history.get("values") if isinstance(history, Mapping) else None
    if not isinstance(body, Mapping) or not isinstance(values, list):
        return _measurement(None, "cancellation_history_missing")
    for item in values:
        if not isinstance(item, Mapping) or not _snapshot_request_matches(item, request_id, body):
            continue
        latency = item.get("latency_ns")
        if _observed_measurement(latency):
            return dict(latency)
    return _measurement(None, "cancellation_completion_unavailable")


def _observed_measurement(value: Any) -> bool:
    """Check one nonnegative observed service measurement."""

    observed = isinstance(value, Mapping) and value.get("status") == "observed"
    measured = value.get("value") if isinstance(value, Mapping) else None
    return (
        observed
        and isinstance(measured, (int, float))
        and not isinstance(measured, bool)
        and math.isfinite(float(measured))
        and measured >= 0
    )


def _attach_cancellation_metric(
    probes: Sequence[Mapping[str, Any]], snapshot: Mapping[str, Any]
) -> None:
    """Replace client-side cancellation timing with terminal service evidence."""

    cancel = next((item for item in probes if item.get("role") == "cancel"), None)
    if not isinstance(cancel, dict):
        return
    metrics = cancel.get("service_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        cancel["service_metrics"] = metrics
    request_id = cancel.get("service_request_id") or cancel.get("request_id")
    metrics["cancel_latency_ns"] = _snapshot_cancellation(snapshot, request_id)


def _snapshot_pressure_item(
    item: Any, request_id: Optional[str], default_clock_domain: Optional[str] = None,
    workload_epoch: Optional[Union[str, int]] = None,
) -> Optional[Dict[str, Any]]:
    """Validate one retained server pressure event."""

    if not isinstance(item, dict):
        return None
    item_id = _pressure_request_id(item)
    latency = item.get("latency_ns")
    interval = _pressure_interval(item, default_clock_domain)
    if not isinstance(item_id, str) or not item_id or not _pressure_matches_request(item, request_id):
        return None
    if latency is not None and (not isinstance(latency, dict) or latency.get("status") != "observed"):
        return None
    if item.get("reason") not in BACKPRESSURE_REASONS or interval is None:
        return None
    return {
        "status": "observed",
        "reason": item["reason"],
        "request_id": item_id,
        "latency_ns": latency,
        "interval_start_ns": interval[0],
        "interval_end_ns": interval[1],
        "clock_domain": interval[2],
        "workload_epoch": workload_epoch,
    }


def _schedule_tasks(
    manifest: Mapping[str, Any],
    engine: Mapping[str, Any],
    parent_id: str,
    parent_messages: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Tuple[Any, ...]]]:
    """Build branch and probe tasks from one observed parent response."""

    prompts = manifest["prompts"]
    parent_history = _history_messages(parent_messages, parent)
    branches: List[Dict[str, Any]] = []
    tasks: List[Tuple[Any, ...]] = []
    branch_count = len(prompts["branches"])
    for index, branch in enumerate(prompts["branches"]):
        branch_id = f"{parent_id}-{branch['id']}"
        mode, fields = _branch_fields(engine, inspection, parent_id, branch_id)
        branch_messages = _prompt_messages(branch)
        if mode == "unavailable":
            branches.append(
                {
                    "branch_id": branch_id,
                    "branch_mode": mode,
                    "status": inspection.get("status", "unverified"),
                    "error": inspection.get("reason", "fork_and_cache_unsupported"),
                    "history_reuse": _history_check(parent_messages, branch_messages, parent, {}),
                }
            )
            continue
        body = _engine_request_body(
            engine,
            request_body(manifest, parent_history + branch_messages, branch_id, engine.get("session_field")),
            _engine_schedule_slot(engine, "branch", index, branch_count),
            proof=True,
        )
        tasks.append(
            (
                "branch",
                branch_id,
                body,
                fields,
                False,
                0.0,
                {
                    "branch_mode": mode,
                    "branch": branch,
                    "fields": fields,
                    "request_body": {**body, **fields},
                },
                STREAM_READ_BYTES,
            )
        )
    schedule = manifest["schedule"]
    new_prompt = schedule["new_prompt"]
    new_id = f"{parent_id}-{new_prompt['id']}"
    tasks.append(
        (
            "new_prompt",
            new_id,
            _engine_request_body(
                engine,
                request_body(manifest, _prompt_messages(new_prompt), new_id, engine.get("session_field")),
                _engine_schedule_slot(engine, "new_prompt", branch_count=branch_count),
            ),
            {},
            False,
            0.0,
            {},
            STREAM_READ_BYTES,
        )
    )
    cancel_prompt = schedule["cancel"]
    cancel_id = f"{parent_id}-{cancel_prompt['id']}"
    cancel_mode, cancel_fields = _branch_fields(engine, inspection, parent_id, cancel_id)
    cancel_body = _engine_request_body(
        engine,
        request_body(
            manifest,
            parent_history + _prompt_messages(cancel_prompt),
            cancel_id,
            engine.get("session_field"),
        ),
        _engine_schedule_slot(engine, "cancel", branch_count=branch_count),
    )
    tasks.append(
        (
            "cancel",
            cancel_id,
            cancel_body,
            cancel_fields,
            True,
            0.0,
            {
                "branch_mode": cancel_mode,
                "branch": cancel_prompt,
                "fields": cancel_fields,
                "request_body": {**cancel_body, **cancel_fields},
            },
            STREAM_READ_BYTES,
        )
    )
    slow_prompt = schedule["slow_reader"]
    slow_id = f"{parent_id}-{slow_prompt['id']}"
    tasks.append(
        (
            "slow_reader",
            slow_id,
            _engine_request_body(
                engine,
                request_body(
                    manifest,
                    parent_history + _prompt_messages(slow_prompt),
                    slow_id,
                    engine.get("session_field"),
                ),
                _engine_schedule_slot(engine, "slow_reader", branch_count=branch_count),
            ),
            {},
            False,
            float(slow_prompt["delay_ms"]) / 1000,
            {},
            int(slow_prompt.get("read_bytes", STREAM_READ_BYTES)),
        )
    )
    return branches, tasks


def _record_probe_result(
    result: Dict[str, Any],
    role: str,
    request_id: str,
    meta: Mapping[str, Any],
    parent_messages: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    branches: List[Dict[str, Any]],
    probes: List[Dict[str, Any]],
) -> None:
    """Attach role-specific history evidence to one completed probe."""

    result["role"] = role
    result["request_id"] = request_id
    if role == "branch":
        branch = meta["branch"]
        result["branch_id"] = request_id
        result["branch_mode"] = meta["branch_mode"]
        result["history_reuse"] = _history_check(
            parent_messages,
            _prompt_messages(branch),
            parent,
            result,
            meta.get("request_body"),
            branch.get("id"),
        )
        branches.append(result)
        return
    if role == "cancel":
        branch = meta["branch"]
        result["branch_mode"] = meta.get("branch_mode", "unavailable")
        result["history_reuse"] = _history_check(
            parent_messages,
            _prompt_messages(branch),
            parent,
            result,
            meta.get("request_body"),
            branch.get("id"),
        )
    probes.append(result)


def _collect_schedule_probes(
    engine: Mapping[str, Any],
    tasks: Sequence[Tuple[Any, ...]],
    parent_messages: Sequence[Mapping[str, Any]],
    parent: Mapping[str, Any],
    timeout_s: float,
    schedule: Mapping[str, Any],
    deadline_ns: Optional[int],
    branches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Run bounded schedule tasks and retain role-specific probe records."""

    probes: List[Dict[str, Any]] = []
    start_barrier = threading.Barrier(len(tasks))
    max_events = schedule.get("max_stream_events", MAX_STREAM_EVENTS)
    max_content_bytes = schedule.get("max_content_bytes", MAX_CONTENT_BYTES)
    max_retained_bytes = MAX_RETAINED_BYTES
    with concurrent.futures.ThreadPoolExecutor(max_workers=schedule["max_workers"]) as pool:
        futures = {
            pool.submit(
                _run_probe,
                engine,
                body,
                fields,
                timeout_s,
                role,
                request_id,
                cancel,
                delay,
                start_barrier,
                max_events,
                max_content_bytes,
                read_bytes,
                deadline_ns,
                max_retained_bytes,
            ): (role, request_id, meta)
            for role, request_id, body, fields, cancel, delay, meta, read_bytes in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            role, request_id, meta = futures[future]
            try:
                result = future.result()
            except Exception as error:  # pragma: no cover - executor boundary
                result = _failed_probe("probe_error", f"{type(error).__name__}: {error}", role, request_id)
            _record_probe_result(result, role, request_id, meta, parent_messages, parent, branches, probes)
    return probes


def _attach_run_history_evidence(
    engine: Mapping[str, Any], parent: Mapping[str, Any], branches: Sequence[Dict[str, Any]],
    timeout_s: float, producer: Optional[Callable[..., Mapping[str, Any]]],
    identity_start: Optional[Mapping[str, Any]] = None,
    slot_copy: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Any, Any]:
    """Fetch end snapshots and attach independent history evidence to branches."""

    selected = producer if producer is not None else _load_history_tokenization_producer()
    metrics_end = fetch_metrics_snapshot(engine, timeout_s)
    service_trace = fetch_service_trace_snapshot(engine, timeout_s)
    identity_end = fetch_server_identity(engine, timeout_s)
    running_identity = {
        "start": identity_start,
        "end": identity_end,
    } if identity_start is not None else identity_end
    producer_engine = {
        **_producer_engine(engine),
        "_service_metrics": metrics_end,
        "_service_trace": service_trace,
        "_running_identity": running_identity,
        "_slot_copy": slot_copy,
    }
    for branch in branches:
        _attach_history_tokenization(producer_engine, parent, branch, selected)
    return metrics_end, service_trace, identity_end


def run_engine(
    manifest: Mapping[str, Any],
    engine: Mapping[str, Any],
    repetition: int,
    timeout_s: float,
    deadline_ns: Optional[int] = None,
    history_tokenization_producer: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run one parent and a bounded concurrent branch schedule."""

    prompts = manifest["prompts"]
    parent_id = f"branch-parent-{engine['id']}-{repetition}"
    parent_messages = _prompt_messages(prompts["parent"])
    identity_start = fetch_server_identity(engine, timeout_s)
    metrics_start = fetch_metrics_snapshot(engine, timeout_s)
    parent = nonstream_request(
        engine,
        _engine_request_body(
            engine,
            request_body(
                manifest,
                parent_messages,
                parent_id,
                engine.get("session_field"),
                stream=False,
            ),
            _engine_schedule_slot(engine, "parent"),
            proof=True,
        ),
        {},
        timeout_s,
        max_content_bytes=manifest["budgets"].get("max_content_bytes", MAX_CONTENT_BYTES),
    )
    parent["request_id"] = parent_id
    inspection = inspect_branch_method(engine, timeout_s)
    schedule = manifest["schedule"]
    branches, tasks = _schedule_tasks(
        manifest, engine, parent_id, parent_messages, parent, inspection
    )
    slot_copy = _prepare_slot_copies(engine, parent_id, parent, tasks, timeout_s)
    probes = _collect_schedule_probes(
        engine,
        tasks,
        parent_messages,
        parent,
        timeout_s,
        {**schedule, **manifest["budgets"]},
        deadline_ns,
        branches,
    )
    branches.sort(key=lambda item: item["branch_id"])
    probes.sort(key=lambda item: (item.get("role", ""), item.get("request_id", "")))
    metrics_end, service_trace, identity_end = _attach_run_history_evidence(
        engine, parent, branches, timeout_s, history_tokenization_producer, identity_start, slot_copy
    )
    slot_copy = _release_slot_files(engine, slot_copy)
    _attach_cancellation_metric(probes, metrics_end)
    slow = next((item for item in probes if item.get("role") == "slow_reader"), None)
    if isinstance(slow, dict):
        pressure_id = slow.get("service_request_id") or slow.get("request_id")
        snapshot_pressure = _snapshot_backpressure(metrics_end, pressure_id)
        if snapshot_pressure.get("status") == "observed":
            slow["backpressure"] = snapshot_pressure
    parent.pop("_content_text", None)
    records = [parent, *branches, *probes]
    pressure = slow.get("backpressure") if isinstance(slow, dict) else None
    trace_for_sibling = service_trace if isinstance(engine.get("trace_endpoint"), str) else None
    return {
        "engine": engine["id"],
        "engine_kind": engine_kind(engine),
        "comparison_support": capability_record(engine_kind(engine)),
        "repetition": repetition,
        "parent": parent,
        "branches": branches,
        "probes": probes,
        "inspection": inspection,
        "schedule": {
            "max_workers": schedule["max_workers"],
            "roles": ["branch", "new_prompt", "cancel", "slow_reader"],
            "barrier": {"status": "observed", "party_count": len(tasks)},
        },
        "overlap": _schedule_overlap(records),
        "sibling_progress": _sibling_progress(records, pressure, trace_for_sibling, metrics_end),
        "service_metrics": {"start": metrics_start, "end": metrics_end},
        "service_trace": service_trace if trace_for_sibling is not None else None,
        "slot_copy": slot_copy,
        "running_identity": {
            "start": identity_start,
            "end": identity_end,
        },
    }


def _record_window(record: Any) -> Optional[Tuple[int, int]]:
    """Read one integer request interval in client monotonic nanoseconds."""

    if not isinstance(record, Mapping):
        return None
    start, end = record.get("request_start_ns"), record.get("request_end_ns")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, end)) or end < start:
        return None
    return start, end


def _slot_exchanges(plan: Any) -> List[Any]:
    """List the save exchange and every restore exchange of one slot copy plan."""

    restores = plan.get("restore") if isinstance(plan, Mapping) else None
    if not isinstance(restores, Mapping):
        return []
    return [plan.get("save"), *restores.values()]


def _preparation_window(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the sequential slot copy interval that precedes the timed barrier."""

    plan = run.get("slot_copy")
    if plan is None:
        return {"status": "not_required"}
    exchanges = _slot_exchanges(plan)
    windows = [_record_window(item) for item in exchanges]
    complete = len(exchanges) >= 2 and all(
        window is not None and item.get("http_status") == 200 for window, item in zip(windows, exchanges)
    )
    if not complete:
        return {"status": "unavailable", "reason": "slot_copy_incomplete"}
    return {"status": "observed", "start_ns": min(w[0] for w in windows), "end_ns": max(w[1] for w in windows)}


def _completed_tokens(records: Sequence[Mapping[str, Any]]) -> Optional[int]:
    """Sum completion tokens over completed records, or None when one count is missing."""

    total = 0
    for record in records:
        if record.get("status") != "completed":
            continue
        usage = record.get("usage")
        count = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        total += count
    return total


def _token_definition(records: Sequence[Mapping[str, Any]]) -> str:
    """Name the count definition that a run's completed requests support.

    Completion counts are comparable across engines only at a limit stop.
    """

    counted = [item for item in records if item.get("status") == "completed"]
    if counted and all(item.get("finish_reason") == "length" for item in counted):
        return TOKEN_COUNT_DEFINITION
    return "unverified_stop_mix"


def _branch_latencies(branches: Sequence[Mapping[str, Any]], origin_ns: int) -> Optional[List[Tuple[float, float]]]:
    """Pair each branch's ready first-token latency with its inclusive latency.

    Ready latency starts at the branch request. Inclusive latency starts at
    `origin_ns`, the start of the slot copy, so the copy is part of the wait.
    """

    pairs = []
    for branch in branches:
        ready, window = _record_ttft(branch), _record_window(branch)
        if ready is None or window is None:
            return None
        pairs.append((ready, ready + (window[0] - origin_ns) / 1_000_000))
    return pairs


def run_accounting(run: Mapping[str, Any]) -> Dict[str, Any]:
    """Account one run with and without its setup, from retained request intervals."""

    branches = [item for item in run.get("branches", []) if isinstance(item, Mapping)]
    timed = [*branches, *(item for item in run.get("probes", []) if isinstance(item, Mapping))]
    windows = [_record_window(item) for item in timed]
    preparation = _preparation_window(run)
    if not branches or None in windows:
        return {"status": "unavailable", "reason": "timed_intervals_missing"}
    if preparation["status"] == "unavailable":
        return {"status": "unavailable", "reason": preparation["reason"]}
    return _observed_accounting(branches, timed, windows, preparation)


def _observed_accounting(
    branches: Sequence[Mapping[str, Any]], timed: Sequence[Mapping[str, Any]],
    windows: Sequence[Tuple[int, int]], preparation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build one run's accounting. The end-to-end clock starts at the slot copy."""

    ready_start, end = min(w[0] for w in windows), max(w[1] for w in windows)
    origin = preparation.get("start_ns", ready_start)
    tokens = _completed_tokens(timed)
    latencies = _branch_latencies(branches, origin)
    if tokens is None or latencies is None or end <= origin:
        return {"status": "unavailable", "reason": "tokens_or_latency_missing"}
    return {
        "status": "observed", "tokens": tokens, "token_count_definition": _token_definition(timed),
        "ready_window_ns": end - ready_start,
        "preparation_ns": ready_start - origin,
        "ready_first_token_ms": [pair[0] for pair in latencies],
        "inclusive_first_token_ms": [pair[1] for pair in latencies],
    }


def _pooled(rows: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    """Pool one per-branch latency list across runs."""

    return [value for row in rows for value in row[field]]


def _engine_token_definition(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the one count definition every run shares, or the unverified mix."""

    definitions = {row["token_count_definition"] for row in rows}
    return next(iter(definitions)) if len(definitions) == 1 else "unverified_stop_mix"


def _engine_accounting(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate run accounting for one engine. Setup time enters the end-to-end window."""

    bad = next((row for row in rows if row.get("status") != "observed"), None)
    if bad is not None or not rows:
        return {"status": "unavailable", "reason": bad.get("reason") if bad else "no_runs"}
    tokens = sum(row["tokens"] for row in rows)
    ready = sum(row["ready_window_ns"] for row in rows)
    setup = sum(row["preparation_ns"] for row in rows)
    return {
        "status": "observed", "run_count": len(rows), "tokens": tokens,
        "token_count_definition": _engine_token_definition(rows),
        "ready_window_ns": ready, "preparation_ns": setup, "end_to_end_window_ns": ready + setup,
        "ready_tokens_per_s": tokens * 1e9 / ready,
        "end_to_end_tokens_per_s": tokens * 1e9 / (ready + setup),
        "ready_first_token_p95_ms": linear_quantile(_pooled(rows, "ready_first_token_ms"), 0.95),
        "inclusive_first_token_p95_ms": linear_quantile(_pooled(rows, "inclusive_first_token_ms"), 0.95),
    }


ACCOUNTING_DIRECTIONS = {
    "ready_tokens_per_s": 1, "end_to_end_tokens_per_s": 1,
    "ready_first_token_p95_ms": -1, "inclusive_first_token_p95_ms": -1,
}


def _accounting_winner(engines: Mapping[str, Mapping[str, Any]], metric: str) -> Optional[str]:
    """Return the engine that strictly wins one metric, or None on a tie."""

    ranked = sorted(engines, key=lambda name: -ACCOUNTING_DIRECTIONS[metric] * engines[name][metric])
    if engines[ranked[0]][metric] == engines[ranked[1]][metric]:
        return None
    return ranked[0]


def compare_accounting(engines: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Name a faster engine only when it wins ready and setup-inclusive metrics alike.

    Both engines must also report the same verified token count definition.
    """

    if len(engines) != 2 or any(item.get("status") != "observed" for item in engines.values()):
        return {"status": "unavailable", "speed_claim": None}
    winners = {metric: _accounting_winner(engines, metric) for metric in ACCOUNTING_DIRECTIONS}
    reversed_by_setup = any(
        winners[ready] is not None and winners[ready] != winners[inclusive]
        for ready, inclusive in (
            ("ready_tokens_per_s", "end_to_end_tokens_per_s"),
            ("ready_first_token_p95_ms", "inclusive_first_token_p95_ms"),
        )
    )
    result = {"status": "observed", "winners": winners, "setup_reverses_ready_winner": reversed_by_setup}
    definitions = {item["token_count_definition"] for item in engines.values()}
    if definitions != {TOKEN_COUNT_DEFINITION}:
        return {**result, "speed_claim": None, "claim_withheld": "token_count_definitions_differ_or_unverified"}
    unanimous = set(winners.values())
    return {**result, "speed_claim": next(iter(unanimous)) if len(unanimous) == 1 else None}


def receipt_end_to_end(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute per-engine accounting and the speed comparison from run records."""

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in receipt.get("runs", []):
        if isinstance(run, Mapping) and isinstance(run.get("engine"), str):
            grouped.setdefault(run["engine"], []).append(run_accounting(run))
    engines = {name: _engine_accounting(rows) for name, rows in sorted(grouped.items())}
    return {"engines": engines, "comparison": compare_accounting(engines)}


def _end_to_end_errors(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require setup-inclusive accounting that recomputes, and setup where declared."""

    errors = []
    if receipt.get("end_to_end") != receipt_end_to_end(receipt):
        errors.append("end-to-end accounting is missing or does not recompute from run records")
    required = {
        item.get("id") for item in manifest.get("engines", [])
        if isinstance(item, Mapping) and isinstance(item.get("slot_copy"), Mapping)
    }
    for run in receipt.get("runs", []):
        if isinstance(run, Mapping) and (run.get("engine") in required) != isinstance(run.get("slot_copy"), Mapping):
            errors.append("run slot copy does not match the engine declaration")
        errors.extend(_slot_cleanup_errors(run))
    return errors


SLOT_CLEANUP_STATUSES = {"removed", "missing", "not_owned"}


def _slot_cleanup_errors(run: Any) -> List[str]:
    """Require each slot copy to state whether its saved file was removed."""

    plan = run.get("slot_copy") if isinstance(run, Mapping) else None
    if not isinstance(plan, Mapping):
        return []
    cleanup = plan.get("cleanup")
    if not isinstance(cleanup, Mapping) or cleanup.get("status") not in SLOT_CLEANUP_STATUSES:
        return ["run slot copy lacks a typed slot file cleanup status"]
    return []


def prompt_provenance(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Hash every declared message sequence before a run."""

    prompts = [
        manifest["prompts"]["parent"],
        *manifest["prompts"]["branches"],
        manifest["schedule"]["new_prompt"],
        manifest["schedule"]["cancel"],
        manifest["schedule"]["slow_reader"],
    ]
    return [
        {
            "id": item["id"],
            "sha256": sha256_bytes(canonical_json(_prompt_messages(item))),
            "message_count": len(_prompt_messages(item)),
            "word_count": sum(len(str(message["content"]).split()) for message in _prompt_messages(item)),
            "minimum_prompt_tokens": item.get("minimum_prompt_tokens"),
        }
        for item in prompts
    ]


def artifact_provenance(
    manifest: Mapping[str, Any], root: Path, before: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Hash the recorded artifacts that exist at run time."""

    result: Dict[str, Any] = {}
    for name, item in manifest.get("artifacts", {}).items():
        try:
            path = root_path(root, item["path"])
        except ValueError:
            result[name] = {
                "path": item.get("path"),
                "sha256": None,
                "status": "unavailable",
                "reason": "artifact_path_escapes_root",
            }
            continue
        row = {
            "path": item["path"],
            "sha256": sha256_file(path) if path.is_file() else None,
            "status": "observed" if path.is_file() else "unavailable",
            "reason": None if path.is_file() else "artifact_missing",
        }
        if isinstance(before, Mapping) and isinstance(before.get(name), Mapping):
            row["before_sha256"] = before[name].get("sha256")
            row["before_status"] = before[name].get("status")
        result[name] = row
    return result


LLAMA_VERSION_LINE = re.compile(r"^version: .*\(build \d+, commit ([0-9a-f]{7,40})\)", re.MULTILINE)


def _llama_source_id(stdout: bytes, stderr: bytes, pinned: Optional[str]) -> Optional[str]:
    """Parse the `--version` line llama.cpp prints to stderr and bind it to the pinned commit.

    The reported commit is short. It counts only as a prefix of the pinned
    full commit, and the emitted id carries the full pinned commit.
    """

    found = LLAMA_VERSION_LINE.search((stderr + b"\n" + stdout).decode("utf-8", "replace"))
    if found is None or not isinstance(pinned, str) or not _valid_commit(pinned) or not pinned.startswith(found.group(1)):
        return None
    return f"llama.cpp:git:{pinned}"


def _pinned_llama_commit(root: Path) -> Optional[str]:
    """Read the pinned llama.cpp commit from the study root, or None when absent."""

    try:
        text = (root / "external" / "PINNED").read_text().strip()
    except OSError:
        return None
    return text if _valid_commit(text) else None


def _build_info_source_id(
    kind: str, output: bytes, stderr: bytes = b"", pinned: Optional[str] = None
) -> Optional[str]:
    """Extract a source identity only when the executable reports one."""

    if kind == "llama.cpp":
        return _llama_source_id(output, stderr, pinned)
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        for field in ("source_id", "git_commit", "source_commit", "commit"):
            candidate = value.get(field)
            if isinstance(candidate, str) and candidate:
                return candidate if field == "source_id" else f"{kind}:git:{candidate}"
    return None


def engine_provenance(engine: Mapping[str, Any], root: Path, timeout_s: float = 10.0) -> Dict[str, Any]:
    """Hash the executable and record build output from the actual process."""

    path_text = engine.get("executable_path")
    if not isinstance(path_text, str) or not path_text:
        return {"status": "unavailable", "reason": "executable_path_missing"}
    try:
        path = root_path(root, path_text)
    except ValueError:
        return {
            "status": "unavailable",
            "reason": "executable_path_escapes_root",
            "executable_path": path_text,
        }
    if not path.is_file():
        return {
            "status": "unavailable",
            "reason": "executable_missing",
            "executable_path": path_text,
        }
    result: Dict[str, Any] = {
        "status": "observed",
        "executable_path": path_text,
        "executable_sha256": sha256_file(path),
    }
    result["build_info"] = _run_build_info(engine, root, timeout_s)
    if engine.get("quality_producer") == "peer_adapter":
        result["linked_libraries"] = _linked_libraries_record(engine, path)
    return result


def _linked_libraries_record(engine: Mapping[str, Any], binary: Path) -> Dict[str, Any]:
    """Resolve the numeric libraries the peer server links, without reading its process map."""

    module = _load_sibling_module("linked_libraries.py")
    try:
        libraries = module.resolve(binary, binary.parent, engine.get("backend"))
    except (module.LinkageError, OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return {"status": "observed", "loaded_library_status": LOADED_LIBRARY_STATUS, "libraries": libraries}


def _run_build_info(engine: Mapping[str, Any], root: Path, timeout_s: float) -> Dict[str, Any]:
    """Run the declared build identity command without shell expansion."""

    command = engine.get("build_info_command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        return {"status": "unavailable", "reason": "build_info_command_missing"}
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}
    return _build_info_result(engine, completed, _pinned_llama_commit(root))


def _build_info_result(engine: Mapping[str, Any], completed: Any, pinned: Optional[str] = None) -> Dict[str, Any]:
    """Hash build output and extract source identity only from reported data."""

    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    source_id = _build_info_source_id(str(engine.get("kind", "engine")), stdout, stderr, pinned)
    result: Dict[str, Any] = {
        "status": "observed" if completed.returncode == 0 else "unavailable",
        "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(stdout),
        "stderr_sha256": sha256_bytes(stderr),
        "source_id": source_id,
        "source_id_status": "observed" if source_id is not None else "unavailable",
        "source_id_reason": None if source_id is not None else "build_output_did_not_report_source_id",
    }
    if completed.returncode != 0:
        result["reason"] = "build_info_command_failed"
    return result


def validate_manifest(manifest: Mapping[str, Any]) -> List[str]:
    """Return all manifest schema errors without applying default gates."""

    phase = manifest.get("phase")
    errors = _validate_header(manifest, phase)
    errors.extend(_validate_budgets(manifest.get("budgets")))
    errors.extend(_sample_policy_errors(manifest.get("budgets"), phase))
    errors.extend(_validate_prompts(manifest.get("prompts")))
    errors.extend(_validate_prompt_ids(manifest))
    errors.extend(_validate_engines(manifest.get("engines")))
    errors.extend(_validate_request(manifest.get("request")))
    errors.extend(_validate_schedule(manifest.get("schedule")))
    errors.extend(_validate_schedule_capacity(manifest))
    errors.extend(_validate_artifacts(manifest.get("artifacts")))
    errors.extend(_validate_evaluation(manifest.get("evaluation"), phase))
    errors.extend(_threshold_scope_errors(manifest))
    if not isinstance(manifest.get("workload_id"), str) or not manifest["workload_id"]:
        errors.append("workload_id must be nonempty")
    if phase == "frozen":
        errors.extend(_validate_frozen_requirements(manifest))
    return errors


def _threshold_scope_errors(manifest: Mapping[str, Any]) -> List[str]:
    """Require threshold scope to match one declared engine target."""

    evaluation = manifest.get("evaluation")
    thresholds = evaluation.get("thresholds") if isinstance(evaluation, dict) else None
    engines = manifest.get("engines")
    if not isinstance(thresholds, list) or not isinstance(engines, list):
        return []
    targets = {item.get("id"): item for item in engines if isinstance(item, dict)}
    return [
        error
        for index, threshold in enumerate(thresholds)
        for error in _threshold_scope_item_errors(threshold, index, targets, manifest)
    ]


def _threshold_scope_item_errors(
    threshold: Any, index: int, targets: Mapping[Any, Mapping[str, Any]], manifest: Mapping[str, Any]
) -> List[str]:
    """Compare one threshold scope with its engine declaration."""

    if not isinstance(threshold, dict):
        return []
    engine = targets.get(threshold.get("engine"))
    if engine is None:
        return [f"evaluation.thresholds[{index}].engine is not in engines"]
    expected = {
        "model": engine.get("model", manifest.get("request", {}).get("model")),
        "backend": engine.get("backend", engine.get("kind")),
        "device": engine.get("device"),
    }
    return [
        f"evaluation.thresholds[{index}].{field} differs from engine"
        for field, value in expected.items()
        if field in threshold and threshold[field] != value
    ]


def _validate_schedule_capacity(manifest: Mapping[str, Any]) -> List[str]:
    """Prevent a start barrier from waiting behind an undersized pool."""

    prompts = manifest.get("prompts")
    schedule = manifest.get("schedule")
    if not isinstance(prompts, dict) or not isinstance(schedule, dict):
        return []
    branches = prompts.get("branches")
    workers = schedule.get("max_workers")
    if not isinstance(branches, list) or not isinstance(workers, int):
        return []
    required = len(branches) + len(REQUIRED_SCHEDULE_ROLES)
    return [f"schedule.max_workers must be at least {required}"] if workers < required else []


def _validate_prompt_ids(manifest: Mapping[str, Any]) -> List[str]:
    """Keep prompt and schedule identifiers unique in one receipt."""

    prompts = manifest.get("prompts", {})
    schedule = manifest.get("schedule", {})
    items = []
    if isinstance(prompts, dict):
        branches = prompts.get("branches") if isinstance(prompts.get("branches"), list) else []
        items.extend([prompts.get("parent"), *branches])
    if isinstance(schedule, dict):
        items.extend(schedule.get(name) for name in ("new_prompt", "cancel", "slow_reader"))
    ids = [item.get("id") for item in items if isinstance(item, dict)]
    return ["prompt IDs must be unique across schedule"] if len(ids) != len(set(ids)) else []


def _validate_header(manifest: Mapping[str, Any], phase: Any) -> List[str]:
    """Validate schema and phase labels."""

    errors = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unexpected schema_version")
    if phase not in {"calibration", "pending", "frozen"}:
        errors.append("phase must be calibration, pending, or frozen")
    if manifest.get("freeze_status") != phase:
        errors.append("freeze_status must equal phase")
    return errors


def _validate_budgets(budgets: Any) -> List[str]:
    """Validate positive bounded study budgets."""

    if not isinstance(budgets, dict):
        return ["budgets must be an object"]
    fields = (
        "repetitions",
        "minimum_samples_for_quantiles",
        "max_history",
        "max_stream_events",
        "max_content_bytes",
        "wall_time_limit_s",
    )
    return [
        f"budgets.{field} must be a positive integer"
        for field in fields
        if not isinstance(budgets.get(field), int) or budgets[field] <= 0
    ]


def _sample_policy_errors(budgets: Any, phase: Any) -> List[str]:
    """Require enough independent observations for each study phase."""

    if not isinstance(budgets, Mapping):
        return []
    repetitions = budgets.get("repetitions")
    minimum = budgets.get("minimum_samples_for_quantiles")
    if not _sample_budget_values_valid(repetitions, minimum):
        return []
    floor = _sample_floor(phase)
    return _sample_floor_errors(repetitions, minimum, floor, phase) + _sample_relation_errors(repetitions, minimum)


def _sample_budget_values_valid(repetitions: Any, minimum: Any) -> bool:
    """Check typed sample budget values before applying phase floors."""

    return all(isinstance(value, int) and not isinstance(value, bool) for value in (repetitions, minimum))


def _sample_floor(phase: Any) -> int:
    """Return the minimum sample count for one study phase."""

    return MINIMUM_CALIBRATION_SAMPLES if phase == "calibration" else MINIMUM_FROZEN_SAMPLES if phase == "frozen" else 1


def _sample_floor_errors(repetitions: int, minimum: int, floor: int, phase: Any) -> List[str]:
    """Report sample counts below the phase floor."""

    errors = []
    if repetitions < floor:
        errors.append(f"budgets.repetitions must be at least {floor} for {phase} evidence")
    if minimum < floor:
        errors.append(f"budgets.minimum_samples_for_quantiles must be at least {floor} for {phase} evidence")
    return errors


def _sample_relation_errors(repetitions: int, minimum: int) -> List[str]:
    """Require the quantile sample target to fit the repetition budget."""

    return ["budgets.minimum_samples_for_quantiles cannot exceed repetitions"] if minimum > repetitions else []


def _validate_artifacts(artifacts: Any) -> List[str]:
    """Validate artifact paths before hashing them."""

    if not isinstance(artifacts, dict):
        return ["artifacts must be an object"]
    errors = []
    for name, item in artifacts.items():
        path = item.get("path") if isinstance(item, dict) else None
        error = _manifest_path_error(path, f"artifact {name}.path")
        if error:
            errors.append(error)
    return errors


def _validate_prompts(prompts: Any) -> List[str]:
    if not _prompt_container_valid(prompts):
        return ["prompts must contain parent and branches"]
    all_prompts = [prompts["parent"], *prompts["branches"]]
    errors = _validate_prompt_items(all_prompts)
    errors.extend(_prompt_id_errors(all_prompts))
    errors.extend(_parent_prompt_errors(prompts["parent"]))
    return errors


def _prompt_container_valid(prompts: Any) -> bool:
    """Check the prompt collection shape before validating its items."""

    return isinstance(prompts, dict) and isinstance(prompts.get("parent"), dict) and isinstance(prompts.get("branches"), list)


def _prompt_id_errors(items: Sequence[Any]) -> List[str]:
    """Reject duplicate prompt identifiers."""

    ids = [item.get("id") for item in items if isinstance(item, dict)]
    return ["prompt IDs must be unique"] if len(ids) != len(set(ids)) else []


def _parent_prompt_errors(parent: Mapping[str, Any]) -> List[str]:
    """Require a positive long-context prompt bound."""

    minimum = parent.get("minimum_prompt_tokens")
    return [] if isinstance(minimum, int) and not isinstance(minimum, bool) and minimum > 0 else [
        "prompts.parent.minimum_prompt_tokens must be positive"
    ]


def _validate_prompt_items(items: Sequence[Any]) -> List[str]:
    """Validate prompt IDs and message fields."""

    for item in items:
        error = _validate_prompt(item)
        if error:
            return [error]
    return []


def _validate_prompt(item: Any) -> Optional[str]:
    """Validate one prompt and its message sequence."""

    error = _prompt_identity_error(item)
    if error:
        return error
    messages = item.get("messages")
    return _prompt_messages_error(item, messages)


def _prompt_identity_error(item: Any) -> Optional[str]:
    """Validate the prompt identity before reading message fields."""

    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
        return "every prompt needs a nonempty id"
    return None


def _prompt_messages_error(item: Mapping[str, Any], messages: Any) -> Optional[str]:
    """Validate text fallback or structured messages."""

    if messages is None and isinstance(item.get("text"), str) and item["text"]:
        return None
    if not isinstance(messages, list) or not messages:
        return f"prompt {item['id']} needs nonempty messages"
    return None if all(_valid_message(message) for message in messages) else f"prompt {item['id']} has an invalid message"


def _valid_message(message: Any) -> bool:
    """Return whether one message has a supported role and content."""

    return (
        isinstance(message, dict)
        and message.get("role") in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
        and bool(message["content"])
    )


def _validate_schedule(schedule: Any) -> List[str]:
    """Require the bounded concurrent probes used by the study."""

    if not isinstance(schedule, dict):
        return ["schedule must be an object"]
    errors = []
    workers = schedule.get("max_workers")
    errors.extend(_schedule_worker_errors(workers))
    for name in ("new_prompt", "cancel", "slow_reader"):
        item = schedule.get(name)
        errors.extend(_validate_prompt_items([item]))
    slow = schedule.get("slow_reader")
    errors.extend(_slow_reader_errors(slow))
    cancel = schedule.get("cancel")
    errors.extend(_cancel_probe_errors(cancel))
    if isinstance(slow, dict) and slow.get("backpressure_required") is not True:
        errors.append("schedule.slow_reader.backpressure_required must be true")
    return errors


def _schedule_worker_errors(workers: Any) -> List[str]:
    """Validate the worker count before the barrier is constructed."""

    if not isinstance(workers, int) or not 0 < workers <= MAX_SCHEDULE_WORKERS:
        return ["schedule.max_workers must be a positive integer"]
    return ["schedule.max_workers must allow one branch and three probes"] if workers < 4 else []


def _slow_reader_errors(slow: Any) -> List[str]:
    """Validate slow-reader delay and read bounds."""

    if not isinstance(slow, dict):
        return []
    errors = []
    delay = slow.get("delay_ms")
    if not isinstance(delay, int) or not 0 < delay <= 60_000:
        errors.append("schedule.slow_reader.delay_ms must be positive")
    read_bytes = slow.get("read_bytes", STREAM_READ_BYTES)
    if not isinstance(read_bytes, int) or not 0 < read_bytes <= STREAM_READ_BYTES:
        errors.append("schedule.slow_reader.read_bytes must be between one and the stream read bound")
    return errors


def _cancel_probe_errors(cancel: Any) -> List[str]:
    """Require cancellation after content and explicit server acknowledgement."""

    if not isinstance(cancel, dict):
        return []
    errors = []
    if cancel.get("after_first_content") is not True:
        errors.append("schedule.cancel.after_first_content must be true")
    if cancel.get("acknowledgement_required") is not True:
        errors.append("schedule.cancel.acknowledgement_required must be true")
    return errors


def _validate_evaluation(evaluation: Any, phase: Any) -> List[str]:
    """Validate explicit evaluation state without inventing gates."""

    if not isinstance(evaluation, dict):
        return ["evaluation must be an object"]
    errors = _evaluation_threshold_errors(evaluation, phase)
    errors.extend(_evaluation_quality_errors(evaluation, phase))
    errors.extend(_evaluation_receipt_errors(evaluation))
    return errors


def _evaluation_threshold_errors(evaluation: Mapping[str, Any], phase: Any) -> List[str]:
    """Validate the closed threshold schema and the frozen nonempty requirement."""

    thresholds = evaluation.get("thresholds")
    if thresholds is None or thresholds == {}:
        return [] if phase != "frozen" else ["frozen evaluation.thresholds must be a nonempty list"]
    if not isinstance(thresholds, list):
        return ["evaluation.thresholds must be a list"]
    errors = []
    seen = set()
    for index, threshold in enumerate(thresholds):
        errors.extend(_threshold_item_errors(threshold, index, seen, phase == "frozen"))
        if phase == "frozen" and isinstance(threshold, Mapping):
            errors.extend(_threshold_calibration_shape_errors(threshold, f"evaluation.thresholds[{index}]"))
    return errors


def _threshold_item_errors(
    threshold: Any, index: int, seen: set, allow_calibration: bool = False
) -> List[str]:
    """Validate one typed threshold and update the identifier set."""

    prefix = f"evaluation.thresholds[{index}]"
    if not isinstance(threshold, dict):
        return [f"{prefix} must be an object"]
    required = {
        "id", "engine", "role", "scenario", "metric", "model", "backend", "device",
        "operator", "value", "unit",
    }
    if allow_calibration:
        required.add("calibration")
    errors = []
    unknown = set(threshold) - required
    missing = required - set(threshold)
    if unknown:
        errors.append(f"{prefix} has unknown fields: {','.join(sorted(unknown))}")
    if missing:
        errors.append(f"{prefix} is missing fields: {','.join(sorted(missing))}")
    core_fields = required - {"calibration"}
    if core_fields.issubset(threshold):
        errors.extend(_threshold_identity_errors(threshold, prefix, seen))
        errors.extend(_threshold_value_errors(threshold, prefix))
    return errors


def _threshold_identity_errors(threshold: Mapping[str, Any], prefix: str, seen: set) -> List[str]:
    """Validate threshold scope and identifier uniqueness."""

    threshold_id = threshold["id"]
    duplicate = isinstance(threshold_id, str) and threshold_id in seen
    errors = []
    if not isinstance(threshold_id, str) or not threshold_id or duplicate:
        errors.append(f"{prefix}.id must be unique and nonempty")
    if isinstance(threshold_id, str):
        seen.add(threshold_id)
    errors.extend(_threshold_scope_field_errors(threshold, prefix))
    errors.extend(_threshold_scenario_errors(threshold, prefix))
    return errors


def _threshold_scope_field_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Require nonempty strings for threshold scope fields."""

    return [
        f"{prefix}.{field} must be nonempty"
        for field in ("engine", "role", "scenario", "metric", "model", "backend", "device")
        if not isinstance(threshold[field], str) or not threshold[field]
    ]


def _threshold_scenario_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Require supported threshold role and scenario labels."""

    errors = []
    for field in ("role", "scenario"):
        if isinstance(threshold.get(field), str) and threshold[field] not in THRESHOLD_SCENARIOS:
            errors.append(f"{prefix}.{field} is not supported")
    return errors


def _threshold_value_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Validate metric, operator, unit, and bound types."""

    errors = _threshold_kind_errors(threshold, prefix)
    errors.extend(_threshold_bound_errors(threshold, prefix))
    errors.extend(_threshold_direction_errors(threshold, prefix))
    errors.extend(_threshold_metric_scenario_errors(threshold, prefix))
    return errors


def _threshold_kind_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Validate threshold metric, operator, and unit declarations."""

    metric = threshold["metric"]
    operator = threshold["operator"]
    unit = threshold["unit"]
    errors = []
    if metric not in THRESHOLD_METRICS:
        errors.append(f"{prefix}.metric is not supported")
    if operator not in THRESHOLD_OPERATORS:
        errors.append(f"{prefix}.operator is not supported")
    if unit not in THRESHOLD_UNITS:
        errors.append(f"{prefix}.unit is not supported")
    elif metric in THRESHOLD_METRIC_UNITS and unit != THRESHOLD_METRIC_UNITS[metric]:
        errors.append(f"{prefix}.unit does not match metric")
    return errors


def _threshold_bound_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Validate one numeric or boolean threshold bound."""

    if threshold["unit"] == "boolean":
        return [] if threshold["value"] is True and threshold["operator"] == "==" else [
            f"{prefix} boolean thresholds require value true and operator =="
        ]
    value = threshold["value"]
    return [] if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else [
        f"{prefix}.value must be finite"
    ]


def _threshold_direction_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Reject threshold directions and bounds that cannot express a useful gate."""

    metric = threshold["metric"]
    if threshold["unit"] == "boolean":
        return []
    rules = {
        "ttft_p95_ms": ({"<", "<="}, 0.0, None),
        "inter_token_latency_p95_ms": ({"<", "<="}, 0.0, None),
        "fork_latency_p95_ms": ({"<", "<="}, 0.0, None),
        "cancel_latency_p95_ms": ({"<", "<="}, 0.0, None),
        "physical_memory_peak_bytes": ({"<", "<="}, 0.0, None),
        "history_reuse_min_tokens": ({">", ">="}, 0.0, None),
        "request_success_rate": ({">", ">="}, 0.0, 1.0),
    }
    rule = rules.get(metric)
    if rule is None:
        return []
    operators, lower, upper = rule
    value = threshold["value"]
    errors = []
    if threshold["operator"] not in operators:
        errors.append(f"{prefix}.operator does not match metric direction")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        return errors
    if value <= lower:
        errors.append(f"{prefix}.value must be greater than zero")
    if upper is not None and value > upper:
        errors.append(f"{prefix}.value must be at most one")
    return errors


def _threshold_calibration_shape_errors(
    threshold: Mapping[str, Any], prefix: str
) -> List[str]:
    """Require a self-contained, hashed calibration decision for one bound."""

    decision = threshold.get("calibration")
    if not isinstance(decision, Mapping):
        return [f"{prefix}.calibration decision is missing"]
    errors = _threshold_calibration_field_errors(decision, prefix)
    if CALIBRATION_DECISION_FIELDS.issubset(decision):
        errors.extend(_threshold_calibration_value_errors(threshold, decision, prefix))
        if decision.get("decision_sha256") != _threshold_decision_digest(threshold, decision):
            errors.append(f"{prefix}.calibration decision hash does not recompute")
    return errors


def _threshold_calibration_field_errors(
    decision: Mapping[str, Any], prefix: str
) -> List[str]:
    """Validate the fields shared by every calibration decision."""

    unknown = set(decision) - CALIBRATION_DECISION_FIELDS
    missing = CALIBRATION_DECISION_FIELDS - set(decision)
    errors = []
    if unknown:
        errors.append(f"{prefix}.calibration has unknown fields: {','.join(sorted(unknown))}")
    if missing:
        errors.append(f"{prefix}.calibration is missing fields: {','.join(sorted(missing))}")
    if "receipt_sha256" in decision and not _valid_digest(decision.get("receipt_sha256")):
        errors.append(f"{prefix}.calibration.receipt_sha256 is invalid")
    if "rule" in decision and decision.get("rule") not in CALIBRATION_RULES:
        errors.append(f"{prefix}.calibration.rule is not supported")
    return errors


def _threshold_calibration_value_errors(
    threshold: Mapping[str, Any], decision: Mapping[str, Any], prefix: str
) -> List[str]:
    """Validate a calibration observation and its selected bound."""

    observation, derived, rule = (
        decision.get("observation"), decision.get("derived_value"), decision.get("rule")
    )
    if threshold.get("unit") == "boolean":
        return [] if observation is True and rule == "boolean_true" and derived is True and threshold.get("value") is True else [
            f"{prefix}.calibration boolean decision is invalid"
        ]
    errors = _threshold_calibration_rule_errors(threshold, rule, prefix)
    if not _finite_positive_or_zero(observation) or not _finite_positive_or_zero(derived):
        errors.append(f"{prefix}.calibration numeric decision is invalid")
    elif not _calibration_bound_matches(observation, derived, rule):
        errors.append(f"{prefix}.calibration derived value does not match its rule")
    if derived != threshold.get("value"):
        errors.append(f"{prefix}.calibration derived value differs from threshold")
    return errors


def _threshold_calibration_rule_errors(
    threshold: Mapping[str, Any], rule: Any, prefix: str
) -> List[str]:
    """Require calibration direction to match the metric bound direction."""

    allowed = CALIBRATION_RULES_BY_METRIC.get(threshold.get("metric"))
    return [] if allowed is None or rule in allowed else [
        f"{prefix}.calibration rule does not match metric direction"
    ]


def _finite_positive_or_zero(value: Any) -> bool:
    """Return whether a numeric calibration value is finite and nonnegative."""

    return _finite_nonnegative(value)


def _calibration_bound_matches(observation: Any, derived: Any, rule: Any) -> bool:
    """Check one fixed calibration selection rule without caller slack."""

    if not _finite_positive_or_zero(observation) or not _finite_positive_or_zero(derived):
        return False
    if rule == "identity":
        expected = float(observation)
    elif rule == "upper_10_percent":
        expected = float(observation) * 1.1
    elif rule == "lower_10_percent":
        expected = float(observation) * 0.9
    else:
        return False
    return math.isclose(float(derived), expected, rel_tol=1e-12, abs_tol=1e-12)


def _threshold_decision_digest(
    threshold: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    """Hash threshold scope and the selected calibration decision."""

    scope = {
        field: threshold.get(field)
        for field in (
            "id", "engine", "role", "scenario", "metric", "model",
            "backend", "device", "operator", "unit",
        )
    }
    material = {**scope, **{field: decision.get(field) for field in CALIBRATION_DECISION_FIELDS if field != "decision_sha256"}}
    return sha256_bytes(canonical_json(material))


def _threshold_metric_scenario_errors(threshold: Mapping[str, Any], prefix: str) -> List[str]:
    """Require scenario labels that match metric semantics."""

    expected = {
        "backpressure_observed": "slow_reader",
        "schedule_overlap_observed": "schedule",
        "physical_memory_peak_bytes": "schedule",
        "fork_latency_p95_ms": "branch",
        "cancel_latency_p95_ms": "cancel",
    }.get(threshold["metric"])
    return [] if expected is None or threshold.get("scenario") == expected else [
        f"{prefix}.scenario does not match metric"
    ]


def _evaluation_quality_errors(evaluation: Mapping[str, Any], phase: Any) -> List[str]:
    """Require an explicit quality state and, before and after freezing, the declared policy."""

    quality = evaluation.get("quality")
    errors = [] if isinstance(quality, str) and quality else ["evaluation.quality must be explicit"]
    if phase in {"pending", "frozen"} and evaluation.get("quality_policy") != QUALITY_POLICY:
        errors.append(f"evaluation.quality_policy must be the declared {QUALITY_POLICY['name']} policy")
    return errors


def _evaluation_receipt_errors(evaluation: Mapping[str, Any]) -> List[str]:
    """Validate the optional evaluation quality record reference."""

    return _evaluation_reference_shape_errors(evaluation.get("quality_record"), "quality_record")


def _evaluation_reference_shape_errors(reference: Any, name: str) -> List[str]:
    """Validate one root-contained receipt reference when it is present."""

    if reference is None:
        return []
    if not isinstance(reference, Mapping):
        return [f"evaluation.{name} needs path and sha256"]
    path_error = _manifest_path_error(reference.get("path"), f"evaluation.{name}.path")
    if path_error:
        return [path_error]
    return [] if _valid_digest(reference.get("sha256")) else [f"evaluation.{name} needs path and sha256"]


def _validate_frozen_requirements(manifest: Mapping[str, Any]) -> List[str]:
    """Require a calibration receipt and disjoint workload before freezing."""

    errors = _frozen_calibration_reference_errors(manifest)
    errors.extend(_freeze_criteria_errors(manifest.get("freeze_criteria")))
    errors.extend(_frozen_quality_requirements(manifest))
    errors.extend(_freeze_engine_identity_errors(manifest.get("engines", [])))
    errors.extend(_frozen_history_tokenization_errors(manifest.get("engines", [])))
    errors.extend(_frozen_threshold_scope_errors(manifest))
    return errors


def _frozen_history_tokenization_errors(engines: Any) -> List[str]:
    """Require independent tokenizer pins in every frozen engine row."""

    fields = (
        "vocab_size", "tokenizer_metadata_sha256", "special_tokens_policy_sha256",
        "producer_source_commit", "producer_executable_sha256", "producer_model_sha256",
        "producer_loaded_library_sha256", "template_config_sha256", "template_bytes_sha256",
    )
    errors = []
    for engine in engines if isinstance(engines, list) else []:
        if not isinstance(engine, Mapping):
            continue
        declaration = engine.get("history_tokenization")
        if not isinstance(declaration, Mapping):
            continue
        errors.extend(
            f"frozen {engine.get('id')} history_tokenization pin is missing: {field}"
            for field in fields
            if field not in declaration
        )
    return errors


def _frozen_calibration_reference_errors(manifest: Mapping[str, Any]) -> List[str]:
    """Require the root-contained calibration receipt link."""

    reference = manifest.get("calibration_receipt")
    path_error = _manifest_path_error(
        reference.get("path") if isinstance(reference, dict) else None,
        "frozen calibration_receipt.path",
    )
    if path_error or not isinstance(reference, dict) or not _valid_digest(reference.get("sha256")):
        return ["frozen manifests require calibration_receipt path and sha256"]
    evaluation = manifest.get("evaluation")
    digest = evaluation.get("calibration_receipt_sha256") if isinstance(evaluation, Mapping) else None
    return [] if digest == reference.get("sha256") else [
        "frozen evaluation calibration digest differs from calibration receipt"
    ]


def _frozen_quality_requirements(manifest: Mapping[str, Any]) -> List[str]:
    """Require observed quality and its comparison record."""

    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return ["frozen manifests require observed quality"]
    errors = _quality_declaration_errors(manifest)
    if evaluation.get("quality") in {"unverified", "pending"}:
        errors.append("frozen manifests require observed quality")
    if not isinstance(evaluation.get("quality_record"), dict):
        errors.append("frozen manifests require quality_record path and sha256")
    if not _valid_digest(evaluation.get("calibration_receipt_sha256")):
        errors.append("frozen manifests require evaluation.calibration_receipt_sha256")
    return errors


def _quality_declaration_errors(manifest: Mapping[str, Any]) -> List[str]:
    """Require one comparison backend and one quality label per comparison engine."""

    engines = [engine for engine in manifest.get("engines", []) if isinstance(engine, Mapping)]
    errors = []
    backends = {engine.get("backend") for engine in engines}
    if len(backends) != 1 or not backends <= set(QUALITY_RECORD_SCHEMAS):
        errors.append("frozen engines must share one quality backend: cuda or metal")
    labels = [engine.get("quality_label") for engine in engines]
    if sorted(map(str, labels)) != sorted(QUALITY_PRODUCERS):
        errors.append("frozen engines must declare quality labels leone and llama_cpp once each")
    return errors


def _quality_engine_declaration_errors(engine: Mapping[str, Any]) -> List[str]:
    """Check one engine's quality label, producer kind, and engine kind."""

    label, name = engine.get("quality_label"), engine.get("id")
    if label not in QUALITY_PRODUCERS:
        return [f"{name}: quality_label must be leone or llama_cpp"]
    errors = []
    if engine.get("quality_producer") != QUALITY_PRODUCERS[label]:
        errors.append(f"{name}: quality_producer must be {QUALITY_PRODUCERS[label]}")
    if (label == "leone") != (engine.get("kind") == "leone"):
        errors.append(f"{name}: quality_label differs from the engine kind")
    return errors


def _quality_record_reference_errors(reference: Any, prefix: str) -> List[str]:
    """Require a root-contained path and a SHA-256 for a comparison record."""

    path = reference.get("path") if isinstance(reference, Mapping) else None
    error = _manifest_path_error(path, f"{prefix}.path")
    if error:
        return [error]
    return [] if _valid_digest(reference.get("sha256")) else [f"{prefix} needs path and sha256"]


def _finite_nonnegative(value: Any) -> bool:
    """Return whether a scalar is finite and nonnegative."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0


def _frozen_threshold_scope_errors(manifest: Mapping[str, Any]) -> List[str]:
    """Require the complete measured threshold matrix for every engine."""

    evaluation = manifest.get("evaluation")
    thresholds = evaluation.get("thresholds") if isinstance(evaluation, dict) else None
    engines = manifest.get("engines")
    if not isinstance(thresholds, list) or not isinstance(engines, list):
        return []
    declared = {}
    for item in thresholds:
        if not isinstance(item, dict):
            continue
        key = (item.get("engine"), item.get("role"), item.get("scenario"))
        declared.setdefault(key, set()).add(item.get("metric"))
    errors = []
    for index, threshold in enumerate(thresholds):
        if isinstance(threshold, Mapping):
            errors.extend(_threshold_calibration_shape_errors(threshold, f"evaluation.thresholds[{index}]"))
    for engine in engines:
        error = _frozen_engine_threshold_error(engine, declared)
        if error:
            errors.append(error)
    return errors


def _frozen_engine_threshold_error(engine: Any, declared: Mapping[Any, set]) -> Optional[str]:
    """Return missing matrix entries for one engine."""

    engine_id = engine.get("id") if isinstance(engine, dict) else None
    unsupported = _unsupported_metrics(engine_kind(engine) if isinstance(engine, dict) else None)
    missing = [
        f"{engine_id}/{role}/{scenario}/{metric}"
        for role, scenario in sorted(REQUIRED_THRESHOLD_SCOPES)
        for metric in sorted(
            REQUIRED_THRESHOLD_METRICS[(role, scenario)] - unsupported
            - declared.get((engine_id, role, scenario), set())
        )
    ]
    waived = [
        f"{engine_id}/{role}/{scenario}/{metric}"
        for (owner, role, scenario), metrics in sorted(declared.items(), key=str)
        if owner == engine_id
        for metric in sorted(metrics & unsupported)
    ]
    if waived:
        return "frozen thresholds name unsupported metrics: " + ",".join(waived)
    return "frozen thresholds miss scopes: " + ",".join(missing) if missing else None


def _freeze_criteria_errors(criteria: Any) -> List[str]:
    """Require every release criterion to be enabled explicitly."""

    required = {
        "disjoint_workload",
        "required_thresholds",
        "quality_receipt_required",
        "calibration_receipt_validated",
        "thresholds_evaluated",
        "quality_artifact_binding",
        "running_identity_binding",
        "required_outcomes",
        "token_history_complete",
        "schedule_overlap",
        "backpressure_evidence",
        "sibling_progress",
    }
    if not isinstance(criteria, dict):
        return ["frozen manifests require freeze criteria"]
    errors = []
    unknown = set(criteria) - required
    if unknown:
        errors.append(f"frozen freeze criteria has unknown fields: {','.join(sorted(unknown))}")
    missing = required - set(criteria)
    if missing:
        errors.append(f"frozen manifests require criteria: {','.join(sorted(missing))}")
    errors.extend(
        f"frozen freeze criterion is not enabled: {criterion}"
        for criterion in required & set(criteria)
        if criteria[criterion] is not True
    )
    return errors


def _freeze_engine_identity_errors(engines: Any) -> List[str]:
    """Require running identity and model artifact declarations per engine."""

    errors = []
    for engine in engines if isinstance(engines, list) else []:
        if not isinstance(engine, dict):
            continue
        if not isinstance(engine.get("identity_endpoint"), str) and engine.get("identity_provider") not in IDENTITY_PROVIDERS:
            errors.append(f"frozen engine requires identity_endpoint: {engine.get('id')}")
        if "cancel_latency_p95_ms" not in _unsupported_metrics(engine_kind(engine)) and not isinstance(
            engine.get("cancel_observation_endpoint"), str
        ):
            errors.append(f"frozen engine requires cancel_observation_endpoint: {engine.get('id')}")
        if not isinstance(engine.get("model_artifact"), str):
            errors.append(f"frozen engine requires model_artifact: {engine.get('id')}")
    return errors


def _validate_engines(engines: Any) -> List[str]:
    if not isinstance(engines, list) or len(engines) < 2:
        return ["engines must contain at least two entries"]
    errors = _engine_entry_errors(engines)
    for engine in engines:
        if isinstance(engine, dict):
            errors.extend(_validate_engine(engine))
    ids = [engine.get("id") for engine in engines if isinstance(engine, dict)]
    if len(ids) != len(set(ids)):
        errors.append("engine IDs must be unique")
    return errors


def _engine_entry_errors(engines: Sequence[Any]) -> List[str]:
    """Report non-object engine entries before field validation."""

    return ["every engine entry must be an object" for engine in engines if not isinstance(engine, dict)]


def _validate_engine(engine: Mapping[str, Any]) -> List[str]:
    """Validate one endpoint, build declaration, and branch method."""

    engine_id = engine.get("id")
    errors = _engine_identity_errors(engine, engine_id)
    errors.extend(_engine_target_errors(engine, engine_id))
    errors.extend(_engine_endpoint_errors(engine, engine_id))
    errors.extend(_engine_method_errors(engine, engine_id))
    errors.extend(_quality_engine_declaration_errors(engine))
    return errors


def _engine_target_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require model, backend, and device identity in every engine row."""

    errors = [
        f"{engine_id}: {field} is required"
        for field in ("model_artifact", "backend", "device")
        if not isinstance(engine.get(field), str) or not engine[field]
    ]
    path_error = _manifest_path_error(engine.get("model_artifact"), f"{engine_id}: model_artifact")
    if path_error:
        errors.append(path_error)
    return errors


def _engine_identity_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate executable and build identity declarations."""

    errors = []
    if not isinstance(engine_id, str) or not isinstance(engine.get("kind"), str):
        errors.append("engine needs id and kind")
    path_error = _manifest_path_error(engine.get("executable_path"), f"{engine_id}: executable_path")
    if path_error:
        errors.append(path_error)
    command = engine.get("build_info_command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        errors.append(f"{engine_id}: build_info_command is required")
    errors.extend(_engine_source_identity_errors(engine, engine_id))
    return errors


def _engine_source_identity_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Reject source identities that bypass the build command."""

    if "source_id" in engine or "source_id_hint" in engine:
        return [f"{engine_id}: source identity must come from build_info_command"]
    return []


def _engine_endpoint_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate endpoint and optional metrics path."""

    errors = []
    if not isinstance(engine.get("base_url"), str):
        errors.append(f"{engine_id}: base_url is required")
    elif not _valid_base_url(engine["base_url"]):
        errors.append(f"{engine_id}: base_url must be an HTTP URL")
    errors.extend(_validate_metrics_endpoint(engine, engine_id))
    for field in ("identity_endpoint", "cancel_observation_endpoint"):
        if field in engine and (not isinstance(engine[field], str) or not engine[field]):
            errors.append(f"{engine_id}: {field} must be a nonempty string")
    return errors


def _engine_method_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate branch, cache, and session request declarations."""

    errors = _branch_method_errors(engine.get("branch_method"), engine_id)
    errors.extend(_cache_method_errors(engine.get("cache_method"), engine_id))
    errors.extend(_slot_copy_declaration_errors(engine, engine_id))
    errors.extend(_comparison_support_errors(engine, engine_id))
    errors.extend(_thinking_policy_errors(engine, engine_id))
    errors.extend(_history_tokenization_declaration_errors(engine, engine_id))
    if "session_field" in engine and engine["session_field"] is not None and not isinstance(engine["session_field"], str):
        errors.append(f"{engine_id}: session_field must be a string or null")
    return errors


def _history_tokenization_declaration_errors(
    engine: Mapping[str, Any], engine_id: Any
) -> List[str]:
    """Require an independent pinned llama.cpp tokenizer declaration."""

    declaration = engine.get("history_tokenization")
    if not isinstance(declaration, Mapping):
        return [f"{engine_id}: history_tokenization producer declaration is required"]
    return (
        _history_tokenization_declaration_shape_errors(declaration, engine_id)
        + _history_tokenization_declaration_path_errors(declaration, engine_id)
        + _history_tokenization_declaration_identity_errors(declaration, engine_id)
        + _history_tokenization_declaration_pin_errors(declaration, engine_id)
    )


def _history_tokenization_declaration_shape_errors(
    declaration: Mapping[str, Any], engine_id: Any
) -> List[str]:
    """Require the producer kind and its required string fields."""

    required = (
        "producer_engine", "producer_executable_path", "producer_model_artifact",
        "producer_template_file", "producer_source_commit", "trusted_public_key_source",
    )
    errors = [
        f"{engine_id}: history_tokenization.{field} is required"
        for field in required
        if not isinstance(declaration.get(field), str) or not declaration[field]
    ]
    if declaration.get("producer_engine") != "llama.cpp":
        errors.append(f"{engine_id}: history_tokenization producer_engine must be llama.cpp")
    return errors


def _history_tokenization_declaration_path_errors(
    declaration: Mapping[str, Any], engine_id: Any
) -> List[str]:
    """Require producer paths to remain relative to the study root."""

    errors = []
    for field in ("producer_executable_path", "producer_model_artifact", "producer_template_file"):
        path_error = _manifest_path_error(
            declaration.get(field), f"{engine_id}: history_tokenization.{field}"
        )
        if path_error:
            errors.append(path_error)
    return errors


def _history_tokenization_declaration_identity_errors(
    declaration: Mapping[str, Any], engine_id: Any
) -> List[str]:
    """Require the producer source and the exact legacy template selection."""

    errors = []
    if not _valid_commit(declaration.get("producer_source_commit")):
        errors.append(f"{engine_id}: history_tokenization producer source commit is invalid")
    if declaration.get("template_mode") != "legacy" or declaration.get("template_file") != LEGACY_CHATML_TEMPLATE:
        errors.append(f"{engine_id}: history_tokenization must select legacy template {LEGACY_CHATML_TEMPLATE}")
    if declaration.get("producer_template_file") != declaration.get("template_file"):
        errors.append(f"{engine_id}: history tokenizer template paths differ")
    if not isinstance(declaration.get("special_tokens_policy"), Mapping):
        errors.append(f"{engine_id}: history_tokenization special_tokens_policy is required")
    return errors


def _history_tokenization_declaration_pin_errors(
    declaration: Mapping[str, Any], engine_id: Any
) -> List[str]:
    """Validate optional producer and tokenizer pins."""

    errors = []
    for field in (
        "template_config_sha256", "template_bytes_sha256", "special_tokens_policy_sha256",
        "tokenizer_metadata_sha256", "producer_executable_sha256", "producer_model_sha256",
        "producer_loaded_library_sha256",
    ):
        if field in declaration and not _valid_digest(declaration[field]):
            errors.append(f"{engine_id}: history_tokenization {field} is invalid")
    vocabulary = declaration.get("vocab_size")
    if vocabulary is not None and (
        not isinstance(vocabulary, int) or isinstance(vocabulary, bool) or vocabulary <= 0
    ):
        errors.append(f"{engine_id}: history_tokenization vocab_size is invalid")
    return errors


IDENTITY_PROVIDERS = {"local_process", "spawned_process"}
SLOT_COPY_DECLARATION = {
    "status": "required",
    "server_flag": "--slot-save-path",
    "save_endpoint": "/slots/{id_slot}?action=save",
    "restore_endpoint": "/slots/{id_slot}?action=restore",
    "restored_roles": ["branch", "cancel"],
    "setup_accounting": "ready_and_end_to_end_reported",
}


def _slot_copy_declaration_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require the parent slot copy setup for llama.cpp and reject it elsewhere."""

    declared, provider = engine.get("slot_copy"), engine.get("identity_provider")
    if not _is_llama_cpp(engine):
        return [f"{engine_id}: slot_copy and identity_provider apply to llama.cpp only"] if (
            declared is not None or provider is not None
        ) else []
    errors = []
    if declared != SLOT_COPY_DECLARATION:
        errors.append(f"{engine_id}: llama.cpp needs the slot_copy declaration")
    if provider not in (None, *IDENTITY_PROVIDERS):
        errors.append(f"{engine_id}: identity_provider must be one of {sorted(IDENTITY_PROVIDERS)}")
    errors.extend(_spawn_declaration_errors(engine, engine_id))
    return errors


def _spawn_declaration_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require a launch declaration for `spawned_process` and reject one elsewhere."""

    spawn = engine.get("spawn")
    if engine.get("identity_provider") != "spawned_process":
        return [f"{engine_id}: spawn needs identity_provider spawned_process"] if spawn is not None else []
    if not isinstance(spawn, Mapping):
        return [f"{engine_id}: spawn needs an argv string list"]
    return _spawn_argv_errors(spawn, engine_id, engine)


def _spawn_argv_errors(spawn: Mapping[str, Any], engine_id: Any, engine: Mapping[str, Any]) -> List[str]:
    """Check the launch argv placeholders, the alias, and the startup timeout."""

    argv, timeout = spawn.get("argv"), spawn.get("startup_timeout_s")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return [f"{engine_id}: spawn needs an argv string list"]
    errors = [
        f"{engine_id}: spawn argv needs the {placeholder} placeholder"
        for placeholder in ("{executable}", "{model}", "{slot_save_path}")
        if not any(placeholder in item for item in argv)
    ]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        errors.append(f"{engine_id}: spawn needs a positive startup_timeout_s")
    return errors + _spawn_alias_errors(argv, engine_id, engine)


def _model_alias(engine: Mapping[str, Any]) -> Optional[str]:
    """The served model name: the model artifact basename, so wire records carry no directory."""

    artifact = engine.get("model_artifact")
    return Path(artifact).name if isinstance(artifact, str) and artifact else None


def _spawn_alias_errors(argv: Sequence[str], engine_id: Any, engine: Mapping[str, Any]) -> List[str]:
    """Require `--alias` to equal the model artifact basename."""

    alias = argv_flags(argv)["alias"]
    if alias is None or alias != _model_alias(engine):
        return [f"{engine_id}: spawn argv needs --alias equal to the model artifact basename"]
    return []


THINKING_POLICY_FIELDS = {"mode", "reasoning_format", "chat_template_kwargs", "generation_prompt_suffix"}
THINKING_POLICY_MODE = "explicit_legacy_chatml"
LEGACY_CHATML_TEMPLATE = "fixtures/qwen3-legacy-chatml.jinja"
LEGACY_CHATML_SERVER_FLAGS = (
    ("--chat-template-file", LEGACY_CHATML_TEMPLATE), ("--reasoning", "off"), ("--reasoning-format", "none"),
)


def _thinking_policy_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require a pinned thinking policy for llama.cpp and reject it elsewhere.

    The policy selects one plain ChatML template file, the bytes Leone renders
    for Qwen3. The embedded Qwen template reuses only part of the history
    prefix, so the fresh tokenizer rejects it. `reasoning_format: none` keeps
    reasoning text in `content`, where the client counts first content and
    completion tokens the same way for both engines. The suffix is the
    generation prompt the other engine renders for the same turns.
    """

    policy = engine.get("thinking_policy")
    if not _is_llama_cpp(engine):
        return [f"{engine_id}: thinking_policy applies to llama.cpp only"] if policy is not None else []
    if not isinstance(policy, Mapping) or set(policy) != THINKING_POLICY_FIELDS:
        return [f"{engine_id}: llama.cpp needs a thinking_policy with {sorted(THINKING_POLICY_FIELDS)}"]
    errors = []
    if policy["mode"] != THINKING_POLICY_MODE or policy["reasoning_format"] != "none":
        errors.append(f"{engine_id}: thinking_policy needs mode {THINKING_POLICY_MODE} and reasoning_format none")
    if not isinstance(policy["chat_template_kwargs"], Mapping):
        errors.append(f"{engine_id}: thinking_policy chat_template_kwargs must be an object")
    if not isinstance(policy["generation_prompt_suffix"], str) or not policy["generation_prompt_suffix"]:
        errors.append(f"{engine_id}: thinking_policy generation_prompt_suffix must be a nonempty string")
    return errors + _legacy_template_errors(engine, engine_id)


def _legacy_template_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require the tokenizer template and the launch argv to name the same legacy template file."""

    errors = []
    declaration = engine.get("history_tokenization")
    if isinstance(declaration, Mapping) and (
        declaration.get("template_mode") != "legacy" or declaration.get("template_file") != LEGACY_CHATML_TEMPLATE
    ):
        errors.append(f"{engine_id}: history_tokenization must select legacy template {LEGACY_CHATML_TEMPLATE}")
    return errors + _legacy_spawn_flag_errors(engine, engine_id)


def _legacy_spawn_flag_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require the declared launch argv to serve the legacy template with reasoning off."""

    spawn = engine.get("spawn")
    argv = spawn.get("argv") if isinstance(spawn, Mapping) else None
    if not isinstance(argv, list):
        return []
    pairs = set(zip(argv, argv[1:]))
    if "--jinja" in argv and all(flag in pairs for flag in LEGACY_CHATML_SERVER_FLAGS):
        return []
    wanted = ", ".join(" ".join(flag) for flag in LEGACY_CHATML_SERVER_FLAGS)
    return [f"{engine_id}: spawn argv must pass --jinja, {wanted}"]


def _comparison_support_errors(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Require the declared support table to equal the fixed evidence contract."""

    expected = capability_record(engine_kind(engine))
    if expected is None:
        return [f"{engine_id}: engine kind has no evidence contract"]
    if engine.get("comparison_support") != expected:
        return [f"{engine_id}: comparison_support differs from the fixed evidence contract"]
    return []


def _branch_method_errors(method: Any, engine_id: Any) -> List[str]:
    """Validate fork request fields when the method is declared."""

    if not isinstance(method, dict) or not isinstance(method.get("name"), str):
        return [f"{engine_id}: branch_method is incomplete"]
    if method.get("status") != "unsupported" and (
        not isinstance(method.get("parent_field"), str)
        or not isinstance(method.get("session_field"), str)
    ):
        return [f"{engine_id}: supported branch_method needs parent and session fields"]
    return _telemetry_declaration_errors(method, engine_id)


def _telemetry_declaration_errors(method: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate optional token and engine clock declarations."""

    errors = []
    boundary = method.get("token_boundary")
    if boundary is not None and not _boundary_declaration_valid(boundary):
        errors.append(f"{engine_id}: token_boundary needs a field")
    timing = method.get("engine_timing")
    if timing is not None and not _timing_declaration_valid(timing):
        errors.append(f"{engine_id}: engine_timing needs field and clock")
    return errors


def _boundary_declaration_valid(boundary: Any) -> bool:
    """Return whether a token boundary identifies one field."""

    return isinstance(boundary, dict) and isinstance(boundary.get("field"), str) and bool(boundary["field"])


def _timing_declaration_valid(timing: Any) -> bool:
    """Return whether engine timing names a field and clock."""

    return (
        isinstance(timing, dict)
        and isinstance(timing.get("field"), str)
        and isinstance(timing.get("clock"), str)
        and bool(timing["clock"])
    )


def _cache_method_errors(cache: Any, engine_id: Any) -> List[str]:
    """Validate optional same-history cache fields."""

    if cache is None:
        return []
    if not isinstance(cache, dict) or cache.get("status") != "supported" or not isinstance(cache.get("name"), str):
        return [f"{engine_id}: cache_method is incomplete"]
    errors = []
    if cache.get("request_field") is not None and not isinstance(cache.get("request_field"), str):
        errors.append(f"{engine_id}: cache_method.request_field must be a string")
    if cache.get("reuse_count_field") is not None and not isinstance(cache.get("reuse_count_field"), str):
        errors.append(f"{engine_id}: cache_method.reuse_count_field must be a string")
    errors.extend(_cache_path_errors(cache, engine_id))
    return errors


def _cache_path_errors(cache: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate an optional nested cache reuse path."""

    path = cache.get("reuse_count_path")
    if path is not None and (not isinstance(path, list) or not all(isinstance(part, str) for part in path)):
        return [f"{engine_id}: cache_method.reuse_count_path must be a string list"]
    return []


def _validate_metrics_endpoint(engine: Mapping[str, Any], engine_id: Any) -> List[str]:
    """Validate an optional structured metrics endpoint."""

    if "metrics_endpoint" in engine and not isinstance(engine["metrics_endpoint"], str):
        return [f"{engine_id}: metrics_endpoint must be a string"]
    if "trace_endpoint" in engine and (
        not isinstance(engine["trace_endpoint"], str) or not engine["trace_endpoint"]
    ):
        return [f"{engine_id}: trace_endpoint must be a nonempty string"]
    return []


def _valid_base_url(value: str) -> bool:
    """Check the endpoint form before a run starts."""

    parsed = urllib.parse.urlparse(value)
    return parsed.scheme == "http" and parsed.hostname is not None


def _validate_request(request: Any) -> List[str]:
    if not isinstance(request, dict):
        return ["request must be an object"]
    errors = []
    errors.extend(_validate_request_names(request))
    for validator in (_validate_max_tokens, _validate_seed):
        error = validator(request)
        if error:
            errors.append(error)
    if not isinstance(request.get("temperature"), (int, float)) or request["temperature"] < 0:
        errors.append("request.temperature must be nonnegative")
    return errors


def _validate_request_names(request: Mapping[str, Any]) -> List[str]:
    """Validate request model and session field names."""

    return [
        f"request.{field} is required"
        for field in ("model", "session_field")
        if not isinstance(request.get(field), str) or not request[field]
    ]


def _validate_max_tokens(request: Mapping[str, Any]) -> Optional[str]:
    """Validate the positive output token bound."""

    return None if isinstance(request.get("max_tokens"), int) and request["max_tokens"] > 0 else "request.max_tokens must be a positive integer"


def _validate_seed(request: Mapping[str, Any]) -> Optional[str]:
    """Validate the nonnegative sampler seed."""

    return None if isinstance(request.get("seed"), int) and request["seed"] >= 0 else "request.seed must be a nonnegative integer"


def _engine_rows(
    manifest: Mapping[str, Any],
    root: Optional[Path] = None,
    before: Optional[Sequence[Mapping[str, Any]]] = None,
    runs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Record declarations and, when running, actual build provenance."""

    return [_engine_row_from_run(item, root, before, runs) for item in manifest["engines"]]


def _engine_row_from_run(
    item: Mapping[str, Any], root: Optional[Path], before: Optional[Sequence[Mapping[str, Any]]],
    runs: Optional[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """Build one engine provenance row."""

    row = _engine_declaration_row(item)
    row["provenance"] = engine_provenance(item, root) if root is not None else {
        "status": "pending", "reason": "plan_does_not_execute_build"
    }
    if before is not None:
        previous = next((candidate for candidate in before if candidate.get("id") == item["id"]), None)
        if isinstance(previous, Mapping):
            row["provenance_before"] = previous.get("provenance")
    if runs is not None:
        _attach_running_identities(row, item["id"], runs)
    return row


def _engine_declaration_row(item: Mapping[str, Any]) -> Dict[str, Any]:
    """Copy the engine declaration fields retained in a receipt."""

    return {
        "id": item["id"], "kind": item["kind"], "base_url": item["base_url"],
        "branch_method": item["branch_method"], "cache_method": item.get("cache_method"),
        "metrics_endpoint": item.get("metrics_endpoint"), "trace_endpoint": item.get("trace_endpoint"),
        "identity_endpoint": item.get("identity_endpoint"),
        "identity_provider": item.get("identity_provider"), "slot_copy": item.get("slot_copy"),
        "comparison_support": item.get("comparison_support"),
        "thinking_policy": item.get("thinking_policy"), "spawn": item.get("spawn"),
        "history_tokenization": item.get("history_tokenization"),
        "cancel_observation_endpoint": item.get("cancel_observation_endpoint"),
        "model_artifact": item.get("model_artifact"), "model": item.get("model"),
        "backend": item.get("backend"), "device": item.get("device"),
        "quality_label": item.get("quality_label"), "quality_producer": item.get("quality_producer"),
        "executable_path": item["executable_path"], "build_info_command": item["build_info_command"],
    }


def _attach_running_identities(
    row: Dict[str, Any], engine_id: Any, runs: Sequence[Mapping[str, Any]]
) -> None:
    """Retain every run identity while keeping the first-row compatibility field."""

    identities = [
        run.get("running_identity")
        for run in runs
        if run.get("engine") == engine_id and isinstance(run.get("running_identity"), dict)
    ]
    if identities:
        row["running_identity"] = identities[0]
        row["running_identities"] = identities


def build_plan(manifest: Mapping[str, Any], manifest_path: Path) -> Dict[str, Any]:
    """Build a reviewable plan without starting servers or measuring output."""

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": manifest["phase"],
        "manifest": {"path": str(manifest_path), "canonical_sha256": sha256_bytes(canonical_json(manifest))},
        "prompts": prompt_provenance(manifest),
        "engines": _engine_rows(manifest),
        "budgets": manifest["budgets"],
        "schedule": manifest["schedule"],
        "evaluation": manifest.get("evaluation", {}),
    }


def _all_records(receipt: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    records = []
    for run in receipt.get("runs", []):
        if not isinstance(run, dict):
            continue
        parent = run.get("parent") if isinstance(run.get("parent"), dict) else {}
        records.append({"status": parent.get("status"), "metrics": parent.get("metrics", {})})
        branches = run.get("branches") if isinstance(run.get("branches"), list) else []
        probes = run.get("probes") if isinstance(run.get("probes"), list) else []
        records.extend(
            {"status": item.get("status"), "metrics": item.get("metrics", {})}
            for item in [*branches, *probes]
            if isinstance(item, dict)
        )
    return records


def _records_by_engine_role(receipt: Mapping[str, Any]) -> Dict[Tuple[str, str], List[Mapping[str, Any]]]:
    """Group retained request records by engine and schedule role."""

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for run in receipt.get("runs", []):
        if not isinstance(run, dict):
            continue
        engine = run.get("engine")
        if not isinstance(engine, str):
            continue
        entries = [("parent", run.get("parent"))]
        entries.extend(("branch", item) for item in run.get("branches", []))
        entries.extend((item.get("role"), item) for item in run.get("probes", []) if isinstance(item, dict))
        for role, item in entries:
            if not isinstance(role, str) or not isinstance(item, dict):
                continue
            grouped.setdefault((engine, role), []).append(
                {"status": item.get("status"), "metrics": item.get("metrics", {}), **item}
            )
    return grouped


def summary_by_engine_role(
    receipt: Mapping[str, Any], minimum_samples: int, max_samples: int
) -> Dict[str, Dict[str, Any]]:
    """Summarize every engine and schedule role without pooling their outcomes."""

    return {
        f"{engine}/{role}": outcome_summary(records, minimum_samples, max_samples)
        for (engine, role), records in sorted(_records_by_engine_role(receipt).items())
    }


def _engine_target(manifest: Mapping[str, Any], engine_id: str) -> Dict[str, Any]:
    """Return the declared model, backend, and device scope for one engine."""

    engine = next((item for item in manifest.get("engines", []) if item.get("id") == engine_id), {})
    return {
        "model": engine.get("model", manifest.get("request", {}).get("model")),
        "backend": engine.get("backend", engine.get("kind")),
        "device": engine.get("device"),
    }


def _measurement_values(records: Sequence[Mapping[str, Any]], field: str) -> List[float]:
    """Read finite numeric measurement values from request records."""

    values = []
    for record in records:
        service = record.get("service_metrics", {})
        measurement = service.get(field) if isinstance(service, dict) else None
        value = measurement.get("value") if isinstance(measurement, dict) else None
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
        ):
            values.append(float(value))
    return values


def _reuse_values(records: Sequence[Mapping[str, Any]]) -> List[int]:
    """Read observed history reuse counts that cover their parent prefix."""

    values = []
    for record in records:
        values.extend(_record_reuse_values(record))
    return values


def _record_reuse_values(record: Mapping[str, Any]) -> List[int]:
    """Read qualifying reuse values from one branch record."""

    reuse = record.get("history_reuse")
    if not isinstance(reuse, Mapping):
        return []
    tokenization = reuse.get("tokenization")
    minimum = (
        tokenization.get("expected_reused_token_count")
        if isinstance(tokenization, Mapping)
        else None
    )
    nested = reuse.get("reuse_count")
    candidates = nested.get("values") if isinstance(nested, Mapping) else None
    if not _valid_reuse_minimum(minimum) or not isinstance(candidates, list):
        return []
    return [value for value in candidates if _qualifying_reuse_value(value, minimum)]


def _valid_reuse_minimum(value: Any) -> bool:
    """Check one declared parent reuse floor."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _qualifying_reuse_value(value: Any, minimum: int) -> bool:
    """Check one observed reuse count against its parent floor."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _threshold_observation(
    threshold: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    all_runs: Sequence[Mapping[str, Any]],
    minimum_samples: int,
) -> Optional[Union[float, bool]]:
    """Compute one declared threshold metric from retained request evidence."""

    metric = threshold["metric"]
    record_value = _record_threshold_observation(metric, records, minimum_samples)
    if record_value is not None:
        return record_value
    return _run_threshold_observation(metric, threshold, records, all_runs, minimum_samples)


def _record_threshold_observation(
    metric: str, records: Sequence[Mapping[str, Any]], minimum_samples: int
) -> Optional[Union[float, bool]]:
    """Compute metrics derived from request records."""

    if metric == "ttft_p95_ms":
        return _quantile_metric(records, _record_ttft, minimum_samples)
    if metric == "inter_token_latency_p95_ms":
        return _itl_quantile_metric(records, minimum_samples)
    if metric in {"fork_latency_p95_ms", "cancel_latency_p95_ms"}:
        field = "fork_latency_ns" if metric.startswith("fork") else "cancel_latency_ns"
        status = "completed" if metric.startswith("fork") else "cancelled"
        return _service_quantile_metric(records, field, minimum_samples, {status})
    if metric == "request_success_rate":
        return _success_rate(records, minimum_samples)
    return None


def _run_threshold_observation(
    metric: str, threshold: Mapping[str, Any], records: Sequence[Mapping[str, Any]],
    all_runs: Sequence[Mapping[str, Any]], minimum_samples: int,
) -> Optional[Union[float, bool]]:
    """Compute metrics derived from cache, transport, or schedule evidence."""

    if metric == "history_reuse_min_tokens":
        qualified = _qualified_records(records, minimum_samples)
        values = _reuse_values(qualified) if qualified else []
        return min(values) if len(values) >= minimum_samples else None
    if metric == "backpressure_observed":
        qualified = _qualified_records(records, minimum_samples)
        return _observed_backpressure(qualified, minimum_samples) if qualified else None
    if metric == "physical_memory_peak_bytes":
        return _physical_memory_peak_observation(threshold, all_runs, minimum_samples)
    if metric == "schedule_overlap_observed":
        return _observed_overlap(threshold, all_runs, minimum_samples)
    return None


def _physical_memory_peak_observation(
    threshold: Mapping[str, Any], all_runs: Sequence[Mapping[str, Any]], minimum_samples: int
) -> Optional[float]:
    """Return the maximum reported physical memory peak for one engine."""

    runs = [
        run
        for run in all_runs
        if isinstance(run, Mapping) and run.get("engine") == threshold.get("engine")
    ]
    if len(runs) < minimum_samples:
        return None
    peaks = [_run_physical_memory_peak(run) for run in runs]
    if any(value is None for value in peaks):
        return None
    return max(value for value in peaks if value is not None)


def _run_physical_memory_peak(run: Mapping[str, Any]) -> Optional[float]:
    """Read one finite physical memory peak from the terminal service snapshot."""

    snapshots = run.get("service_metrics")
    end = snapshots.get("end") if isinstance(snapshots, Mapping) else None
    body = end.get("body") if isinstance(end, Mapping) else None
    if not isinstance(body, Mapping):
        return None
    if not _memory_snapshot_usable(body):
        return None
    return _memory_value(body.get("physical_tracker_peak_bytes"))


def _memory_snapshot_usable(body: Mapping[str, Any]) -> bool:
    """Reject physical peaks recorded after loss, overflow, or degradation."""

    return (
        _memory_topology_valid(body)
        and _memory_peak_valid(body)
        and _memory_collection_state_valid(body)
        and not _memory_history_dropped(body)
    )


def _memory_topology_valid(body: Mapping[str, Any]) -> bool:
    """Require a supported memory ledger topology and its selected ledger label."""

    topology = body.get("memory_topology")
    ledger = body.get("physical_tracker_ledger")
    expected = {
        "discrete": "backend_memory_tracker_root",
        "unified_parent": "parent_memory_tracker_root",
        "cpu_parent": "parent_memory_tracker_root",
        "backend_child_fallback": "backend_child_tracker",
    }.get(topology)
    return expected is not None and ledger == expected


def _memory_peak_valid(body: Mapping[str, Any]) -> bool:
    """Require the selected parent peak to use the producer definition."""

    peak = body.get("physical_tracker_peak_bytes")
    return (
        body.get("physical_peak_definition") == PHYSICAL_PEAK_DEFINITION
        and isinstance(peak, Mapping)
        and set(peak).issubset({"status", "value", "reason"})
        and {"status", "value"}.issubset(peak)
        and peak.get("status") == "observed"
        and ("reason" not in peak or peak.get("reason") is None)
        and _finite_memory_scalar(peak.get("value")) is not None
    )


def _memory_collection_state_valid(body: Mapping[str, Any]) -> bool:
    """Require loss, overflow, and degradation counters to be clear."""

    flags_clear = all(
        isinstance(body.get(field), bool) and body.get(field) is False
        for field in ("counter_overflowed", "degraded", "memory_topology_conflict")
    )
    errors = body.get("collection_errors")
    losses = body.get("collection_losses")
    return flags_clear and _nonnegative_int(errors) and errors == 0 and _memory_losses_clear(losses)


def _memory_losses_clear(losses: Any) -> bool:
    """Return whether every retained collection-loss counter is zero."""

    return _memory_collection_losses_valid(losses) and not any(
        _positive_integer(value) for value in losses.values() if not isinstance(value, bool)
    )


def _memory_collection_losses_valid(value: Any) -> bool:
    """Require every producer collection loss counter and its overflow bit."""

    if not isinstance(value, Mapping) or not MEMORY_COLLECTION_LOSS_FIELDS.issubset(value):
        return False
    counters = MEMORY_COLLECTION_LOSS_FIELDS - {"overflowed"}
    return isinstance(value.get("overflowed"), bool) and value["overflowed"] is False and all(
        _nonnegative_int(value.get(field)) for field in counters
    )


def _positive_integer(value: Any) -> bool:
    """Return whether a counter records at least one lost observation."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _memory_history_dropped(body: Mapping[str, Any]) -> bool:
    """Check physical memory histories for discarded observations."""

    histories = (body.get("physical_bytes"), body.get("physical_bytes_samples"))
    present = [history for history in histories if history is not None]
    return not present or any(not _memory_history_valid(history) for history in present)


def _memory_history_valid(history: Any) -> bool:
    """Require complete, lossless physical memory sample history."""

    required = {"capacity", "sample_count", "dropped_count", "values"}
    if not isinstance(history, Mapping) or not required.issubset(history):
        return False
    capacity, count, dropped, values = (
        history.get("capacity"), history.get("sample_count"),
        history.get("dropped_count"), history.get("values"),
    )
    if not _memory_history_shape_valid(capacity, count, dropped, values):
        return False
    return all(_memory_history_value_valid(value) for value in values)


def _memory_history_value_valid(value: Any) -> bool:
    """Validate one scalar or backend-class physical memory sample."""

    if _finite_memory_scalar(value) is not None:
        return True
    if not isinstance(value, Mapping) or set(value) != {"at_ns", "bytes_by_class"}:
        return False
    timestamp = value.get("at_ns")
    classes = value.get("bytes_by_class")
    return (
        _nonnegative_int(timestamp)
        and isinstance(classes, Mapping)
        and bool(classes)
        and all(
            isinstance(name, str) and bool(name) and _nonnegative_int(bytes_value)
            for name, bytes_value in classes.items()
        )
    )


def _memory_history_shape_valid(
    capacity: Any, count: Any, dropped: Any, values: Any
) -> bool:
    """Check memory sample counts and loss counters."""

    return (
        _positive_integer(capacity)
        and _nonnegative_int(count)
        and _nonnegative_int(dropped)
        and isinstance(values, list)
        and dropped == 0
        and count == len(values) + dropped
        and count <= capacity
    )


def _memory_value(value: Any) -> Optional[float]:
    """Sum one nonnegative physical memory measurement without treating absence as zero."""

    if isinstance(value, Mapping) and "value" in value:
        return _memory_measurement_value(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_memory_scalar(value)
    return _memory_class_sum(value)


def _memory_measurement_value(value: Mapping[str, Any]) -> Optional[float]:
    """Read an observed value from a serialized measurement wrapper."""

    return (
        _finite_memory_scalar(value.get("value"))
        if isinstance(value, Mapping) and value.get("status") == "observed"
        else None
    )


def _finite_memory_scalar(value: Any) -> Optional[float]:
    """Return one finite nonnegative memory scalar."""

    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
        else None
    )


def _memory_class_sum(value: Any) -> Optional[float]:
    """Sum finite nonnegative memory classes."""

    if not isinstance(value, Mapping) or not value:
        return None
    values = list(value.values())
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        return None
    return sum(float(item) for item in values) if all(math.isfinite(float(item)) and item >= 0 for item in values) else None


def _qualified_records(
    records: Sequence[Mapping[str, Any]], minimum_samples: int, statuses: Optional[set] = None
) -> List[Mapping[str, Any]]:
    """Return complete records only when the declared sample count is met."""

    accepted = statuses or {"completed"}
    qualified = [
        record
        for record in records
        if record.get("status") in accepted
        and record.get("history_complete") is True
        and not _request_interval_errors(record)
    ]
    return qualified if len(qualified) >= minimum_samples else []


def _quantile_metric(
    records: Sequence[Mapping[str, Any]], collector: Any, minimum_samples: int = 1
) -> Optional[float]:
    """Return p95 for one scalar record collector."""

    qualified = _qualified_records(records, minimum_samples)
    values = [value for record in qualified for value in [collector(record)] if value is not None]
    return linear_quantile(values, 0.95) if len(values) >= minimum_samples else None


def _itl_quantile_metric(records: Sequence[Mapping[str, Any]], minimum_samples: int = 1) -> Optional[float]:
    """Return p95 for all explicit token-boundary intervals."""

    qualified = _qualified_records(records, minimum_samples)
    values = [value for record in qualified for value in _record_itl(record)]
    return linear_quantile(values, 0.95) if len(values) >= minimum_samples else None


def _service_quantile_metric(
    records: Sequence[Mapping[str, Any]],
    field: str,
    minimum_samples: int = 1,
    statuses: Optional[set] = None,
) -> Optional[float]:
    """Return p95 for one server nanosecond measurement field."""

    qualified = _qualified_records(records, minimum_samples, statuses)
    values = _measurement_values(qualified, field)
    return linear_quantile([value / 1_000_000 for value in values], 0.95) if len(values) >= minimum_samples else None


def _success_rate(records: Sequence[Mapping[str, Any]], minimum_samples: int = 1) -> Optional[float]:
    """Return the completed request ratio for one role."""

    if len(records) < minimum_samples or any(record.get("history_complete") is not True for record in records):
        return None
    return sum(record.get("status") == "completed" for record in records) / len(records) if records else None


def _observed_backpressure(records: Sequence[Mapping[str, Any]], minimum_samples: int = 1) -> Optional[bool]:
    """Return whether a slow-reader record contains server evidence."""

    observed = sum(
        record.get("status") == "completed"
        and isinstance(record.get("backpressure"), dict)
        and record["backpressure"].get("status") == "observed"
        and isinstance(record["backpressure"].get("request_id"), str)
        for record in records
    )
    return True if observed >= minimum_samples else None


def _observed_overlap(
    threshold: Mapping[str, Any], all_runs: Sequence[Mapping[str, Any]], minimum_samples: int = 1
) -> bool:
    """Return whether every run for one engine recorded overlapping intervals."""

    rows = [
        run
        for run in all_runs
        if isinstance(run, dict) and run.get("engine") == threshold["engine"]
    ]
    if len(rows) < minimum_samples:
        return False
    return bool(rows) and all(_overlap_run_valid(run) for run in rows)


def _overlap_run_valid(run: Mapping[str, Any]) -> bool:
    """Check one complete schedule overlap claim."""

    overlap = run.get("overlap")
    return (
        isinstance(overlap, Mapping)
        and overlap.get("status") == "observed"
        and overlap.get("overlap") is True
        and overlap.get("required_overlap") is True
        and _schedule_overlap(_run_records(run)) == overlap
    )


def _compare_threshold(observed: Union[float, bool], operator: str, expected: Union[float, bool]) -> bool:
    """Compare one observed value with one typed threshold."""

    if operator == "<":
        return observed < expected
    if operator == "<=":
        return observed <= expected
    if operator == "==":
        return observed == expected
    if operator == ">=":
        return observed >= expected
    return observed > expected


def evaluate_thresholds(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate each typed frozen threshold against one engine and role."""

    thresholds = manifest.get("evaluation", {}).get("thresholds", [])
    grouped = _records_by_engine_role(receipt)
    runs = receipt.get("runs", [])
    budgets = manifest.get("budgets", {})
    minimum_samples = budgets.get("minimum_samples_for_quantiles", 1)
    results = []
    for threshold in thresholds if isinstance(thresholds, list) else []:
        key = (threshold["engine"], threshold["role"])
        records = grouped.get(key, [])
        target = _engine_target(manifest, threshold["engine"])
        scope_matches = all(target.get(field) == threshold[field] for field in ("model", "backend", "device"))
        observed = (
            _threshold_observation(threshold, records, runs, minimum_samples) if scope_matches else None
        )
        passed = observed is not None and _compare_threshold(observed, threshold["operator"], threshold["value"])
        results.append(
            {
                "id": threshold["id"],
                "status": "pass" if passed else "fail",
                "scope_matches": scope_matches,
                "observed": observed,
                "expected": threshold["value"],
                "operator": threshold["operator"],
                "metric": threshold["metric"],
                "engine": threshold["engine"],
                "role": threshold["role"],
                "reason": None if passed else _threshold_failure_reason(records, minimum_samples, observed),
            }
        )
    return results


def _threshold_failure_reason(
    records: Sequence[Mapping[str, Any]], minimum_samples: int, observed: Optional[Union[float, bool]]
) -> str:
    """Classify an unavailable threshold without converting missing data to zero."""

    if observed is not None:
        return "threshold_failed"
    if len(_qualified_records(records, minimum_samples)) < minimum_samples:
        return "insufficient_samples"
    return "measurement_unavailable"


def _threshold_evaluation_errors(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Reject frozen receipts without complete evaluated threshold results."""

    if manifest.get("phase") != "frozen":
        return []
    expected = evaluate_thresholds(receipt, manifest)
    evaluation = receipt.get("evaluation")
    actual = evaluation.get("threshold_results") if isinstance(evaluation, dict) else None
    if actual != expected:
        return ["frozen threshold results do not recompute from outcomes"]
    return [f"frozen threshold failed: {item['id']}" for item in expected if item.get("status") != "pass"]


# Offline validation checks recorded structure and recorded digests of models,
# plans, and binaries. It never reads them and never claims to re-execute
# anything. The flag is scoped to one `validate_receipt` call.
_OFFLINE: "contextvars.ContextVar[bool]" = contextvars.ContextVar("branching_offline", default=False)


def _offline() -> bool:
    """Return whether the running validation must not require local models or binaries."""

    return _OFFLINE.get()


def _file_digest_mismatch(path: Path, recorded: Any) -> bool:
    """Return whether a file differs from its recorded digest. Offline, only a malformed digest differs."""

    if _offline():
        return not _valid_digest(recorded)
    return sha256_file(path) != recorded


def _file_matches(path: Path, recorded: Any) -> bool:
    """Return whether a required file exists with its recorded digest. Offline, a well-formed digest suffices."""

    if _offline():
        return _valid_digest(recorded)
    return path.is_file() and sha256_file(path) == recorded


def validate_receipt(
    path: Path, root: Path, offline: bool = False, source_manifest: Optional[Path] = None,
    source_scope: str = "repository",
) -> List[str]:
    """Validate provenance, counts, outcomes, and recomputed summaries.

    `offline` skips reading local models, plans, and binaries and checks their
    recorded digests instead. `source_manifest` binds the receipt to the source
    inputs by content hash under `source_scope`. Neither option re-executes a
    tokenizer or a server.
    """

    token = _OFFLINE.set(offline)
    try:
        if source_scope == "archive" and source_manifest is None:
            return ["archive source scope requires --source-manifest"]
        return _validate_receipt(path, root, source_manifest, source_scope)
    finally:
        _OFFLINE.reset(token)


def _validate_receipt(
    path: Path, root: Path, source_manifest: Optional[Path], source_scope: str
) -> List[str]:
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read receipt: {error}"]
    errors = []
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION:
        return ["unexpected receipt schema_version"]
    manifest, manifest_errors = _receipt_manifest(receipt, root)
    errors.extend(manifest_errors)
    if manifest is None:
        return errors
    if manifest_errors:
        return errors
    errors.extend(_receipt_engine_errors(receipt, manifest, root))
    errors.extend(_receipt_artifact_errors(receipt, manifest, root))
    errors.extend(_receipt_run_errors(receipt, manifest))
    errors.extend(_receipt_frozen_errors(receipt, manifest, root))
    if receipt.get("prompts") != prompt_provenance(manifest):
        errors.append("prompt hashes differ from manifest")
    expected = outcome_summary(
        _all_records(receipt),
        manifest["budgets"]["minimum_samples_for_quantiles"],
        manifest["budgets"]["max_history"],
    )
    if receipt.get("summary") != expected:
        errors.append("summary does not recompute from outcomes")
    expected_roles = summary_by_engine_role(
        receipt,
        manifest["budgets"]["minimum_samples_for_quantiles"],
        manifest["budgets"]["max_history"],
    )
    if receipt.get("summary_by_engine_role") != expected_roles:
        errors.append("per-engine role summaries do not recompute from outcomes")
    errors.extend(_end_to_end_errors(receipt, manifest))
    errors.extend(_threshold_evaluation_errors(receipt, manifest))
    if source_manifest is not None:
        errors.extend(_source_binding_errors(receipt.get("source"), root, source_manifest, source_scope))
    return errors


def _git_source(root: Path) -> Dict[str, Any]:
    """Record the source commit and whether tracked files are clean, or why neither is known."""

    def git(*arguments: str) -> Optional[str]:
        try:
            done = subprocess.run(["git", *arguments], cwd=root, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None

    commit, dirty = git("rev-parse", "HEAD"), git("status", "--porcelain", "--untracked-files=no")
    if commit is None or dirty is None or not _valid_commit(commit.strip()):
        return {"status": "unavailable", "reason": "git_source_unreadable"}
    return {"status": "observed", "commit": commit.strip(), "tracked_tree_clean": not dirty.strip()}


def _source_binding_errors(
    source: Any, root: Path, relative: Path, scope: str = "repository"
) -> List[str]:
    """Bind a receipt to its source inputs: the manifest names the commit and every file hash."""

    if not isinstance(source, Mapping) or source.get("status") != "observed" or not _valid_commit(source.get("commit")):
        return ["receipt does not record an observed source commit"]
    manifest = _safe_relative_file(root, relative)
    if manifest is None:
        return ["source input manifest path is unsafe"]
    try:
        body = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        body = None
    if not isinstance(body, Mapping) or body.get("source_commit") != source["commit"]:
        return ["source inputs do not match the recorded source commit"]
    return _source_scope_errors(body, source["commit"], root, relative, scope)


def _source_scope_errors(
    body: Mapping[str, Any], commit: str, root: Path, relative: Path, scope: str
) -> List[str]:
    """Check the recorded source set with the shared checker.

    `repository` compares the whole recorded set with the tracked files. `archive`
    checks each recorded file that is present and requires the files this harness
    loads, because an archive omits `crates`.
    """

    if scope not in SOURCE_SCOPES:
        return [f"source scope is not supported: {scope}"]
    checker = _load_sibling_module("source_inputs.py")
    try:
        if scope == "archive":
            checker.check_packaged_files(root, body, ARCHIVE_REQUIRED_SOURCES)
        else:
            checker.check(root, commit, relative)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        return [f"source inputs differ from the recorded manifest: {error}"]
    return []


def _load_sibling_module(filename: str) -> Any:
    """Load a helper module that sits beside this harness."""

    if filename not in _SIBLING_MODULES:
        path = Path(__file__).resolve().parent / filename
        spec = importlib.util.spec_from_file_location(f"leone_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise StudyError(f"helper module cannot be loaded: {filename}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SIBLING_MODULES[filename] = module
    return _SIBLING_MODULES[filename]


def _safe_relative_file(root: Path, relative: Path) -> Optional[Path]:
    """Resolve a relative regular file under root without following any symlink."""

    if relative.is_absolute() or ".." in relative.parts:
        return None
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return None
    return current if current.is_file() else None


def _receipt_manifest(receipt: Mapping[str, Any], root: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Load the manifest named by a receipt and verify its canonical hash."""

    info = receipt.get("manifest", {})
    path_value = info.get("path") if isinstance(info, dict) else None
    path_error = _manifest_path_error(path_value, "receipt manifest.path")
    if path_error:
        return None, [path_error]
    try:
        path = root_path(root, path_value)
    except ValueError:
        return None, ["recorded manifest path escapes root"]
    if not path.is_file():
        return None, ["recorded manifest is missing"]
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, ["recorded manifest is not JSON"]
    if not isinstance(manifest, dict):
        return None, ["recorded manifest is not an object"]
    errors = validate_manifest(manifest)
    if info.get("canonical_sha256") != sha256_bytes(canonical_json(manifest)):
        errors.append("manifest canonical SHA-256 does not match")
    if receipt.get("phase") != manifest.get("phase"):
        errors.append("receipt phase differs from manifest")
    if receipt.get("workload_id") != manifest.get("workload_id"):
        errors.append("receipt workload_id differs from manifest")
    return manifest, errors


def _receipt_engine_errors(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], root: Path
) -> List[str]:
    """Verify actual executable provenance for every configured engine."""

    rows = receipt.get("engines", [])
    if not isinstance(rows, list):
        rows = []
    return [
        error
        for engine in manifest.get("engines", [])
        for error in [_receipt_engine_error(
            engine, rows, root, manifest.get("phase"), manifest.get("budgets", {}).get("repetitions")
        )]
        if error
    ]


def _receipt_engine_error(
    engine: Mapping[str, Any], rows: Any, root: Path, phase: Any = None, expected_count: Optional[int] = None
) -> Optional[str]:
    """Validate one engine's observed executable identity."""

    row = _engine_row(rows, engine.get("id"))
    if row is None or not isinstance(row.get("provenance"), dict):
        return f"engine provenance is missing: {engine.get('id')}"
    provenance = row["provenance"]
    declaration_error = _engine_declaration_error(row, engine)
    if declaration_error:
        return declaration_error
    status_error = _engine_status_error(provenance, engine, root, phase)
    if status_error:
        return status_error
    provenance_error = _provenance_change_error(row, provenance, engine)
    if provenance_error:
        return provenance_error
    if phase != "frozen":
        return None
    return _frozen_identity_error(row, engine, root, expected_count)


def _provenance_change_error(
    row: Mapping[str, Any], provenance: Mapping[str, Any], engine: Mapping[str, Any]
) -> Optional[str]:
    """Reject executable or build changes between the two local hashes."""

    previous = row.get("provenance_before")
    if not isinstance(previous, dict) or previous.get("status") != "observed":
        return None
    if previous.get("executable_sha256") != provenance.get("executable_sha256"):
        return f"engine executable changed during run: {engine.get('id')}"
    if previous.get("linked_libraries") != provenance.get("linked_libraries"):
        return f"engine linked libraries changed during run: {engine.get('id')}"
    before_build = previous.get("build_info", {})
    after_build = provenance.get("build_info", {})
    if before_build.get("source_id") != after_build.get("source_id"):
        return f"engine build identity changed during run: {engine.get('id')}"
    return None


def _frozen_identity_error(
    row: Mapping[str, Any], engine: Mapping[str, Any], root: Path, expected_count: Optional[int] = None
) -> Optional[str]:
    """Require every frozen run to retain a stable serving-process identity."""

    identities = row.get("running_identities")
    if not _identity_count_valid(identities, expected_count):
        return f"running server identities are incomplete: {engine.get('id')}"
    error = _identity_run_error(row, identities, engine, root)
    if error:
        return error
    if not _identities_match(identities):
        return f"running server identity changed between runs: {engine.get('id')}"
    return None


def _identity_count_valid(identities: Any, expected_count: Optional[int]) -> bool:
    """Check the retained identity list length."""

    return isinstance(identities, list) and bool(identities) and (
        expected_count is None or len(identities) == expected_count
    )


def _identity_run_error(
    row: Mapping[str, Any], identities: Sequence[Any], engine: Mapping[str, Any], root: Path
) -> Optional[str]:
    """Validate each per-run identity against local provenance and artifacts."""

    for identity in identities:
        error = _running_identity_error({**row, "running_identity": identity}, engine, root)
        if error:
            return error
    return None


def _identities_match(identities: Sequence[Any]) -> bool:
    """Check process-independent identity fields across runs."""

    return len({_identity_fingerprint(identity) for identity in identities}) == 1


def _identity_fingerprint(identity: Any) -> Tuple[Any, ...]:
    """Keep process-independent fields for cross-run identity comparison."""

    if not isinstance(identity, Mapping):
        return (None,)
    start = identity.get("start") if isinstance(identity.get("start"), Mapping) else {}
    payload = start.get("identity") if isinstance(start.get("identity"), Mapping) else {}
    return tuple(
        payload.get(field)
        for field in ("source_id", "executable_sha256", "model_sha256", "process_start_ns")
    )


def _engine_row(rows: Any, engine_id: Any) -> Optional[Mapping[str, Any]]:
    """Find one recorded engine row."""

    return next((item for item in rows if isinstance(item, dict) and item.get("id") == engine_id), None)


def _engine_declaration_error(row: Mapping[str, Any], engine: Mapping[str, Any]) -> Optional[str]:
    """Check that receipt provenance uses the declared executable."""

    fields = (
        "executable_path",
        "build_info_command",
        "trace_endpoint",
        "identity_endpoint",
        "identity_provider",
        "slot_copy",
        "comparison_support",
        "thinking_policy",
        "spawn",
        "history_tokenization",
        "cancel_observation_endpoint",
        "model_artifact",
        "model",
        "backend",
        "device",
        "quality_label",
        "quality_producer",
    )
    if any(row.get(field) != engine.get(field) for field in fields):
        return f"engine provenance declaration differs: {engine.get('id')}"
    return None


def _engine_status_error(
    provenance: Mapping[str, Any], engine: Mapping[str, Any], root: Path, manifest_phase: Any = None
) -> Optional[str]:
    """Check observed hashes or an explicit unavailable state."""

    try:
        path = root_path(root, str(provenance.get("executable_path", "")))
    except ValueError:
        return f"executable path escapes root: {engine.get('id')}"
    if provenance.get("status") == "observed" and (path.is_file() or _offline()):
        if _file_digest_mismatch(path, provenance.get("executable_sha256")):
            return f"executable SHA-256 does not match: {engine.get('id')}"
        if manifest_phase == "frozen" and provenance.get("build_info", {}).get("source_id_status") != "observed":
            return f"running build source identity is unavailable: {engine.get('id')}"
        return None
    if provenance.get("status") != "unavailable":
        return f"engine provenance status is invalid: {engine.get('id')}"
    if manifest_phase == "frozen":
        return f"engine executable provenance is unavailable: {engine.get('id')}"
    return None


def _running_identity_error(row: Mapping[str, Any], engine: Mapping[str, Any], root: Path) -> Optional[str]:
    """Require start and end identity from the process that served the run."""

    identity = row.get("running_identity")
    if not isinstance(identity, dict):
        return f"running server identity is invalid: {engine.get('id')}"
    starts, ends = identity.get("start"), identity.get("end")
    if not _identity_observations_valid(starts, ends):
        return f"running server identity is incomplete: {engine.get('id')}"
    start_value = starts.get("identity", {})
    end_value = ends.get("identity", {})
    if not _identity_payloads_valid(start_value, end_value):
        return f"running server identity payload is invalid: {engine.get('id')}"
    return _identity_value_error(row, start_value, end_value, engine, root)


def _identity_value_error(
    row: Mapping[str, Any],
    start_value: Mapping[str, Any],
    end_value: Mapping[str, Any],
    engine: Mapping[str, Any],
    root: Path,
) -> Optional[str]:
    """Compare process identity with itself and with local artifacts."""

    if start_value != end_value:
        return f"running server identity changed during run: {engine.get('id')}"
    build_source = row.get("provenance", {}).get("build_info", {}).get("source_id")
    if isinstance(build_source, str) and start_value.get("source_id") != build_source:
        return f"running server source identity differs from build: {engine.get('id')}"
    try:
        executable = root_path(root, str(engine.get("executable_path", "")))
    except ValueError:
        return f"running executable path escapes root: {engine.get('id')}"
    if not _offline() and executable.is_file() and start_value.get("executable_sha256") != sha256_file(executable):
        return f"running executable SHA-256 does not match: {engine.get('id')}"
    return _model_identity_error(start_value, engine, root)


def _model_identity_error(identity: Mapping[str, Any], engine: Mapping[str, Any], root: Path) -> Optional[str]:
    """Compare running model identity with the declared model artifact."""

    artifact = engine.get("model_artifact")
    if not isinstance(artifact, str):
        return None
    try:
        model_path = root_path(root, artifact)
    except ValueError:
        return f"running model path escapes root: {engine.get('id')}"
    if not _offline() and model_path.is_file() and identity.get("model_sha256") != sha256_file(model_path):
        return f"running model SHA-256 does not match: {engine.get('id')}"
    return None


def _identity_observations_valid(starts: Any, ends: Any) -> bool:
    """Check start and end identity observation wrappers."""

    return (
        isinstance(starts, dict)
        and isinstance(ends, dict)
        and starts.get("status") == "observed"
        and ends.get("status") == "observed"
    )


def _identity_payloads_valid(starts: Any, ends: Any) -> bool:
    """Check start and end identity payload objects."""

    return (
        isinstance(starts, dict)
        and isinstance(ends, dict)
        and _identity_fields_valid(starts)
        and _identity_fields_valid(ends)
    )


def _receipt_artifact_errors(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], root: Path
) -> List[str]:
    """Verify recorded artifact paths and hashes, including unavailable files."""

    recorded = receipt.get("artifacts", {})
    frozen = manifest.get("phase") == "frozen"
    return [
        error
        for name, item in manifest.get("artifacts", {}).items()
        for error in _receipt_artifact_item_errors(name, item, recorded, frozen, root)
    ]


def _receipt_artifact_item_errors(
    name: str, item: Any, recorded: Any, frozen: bool, root: Path
) -> List[str]:
    """Verify one recorded artifact path, status, and digest."""

    row = recorded.get(name) if isinstance(recorded, dict) else None
    try:
        path = root_path(root, item["path"])
    except (KeyError, ValueError, TypeError):
        return [f"artifact path escapes root: {name}"]
    if not isinstance(row, dict) or row.get("path") != item["path"]:
        return [f"artifact provenance is missing: {name}"]
    if _offline() and name != "quality":
        return _receipt_recorded_artifact_errors(name, row)
    return _receipt_present_artifact_errors(name, row, path) if path.is_file() else _receipt_missing_artifact_errors(name, row, frozen)


def _receipt_recorded_artifact_errors(name: str, row: Mapping[str, Any]) -> List[str]:
    """Check one artifact's recorded status and digests without reading its bytes."""

    errors = []
    if row.get("status") != "observed" or not _valid_digest(row.get("sha256")):
        errors.append(f"artifact recorded digest or status is invalid: {name}")
    if row.get("before_status") == "observed" and row.get("before_sha256") != row.get("sha256"):
        errors.append(f"artifact changed during run: {name}")
    return errors


def _receipt_present_artifact_errors(name: str, row: Mapping[str, Any], path: Path) -> List[str]:
    """Check one present artifact's bytes and before-run digest."""

    errors = []
    if row.get("status") != "observed" or row.get("sha256") != sha256_file(path):
        errors.append(f"artifact SHA-256 or status does not match: {name}")
    if row.get("before_status") == "observed" and row.get("before_sha256") != row.get("sha256"):
        errors.append(f"artifact changed during run: {name}")
    return errors


def _receipt_missing_artifact_errors(name: str, row: Mapping[str, Any], frozen: bool) -> List[str]:
    """Reject unavailable artifacts when the phase requires their bytes."""

    return [f"required artifact is unavailable: {name}"] if frozen or row.get("status") != "unavailable" else []


def _receipt_frozen_errors(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], root: Path
) -> List[str]:
    """Verify calibration linkage and disjoint workload for a frozen receipt."""

    if manifest.get("phase") != "frozen":
        return []
    errors, _ = _frozen_manifest_reference(manifest, root, _dig(receipt, "artifacts", "model", "sha256"))
    if errors:
        return errors
    errors.extend(_quality_runtime_binding_errors(receipt, manifest, root))
    if _dig(receipt, "evaluation", "quality_binding") != _quality_binding_declaration(manifest):
        errors.append("receipt quality binding differs from the manifest declaration")
    reference = manifest["calibration_receipt"]
    receipt_evaluation = receipt.get("evaluation")
    manifest_evaluation = manifest.get("evaluation")
    calibration_digest = manifest_evaluation.get("calibration_receipt_sha256") if isinstance(manifest_evaluation, dict) else None
    if calibration_digest != reference.get("sha256"):
        errors.append("frozen thresholds are not bound to the calibration receipt")
    receipt_thresholds = receipt_evaluation.get("thresholds") if isinstance(receipt_evaluation, dict) else None
    manifest_thresholds = manifest_evaluation.get("thresholds") if isinstance(manifest_evaluation, dict) else None
    if receipt_thresholds != manifest_thresholds:
        errors.append("receipt evaluation thresholds differ from frozen manifest")
    return errors


def _quality_build_source_commit(build: Any) -> Optional[str]:
    """Read a commit from typed build metadata or the documented source ID."""

    if not isinstance(build, Mapping):
        return None
    typed = build.get("source_commit")
    if _valid_commit(typed):
        return typed
    source_id = build.get("source_id")
    if not isinstance(source_id, str):
        return None
    return _quality_source_id_commit(source_id)


def _quality_source_id_commit(source_id: str) -> Optional[str]:
    """Parse a complete source ID emitted by a build or running service."""

    if _valid_commit(source_id):
        return source_id
    parts = source_id.split(":")
    if len(parts) == 2 and _valid_commit(parts[1]):
        return parts[1]
    if len(parts) == 3 and parts[1] == "git" and _valid_commit(parts[2]):
        return parts[2]
    if len(parts) == 3 and _valid_commit(parts[1]) and parts[2] in {"true", "false"}:
        return parts[1]
    return None


def _frozen_workload_errors(
    calibration: Any, manifest: Mapping[str, Any], root: Optional[Path] = None
) -> List[str]:
    """Check phase, workload identifier, and prompt disjointness."""

    if not isinstance(calibration, dict) or calibration.get("phase") != "calibration":
        return ["calibration receipt phase is invalid"]
    errors = _calibration_shape_errors(calibration, root)
    errors.extend(_frozen_manifest_context_errors(calibration, manifest, root))
    errors.extend(_frozen_prompt_hash_errors(calibration, manifest))
    return errors


def _frozen_manifest_context_errors(
    calibration: Mapping[str, Any], manifest: Mapping[str, Any], root: Optional[Path]
) -> List[str]:
    """Validate the canonical calibration manifest context."""

    if root is None:
        return ["calibration receipt canonical manifest context is unavailable"]
    errors = _calibration_manifest_errors(calibration, manifest, root)
    if calibration.get("workload_id") == manifest.get("workload_id"):
        errors.append("frozen workload_id is not disjoint from calibration")
    return errors


def _frozen_prompt_hash_errors(calibration: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require complete and hash-disjoint prompt provenance."""

    old_prompts = calibration.get("prompts") if isinstance(calibration.get("prompts"), list) else []
    old_hashes = {item.get("sha256") for item in old_prompts if isinstance(item, dict)}
    new_hashes = {item["sha256"] for item in prompt_provenance(manifest)}
    errors = []
    if None in old_hashes or len(old_hashes) != len(old_prompts):
        errors.append("calibration prompt provenance is incomplete or duplicated")
    if old_hashes & new_hashes:
        errors.append("frozen prompts are not disjoint from calibration")
    return errors


def _calibration_manifest_errors(
    calibration: Mapping[str, Any], frozen_manifest: Mapping[str, Any], root: Path
) -> List[str]:
    """Bind calibration evidence to its canonical calibration manifest."""

    reference = calibration.get("manifest")
    if not isinstance(reference, dict):
        return ["calibration receipt manifest reference is missing"]
    path_error = _manifest_path_error(reference.get("path"), "calibration receipt manifest.path")
    if path_error or not _valid_digest(reference.get("canonical_sha256")):
        return ["calibration receipt manifest path and canonical hash are required"]
    value, errors = _load_calibration_manifest(reference, root)
    if value is None:
        return errors
    errors.extend(_calibration_manifest_binding_errors(calibration, value, reference))
    errors.extend(_calibration_artifact_binding_errors(calibration, value))
    errors.extend(_calibration_engine_declaration_errors(calibration, value))
    errors.extend(_calibration_frozen_scope_errors(value, frozen_manifest))
    if frozen_manifest.get("phase") == "frozen":
        errors.extend(_prompt_disjointness_errors(value, frozen_manifest))
        errors.extend(_calibration_metric_support_errors(calibration, frozen_manifest))
    return errors


def _prompt_disjointness_errors(
    calibration_manifest: Mapping[str, Any], frozen_manifest: Mapping[str, Any]
) -> List[str]:
    """Reject frozen prompts that are near copies of calibration prompts."""

    old = _manifest_prompts_by_id(calibration_manifest)
    new = _manifest_prompts_by_id(frozen_manifest)
    errors = []
    for prompt_id, prompt in new.items():
        previous = old.get(prompt_id)
        if previous is None:
            errors.append(f"calibration prompt is missing for frozen prompt: {prompt_id}")
            continue
        ratio = difflib.SequenceMatcher(
            None,
            canonical_json(_prompt_messages(previous)),
            canonical_json(_prompt_messages(prompt)),
        ).ratio()
        if ratio > PROMPT_DISJOINT_MAX_SEQUENCE_RATIO:
            errors.append(f"calibration and frozen prompt content is not meaningfully disjoint: {prompt_id}")
    return errors


def _manifest_prompts_by_id(manifest: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """Index parent, branch, and scheduled prompts by their declared IDs."""

    prompts = manifest.get("prompts") if isinstance(manifest.get("prompts"), Mapping) else {}
    values: List[Any] = [prompts.get("parent")]
    branches = prompts.get("branches") if isinstance(prompts.get("branches"), list) else []
    values.extend(branches)
    schedule = manifest.get("schedule") if isinstance(manifest.get("schedule"), Mapping) else {}
    values.extend(schedule.get(role) for role in REQUIRED_SCHEDULE_ROLES)
    return {
        item["id"]: item
        for item in values
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }


def _calibration_metric_support_errors(
    calibration: Mapping[str, Any], frozen_manifest: Mapping[str, Any]
) -> List[str]:
    """Require every frozen metric to have qualified calibration observations."""

    evaluation = frozen_manifest.get("evaluation")
    thresholds = evaluation.get("thresholds") if isinstance(evaluation, Mapping) else None
    budgets = calibration.get("budgets")
    minimum = budgets.get("minimum_samples_for_quantiles") if isinstance(budgets, Mapping) else None
    if not isinstance(thresholds, list) or not isinstance(minimum, int) or minimum <= 0:
        return []
    grouped = _records_by_engine_role(calibration)
    runs = calibration.get("runs") if isinstance(calibration.get("runs"), list) else []
    receipt_digest = _frozen_calibration_digest(frozen_manifest)
    errors = []
    for threshold in thresholds:
        if not isinstance(threshold, Mapping):
            continue
        records = grouped.get((threshold.get("engine"), threshold.get("role")), [])
        observed = _threshold_observation(threshold, records, runs, minimum)
        if observed is None:
            errors.append(
                "calibration receipt lacks selected metric evidence: "
                f"{threshold.get('engine')}/{threshold.get('role')}/{threshold.get('metric')}"
            )
            continue
        errors.extend(_threshold_calibration_observation_errors(threshold, observed, receipt_digest))
    return errors


def _frozen_calibration_digest(manifest: Mapping[str, Any]) -> Optional[str]:
    """Read the digest that every frozen calibration decision must name."""

    reference = manifest.get("calibration_receipt")
    if isinstance(reference, Mapping) and _valid_digest(reference.get("sha256")):
        return reference["sha256"]
    evaluation = manifest.get("evaluation")
    digest = evaluation.get("calibration_receipt_sha256") if isinstance(evaluation, Mapping) else None
    return digest if _valid_digest(digest) else None


def _threshold_calibration_observation_errors(
    threshold: Mapping[str, Any], observed: Union[float, bool], receipt_digest: Optional[str]
) -> List[str]:
    """Require one selected bound to equal its recomputed calibration decision."""

    errors = _threshold_calibration_shape_errors(threshold, "evaluation.threshold")
    decision = threshold.get("calibration")
    if not isinstance(decision, Mapping):
        return errors
    if receipt_digest is None or decision.get("receipt_sha256") != receipt_digest:
        errors.append("threshold calibration decision is not bound to the frozen receipt")
    expected = decision.get("observation")
    if isinstance(observed, bool) or isinstance(expected, bool):
        if observed is not expected:
            errors.append("threshold calibration observation does not recompute")
    elif not _same_numeric(observed, expected):
        errors.append("threshold calibration observation does not recompute")
    return errors


def _same_numeric(left: Any, right: Any) -> bool:
    """Compare finite numeric observations without accepting type confusions."""

    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and isinstance(right, (int, float))
        and not isinstance(right, bool)
        and math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    )


def _load_calibration_manifest(
    reference: Mapping[str, Any], root: Path
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Load one root-contained canonical calibration manifest."""

    try:
        path = root_path(root, reference["path"])
    except ValueError:
        return None, ["calibration receipt manifest path escapes root"]
    if not path.is_file():
        return None, ["calibration receipt manifest is missing"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, ["calibration receipt manifest is not valid JSON"]
    if not isinstance(value, dict):
        return None, ["calibration receipt manifest is not an object"]
    return value, validate_manifest(value)


def _calibration_manifest_binding_errors(
    calibration: Mapping[str, Any], value: Mapping[str, Any], reference: Mapping[str, Any]
) -> List[str]:
    """Check canonical hash, phase, workload, prompts, and budgets."""

    errors = []
    if reference["canonical_sha256"] != sha256_bytes(canonical_json(value)):
        errors.append("calibration receipt manifest canonical SHA-256 does not match")
    if value.get("phase") != "calibration":
        errors.append("calibration receipt manifest is not a calibration manifest")
    if value.get("workload_id") != calibration.get("workload_id"):
        errors.append("calibration receipt workload_id is not bound to its manifest")
    if calibration.get("prompts") != prompt_provenance(value):
        errors.append("calibration receipt prompts do not match its manifest")
    if calibration.get("budgets") != value.get("budgets"):
        errors.append("calibration receipt budgets do not match its manifest")
    return errors


def _calibration_artifact_binding_errors(
    calibration: Mapping[str, Any], calibration_manifest: Mapping[str, Any]
) -> List[str]:
    """Require calibration artifact rows to match the canonical manifest."""

    rows = calibration.get("artifacts")
    declared = calibration_manifest.get("artifacts")
    if not isinstance(rows, dict) or not isinstance(declared, dict):
        return ["calibration artifact declarations are missing"]
    errors = []
    if set(rows) != set(declared):
        errors.append("calibration artifact declarations differ from its manifest")
    for name, item in declared.items():
        row = rows.get(name)
        if not isinstance(row, dict) or row.get("path") != item.get("path"):
            errors.append(f"calibration artifact declaration differs: {name}")
    return errors


def _calibration_engine_declaration_errors(
    calibration: Mapping[str, Any], calibration_manifest: Mapping[str, Any]
) -> List[str]:
    """Require each calibration engine row to match its manifest declaration."""

    rows = calibration.get("engines")
    if not isinstance(rows, list):
        return ["calibration receipt engine declarations are missing"]
    declared = calibration_manifest.get("engines", [])
    errors = _calibration_engine_set_errors(rows, declared)
    errors.extend(_calibration_engine_row_errors(rows, declared))
    return errors


def _calibration_engine_set_errors(rows: Sequence[Any], declared: Sequence[Any]) -> List[str]:
    """Check calibration engine ID sets."""

    declared_ids = {item.get("id") for item in declared if isinstance(item, dict) and isinstance(item.get("id"), str)}
    row_ids = {item.get("id") for item in rows if isinstance(item, dict) and isinstance(item.get("id"), str)}
    errors = []
    if len(row_ids) != len(rows):
        errors.append("calibration receipt engine declarations are duplicated or invalid")
    if row_ids != declared_ids:
        errors.append("calibration receipt engine declarations differ from its manifest")
    return errors


def _calibration_engine_row_errors(rows: Sequence[Any], declared: Sequence[Any]) -> List[str]:
    """Compare each declared engine with its calibration row."""

    errors = []
    for engine in declared:
        if not isinstance(engine, dict):
            continue
        row = _engine_row(rows, engine.get("id"))
        if row is None:
            errors.append(f"calibration receipt engine is missing: {engine.get('id')}")
            continue
        error = _engine_declaration_error(row, engine)
        if error:
            errors.append(error.replace("engine provenance declaration differs", "calibration engine declaration differs"))
    return errors


def _calibration_frozen_scope_errors(
    calibration_manifest: Mapping[str, Any], frozen_manifest: Mapping[str, Any]
) -> List[str]:
    """Require frozen and calibration engines to target the same artifacts."""

    def scope(value: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple(value.get(field) for field in ("id", "kind", "model", "backend", "device", "model_artifact"))

    calibration_scopes = [scope(item) for item in calibration_manifest.get("engines", []) if isinstance(item, dict)]
    frozen_scopes = [scope(item) for item in frozen_manifest.get("engines", []) if isinstance(item, dict)]
    return [] if calibration_scopes == frozen_scopes else ["calibration and frozen engine scopes differ"]


def _calibration_shape_errors(calibration: Mapping[str, Any], root: Optional[Path] = None) -> List[str]:
    """Require a nonempty calibration receipt with evaluated observations."""

    errors = _calibration_header_errors(calibration)
    errors.extend(_calibration_run_errors(calibration))
    errors.extend(_calibration_artifact_errors(calibration, root))
    evaluation = calibration.get("evaluation")
    evidence = evaluation.get("calibration_evidence") if isinstance(evaluation, dict) else None
    if not isinstance(evidence, dict) or evidence.get("status") != "observed":
        errors.append("calibration receipt has no evaluated calibration evidence")
    else:
        errors.extend(_calibration_evidence_errors(calibration, evidence))
    return errors


def _calibration_header_errors(calibration: Mapping[str, Any]) -> List[str]:
    """Require schema, workload, and prompt provenance fields."""

    errors = []
    if calibration.get("schema_version") != SCHEMA_VERSION:
        errors.append("calibration receipt schema is invalid")
    if not isinstance(calibration.get("workload_id"), str) or not calibration["workload_id"]:
        errors.append("calibration receipt workload_id is missing")
    prompts = calibration.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        errors.append("calibration receipt has no prompt provenance")
    elif any(not isinstance(item, dict) or not _valid_digest(item.get("sha256")) for item in prompts):
        errors.append("calibration receipt prompt provenance is invalid")
    return errors


def _calibration_run_errors(calibration: Mapping[str, Any]) -> List[str]:
    """Require completed calibration runs and per-role summaries."""

    runs = calibration.get("runs")
    if not isinstance(runs, list) or not runs:
        return ["calibration receipt has no runs"]
    errors = _calibration_summary_errors(calibration)
    budgets = calibration.get("budgets", {})
    engines = calibration.get("engines", [])
    errors.extend(_calibration_run_count_errors(runs, budgets, engines))
    records = _all_records(calibration)
    if not any(record.get("status") == "completed" for record in records):
        errors.append("calibration receipt has no completed evidence")
    errors.extend(_calibration_run_item_errors(runs, calibration, budgets, engines))
    return errors


def _calibration_run_count_errors(runs: Sequence[Any], budgets: Any, engines: Any) -> List[str]:
    """Check calibration run count and budget shapes."""

    expected = budgets.get("repetitions", 0) * len(engines) if isinstance(budgets, dict) and isinstance(engines, list) else 0
    return [] if len(runs) == expected else [
        "calibration receipt run count differs from repetitions times engines"
    ]


def _calibration_run_item_errors(
    runs: Sequence[Any], calibration: Mapping[str, Any], budgets: Any, engines: Any
) -> List[str]:
    """Validate each calibration run and engine coverage."""

    seen = set()
    engine_ids = {item.get("id") for item in engines if isinstance(item, dict)}
    observed_engines = set()
    errors = []
    for run in runs:
        run_errors, engine_id = _calibration_run_item_error(run, calibration, budgets, engine_ids, seen)
        errors.extend(run_errors)
        if engine_id is not None:
            observed_engines.add(engine_id)
    if engine_ids and not engine_ids.issubset(observed_engines):
        errors.append("calibration receipt does not cover every engine")
    return errors


def _calibration_run_item_error(
    run: Any, calibration: Mapping[str, Any], budgets: Mapping[str, Any],
    engine_ids: set, seen: set,
) -> Tuple[List[str], Optional[str]]:
    """Validate one calibration run identity and evidence."""

    if not isinstance(run, dict):
        return ["calibration receipt contains an invalid run"], None
    key = (run.get("engine"), run.get("repetition"))
    errors = []
    if run.get("engine") not in engine_ids:
        errors.append("calibration receipt names an engine outside its manifest")
    if key in seen:
        errors.append("calibration receipt repeats an engine and repetition pair")
    seen.add(key)
    if not isinstance(run.get("repetition"), int) or not 0 <= run["repetition"] < budgets.get("repetitions", 0):
        errors.append("calibration receipt repetition is outside the manifest")
    errors.extend(_calibration_run_record_errors(run, calibration))
    return errors, run.get("engine")


def _calibration_summary_errors(calibration: Mapping[str, Any]) -> List[str]:
    """Require a recomputable per-engine role summary."""

    summary = calibration.get("summary_by_engine_role")
    if not isinstance(summary, dict) or not summary:
        return ["calibration receipt has no per-engine role summaries"]
    budgets = calibration.get("budgets", {})
    minimum = budgets.get("minimum_samples_for_quantiles") if isinstance(budgets, dict) else None
    maximum = budgets.get("max_history") if isinstance(budgets, dict) else None
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        return ["calibration receipt budgets are missing"]
    expected = summary_by_engine_role(calibration, minimum, maximum)
    if summary != expected:
        return ["calibration per-engine role summaries do not recompute"]
    expected_summary = outcome_summary(_all_records(calibration), minimum, maximum)
    errors = [] if calibration.get("summary") == expected_summary else ["calibration summary does not recompute from outcomes"]
    return errors + _calibration_sample_errors(calibration, minimum)


def _calibration_sample_errors(calibration: Mapping[str, Any], minimum: int) -> List[str]:
    """Require enough successful observations for every calibration role."""

    grouped = _records_by_engine_role(calibration)
    errors = []
    for engine in calibration.get("engines", []):
        if not isinstance(engine, dict):
            continue
        unacknowledged = "cancel_latency_p95_ms" in _unsupported_metrics(engine_kind(engine))
        for role in ("parent", "branch", "new_prompt", "cancel", "slow_reader"):
            records = grouped.get((engine.get("id"), role), [])
            expected_status = "completed"
            if role == "cancel":
                expected_status = "cancel_acknowledgement_unavailable" if unacknowledged else "cancelled"
            count = sum(
                record.get("status") == expected_status and record.get("history_complete") is True
                for record in records
            )
            if count < minimum:
                errors.append(
                    f"calibration role has fewer than minimum samples: {engine.get('id')}/{role}"
                )
    return errors


def _calibration_run_record_errors(run: Mapping[str, Any], calibration: Mapping[str, Any]) -> List[str]:
    """Require one calibration run to contain the bounded concurrent schedule."""

    errors = _calibration_request_identity_errors(run)
    errors.extend(_calibration_parent_errors(run, calibration))
    branches = run.get("branches")
    errors.extend(_calibration_branch_errors(branches, calibration))
    probes = run.get("probes")
    errors.extend(_calibration_probe_shape_errors(probes))
    overlap = run.get("overlap")
    if overlap != _schedule_overlap(_run_records(run)):
        errors.append("calibration receipt run lacks overlap evidence")
    errors.extend(_record_validation_errors(_run_records(run)))
    parent = run.get("parent") if isinstance(run.get("parent"), dict) else {}
    if parent.get("status") != "completed":
        errors.append("calibration receipt run parent did not complete")
    errors.extend(_calibration_branch_outcome_errors(parent, branches, run, calibration))
    errors.extend(_calibration_contract_errors(run, probes, calibration))
    return errors


def _calibration_contract_errors(run: Mapping[str, Any], probes: Any, calibration: Mapping[str, Any]) -> List[str]:
    """Apply the engine's fixed evidence contract to one calibration run."""

    unsupported = _unsupported_metrics(_run_engine_kind(run, calibration))
    errors = _run_support_record_errors(run, calibration)
    errors.extend(_calibration_probe_outcome_errors(probes, unsupported))
    errors.extend(_calibration_token_errors(run, unsupported))
    errors.extend(_frozen_limit_stop_errors(run))
    errors.extend(_frozen_wire_errors(run, unsupported))
    return errors


def _calibration_request_identity_errors(run: Mapping[str, Any]) -> List[str]:
    """Require stable client and service identities for calibration requests."""

    records = _run_records(run)
    errors = _calibration_parent_request_error(run)
    errors.extend(_calibration_branch_request_errors(run))
    errors.extend(_calibration_record_identity_errors(records))
    return errors


def _calibration_parent_request_error(run: Mapping[str, Any]) -> List[str]:
    """Check the deterministic parent request identity."""

    parent_id = f"branch-parent-{run.get('engine')}-{run.get('repetition')}"
    parent = run.get("parent") if isinstance(run.get("parent"), dict) else {}
    return [] if parent.get("request_id") == parent_id else [
        "calibration parent request_id differs from schedule"
    ]


def _calibration_branch_request_errors(run: Mapping[str, Any]) -> List[str]:
    """Check branch request IDs against their branch IDs."""

    branches = run.get("branches") if isinstance(run.get("branches"), list) else []
    return [
        "calibration branch request_id differs from branch_id"
        for branch in branches
        if not isinstance(branch, dict) or branch.get("request_id") != branch.get("branch_id")
    ]


def _calibration_record_identity_errors(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Require unique client IDs and service IDs for terminal requests."""

    errors = []
    if any(not isinstance(record.get("request_id"), str) or not record["request_id"] for record in records):
        errors.append("calibration request identity is missing")
    ids = [record.get("request_id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("calibration request IDs are not unique")
    if any(record.get("status") in SERVICE_OUTCOMES and not _nonempty_string(record.get("service_request_id")) for record in records):
        errors.append("calibration service request identity is missing")
    return errors


def _nonempty_string(value: Any) -> bool:
    """Return whether a value is a nonempty string."""

    return isinstance(value, str) and bool(value)


def _calibration_branch_errors(branches: Any, calibration: Mapping[str, Any]) -> List[str]:
    """Require one calibration branch for each canonical branch prompt."""

    expected = _calibration_branch_count(calibration)
    if not isinstance(branches, list) or len(branches) != expected or any(not isinstance(item, dict) for item in branches):
        return ["calibration receipt run has invalid branch records"]
    return []


def _calibration_branch_count(calibration: Mapping[str, Any]) -> int:
    """Count canonical prompt entries that require a branch request."""

    prompts = calibration.get("prompts", [])
    schedule = calibration.get("schedule")
    schedule_items = schedule.values() if isinstance(schedule, dict) else []
    schedule_ids = {item.get("id") for item in schedule_items if isinstance(item, dict)}
    return sum(
        1 for item in prompts
        if isinstance(item, dict) and item.get("id") not in schedule_ids and item.get("id") != "parent"
    ) if isinstance(prompts, list) else 0


def _calibration_probe_shape_errors(probes: Any) -> List[str]:
    """Require one record for each bounded calibration probe."""

    if not isinstance(probes, list) or any(not isinstance(item, dict) for item in probes):
        return ["calibration receipt run has invalid probe records"]
    roles = {item.get("role") for item in probes}
    return [] if roles == REQUIRED_SCHEDULE_ROLES and len(probes) == len(REQUIRED_SCHEDULE_ROLES) else [
        "calibration receipt run lacks a required schedule role"
    ]


def _calibration_branch_outcome_errors(
    parent: Mapping[str, Any], branches: Any, run: Mapping[str, Any], calibration: Mapping[str, Any]
) -> List[str]:
    """Require completed branches with positive history reuse."""

    if not isinstance(branches, list):
        return []
    errors = [
        "calibration receipt branch did not complete"
        for item in branches
        if not isinstance(item, dict) or item.get("status") != "completed"
    ]
    if parent.get("status") == "completed":
        for item in branches:
            if isinstance(item, dict):
                errors.extend(_history_reuse_errors(
                    parent, item, run=run, engine=_history_engine_declaration(calibration, run.get("engine"), None)
                ))
    return errors


def _probe_outcome_ok(item: Any, cancelled_status: str) -> bool:
    """Return whether one probe ended in the outcome its role requires."""

    if not isinstance(item, dict):
        return False
    return item.get("status") == (cancelled_status if item.get("role") == "cancel" else "completed")


def _calibration_probe_outcome_errors(probes: Any, unsupported: Any = frozenset()) -> List[str]:
    """Require completed probes and a cancelled cancellation probe.

    An engine without cancel acknowledgement ends the cancel probe as
    `cancel_acknowledgement_unavailable`, with no acknowledgement observed.
    """

    if not isinstance(probes, list):
        return []
    unacknowledged = "cancel_latency_p95_ms" in unsupported
    cancelled = "cancel_acknowledgement_unavailable" if unacknowledged else "cancelled"
    errors = [
        "calibration receipt probe outcome is not successful"
        for item in probes
        if not _probe_outcome_ok(item, cancelled)
    ]
    if unacknowledged:
        cancel = next((item for item in probes if isinstance(item, dict) and item.get("role") == "cancel"), None)
        errors.extend(_unsupported_cancel_errors(cancel))
    return errors


def _calibration_token_errors(run: Mapping[str, Any], unsupported: Any = frozenset()) -> List[str]:
    """Require raw token evidence for every completed calibration request."""

    return _frozen_token_errors(run, unsupported)


def _calibration_parent_errors(run: Mapping[str, Any], calibration: Mapping[str, Any]) -> List[str]:
    """Require one completed parent with the declared prompt-token minimum."""

    parent = run.get("parent")
    if not isinstance(parent, dict):
        return ["calibration receipt run has no parent record"]
    errors = [] if parent.get("status") == "completed" else ["calibration receipt run parent did not complete"]
    errors.extend(_calibration_parent_usage_error(parent, calibration))
    return errors


def _calibration_parent_usage_error(
    parent: Mapping[str, Any], calibration: Mapping[str, Any]
) -> List[str]:
    """Require the parent prompt token minimum."""

    parent_prompt = next(
        (item for item in calibration.get("prompts", []) if isinstance(item, dict) and item.get("id") == "parent"),
        {},
    )
    minimum = parent_prompt.get("minimum_prompt_tokens")
    usage = parent.get("usage") if isinstance(parent.get("usage"), dict) else {}
    prompt_tokens = usage.get("prompt_tokens")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or prompt_tokens < minimum
    ):
        return ["calibration receipt parent misses the long-context minimum"]
    return []


def _calibration_artifact_errors(
    calibration: Mapping[str, Any], root: Optional[Path] = None
) -> List[str]:
    """Require observed artifact and executable evidence before freezing."""

    artifacts = calibration.get("artifacts")
    errors = _calibration_artifact_rows_errors(artifacts, root)
    engines = calibration.get("engines")
    if not isinstance(engines, list) or not engines:
        return errors + ["calibration receipt has no engine provenance"]
    for engine in engines:
        errors.extend(_calibration_engine_artifact_errors(engine, root))
    return errors


def _calibration_artifact_rows_errors(artifacts: Any, root: Optional[Path]) -> List[str]:
    """Verify calibration artifact rows and bytes."""

    if not isinstance(artifacts, dict) or not artifacts:
        return ["calibration receipt has no artifact provenance"]
    errors = []
    for name, item in artifacts.items():
        if not isinstance(item, dict) or item.get("status") != "observed" or not _valid_digest(item.get("sha256")):
            errors.append(f"calibration artifact is unavailable: {name}")
            continue
        if root is not None:
            errors.extend(_calibration_artifact_file_errors(name, item, root))
    return errors


def _calibration_artifact_file_errors(name: str, item: Mapping[str, Any], root: Path) -> List[str]:
    """Verify one root-contained calibration artifact."""

    path_error = _manifest_path_error(item.get("path"), f"calibration artifact {name}.path")
    if path_error:
        return [f"calibration artifact is unavailable: {name}"]
    try:
        path = root_path(root, item["path"])
    except ValueError:
        return [f"calibration artifact path escapes root: {name}"]
    matches = _valid_digest(item["sha256"]) if _offline() and name != "quality" else (
        path.is_file() and sha256_file(path) == item["sha256"]
    )
    return [] if matches else [f"calibration artifact SHA-256 does not match: {name}"]


def _calibration_engine_artifact_errors(engine: Any, root: Optional[Path]) -> List[str]:
    """Verify one calibration executable provenance row."""

    if not isinstance(engine, dict):
        return ["calibration receipt contains an invalid engine"]
    provenance = engine.get("provenance")
    errors = _calibration_provenance_errors(engine, provenance)
    if root is not None and isinstance(provenance, dict) and provenance.get("status") == "observed":
        errors.extend(_calibration_executable_file_errors(engine, provenance, root))
    if "source_id" in engine:
        errors.append(f"calibration engine copied a source identity: {engine.get('id')}")
    return errors


def _calibration_provenance_errors(engine: Mapping[str, Any], provenance: Any) -> List[str]:
    """Require observed executable and build provenance."""

    build_info = provenance.get("build_info") if isinstance(provenance, dict) else None
    valid = (
        isinstance(provenance, dict)
        and provenance.get("status") == "observed"
        and _valid_digest(provenance.get("executable_sha256"))
        and isinstance(build_info, dict)
        and build_info.get("status") == "observed"
        and build_info.get("source_id_status") == "observed"
    )
    return [] if valid else [f"calibration executable provenance is incomplete: {engine.get('id')}" ]


def _calibration_executable_file_errors(
    engine: Mapping[str, Any], provenance: Mapping[str, Any], root: Path
) -> List[str]:
    """Verify one root-contained calibration executable."""

    if provenance.get("executable_path") != engine.get("executable_path"):
        return [f"calibration executable declaration differs: {engine.get('id')}"]
    try:
        executable = root_path(root, str(provenance.get("executable_path", "")))
    except ValueError:
        return [f"calibration executable path escapes root: {engine.get('id')}"]
    return [] if _file_matches(executable, provenance.get("executable_sha256")) else [
        f"calibration executable SHA-256 does not match: {engine.get('id')}"
    ]


def _calibration_evidence_errors(
    calibration: Mapping[str, Any], evidence: Mapping[str, Any]
) -> List[str]:
    """Recompute the compact calibration evidence fields."""

    errors = _calibration_evidence_identity_errors(calibration, evidence)
    budgets = calibration.get("budgets")
    if not isinstance(budgets, dict):
        return errors + ["calibration receipt budgets are missing"]
    minimum = budgets.get("minimum_samples_for_quantiles")
    maximum = budgets.get("max_history")
    if (
        not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum <= 0
        or not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum <= 0
    ):
        return errors + ["calibration receipt budgets are invalid"]
    errors.extend(_calibration_evidence_summary_errors(calibration, evidence, minimum, maximum))
    return errors


def _calibration_evidence_identity_errors(
    calibration: Mapping[str, Any], evidence: Mapping[str, Any]
) -> List[str]:
    """Check workload and prompt identities in compact evidence."""

    errors = []
    if evidence.get("workload_id") != calibration.get("workload_id"):
        errors.append("calibration evidence workload_id does not match receipt")
    if not _valid_digest(evidence.get("summary_sha256")):
        errors.append("calibration evidence summary hash is invalid")
    prompts = calibration.get("prompts") if isinstance(calibration.get("prompts"), list) else []
    prompt_hashes = [item.get("sha256") for item in prompts if isinstance(item, dict)]
    if evidence.get("prompt_hashes") != prompt_hashes:
        errors.append("calibration evidence prompt hashes do not match receipt")
    return errors


def _calibration_evidence_summary_errors(
    calibration: Mapping[str, Any], evidence: Mapping[str, Any], minimum: int, maximum: int
) -> List[str]:
    """Recompute the calibration summary and its content hash."""

    summary = outcome_summary(_all_records(calibration), minimum, maximum)
    expected = sha256_bytes(canonical_json(summary))
    errors = [] if evidence.get("summary_sha256") == expected else [
        "calibration evidence summary does not match receipt"
    ]
    if calibration.get("summary") != summary:
        errors.append("calibration summary does not recompute from outcomes")
    return errors


def _calibration_reference_errors(manifest: Mapping[str, Any], root: Path) -> List[str]:
    """Check the receipt hash required by a frozen manifest."""

    reference = manifest.get("calibration_receipt")
    if not isinstance(reference, dict):
        return ["calibration receipt reference is missing"]
    path_value = reference.get("path")
    digest = reference.get("sha256")
    if _manifest_path_error(path_value, "calibration receipt.path") or not _valid_digest(digest):
        return ["calibration receipt path and sha256 are required"]
    try:
        path = root_path(root, path_value)
    except ValueError:
        return ["calibration receipt path escapes root"]
    if not path.is_file():
        return ["calibration receipt is missing"]
    return [] if sha256_file(path) == digest else ["calibration receipt SHA-256 does not match"]


def _frozen_manifest_reference(
    manifest: Mapping[str, Any], root: Path, model_sha256: Optional[str] = None
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """Load and validate the calibration receipt and quality records of a frozen run."""

    errors = _calibration_reference_errors(manifest, root)
    errors.extend(_quality_reference_errors(manifest, root, model_sha256))
    if errors:
        return errors, None
    reference = manifest["calibration_receipt"]
    path = root_path(root, reference["path"])
    try:
        calibration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["calibration receipt is not valid JSON"], None
    return _frozen_workload_errors(calibration, manifest, root), calibration if isinstance(calibration, dict) else None




def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping, or an empty one, so a lookup chain never raises."""

    return value if isinstance(value, Mapping) else {}


def _dig(value: Any, *keys: str) -> Any:
    """Follow mapping keys and return None at the first missing step."""

    for key in keys:
        value = _mapping(value).get(key)
    return value


def _quality_reference_errors(
    manifest: Mapping[str, Any], root: Path, model_sha256: Optional[str]
) -> List[str]:
    """Bind the evaluation comparison record to the manifest by path and SHA-256.

    This checks structure, model, corpus, and sidecar joins. It does not
    recompute KLD, top-1 agreement, or the needle argmax. `check-quality`
    (`validate-quality-stage.py comparison`) does.
    """

    record, directory, errors = _load_quality_record(_dig(manifest, "evaluation", "quality_record"), root, "evaluation")
    if record is not None:
        errors.extend(_quality_record_errors(record, directory, _quality_backend(manifest), model_sha256, "evaluation"))
    return errors


def _quality_backend(manifest: Mapping[str, Any]) -> Optional[str]:
    """Return the one backend the manifest engines share, or None."""

    backends = {engine.get("backend") for engine in manifest.get("engines", []) if isinstance(engine, Mapping)}
    return backends.pop() if len(backends) == 1 else None


def _load_quality_record(
    reference: Any, root: Path, name: str
) -> Tuple[Optional[Dict[str, Any]], Optional[Path], List[str]]:
    """Read one comparison record after its path and SHA-256 check."""

    problem = _quality_record_reference_errors(reference, f"{name} quality record")
    if problem:
        return None, None, problem
    try:
        path = root_path(root, reference["path"])
        matches = path.is_file() and sha256_file(path) == reference["sha256"]
        value = json.loads(path.read_text(encoding="utf-8")) if matches else None
    except (OSError, ValueError):
        return None, None, [f"{name} quality record cannot be read"]
    if not matches:
        return None, None, [f"{name} quality record SHA-256 does not match"]
    return (value, path.parent, []) if isinstance(value, dict) else (None, None, [f"{name} quality record is not an object"])


def _quality_record_errors(
    record: Mapping[str, Any], directory: Optional[Path], backend: Optional[str],
    model_sha256: Optional[str], name: str,
) -> List[str]:
    """Check schema, subject model, policy sample contract, both quality rows, and the long-context task."""

    errors = []
    if record.get("schema_version") != QUALITY_RECORD_SCHEMAS.get(str(backend)):
        errors.append(f"{name} quality record schema does not match the {backend} backend")
    subject = _dig(record, "models", "subject", "sha256")
    if not _valid_digest(subject) or (model_sha256 is not None and subject != model_sha256):
        errors.append(f"{name} quality record subject model differs from the served model")
    if _dig(record, "validation", "trusted", "sample_contract") != QUALITY_POLICY["sample_contract"]:
        errors.append(f"{name} quality record sample contract differs from the declared policy")
    for label in QUALITY_PRODUCERS:
        errors.extend(_quality_row_errors(record, label, directory, f"{name} quality record {label}"))
    errors.extend(_quality_common_oracle_errors(record, name))
    errors.extend(_quality_task_errors(record, name))
    return errors


def _quality_common_oracle_errors(record: Mapping[str, Any], name: str) -> List[str]:
    """Require both schema-3 quality rows to identify the same oracle."""

    oracles = [_dig(record, "quality", label, "receipt", "oracle") for label in QUALITY_PRODUCERS]
    if any(not isinstance(oracle, Mapping) for oracle in oracles):
        return [f"{name} quality receipts must declare a common oracle"]
    if oracles[0] != oracles[1]:
        return [f"{name} quality receipts do not share a common oracle"]
    return []


def _quality_row_errors(
    record: Mapping[str, Any], label: str, directory: Optional[Path], prefix: str
) -> List[str]:
    """Check one labelled quality row: sidecar bytes, inline receipt, and its metrics."""

    entry = _dig(record, "quality", label)
    if not isinstance(entry, Mapping):
        return [f"{prefix} row is missing"]
    errors = _quality_sidecar_errors(entry, directory, prefix)
    errors.extend(_quality_receipt_errors(entry.get("receipt"), record, prefix))
    return errors


def _quality_sidecar_errors(entry: Mapping[str, Any], directory: Optional[Path], prefix: str) -> List[str]:
    """Require the sidecar file beside the record to match its hash and the inline receipt."""

    name = entry.get("path")
    if directory is None or not isinstance(name, str) or Path(name).name != name or not _valid_digest(entry.get("sha256")):
        return [f"{prefix} sidecar name or digest is invalid"]
    path = directory / name
    try:
        matches = path.is_file() and sha256_file(path) == entry["sha256"]
        body = json.loads(path.read_text(encoding="utf-8")) if matches else None
    except (OSError, ValueError):
        return [f"{prefix} sidecar cannot be read"]
    if not matches:
        return [f"{prefix} sidecar SHA-256 does not match"]
    return [] if body == entry.get("receipt") else [f"{prefix} inline receipt differs from its sidecar"]


def _quality_receipt_errors(receipt: Any, record: Mapping[str, Any], prefix: str) -> List[str]:
    """Bind a schema-3 quality receipt to the record model and corpus, and check its metrics."""

    if not isinstance(receipt, Mapping):
        return [f"{prefix} receipt is missing"]
    errors = []
    if receipt.get("schema_version") != QUALITY_RECEIPT_SCHEMA:
        errors.append(f"{prefix} receipt schema is not {QUALITY_RECEIPT_SCHEMA}")
    if _dig(receipt, "subject", "model_artifact", "sha256") != _dig(record, "models", "subject", "sha256"):
        errors.append(f"{prefix} receipt subject model differs from the record")
    if _dig(receipt, "corpus", "sha256") != _dig(record, "corpus", "sha256"):
        errors.append(f"{prefix} receipt corpus differs from the record")
    errors.extend(_quality_metric_errors(receipt, prefix))
    return errors


def _quality_metric_errors(receipt: Mapping[str, Any], prefix: str) -> List[str]:
    """Require the three fields the study reads: KLD max, top-1 agreement, and sample count."""

    kld, top1 = _dig(receipt, "metrics", "kld", "max"), _dig(receipt, "metrics", "top1_agreement")
    count = receipt.get("sample_count")
    errors = []
    if not _finite_nonnegative(kld):
        errors.append(f"{prefix} receipt lacks a finite KLD max")
    if not _finite_nonnegative(top1) or top1 > 1:
        errors.append(f"{prefix} receipt top1_agreement is invalid")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        errors.append(f"{prefix} receipt sample_count is invalid")
    return errors


def _quality_task_errors(record: Mapping[str, Any], name: str) -> List[str]:
    """Bind the long-context task: the Metal result, or the CUDA manifest and answer rows.

    A CUDA record carries no result. The canonical validator checks the argmax
    of the retained logits against the answer tokens.
    """

    task = _dig(record, "tasks", "long_context")
    prefix = f"{name} quality record task"
    if record.get("schema_version") == QUALITY_RECORD_SCHEMAS["metal"]:
        return _quality_task_result_errors(_dig(task, "result"), prefix)
    answer = _dig(task, "manifest", "body", "task", "answer")
    rows, tokens, first = _dig(task, "rows"), _dig(answer, "token_ids"), _dig(answer, "row")
    valid = isinstance(tokens, list) and tokens and isinstance(first, int) and rows == list(range(first, first + len(tokens)))
    return [] if valid else [f"{prefix} rows differ from the answer tokens"]


def _quality_task_result_errors(result: Any, prefix: str) -> List[str]:
    """Require a passed `leone.quality-task-result.v2` whose argmax equals the expected answer."""

    argmax = _dig(result, "argmax")
    expected = _dig(argmax, "expected")
    labels = ("oracle", "llama_cpp_metal", "leone_metal")
    valid = (
        _dig(result, "schema_version") == TASK_RESULT_SCHEMA and _dig(result, "passed") is True
        and isinstance(expected, list) and bool(expected) and all(_dig(argmax, item) == expected for item in labels)
    )
    return [] if valid else [f"{prefix} result is not a passed {TASK_RESULT_SCHEMA}"]


def _quality_runtime_binding_errors(
    receipt: Mapping[str, Any], manifest: Mapping[str, Any], root: Path
) -> List[str]:
    """Join the evaluation record to each serving row: same bytes for Leone, same libraries for the peer."""

    record, _, errors = _load_quality_record(_dig(manifest, "evaluation", "quality_record"), root, "evaluation")
    if record is None:
        return errors
    rows = receipt.get("engines", [])
    for engine in manifest.get("engines", []):
        row = _engine_row(rows, engine.get("id")) if isinstance(rows, list) else None
        errors.extend(_quality_engine_binding_errors(record, engine, row, root))
    return errors


def _quality_engine_binding_errors(
    record: Mapping[str, Any], engine: Mapping[str, Any], row: Optional[Mapping[str, Any]], root: Path
) -> List[str]:
    """Check one serving row against the model, executable, and libraries of the record."""

    name = engine.get("id")
    if row is None:
        return [f"quality binding has no serving row: {name}"]
    errors = []
    if _dig(row, "running_identity", "start", "identity", "model_sha256") != _dig(record, "models", "subject", "sha256"):
        errors.append(f"quality model differs from the serving process: {name}")
    if engine.get("quality_producer") == "same_executable":
        errors.extend(_quality_same_executable_errors(record, row, name))
    else:
        errors.extend(_quality_peer_library_errors(record, engine, row, root))
    return errors


def _quality_same_executable_errors(record: Mapping[str, Any], row: Mapping[str, Any], name: Any) -> List[str]:
    """Require the served executable bytes to equal the native eval executable of the record.

    The top-level `executable` of a record is the statistics producer. It runs
    on its own target, so it is not compared here. `check-quality` verifies it.
    Identical bytes do not show an identical code path: serve, eval, and quality
    are three subcommands of one binary.
    """

    served = _dig(row, "provenance", "executable_sha256")
    native = _quality_native_executable(record)
    errors = [] if _valid_digest(native) and native == served else [
        f"quality native executable differs from the serving process: {name}"
    ]
    commit = _quality_build_source_commit(_dig(row, "provenance", "build_info"))
    if commit is None or commit != _dig(record, "subjects", "leone", "engine", "git_commit"):
        errors.append(f"quality source commit differs from the serving process: {name}")
    return errors


def _quality_native_executable(record: Mapping[str, Any]) -> Any:
    """Return the native eval executable hash: CUDA `producers.leone`, Metal `stages.metal_subject.native`."""

    body = _dig(record, "producers", "leone", "body") or _dig(record, "stages", "metal_subject", "body", "native", "manifest", "body")
    return _dig(body, "executable", "sha256")


def _quality_peer_library_errors(
    record: Mapping[str, Any], engine: Mapping[str, Any], row: Mapping[str, Any], root: Path
) -> List[str]:
    """Join the peer server to its quality adapter by pinned commit and numeric library hashes.

    The two executables differ by construction and are not compared. Linkage is
    resolved with ldd or otool, not read from the running process.
    """

    name = engine.get("id")
    errors = []
    if _dig(record, "subjects", "llama_cpp", "engine", "git_commit") != _pinned_llama_commit(root):
        errors.append(f"quality peer commit differs from external/PINNED: {name}")
    served = _dig(row, "provenance", "linked_libraries")
    if _dig(served, "status") != "observed" or _dig(served, "loaded_library_status") != LOADED_LIBRARY_STATUS:
        return errors + [f"peer serving libraries are not recorded as resolved linkage: {name}"]
    libraries = _load_sibling_module("linked_libraries.py")
    adapter_libraries = _quality_adapter_libraries(record)
    server_libraries = served.get("libraries") or []
    invalid = libraries.invalid_core_families(adapter_libraries, engine.get("backend"))
    invalid.update(libraries.invalid_core_families(server_libraries, engine.get("backend")))
    errors.extend(f"quality peer library digest is invalid: {name} {family}" for family in sorted(invalid))
    adapter = libraries.core_hashes(adapter_libraries, engine.get("backend"))
    server = libraries.core_hashes(server_libraries, engine.get("backend"))
    errors.extend(
        f"quality peer library differs from the adapter: {name} {family}"
        for family in libraries.core_families(engine.get("backend"))
        if not adapter.get(family) or adapter.get(family) != server.get(family)
    )
    return errors


def _quality_adapter_libraries(record: Mapping[str, Any]) -> List[Any]:
    """Return the adapter's recorded linked libraries: CUDA producer manifest or Metal stage body."""

    body = _dig(record, "producers", "llama_cpp", "body") or _dig(record, "stages", "metal_subject", "body", "subject", "manifest", "body")
    libraries = _dig(body, "executable", "linked_libraries")
    return libraries if isinstance(libraries, list) else []


def _quality_binding_declaration(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """State what the quality binding does and does not cover, per engine."""

    engines = {}
    for engine in manifest["engines"]:
        row = {"quality_label": engine["quality_label"], "quality_producer": engine["quality_producer"]}
        if engine["quality_producer"] == "peer_adapter":
            row["loaded_library_status"] = LOADED_LIBRARY_STATUS
        engines[engine["id"]] = row
    return {
        "quality_policy": QUALITY_POLICY["name"], "quality_path_status": QUALITY_PATH_STATUS,
        "quality_recomputation_status": QUALITY_RECOMPUTATION_STATUS, "engines": engines,
    }


def _receipt_run_errors(receipt: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Verify the complete concurrent schedule before evaluating evidence."""

    runs = receipt.get("runs")
    expected = manifest["budgets"]["repetitions"] * len(manifest["engines"])
    errors = []
    if not isinstance(runs, list) or len(runs) != expected:
        errors.append("run count differs from repetitions times engines")
        return errors
    branch_count = len(manifest["prompts"]["branches"])
    required_roles = REQUIRED_SCHEDULE_ROLES
    seen = set()
    for run in runs:
        if isinstance(run, dict):
            key = (run.get("engine"), run.get("repetition"))
            if key in seen:
                errors.append("run engine and repetition pair is duplicated")
            seen.add(key)
    successful = 0
    for run in runs:
        run_errors, run_successes = _run_validation_errors(run, manifest, branch_count, required_roles)
        errors.extend(run_errors)
        successful += run_successes
    if not successful:
        errors.append("receipt contains no completed request evidence")
    return errors


def _run_validation_errors(
    run: Any, manifest: Mapping[str, Any], branch_count: int, required_roles: set
) -> Tuple[List[str], int]:
    """Validate one run and return its errors and completed count."""

    if not isinstance(run, dict):
        return ["run is not an object"], 0
    errors = []
    key = (run.get("engine"), run.get("repetition"))
    errors.extend(_run_shape_errors(run, manifest, key, branch_count, required_roles))
    records = _run_records(run)
    errors.extend(_record_validation_errors(records))
    errors.extend(_parent_context_errors(run, manifest))
    if manifest.get("phase") == "frozen":
        errors.extend(_frozen_run_evidence_errors(run, manifest))
    return errors, sum(record.get("status") == "completed" for record in records)


def _run_shape_errors(
    run: Mapping[str, Any], manifest: Mapping[str, Any], key: Tuple[Any, Any], branch_count: int, required_roles: set
) -> List[str]:
    """Validate one run identity, branch count, and role set."""

    errors = _run_key_errors(key, manifest)
    errors.extend(_run_record_shape_errors(run, branch_count, required_roles))
    errors.extend(_run_request_id_errors(run, manifest))
    errors.extend(_run_overlap_errors(run))
    errors.extend(_run_support_record_errors(run, manifest))
    return errors


def _run_support_record_errors(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require each run to carry the fixed evidence contract of its declared engine kind."""

    kind = _run_engine_kind(run, manifest)
    expected = capability_record(kind)
    if expected is None:
        return ["run engine kind has no evidence contract"]
    errors = []
    if run.get("engine_kind") != kind:
        errors.append("run engine kind differs from the manifest declaration")
    if run.get("comparison_support") != expected:
        errors.append("run comparison_support differs from the fixed evidence contract")
    return errors


def _run_key_errors(key: Tuple[Any, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require an engine and repetition declared by the manifest."""

    errors = []
    engine_ids = {engine.get("id") for engine in manifest["engines"]}
    if key[0] not in engine_ids:
        errors.append("run names an engine outside the manifest")
    if not isinstance(key[1], int) or not 0 <= key[1] < manifest["budgets"]["repetitions"]:
        errors.append("run repetition is outside the manifest")
    return errors


def _run_record_shape_errors(
    run: Mapping[str, Any], branch_count: int, required_roles: set
) -> List[str]:
    """Require parent, branches, and schedule probe record shapes."""

    errors = [] if isinstance(run.get("parent"), dict) else ["run parent record is missing or invalid"]
    branches = run.get("branches") if isinstance(run.get("branches"), list) else []
    errors.extend(_run_branch_shape_errors(branches, branch_count))
    probes = run.get("probes") if isinstance(run.get("probes"), list) else []
    errors.extend(_run_probe_shape_errors(probes, required_roles))
    return errors


def _run_branch_shape_errors(branches: Sequence[Any], expected: int) -> List[str]:
    """Require the declared number of object branch records."""

    errors = []
    if len(branches) != expected:
        errors.append("run branch count differs from manifest")
    if any(not isinstance(item, dict) for item in branches):
        errors.append("run contains an invalid branch record")
    return errors


def _run_probe_shape_errors(probes: Sequence[Any], required_roles: set) -> List[str]:
    """Require one object probe for every bounded role."""

    errors = []
    if any(not isinstance(item, dict) for item in probes):
        errors.append("run contains an invalid probe record")
    roles = {item.get("role") for item in probes if isinstance(item, dict)}
    if roles != required_roles or len(probes) != len(required_roles):
        errors.append("run does not contain every bounded schedule probe")
    return errors


def _run_overlap_errors(run: Mapping[str, Any]) -> List[str]:
    """Require recorded overlap to recompute from request intervals."""

    expected_overlap = _schedule_overlap(_run_records(run))
    return [] if run.get("overlap") == expected_overlap else [
        "run overlap does not recompute from request intervals"
    ]


def _run_request_id_errors(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require one stable request identity for each declared schedule item."""

    expected = _expected_request_ids(run, manifest)
    errors = []
    parent = run.get("parent")
    if isinstance(parent, dict) and parent.get("request_id") != expected["parent"]:
        errors.append("parent request_id differs from the declared schedule")
    errors.extend(_branch_request_id_errors(run, expected))
    errors.extend(_probe_request_id_errors(run, expected))
    actual = [
        item.get("request_id")
        for item in _run_records(run)
        if isinstance(item, dict)
    ]
    if len(actual) != len(set(actual)):
        errors.append("run request IDs are not unique")
    return errors


def _expected_request_ids(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> Dict[str, str]:
    """Build deterministic request IDs from engine, repetition, and prompt IDs."""

    parent_id = f"branch-parent-{run.get('engine')}-{run.get('repetition')}"
    expected = {"parent": parent_id}
    prompts = manifest.get("prompts", {})
    branches = prompts.get("branches", []) if isinstance(prompts, dict) else []
    expected.update({f"branch:{item['id']}": f"{parent_id}-{item['id']}" for item in branches if isinstance(item, dict)})
    schedule = manifest.get("schedule", {})
    for role in REQUIRED_SCHEDULE_ROLES:
        item = schedule.get(role) if isinstance(schedule, dict) else None
        if isinstance(item, dict):
            expected[role] = f"{parent_id}-{item['id']}"
    return expected


def _branch_request_id_errors(run: Mapping[str, Any], expected: Mapping[str, str]) -> List[str]:
    """Check branch request and branch identity fields."""

    branch_ids = {value for key, value in expected.items() if key.startswith("branch:")}
    return [
        "branch request_id differs from the declared schedule"
        for branch in run.get("branches", []) if isinstance(run.get("branches"), list)
        if not isinstance(branch, dict)
        or branch.get("request_id") != branch.get("branch_id")
        or branch.get("branch_id") not in branch_ids
    ]


def _probe_request_id_errors(run: Mapping[str, Any], expected: Mapping[str, str]) -> List[str]:
    """Check one request identity for every schedule probe."""

    return [
        "probe request_id differs from the declared schedule"
        for probe in run.get("probes", []) if isinstance(run.get("probes"), list)
        if not isinstance(probe, dict)
        or probe.get("role") not in REQUIRED_SCHEDULE_ROLES
        or probe.get("request_id") != expected.get(probe.get("role"))
    ]


def _record_validation_errors(records: Sequence[Mapping[str, Any]]) -> List[str]:
    """Reject unknown outcomes and incomplete completed-request histories."""

    errors = []
    for record in records:
        status = record.get("status")
        if status not in KNOWN_OUTCOMES:
            errors.append(f"run contains unknown outcome: {status}")
        if record.get("history_complete") is not True:
            errors.append("request has incomplete bounded history")
        errors.extend(_request_interval_errors(record))
    return errors


def _request_interval_errors(record: Mapping[str, Any]) -> List[str]:
    """Require terminal records to retain one ordered client interval."""

    started = record.get("request_start_ns")
    ended = record.get("request_end_ns")
    if not _valid_request_bounds(started, ended):
        return ["request interval is missing or invalid"]
    first = record.get("first_content_ns")
    if first is not None and not _valid_first_content(first, started, ended):
        return ["first content timestamp is outside request interval"]
    return []


def _valid_request_bounds(started: Any, ended: Any) -> bool:
    """Check nonnegative ordered request timestamps."""

    return (
        isinstance(started, int)
        and not isinstance(started, bool)
        and isinstance(ended, int)
        and not isinstance(ended, bool)
        and started >= 0
        and ended >= started
    )


def _valid_first_content(first: Any, started: int, ended: int) -> bool:
    """Check that first content arrived inside the request interval."""

    return isinstance(first, int) and not isinstance(first, bool) and started <= first <= ended


def _parent_context_errors(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> List[str]:
    """Require observed prompt usage to meet the declared long-context bound."""

    parent = run.get("parent") if isinstance(run.get("parent"), dict) else {}
    if parent.get("status") != "completed":
        return []
    minimum = manifest["prompts"]["parent"]["minimum_prompt_tokens"]
    usage = parent.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) or prompt_tokens < minimum:
        return ["completed parent does not meet the long-context prompt-token minimum"]
    return []


def _run_records(run: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """Return parent, branch, and probe records from one run."""

    records = [run.get("parent", {})]
    records.extend(run.get("branches", []) if isinstance(run.get("branches"), list) else [])
    records.extend(run.get("probes", []) if isinstance(run.get("probes"), list) else [])
    return [item for item in records if isinstance(item, dict)]


def _frozen_run_evidence_errors(
    run: Mapping[str, Any], manifest: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Require the observations that make the frozen schedule meaningful.

    The engine kind selects the evidence contract. A metric the contract marks
    unsupported must be absent, never observed, and the shared metrics of a
    non-Leone engine must recompute from the retained wire bytes.
    """

    kind = _run_engine_kind(run, manifest)
    unsupported = _unsupported_metrics(kind)
    errors = _frozen_parent_branch_errors(run, manifest)
    probes = {item.get("role"): item for item in run.get("probes", []) if isinstance(item, dict)}
    sibling_progress = run.get("sibling_progress") if isinstance(run.get("sibling_progress"), dict) else None
    service_metrics = run.get("service_metrics") if isinstance(run.get("service_metrics"), dict) else None
    service_trace = run.get("service_trace") if isinstance(run.get("service_trace"), dict) else None
    errors.extend(_frozen_probe_errors(
        probes, sibling_progress, service_metrics, _run_records(run), service_trace,
        run.get("running_identity") if isinstance(run.get("running_identity"), Mapping) else None,
        unsupported,
    ))
    errors.extend(_frozen_token_errors(run, unsupported))
    errors.extend(_frozen_metrics_errors(run, unsupported))
    errors.extend(_frozen_limit_stop_errors(run))
    errors.extend(_frozen_wire_errors(run, unsupported))
    return errors


def _frozen_limit_stop_errors(run: Mapping[str, Any]) -> List[str]:
    """Require every counted request to stop at the token limit.

    A natural stop has no boundary rule shared by both engines, so its
    completion count is not comparable and the workload must not produce one.
    """

    counted = [run.get("parent"), *run.get("branches", [])] + [
        item for item in run.get("probes", []) if isinstance(item, Mapping) and item.get("role") != "cancel"
    ]
    return [
        "frozen run request stopped before the token limit"
        for record in counted
        if isinstance(record, Mapping) and record.get("status") == "completed"
        and record.get("finish_reason") != "length"
    ]


def _wire_events(record: Mapping[str, Any]) -> Optional[List[Tuple[Mapping[str, Any], int]]]:
    """Read the retained raw events as (event, receive time) pairs, or None when malformed."""

    rows = _record_raw_events(record)
    if rows is None:
        return None
    events = []
    for row in rows:
        event, received = (row.get("event"), row.get("received_ns")) if isinstance(row, Mapping) else (None, None)
        if not isinstance(event, Mapping) or isinstance(received, bool) or not isinstance(received, int):
            return None
        events.append((event, received))
    return events


def _wire_checks(
    record: Mapping[str, Any], events: Sequence[Tuple[Mapping[str, Any], int]], stream: bool
) -> List[Tuple[bool, str]]:
    """Pair each recomputed wire fact with the name of the record field it must match."""

    content, times, usage, _ = _content_and_timestamps(events)
    if not stream:
        content = _nonstream_content(events[0][0]) or ""
    metric = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    return [
        (sha256_bytes(content.encode()) == record.get("content_sha256"), "content digest"),
        (usage == record.get("usage"), "usage"),
        (_finish_reason([event for event, _ in events]) == record.get("finish_reason"), "finish reason"),
        (not stream or metric.get("content_receive_ns") == times, "content receive times"),
    ]


def _wire_record_errors(record: Mapping[str, Any], stream: bool) -> List[str]:
    """Recompute one record's request digest, content, usage, and finish reason from raw bytes."""

    request, events = _record_request_bytes(record), _wire_events(record)
    if request is None or sha256_bytes(request) != record.get("request_body_sha256"):
        return ["frozen run request bytes are missing or differ from the request digest"]
    if events is None:
        return ["frozen run raw stream events are missing"]
    return [
        f"frozen run {name} does not recompute from the retained wire events"
        for ok, name in _wire_checks(record, events, stream)
        if not ok
    ]


def _frozen_wire_errors(run: Mapping[str, Any], unsupported: Any) -> List[str]:
    """Require wire proof for every measured shared metric of an engine with unsupported metrics."""

    if not unsupported:
        return []
    errors = []
    parent = run.get("parent")
    if isinstance(parent, Mapping) and parent.get("status") == "completed":
        errors.extend(_wire_record_errors(parent, False))
    for record in [*run.get("branches", []), *run.get("probes", [])]:
        if isinstance(record, Mapping) and record.get("status") == "completed":
            errors.extend(_wire_record_errors(record, True))
    return errors


def _frozen_parent_branch_errors(
    run: Mapping[str, Any], manifest: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Require a completed parent and observed reuse on every branch."""

    errors = []
    parent = run.get("parent") if isinstance(run.get("parent"), dict) else {}
    if parent.get("status") != "completed":
        errors.append("frozen run parent did not complete")
    branches = run.get("branches") if isinstance(run.get("branches"), list) else []
    if not branches:
        return errors + ["frozen run lacks verified branch history reuse"]
    if not _valid_digest(parent.get("content_sha256")):
        errors.append("frozen run parent content identity is missing")
    for branch in branches:
        if not isinstance(branch, dict) or branch.get("status") != "completed":
            errors.append("frozen run branch did not complete")
            continue
        errors.extend(_history_reuse_errors(parent, branch, manifest, run.get("engine"), run))
    return errors


def _history_material_parts(material: Any) -> Optional[Tuple[List[Any], List[Any], Mapping[str, Any]]]:
    """Return canonical prefix, request messages, and body when present."""

    if not isinstance(material, Mapping):
        return None
    prefix = material.get("prefix_messages")
    request = material.get("request_messages")
    body = material.get("request_body")
    if not isinstance(prefix, list) or not isinstance(request, list) or not isinstance(body, Mapping):
        return None
    return prefix, request, body


def _history_material_hash_errors(
    history: Mapping[str, Any], prefix: Sequence[Any], request: Sequence[Any], body: Mapping[str, Any]
) -> List[str]:
    """Check hashes, counts, and body messages against canonical material."""

    errors = []
    if body.get("messages") != list(request):
        errors.append("frozen run branch request body history differs from canonical material")
    if history.get("prefix_sha256") != sha256_bytes(canonical_json(prefix)):
        errors.append("frozen run branch prefix hash does not recompute from canonical material")
    if history.get("request_sha256") != sha256_bytes(canonical_json(request)):
        errors.append("frozen run branch request hash does not recompute from canonical material")
    if history.get("request_body_sha256") != sha256_bytes(canonical_json(dict(body))):
        errors.append("frozen run branch body hash does not recompute from canonical material")
    if history.get("request_message_count") != len(request) or history.get("prefix_message_count") != len(prefix):
        errors.append("frozen run branch message counts do not recompute from canonical material")
    return errors


def _history_parent_response_error(
    prefix: Sequence[Any], parent: Mapping[str, Any]
) -> Optional[str]:
    """Require the observed parent response at the end of the branch prefix."""

    assistant = prefix[-1] if prefix else None
    valid = (
        isinstance(assistant, Mapping)
        and assistant.get("role") == "assistant"
        and isinstance(assistant.get("content"), str)
        and sha256_bytes(assistant["content"].encode()) == parent.get("content_sha256")
    )
    return None if valid else "frozen run branch prefix does not contain the observed parent response"


def _history_material_shape_errors(
    history: Mapping[str, Any], parent: Mapping[str, Any]
) -> List[str]:
    """Recompute branch request hashes from retained canonical request material."""

    parts = _history_material_parts(history.get("request_material"))
    if parts is None:
        material = history.get("request_material")
        reason = "missing" if not isinstance(material, Mapping) else "incomplete"
        return [f"frozen run branch canonical request material is {reason}"]
    prefix, request, body = parts
    errors = _history_material_hash_errors(history, prefix, request, body)
    response_error = _history_parent_response_error(prefix, parent)
    if response_error is not None:
        errors.append(response_error)
    return errors


def _history_manifest_prompts(
    material: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Optional[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Find the declared parent and branch prompts for one request material row."""

    prompts = manifest.get("prompts") if isinstance(manifest.get("prompts"), Mapping) else {}
    parent_prompt = prompts.get("parent") if isinstance(prompts, Mapping) else None
    branches = prompts.get("branches") if isinstance(prompts, Mapping) else None
    branch_id = material.get("branch_prompt_id")
    branch_prompt = next(
        (item for item in branches if isinstance(item, Mapping) and item.get("id") == branch_id),
        None,
    ) if isinstance(branches, list) else None
    if not isinstance(parent_prompt, Mapping) or not isinstance(branch_prompt, Mapping):
        return None
    return parent_prompt, branch_prompt


def _history_prompt_errors(
    material: Mapping[str, Any], parent_prompt: Mapping[str, Any], branch_prompt: Mapping[str, Any]
) -> List[str]:
    """Bind retained prefix and branch messages to declared prompt content."""

    prefix = material.get("prefix_messages")
    request = material.get("request_messages")
    expected_parent = _prompt_messages(parent_prompt)
    expected_branch = _prompt_messages(branch_prompt)
    if not isinstance(prefix, list) or not isinstance(request, list):
        return ["frozen run branch canonical request material is incomplete"]
    errors = []
    if prefix[: len(expected_parent)] != expected_parent:
        errors.append("frozen run branch parent prompt differs from the manifest")
    if len(prefix) != len(expected_parent) + 1:
        errors.append("frozen run branch prefix contains an unexpected extra turn")
    if request[len(prefix) :] != expected_branch:
        errors.append("frozen run branch prompt differs from the manifest")
    return errors


def _history_request_field_errors(
    material: Mapping[str, Any], manifest: Mapping[str, Any]
) -> List[str]:
    """Bind retained request controls to the frozen request declaration."""

    body = material.get("request_body")
    declared_request = manifest.get("request")
    if not isinstance(body, Mapping) or not isinstance(declared_request, Mapping):
        return ["frozen run branch request declaration is missing"]
    errors = []
    expected_fields = {
        "model": declared_request.get("model"),
        "max_tokens": declared_request.get("max_tokens"),
        "temperature": declared_request.get("temperature"),
        "seed": declared_request.get("seed"),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    for field, expected in expected_fields.items():
        if body.get(field) != expected:
            errors.append(f"frozen run branch request field differs from manifest: {field}")
    return errors


def _history_branch_field_errors(
    material: Mapping[str, Any], manifest: Mapping[str, Any], parent: Mapping[str, Any],
    branch: Mapping[str, Any], engine_id: Optional[str],
) -> List[str]:
    """Bind fork or cache control fields to the declared request identities."""

    branch_id = _history_branch_id(branch)
    if branch_id is None:
        return []
    body = material.get("request_body")
    if not isinstance(body, Mapping):
        return ["frozen run branch request controls are missing"]
    engine, inferred_mode = _history_engine_and_mode(manifest, engine_id, body)
    if not isinstance(engine, Mapping):
        return ["frozen run branch engine declaration is missing"]
    mode = branch.get("branch_mode") or inferred_mode
    parent_id = _history_parent_id(parent, branch_id)
    if parent_id is None:
        return ["frozen run branch request identities are missing"]
    return _history_mode_errors(mode, body, engine, parent_id, branch_id)


def _history_branch_id(branch: Mapping[str, Any]) -> Optional[str]:
    """Read one stable branch request identity."""

    value = branch.get("request_id") or branch.get("branch_id")
    return value if isinstance(value, str) and value else None


def _history_parent_id(parent: Mapping[str, Any], branch_id: str) -> Optional[str]:
    """Read or derive the parent identity for one branch request."""

    value = parent.get("request_id")
    if isinstance(value, str) and value:
        return value
    derived = branch_id.rsplit("-", 1)[0]
    return derived if derived else None


def _history_mode_errors(
    mode: Optional[str], body: Mapping[str, Any], engine: Mapping[str, Any],
    parent_id: str, branch_id: str,
) -> List[str]:
    """Validate the request controls for one declared branch mode."""

    if mode == "fork":
        return _history_fork_identity_errors(body, engine, parent_id, branch_id)
    if mode == "cached_history":
        return _history_cache_control_errors(body, engine)
    return ["frozen run branch method is unsupported"]


def _history_engine_and_mode(
    manifest: Mapping[str, Any], engine_id: Optional[str], body: Mapping[str, Any]
) -> Tuple[Optional[Mapping[str, Any]], Optional[str]]:
    """Find the declared engine that owns one retained branch request."""

    engines = manifest.get("engines")
    if not isinstance(engines, list):
        return None, None
    declared = _declared_history_engine(engines, engine_id)
    if isinstance(declared, Mapping):
        mode = "fork" if _history_fork_fields_present(declared, body) else None
        return declared, mode
    for item in engines:
        if isinstance(item, Mapping) and _history_fork_fields_present(item, body):
            return item, "fork"
    return None, None


def _declared_history_engine(
    engines: Sequence[Any], engine_id: Optional[str]
) -> Optional[Mapping[str, Any]]:
    """Find one engine declaration by stable ID."""

    for item in engines:
        if isinstance(item, Mapping) and item.get("id") == engine_id:
            return item
    return None


def _history_fork_fields_present(engine: Mapping[str, Any], body: Mapping[str, Any]) -> bool:
    """Check whether a body carries both declared fork controls."""

    method = engine.get("branch_method")
    if not isinstance(method, Mapping):
        return False
    fields = (method.get("parent_field"), method.get("session_field"))
    return all(isinstance(field, str) and field and field in body for field in fields)


def _history_fork_identity_errors(
    body: Mapping[str, Any], engine: Mapping[str, Any], parent_id: str, branch_id: str
) -> List[str]:
    """Check the two identities carried by a fork request."""

    method = engine.get("branch_method")
    if not isinstance(method, Mapping):
        return ["frozen run branch fork declaration is missing"]
    parent_field, session_field = method.get("parent_field"), method.get("session_field")
    if not _valid_control_field(parent_field) or not _valid_control_field(session_field):
        return ["frozen run branch fork fields are incomplete"]
    errors = []
    if body.get(parent_field) != parent_id:
        errors.append("frozen run branch fork parent identity differs")
    if body.get(session_field) != branch_id:
        errors.append("frozen run branch fork session identity differs")
    return errors


def _valid_control_field(value: Any) -> bool:
    """Check one declared request control field name."""

    return isinstance(value, str) and bool(value)


def _history_cache_control_errors(body: Mapping[str, Any], engine: Mapping[str, Any]) -> List[str]:
    """Check the declared same-history cache control."""

    method = engine.get("cache_method")
    field = method.get("request_field") if isinstance(method, Mapping) else None
    expected = method.get("request_value", True) if isinstance(method, Mapping) else None
    return [] if isinstance(field, str) and body.get(field) == expected else [
        "frozen run branch cache control differs"
    ]


def _history_manifest_errors(
    material: Mapping[str, Any], manifest: Mapping[str, Any], parent: Optional[Mapping[str, Any]] = None,
    branch: Optional[Mapping[str, Any]] = None, engine_id: Optional[str] = None,
) -> List[str]:
    """Bind branch material to the frozen prompt and request declarations."""

    prompts = _history_manifest_prompts(material, manifest)
    if prompts is None:
        return ["frozen run branch prompt identity is not bound to the manifest"]
    parent_prompt, branch_prompt = prompts
    errors = _history_prompt_errors(material, parent_prompt, branch_prompt)
    errors.extend(_history_request_field_errors(material, manifest))
    if isinstance(parent, Mapping) and isinstance(branch, Mapping):
        errors.extend(_history_branch_field_errors(material, manifest, parent, branch, engine_id))
    return errors


def _history_required_reuse_tokens(
    parent: Mapping[str, Any], manifest: Optional[Mapping[str, Any]]
) -> int:
    """Read the complete parent prefix size, including its assistant response."""

    usage = parent.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, Mapping) else None
    if isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0:
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens > 0:
            return prompt_tokens + completion_tokens
        return prompt_tokens
    return 1


def _history_reuse_count_error(
    history: Mapping[str, Any], parent: Mapping[str, Any], manifest: Optional[Mapping[str, Any]]
) -> Optional[str]:
    """Require an observed reuse list before tokenized prefix checks."""

    reuse = history.get("reuse_count")
    values = reuse.get("values") if isinstance(reuse, dict) else None
    if (
        not isinstance(reuse, dict)
        or reuse.get("status") != "observed"
        or not isinstance(values, list)
        or not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in values)
    ):
        return "frozen run branch reuse count is invalid"
    return None


def _history_reuse_identity_errors(
    history: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any]
) -> List[str]:
    """Require parent and body identities to match the branch record."""

    errors = []
    if history.get("parent_content_sha256") != parent.get("content_sha256"):
        errors.append("frozen run branch parent content identity differs")
    if not _valid_digest(branch.get("request_body_sha256")) or history.get("request_body_sha256") != branch.get("request_body_sha256"):
        errors.append("frozen run branch history request identity differs")
    return errors


def _history_reuse_shape_errors(history: Mapping[str, Any]) -> List[str]:
    """Require complete branch history hashes and message counts."""

    errors = []
    if not all(_valid_digest(history.get(field)) for field in ("prefix_sha256", "request_sha256")):
        errors.append("frozen run branch history hashes are incomplete")
    if (
        not isinstance(history.get("prefix_message_count"), int)
        or not isinstance(history.get("request_message_count"), int)
        or history["prefix_message_count"] < 0
        or history["request_message_count"] < 0
    ):
        errors.append("frozen run branch history message counts are incomplete")
    elif history["request_message_count"] <= history["prefix_message_count"]:
        errors.append("frozen run branch history has no appended branch message")
    return errors


def _history_tokenization_errors(
    history: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any],
    manifest: Optional[Mapping[str, Any]] = None, engine_id: Optional[str] = None,
    run: Optional[Mapping[str, Any]] = None, engine: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Require independently reconstructed plain length-stop history."""

    tokenization = history.get("tokenization")
    if not isinstance(tokenization, Mapping) or tokenization.get("status") != "observed":
        return ["frozen run branch canonical tokenization is missing"]
    material = history.get("request_material")
    parts = _history_material_parts(material)
    if parts is None:
        return ["frozen run branch canonical tokenization has no request material"]
    prefix, request, _ = parts
    declared_engine = _history_engine_declaration(manifest, engine_id, engine)
    errors = _history_token_shape_errors(tokenization)
    errors.extend(_history_token_oracle_errors(tokenization, engine_id, declared_engine))
    errors.extend(_history_token_provenance_errors(tokenization, manifest, engine_id))
    errors.extend(_history_token_material_errors(tokenization, history, prefix, request))
    errors.extend(_history_token_request_identity_errors(tokenization, parent, branch))
    errors.extend(_history_token_runtime_identity_errors(tokenization, run))
    errors.extend(_history_token_boundary_errors(tokenization, history, parent, branch))
    errors.extend(_history_claim_errors(tokenization, parent, branch, declared_engine, run))
    errors.extend(_history_slot_copy_errors(tokenization, parent, branch))
    return errors


def _history_engine_declaration(
    manifest: Optional[Mapping[str, Any]], engine_id: Optional[str], engine: Optional[Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    """Select the declared engine that owns one history record."""

    if isinstance(engine, Mapping):
        return engine
    engines = manifest.get("engines") if isinstance(manifest, Mapping) else None
    return _declared_history_engine(engines, engine_id) if isinstance(engines, list) else None


def _history_claim_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any],
    engine: Optional[Mapping[str, Any]], run: Optional[Mapping[str, Any]],
) -> List[str]:
    """Check response claims with the acceptance rule of the declared engine."""

    if isinstance(engine, Mapping) and _is_llama_cpp(engine):
        return _llama_history_claim_errors(tokenization, parent, branch, engine, run)
    return _history_signed_claim_errors(tokenization, parent, branch)


def _llama_history_claim_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any],
    engine: Mapping[str, Any], run: Optional[Mapping[str, Any]],
) -> List[str]:
    """Recompute llama.cpp history evidence from the retained raw exchanges.

    llama.cpp signs nothing. The producer module rebuilds the whole evidence
    record from the raw requests, raw responses, slot exchanges, and tokenizer
    exchanges. The retained record must equal the rebuilt one.
    """

    oracle = tokenization.get("oracle") if isinstance(tokenization.get("oracle"), Mapping) else {}
    errors = []
    if oracle.get("proof_adapter") != "llama.cpp.verbose":
        errors.append("frozen run branch llama.cpp proof adapter is missing")
    if oracle.get("evidence_class") != "retained_and_recomputed":
        errors.append("frozen run branch llama.cpp evidence class must be retained_and_recomputed")
    errors.extend(_llama_thinking_errors(oracle, engine))
    if not isinstance(run, Mapping):
        return errors + ["frozen run branch llama.cpp evidence needs its run record"]
    errors.extend(_llama_process_binding_errors(oracle, run, engine))
    errors.extend(_llama_recompute_errors(tokenization, oracle, parent, branch, engine, run))
    return errors


def _llama_thinking_errors(oracle: Mapping[str, Any], engine: Mapping[str, Any]) -> List[str]:
    """Bind the retained template policy and kwargs to the declared thinking policy."""

    policy = engine.get("thinking_policy")
    if not isinstance(policy, Mapping):
        return ["frozen run branch llama.cpp thinking policy is undeclared"]
    retained = oracle.get("special_tokens_policy")
    expected = _thinking_policy_fields(policy)
    errors = []
    if not isinstance(retained, Mapping) or any(retained.get(key) != value for key, value in expected.items()):
        errors.append("frozen run branch tokenizer policy differs from the declared thinking policy")
    kwargs = expected["chat_template_kwargs"]
    if oracle.get("apply_template_kwargs") != {"parent": kwargs, "branch": kwargs}:
        errors.append("frozen run branch apply-template kwargs differ from the declared thinking policy")
    return errors


def _llama_process_binding_errors(
    oracle: Mapping[str, Any], run: Mapping[str, Any], engine: Mapping[str, Any]
) -> List[str]:
    """Bind the pinned tokenizer binary, model, and server flags to the serving process."""

    identity = _history_identity_body(run.get("running_identity"))
    if not isinstance(identity, Mapping):
        return ["frozen run branch llama.cpp serving process identity is missing"]
    errors = []
    if identity.get("executable_sha256") != oracle.get("executable_sha256"):
        errors.append("frozen run branch tokenizer binary differs from the serving process")
    if identity.get("model_sha256") != oracle.get("gguf_sha256"):
        errors.append("frozen run branch tokenizer model differs from the serving process")
    errors.extend(_llama_flag_errors(identity, engine))
    errors.extend(_llama_template_binding_errors(identity, oracle, engine))
    return errors


def _llama_template_binding_errors(
    identity: Mapping[str, Any], oracle: Mapping[str, Any], engine: Mapping[str, Any]
) -> List[str]:
    """Require a spawned server to have served the template bytes the tokenizer used."""

    if engine.get("identity_provider") != "spawned_process":
        return []
    served = identity.get("template_sha256")
    if not _valid_digest(served) or served != oracle.get("template_bytes_sha256"):
        return ["frozen run llama.cpp served template differs from the tokenizer template"]
    return []


def _llama_flag_errors(identity: Mapping[str, Any], engine: Mapping[str, Any]) -> List[str]:
    """Require the process argv hashes and the flags the slot copy protocol needs."""

    errors = []
    if not all(_valid_digest(identity.get(field)) for field in ("argv_sha256", "argv_roles_sha256")):
        errors.append("frozen run llama.cpp process argv hashes are missing")
    flags = identity.get("flags")
    parallel = flags.get("parallel") if isinstance(flags, Mapping) else None
    if not isinstance(flags, Mapping) or flags.get("slot_save_path") is not True:
        errors.append("frozen run llama.cpp process lacks the slot save path flag")
    if isinstance(parallel, bool) or not isinstance(parallel, int) or parallel < engine.get("slot_count", 0):
        errors.append("frozen run llama.cpp process serves fewer slots than declared")
    if identity.get("workload_epoch") != identity.get("process_instance_id"):
        errors.append("frozen run llama.cpp workload epoch is not the process instance")
    return errors + _served_alias_errors(identity, engine)


def _served_alias_errors(identity: Mapping[str, Any], engine: Mapping[str, Any]) -> List[str]:
    """Require a spawned server to have been launched with the model artifact basename as its alias."""

    alias = _identity_alias(identity)
    if engine.get("identity_provider") == "spawned_process" and (alias is None or alias != _model_alias(engine)):
        return ["frozen run llama.cpp served alias differs from the model artifact"]
    return []


def _identity_alias(identity: Mapping[str, Any]) -> Optional[str]:
    """Read the `--alias` the serving process was launched with."""

    flags = identity.get("flags")
    alias = flags.get("alias") if isinstance(flags, Mapping) else None
    return alias if isinstance(alias, str) and alias else None


def _llama_recompute_errors(
    tokenization: Mapping[str, Any], oracle: Mapping[str, Any], parent: Mapping[str, Any],
    branch: Mapping[str, Any], engine: Mapping[str, Any], run: Mapping[str, Any],
) -> List[str]:
    """Rebuild the evidence record and compare it with the retained one."""

    module = _load_history_producer_module()
    recompute = getattr(module, "recompute_llama_history", None)
    if not callable(recompute):
        return ["frozen run branch llama.cpp evidence cannot recompute without the producer module"]
    context = _history_run_engine(engine, run)
    parent_id = parent.get("service_request_id") or parent.get("response_id") or parent.get("request_id")
    inputs = (
        _record_request_bytes(parent), _history_raw_events(parent, context),
        _record_request_bytes(branch), _history_raw_events(branch, context, parent_id),
    )
    if any(value is None for value in inputs):
        return ["frozen run branch llama.cpp raw request or response events are missing"]
    rebuilt = recompute(context, oracle, *inputs)
    if rebuilt.get("status") != "observed":
        return [f"frozen run branch llama.cpp evidence does not recompute: {rebuilt.get('reason')}"]
    if {field: rebuilt.get(field) for field in HISTORY_TOKEN_FIELDS} != dict(tokenization):
        return ["frozen run branch llama.cpp evidence differs from the recomputed record"]
    return []


def _history_run_engine(engine: Mapping[str, Any], run: Mapping[str, Any]) -> Dict[str, Any]:
    """Rebuild the engine mapping that the producer saw at the end of one run."""

    metrics = run.get("service_metrics") if isinstance(run.get("service_metrics"), Mapping) else {}
    return {
        **_producer_engine(engine),
        "_service_metrics": metrics.get("end"),
        "_service_trace": run.get("service_trace"),
        "_running_identity": run.get("running_identity"),
        "_slot_copy": run.get("slot_copy"),
        "served_model_alias": _identity_alias(_history_identity_body(run.get("running_identity")) or {}),
    }


def _history_slot_copy_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any]
) -> List[str]:
    """Require a slot copy exactly when the branch runs outside the parent slot."""

    oracle = tokenization.get("oracle")
    copy = oracle.get("slot_copy") if isinstance(oracle, Mapping) else None
    parent_slot, branch_slot = _record_request_slot(parent), _record_request_slot(branch)
    if parent_slot == branch_slot:
        return [] if copy is None else ["frozen run branch has a slot copy without a slot change"]
    if parent_slot is None or branch_slot is None:
        return ["frozen run branch slot identity is incomplete"]
    if not isinstance(copy, Mapping):
        return ["frozen run branch outside the parent slot lacks a slot copy"]
    errors = _slot_copy_value_errors(copy, tokenization.get("parent_evaluated_token_count"))
    if copy.get("parent_slot") != parent_slot or copy.get("branch_slot") != branch_slot:
        errors.append("frozen run branch slot copy names other slots")
    return errors


def _slot_copy_value_errors(copy: Mapping[str, Any], evaluated: Any) -> List[str]:
    """Check the counts, file identity, and exchange digests of one slot copy record."""

    errors = []
    if copy.get("saved_token_count") != evaluated or copy.get("restored_token_count") != evaluated:
        errors.append("frozen run branch slot copy does not cover the evaluated history")
    if not _valid_token_count(copy.get("byte_count")) or copy.get("byte_count") == 0:
        errors.append("frozen run branch slot copy has no bytes")
    if not isinstance(copy.get("filename"), str) or not copy["filename"]:
        errors.append("frozen run branch slot copy has no file identity")
    digests = ("save_request_sha256", "save_response_sha256", "restore_request_sha256", "restore_response_sha256")
    if not all(_valid_digest(copy.get(field)) for field in digests):
        errors.append("frozen run branch slot copy exchange digests are invalid")
    return errors


def _history_token_shape_errors(tokenization: Mapping[str, Any]) -> List[str]:
    """Validate the exact independent history evidence shape."""

    if set(tokenization) != HISTORY_TOKEN_FIELDS:
        return ["frozen run branch canonical tokenization fields are incomplete"]
    errors = []
    if tokenization.get("schema_version") != HISTORY_TOKEN_SCHEMA:
        errors.append("frozen run branch canonical tokenization schema is unsupported")
    if tokenization.get("status") != "observed":
        errors.append("frozen run branch canonical tokenization status is invalid")
    if tokenization.get("scope") != HISTORY_TOKEN_SCOPE:
        errors.append("frozen run branch canonical tokenization scope is unsupported")
    for field in ("process_instance_id", "workload_epoch"):
        if not isinstance(tokenization.get(field), (str, int)) or isinstance(tokenization.get(field), bool):
            errors.append(f"frozen run branch canonical tokenization identity is invalid: {field}")
    return errors


def _history_token_oracle_errors(
    tokenization: Mapping[str, Any], engine_id: Optional[str],
    engine: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Require pinned tokenizer, template, and raw oracle provenance."""

    oracle = tokenization.get("oracle")
    if not isinstance(oracle, Mapping):
        return ["frozen run branch tokenizer oracle is missing"]
    errors = _history_oracle_missing_errors(oracle)
    errors.extend(_history_oracle_engine_errors(oracle, engine_id, engine))
    errors.extend(_history_oracle_digest_errors(oracle))
    errors.extend(_history_oracle_library_errors(oracle))
    errors.extend(_history_oracle_tokenizer_errors(oracle))
    return errors


def _history_oracle_missing_errors(oracle: Mapping[str, Any]) -> List[str]:
    """Require each field in the independent tokenizer oracle."""

    return [
        f"frozen run branch tokenizer oracle field is missing: {field}"
        for field in HISTORY_ORACLE_REQUIRED - set(oracle)
    ]


def _history_oracle_engine_errors(
    oracle: Mapping[str, Any], engine_id: Optional[str], engine: Optional[Mapping[str, Any]] = None
) -> List[str]:
    """Bind oracle engine and source to the declared runtime."""

    declaration = engine.get("history_tokenization") if isinstance(engine, Mapping) else None
    producer = declaration.get("producer_engine") if isinstance(declaration, Mapping) else None
    if isinstance(producer, str) and producer:
        expected = {producer}
    elif engine_id is None:
        expected = None
    elif engine_id == "llama_cpp":
        expected = {engine_id, "llama.cpp"}
    else:
        expected = {engine_id}
    errors = []
    if expected is not None and oracle.get("engine") not in expected:
        errors.append("frozen run branch tokenizer oracle engine differs")
    if not _valid_commit(oracle.get("source_commit")):
        errors.append("frozen run branch tokenizer oracle source is invalid")
    return errors


def _history_oracle_digest_errors(oracle: Mapping[str, Any]) -> List[str]:
    """Require digests for the executable, model, template, and tokenizer calls."""

    fields = (
        "executable_sha256", "gguf_sha256", "template_config_sha256", "template_bytes_sha256",
        "special_tokens_policy_sha256", "apply_template_request_sha256",
        "apply_template_response_sha256", "tokenize_request_sha256", "tokenize_response_sha256",
    )
    return [
        f"frozen run branch tokenizer oracle digest is invalid: {field}"
        for field in fields
        if not _valid_digest(oracle.get(field))
    ]


def _history_oracle_library_errors(oracle: Mapping[str, Any]) -> List[str]:
    """Require hashes for every loaded tokenizer library."""

    loaded = oracle.get("loaded_library_sha256")
    values = loaded if isinstance(loaded, list) else [loaded]
    if values and all(_valid_digest(value) for value in values):
        return []
    return ["frozen run branch tokenizer oracle loaded library is invalid"]


def _history_oracle_tokenizer_errors(oracle: Mapping[str, Any]) -> List[str]:
    """Require named tokenizer metadata and a positive vocabulary size."""

    metadata = oracle.get("tokenizer_metadata_sha256")
    named_scheme = oracle.get("tokenizer_hash_scheme")
    has_metadata = _valid_digest(metadata)
    has_named_hash = _valid_digest(oracle.get("tokenizer_sha256")) and isinstance(named_scheme, str) and bool(named_scheme)
    errors = []
    if not has_metadata and not has_named_hash:
        errors.append("frozen run branch tokenizer metadata provenance is missing")
    vocabulary = oracle.get("vocab_size")
    if not isinstance(vocabulary, int) or isinstance(vocabulary, bool) or vocabulary <= 0:
        errors.append("frozen run branch tokenizer vocabulary size is invalid")
    return errors


def _history_token_id_errors(tokenization: Mapping[str, Any]) -> List[str]:
    """Validate all retained token arrays against the pinned vocabulary."""

    oracle = tokenization.get("oracle")
    vocabulary = oracle.get("vocab_size") if isinstance(oracle, Mapping) else None
    if not isinstance(vocabulary, int) or isinstance(vocabulary, bool) or vocabulary <= 0:
        return ["frozen run branch tokenization vocabulary size is invalid"]
    return _history_token_id_field_errors(
        tokenization, vocabulary,
        ("parent_prompt_token_ids", "parent_generated_token_ids", "parent_evaluated_token_ids", "request_token_ids"),
    )


def _history_token_id_field_errors(
    tokenization: Mapping[str, Any], vocabulary: int, fields: Sequence[str]
) -> List[str]:
    """Require every retained token ID to fit the declared vocabulary."""

    errors = []
    for field in fields:
        if not _history_token_id_values_valid(tokenization.get(field), vocabulary):
            errors.append(f"frozen run branch {field} are invalid")
    return errors


def _history_token_id_values_valid(values: Any, vocabulary: int) -> bool:
    """Check one nonempty token ID array against its vocabulary bound."""

    return isinstance(values, list) and bool(values) and all(
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value < vocabulary
        for value in values
    )


def _history_token_provenance_errors(
    tokenization: Mapping[str, Any], manifest: Optional[Mapping[str, Any]], engine_id: Optional[str]
) -> List[str]:
    """Bind tokenizer and template digests to the frozen engine declaration."""

    if not isinstance(manifest, Mapping):
        return []
    declaration = next(
        (
            engine.get("history_tokenization")
            for engine in manifest.get("engines", [])
            if isinstance(engine, Mapping) and engine.get("id") == engine_id
        ),
        None,
    )
    if not isinstance(declaration, Mapping):
        return ["frozen run branch tokenizer provenance is undeclared"]
    oracle = tokenization.get("oracle") if isinstance(tokenization.get("oracle"), Mapping) else {}
    values, declared = _history_token_provenance_values(oracle, declaration)
    return [
        f"frozen run branch tokenizer provenance differs: {field}"
        for field in values
        if values[field] != declared[field]
    ]


def _history_token_provenance_values(
    oracle: Mapping[str, Any], declaration: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return oracle and declaration values for every declared tokenizer pin."""

    values = {
        "tokenizer_sha256": oracle.get("tokenizer_metadata_sha256", oracle.get("tokenizer_sha256")),
        "chat_template_sha256": oracle.get("template_config_sha256"),
        "special_tokens_policy_sha256": oracle.get("special_tokens_policy_sha256"),
        "vocab_size": oracle.get("vocab_size"),
    }
    declared = {
        "tokenizer_sha256": declaration.get("tokenizer_metadata_sha256", declaration.get("tokenizer_sha256")),
        "chat_template_sha256": declaration.get("template_config_sha256", declaration.get("chat_template_sha256")),
        "special_tokens_policy_sha256": declaration.get("special_tokens_policy_sha256"),
        "vocab_size": declaration.get("vocab_size"),
    }
    identity = {
        "producer_source_commit": "source_commit",
        "producer_executable_sha256": "executable_sha256",
        "producer_model_sha256": "gguf_sha256",
        "producer_loaded_library_sha256": "loaded_library_sha256",
        "template_config_sha256": "template_config_sha256",
        "template_bytes_sha256": "template_bytes_sha256",
    }
    for declaration_field, oracle_field in identity.items():
        if declaration_field in declaration:
            value = oracle.get(oracle_field)
            if declaration_field == "producer_model_sha256" and value is None:
                value = oracle.get("model_sha256")
            values[declaration_field] = value
            declared[declaration_field] = declaration.get(declaration_field)
    return values, declared


def _history_token_material_errors(
    tokenization: Mapping[str, Any], history: Mapping[str, Any],
    prefix: Sequence[Any], request: Sequence[Any],
) -> List[str]:
    """Bind independently tokenized arrays to retained canonical messages."""

    errors = _history_token_id_errors(tokenization)
    errors.extend(_history_token_message_identity_errors(tokenization, prefix, request))
    return errors


def _history_token_message_identity_errors(
    tokenization: Mapping[str, Any], prefix: Sequence[Any], request: Sequence[Any]
) -> List[str]:
    """Bind tokenization to the exact retained canonical messages."""

    errors = []
    if tokenization.get("canonical_prefix_sha256") != sha256_bytes(canonical_json(prefix)):
        errors.append("frozen run branch token prefix message identity differs")
    if tokenization.get("canonical_request_sha256") != sha256_bytes(canonical_json(request)):
        errors.append("frozen run branch token request message identity differs")
    return errors


def _history_reuse_values_valid(values: Any) -> bool:
    """Return whether all retained reuse observations are nonnegative integers."""

    return isinstance(values, list) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in values
    )


def _history_token_reuse_value_errors(
    values: Any, expected_count: Any, request_count: Any = None
) -> List[str]:
    """Require every reuse observation to equal the exact retained prefix."""

    errors = []
    if not _history_reuse_values_valid(values) or not values:
        errors.append("frozen run branch reuse count is invalid")
    elif any(value != expected_count for value in values):
        errors.append("frozen run branch reuse count does not cover tokenized prefix")
    if request_count is not None and (
        not _valid_token_count(request_count)
        or not _valid_token_count(expected_count)
        or expected_count > request_count
    ):
        errors.append("frozen run branch reuse count exceeds tokenized request")
    return errors


def _valid_token_count(value: Any) -> bool:
    """Return whether a token count is a nonnegative integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _history_token_request_identity_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any]
) -> List[str]:
    """Bind producer request and receipt identities to the retained records."""

    parent_id = parent.get("service_request_id") or parent.get("request_id")
    branch_id = branch.get("service_request_id") or branch.get("request_id")
    errors = []
    if tokenization.get("parent_service_request_id") != parent_id:
        errors.append("frozen run branch tokenization parent request identity differs")
    if tokenization.get("service_request_id") != branch_id:
        errors.append("frozen run branch tokenization request identity differs")
    for record, field, label in (
        (parent, "parent_request_sha256", "parent"), (branch, "request_sha256", "branch")
    ):
        expected = _record_request_digest(record)
        if expected is None or tokenization.get(field) != expected:
            errors.append(f"frozen run branch tokenization {label} request digest differs")
    return errors


def _nested_mapping(value: Any, *keys: str) -> Optional[Mapping[str, Any]]:
    """Walk nested mappings by key, returning the final mapping or None."""

    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value if isinstance(value, Mapping) else None


def _history_token_runtime_identity_errors(
    tokenization: Mapping[str, Any], run: Optional[Mapping[str, Any]]
) -> List[str]:
    """Bind producer process and epoch identities to one frozen run.

    The end metrics body names the workload epoch. An engine without one names
    it in its start identity, where it equals the process instance.
    """

    if not isinstance(run, Mapping):
        return []
    body = _nested_mapping(run, "service_metrics", "end", "body")
    expected = _nested_mapping(run, "running_identity", "start", "identity")
    epoch = body if body is not None else (expected if expected and "workload_epoch" in expected else None)
    errors = []
    if expected is not None and tokenization.get("process_instance_id") != expected.get("process_instance_id"):
        errors.append("frozen run branch tokenization process identity differs")
    if epoch is not None and tokenization.get("workload_epoch") != epoch.get("workload_epoch"):
        errors.append("frozen run branch tokenization workload epoch differs")
    return errors


def _history_token_boundary_errors(
    tokenization: Mapping[str, Any], history: Mapping[str, Any],
    parent: Mapping[str, Any], branch: Mapping[str, Any],
) -> List[str]:
    """Recompute the plain length-stop evaluated prefix and exact reuse."""

    errors = _history_token_shape_value_errors(tokenization)
    errors.extend(_history_reuse_count_match_errors(tokenization, history))
    errors.extend(_history_token_reuse_identity_errors(tokenization, branch))
    errors.extend(_history_parent_usage_errors(tokenization, parent))
    return errors


def _history_parent_usage_errors(tokenization: Mapping[str, Any], parent: Mapping[str, Any]) -> List[str]:
    """Bind the parent usage counts to the prompt and response token IDs of the proof.

    The non-stream parent has no timed token markers. Its usage counts are the
    only token facts the client observed, so they must equal the lengths of the
    signed (Leone) or recomputed (llama.cpp) prompt and response ID arrays.
    """

    usage = parent.get("usage")
    prompt = tokenization.get("parent_prompt_token_ids")
    generated = tokenization.get("parent_generated_token_ids")
    if not isinstance(usage, Mapping) or not isinstance(prompt, list) or not isinstance(generated, list):
        return ["frozen run parent usage or token IDs are missing"]
    errors = []
    for field, ids in (("prompt_tokens", prompt), ("completion_tokens", generated)):
        if type(usage.get(field)) is not int or usage[field] != len(ids):
            errors.append(f"frozen run parent usage {field} differs from the proof token IDs")
    return errors


def _history_token_shape_value_errors(tokenization: Mapping[str, Any]) -> List[str]:
    """Require exact token arrays and the reviewed plain stop boundary."""

    errors = _history_token_id_errors(tokenization)
    parent_prompt = tokenization.get("parent_prompt_token_ids")
    generated = tokenization.get("parent_generated_token_ids")
    evaluated = tokenization.get("parent_evaluated_token_ids")
    request = tokenization.get("request_token_ids")
    if not all(isinstance(value, list) for value in (parent_prompt, generated, evaluated, request)):
        return errors + ["frozen run branch tokenization arrays are missing"]
    if len(generated) < 2:
        errors.append("frozen run branch plain stop needs at least two generated tokens")
        return errors
    expected_count = len(parent_prompt) + len(generated) - 1
    expected = [*parent_prompt, *generated][:expected_count]
    if tokenization.get("parent_evaluated_token_count") != expected_count:
        errors.append("frozen run branch evaluated token count differs from plain stop")
    if tokenization.get("expected_reused_token_count") != expected_count:
        errors.append("frozen run branch expected reuse count differs from evaluated history")
    if evaluated != expected:
        errors.append("frozen run branch evaluated tokens differ from prompt and response")
    if len(evaluated) <= len(parent_prompt):
        errors.append("frozen run branch evaluated history lacks generated context")
    if len(request) <= len(evaluated) or request[: len(evaluated)] != evaluated:
        errors.append("frozen run branch evaluated history is not a strict request prefix")
    return errors


def _history_reuse_count_match_errors(tokenization: Mapping[str, Any], history: Mapping[str, Any]) -> List[str]:
    """Require signed reuse and every retained telemetry value to equal E."""

    expected = tokenization.get("expected_reused_token_count")
    observed = tokenization.get("observed_reused_token_count")
    values = history.get("reuse_count", {}).get("values") if isinstance(history.get("reuse_count"), Mapping) else None
    errors = []
    if not isinstance(expected, int) or not isinstance(observed, int) or observed != expected:
        errors.append("frozen run branch signed reuse count differs from evaluated history")
    errors.extend(_history_token_reuse_value_errors(values, expected))
    return errors


def _history_token_reuse_identity_errors(
    tokenization: Mapping[str, Any], branch: Mapping[str, Any]
) -> List[str]:
    """Bind tokenized reuse to the branch service request."""

    service_id = branch.get("service_request_id") or branch.get("request_id")
    return [] if tokenization.get("service_request_id") == service_id else [
        "frozen run branch tokenization request identity differs"
    ]


def _history_token_uint32_sha256(values: Sequence[Any]) -> Optional[str]:
    """Hash token IDs with the receipt's concatenated little-endian uint32 rule."""

    if not values or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 0xFFFFFFFF for value in values):
        return None
    return sha256_bytes(b"".join(value.to_bytes(4, "little") for value in values))


def _record_request_digest(record: Mapping[str, Any]) -> Optional[str]:
    """Hash the exact request bytes retained by the runner."""

    value = record.get("_request_bytes_hex")
    if not isinstance(value, str) or not value:
        return None
    try:
        return sha256_bytes(bytes.fromhex(value))
    except ValueError:
        return None


def _record_receipt(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Read one signed response receipt from a retained response."""

    direct = record.get("response_receipt")
    if isinstance(direct, Mapping):
        return direct
    for item in record.get("_raw_events", []) if isinstance(record.get("_raw_events"), list) else []:
        event = item.get("event") if isinstance(item, Mapping) else None
        receipt = event.get("leone_receipt") if isinstance(event, Mapping) else None
        if isinstance(receipt, Mapping):
            return receipt
    return None


def _history_signed_claim_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any]
) -> List[str]:
    """Match independent arrays to the signed parent and branch claims."""

    parent_receipt = _record_receipt(parent)
    branch_receipt = _record_receipt(branch)
    errors = _history_receipt_digest_errors(tokenization, parent, branch, parent_receipt, branch_receipt)
    errors.extend(_history_receipt_shape_errors(parent_receipt, "parent"))
    errors.extend(_history_receipt_shape_errors(branch_receipt, "branch"))
    parent_claim = parent_receipt.get("claim") if isinstance(parent_receipt, Mapping) else None
    branch_claim = branch_receipt.get("claim") if isinstance(branch_receipt, Mapping) else None
    if not isinstance(parent_claim, Mapping) or not isinstance(branch_claim, Mapping):
        return errors + ["frozen run branch signed response claims are missing"]
    if any(not isinstance(tokenization.get(field), list) for field in (
        "parent_prompt_token_ids", "parent_generated_token_ids", "request_token_ids"
    )):
        return errors + ["frozen run branch signed response claim arrays are missing"]
    errors.extend(_history_parent_claim_errors(tokenization, parent_claim))
    errors.extend(_history_branch_claim_errors(tokenization, branch, branch_claim))
    return errors


def _history_receipt_shape_errors(receipt: Optional[Mapping[str, Any]], label: str) -> List[str]:
    """Require the retained receipt to carry a verifiable signature shape."""

    if not isinstance(receipt, Mapping):
        return [f"frozen run branch {label} response receipt is missing"]
    public_key = receipt.get("public_key_ed25519")
    signature = receipt.get("signature_ed25519")
    valid_key = isinstance(public_key, str) and len(public_key) == 64 and all(
        character in "0123456789abcdef" for character in public_key
    )
    valid_signature = isinstance(signature, str) and len(signature) == 128 and all(
        character in "0123456789abcdef" for character in signature
    )
    return [] if valid_key and valid_signature else [
        f"frozen run branch {label} response receipt signature shape is invalid"
    ]


def _history_receipt_digest_errors(
    tokenization: Mapping[str, Any], parent: Mapping[str, Any], branch: Mapping[str, Any],
    parent_receipt: Optional[Mapping[str, Any]], branch_receipt: Optional[Mapping[str, Any]],
) -> List[str]:
    """Bind producer receipt digests to the retained response objects."""

    errors = []
    for label, evidence, record, receipt in (
        ("parent", "parent_receipt_sha256", parent, parent_receipt),
        ("branch", "branch_receipt_sha256", branch, branch_receipt),
    ):
        expected = record.get("_response_receipt_sha256")
        if receipt is None or not _valid_digest(expected) or tokenization.get(evidence) != expected:
            errors.append(f"frozen run branch {label} response receipt identity is missing")
        elif tokenization.get(evidence) != sha256_bytes(canonical_json(receipt)):
            errors.append(f"frozen run branch {label} response receipt digest does not recompute")
    return errors


def _history_parent_claim_errors(tokenization: Mapping[str, Any], claim: Mapping[str, Any]) -> List[str]:
    """Match the parent response claim to P, G, and P plus G."""

    prompt = tokenization["parent_prompt_token_ids"]
    generated = tokenization["parent_generated_token_ids"]
    errors = _history_claim_token_errors(
        claim, _history_token_uint32_sha256(prompt), _history_token_uint32_sha256(generated),
        _history_token_uint32_sha256([*prompt, *generated]),
        len(prompt), len(generated), len(prompt) + len(generated),
    )
    if claim.get("finish_reason") != "length" or claim.get("cancelled") is not False:
        errors.append("frozen run branch parent claim is not a plain length completion")
    return errors


def _history_branch_claim_errors(
    tokenization: Mapping[str, Any], branch: Mapping[str, Any], claim: Mapping[str, Any]
) -> List[str]:
    """Match the branch prompt and signed cache counts to the evidence."""

    request = tokenization["request_token_ids"]
    errors = _history_claim_token_errors(
        claim, _history_token_uint32_sha256(request), None, None, len(request), None, None
    )
    session = claim.get("session")
    expected = tokenization.get("expected_reused_token_count")
    observed = tokenization.get("observed_reused_token_count")
    branch_id = branch.get("service_request_id") or branch.get("request_id")
    if not isinstance(session, Mapping):
        return errors + ["frozen run branch signed session claim is missing"]
    if session.get("session_id") != branch_id:
        errors.append("frozen run branch signed session identity differs")
    if session.get("cached_tokens") != expected:
        errors.append("frozen run branch signed cached token count differs")
    if session.get("reused_tokens") != observed:
        errors.append("frozen run branch signed reused token count differs")
    return errors


def _history_claim_token_errors(
    claim: Mapping[str, Any], prompt_digest: Optional[str], response_digest: Optional[str],
    transcript_digest: Optional[str], prompt_count: int, response_count: Optional[int],
    transcript_count: Optional[int],
) -> List[str]:
    """Compare one signed claim's token hashes and counts with oracle arrays."""

    errors = []
    for field, expected, count in (
        ("prompt_tokens_sha256", prompt_digest, prompt_count),
        ("response_tokens_sha256", response_digest, response_count),
        ("transcript_sha256", transcript_digest, transcript_count),
    ):
        if expected is not None and claim.get(field) != expected:
            errors.append(f"frozen run branch signed claim differs: {field}")
        count_field = {
            "prompt_tokens_sha256": "prompt_tokens",
            "response_tokens_sha256": "generated_tokens",
            "transcript_sha256": None,
        }[field]
        if count is not None and count_field is not None and claim.get(count_field) != count:
            errors.append(f"frozen run branch signed claim count differs: {count_field}")
    return errors


def _history_reuse_errors(
    parent: Mapping[str, Any],
    branch: Mapping[str, Any],
    manifest: Optional[Mapping[str, Any]] = None,
    engine_id: Optional[str] = None,
    run: Optional[Mapping[str, Any]] = None,
    engine: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Check branch reuse claims against the recorded request identities."""

    history = branch.get("history_reuse")
    if not isinstance(history, dict) or history.get("status") != "observed":
        return ["frozen run lacks verified branch history reuse"]
    errors = []
    count_error = _history_reuse_count_error(history, parent, manifest)
    if count_error is not None:
        errors.append(count_error)
    errors.extend(_history_reuse_identity_errors(history, parent, branch))
    errors.extend(_history_reuse_shape_errors(history))
    errors.extend(_history_material_shape_errors(history, parent))
    errors.extend(_history_tokenization_errors(history, parent, branch, manifest, engine_id, run, engine))
    if manifest is not None and isinstance(history.get("request_material"), Mapping):
        errors.extend(_history_manifest_errors(
            history["request_material"], manifest, parent, branch, engine_id
        ))
    return errors


def _frozen_probe_errors(
    probes: Mapping[str, Mapping[str, Any]],
    sibling_progress: Optional[Mapping[str, Any]] = None,
    service_metrics: Optional[Mapping[str, Any]] = None,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
    service_trace: Optional[Mapping[str, Any]] = None,
    running_identity: Optional[Mapping[str, Any]] = None,
    unsupported: Any = frozenset(),
) -> List[str]:
    """Require completion, cancellation, and backpressure probe evidence."""

    errors = _frozen_probe_outcome_errors(probes, unsupported)
    if "backpressure_observed" in unsupported:
        errors.extend(_unsupported_pressure_errors(probes, sibling_progress))
        return errors
    errors.extend(_frozen_pressure_errors(
        probes, sibling_progress, service_metrics, records, service_trace, running_identity
    ))
    return errors


def _frozen_probe_outcome_errors(
    probes: Mapping[str, Mapping[str, Any]], unsupported: Any = frozenset()
) -> List[str]:
    """Require completion and cancellation outcomes for the bounded probes."""

    errors = _frozen_new_prompt_errors(probes.get("new_prompt"))
    errors.extend(_frozen_cancel_errors(probes.get("cancel"), "cancel_latency_p95_ms" in unsupported))
    errors.extend(_frozen_slow_reader_errors(probes.get("slow_reader")))
    return errors


def _frozen_new_prompt_errors(value: Any) -> List[str]:
    """Require completion of the independent new prompt."""

    return [] if isinstance(value, dict) and value.get("status") == "completed" else [
        "frozen run new prompt did not complete"
    ]


def _unsupported_cancel_errors(value: Any) -> List[str]:
    """Require an engine without cancel acknowledgement to report none, never a Leone shape."""

    if not isinstance(value, Mapping) or value.get("status") != "cancel_acknowledgement_unavailable":
        return ["frozen run cancel probe must end as cancel_acknowledgement_unavailable"]
    acknowledgement = value.get("cancel_ack")
    metrics = value.get("service_metrics")
    latency = metrics.get("cancel_latency_ns") if isinstance(metrics, Mapping) else None
    errors = []
    if not isinstance(acknowledgement, Mapping) or acknowledgement.get("status") == "observed":
        errors.append("frozen run unsupported cancel acknowledgement is present or missing its typed state")
    if isinstance(latency, Mapping) and latency.get("status") == "observed":
        errors.append("frozen run unsupported cancel latency is observed")
    return errors


def _unsupported_pressure_errors(
    probes: Mapping[str, Mapping[str, Any]], sibling_progress: Optional[Mapping[str, Any]]
) -> List[str]:
    """Reject observed backpressure or sibling progress for an engine without server events."""

    slow = probes.get("slow_reader")
    pressure = slow.get("backpressure") if isinstance(slow, Mapping) else None
    errors = []
    if isinstance(pressure, Mapping) and pressure.get("status") == "observed":
        errors.append("frozen run unsupported backpressure is observed")
    if isinstance(sibling_progress, Mapping) and sibling_progress.get("status") == "observed":
        errors.append("frozen run unsupported sibling progress is observed")
    return errors


def _frozen_cancel_errors(value: Any, unsupported: bool = False) -> List[str]:
    """Require cancellation acknowledgement and reclamation."""

    if unsupported:
        return _unsupported_cancel_errors(value)
    if not isinstance(value, Mapping) or value.get("status") != "cancelled":
        return ["frozen run lacks cancellation acknowledgement and terminal latency"]
    if not _cancel_ack_valid(value):
        return ["frozen run lacks cancellation acknowledgement and terminal latency"]
    if not _cancel_latency_valid(value):
        return ["frozen run lacks cancellation acknowledgement and terminal latency"]
    return []


def _cancel_ack_valid(value: Mapping[str, Any]) -> bool:
    """Check cancellation identity and server acknowledgement."""

    request_id = value.get("service_request_id")
    acknowledgement = value.get("cancel_ack")
    return (
        isinstance(request_id, str)
        and isinstance(acknowledgement, Mapping)
        and acknowledgement.get("status") == "observed"
        and acknowledgement.get("request_id") == request_id
    )


def _cancel_latency_valid(value: Mapping[str, Any]) -> bool:
    """Check terminal cancellation latency evidence."""

    metrics = value.get("service_metrics")
    latency = metrics.get("cancel_latency_ns") if isinstance(metrics, Mapping) else None
    return _observed_measurement(latency)


def _frozen_slow_reader_errors(value: Any) -> List[str]:
    """Require completion and service identity for the slow reader."""

    valid = isinstance(value, dict) and value.get("status") == "completed"
    errors = [] if valid else ["frozen run slow-reader probe did not complete"]
    if not isinstance(value, dict) or not isinstance(value.get("service_request_id"), str):
        errors.append("frozen run slow-reader service request identity is missing")
    return errors


def _frozen_pressure_sources(
    slow: Mapping[str, Any], service_metrics: Optional[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Recompute pressure evidence and select one request-bound service row."""

    end_snapshot = service_metrics.get("end") if isinstance(service_metrics, dict) else {}
    request_id = slow.get("service_request_id") or slow.get("request_id")
    direct = _backpressure_evidence(slow)
    if direct.get("status") == "observed" and direct.get("request_id") != request_id:
        direct = _pressure_unavailable("pressure_event_names_another_request")
    snapshot = _snapshot_backpressure(end_snapshot, request_id)
    effective = snapshot if snapshot.get("status") == "observed" else direct
    return direct, snapshot, effective


def _pressure_unavailable(reason: str) -> Dict[str, Any]:
    """Return an explicit unavailable state for an invalid pressure source."""

    return {"status": "unavailable", "reason": reason}


def _frozen_pressure_claim_errors(
    backpressure: Mapping[str, Any],
    direct: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    request_id: Any,
) -> List[str]:
    """Require a backpressure claim to match retained server evidence."""

    errors = []
    selected = snapshot if snapshot.get("status") == "observed" else direct
    if backpressure != selected:
        errors.append("frozen run backpressure claim does not recompute from selected service evidence")
    if backpressure.get("status") == "observed" and backpressure.get("request_id") != request_id:
        errors.append("frozen run backpressure evidence names another request")
    if selected.get("status") != "observed":
        errors.append("frozen run lacks server backpressure evidence")
    if selected.get("status") == "observed" and selected.get("request_id") != request_id:
        errors.append("frozen run backpressure evidence names another request")
    if selected.get("status") == "observed" and selected.get("workload_epoch") is None:
        errors.append("frozen run backpressure workload epoch is missing")
    return errors


def _frozen_sibling_errors(
    sibling_progress: Optional[Mapping[str, Any]],
    records: Optional[Sequence[Mapping[str, Any]]],
    pressure: Mapping[str, Any],
    service_trace: Optional[Mapping[str, Any]] = None,
    service_metrics: Optional[Mapping[str, Any]] = None,
    running_identity: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Require sibling progress to fall inside the measured pressure interval."""

    expected = _sibling_progress(records or [], pressure, service_trace, service_metrics, running_identity)
    errors = [] if sibling_progress == expected else [
        "frozen run sibling progress claim does not recompute from request events"
    ]
    if not isinstance(sibling_progress, Mapping) or sibling_progress.get("status") != "observed":
        errors.append("frozen run lacks sibling progress during slow-reader pressure")
    return errors


def _frozen_pressure_errors(
    probes: Mapping[str, Mapping[str, Any]],
    sibling_progress: Optional[Mapping[str, Any]],
    service_metrics: Optional[Mapping[str, Any]],
    records: Optional[Sequence[Mapping[str, Any]]],
    service_trace: Optional[Mapping[str, Any]] = None,
    running_identity: Optional[Mapping[str, Any]] = None,
) -> List[str]:
    """Require pressure identity and sibling progress during that interval."""

    slow = probes.get("slow_reader") if isinstance(probes.get("slow_reader"), dict) else {}
    backpressure = slow.get("backpressure") if isinstance(slow.get("backpressure"), dict) else {}
    request_id = slow.get("service_request_id") or slow.get("request_id")
    direct_pressure, snapshot_pressure, effective_pressure = _frozen_pressure_sources(slow, service_metrics)
    errors = _frozen_pressure_claim_errors(backpressure, direct_pressure, snapshot_pressure, request_id)
    errors.extend(_frozen_sibling_errors(
        sibling_progress, records, effective_pressure, service_trace, service_metrics, running_identity
    ))
    return errors


def _frozen_token_errors(run: Mapping[str, Any], unsupported: Any = frozenset()) -> List[str]:
    """Require complete token boundary histories for completed streamed requests.

    The non-stream parent has no client timing by construction, so only its
    completion count is required here.
    """

    errors = []
    boundaries_unsupported = "inter_token_latency_p95_ms" in unsupported
    for item in _run_records(run):
        if item.get("status") != "completed":
            continue
        if item is run.get("parent"):
            errors.extend(_parent_count_errors(item))
        else:
            errors.extend(_strict_token_record_errors(item, boundaries_unsupported))
    return errors


def _parent_count_errors(parent: Mapping[str, Any]) -> List[str]:
    """Require a positive completion count on the completed non-stream parent."""

    usage = parent.get("usage")
    count = usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    if isinstance(count, int) and not isinstance(count, bool) and count > 0:
        return []
    return ["frozen run token usage count is missing"]


def _client_timing_record_errors(record: Mapping[str, Any]) -> List[str]:
    """Require client content timing and a usage count, and no token boundary claim.

    An SSE chunk is not a token: chunks can merge tokens or carry none. An
    engine without a per-token marker reports no inter-token latency at all.
    """

    metric = record.get("metrics")
    if not isinstance(metric, dict):
        return ["frozen run lacks client content timing"]
    errors = _parent_count_errors(record)
    p95 = metric.get("inter_token_latency_p95_ms")
    if record.get("token_boundary_complete") is True or metric.get("token_boundaries_verified") is True:
        errors.append("frozen run claims token boundaries without a token marker")
    if metric.get("token_boundary_receive_ns") or (isinstance(p95, Mapping) and p95.get("status") == "observed"):
        errors.append("frozen run claims inter-token latency without a token marker")
    errors.extend(_strict_content_timing_errors(record, metric))
    return errors


def _strict_token_record_errors(record: Mapping[str, Any], boundaries_unsupported: bool = False) -> List[str]:
    """Require token boundaries, usage, and client timing to describe one response."""

    if boundaries_unsupported:
        return _client_timing_record_errors(record)
    metric = record.get("metrics")
    usage = record.get("usage")
    if record.get("token_boundary_complete") is not True or not isinstance(metric, dict):
        return ["frozen run lacks complete token boundary history"]
    count = usage.get("completion_tokens") if isinstance(usage, dict) else None
    boundaries = metric.get("token_boundary_receive_ns")
    errors = _strict_token_usage_errors(metric, boundaries, count)
    errors.extend(_strict_token_timing_errors(record, metric, boundaries))
    errors.extend(_strict_content_timing_errors(record, metric))
    return errors


def _strict_token_usage_errors(metric: Mapping[str, Any], boundaries: Any, count: Any) -> List[str]:
    """Require token count and raw boundary count to agree."""

    errors = []
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        errors.append("frozen run token usage count is missing")
    if not isinstance(boundaries, list) or len(boundaries) != count:
        errors.append("frozen run token boundaries do not match usage count")
    if metric.get("token_boundaries_verified") is not True:
        errors.append("frozen run token boundary verification is missing")
    indexes = metric.get("token_boundary_indexes")
    if not isinstance(indexes, list) or not _valid_token_indexes(indexes, {"completion_tokens": count}):
        errors.append("frozen run token boundary indexes are missing or invalid")
    return errors


def _strict_token_timing_errors(
    record: Mapping[str, Any], metric: Mapping[str, Any], boundaries: Any
) -> List[str]:
    """Require ordered token boundaries after request start."""

    errors = []
    if metric.get("request_start_ns") != record.get("request_start_ns"):
        errors.append("frozen run token timing start differs from request interval")
    if isinstance(boundaries, list):
        try:
            _check_timestamps(record.get("request_start_ns"), boundaries)
        except (TypeError, ValueError):
            errors.append("frozen run token boundary timestamps are invalid")
    return errors


def _strict_content_timing_errors(record: Mapping[str, Any], metric: Mapping[str, Any]) -> List[str]:
    """Require ordered content timestamps inside the request interval."""

    errors = []
    content = metric.get("content_receive_ns")
    if not isinstance(content, list) or not content:
        errors.append("frozen run content timing is missing")
    else:
        try:
            _check_timestamps(record.get("request_start_ns"), content)
        except (TypeError, ValueError):
            errors.append("frozen run content timestamps are invalid")
        ended = record.get("request_end_ns")
        if isinstance(ended, int) and any(value > ended for value in content if isinstance(value, int)):
            errors.append("frozen run content timestamp exceeds request interval")
        if record.get("first_content_ns") != content[0]:
            errors.append("frozen run first content timestamp differs from raw history")
    return errors


SNAPSHOT_METRICS = {
    "fork_latency_p95_ms", "cancel_latency_p95_ms", "backpressure_observed", "physical_memory_peak_bytes",
}


def _unsupported_metrics_errors(run: Mapping[str, Any]) -> List[str]:
    """Reject observed service snapshots, traces, and fork latency for an engine that has none."""

    snapshots = run.get("service_metrics")
    trace = run.get("service_trace")
    observed = [
        isinstance(snapshots, Mapping) and isinstance(snapshots.get(side), Mapping)
        and snapshots[side].get("status") == "observed"
        for side in ("start", "end")
    ]
    observed.append(isinstance(trace, Mapping) and trace.get("status") == "observed")
    for branch in run.get("branches", []):
        metrics = branch.get("service_metrics") if isinstance(branch, Mapping) else None
        fork = metrics.get("fork_latency_ns") if isinstance(metrics, Mapping) else None
        observed.append(isinstance(fork, Mapping) and fork.get("status") == "observed")
    return ["frozen run unsupported service telemetry is observed"] if any(observed) else []


def _frozen_metrics_errors(run: Mapping[str, Any], unsupported: Any = frozenset()) -> List[str]:
    """Require start and end snapshots from one declared workload epoch."""

    if SNAPSHOT_METRICS <= set(unsupported):
        return _unsupported_metrics_errors(run)
    start, end = _metric_snapshot_pair(run)
    if not _observed_snapshot_pair(start, end):
        return ["frozen run lacks observed service metric snapshots"]
    shape_errors = _metric_snapshot_shape_errors(start, end)
    if shape_errors:
        return shape_errors
    errors = _metric_snapshot_delta_errors(start, end)
    errors.extend(_memory_snapshot_pair_errors(start["body"], end["body"]))
    errors.extend(_memory_snapshot_identity_errors(start["body"], end["body"], run))
    errors.extend(_snapshot_history_errors(start["body"], end["body"], run))
    return errors


def _memory_snapshot_pair_errors(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> List[str]:
    """Require lossless physical memory collection across one workload."""

    errors = []
    for label, body in (("start", start), ("end", end)):
        if not _memory_snapshot_usable(body):
            errors.append(f"frozen run {label} physical memory evidence is degraded or truncated")
    errors.extend(_memory_snapshot_continuity_errors(start, end))
    return errors


def _memory_snapshot_continuity_errors(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> List[str]:
    """Require one workload epoch and monotonic selected ledger peak."""

    errors = []
    start_epoch = _snapshot_epoch({"body": start})
    end_epoch = _snapshot_epoch({"body": end})
    if start_epoch is not None and end_epoch is not None and start_epoch != end_epoch:
        errors.append("frozen run physical memory snapshots use different workload epochs")
    for field in ("memory_topology", "physical_tracker_ledger"):
        if start.get(field) != end.get(field):
            errors.append(f"frozen run physical memory snapshots change selected ledger: {field}")
    start_peak = _memory_measurement_value(start.get("physical_tracker_peak_bytes"))
    end_peak = _memory_measurement_value(end.get("physical_tracker_peak_bytes"))
    if start_peak is not None and end_peak is not None and end_peak < start_peak:
        errors.append("frozen run physical memory peak resets during workload")
    return errors


def _memory_snapshot_identity_errors(
    start: Mapping[str, Any], end: Mapping[str, Any], run: Mapping[str, Any]
) -> List[str]:
    """Bind memory snapshots to one producer process and source identity."""

    errors = []
    for field in ("process_instance_id", "source_id", "workload_epoch"):
        error = _memory_snapshot_identity_field_error(start, end, field)
        if error is not None:
            errors.append(error)
    identity = run.get("running_identity") if isinstance(run.get("running_identity"), Mapping) else None
    start_observation = identity.get("start") if isinstance(identity, Mapping) else None
    expected = start_observation.get("identity") if isinstance(start_observation, Mapping) else None
    errors.extend(_memory_snapshot_expected_identity_errors(start, expected))
    return errors


def _memory_snapshot_identity_field_error(
    start: Mapping[str, Any], end: Mapping[str, Any], field: str
) -> Optional[str]:
    """Compare one producer identity field across memory snapshots."""

    start_value, end_value = start.get(field), end.get(field)
    if not isinstance(start_value, str) or not start_value or not isinstance(end_value, str) or not end_value:
        return f"frozen run physical memory {field} identity is missing"
    return None if start_value == end_value else f"frozen run physical memory snapshots change identity: {field}"


def _memory_snapshot_expected_identity_errors(
    start: Mapping[str, Any], expected: Any
) -> List[str]:
    """Compare the selected memory snapshot with the run identity."""

    if not isinstance(expected, Mapping):
        return []
    return [
        f"frozen run physical memory identity differs: {field}"
        for field in ("process_instance_id", "source_id")
        if isinstance(expected.get(field), str) and start.get(field) != expected[field]
    ]


def _metric_snapshot_pair(run: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Read start and end service snapshots from a run."""

    snapshots = run.get("service_metrics") if isinstance(run.get("service_metrics"), dict) else {}
    start = snapshots.get("start") if isinstance(snapshots.get("start"), dict) else {}
    end = snapshots.get("end") if isinstance(snapshots.get("end"), dict) else {}
    return start, end


def _observed_snapshot_pair(start: Mapping[str, Any], end: Mapping[str, Any]) -> bool:
    """Check that both service snapshots reported observed bodies."""

    return start.get("status") == "observed" and end.get("status") == "observed"


def _metric_snapshot_shape_errors(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> List[str]:
    """Validate both snapshot bodies and loss counters."""

    errors = _snapshot_shape_errors(start)
    errors.extend(_snapshot_shape_errors(end))
    if errors:
        return errors
    return ["frozen run service metric history is truncated"] if any(
        _snapshot_history_truncated(snapshot["body"]) for snapshot in (start, end)
    ) else []


def _metric_snapshot_delta_errors(
    start: Mapping[str, Any], end: Mapping[str, Any]
) -> List[str]:
    """Validate request count, outcomes, and workload epoch continuity."""

    start_body, end_body = start["body"], end["body"]
    start_count, end_count = start_body.get("request_count"), end_body.get("request_count")
    if not _valid_count_delta(start_count, end_count):
        return ["frozen run service metric request delta is unavailable"]
    if not end_body.get("outcomes"):
        return ["frozen run service metric outcomes are empty"]
    start_epoch, end_epoch = _snapshot_epoch(start), _snapshot_epoch(end)
    if start_epoch is None or end_epoch is None:
        return ["frozen run lacks service metric workload epoch"]
    return [] if start_epoch == end_epoch else [
        "frozen run service metric snapshots use different workload epochs"
    ]


def _valid_count_delta(start: Any, end: Any) -> bool:
    """Require a positive terminal request count delta."""

    return _nonnegative_int(start) and _positive_int(end) and end > start


def _snapshot_history_truncated(body: Mapping[str, Any]) -> bool:
    """Return whether a required service history discarded any observations."""

    for field in ("cancellation", "slow_client", "slow_client_intervals"):
        history = body.get(field)
        if isinstance(history, dict) and history.get("dropped_count") != 0:
            return True
    return False


def _snapshot_history_errors(
    start_body: Mapping[str, Any], end_body: Mapping[str, Any], run: Mapping[str, Any]
) -> List[str]:
    """Bind one metrics delta to the request IDs and outcomes in one run."""

    start_history, end_history = _snapshot_history_pair(start_body, end_body)
    if start_history is None or end_history is None:
        return ["frozen run service metric request history is missing"]
    if start_history.get("dropped_count") != 0 or end_history.get("dropped_count") != 0:
        return ["frozen run service metric request history is truncated"]
    start_rows = _snapshot_request_rows(start_history)
    end_rows = _snapshot_request_rows(end_history)
    if start_rows is None or end_rows is None:
        return ["frozen run service metric request identities are invalid"]
    expected, error = _expected_service_outcomes(run)
    if error:
        return [error]
    if _service_ids_preexist(start_rows, expected):
        return ["frozen run service request was present before the workload"]
    return _snapshot_history_delta_errors(start_body, end_body, end_rows, expected)


def _snapshot_history_pair(
    start_body: Mapping[str, Any], end_body: Mapping[str, Any]
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """Read both canonical terminal request histories."""

    return _request_history(start_body), _request_history(end_body)


def _service_ids_preexist(rows: Sequence[Mapping[str, Any]], expected: Mapping[str, str]) -> bool:
    """Check that no workload request was present in the start snapshot."""

    return bool({row["request_id"] for row in rows} & set(expected))


def _snapshot_history_delta_errors(
    start_body: Mapping[str, Any], end_body: Mapping[str, Any],
    end_rows: Sequence[Mapping[str, Any]], expected: Mapping[str, str],
) -> List[str]:
    """Bind terminal rows, counts, and cumulative outcomes to one run."""

    end_by_id = {row["request_id"]: row for row in end_rows}
    errors = _service_row_outcome_errors(expected, end_by_id)
    if errors:
        return errors
    if end_body.get("request_count") - start_body.get("request_count") != len(expected):
        return ["frozen run service metric request delta does not match request identities"]
    return _service_outcome_delta_errors(start_body, end_body, expected)


def _expected_service_outcomes(
    run: Mapping[str, Any],
) -> Tuple[Dict[str, str], Optional[str]]:
    """Map every terminal harness record to its service identity and outcome."""

    records = _run_records(run)
    expected = {
        record.get("service_request_id"): SERVICE_OUTCOMES[record.get("status")]
        for record in records
        if record.get("service_request_id") and record.get("status") in SERVICE_OUTCOMES
    }
    terminal_count = sum(record.get("status") in SERVICE_OUTCOMES for record in records)
    if len(expected) != terminal_count:
        return {}, "frozen run lacks service request identity for one outcome"
    return expected, None


def _service_row_outcome_errors(
    expected: Mapping[str, str], end_by_id: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    """Check service history outcome labels for one workload delta."""

    return [
        "frozen run service request outcome does not match receipt"
        for request_id, outcome in expected.items()
        if end_by_id.get(request_id, {}).get("outcome") != outcome
    ]


def _service_outcome_delta_errors(
    start_body: Mapping[str, Any], end_body: Mapping[str, Any], expected: Mapping[str, str]
) -> List[str]:
    """Check cumulative service outcome counters against one run."""

    expected_outcomes = Counter(expected.values())
    actual_outcomes = Counter()
    for outcome, count in end_body.get("outcomes", {}).items():
        before = start_body.get("outcomes", {}).get(outcome, 0)
        if not isinstance(count, int) or not isinstance(before, int) or count < before:
            return ["frozen run service metric outcome counters are invalid"]
        actual_outcomes[outcome] = count - before
    return [] if actual_outcomes == expected_outcomes else [
        "frozen run service metric outcome delta does not match receipt"
    ]


def _snapshot_request_rows(history: Any) -> Optional[List[Mapping[str, Any]]]:
    """Return complete request rows from one untruncated service history."""

    values = history.get("values") if isinstance(history, dict) else None
    if not isinstance(values, list):
        return None
    rows = [value for value in values if _snapshot_request_row_valid(value)]
    if len(rows) != len(values):
        return None
    ids = [row["request_id"] for row in rows]
    return rows if len(ids) == len(set(ids)) else None


def _snapshot_request_row_valid(value: Any) -> bool:
    """Check the identity and terminal outcome fields of one history row."""

    return (
        isinstance(value, dict)
        and isinstance(value.get("request_id"), str)
        and bool(value["request_id"])
        and isinstance(value.get("outcome"), str)
        and bool(value["outcome"])
    )


def _snapshot_shape_errors(snapshot: Mapping[str, Any]) -> List[str]:
    """Require outcomes and bounded request history in one snapshot."""

    body = snapshot.get("body")
    if not isinstance(body, dict):
        return ["service metric snapshot body is invalid"]
    errors = []
    if not _outcome_counters_valid(body.get("outcomes")):
        errors.append("service metric snapshot outcomes are missing")
    history = _request_history(body)
    if not isinstance(history, dict):
        errors.append("service metric snapshot request history is missing")
    elif not _history_wire_valid(history):
        errors.append("service metric snapshot request history is invalid")
    return errors


def _outcome_counters_valid(outcomes: Any) -> bool:
    """Require nonnegative integer counters for every recorded outcome."""

    return isinstance(outcomes, dict) and all(
        isinstance(name, str)
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        for name, count in outcomes.items()
    )


def _request_history(body: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Read the canonical terminal request history from either wire spelling."""

    requests = body.get("requests")
    terminal = body.get("terminal_requests")
    if isinstance(requests, dict) and isinstance(terminal, dict):
        return requests if requests == terminal else None
    if isinstance(requests, dict):
        return requests
    return terminal if isinstance(terminal, dict) else None


def _history_wire_valid(history: Mapping[str, Any]) -> bool:
    """Check retained values and loss counters in one serialized history."""

    values = history.get("values")
    capacity = history.get("capacity")
    dropped = history.get("dropped_count")
    total = history.get("sample_count")
    return (
        isinstance(values, list)
        and _positive_int(capacity)
        and len(values) <= capacity
        and _nonnegative_int(dropped)
        and _nonnegative_int(total)
        and total >= len(values)
        and dropped == total - len(values)
    )


def _positive_int(value: Any) -> bool:
    """Return whether a value is a positive integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    """Return whether a value is a nonnegative integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _snapshot_epoch(snapshot: Mapping[str, Any]) -> Optional[Union[str, int]]:
    """Read a nonempty workload epoch from one metrics snapshot body."""

    body = snapshot.get("body") if isinstance(snapshot, Mapping) else None
    if not isinstance(body, dict):
        return None
    value = body.get("workload_epoch")
    if isinstance(value, int) and not isinstance(value, bool):
        return value if value >= 0 else None
    if isinstance(value, str):
        return value if value else None
    return None


def run_study(manifest: Mapping[str, Any], root: Path, manifest_path: Path) -> Dict[str, Any]:
    """Run configured endpoints and retain missing observations explicitly."""

    _validate_run_phase(manifest, root)
    try:
        receipt_manifest_path = _relative_root_path(root, manifest_path)
    except ValueError as error:
        raise StudyError(f"manifest path is outside study root: {error}") from error
    budgets = manifest["budgets"]
    timeout_s = float(manifest.get("timeout_s", 60))
    deadline = time.monotonic() + float(budgets["wall_time_limit_s"])
    producer_paths = _study_producer_paths(root, manifest["engines"])
    provenance_before = _engine_rows(manifest, root)
    artifacts_before = artifact_provenance(manifest, root)
    with _study_engines(manifest, root, provenance_before, producer_paths) as bound:
        runs = _study_runs(bound, timeout_s, deadline, _load_history_tokenization_producer())
    evaluation = _study_evaluation(manifest, runs, budgets)
    if time.monotonic() >= deadline:
        raise StudyError("study wall-time budget exhausted before final receipt")
    receipt = _study_receipt(
        manifest, root, receipt_manifest_path, provenance_before, artifacts_before,
        runs, evaluation,
    )
    if time.monotonic() >= deadline:
        raise StudyError("study wall-time budget exhausted after final receipt")
    return receipt


@contextlib.contextmanager
def _study_engines(
    manifest: Mapping[str, Any], root: Path, rows: Sequence[Mapping[str, Any]],
    producer_paths: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Any:
    """Bind each engine to its build source and start the servers the harness owns.

    A `spawned_process` engine gets one server for the whole study. The server
    stops, and its private slot directory goes, when the study ends or fails.
    """

    with contextlib.ExitStack() as stack:
        bound = []
        for engine, row in zip(manifest["engines"], rows):
            item = dict(engine)
            build = row.get("provenance", {}).get("build_info", {}) if isinstance(row.get("provenance"), Mapping) else {}
            if isinstance(build.get("source_id"), str):
                item["_build_source_id"] = build["source_id"]
            if producer_paths is None:
                item["_producer_paths"] = _producer_paths(root, engine)
            else:
                item["_producer_paths"] = dict(producer_paths[engine["id"]])
            if engine.get("identity_provider") == "spawned_process":
                server = stack.enter_context(SpawnedServer(item, root))
                item.update({"_spawned": server, "_slot_dir": server.slot_dir})
            bound.append(item)
        yield {**manifest, "engines": bound}


def _study_runs(
    manifest: Mapping[str, Any], timeout_s: float, deadline: float,
    history_tokenization_producer: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run every engine and repetition within one wall-time budget."""

    runs = []
    deadline_ns = int(deadline * 1_000_000_000)
    for repetition in range(manifest["budgets"]["repetitions"]):
        for engine in manifest["engines"]:
            if time.monotonic() >= deadline:
                raise StudyError("study wall-time budget exhausted before all runs completed")
            run = run_engine(
                manifest, engine, repetition, timeout_s, deadline_ns,
                history_tokenization_producer,
            )
            if time.monotonic() >= deadline:
                raise StudyError("study wall-time budget exhausted after run")
            runs.append(run)
    return runs


def _validate_run_phase(manifest: Mapping[str, Any], root: Path) -> None:
    """Reject pending runs and validate frozen references before work starts."""

    if manifest.get("phase") == "pending":
        raise StudyError("pending manifest cannot run before an explicit frozen gate")
    if manifest.get("phase") == "frozen":
        model_sha256 = _model_artifact_sha256(manifest, root)
        errors, _ = _frozen_manifest_reference(manifest, root, model_sha256)
        if model_sha256 is None:
            errors.append("model artifact is unreadable")
        if errors:
            raise StudyError("frozen manifest validation failed: " + "; ".join(errors))


def _model_artifact_sha256(manifest: Mapping[str, Any], root: Path) -> Optional[str]:
    """Hash the declared model artifact, or return None when it is absent."""

    try:
        path = root_path(root, manifest["artifacts"]["model"]["path"])
    except (KeyError, TypeError, ValueError):
        return None
    return cached_file_sha256(path) if path.is_file() else None


def _study_evaluation(
    manifest: Mapping[str, Any], runs: Sequence[Mapping[str, Any]], budgets: Mapping[str, Any]
) -> Dict[str, Any]:
    """Build phase-specific evaluation evidence from retained runs."""

    evaluation = dict(manifest.get("evaluation", {}))
    context = {"runs": runs}
    if manifest.get("phase") == "frozen":
        evaluation["threshold_results"] = evaluate_thresholds(context, manifest)
        evaluation["quality_binding"] = _quality_binding_declaration(manifest)
    if manifest.get("phase") == "calibration":
        records = _all_records(context)
        summary = outcome_summary(records, budgets["minimum_samples_for_quantiles"], budgets["max_history"])
        evaluation["calibration_evidence"] = {
            "status": "observed", "workload_id": manifest.get("workload_id"),
            "summary_sha256": sha256_bytes(canonical_json(summary)),
            "prompt_hashes": [item["sha256"] for item in prompt_provenance(manifest)],
        }
    return evaluation


def _study_receipt(
    manifest: Mapping[str, Any], root: Path, receipt_manifest_path: str,
    provenance_before: Sequence[Mapping[str, Any]], artifacts_before: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]], evaluation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble one study receipt from generated observations."""

    budgets = manifest["budgets"]
    records = _all_records({"runs": runs})
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": _git_source(root),
        "phase": manifest["phase"],
        "workload_id": manifest.get("workload_id"),
        "manifest": {"path": receipt_manifest_path, "canonical_sha256": sha256_bytes(canonical_json(manifest))},
        "prompts": prompt_provenance(manifest),
        "budgets": manifest["budgets"],
        "schedule": manifest["schedule"],
        "artifacts": artifact_provenance(manifest, root, artifacts_before),
        "engines": _engine_rows(manifest, root, provenance_before, runs),
        "runs": runs,
        "summary": outcome_summary(
            records,
            budgets["minimum_samples_for_quantiles"],
            budgets["max_history"],
        ),
        "summary_by_engine_role": summary_by_engine_role(
            {"runs": runs},
            budgets["minimum_samples_for_quantiles"],
            budgets["max_history"],
        ),
        "end_to_end": receipt_end_to_end({"runs": runs}),
        "evaluation": evaluation,
    }


def load_manifest(path: Path) -> Dict[str, Any]:
    """Read one manifest and raise on schema errors."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError(f"cannot read manifest: {error}") from error
    if not isinstance(value, dict):
        raise StudyError("manifest root is not an object")
    errors = validate_manifest(value)
    if errors:
        raise StudyError("manifest validation failed: " + "; ".join(errors))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate, print a plan, or run one branching study."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("calibration", "frozen"))
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--source-manifest", type=Path, help="source input manifest, relative to --root, to bind by content hash")
    parser.add_argument("--source-scope", choices=SOURCE_SCOPES, default="repository", help="check the whole recorded source set, or only the files an archive carries")
    parser.add_argument("--offline", action="store_true", help="check recorded digests without local models or binaries")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.validate_receipt:
        return _validate_receipt_command(
            arguments.validate_receipt, arguments.root, arguments.offline,
            arguments.source_manifest, arguments.source_scope,
        )
    if arguments.manifest is None:
        raise StudyError("--manifest is required unless validating a receipt")
    manifest = load_manifest(arguments.manifest)
    if arguments.mode and arguments.mode != manifest["phase"]:
        raise StudyError("--mode differs from manifest phase")
    return _study_command(arguments, manifest)


def _validate_receipt_command(
    path: Path, root: Path, offline: bool = False, source_manifest: Optional[Path] = None,
    source_scope: str = "repository",
) -> int:
    """Print receipt validation errors and return its process status."""

    errors = validate_receipt(path, root, offline, source_manifest, source_scope)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print(path)
    return 0


def _write_append_only(path: Path, value: Mapping[str, Any]) -> None:
    """Create one bounded receipt and remove it if writing fails."""

    payload = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_RECEIPT_BYTES:
        raise StudyError("branching receipt exceeds the 64 MiB byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise StudyError(f"refusing to replace existing receipt: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            written = stream.write(payload)
            if written != len(payload):
                raise StudyError("branching receipt write was short")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _study_command(arguments: argparse.Namespace, manifest: Mapping[str, Any]) -> int:
    """Print a plan or write one completed study receipt."""

    if arguments.plan or not arguments.run:
        print(json.dumps(build_plan(manifest, arguments.manifest), indent=2))
        return 0
    if arguments.output is None:
        raise StudyError("--run requires --output")
    if arguments.output.exists() or arguments.output.is_symlink():
        raise StudyError("the output already exists")
    receipt = run_study(manifest, arguments.root, arguments.manifest)
    _write_append_only(arguments.output, receipt)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except StudyError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
