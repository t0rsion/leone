#!/usr/bin/env python3
"""Run a reproducible streaming comparison for Leone and llama.cpp servers.

The harness sends the same fixed request schedule to three server variants.
It records HTTP content-event timing. It does not infer token timing from
stream chunks because one chunk can contain several token pieces.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import itertools
import json
import os
import platform
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "leone.concurrent-service-study.v1"
QUANTILE_DEFINITION = (
    "linear interpolation between adjacent sorted observations at h=(n-1)p"
)
CONTENT_EVENT_INTERVAL_DEFINITION = (
    "elapsed monotonic time between read completions containing successive nonempty SSE delta.content events"
)
TOKEN_BOUNDARY_LIMIT = (
    "SSE content events do not identify token boundaries; token ITL is not reported"
)
DEFAULT_QUANTILES = (0.50, 0.90, 0.95, 0.99)
VARIANT_IDS = ("leone_batch1", "leone_batch4", "llama_server")
SAFE_BUILD_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "GGML_CUDA_GRAPH_OPT",
    "GGML_CUDA_FORCE_MMQ",
    "LEONE_CUDA_GRAPH_OPT",
)
CONTROLLED_LAUNCH_FLAGS = {
    "--model", "--host", "--port", "--ctx-size", "--parallel", "--cache-type-k",
    "--cache-type-v", "--log-file", "--bind", "--sessions", "--batch-size", "--kv",
    "--plan", "--backend", "--prefill-chunk", "--context-limit", "-m", "-c", "-np",
    "-ctk", "-ctv", "--no-cache-prompt", "--cache-prompt", "--cache-ram",
    "--cont-batching", "--no-cont-batching", "--flash-attn", "-fa", "--chat-template",
    "--jinja", "--no-jinja", "--reasoning-format", "--receipt-dir", "--signing-key",
}


class HarnessError(RuntimeError):
    """Reports an invalid manifest or unrecoverable study setup."""


class MalformedSseError(ValueError):
    """Reports an SSE line that does not use the supported data format."""


class IncompleteStreamError(ValueError):
    """Reports a stream that ended before its SSE framing completed."""


def monotonic_ns() -> int:
    """Return a steady nanosecond timestamp for interval measurements."""

    return time.monotonic_ns()


def canonical_json(value: Any) -> bytes:
    """Encode JSON with stable ordering for provenance hashes."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


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


def _root_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def quantile(values: Sequence[float], probability: float) -> Optional[float]:
    """Return one empirical quantile, or None for an empty sample."""

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
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def empirical_quantiles(
    values: Sequence[float],
    minimum_samples: int,
    probabilities: Sequence[float] = DEFAULT_QUANTILES,
) -> Dict[str, Any]:
    """Return quantiles and their sample-count qualification."""

    count = len(values)
    result: Dict[str, Any] = {
        "definition": QUANTILE_DEFINITION,
        "sample_count": count,
        "minimum_samples": minimum_samples,
        "qualified": count >= minimum_samples,
        "values": {},
    }
    if count >= minimum_samples:
        result["values"] = {
            f"p{int(probability * 100):02d}": quantile(values, probability)
            for probability in probabilities
        }
    return result


@dataclass(frozen=True)
class SseEvent:
    """One decoded SSE data event and its receive timestamp."""

    data: str
    received_ns: int


class SseEventParser:
    """Parse SSE data fields while retaining event receive timestamps."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._data_lines: List[str] = []
        self.events_seen = 0

    def feed(self, chunk: bytes, received_ns: Optional[int] = None) -> List[SseEvent]:
        """Consume bytes and return all complete events in the chunk."""

        if not isinstance(chunk, (bytes, bytearray)):
            raise TypeError("SSE chunks must be bytes")
        self._buffer.extend(chunk)
        timestamp = received_ns if received_ns is not None else monotonic_ns()
        events: List[SseEvent] = []
        while True:
            separator = self._find_line_separator()
            if separator is None:
                break
            line, consumed = separator
            del self._buffer[:consumed]
            if line == b"":
                event = self._finish_event(timestamp)
                if event is not None:
                    events.append(event)
                continue
            self._parse_line(line, timestamp)
        return events

    def finish(self) -> None:
        """Reject an unterminated SSE event or partial UTF-8 line."""

        if self._buffer or self._data_lines:
            raise IncompleteStreamError("SSE stream ended before a blank event line")

    def _find_line_separator(self) -> Optional[Tuple[bytes, int]]:
        newline = self._buffer.find(b"\n")
        if newline < 0:
            return None
        line = bytes(self._buffer[:newline])
        if line.endswith(b"\r"):
            line = line[:-1]
        return line, newline + 1

    def _parse_line(self, line: bytes, timestamp: int) -> None:
        if line.startswith(b":"):
            return
        if not line.startswith(b"data:"):
            raise MalformedSseError(
                "unsupported SSE field: " + line[:80].decode("utf-8", "replace")
            )
        try:
            text = line[5:].decode("utf-8")
        except UnicodeDecodeError as error:
            raise MalformedSseError("SSE data is not UTF-8") from error
        if text.startswith(" "):
            text = text[1:]
        self._data_lines.append(text)

    def _finish_event(self, timestamp: int) -> Optional[SseEvent]:
        if not self._data_lines:
            return None
        data = "\n".join(self._data_lines)
        self._data_lines = []
        self.events_seen += 1
        return SseEvent(data=data, received_ns=timestamp)


def parse_sse_events(chunks: Iterable[bytes]) -> List[SseEvent]:
    """Parse complete SSE chunks and require complete framing."""

    parser = SseEventParser()
    events: List[SseEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    parser.finish()
    return events


def _json_object(data: str) -> Dict[str, Any]:
    try:
        value = json.loads(data)
    except json.JSONDecodeError as error:
        raise MalformedSseError("SSE data is not JSON") from error
    if not isinstance(value, dict):
        raise MalformedSseError("SSE JSON data must be an object")
    return value


@dataclass
class ParsedStream:
    """Decoded content events and terminal usage from one response."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    content_events: List[Tuple[str, int]] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    done_seen: bool = False
    role_seen: bool = False
    terminal_seen: bool = False
    response_id: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None

    def consume(self, event: SseEvent) -> bool:
        """Consume one event and return True when it contains content."""

        if event.data == "[DONE]":
            self._consume_done()
            return False
        if self.done_seen:
            raise MalformedSseError("SSE data appeared after [DONE]")
        content_count = len(self.content_events)
        payload = _json_object(event.data)
        self.events.append(payload)
        self._consume_metadata(payload)
        choices = payload.get("choices")
        if not isinstance(choices, list):
            raise MalformedSseError("SSE event has no choices list")
        if not choices:
            if payload.get("usage") is None:
                raise MalformedSseError("empty choices without usage")
            return False
        if len(choices) != 1:
            raise MalformedSseError("SSE event has multiple choices")
        self._consume_choice(choices[0], event.received_ns)
        return len(self.content_events) > content_count

    def _consume_done(self) -> None:
        if self.done_seen:
            raise MalformedSseError("duplicate SSE [DONE] event")
        self.done_seen = True

    def _consume_metadata(self, payload: Mapping[str, Any]) -> None:
        response_id = payload.get("id")
        if isinstance(response_id, str):
            self.response_id = response_id
        telemetry = payload.get("leone_telemetry")
        if telemetry is not None:
            if not isinstance(telemetry, dict):
                raise MalformedSseError("leone_telemetry is not an object")
            self.telemetry = telemetry
        usage = payload.get("usage")
        if usage is not None:
            self.usage = _parse_usage(usage)

    def _consume_choice(self, choice: Any, received_ns: int) -> None:
        if not isinstance(choice, dict):
            raise MalformedSseError("SSE choice is not an object")
        if choice.get("finish_reason") is not None:
            self.terminal_seen = True
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return
        role = delta.get("role")
        self.role_seen = self.role_seen or isinstance(role, str)
        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            raise MalformedSseError("SSE delta.content is not a string")
        if content:
            self.content_events.append((content, received_ns))

    def finish(self) -> None:
        """Require a terminal marker and terminal token usage."""

        if not self.done_seen:
            raise IncompleteStreamError("SSE stream has no [DONE] event")
        if not self.terminal_seen:
            raise IncompleteStreamError("SSE stream has no terminal choice")


def _parse_usage(value: Any) -> Dict[str, int]:
    if not isinstance(value, dict):
        raise MalformedSseError("SSE usage is not an object")
    result: Dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise MalformedSseError(f"SSE usage.{key} is not a non-negative integer")
        result[key] = item
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise MalformedSseError("SSE usage.total_tokens does not equal prompt plus completion")
    return result


def _stream_metrics(
    parsed: ParsedStream,
    request_start_ns: int,
    response_header_ns: Optional[int],
    completed_ns: Optional[int],
) -> Dict[str, Any]:
    timestamps_ms = [
        (timestamp - request_start_ns) / 1_000_000
        for _, timestamp in parsed.content_events
    ]
    intervals: List[Optional[float]] = []
    for index, timestamp in enumerate(timestamps_ms):
        if index == 0:
            intervals.append(None)
        else:
            intervals.append(timestamp - timestamps_ms[index - 1])
    first_content = timestamps_ms[0] if timestamps_ms else None
    completion_ms = (
        (completed_ns - request_start_ns) / 1_000_000
        if completed_ns is not None
        else None
    )
    return {
        "time_to_response_headers_ms": (
            (response_header_ns - request_start_ns) / 1_000_000
            if response_header_ns is not None
            else None
        ),
        "first_nonempty_content_ttft_ms": first_content,
        "content_event_count": len(parsed.content_events),
        "content_event_elapsed_ms": timestamps_ms,
        "content_event_intervals_ms": intervals,
        "content_event_intervals_definition": CONTENT_EVENT_INTERVAL_DEFINITION,
        "response_header_timing_definition": "elapsed monotonic time until HTTP response headers are available",
        "token_boundaries_verified": False,
        "token_itl_ms": None,
        "token_timing_limit": TOKEN_BOUNDARY_LIMIT,
        "completion_latency_ms": completion_ms,
    }


@dataclass
class RequestResult:
    """Record one request, including a non-success outcome."""

    request_id: str
    prompt_id: str
    planned_offset_ms: float
    status: str
    http_status: Optional[int] = None
    error: Optional[str] = None
    start_elapsed_ms: Optional[float] = None
    response_body_sha256: Optional[str] = None
    request_body_sha256: Optional[str] = None
    response_id: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    content_sha256: Optional[str] = None
    content_bytes: Optional[int] = None
    telemetry: Optional[Dict[str, Any]] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-compatible request outcome."""

        result: Dict[str, Any] = {
            "request_id": self.request_id,
            "prompt_id": self.prompt_id,
            "planned_offset_ms": self.planned_offset_ms,
            "status": self.status,
            "http_status": self.http_status,
            "error": self.error,
            "start_elapsed_ms": self.start_elapsed_ms,
            "response_body_sha256": self.response_body_sha256,
            "request_body_sha256": self.request_body_sha256,
            "response_id": self.response_id,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "content": {
                "sha256": self.content_sha256,
                "bytes": self.content_bytes,
            },
            "leone_telemetry": self.telemetry,
        }
        result.update(self.metrics)
        return result


def _elapsed_ms(start_ns: int, value_ns: Optional[int]) -> Optional[float]:
    if value_ns is None:
        return None
    return (value_ns - start_ns) / 1_000_000


def _request_body(request: Mapping[str, Any], model_name: str, stream: bool) -> bytes:
    request_id = str(request["id"])
    body = {
        "model": request.get("model", model_name),
        "messages": [{"role": "user", "content": request["prompt"]}],
        "max_tokens": request["max_tokens"],
        "temperature": request.get("temperature", 0.0),
        "seed": request.get("seed", 0),
        "stream": stream,
        "stream_options": {"include_usage": True},
        "user": f"concurrent-service-{request_id}",
        "leone_session": f"concurrent-service-{request_id}",
    }
    return canonical_json(body)


def _connection_parts(base_url: str) -> Tuple[str, int, str]:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise HarnessError(f"only loopback HTTP URLs are supported: {base_url}")
    port = parsed.port or 80
    path = parsed.path.rstrip("/")
    return parsed.hostname, port, path


def _read_error_body(response: http.client.HTTPResponse) -> Tuple[bytes, Optional[str]]:
    try:
        body = response.read(1024 * 1024)
        return body, None
    except (OSError, http.client.HTTPException) as error:
        return b"", str(error)


def set_remaining_timeout(transport: Optional[socket.socket], started_ns: int, timeout_s: float) -> None:
    remaining = timeout_s - (monotonic_ns() - started_ns) / 1_000_000_000
    if remaining <= 0:
        raise TimeoutError("request wall-time limit exceeded")
    if transport is not None:
        transport.settimeout(remaining)


def _read_stream(
    response: http.client.HTTPResponse,
    transport: Optional[socket.socket],
    started_ns: int,
    timeout_s: float,
    parser: SseEventParser,
    parsed: ParsedStream,
    raw_body: bytearray,
    result: RequestResult,
    disconnect_after_content: bool,
    response_header_ns: int,
) -> bool:
    read_method = getattr(response, "read1", response.read)
    while True:
        set_remaining_timeout(transport, started_ns, timeout_s)
        chunk = read_method(4096)
        if not chunk:
            break
        received_ns = monotonic_ns()
        raw_body.extend(chunk)
        for event in parser.feed(chunk, received_ns):
            has_content = parsed.consume(event)
            if disconnect_after_content and has_content:
                result.status = "intentional_disconnect"
                result.error = "client closed after first nonempty content event"
                result.metrics = _stream_metrics(
                    parsed, started_ns, response_header_ns, None
                )
                result.response_body_sha256 = sha256_bytes(bytes(raw_body))
                return True
    return False


def _record_stream_result(
    parsed: ParsedStream,
    raw_body: bytearray,
    result: RequestResult,
    started_ns: int,
    response_header_ns: Optional[int],
    completed_ns: Optional[int],
) -> None:
    if completed_ns is not None:
        if parsed.usage is None:
            result.status = "completed_missing_usage"
            result.error = "terminal SSE event did not include usage"
        else:
            result.status = "completed"
            result.prompt_tokens = parsed.usage["prompt_tokens"]
            result.completion_tokens = parsed.usage["completion_tokens"]
            result.total_tokens = parsed.usage["total_tokens"]
        text = "".join(content for content, _ in parsed.content_events)
        encoded = text.encode("utf-8")
        result.content_sha256 = sha256_bytes(encoded)
        result.content_bytes = len(encoded)
        result.response_id = parsed.response_id
        result.response_body_sha256 = sha256_bytes(bytes(raw_body))
        result.metrics = _stream_metrics(
            parsed, started_ns, response_header_ns, completed_ns
        )
    elif parsed.content_events and result.content_sha256 is None:
        partial_text = "".join(content for content, _ in parsed.content_events).encode("utf-8")
        result.content_sha256 = sha256_bytes(partial_text)
        result.content_bytes = len(partial_text)
    result.telemetry = parsed.telemetry
    if completed_ns is None:
        result.response_body_sha256 = sha256_bytes(bytes(raw_body)) if raw_body else None
        result.metrics = _stream_metrics(parsed, started_ns, response_header_ns, None)


def perform_stream_request(
    base_url: str,
    request: Mapping[str, Any],
    model_name: str,
    timeout_s: float,
    planned_offset_ms: float,
    request_id: str,
    prompt_id: str,
    disconnect_after_content: bool = False,
) -> RequestResult:
    """Send one streaming request and retain timing only from observed bytes."""

    host, port, base_path = _connection_parts(base_url)
    body = _request_body(request, model_name, stream=True)
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Content-Length": str(len(body)),
        "Connection": "close",
    }
    started_ns = monotonic_ns()
    result = RequestResult(
        request_id=request_id,
        prompt_id=prompt_id,
        planned_offset_ms=planned_offset_ms,
        status="connection_error",
        start_elapsed_ms=None,
        request_body_sha256=sha256_bytes(body),
    )
    connection: Optional[http.client.HTTPConnection] = None
    raw_body = bytearray()
    parsed = ParsedStream()
    parser = SseEventParser()
    response_header_ns: Optional[int] = None
    completed_ns: Optional[int] = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        connection.request("POST", base_path + "/v1/chat/completions", body, headers)
        transport = connection.sock
        set_remaining_timeout(transport, started_ns, timeout_s)
        response = connection.getresponse()
        response_header_ns = monotonic_ns()
        result.http_status = response.status
        if response.status != 200:
            error_body, error = _read_error_body(response)
            raw_body.extend(error_body)
            result.status = "http_error"
            result.error = error or error_body[:512].decode("utf-8", "replace")
            result.response_body_sha256 = sha256_bytes(bytes(raw_body))
            result.metrics = _stream_metrics(parsed, started_ns, response_header_ns, None)
            return result
        if _read_stream(
            response,
            transport,
            started_ns,
            timeout_s,
            parser,
            parsed,
            raw_body,
            result,
            disconnect_after_content,
            response_header_ns,
        ):
            return result
        parser.finish()
        parsed.finish()
        completed_ns = monotonic_ns()
        _record_stream_result(
            parsed, raw_body, result, started_ns, response_header_ns, completed_ns
        )
        return result
    except socket.timeout as error:
        result.status = "timeout"
        result.error = str(error) or "socket timeout"
    except IncompleteStreamError as error:
        result.status = "incomplete_stream"
        result.error = str(error)
    except MalformedSseError as error:
        result.status = "malformed_sse"
        result.error = str(error)
    except (OSError, http.client.HTTPException, ValueError) as error:
        result.status = "connection_error"
        result.error = str(error)
    finally:
        if connection is not None:
            connection.close()
    _record_stream_result(parsed, raw_body, result, started_ns, response_header_ns, None)
    return result


def _run_at_offset(
    base_url: str,
    request: Mapping[str, Any],
    model_name: str,
    timeout_s: float,
    epoch_ns: int,
    request_id: str,
    prompt_id: str,
    planned_offset_ms: float,
) -> RequestResult:
    deadline_ns = epoch_ns + int(planned_offset_ms * 1_000_000)
    while True:
        remaining_ns = deadline_ns - monotonic_ns()
        if remaining_ns <= 0:
            break
        time.sleep(min(remaining_ns / 1_000_000_000, 0.010))
    actual_start_ns = monotonic_ns()
    result = perform_stream_request(
        base_url,
        request,
        model_name,
        timeout_s,
        planned_offset_ms,
        request_id,
        prompt_id,
    )
    result.start_elapsed_ms = _elapsed_ms(epoch_ns, actual_start_ns)
    return result


def run_offered_load(
    base_url: str,
    requests: Sequence[Mapping[str, Any]],
    offsets_ms: Sequence[float],
    model_name: str,
    timeout_s: float,
    minimum_samples: int = 1,
) -> Tuple[List[RequestResult], Dict[str, Any]]:
    """Run one fixed staggered schedule and retain all outcomes."""

    if len(requests) != len(offsets_ms):
        raise HarnessError("request schedule and offset counts differ")
    epoch_ns = monotonic_ns()
    results: List[RequestResult] = []
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(
                _run_at_offset,
                base_url,
                request,
                model_name,
                timeout_s,
                epoch_ns,
                str(request["id"]),
                str(request["prompt_id"]),
                float(offset_ms),
            )
            for request, offset_ms in zip(requests, offsets_ms)
        ]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:  # keep one client failure in the receipt
                request = requests[len(results)]
                results.append(
                    RequestResult(
                        request_id=str(request["id"]),
                        prompt_id=str(request["prompt_id"]),
                        planned_offset_ms=float(offsets_ms[len(results)]),
                        status="client_error",
                        error=f"{type(error).__name__}: {error}",
                    )
                )
    results.sort(key=lambda result: result.request_id)
    summary = summarize_requests(results, minimum_samples=minimum_samples)
    wall_ms = (monotonic_ns() - epoch_ns) / 1_000_000
    token_count = sum(result.completion_tokens or 0 for result in results)
    summary["offered_load_wall_ms"] = wall_ms
    summary["aggregate_completion_tok_s"] = (
        token_count / (wall_ms / 1000.0) if wall_ms > 0 and token_count else None
    )
    return results, summary


def summarize_requests(
    results: Sequence[RequestResult], minimum_samples: int
) -> Dict[str, Any]:
    """Summarize outcomes and qualified request-level metrics."""

    successful = [result for result in results if result.status == "completed"]
    ttft = _successful_metric_values(successful, "first_nonempty_content_ttft_ms")
    completion_latency = _successful_metric_values(successful, "completion_latency_ms")
    completion_rates = _completion_rates(successful)
    interval_values = _interval_values(successful)
    return {
        "request_count": len(results),
        "completed_count": len(successful),
        "loss_count": len(results) - len(successful),
        "outcomes": dict(Counter(result.status for result in results)),
        "completion_tokens_observed": sum(
            result.completion_tokens or 0 for result in successful
        ),
        "prompt_tokens_observed": sum(
            result.prompt_tokens or 0 for result in successful
        ),
        "ttft_ms": empirical_quantiles(ttft, minimum_samples),
        "completion_latency_ms": empirical_quantiles(
            completion_latency, minimum_samples
        ),
        "completion_tok_s": empirical_quantiles(completion_rates, minimum_samples),
        "zero_content_event_intervals": sum(value == 0 for value in interval_values),
        "content_event_interval_ms": empirical_quantiles(
            interval_values, minimum_samples
        ),
        "token_itl_ms": None,
        "token_boundaries_verified": False,
    }


def _successful_metric_values(
    successful: Sequence[RequestResult], key: str
) -> List[float]:
    return [
        float(result.metrics[key])
        for result in successful
        if result.metrics.get(key) is not None
    ]


def _completion_rates(successful: Sequence[RequestResult]) -> List[float]:
    return [
        result.completion_tokens / (result.metrics["completion_latency_ms"] / 1000.0)
        for result in successful
        if result.completion_tokens is not None
        and result.metrics.get("completion_latency_ms")
        and result.metrics["completion_latency_ms"] > 0
    ]


def _interval_values(successful: Sequence[RequestResult]) -> List[float]:
    return [
        interval
        for result in successful
        for interval in result.metrics.get("content_event_intervals_ms", [])
        if interval is not None
    ]


def _request_path(base_url: str, endpoint: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    base_path = parsed.path.rstrip("/")
    return base_path + "/" + endpoint.lstrip("/")


def get_json(base_url: str, endpoint: str, timeout_s: float) -> Dict[str, Any]:
    """Fetch a JSON endpoint and reject non-object responses."""

    host, port, _ = _connection_parts(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        connection.request("GET", _request_path(base_url, endpoint))
        response = connection.getresponse()
        body = response.read(1024 * 1024)
        if response.status < 200 or response.status >= 300:
            raise HarnessError(f"GET {endpoint} returned HTTP {response.status}")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise HarnessError(f"GET {endpoint} did not return an object")
        return value
    finally:
        connection.close()


def wait_ready(base_url: str, timeout_s: float, startup_timeout_s: float) -> Dict[str, Any]:
    """Wait for a server health endpoint and return its response."""

    deadline = time.monotonic() + startup_timeout_s
    last_error = "server did not answer health checks"
    while time.monotonic() < deadline:
        try:
            health = get_json(base_url, "/health", timeout_s)
            return health
        except (OSError, http.client.HTTPException, HarnessError, ValueError) as error:
            last_error = str(error)
            time.sleep(0.100)
    raise HarnessError(last_error)


def _require_port_free(base_url: str) -> None:
    """Reject a study port already owned by another process."""

    host, port, _ = _connection_parts(base_url)
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.1)
        if probe.connect_ex((host, port)) == 0:
            raise HarnessError(f"study port is already in use: {host}:{port}")
    finally:
        probe.close()


def resident_progress_trace(trace: Any) -> Dict[str, Any]:
    """Validate the trace contract for resident decode during prefill."""

    result = {
        "schema_version": None,
        "event_count": 0,
        "prefill_events": 0,
        "resident_progress_events": 0,
        "resident_progress_proven": False,
        "error": None,
    }
    if not isinstance(trace, dict):
        result["error"] = "trace is not an object"
        return result
    result["schema_version"] = trace.get("schema_version")
    if result["schema_version"] != "leone.service-trace.v1":
        result["error"] = "trace schema_version is not leone.service-trace.v1"
        return result
    events = trace.get("events")
    if not isinstance(events, list):
        result["error"] = "trace.events is not a list"
        return result
    result["event_count"] = len(events)
    prefill_events, progress_events, proven, error = _scan_trace_events(events)
    result["prefill_events"] = prefill_events
    result["resident_progress_events"] = progress_events
    result["resident_progress_proven"] = proven
    if error:
        result["error"] = error
    elif not proven:
        result["error"] = (
            "trace lacks resident decode between two chunks of another request"
        )
    return result


@dataclass
class _TraceState:
    prefill: int = 0
    progress: int = 0
    proven: bool = False
    previous_chunks: set[str] = field(default_factory=set)
    interleaved: set[str] = field(default_factory=set)


def _scan_trace_events(
    events: Sequence[Any],
) -> Tuple[int, int, bool, Optional[str]]:
    state = _TraceState()
    for event in events:
        error = _scan_trace_event(event, state)
        if error:
            return state.prefill, state.progress, state.proven, error
    return state.prefill, state.progress, state.proven, None


def _scan_trace_event(
    event: Any,
    state: _TraceState,
) -> Optional[str]:
    if not isinstance(event, dict):
        return "trace contains a non-object event"
    request_id = event.get("request_id")
    if not isinstance(request_id, (str, int)):
        return "trace event lacks a request identity"
    request_id = str(request_id)
    if event.get("kind") == "prefill_chunk":
        state.prefill += 1
        if event.get("processed_tokens", 0) > 0:
            if request_id in state.interleaved:
                state.proven = True
            if not event.get("ready", False):
                state.previous_chunks.add(request_id)
    elif event.get("kind") == "resident_decode_progress":
        _scan_progress_event(event, request_id, state)
    return None


def _scan_progress_event(
    event: Mapping[str, Any],
    request_id: str,
    state: _TraceState,
) -> None:
    state.progress += 1
    during = event.get("during_prefill_request_id")
    if not isinstance(during, (str, int)):
        return
    during = str(during)
    if during != request_id and during in state.previous_chunks and event.get("emitted_tokens", 0) > 0:
        state.interleaved.add(during)


def run_disconnect_probe(
    base_url: str,
    request: Mapping[str, Any],
    model_name: str,
    timeout_s: float,
    recovery_wait_s: float,
) -> Dict[str, Any]:
    """Close one stream after content, then record an independent recovery call."""

    disconnect = perform_stream_request(
        base_url,
        dict(request, id="disconnect-probe"),
        model_name,
        timeout_s,
        planned_offset_ms=0.0,
        request_id="disconnect-probe",
        prompt_id=str(request["prompt_id"]),
        disconnect_after_content=True,
    )
    time.sleep(recovery_wait_s)
    recovery = perform_stream_request(
        base_url,
        dict(request, id="disconnect-recovery"),
        model_name,
        timeout_s,
        planned_offset_ms=0.0,
        request_id="disconnect-recovery",
        prompt_id=str(request["prompt_id"]),
    )
    return {
        "recovery_session": "fresh session distinct from the disconnected session",
        "disconnect": disconnect.to_dict(),
        "recovery": recovery.to_dict(),
        "disconnect_succeeded": disconnect.status == "intentional_disconnect",
        "recovery_succeeded": recovery.status == "completed",
        "passed": disconnect.status == "intentional_disconnect" and recovery.status == "completed",
    }


def _run_command(
    argv: Sequence[str], timeout_s: float = 10.0
) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return 127, "", str(error)
    return completed.returncode, completed.stdout, completed.stderr


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _git_provenance(root: Path) -> Dict[str, Any]:
    commit_code, commit_out, commit_err = _run_command(
        ["git", "-C", str(root), "rev-parse", "HEAD"], timeout_s=5
    )
    commit = commit_out.strip() if commit_code == 0 else None
    status_code, status_out, status_err = _run_command(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], timeout_s=5
    )
    return {
        "commit": commit,
        "commit_verified": bool(commit and re.fullmatch(r"[0-9a-f]{40}", commit)),
        "tracked_tree_clean": status_code == 0 and not status_out.strip(),
        "status_error": status_err.strip() or None,
        "rev_parse_error": commit_err.strip() or None,
    }


def _build_info(binary: Path) -> Dict[str, Any]:
    attempts = [[str(binary), "--build-info"], [str(binary), "--build-info", "--json"]]
    code, stdout, stderr = 127, "", ""
    used_argv = attempts[-1]
    for attempt in attempts:
        code, stdout, stderr = _run_command(attempt, timeout_s=10)
        used_argv = attempt
        if code == 0:
            break
    result: Dict[str, Any] = {
        "argv": used_argv,
        "returncode": code,
        "available": False,
        "error": stderr.strip() or None,
    }
    if code == 0:
        try:
            value = json.loads(stdout)
        except json.JSONDecodeError as error:
            result["error"] = f"build-info output is not JSON: {error}"
        else:
            if isinstance(value, dict):
                result["available"] = True
                result["value"] = value
            else:
                result["error"] = "build-info output is not an object"
    return result


def _ldd_paths(binary: Path) -> Tuple[List[str], Optional[str]]:
    code, stdout, stderr = _run_command(["ldd", str(binary)], timeout_s=10)
    if code != 0:
        return [], stderr.strip() or f"ldd returned {code}"
    paths: List[str] = []
    for line in stdout.splitlines():
        match = re.search(r"=>\s+(/[^ ]+)", line)
        if match:
            paths.append(match.group(1))
            continue
        direct = re.match(r"\s*(/[^ ]+)\s+\(0x", line)
        if direct:
            paths.append(direct.group(1))
    return sorted(set(paths)), None


def binary_provenance(root: Path, binary: Path) -> Dict[str, Any]:
    """Hash an executable and every resolved shared object reported by ldd."""

    if not binary.exists():
        raise HarnessError(f"binary does not exist: {binary}")
    libraries, error = _ldd_paths(binary)
    files = [binary] + [Path(path) for path in libraries]
    hashes = []
    hash_errors = []
    for path in files:
        try:
            hashes.append(
                {
                    "path": _relative_path(root, path),
                    "sha256": sha256_file(path),
                    "kind": "executable" if path == binary else "shared_library",
                }
            )
        except OSError as file_error:
            hash_errors.append({"path": str(path), "error": str(file_error)})
    return {
        "path": _relative_path(root, binary),
        "binary_hashes": hashes,
        "ldd_error": error,
        "hash_errors": hash_errors,
        "build_info": _build_info(binary),
    }


def llama_source_provenance(root: Path) -> Dict[str, Any]:
    """Verify the llama.cpp checkout against external/PINNED."""

    checkout = root / "external" / "llama.cpp"
    pinned_path = root / "external" / "PINNED"
    code, stdout, stderr = _run_command(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], timeout_s=5
    )
    commit = stdout.strip() if code == 0 else None
    pinned = pinned_path.read_text(encoding="utf-8").strip() if pinned_path.exists() else None
    status_code, status_out, status_err = _run_command(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
        timeout_s=5,
    )
    return {
        "checkout": _relative_path(root, checkout),
        "pinned_file": _relative_path(root, pinned_path),
        "commit": commit,
        "pinned_commit": pinned,
        "commit_matches_pinned": bool(commit and pinned and commit == pinned),
        "tracked_tree_clean": status_code == 0 and not status_out.strip(),
        "error": stderr.strip() or status_err.strip() or None,
    }


def nvidia_metadata() -> Dict[str, Any]:
    """Capture hardware, driver, clock, and power data without inventing values."""

    executable = shutil.which("nvidia-smi")
    query = (
        "name,driver_version,pstate,clocks.sm,clocks.mem,power.draw,power.limit,"
        "temperature.gpu,utilization.gpu"
    )
    result: Dict[str, Any] = {
        "available": False,
        "executable": executable,
        "query": query,
        "sample_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if executable is None:
        result["error"] = "nvidia-smi is not in PATH"
        return result
    code, stdout, stderr = _run_command(
        [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        timeout_s=10,
    )
    result.update(
        {
            "available": code == 0,
            "returncode": code,
            "stdout": stdout.strip() or None,
            "stderr": stderr.strip() or None,
        }
    )
    return result


def system_metadata() -> Dict[str, Any]:
    """Capture host and process metadata needed to reproduce the run."""

    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python": sys.version,
        "cpu_affinity": affinity,
        "pid": os.getpid(),
        "environment": {
            key: os.environ[key] for key in SAFE_BUILD_ENV_KEYS if key in os.environ
        },
    }


def _format_arg(argument: str, values: Mapping[str, str]) -> str:
    try:
        return argument.format_map(values)
    except KeyError as error:
        raise HarnessError(f"unknown launch argument placeholder: {error}") from error


def _common_values(
    root: Path,
    manifest: Mapping[str, Any],
    model: Path,
    plan: Optional[Path],
    host: str,
    port: int,
    batch_size: int,
    log_path: Path,
    receipt_dir: Path,
    signing_key: Path,
) -> Dict[str, str]:
    workload = manifest["workload"]
    server = manifest["server"]
    return {
        "root": str(root),
        "model": str(model),
        "plan": str(plan) if plan else "",
        "host": host,
        "port": str(port),
        "bind": f"{host}:{port}",
        "batch_size": str(batch_size),
        "sessions": str(server["sessions"]),
        "ctx_size": str(server["context_tokens"]),
        "kv": str(server["kv"]),
        "max_output_tokens": str(workload["max_output_tokens"]),
        "log": str(log_path),
        "receipt_dir": str(receipt_dir),
        "signing_key": str(signing_key),
    }


def build_launch_argv(
    root: Path,
    manifest: Mapping[str, Any],
    variant: Mapping[str, Any],
    model: Path,
    plan: Optional[Path],
    host: str,
    port: int,
    log_path: Path,
    receipt_dir: Path,
    signing_key: Path,
    leone_binary: Path,
    llama_binary: Path,
    cpu_mask: Optional[str],
) -> List[str]:
    """Build one explicit server command from manifest values."""

    kind = variant["kind"]
    batch_size = int(variant.get("batch_size", manifest["server"]["sessions"]))
    values = _common_values(
        root,
        manifest,
        model,
        plan,
        host,
        port,
        batch_size,
        log_path,
        receipt_dir,
        signing_key,
    )
    if kind == "leone":
        argv = _leone_launch_argv(
            manifest, model, plan, host, port, batch_size, receipt_dir, signing_key, leone_binary
        )
    elif kind == "llama_server":
        argv = _llama_launch_argv(
            manifest, model, host, port, log_path, llama_binary, values
        )
    else:
        raise HarnessError(f"unsupported variant kind: {kind}")
    extra = variant.get("extra_launch_args", [])
    _validate_extra_launch_args(extra)
    argv.extend(_format_arg(item, values) for item in extra)
    if cpu_mask:
        argv = ["taskset", "-c", cpu_mask] + argv
    return argv


def _leone_launch_argv(
    manifest: Mapping[str, Any],
    model: Path,
    plan: Optional[Path],
    host: str,
    port: int,
    batch_size: int,
    receipt_dir: Path,
    signing_key: Path,
    binary: Path,
) -> List[str]:
    if plan is None:
        raise HarnessError("Leone requires an execution plan")
    server = manifest["server"]
    return [
        str(binary), "serve", "-m", str(model), "--plan", str(plan), "--backend", "cuda",
        "--bind", f"{host}:{port}", "--sessions", str(server["sessions"]),
        "--batch-size", str(batch_size), "--context-limit", str(server["context_tokens"]),
        "--prefill-chunk", str(server["prefill_chunk_tokens"]), "--kv", str(server["kv"]),
        "--receipt-dir", str(receipt_dir), "--signing-key", str(signing_key),
    ]


def _llama_launch_argv(
    manifest: Mapping[str, Any],
    model: Path,
    host: str,
    port: int,
    log_path: Path,
    binary: Path,
    values: Mapping[str, str],
) -> List[str]:
    server = manifest["server"]
    argv = [str(binary), "--model", str(model), "--host", host, "--port", str(port)]
    argv.extend(_format_arg(str(item), values) for item in server["llama_launch_args"])
    argv.extend([
        "--ctx-size", str(server.get("context_tokens_total", server["context_tokens"])),
        "--parallel", str(server["sessions"]), "--cache-type-k", str(server["kv"]),
        "--cache-type-v", str(server["kv"]), "--log-file", str(log_path),
    ])
    return argv


def _validate_extra_launch_args(extra: Any) -> None:
    if not isinstance(extra, list) or not all(isinstance(item, str) for item in extra):
        raise HarnessError("variant extra_launch_args must be a string list")
    if any(item.split("=", 1)[0] in CONTROLLED_LAUNCH_FLAGS for item in extra):
        raise HarnessError("variant extra_launch_args cannot override controlled flags")


class ManagedServer:
    """Own one server process and its temporary launch artifacts."""

    def __init__(
        self,
        argv: Sequence[str],
        base_url: str,
        log_path: Path,
        cwd: Path,
        startup_timeout_s: float,
        request_timeout_s: float,
    ) -> None:
        self.argv = list(argv)
        self.base_url = base_url
        self.log_path = log_path
        self.cwd = cwd
        self.startup_timeout_s = startup_timeout_s
        self.request_timeout_s = request_timeout_s
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.health: Optional[Dict[str, Any]] = None

    def start(self) -> Dict[str, Any]:
        """Start the process and wait for GET /health."""

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        _require_port_free(self.base_url)
        log_stream = self.log_path.open("wb")
        try:
            self.process = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
        except OSError:
            log_stream.close()
            raise
        finally:
            log_stream.close()
        try:
            self.health = wait_ready(
                self.base_url,
                self.request_timeout_s,
                self.startup_timeout_s,
            )
        except Exception:
            self.stop()
            raise
        return self.health

    def stop(self) -> Dict[str, Any]:
        """Stop the process and return its exit status."""

        if self.process is None:
            return {"returncode": None, "signal": None}
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return {
            "returncode": process.returncode,
            "signal": -process.returncode if process.returncode and process.returncode < 0 else None,
        }


def _balanced_orders(variant_ids: Sequence[str], repetitions: int) -> List[List[str]]:
    permutations = list(itertools.permutations(variant_ids))
    return [list(permutations[index % len(permutations)]) for index in range(repetitions)]


def _position_counts(orders: Sequence[Sequence[str]], variant_ids: Sequence[str]) -> Dict[str, List[int]]:
    result = {variant: [0] * len(variant_ids) for variant in variant_ids}
    for order in orders:
        for position, variant in enumerate(order):
            result[variant][position] += 1
    return result


def _validate_manifest(manifest: Mapping[str, Any], allow_pending_freeze: bool) -> None:
    phase, budgets = _manifest_header(manifest, allow_pending_freeze)
    workload, server, variants = _manifest_sections(manifest)
    requests, prompts = _workload_shape(workload)
    prompt_ids = {str(item.get("id")) for item in prompts if isinstance(item, dict)}
    _validate_requests(requests, prompt_ids, workload)
    _validate_prompt_texts(prompts)
    _validate_workload_budget(workload, budgets, requests, phase, server)
    _validate_server(server)
    _validate_template(manifest)
    _validate_launch_args(server, variants)
    _validate_trace(manifest)


def _manifest_header(
    manifest: Mapping[str, Any], allow_pending_freeze: bool
) -> Tuple[str, Dict[str, Any]]:
    if manifest.get("schema_version") != 1:
        raise HarnessError("concurrent-service manifest schema_version must be 1")
    phase = manifest.get("phase")
    if phase not in ("exploratory", "frozen"):
        raise HarnessError("manifest phase must be exploratory or frozen")
    if phase == "frozen" and manifest.get("freeze_status") != "frozen" and not allow_pending_freeze:
        raise HarnessError(
            "frozen manifest is pending exploratory calibration; set freeze_status=frozen after calibration"
        )
    budgets = manifest.get("budgets")
    if not isinstance(budgets, dict) or not budgets.get("declared_before_tuning"):
        raise HarnessError("manifest budgets must be preregistered")
    return phase, budgets


def _manifest_sections(
    manifest: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Mapping[str, Any]]]:
    workload = manifest.get("workload")
    server = manifest.get("server")
    variants = manifest.get("variants")
    if not isinstance(workload, dict) or not isinstance(server, dict):
        raise HarnessError("manifest requires workload and server objects")
    if not isinstance(variants, list):
        raise HarnessError("manifest variants must be a list")
    actual_ids = tuple(str(item.get("id")) for item in variants if isinstance(item, dict))
    if actual_ids != VARIANT_IDS:
        raise HarnessError(f"manifest variants must be ordered as {VARIANT_IDS}")
    return workload, server, variants


def _workload_shape(
    workload: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    requests = workload.get("requests")
    offsets = workload.get("stagger_ms")
    prompts = workload.get("prompts")
    if not isinstance(requests, list) or not requests:
        raise HarnessError("workload.requests must be non-empty")
    if not isinstance(offsets, list) or len(offsets) != len(requests):
        raise HarnessError("workload.stagger_ms must match request count")
    _validate_offsets(offsets)
    if not isinstance(prompts, list) or not prompts:
        raise HarnessError("workload.prompts must be non-empty")
    return requests, prompts


def _validate_offsets(offsets: Sequence[Any]) -> None:
    if any(not isinstance(item, (int, float)) or item < 0 for item in offsets):
        raise HarnessError("workload.stagger_ms must contain non-negative numbers")
    if list(offsets) != sorted(offsets):
        raise HarnessError("workload.stagger_ms must be non-decreasing")


def _validate_request(
    request: Mapping[str, Any],
    request_ids: set[str],
    prompt_ids: set[str],
    workload: Mapping[str, Any],
) -> None:
    request_id = str(request.get("id"))
    if request_id in request_ids:
        raise HarnessError(f"duplicate request id: {request_id}")
    request_ids.add(request_id)
    if request.get("prompt_id") not in prompt_ids:
        raise HarnessError(f"unknown prompt id for request {request_id}")
    if not isinstance(request.get("max_tokens"), int) or request["max_tokens"] <= 0:
        raise HarnessError(f"request {request_id} has invalid max_tokens")
    if request["max_tokens"] > int(workload.get("max_output_tokens", 0)):
        raise HarnessError(f"request {request_id} exceeds workload max_output_tokens")
    if "prompt" in request:
        raise HarnessError("requests refer to full prompts by prompt_id")


def _validate_requests(
    requests: Sequence[Any], prompt_ids: set[str], workload: Mapping[str, Any]
) -> None:
    request_ids = set()
    for request in requests:
        if not isinstance(request, dict):
            raise HarnessError("each workload request must be an object")
        _validate_request(request, request_ids, prompt_ids, workload)


def _validate_prompt_texts(prompts: Sequence[Any]) -> None:
    prompt_texts = [item.get("text") for item in prompts if isinstance(item, dict)]
    if any(not isinstance(text, str) or not text for text in prompt_texts):
        raise HarnessError("each prompt must have non-empty text")


def _validate_workload_budget(
    workload: Mapping[str, Any],
    budgets: Mapping[str, Any],
    requests: Sequence[Any],
    phase: str,
    server: Mapping[str, Any],
) -> None:
    repetitions = workload.get("repetitions")
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise HarnessError("workload.repetitions must be positive")
    if repetitions > int(budgets.get("max_repetitions", 0)):
        raise HarnessError("workload.repetitions exceeds preregistered budget")
    if len(requests) > int(budgets.get("max_requests_per_repetition", 0)):
        raise HarnessError("request count exceeds preregistered budget")
    if int(workload.get("max_output_tokens", 0)) > int(budgets.get("max_output_tokens", 0)):
        raise HarnessError("max output exceeds preregistered budget")
    minimum_samples = int(budgets.get("minimum_samples_for_quantiles", 0))
    if minimum_samples <= 0:
        raise HarnessError("minimum_samples_for_quantiles must be positive")
    if phase == "frozen" and repetitions < minimum_samples:
        raise HarnessError("frozen workload has too few repetitions for quantiles")
    if int(server.get("sessions", 0)) < len(requests):
        raise HarnessError("server sessions must admit the whole schedule")


def _validate_server(server: Mapping[str, Any]) -> None:
    if not isinstance(server.get("prefill_chunk_tokens"), int) or server["prefill_chunk_tokens"] <= 0:
        raise HarnessError("server.prefill_chunk_tokens must be positive")
    if server.get("kv") not in ("f16", "q8", "q8_0", "bf16"):
        raise HarnessError("server.kv must name a supported KV type")
    if not isinstance(server.get("llama_launch_args"), list) or not all(
        isinstance(item, str) for item in server["llama_launch_args"]
    ):
        raise HarnessError("server.llama_launch_args must be a string list")
    if server.get("context_tokens_total") != int(server["context_tokens"]) * int(
        server["sessions"]
    ):
        raise HarnessError("llama context_tokens_total must equal sessions times per-request context")


def _validate_template(manifest: Mapping[str, Any]) -> None:
    template = manifest.get("template")
    if not isinstance(template, dict) or not template.get("assert_prompt_token_counts_match"):
        raise HarnessError("template must require matching prompt token counts")


def _validate_launch_args(
    server: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> None:
    required_template_flags = (
        "--no-jinja", "--chat-template", "chatml", "--reasoning-format", "none"
    )
    launch_args = tuple(server["llama_launch_args"])
    if not all(flag in launch_args for flag in required_template_flags):
        raise HarnessError("llama launch args must pin the chatml template and reasoning format")
    for variant in variants:
        if variant["kind"] == "llama_server" and "batch_size" in variant:
            raise HarnessError("llama_server uses server.sessions slots; batch_size is unused")
    required_runtime_flags = {
        "--cache-ram": "0", "--flash-attn": "on", "--chat-template": "chatml",
        "--reasoning-format": "none",
    }
    for flag, value in required_runtime_flags.items():
        if launch_args.count(flag) != 1 or launch_args[launch_args.index(flag) + 1:launch_args.index(flag) + 2] != (value,):
            raise HarnessError(f"llama launch args must pin {flag} {value}")
    if not {"--no-cache-prompt", "--cont-batching", "--no-jinja"}.issubset(launch_args):
        raise HarnessError("llama launch args must disable prompt caching and enable continuous batching")


def _validate_trace(manifest: Mapping[str, Any]) -> None:
    trace = manifest.get("trace", {})
    if not isinstance(trace, dict):
        raise HarnessError("manifest trace must be an object")
    if trace.get("schema_version") != "leone.service-trace.v1":
        raise HarnessError("manifest trace schema_version must be leone.service-trace.v1")


def _load_manifest(path: Path, allow_pending_freeze: bool) -> Tuple[Dict[str, Any], Dict[str, str]]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise HarnessError("manifest root must be an object")
    _validate_manifest(manifest, allow_pending_freeze)
    digests = {
        "file_sha256": sha256_bytes(raw),
        "canonical_sha256": sha256_bytes(canonical_json(manifest)),
    }
    return manifest, digests


def _expand_requests(manifest: Mapping[str, Any]) -> List[Dict[str, Any]]:
    prompts = {
        str(prompt["id"]): str(prompt["text"])
        for prompt in manifest["workload"]["prompts"]
    }
    expanded = []
    for request in manifest["workload"]["requests"]:
        item = dict(request)
        item["prompt"] = prompts[str(request["prompt_id"])]
        item["prompt_sha256"] = sha256_bytes(item["prompt"].encode("utf-8"))
        expanded.append(item)
    return expanded


def _quality_provenance(root: Path, path: Path, model_sha: str) -> Dict[str, Any]:
    if not path.exists():
        raise HarnessError(f"quality receipt does not exist: {path}")
    value = _read_quality_receipt(path)
    subject = value.get("subject", {}) if isinstance(value, dict) else {}
    subject_sha = _quality_subject_sha(value, subject)
    comparison_schema = value.get("schema_version") == "leone.quality-comparison.v1"
    executions = value.get("executions", {})
    if comparison_schema and (
        not isinstance(executions, dict)
        or not {"leone_q4", "llama_q4"}.issubset(executions)
    ):
        raise HarnessError("quality comparison lacks Leone and llama.cpp quality rows")
    comparison_engines = sorted(executions) if comparison_schema and isinstance(executions, dict) else []
    if subject_sha != model_sha:
        raise HarnessError("quality receipt subject does not match the served model")
    return _quality_result(
        value,
        subject,
        path,
        root,
        subject_sha,
        comparison_schema,
        comparison_engines,
        executions,
    )


def _quality_result(
    value: Mapping[str, Any],
    subject: Any,
    path: Path,
    root: Path,
    subject_sha: Optional[str],
    comparison_schema: bool,
    comparison_engines: Sequence[str],
    executions: Mapping[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": _relative_path(root, path),
        "sha256": sha256_file(path),
        "receipt_id": value.get("receipt_id"),
        "subject_model_sha256": subject_sha,
        "subject_engine": subject.get("engine") if isinstance(subject, dict) else None,
        "comparison_schema": comparison_schema,
        "comparison_engines": comparison_engines if comparison_schema else [],
        "corpus": value.get("corpus"),
        "oracle": value.get("oracle"),
        "metrics": value.get("metrics"),
        "sample_count": value.get("sample_count"),
    }
    if comparison_schema:
        result.update(
            corpus=value["corpus"], oracle=value["model_family"]["oracle"],
            metrics={name: row["quality"] for name, row in executions.items() if "quality" in row},
            sample_count=value["input"]["scored_positions"],
            input=value["input"], build_info=value["build_info"], executable=value["executable"],
        )
    return result


def _read_quality_receipt(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read quality receipt: {error}") from error


def _quality_subject_sha(value: Any, subject: Any) -> Optional[str]:
    if value.get("schema_version") == "leone.quality-comparison.v1":
        model_family = value.get("model_family", {})
        subject_model = model_family.get("subject", {}) if isinstance(model_family, dict) else {}
        return subject_model.get("sha256") if isinstance(subject_model, dict) else None
    artifact = subject.get("model_artifact", {}) if isinstance(subject, dict) else {}
    return artifact.get("sha256") if isinstance(artifact, dict) else None


def validate_quality_build(quality, build, binary_sha):
    for record in quality.values():
        if not record.get("comparison_schema"):
            raise HarnessError("frozen comparison requires the common quality-comparison schema")
        if record.get("build_info") != build or record.get("executable", {}).get("sha256") != binary_sha:
            raise HarnessError("quality and service must use the same Leone build and executable")
        for metrics in record["metrics"].values():
            if not metrics.get("receipt_sha256") or not metrics.get("kld", {}).get("definition"):
                raise HarnessError("frozen quality requires full-precision linked receipts and a KLD definition")



def _write_append_only(path: Path, value: Mapping[str, Any]) -> None:
    """Create one receipt and refuse to replace an existing record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    mode = 0o644
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as error:
        raise HarnessError(f"refusing to replace existing receipt: {path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _tail(path: Path, limit: int = 16_384) -> str:
    try:
        data = path.read_bytes()
    except OSError as error:
        return str(error)
    return data[-limit:].decode("utf-8", "replace")


def sample_load_hardware(stop, samples, origin):
    while not stop.is_set():
        sample = nvidia_metadata()
        sample["elapsed_ms"] = (monotonic_ns() - origin) / 1_000_000
        samples.append(sample)
        stop.wait(0.5)


@dataclass
class VariantOutcome:
    health: Optional[Dict[str, Any]] = None
    requests: List[RequestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    disconnect: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=lambda: {"status": "not_requested"})


def _run_variant(
    root: Path,
    manifest: Mapping[str, Any],
    variant: Mapping[str, Any],
    expanded_requests: Sequence[Mapping[str, Any]],
    offsets_ms: Sequence[float],
    repetition: int,
    order_position: int,
    model: Path,
    plan: Optional[Path],
    leone_binary: Path,
    llama_binary: Path,
    base_port: int,
    artifact_dir: Path,
    cpu_mask: Optional[str],
) -> Dict[str, Any]:
    variant_id = str(variant["id"])
    port = base_port + repetition * 10 + order_position
    run_dir = artifact_dir / f"rep-{repetition:03d}" / variant_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "server.log"
    receipt_dir = run_dir / "response-receipts"
    signing_key = run_dir / "response.key"
    argv = build_launch_argv(
        root,
        manifest,
        variant,
        model,
        plan,
        str(manifest["server"]["host"]),
        port,
        log_path,
        receipt_dir,
        signing_key,
        leone_binary,
        llama_binary,
        cpu_mask,
    )
    base_url = f"http://{manifest['server']['host']}:{port}"
    server = ManagedServer(
        argv,
        base_url,
        log_path,
        root,
        float(manifest["server"]["startup_timeout_s"]),
        float(manifest["workload"]["request_timeout_s"]),
    )
    hardware_start = nvidia_metadata()
    started_ns = monotonic_ns()
    startup_error: Optional[str] = None
    outcome = VariantOutcome()
    stop_status: Dict[str, Any] = {"returncode": None, "signal": None}
    load_samples = []
    try:
        _run_variant_work(
            server,
            manifest,
            variant,
            base_url,
            expanded_requests,
            offsets_ms,
            load_samples,
            outcome,
        )
    except Exception as error:
        startup_error = f"{type(error).__name__}: {error}"
        if not outcome.disconnect:
            outcome.disconnect = {"status": "not_run", "error": startup_error}
    finally:
        stop_status = server.stop()
    finished_ns = monotonic_ns()
    hardware_end = nvidia_metadata()
    log_hash = None
    if log_path.exists():
        try:
            log_hash = sha256_file(log_path)
        except OSError:
            log_hash = None
    if startup_error and not outcome.requests:
        outcome.summary = {
            "request_count": len(expanded_requests),
            "completed_count": 0,
            "loss_count": len(expanded_requests),
            "outcomes": {"server_error": len(expanded_requests)},
            "error": startup_error,
            "token_itl_ms": None,
            "token_boundaries_verified": False,
        }
    return {
        "variant": variant_id,
        "repetition": repetition,
        "order_position": order_position,
        "port": port,
        "base_url": base_url,
        "launch_argv": argv,
        "health": outcome.health,
        "startup_error": startup_error,
        "server_wall_ms": (finished_ns - started_ns) / 1_000_000,
        "process": stop_status,
        "server_log_sha256": log_hash,
        "hardware_start": hardware_start,
        "hardware_end": hardware_end,
        "hardware_during_service": load_samples,
        "requests": [result.to_dict() for result in outcome.requests],
        "summary": outcome.summary,
        "disconnect_recovery": outcome.disconnect,
        "resident_trace": outcome.trace,
    }


def _run_variant_work(
    server: ManagedServer,
    manifest: Mapping[str, Any],
    variant: Mapping[str, Any],
    base_url: str,
    expanded_requests: Sequence[Mapping[str, Any]],
    offsets_ms: Sequence[float],
    load_samples: List[Dict[str, Any]],
    outcome: VariantOutcome,
) -> None:
    load_stop = threading.Event()
    sampler = threading.Thread(
        target=sample_load_hardware,
        args=(load_stop, load_samples, monotonic_ns()),
        daemon=True,
    )
    try:
        outcome.health = server.start()
        sampler.start()
        outcome.requests, outcome.summary = run_offered_load(
            base_url,
            expanded_requests,
            offsets_ms,
            str(manifest["workload"].get("model_name", "leone")),
            float(manifest["workload"]["request_timeout_s"]),
            int(manifest["budgets"]["minimum_samples_for_quantiles"]),
        )
        outcome.disconnect = _run_variant_disconnect(manifest, base_url, expanded_requests)
        _capture_variant_trace(manifest, variant, base_url, outcome)
    finally:
        load_stop.set()
        if sampler.ident is not None:
            sampler.join(timeout=12)


def _run_variant_disconnect(
    manifest: Mapping[str, Any],
    base_url: str,
    expanded_requests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    request = dict(expanded_requests[0])
    request["max_tokens"] = int(manifest["workload"]["disconnect_max_tokens"])
    workload = manifest["workload"]
    return run_disconnect_probe(
        base_url,
        request,
        str(workload.get("model_name", "leone")),
        float(workload["request_timeout_s"]),
        float(workload["disconnect_recovery_wait_s"]),
    )


def _capture_variant_trace(
    manifest: Mapping[str, Any], variant: Mapping[str, Any], base_url: str,
    outcome: VariantOutcome,
) -> None:
    trace_config = manifest.get("trace", {})
    endpoint = trace_config.get("endpoint")
    if not endpoint or variant["kind"] != "leone":
        return
    try:
        trace = get_json(
            base_url,
            str(endpoint),
            float(manifest["workload"]["request_timeout_s"]),
        )
        outcome.trace = {
            "status": "captured",
            "validation": resident_progress_trace(trace),
            "body_sha256": sha256_bytes(canonical_json(trace)),
            "body": trace,
        }
    except Exception as error:
        outcome.trace = {"status": "unavailable", "error": str(error)}
        if trace_config.get("required"):
            raise


def _paired_comparison(
    runs: Sequence[Mapping[str, Any]], metric: str, minimum_samples: int
) -> Dict[str, Any]:
    by_variant: Dict[str, Dict[Tuple[int, str], float]] = {}
    for run in runs:
        variant = str(run["variant"])
        values = by_variant.setdefault(variant, {})
        for item in run.get("requests", []):
            if item.get("status") != "completed":
                continue
            value = item.get(metric)
            if isinstance(value, (int, float)):
                values[(int(run["repetition"]), str(item["request_id"]))] = float(value)
    baseline = by_variant.get("leone_batch1", {})
    comparisons: Dict[str, Any] = {}
    for variant, values in by_variant.items():
        if variant == "leone_batch1":
            continue
        ratios = [
            values[key] / baseline[key]
            for key in sorted(values.keys() & baseline.keys())
            if baseline[key] > 0
        ]
        comparisons[variant] = {
            "baseline": "leone_batch1",
            "metric": metric,
            "paired_sample_count": len(ratios),
            "ratio_definition": f"{variant} value divided by leone_batch1 value for the same repetition and request",
            "ratios": empirical_quantiles(ratios, minimum_samples),
        }
    return comparisons


def _aggregate_runs(
    runs: Sequence[Mapping[str, Any]], minimum_samples: int
) -> Dict[str, Any]:
    variants: Dict[str, List[Mapping[str, Any]]] = {}
    for run in runs:
        variants.setdefault(str(run["variant"]), []).append(run)
    aggregate = {
        variant: _aggregate_variant(variant_runs, minimum_samples)
        for variant, variant_runs in variants.items()
    }
    for metric in ("first_nonempty_content_ttft_ms", "completion_latency_ms"):
        key = "ttft_ms" if metric.startswith("first") else "completion_latency_ms"
        aggregate[f"paired_{key}"] = _paired_comparison(runs, metric, minimum_samples)
    return aggregate


def _aggregate_variant(
    runs: Sequence[Mapping[str, Any]], minimum_samples: int
) -> Dict[str, Any]:
    results = [
        _request_result_from_record(item)
        for run in runs
        for item in run.get("requests", [])
    ]
    summary = summarize_requests(results, minimum_samples)
    run_rates = [
        float(run["summary"]["aggregate_completion_tok_s"])
        for run in runs
        if isinstance(run.get("summary", {}).get("aggregate_completion_tok_s"), (int, float))
    ]
    summary["aggregate_completion_tok_s"] = empirical_quantiles(
        run_rates, minimum_samples
    )
    return summary


def _request_result_from_record(item: Mapping[str, Any]) -> RequestResult:
    usage = item.get("usage", {})
    content = item.get("content", {})
    return RequestResult(
        request_id=str(item["request_id"]),
        prompt_id=str(item["prompt_id"]),
        planned_offset_ms=float(item["planned_offset_ms"]),
        status=str(item["status"]),
        http_status=item.get("http_status"),
        error=item.get("error"),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        total_tokens=usage.get("total_tokens"),
        content_sha256=content.get("sha256"),
        content_bytes=content.get("bytes"),
        metrics={
            key: item.get(key)
            for key in (
                "first_nonempty_content_ttft_ms",
                "completion_latency_ms",
                "content_event_intervals_ms",
            )
            if key in item
        },
    )


def _prompt_token_comparison(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Check reported prompt-token counts for every shared request."""

    observed = _observed_prompt_tokens(runs)
    missing, mismatches = _prompt_token_issues(observed)
    expected_keys = {
        (int(run["repetition"]), str(item["request_id"]))
        for run in runs
        for item in run.get("requests", [])
    }
    missing.extend(_missing_prompt_keys(expected_keys, observed))
    return {
        "required": True,
        "all_shared_prompt_token_counts_match": not missing and not mismatches,
        "missing": missing,
        "mismatches": mismatches,
    }


def _observed_prompt_tokens(
    runs: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[int, str], Dict[str, int]]:
    observed: Dict[Tuple[int, str], Dict[str, int]] = {}
    for run in runs:
        variant = str(run["variant"])
        repetition = int(run["repetition"])
        for item in run.get("requests", []):
            usage = item.get("usage", {})
            count = usage.get("prompt_tokens")
            if item.get("status") != "completed" or not isinstance(count, int):
                continue
            observed.setdefault((repetition, str(item["request_id"])), {})[variant] = count
    return observed


def _prompt_token_issues(
    observed: Mapping[Tuple[int, str], Mapping[str, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    missing: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    expected_variants = set(VARIANT_IDS)
    for key in sorted(observed):
        values = observed[key]
        if set(values) != expected_variants:
            missing.append(
                {
                    "repetition": key[0],
                    "request_id": key[1],
                    "variants": sorted(expected_variants - set(values)),
                }
            )
        elif len(set(values.values())) != 1:
            mismatches.append(
                {
                    "repetition": key[0],
                    "request_id": key[1],
                    "prompt_tokens": values,
                }
            )
    return missing, mismatches


def _missing_prompt_keys(
    expected_keys: Iterable[Tuple[int, str]],
    observed: Mapping[Tuple[int, str], Mapping[str, int]],
) -> List[Dict[str, Any]]:
    expected_variants = set(VARIANT_IDS)
    expected = set(expected_keys)
    return [
        {
            "repetition": key[0],
            "request_id": key[1],
            "variants": sorted(expected_variants),
        }
        for key in sorted(expected - set(observed))
    ]


def _context_limit_check(
    runs: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    per_request_limit: int,
) -> Dict[str, Any]:
    """Check observed prompt plus completion usage against the request limit."""

    max_tokens = {str(request["id"]): int(request["max_tokens"]) for request in requests}
    violations: List[Dict[str, Any]] = []
    observed = 0
    for run in runs:
        for item in run.get("requests", []):
            usage = item.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
                continue
            observed += 1
            total = prompt_tokens + completion_tokens
            if total > per_request_limit:
                violations.append(
                    {
                        "variant": run.get("variant"),
                        "repetition": run.get("repetition"),
                        "request_id": item.get("request_id"),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "limit": per_request_limit,
                    }
                )
    return {
        "per_request_limit": per_request_limit,
        "observed_completed_requests": observed,
        "violations": violations,
        "passed": not violations,
        "max_tokens_by_request": max_tokens,
    }


def release_outcome_errors(receipt: Mapping[str, Any]) -> List[str]:
    """Check completion, recovery, and trace evidence for the frozen workload."""
    errors = []
    if receipt.get("budget_exhausted"):
        errors.append("study budget was exhausted")
    for run in receipt.get("runs", []):
        label = f"{run.get('variant')} repetition {run.get('repetition')}"
        errors.extend(_run_outcome_errors(run, label))
    return errors


def _run_outcome_errors(run: Mapping[str, Any], label: str) -> List[str]:
    errors = []
    if run.get("startup_error"):
        errors.append(f"{label}: server or telemetry error")
    if any(item.get("status") != "completed" for item in run.get("requests", [])):
        errors.append(f"{label}: incomplete request")
    if not run.get("disconnect_recovery", {}).get("passed"):
        errors.append(f"{label}: disconnect recovery failed")
    if str(run.get("variant", "")).startswith("leone_"):
        errors.extend(_trace_outcome_errors(run, label))
    return errors


def _trace_outcome_errors(run: Mapping[str, Any], label: str) -> List[str]:
    errors = []
    trace_record = run.get("resident_trace", {})
    body = trace_record.get("body")
    validation = resident_progress_trace(body)
    if not validation["resident_progress_proven"]:
        errors.append(f"{label}: resident progress was not demonstrated")
    elif validation != trace_record.get("validation"):
        errors.append(f"{label}: trace validation differs")
    if isinstance(body, dict):
        if sha256_bytes(canonical_json(body)) != trace_record.get("body_sha256"):
            errors.append(f"{label}: trace body hash differs")
        if body.get("dropped_events", 0) or body.get("dropped_memory_samples", 0):
            errors.append(f"{label}: trace is partial")
        if not body.get("memory_samples"):
            errors.append(f"{label}: physical memory samples are missing")
    return errors


def validate_recorded_receipt(path: Path, root: Path) -> List[str]:
    """Validate hashes, counts, summaries, and flags in one recorded study."""

    errors: List[str] = []
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"cannot read receipt: {error}"]
    if not isinstance(receipt, dict):
        return ["receipt root is not an object"]
    if receipt.get("schema_version") != SCHEMA_VERSION:
        errors.append("unexpected receipt schema_version")
    manifest_errors, manifest_body = _validate_manifest_info(receipt, root)
    errors.extend(manifest_errors)
    errors.extend(_validate_model_and_plan(receipt, root))
    errors.extend(_validate_quality_provenance(receipt, root))
    errors.extend(_validate_source_and_binaries(receipt, root))
    runs = receipt.get("runs")
    if not isinstance(runs, list):
        errors.append("receipt runs is not a list")
        return errors
    errors.extend(_validate_run_shapes(receipt, runs))
    errors.extend(_validate_derived_receipt_fields(receipt, runs))
    recorded_requests = receipt.get("workload", {}).get("requests", [])
    if receipt.get("phase") != "frozen":
        errors.append("release evidence is not frozen")
    errors.extend(
        _validate_manifest_consistency(receipt, manifest_body, runs, recorded_requests)
    )
    errors.extend(release_outcome_errors(receipt))
    return errors


def _validate_manifest_info(
    receipt: Mapping[str, Any], root: Path
) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    manifest_info = receipt.get("manifest", {})
    manifest_body = manifest_info.get("body") if isinstance(manifest_info, dict) else None
    if not isinstance(manifest_body, dict):
        return ["receipt does not contain manifest.body"], None
    errors = []
    manifest_path_value = manifest_info.get("path")
    if isinstance(manifest_path_value, str):
        errors.extend(_validate_manifest_file(manifest_info, manifest_body, root / manifest_path_value))
    if manifest_info.get("canonical_sha256") != sha256_bytes(canonical_json(manifest_body)):
        errors.append("manifest canonical SHA-256 does not match body")
    return errors, manifest_body


def _validate_manifest_file(
    manifest_info: Mapping[str, Any], manifest_body: Mapping[str, Any], path: Path
) -> List[str]:
    if not path.is_file():
        return [f"recorded manifest is missing: {path}"]
    raw = path.read_bytes()
    errors = []
    if manifest_info.get("file_sha256") != sha256_bytes(raw):
        errors.append("manifest file SHA-256 does not match receipt")
    try:
        on_disk = json.loads(raw)
    except json.JSONDecodeError:
        errors.append("manifest file is not JSON")
    else:
        if on_disk != manifest_body:
            errors.append("manifest body differs from recorded manifest")
    return errors


def _validate_model_and_plan(receipt: Mapping[str, Any], root: Path) -> List[str]:
    errors = []
    model_info = receipt.get("model", {})
    if isinstance(model_info, dict) and isinstance(model_info.get("path"), str):
        model_path = root / model_info["path"]
        if not model_path.is_file():
            errors.append(f"recorded model is missing: {model_path}")
        elif model_info.get("sha256") != sha256_file(model_path):
            errors.append("model SHA-256 does not match receipt")
    else:
        errors.append("receipt does not contain model path")
    plan = receipt.get("plan")
    if plan is not None:
        errors.extend(_validate_optional_file(plan, root, "plan"))
    return errors


def _validate_optional_file(info: Any, root: Path, name: str) -> List[str]:
    if not isinstance(info, dict) or not isinstance(info.get("path"), str):
        return [f"receipt {name} provenance is malformed"]
    error = _file_hash_error(
        info, root, f"recorded {name} is missing", f"{name} SHA-256 does not match receipt"
    )
    return [error] if error else []


def _validate_quality_provenance(receipt: Mapping[str, Any], root: Path) -> List[str]:
    quality = receipt.get("quality", {})
    quality_items = (
        quality.items()
        if isinstance(quality, dict) and quality and all(
            isinstance(item, dict) for item in quality.values()
        )
        else []
    )
    if not quality_items:
        return ["receipt does not contain quality provenance"]
    errors = []
    for engine, quality_info in quality_items:
        errors.extend(_validate_quality_item(engine, quality_info, root))
    return errors


def _validate_quality_item(
    engine: str, quality_info: Mapping[str, Any], root: Path
) -> List[str]:
    path = quality_info.get("path")
    if not isinstance(path, str):
        return [f"quality provenance has no path: {engine}"]
    errors = []
    error = _file_hash_error(
        quality_info,
        root,
        "recorded quality receipt is missing",
        f"quality receipt SHA-256 does not match receipt: {engine}",
    )
    if error:
        errors.append(error)
    if engine == "common" and not {"leone_q4", "llama_q4"}.issubset(
        set(quality_info.get("comparison_engines", []))
    ):
        errors.append("common quality receipt lacks both engine rows")
    return errors


def _file_hash_error(
    info: Mapping[str, Any], root: Path, missing: str, changed: str
) -> Optional[str]:
    target = root / info["path"]
    if not target.is_file():
        return f"{missing}: {target}"
    if info.get("sha256") != sha256_file(target):
        return changed
    return None


def _validate_source_and_binaries(receipt: Mapping[str, Any], root: Path) -> List[str]:
    errors = _validate_source(receipt.get("source", {}), root)
    llama_info = receipt.get("llama_cpp", {})
    if isinstance(llama_info, dict) and not llama_info.get("commit_matches_pinned"):
        errors.append("llama.cpp source does not match external/PINNED")
    errors.extend(_validate_binary_hashes(receipt.get("binaries") or {}, root))
    return errors


def _validate_source(source: Any, root: Path) -> List[str]:
    if not isinstance(source, dict) or not isinstance(source.get("commit"), str):
        return ["receipt does not contain a source commit"]
    code, _, _ = _run_command(
        [sys.executable, str(root / "scripts/source_inputs.py"), "check", source["commit"]],
        timeout_s=5,
    )
    return [] if code == 0 else ["recorded source inputs do not match"]


def _validate_binary_hashes(binaries: Any, root: Path) -> List[str]:
    errors = []
    for binary_name, binary_info in binaries.items():
        if not isinstance(binary_info, dict):
            errors.append(f"binary provenance is malformed: {binary_name}")
            continue
        for item in binary_info.get("binary_hashes", []):
            errors.extend(_validate_binary_hash(binary_name, item, root))
    return errors


def _validate_binary_hash(
    binary_name: str, item: Any, root: Path
) -> List[str]:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        return [f"binary hash entry is malformed: {binary_name}"]
    error = _file_hash_error(
        item, root, "linked binary file is missing", f"linked binary hash changed: {root / item['path']}"
    )
    return [error] if error else []


def _validate_run_shapes(
    receipt: Mapping[str, Any], runs: Sequence[Any]
) -> List[str]:
    errors = []
    expected_repetitions = receipt.get("workload", {}).get("repetitions")
    if isinstance(expected_repetitions, int) and len(runs) != expected_repetitions * len(VARIANT_IDS):
        errors.append("run count does not equal repetitions times variants")
    expected_count = receipt.get("workload", {}).get("request_count_per_repetition")
    for run in runs:
        errors.extend(_validate_run_shape(run, expected_count))
    return errors


def _validate_run_shape(run: Any, expected_count: Any) -> List[str]:
    if not isinstance(run, dict):
        return ["receipt contains a non-object run"]
    requests = run.get("requests")
    if not isinstance(requests, list):
        return [f"run {run.get('variant')} has no request list"]
    errors = []
    if isinstance(expected_count, int) and len(requests) != expected_count:
        errors.append(f"run {run.get('variant')} has the wrong request count")
    for item in requests:
        if not isinstance(item, dict):
            errors.append("run contains a non-object request")
        elif item.get("token_itl_ms") is not None or item.get("token_boundaries_verified"):
            errors.append("receipt reports an unverified token timing metric")
    return errors


def _validate_derived_receipt_fields(
    receipt: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]
) -> List[str]:
    errors = []
    minimum = int(receipt.get("workload", {}).get("quantiles", {}).get("minimum_samples", 1))
    expected_summary = _aggregate_runs(runs, minimum)
    if canonical_json(expected_summary) != canonical_json(receipt.get("summary")):
        errors.append("receipt summary does not recompute from request outcomes")
    prompt_check = _prompt_token_comparison(runs)
    if receipt.get("comparability", {}).get("prompt_tokens") != prompt_check:
        errors.append("prompt-token comparability check does not recompute")
    recorded_requests = receipt.get("workload", {}).get("requests", [])
    context = receipt.get("comparability", {}).get("context_limits", {})
    per_request_limit = context.get("per_request_limit")
    if isinstance(per_request_limit, int) and isinstance(recorded_requests, list):
        context_check = _context_limit_check(runs, recorded_requests, per_request_limit)
        if context != context_check:
            errors.append("context limit check does not recompute")
    return errors


def _validate_manifest_consistency(
    receipt: Mapping[str, Any],
    manifest_body: Optional[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
    recorded_requests: Any,
) -> List[str]:
    if not isinstance(manifest_body, dict):
        return []
    errors = []
    try:
        _validate_manifest(manifest_body, False)
        expected_orders = _balanced_orders(VARIANT_IDS, manifest_body["workload"]["repetitions"])
        if receipt.get("ordering", {}).get("orders") != expected_orders:
            errors.append("run ordering differs from balanced manifest schedule")
        expected_requests = _expand_requests(manifest_body)
        if recorded_requests != expected_requests:
            errors.append("recorded workload differs from manifest")
        _validate_manifest_runs(receipt, runs, expected_orders, expected_requests, manifest_body, errors)
        validate_quality_build(
            receipt["quality"],
            receipt["binaries"]["leone"]["build_info"]["value"],
            receipt["binaries"]["leone"]["binary_hashes"][0]["sha256"],
        )
    except (HarnessError, MalformedSseError, KeyError, IndexError, TypeError) as error:
        errors.append(f"manifest consistency failed: {error}")
    return errors


def _validate_manifest_runs(
    receipt: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    expected_orders: Sequence[Sequence[str]],
    expected_requests: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    errors: List[str],
) -> None:
    by_id = {item["id"]: item for item in expected_requests}
    offsets_by_id = dict(
        zip((request["id"] for request in expected_requests), manifest["workload"]["stagger_ms"])
    )
    for index, run in enumerate(runs):
        repetition, position = divmod(index, len(VARIANT_IDS))
        if (run["repetition"], run["order_position"], run["variant"]) != (
            repetition, position, expected_orders[repetition][position]
        ):
            errors.append("run identity differs from balanced schedule")
        for item in run["requests"]:
            request_id = item["request_id"]
            expected = _request_body(by_id[request_id], manifest["workload"]["model_name"], True)
            if item.get("request_body_sha256") != sha256_bytes(expected):
                errors.append("request body hash differs from manifest")
            if item.get("planned_offset_ms") != offsets_by_id[request_id]:
                errors.append("planned request offset differs from manifest")
            _parse_usage(item["usage"])


def _manifest_paths(
    root: Path, manifest: Mapping[str, Any], args: argparse.Namespace
) -> Tuple[Path, Dict[str, Path], Optional[Path]]:
    model_value = args.model or manifest.get("model", {}).get("path")
    quality_body = manifest.get("quality", {})
    plan_value = args.plan or manifest.get("model", {}).get("plan")
    if not model_value or not isinstance(quality_body, dict):
        raise HarnessError("manifest or CLI must provide model and quality receipt paths")
    model = (root / model_value).resolve()
    quality_paths = _quality_paths(root, quality_body, args.quality_receipt)
    plan = (root / plan_value).resolve() if plan_value else None
    _require_manifest_paths(model, quality_paths, plan)
    return model, quality_paths, plan


def _quality_paths(
    root: Path, quality_body: Mapping[str, Any], quality_receipt: Optional[str]
) -> Dict[str, Path]:
    quality_paths: Dict[str, Path] = {}
    if quality_receipt:
        quality_paths["leone"] = (root / quality_receipt).resolve()
    elif isinstance(quality_body.get("paths"), dict):
        for engine, value in quality_body["paths"].items():
            if not isinstance(value, str):
                raise HarnessError(f"quality.paths.{engine} must be a string")
            quality_paths[str(engine)] = (root / value).resolve()
    elif isinstance(quality_body.get("path"), str):
        scope = quality_body.get("scope", ["leone"])
        key = "common" if isinstance(scope, list) and len(scope) > 1 else "leone"
        quality_paths[key] = (root / quality_body["path"]).resolve()
    else:
        raise HarnessError("manifest or CLI must provide quality receipt paths")
    return quality_paths


def _require_manifest_paths(
    model: Path, quality_paths: Mapping[str, Path], plan: Optional[Path]
) -> None:
    if not model.is_file():
        raise HarnessError(f"model does not exist: {model}")
    for quality in quality_paths.values():
        if not quality.is_file():
            raise HarnessError(f"quality receipt does not exist: {quality}")
    if plan is not None and not plan.is_file():
        raise HarnessError(f"plan does not exist: {plan}")


def _dry_run(
    root: Path,
    manifest: Mapping[str, Any],
    model: Path,
    plan: Optional[Path],
    leone_binary: Path,
    llama_binary: Path,
    cpu_mask: Optional[str],
) -> Dict[str, Any]:
    """Render launch commands and the fixed schedule without starting servers."""

    with tempfile.TemporaryDirectory(prefix="leone-concurrent-dry-run-") as temp:
        temp_path = Path(temp)
        rendered = []
        for variant in manifest["variants"]:
            rendered.append(
                {
                    "variant": variant["id"],
                    "argv": build_launch_argv(
                        root,
                        manifest,
                        variant,
                        model,
                        plan,
                        str(manifest["server"]["host"]),
                        int(manifest["server"]["base_port"]),
                        temp_path / "server.log",
                        temp_path / "responses",
                        temp_path / "response.key",
                        leone_binary,
                        llama_binary,
                        cpu_mask,
                    ),
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_phase": manifest["phase"],
        "launches": rendered,
        "requests": _expand_requests(manifest),
        "stagger_ms": manifest["workload"]["stagger_ms"],
        "orders": _balanced_orders(VARIANT_IDS, int(manifest["workload"]["repetitions"])),
    }


def public_metadata(value: Any, root: Path, artifact_dir: Path) -> Any:
    """Replace local workspace and temporary artifact prefixes before recording."""
    if isinstance(value, str):
        return value.replace(str(artifact_dir), "<run-artifacts>").replace(str(root), ".")
    if isinstance(value, list):
        return [public_metadata(item, root, artifact_dir) for item in value]
    if isinstance(value, dict):
        return {key: public_metadata(item, root, artifact_dir) for key, item in value.items()}
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the preregistered streaming service comparison."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--validate-receipt", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--plan")
    parser.add_argument("--quality-receipt")
    parser.add_argument("--leone-binary", type=Path, default=Path("target/release/leone"))
    parser.add_argument(
        "--llama-binary",
        type=Path,
        default=Path("external/llama.cpp/build/bin/llama-server"),
    )
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--cpu-mask")
    parser.add_argument("--base-port", type=int)
    parser.add_argument("--allow-pending-freeze", action="store_true")
    parser.add_argument("--allow-dirty-source", action="store_true")
    parser.add_argument("--allow-unverified-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _study_provenance(
    root: Path,
    args: argparse.Namespace,
    model: Path,
    quality_paths: Mapping[str, Path],
    leone_binary: Path,
    llama_binary: Path,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    source = _git_provenance(root)
    if not source["tracked_tree_clean"] and not args.allow_dirty_source:
        raise HarnessError(
            "source tree has tracked changes; pass --allow-dirty-source for exploration"
        )
    model_sha = sha256_file(model)
    quality = {
        engine: _quality_provenance(root, quality_path, model_sha)
        for engine, quality_path in sorted(quality_paths.items())
    }
    binaries = {
        "leone": binary_provenance(root, leone_binary),
        "llama_server": binary_provenance(root, llama_binary),
    }
    llama_source = llama_source_provenance(root)
    build_info = binaries["leone"]["build_info"]
    build_commit = (
        build_info.get("value", {}).get("source_commit")
        if build_info.get("available")
        else None
    )
    source["build_info_commit"] = build_commit
    source["build_info_commit_matches"] = bool(
        build_commit and build_commit == source["commit"]
    )
    source["build_info_source_tree_dirty"] = (
        build_info.get("value", {}).get("source_tree_dirty")
        if build_info.get("available")
        else None
    )
    return source, model_sha, quality, binaries, llama_source


def _validate_frozen_inputs(
    manifest: Mapping[str, Any],
    args: argparse.Namespace,
    quality: Mapping[str, Any],
    binaries: Mapping[str, Any],
    source: Mapping[str, Any],
    llama_source: Mapping[str, Any],
    leone_binary: Path,
) -> None:
    if manifest["phase"] != "frozen" or args.allow_unverified_build:
        return
    build_info = binaries["leone"]["build_info"]
    validate_quality_build(quality, build_info.get("value"), sha256_file(leone_binary))
    _require_common_quality(quality)
    _validate_frozen_provenance(source, llama_source)


def _require_common_quality(quality: Mapping[str, Any]) -> None:
    common_quality = quality.get("common")
    if not (
        isinstance(common_quality, dict)
        and common_quality.get("comparison_schema")
        and {"leone_q4", "llama_q4"}.issubset(
            set(common_quality.get("comparison_engines", []))
        )
    ):
        raise HarnessError("frozen run requires a common quality comparison")


def _validate_frozen_provenance(
    source: Mapping[str, Any], llama_source: Mapping[str, Any]
) -> None:
    if not source["commit_verified"] or not source["build_info_commit_matches"]:
        raise HarnessError("frozen run requires build-info source_commit matching git HEAD")
    if source["build_info_source_tree_dirty"] is not False:
        raise HarnessError("frozen run requires a clean tracked source build")
    if not llama_source["commit_matches_pinned"] or not llama_source["tracked_tree_clean"]:
        raise HarnessError("frozen run requires a clean llama.cpp checkout at external/PINNED")


def _execute_study_runs(
    root: Path,
    manifest: Mapping[str, Any],
    expanded_requests: Sequence[Mapping[str, Any]],
    offsets: Sequence[float],
    orders: Sequence[Sequence[str]],
    model: Path,
    plan: Optional[Path],
    leone_binary: Path,
    llama_binary: Path,
    artifact_dir: Path,
    cpu_mask: Optional[str],
    base_port: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    started = time.monotonic()
    runs: List[Dict[str, Any]] = []
    for repetition, order in enumerate(orders):
        for order_position, variant_id in enumerate(order):
            if time.monotonic() - started >= manifest["budgets"]["wall_time_limit_s"]:
                return runs, True
            variant = next(item for item in manifest["variants"] if item["id"] == variant_id)
            runs.append(
                _run_variant(
                    root,
                    manifest,
                    variant,
                    expanded_requests,
                    offsets,
                    repetition,
                    order_position,
                    model,
                    plan,
                    leone_binary,
                    llama_binary,
                    base_port,
                    artifact_dir,
                    cpu_mask,
                )
            )
    return runs, False


def _build_receipt(
    root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    manifest_digest: Mapping[str, str],
    source: Mapping[str, Any],
    model: Path,
    model_sha: str,
    plan: Optional[Path],
    quality: Mapping[str, Any],
    binaries: Mapping[str, Any],
    llama_source: Mapping[str, Any],
    hardware_start: Mapping[str, Any],
    budget_exhausted: bool,
    expanded_requests: Sequence[Mapping[str, Any]],
    offsets: Sequence[float],
    repetitions: int,
    orders: Sequence[Sequence[str]],
    runs: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    comparability: Mapping[str, Any],
) -> Dict[str, Any]:
    position_counts = _position_counts(orders, VARIANT_IDS)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": manifest["phase"],
        "manifest": {
            "path": _relative_path(root, manifest_path),
            **manifest_digest,
            "body": manifest,
        },
        "source": source,
        "model": {"path": _relative_path(root, model), "sha256": model_sha},
        "plan": (
            {"path": _relative_path(root, plan), "sha256": sha256_file(plan)}
            if plan is not None
            else None
        ),
        "quality": quality,
        "binaries": binaries,
        "llama_cpp": llama_source,
        "system": system_metadata(),
        "hardware": {"study_start": hardware_start, "study_end": nvidia_metadata()},
        "budget_exhausted": budget_exhausted,
        "budget_limit": "Wall time is checked before each run; a started run completes or times out.",
        "roofline": {
            "status": "unavailable",
            "reason": "mixed streaming service runs do not expose per-tensor-class byte counters; no batch-1 eta is reused",
        },
        "workload": {
            "request_count_per_repetition": len(expanded_requests),
            "repetitions": repetitions,
            "stagger_ms": offsets,
            "requests": expanded_requests,
            "quantiles": {
                "definition": QUANTILE_DEFINITION,
                "minimum_samples": manifest["budgets"]["minimum_samples_for_quantiles"],
            },
            "stream_timing": {
                "first_content_field": "choices[0].delta.content",
                "first_content_definition": "elapsed monotonic time until the client receives a complete SSE frame with nonempty delta.content",
                "content_event_intervals_definition": CONTENT_EVENT_INTERVAL_DEFINITION,
                "token_boundaries_verified": False,
                "token_itl_ms": None,
                "limit": TOKEN_BOUNDARY_LIMIT,
            },
            "completion_throughput_definition": "completion usage token count divided by measured request start to stream completion latency; offered-load throughput divides all observed completion tokens by the fixed schedule wall time",
        },
        "ordering": {
            "orders": orders,
            "position_counts": position_counts,
            "balanced": all(len(set(counts)) <= 1 for counts in position_counts.values()),
        },
        "runs": runs,
        "summary": summary,
        "comparability": comparability,
        "limits": [
            "SSE content events are transport events, not verified token boundaries.",
            "Events completed by one socket read share the read-completion timestamp; zero intervals are counted.",
            "Prompt token counts match, but cross-engine prompt token identities are not captured by the API.",
            f"llama.cpp can decode up to {manifest['server']['sessions']} slots; Leone dispatch limits are recorded per variant.",
            "Clocks are not locked by this harness; hardware samples record the observed state.",
            "nvidia-smi runs every 0.5 seconds on the study CPU mask; its CPU and driver queries can perturb timing.",
            "Completion throughput uses terminal usage.completion_tokens divided by request start to stream completion.",
            "A missing usage field is recorded as a loss for token throughput.",
            "Cross-engine content identity is recorded by digest and is not a token identity claim.",
        ],
    }


def _finish_study(
    args: argparse.Namespace,
    root: Path,
    artifact_dir: Path,
    manifest: Mapping[str, Any],
    receipt: Dict[str, Any],
    budget_exhausted: bool,
    comparability: Mapping[str, Any],
) -> int:
    output = _root_path(root, args.output)
    receipt = public_metadata(receipt, root, artifact_dir)
    _write_append_only(output, receipt)
    print(output)
    if manifest["phase"] == "frozen":
        errors = release_outcome_errors(receipt)
        if errors:
            print("error: " + "; ".join(errors), file=sys.stderr)
            return 1
    if budget_exhausted:
        print("error: study budget exhausted", file=sys.stderr)
        return 1
    if not comparability["prompt_tokens"]["all_shared_prompt_token_counts_match"]:
        print("error: shared prompt token counts do not match", file=sys.stderr)
        return 1
    if not comparability["context_limits"]["passed"]:
        print("error: observed usage exceeds the context limit", file=sys.stderr)
        return 1
    return 0


def _run_study(args: argparse.Namespace, root: Path) -> int:
    manifest_path = _root_path(root, args.manifest)
    manifest, manifest_digest = _load_manifest(manifest_path, args.allow_pending_freeze)
    model, quality_paths, plan = _manifest_paths(root, manifest, args)
    leone_binary = _root_path(root, args.leone_binary).resolve()
    llama_binary = _root_path(root, args.llama_binary).resolve()
    cpu_mask = args.cpu_mask or manifest["server"].get("cpu_mask")
    if args.dry_run:
        print(json.dumps(_dry_run(root, manifest, model, plan, leone_binary, llama_binary, cpu_mask), indent=2))
        return 0
    if args.output is None:
        raise HarnessError("--output is required unless --dry-run is used")
    source, model_sha, quality, binaries, llama_source = _study_provenance(
        root, args, model, quality_paths, leone_binary, llama_binary
    )
    _validate_frozen_inputs(
        manifest, args, quality, binaries, source, llama_source, leone_binary
    )
    workload = manifest["workload"]
    expanded_requests = _expand_requests(manifest)
    offsets = [float(value) for value in workload["stagger_ms"]]
    repetitions = int(workload["repetitions"])
    orders = _balanced_orders(VARIANT_IDS, repetitions)
    artifact_dir = args.artifact_dir or Path(
        tempfile.mkdtemp(prefix="leone-concurrent-service-")
    )
    artifact_dir = _root_path(root, artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    hardware_start = nvidia_metadata()
    runs, budget_exhausted = _execute_study_runs(
        root,
        manifest,
        expanded_requests,
        offsets,
        orders,
        model,
        plan,
        leone_binary,
        llama_binary,
        artifact_dir,
        cpu_mask,
        int(args.base_port or manifest["server"]["base_port"]),
    )
    minimum_samples = int(manifest["budgets"]["minimum_samples_for_quantiles"])
    summary = _aggregate_runs(runs, minimum_samples)
    server = manifest["server"]
    comparability = {
        "prompt_tokens": _prompt_token_comparison(runs),
        "context_limits": _context_limit_check(
            runs, expanded_requests, int(server["context_tokens"])
        ),
        "template": manifest["template"],
        "cache_policy": {
            "leone_session": "one unique session id per request id and fresh server process per run",
            "llama_prompt_cache": "disabled by --no-cache-prompt and --cache-ram 0",
            "shared_context_tokens_per_request": server["context_tokens"],
            "llama_context_tokens_total": server["context_tokens_total"],
        },
    }
    receipt = _build_receipt(
        root,
        manifest_path,
        manifest,
        manifest_digest,
        source,
        model,
        model_sha,
        plan,
        quality,
        binaries,
        llama_source,
        hardware_start,
        budget_exhausted,
        expanded_requests,
        offsets,
        repetitions,
        orders,
        runs,
        summary,
        comparability,
    )
    return _finish_study(
        args, root, artifact_dir, manifest, receipt, budget_exhausted, comparability
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the study or render its launch plan."""

    args = _parser().parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.validate_receipt is not None:
        receipt_path = _root_path(root, args.validate_receipt)
        errors = validate_recorded_receipt(receipt_path, root)
        print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.manifest is None:
        print("error: --manifest is required", file=sys.stderr)
        return 2
    try:
        return _run_study(args, root)
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
