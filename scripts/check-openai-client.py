#!/usr/bin/env python3
"""Checks the OpenAI Python client against a loopback Leone server."""

import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import socket
import struct
import time
import tempfile
import urllib.parse

from openai import BadRequestError, OpenAI


def digest(path):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def value_digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_client(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1"):
        raise ValueError("the client check requires a loopback HTTP server")
    return OpenAI(api_key="local-test", base_url=base_url, max_retries=0, timeout=180)


def verify_receipt(binary, receipt):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "response.json"
        path.write_text(json.dumps(receipt))
        subprocess.run(
            [str(binary), "receipt", "verify-response", str(path)],
            check=True, stdout=subprocess.DEVNULL, text=True,
        )


def check_response_usage(name, response, claim):
    usage = response.get("usage")
    if not usage:
        raise ValueError(f"{name}: usage differs from the signed token count")
    expected_usage = {
        "prompt_tokens": claim["prompt_tokens"],
        "completion_tokens": claim["generated_tokens"],
        "total_tokens": claim["prompt_tokens"] + claim["generated_tokens"],
    }
    if any(usage.get(field) != value for field, value in expected_usage.items()):
        raise ValueError(f"{name}: usage differs from the signed token count")
    return usage


def check_response_session(name, claim, expected_session):
    if expected_session is not None and claim["session"]["session_id"] != expected_session:
        raise ValueError(f"{name}: receipt session does not match the request")


def check_response_content(name, content, require_content):
    content = content or ""
    if require_content and not content:
        raise ValueError(f"{name}: the response has no text")
    return content


def response_record(
    binary, response, content, name, require_content=True, expected_session=None
):
    receipt = response.get("leone_receipt")
    if not receipt:
        raise ValueError(f"{name}: the response has no receipt")
    verify_receipt(binary, receipt)
    claim = receipt["claim"]
    usage = check_response_usage(name, response, claim)
    check_response_session(name, claim, expected_session)
    if claim["cancelled"]:
        raise ValueError(f"{name}: a completed workflow request was cancelled")
    content = check_response_content(name, content, require_content)
    record = {
        "name": name,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "usage": usage,
        "claim": claim,
        "leone_receipt": receipt,
        "receipt_sha256": value_digest(receipt),
        "signature_verified": True,
    }
    if expected_session is not None:
        record["expected_session"] = expected_session
    return record


def complete(client, binary, messages, name, **extra):
    response = client.chat.completions.create(
        model="leone", messages=messages, max_tokens=32, temperature=0,
        seed=0, extra_body={"leone_session": name, **extra},
    )
    return response_record(
        binary, response.model_dump(), response.choices[0].message.content, name,
        expected_session=name,
    ), response.choices[0].message.content


def append_tool_delta(tool_calls, delta):
    call = tool_calls.setdefault(
        delta.index,
        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
    )
    if delta.id:
        call["id"] = delta.id
    if delta.type:
        call["type"] = delta.type
    if delta.function is not None:
        call["function"]["name"] += delta.function.name or ""
        call["function"]["arguments"] += delta.function.arguments or ""


def collect_chunk(chunk, text, tool_calls):
    terminal = next(
        (chunk.model_dump() for choice in chunk.choices if choice.finish_reason),
        None,
    )
    for choice in chunk.choices:
        if choice.delta.content:
            text.append(choice.delta.content)
        for delta in choice.delta.tool_calls or []:
            append_tool_delta(tool_calls, delta)
    return terminal


def collect_stream(stream):
    terminal = None
    usage_chunk = None
    text = []
    tool_calls = {}
    for chunk in stream:
        usage_chunk = chunk.model_dump() if chunk.usage is not None else usage_chunk
        terminal = collect_chunk(chunk, text, tool_calls) or terminal
    calls = [tool_calls[index] for index in sorted(tool_calls)]
    return terminal, usage_chunk, "".join(text), calls


def open_stream(client, messages, extra):
    return client.chat.completions.create(
        model="leone", messages=messages, max_tokens=64, temperature=0,
        seed=0, stream=True, stream_options={"include_usage": True}, extra_body=extra,
    )


def stream_task(base_url, binary, messages, name, parent=None):
    extra = {"leone_session": name}
    if parent:
        extra["leone_fork_session"] = parent
    with make_client(base_url) as client:
        with open_stream(client, messages, extra) as stream:
            terminal, usage_chunk, text, _ = collect_stream(stream)
    if terminal is None:
        raise ValueError(f"{name}: the stream has no terminal event")
    if usage_chunk is None:
        raise ValueError(f"{name}: include_usage produced no usage event")
    terminal = dict(terminal)
    terminal["usage"] = usage_chunk["usage"]
    record = response_record(binary, terminal, text, name, expected_session=name)
    if parent:
        session = record["claim"]["session"]
        if session["reuse_class"] != "device-fork":
            raise ValueError(f"{name}: fork did not report device-fork reuse")
        if session["reused_tokens"] <= 0:
            raise ValueError(f"{name}: fork did not reuse parent tokens")
    return record


def check_stop(client, binary):
    response = client.chat.completions.create(
        model="leone", messages=[{"role": "user", "content": "Reply with the word STOP, followed by one sentence."}],
        max_tokens=32, stop=["STOP"],
        extra_body={"leone_session": "client-stop"},
    )
    record = response_record(
        binary, response.model_dump(), response.choices[0].message.content, "stop",
        require_content=False,
        expected_session="client-stop",
    )
    if response.choices[0].finish_reason != "stop":
        raise ValueError("stop: the model did not hit the matched stop sequence")
    if "STOP" in (response.choices[0].message.content or ""):
        raise ValueError("stop: the matched marker remained in content")
    record["name"] = "stop"
    return record


def validate_streamed_tool_call(terminal, usage_chunk, calls):
    if terminal is None or usage_chunk is None:
        raise ValueError("tools: stream has no terminal or usage event")
    if terminal["choices"][0]["finish_reason"] != "tool_calls":
        raise ValueError("tools: required tool choice did not finish with tool_calls")
    if len(calls) != 1:
        raise ValueError("tools: expected one streamed tool call")
    call = calls[0]
    if not call["id"] or call["type"] != "function":
        raise ValueError("tools: streamed call has no valid id or type")
    if call["function"]["name"] != "weather":
        raise ValueError("tools: named choice returned the wrong function")
    arguments = json.loads(call["function"]["arguments"])
    if arguments != {"city": "Oslo"}:
        raise ValueError(f"tools: unexpected arguments {arguments!r}")
    return call, arguments


def tool_definitions():
    return [{
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Read the weather",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]


def check_streamed_tool(client, binary, tools, request):
    with client.chat.completions.create(
        model="leone", messages=request, tools=tools,
        tool_choice={"type": "function", "function": {"name": "weather"}},
        max_tokens=64, temperature=0, seed=0, stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "leone_session": "client-tool-stream",
            "leone_template": "official",
        },
    ) as stream:
        terminal, usage_chunk, content, calls = collect_stream(stream)
    call, arguments = validate_streamed_tool_call(terminal, usage_chunk, calls)
    terminal = dict(terminal)
    terminal["usage"] = usage_chunk["usage"]
    return call, response_record(
        binary, terminal, content, "tool-stream", require_content=False,
        expected_session="client-tool-stream",
    )


def check_assistant_tool(client, binary, tools, request):
    assistant_response = client.chat.completions.create(
        model="leone", messages=request, tools=tools,
        tool_choice={"type": "function", "function": {"name": "weather"}},
        max_tokens=64, temperature=0, seed=0,
        extra_body={
            "leone_session": "client-tool-roundtrip",
            "leone_template": "official",
        },
    )
    assistant = assistant_response.choices[0].message
    assistant_history = assistant.model_dump()
    for field in ("refusal", "annotations", "audio", "function_call"):
        if field not in assistant_history or assistant_history[field] is not None:
            raise ValueError(f"tools: SDK assistant message field {field} was not null")
    assistant_calls = assistant_history.get("tool_calls") or []
    if len(assistant_calls) != 1:
        raise ValueError("tools: SDK response has no single tool call")
    assistant_call = assistant_calls[0]
    if assistant_call["function"]["name"] != "weather":
        raise ValueError("tools: SDK response named the wrong function")
    assistant_arguments = json.loads(assistant_call["function"]["arguments"])
    if assistant_arguments != {"city": "Oslo"}:
        raise ValueError(f"tools: SDK response arguments {assistant_arguments!r}")
    assistant_record = response_record(
        binary, assistant_response.model_dump(), assistant.content,
        "tool-assistant", require_content=False, expected_session="client-tool-roundtrip",
    )
    return assistant_history, assistant_call, assistant_arguments, assistant_record


def tool_followup_messages(request, assistant_history, assistant_call):
    return request + [assistant_history] + [
        {
            "role": "tool",
            "name": assistant_call["function"]["name"],
            "tool_call_id": assistant_call["id"],
            "content": '{"temperature_c":12}',
        }
    ] + [{
        "role": "user",
        "content": "State the tool result in one sentence without calling another tool.",
    }]


def check_tool_followup(
    client, binary, tools, request, call, initial_record,
    assistant_history, assistant_call, assistant_arguments, assistant_record,
):
    followup = tool_followup_messages(request, assistant_history, assistant_call)
    followup_response = client.chat.completions.create(
        model="leone", messages=followup, tools=tools, max_tokens=32, temperature=0,
        seed=0, extra_body={
            "leone_session": "client-tool-roundtrip",
            "leone_template": "official",
        },
    )
    record = response_record(
        binary, followup_response.model_dump(),
        followup_response.choices[0].message.content, "tool-roundtrip",
        expected_session="client-tool-roundtrip",
    )
    if record["claim"]["session"]["reused_tokens"] == 0:
        raise ValueError("tools: follow-up did not reuse the tool prompt")
    if followup_response.choices[0].finish_reason not in ("stop", "length"):
        raise ValueError("tools: follow-up did not finish as assistant content")
    record["calls"] = [call]
    record["streamed_call"] = True
    record["official_template"] = True
    record["tool_response"] = {
        "call_id": assistant_call["id"], "city": assistant_arguments["city"],
    }
    record["initial_stream"] = initial_record
    record["assistant_message"] = assistant_history
    record["assistant_response"] = assistant_record
    followup_message = followup_response.choices[0].message.model_dump()
    if followup_message.get("tool_calls"):
        raise ValueError("tools: follow-up called a tool after the tool result")
    record["followup_message"] = followup_message
    return record


def check_tools(client, binary):
    tools = tool_definitions()
    request = [{"role": "user", "content": "Call weather for Oslo."}]
    call, initial_record = check_streamed_tool(client, binary, tools, request)
    assistant = check_assistant_tool(client, binary, tools, request)
    return check_tool_followup(
        client, binary, tools, request, call, initial_record, *assistant
    )


def check_strict_tool_request(client):
    tools = [{
        "type": "function",
        "function": {"name": "weather", "strict": True, "parameters": {"type": "object"}},
    }]
    try:
        client.chat.completions.create(
            model="leone", messages=[{"role": "user", "content": "Call weather."}],
            tools=tools, tool_choice="required", max_tokens=1,
            extra_body={"leone_session": "client-strict-rejected"},
        )
    except BadRequestError as error:
        if "strict=true is unsupported" not in str(error).lower():
            raise ValueError(f"strict-tool: unexpected error {error}") from error
    else:
        raise ValueError("strict-tool: strict tool decoding was accepted")
    if not client.models.list().data:
        raise ValueError("strict-tool: server stopped after request validation")
    return {"name": "strict-tool", "http_status": 400, "server_healthy": True}


def check_truncated_tool_request(client):
    tools = [{
        "type": "function",
        "function": {"name": "weather", "parameters": {"type": "object"}},
    }]
    try:
        client.chat.completions.create(
            model="leone", messages=[{"role": "user", "content": "Call weather."}],
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "weather"}},
            max_tokens=1, temperature=0,
            extra_body={"leone_session": "client-tool-truncated", "leone_template": "official"},
        )
    except BadRequestError as error:
        if "tool call" not in str(error).lower():
            raise ValueError(f"tool-truncation: unexpected error {error}") from error
    else:
        raise ValueError("tool-truncation: incomplete named tool call was accepted")
    if not client.models.list().data:
        raise ValueError("tool-truncation: server stopped after request-local failure")
    return {"name": "tool-truncation", "http_status": 400, "server_healthy": True}


def check_disconnect(client):
    observed = False
    with client.chat.completions.create(
        model="leone", messages=[{"role": "user", "content": "Count upward from one."}],
        max_tokens=512, temperature=0, stream=True,
        extra_body={"leone_session": "client-cancelled"},
    ) as stream:
        for chunk in stream:
            if any(choice.delta.content for choice in chunk.choices):
                observed = True
                break
    if not observed:
        raise ValueError("the disconnect probe received no content")
    return {"name": "disconnect-after-content", "content_observed": observed}


def check_context_growth(client, binary, base):
    first, text = complete(client, binary, base, "client-context-growth")
    messages = base + [{"role": "assistant", "content": text}, {
        "role": "user", "content": "A cache owns separate state for every request. " * 100 + "Summarize in one sentence."
    }]
    grown, _ = complete(client, binary, messages, "client-context-growth")
    if first["claim"]["session"]["reuse_class"] != "cold":
        raise ValueError("context growth: initial request was not cold")
    grown_session = grown["claim"]["session"]
    if grown_session["reuse_class"] != "append-only" or grown_session["reused_tokens"] <= 0:
        raise ValueError("context growth: grown request did not reuse the initial session")
    if grown["usage"]["prompt_tokens"] <= 512:
        raise ValueError("context growth probe did not cross the initial capacity bucket")
    return {"name": "context-growth", "initial": first, "grown": grown}


def check_exact_answer(client, binary):
    prompt = "Reply with the single character 4 and no other text."
    response = client.chat.completions.create(
        model="leone", messages=[{"role": "user", "content": prompt}],
        max_tokens=4, temperature=0, seed=0,
        extra_body={"leone_session": "client-exact-answer"},
    )
    content = response.choices[0].message.content or ""
    if content != "4":
        raise ValueError("client exact-answer task returned an unexpected answer")
    record = response_record(
        binary, response.model_dump(), content, "client-exact-answer",
        expected_session="client-exact-answer",
    )
    record.update({
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "expected_answer_sha256": hashlib.sha256(b"4").hexdigest(),
        "normalized_answer_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "exact_match": True,
    })
    return record


def reset_before_content(base_url, sequence):
    parsed = urllib.parse.urlsplit(base_url)
    body = json.dumps({"model": "leone", "messages": [{"role": "user", "content":
        "A cancelled prompt must release all of its state. " * 100}],
        "max_tokens": 64, "temperature": 0, "stream": True,
        "leone_session": f"client-reset-{sequence}"}).encode()
    request = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {parsed.hostname}\r\n"
               f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
    with socket.create_connection((parsed.hostname, parsed.port), timeout=10) as connection:
        connection.sendall(request)
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))


def check_reset_isolation(arguments, base):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(stream_task, arguments.base_url, arguments.binary, base, "client-reset-survivor")
        for sequence in range(6):
            time.sleep(0.015)
            reset_before_content(arguments.base_url, sequence)
        survivor = future.result()
    return {"name": "reset-isolation", "reset_attempts": 6, "survivor": survivor,
            "limit": "TCP reset timing does not identify the exact scheduler phase; unit tests cover cancelled prefill output."}


def run_workflow(arguments):
    base = [{"role": "user", "content": (
        "A local inference server keeps each request in a separate session. "
        "Name one reason that cancellation must release request state."
    )}]
    records = []
    with make_client(arguments.base_url) as client:
        models = client.models.list()
        if not models.data:
            raise ValueError("the server returned an empty model list")
        parent, parent_text = complete(client, arguments.binary, base, "client-parent")
        records.append(parent)
        continuation = base + [{"role": "assistant", "content": parent_text}]
        questions = ["Explain the memory consequence.", "Explain the scheduling consequence."]
        branches = [continuation + [{"role": "user", "content": question}]
                    for question in questions]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(
                stream_task, arguments.base_url, arguments.binary, messages,
                f"client-branch-{index}", "client-parent",
            ) for index, messages in enumerate(branches)]
            records.extend(future.result() for future in futures)
        records.append(check_context_growth(client, arguments.binary, base))
        records.append(check_exact_answer(client, arguments.binary))
        records.append(check_reset_isolation(arguments, base))
        records.append(check_stop(client, arguments.binary))
        records.append(check_tools(client, arguments.binary))
        records.append(check_strict_tool_request(client))
        records.append(check_truncated_tool_request(client))
        records.append(check_disconnect(client))
        recovery, _ = complete(client, arguments.binary, base, "client-recovery")
        records.append(recovery)
    return {"messages": base, "branch_questions": questions, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda", "metal"))
    parser.add_argument("--kv", choices=("q8", "f16", "f32"))
    parser.add_argument("--batch-size", type=int)
    arguments = parser.parse_args()
    if arguments.batch_size is not None and not 1 <= arguments.batch_size <= 8:
        parser.error("--batch-size must be from 1 to 8")
    if arguments.output.exists():
        parser.error("the output already exists; select a new receipt path")
    build = json.loads(subprocess.check_output(
        [str(arguments.binary), "--build-info"], text=True,
    ))
    workflow = run_workflow(arguments)
    report = {
        "schema_version": "leone.openai-client-check.v2",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "client": {
            "name": "openai-python", "version": importlib.metadata.version("openai"),
            "dependencies": dict(sorted(
                (package.metadata["Name"], package.version)
                for package in importlib.metadata.distributions()
            )),
        },
        "binary_sha256": digest(arguments.binary),
        "model_sha256": digest(arguments.model) if arguments.model else None,
        "script_sha256": digest(__file__),
        "receipt_verifier": {
            "command": "receipt verify-response",
            "binary_sha256": digest(arguments.binary),
        },
        "runtime": {
            "backend": arguments.backend,
            "kv": arguments.kv,
            "batch_size": arguments.batch_size,
        },
        "build": build,
        "workflow": workflow,
        "passed": True,
        "limits": [
            "The check covers streaming, concurrent forks, disconnect recovery, and a typed error.",
            "Signature verification checks the signed claim, not model correctness.",
            "The check does not observe server-side cancellation cleanup.",
            "The check does not measure performance.",
            "The exact-answer task checks one deterministic response, not general model quality.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
