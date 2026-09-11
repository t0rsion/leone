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


def response_record(binary, response, content, name):
    receipt = response.get("leone_receipt")
    if not receipt:
        raise ValueError(f"{name}: the response has no receipt")
    verify_receipt(binary, receipt)
    usage = response.get("usage")
    if not usage or usage["completion_tokens"] != receipt["claim"]["generated_tokens"]:
        raise ValueError(f"{name}: usage differs from the signed token count")
    if receipt["claim"]["cancelled"]:
        raise ValueError(f"{name}: a completed workflow request was cancelled")
    if not content:
        raise ValueError(f"{name}: the response has no text")
    return {
        "name": name,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "usage": usage,
        "claim": receipt["claim"],
        "signature_verified": True,
    }


def complete(client, binary, messages, name, **extra):
    response = client.chat.completions.create(
        model="leone", messages=messages, max_tokens=32, temperature=0,
        seed=0, extra_body={"leone_session": name, **extra},
    )
    return response_record(
        binary, response.model_dump(), response.choices[0].message.content, name,
    ), response.choices[0].message.content


def stream_task(base_url, binary, messages, name, parent=None):
    extra = {"leone_session": name}
    if parent:
        extra["leone_fork_session"] = parent
    terminal = None
    text = []
    with make_client(base_url) as client:
        with client.chat.completions.create(
            model="leone", messages=messages, max_tokens=64, temperature=0,
            seed=0, stream=True, extra_body=extra,
        ) as stream:
            for chunk in stream:
                for choice in chunk.choices:
                    if choice.delta.content:
                        text.append(choice.delta.content)
                    if choice.finish_reason:
                        terminal = chunk.model_dump()
    if terminal is None:
        raise ValueError(f"{name}: the stream has no terminal event")
    return response_record(binary, terminal, "".join(text), name)


def check_rejection(client):
    try:
        client.chat.completions.create(
            model="leone", messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1, stop=["END"],
        )
    except BadRequestError as error:
        if error.status_code != 400 or not isinstance(error.body, dict):
            raise ValueError("unsupported stop did not return a typed 400") from error
        return {"name": "unsupported-stop", "http_status": error.status_code,
                "error": error.body}
    raise ValueError("the server accepted unsupported stop sequences")


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
    if grown["usage"]["prompt_tokens"] <= 512:
        raise ValueError("context growth probe did not cross the initial capacity bucket")
    return {"name": "context-growth", "initial": first, "grown": grown}


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
        records.append(check_reset_isolation(arguments, base))
        records.append(check_rejection(client))
        records.append(check_disconnect(client))
        recovery, _ = complete(client, arguments.binary, base, "client-recovery")
        records.append(recovery)
    return {"messages": base, "branch_questions": questions, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        parser.error("the output already exists; select a new receipt path")
    build = json.loads(subprocess.check_output(
        [str(arguments.binary), "--build-info"], text=True,
    ))
    workflow = run_workflow(arguments)
    report = {
        "schema_version": "leone.openai-client-check.v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "client": {
            "name": "openai-python", "version": importlib.metadata.version("openai"),
            "dependencies": dict(sorted(
                (package.metadata["Name"], package.version)
                for package in importlib.metadata.distributions()
            )),
        },
        "binary_sha256": digest(arguments.binary),
        "script_sha256": digest(__file__),
        "build": build,
        "workflow": workflow,
        "passed": True,
        "limits": [
            "The check covers streaming, concurrent forks, disconnect recovery, and a typed error.",
            "Signature verification checks the signed claim, not model correctness.",
            "The check does not observe server-side cancellation cleanup.",
            "The check does not measure performance or establish tool-call accuracy.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(arguments.output)


if __name__ == "__main__":
    main()
