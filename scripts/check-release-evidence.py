#!/usr/bin/env python3
"""Validate linked release records in offline or local-input mode."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from release_evidence_manifest import load as load_evidence_manifest
from release_evidence_manifest import validate as validate_evidence_manifest

ROOT = Path(__file__).resolve().parents[1]
V03_MANIFEST = Path("packaging/release-evidence.v0.3.json")
EXACT_PROMPT_SHA256 = "9c3720897547bc33978392e5c409419ed1500768f66595f23ca5e11e7221259c"
EXACT_ANSWER_SHA256 = "4b227777d4dd1fc61c6f884f48641d02b4d121d3fd328cb08b5531fcacdabf8a"


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    with (ROOT / path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def digest_file(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def value_digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(commit, offline, source_manifest=Path("receipts/source-inputs.json")):
    source_path = _root_regular_file(source_manifest, "source input manifest")
    source_relative = source_path.relative_to(ROOT)
    if not offline:
        subprocess.run(
            [
                sys.executable,
                "scripts/source_inputs.py",
                "check",
                commit,
                "--manifest",
                str(source_relative),
            ],
            cwd=ROOT,
            check=True,
        )
        return
    manifest = json.loads(source_path.read_text(encoding="utf-8"))
    require(manifest["source_commit"] == commit, "source manifest commit differs")
    _check_source_records(manifest)


def _check_source_records(manifest):
    records = {}
    _check_source_group(manifest.get("files"), "files", records, required=True)
    for field in ("workload_files", "evidence_files"):
        _check_source_group(manifest.get(field), field, records, required=False)


def _check_source_group(values, field, records, required):
    if values is None:
        require(not required, "source manifest has no files")
        return
    require(isinstance(values, dict), f"source manifest {field} are malformed")
    if required:
        require(values, "source manifest has no files")
    for name, value in values.items():
        _check_source_record(name, value, records)


def _check_source_record(name, value, records):
    name_path = Path(name) if isinstance(name, str) else Path()
    require(
        isinstance(name, str)
        and not name_path.is_absolute()
        and ".." not in name_path.parts,
        "source manifest contains an unsafe path",
    )
    require(
        name not in records
        and isinstance(value, dict)
        and isinstance(value.get("executable"), bool)
        and valid_digest(value.get("sha256")),
        "source manifest contains an invalid or repeated file record",
    )
    candidate = ROOT / name_path
    if not candidate.exists() and not candidate.is_symlink():
        records[name] = value
        return
    path = _root_regular_file(name_path, "source input")
    require(
        bool(path.stat().st_mode & 0o111) == value["executable"],
        "source input executable mode differs",
    )
    require(digest_file(path) == value["sha256"], "source input hash differs")
    records[name] = value


def check_build(
    build, commit, offline, source_manifest=Path("receipts/source-inputs.json")
):
    require(build["schema_version"] == "leone.build-info.v1", "unknown build schema")
    require(build["source_commit"] == commit, "build and record commits differ")
    require(build["source_tree_dirty"] is False, "evidence uses a dirty source build")
    require(build["profile"] == "release", "evidence uses a non-release build")
    check_source(commit, offline, source_manifest)


def check_adapter(manifest):
    adapter = manifest["adapter"]
    require(adapter["path"] == "research/oracle/llama_logits.cpp", "unexpected oracle adapter")
    require(digest(adapter["path"]) == adapter["sha256"], "oracle adapter changed")


def check_kld(quality, linked):
    require(quality["kld"]["definition"] == linked["metrics"]["kld"]["definition"], "KLD definition differs")
    for key, value in quality["kld"].items():
        if key == "definition":
            continue
        require(math.isfinite(value) and value >= 0, "invalid quality metric")
        require(linked["metrics"]["kld"][key] == value, "KLD summary differs")


def check_quality(path, oracle_storage, offline):
    record = load(path)
    require(record["schema_version"] == "leone.quality-comparison.v1", "unknown quality schema")
    check_build(record["build_info"], record["source_commit"], offline)
    require(digest(record["corpus"]["path"]) == record["corpus"]["sha256"], "quality corpus changed")
    require(record["input"]["scored_positions"] >= 2399, "quality sample count below declared release corpus")
    oracle = record["executions"]["oracle"]["manifest"]
    require(oracle["model"]["storage_type"] == oracle_storage, "incorrect oracle storage")
    check_adapter(oracle)
    pin = (ROOT / "external/PINNED").read_text().strip()
    require(oracle["engine"]["git_commit"] == pin, "oracle differs from pinned llama.cpp")
    for engine in ("leone_q4", "llama_q4"):
        execution = record["executions"][engine]
        quality = execution["quality"]
        linked = load(quality["receipt"])
        require(digest(quality["receipt"]) == quality["receipt_sha256"], "linked quality record changed")
        require(linked["corpus"]["n_tokens_scored"] == record["input"]["scored_positions"], "scored row mismatch")
        require(linked["oracle"]["artifact_sha256"] == oracle["logits"]["sha256"], "oracle logits differ")
        require(linked["subject"]["model_artifact"]["sha256"] == record["model_family"]["subject"]["sha256"], "subject model differs")
        require(linked["metrics"]["top1_agreement"] == quality["top1_agreement"], "top-1 summary differs")
        require(0 <= quality["top1_agreement"] <= 1, "invalid top-1 agreement")
        check_kld(quality, linked)
        if engine == "leone_q4":
            require(execution["backend"] == "cuda", "Leone quality is not CUDA evaluation")
            require(execution["prefill_path"] == "chunked", "Leone quality does not use chunked prefill")
        else:
            check_adapter(execution["manifest"])
            require(execution["device"] == "cuda", "llama.cpp quality is not CUDA evaluation")
            require(execution["manifest"]["input"] == oracle["input"], "quality token windows differ")
    return record["model_family"]["subject"]["sha256"]


def verify_response(verifier, receipt):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "response.json"
        path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run(
            [str(verifier), "receipt", "verify-response", str(path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )


def walk_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def is_receipt(value):
    return isinstance(value, dict) and {
        "claim", "public_key_ed25519", "signature_ed25519"
    }.issubset(value)


SIGNED_RESPONSE_PATHS = (
    (("client-parent",), "client-parent"),
    (("client-branch-0",), "client-branch-0"),
    (("client-branch-1",), "client-branch-1"),
    (("context-growth", "initial"), "client-context-growth"),
    (("context-growth", "grown"), "client-context-growth"),
    (("client-exact-answer",), "client-exact-answer"),
    (("reset-isolation", "survivor"), "client-reset-survivor"),
    (("stop",), "stop"),
    (("tool-roundtrip", "initial_stream"), "tool-stream"),
    (("tool-roundtrip", "assistant_response"), "tool-assistant"),
    (("tool-roundtrip",), "tool-roundtrip"),
    (("client-recovery",), "client-recovery"),
)
RESPONSE_MARKERS = {
    "claim", "signature_verified", "leone_receipt", "receipt_sha256", "expected_session"
}
CLIENT_RECORD_NAMES = {
    "client-parent", "client-branch-0", "client-branch-1", "context-growth",
    "client-exact-answer", "reset-isolation", "stop", "tool-roundtrip",
    "strict-tool", "tool-truncation", "disconnect-after-content", "client-recovery",
}


def check_client_provenance(record, offline, binary_path, model_path, expected_model_sha256):
    require(record["schema_version"] == "leone.openai-client-check.v2", "unknown client schema")
    require(record["passed"] is True, "client workflow failed")
    check_build(record["build"], record["build"]["source_commit"], offline)
    require(record["script_sha256"] == digest("scripts/check-openai-client.py"), "client script changed")
    require(valid_digest(record.get("binary_sha256")), "client binary hash is missing")
    require(valid_digest(record.get("model_sha256")), "client model hash is missing")
    require(record["model_sha256"] == expected_model_sha256, "client model differs from pinned quality receipt")
    verifier = record.get("receipt_verifier")
    require(
        isinstance(verifier, dict)
        and verifier.get("command") == "receipt verify-response"
        and verifier.get("binary_sha256") == record["binary_sha256"],
        "client receipt verifier provenance is missing",
    )
    check_client_artifacts(record, binary_path, model_path)
    check_client_runtime(record)


def check_client_artifacts(record, binary_path, model_path):
    if binary_path is not None:
        require(digest(binary_path) == record["binary_sha256"], "client binary differs from evidence")
    if model_path is not None:
        require(digest(model_path) == record["model_sha256"], "client model differs from evidence")


def check_client_runtime(record):
    runtime = record.get("runtime")
    require(isinstance(runtime, dict), "client runtime provenance is missing")
    require(runtime.get("backend") in {"cpu", "cuda", "metal"}, "client backend is missing")
    require(runtime.get("kv") in {"q8", "f16", "f32"}, "client KV type is missing")
    require(type(runtime.get("batch_size")) is int and 1 <= runtime["batch_size"] <= 8, "client batch size is invalid")


def client_response_records(record):
    raw_records = record["workflow"]["records"]
    require(isinstance(raw_records, list), "client workflow records are missing")
    names = [item.get("name") for item in raw_records if isinstance(item, dict)]
    require(len(names) == len(raw_records) and len(names) == len(set(names)), "client workflow names are not unique")
    require(set(names) == CLIENT_RECORD_NAMES, "client workflow records differ from the release contract")
    records = {item["name"]: item for item in raw_records}
    response_nodes = []
    for path, name in SIGNED_RESPONSE_PATHS:
        node = records[path[0]]
        for key in path[1:]:
            require(isinstance(node, dict) and key in node, f"client response path is missing: {'.'.join(path)}")
            node = node[key]
        require(isinstance(node, dict) and node.get("name") == name, f"client response path is invalid: {'.'.join(path)}")
        response_nodes.append(node)
    return records, response_nodes


def validate_client_response(node):
    require("leone_receipt" in node and "receipt_sha256" in node, "client response receipt is missing")
    receipt = node.get("leone_receipt")
    require(isinstance(receipt, dict), "client response receipt is not an object")
    require(node.get("receipt_sha256") == value_digest(receipt), "client response receipt hash differs")
    require(node.get("claim") == receipt.get("claim"), "client claim is not copied from its receipt")
    require(node.get("signature_verified") is True, "client response signature failed")
    claim = receipt.get("claim")
    require(isinstance(claim, dict), "client response claim is missing")
    expected_session = node.get("expected_session")
    require(isinstance(expected_session, str), "client response session expectation is missing")
    require(claim.get("session", {}).get("session_id") == expected_session, "client response session differs")
    usage = node.get("usage")
    require(isinstance(usage, dict), "client response usage is missing")
    prompt_tokens = claim.get("prompt_tokens")
    completion_tokens = claim.get("generated_tokens")
    require(type(prompt_tokens) is int and prompt_tokens >= 0, "client response prompt count is invalid")
    require(type(completion_tokens) is int and completion_tokens >= 0, "client response completion count is invalid")
    require(
        all(
            usage.get(field) == expected
            for field, expected in {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }.items()
        ),
        "client response usage differs from its receipt",
    )
    return receipt


def reject_unsigned_client_nodes(workflow, response_ids):
    for node in walk_dicts(workflow):
        if is_receipt(node):
            continue
        if RESPONSE_MARKERS.intersection(node) and id(node) not in response_ids:
            raise ValueError("client workflow contains an unsigned response record")


def validate_client_responses(record, response_nodes, binary_path, receipt_validator=None):
    response_ids = {id(node) for node in response_nodes}
    reject_unsigned_client_nodes(record["workflow"], response_ids)
    receipt_nodes = []
    for node in response_nodes:
        receipt = validate_client_response(node)
        receipt_nodes.append((node, receipt))
    require(receipt_nodes, "client workflow has no signed response receipts")
    require(all(valid_digest(receipt.get("claim", {}).get("model_sha256")) for _, receipt in receipt_nodes), "client receipt claim is invalid")
    require(all(receipt["claim"]["model_sha256"] == record["model_sha256"] for _, receipt in receipt_nodes), "client model hash differs")
    verifier = receipt_validator or binary_path
    if verifier is not None:
        for _, receipt in receipt_nodes:
            verify_response(verifier, receipt)
    return receipt_nodes


def check_client_sessions(records):
    for name in ("client-parent", "client-branch-0", "client-branch-1", "client-recovery"):
        require(records[name]["signature_verified"] is True, "client response signature failed")
        require(records[name]["usage"]["completion_tokens"] > 0, "client response is empty")
    for name in ("client-branch-0", "client-branch-1"):
        session = records[name]["claim"]["session"]
        require(session["reuse_class"] == "device-fork", "client fork did not preserve device-fork reuse")
        require(type(session["reused_tokens"]) is int and session["reused_tokens"] > 0, "client fork reused no parent tokens")
    require(records["stop"]["signature_verified"] is True, "matched stop signature failed")
    require(records["stop"]["claim"]["finish_reason"] == "stop", "matched stop did not finish with stop")
    context_initial = records["context-growth"]["initial"]["claim"]["session"]
    context_grown = records["context-growth"]["grown"]["claim"]["session"]
    require(context_initial["reuse_class"] == "cold", "context growth did not start cold")
    require(context_grown["reuse_class"] == "append-only", "context growth did not append to the session")
    require(type(context_grown["reused_tokens"]) is int and context_grown["reused_tokens"] > 0, "context growth reused no initial tokens")
    require(records["context-growth"]["grown"]["usage"]["prompt_tokens"] > 512, "context growth not exercised")
    require(
        records["context-growth"]["initial"]["claim"]["receipt_id"]
        != records["context-growth"]["grown"]["claim"]["receipt_id"],
        "context growth reused one receipt",
    )


def check_client_tools(records):
    require(records["tool-roundtrip"]["streamed_call"] is True, "tool call was not streamed")
    require(records["tool-roundtrip"]["official_template"] is True, "official tool template was not used")
    require(records["tool-roundtrip"]["claim"]["session"]["reused_tokens"] > 0, "tool follow-up reused no prompt tokens")
    require(records["tool-roundtrip"]["assistant_message"]["role"] == "assistant", "SDK assistant message missing")
    require(records["tool-roundtrip"]["assistant_response"]["signature_verified"] is True, "tool assistant signature failed")
    require(records["tool-truncation"]["http_status"] == 400, "truncated tool output was accepted")
    require(records["tool-truncation"]["server_healthy"] is True, "tool failure stopped the server")
    require(records["strict-tool"]["http_status"] == 400, "strict tool decoding was accepted")
    require(records["strict-tool"]["server_healthy"] is True, "strict tool rejection stopped the server")


def check_client_exact_answer(record):
    exact = record["client-exact-answer"]
    require(exact["signature_verified"] is True, "exact-answer response signature failed")
    require(exact["exact_match"] is True, "exact-answer client task failed")
    require(exact["prompt_sha256"] == EXACT_PROMPT_SHA256, "exact-answer prompt differs")
    require(exact["expected_answer_sha256"] == EXACT_ANSWER_SHA256, "exact-answer target differs")
    require(exact["normalized_answer_sha256"] == EXACT_ANSWER_SHA256, "exact-answer response differs")
    require(exact["content_sha256"] == EXACT_ANSWER_SHA256, "exact-answer content differs")


def check_client(
    offline,
    binary_path=None,
    model_path=None,
    expected_model_sha256=None,
    receipt_validator=None,
):
    record = load("receipts/openai-client-check.json")
    check_client_provenance(record, offline, binary_path, model_path, expected_model_sha256)
    records, response_nodes = client_response_records(record)
    validate_client_responses(
        record,
        response_nodes,
        None if offline else binary_path,
        receipt_validator,
    )
    check_client_sessions(records)
    check_client_tools(records)
    check_client_exact_answer(records)
    require(records["reset-isolation"]["survivor"]["claim"]["cancelled"] is False, "TCP reset cancelled its neighbor")
    require(records["disconnect-after-content"]["content_observed"] is True, "disconnect lacked content")


def check_client_v04(
    path,
    source_manifest,
    platform_name,
    target,
    backend,
    binary=None,
    binary_sha256=None,
    model_family=None,
    model_sha256=None,
    receipt_validator=None,
):
    """Validate one OpenAI client receipt and every retained signed response."""
    record = json.loads(path.read_text(encoding="utf-8"))
    _check_client_v2_identity(record, platform_name, target, backend)
    _check_client_v2_build(record, target, source_manifest)
    _check_client_v2_artifacts(record, binary, binary_sha256, model_family, model_sha256)
    _check_client_v2_workflow(record, receipt_validator)


def _check_client_v2_identity(record, platform_name, target, backend):
    require(record.get("schema_version") == "leone.openai-client-check.v2", "unknown client schema")
    require(record.get("passed") is True, "client workflow failed")
    for key, expected in (("platform", platform_name), ("target", target)):
        if key in record:
            require(record[key] == expected, f"client {key} differs from release identity")
    runtime = record.get("runtime")
    require(isinstance(runtime, dict), "client runtime provenance is missing")
    require(runtime.get("backend") == backend, "client backend differs from release identity")
    require(runtime.get("kv") in {"q8", "f16", "f32"}, "client KV type is missing")
    batch_size = runtime.get("batch_size")
    require(type(batch_size) is int and 1 <= batch_size <= 8, "client batch size is invalid")


def _check_client_v2_build(record, target, source_manifest):
    build = record.get("build")
    require(isinstance(build, dict), "client build provenance is missing")
    commit = build.get("source_commit")
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit.lower()),
        "client build source commit is invalid",
    )
    require(build.get("target") == target, "client build target differs from release identity")
    check_build(build, commit, True, source_manifest)
    script = _root_regular_file(Path("scripts/check-openai-client.py"), "client checker")
    require(record.get("script_sha256") == digest_file(script), "client script changed")


def _check_client_v2_artifacts(record, binary, binary_sha256, model_family, model_sha256):
    require(valid_digest(binary_sha256), "client binary identity is incomplete")
    require(isinstance(model_family, str) and model_family, "client model identity is incomplete")
    require(valid_digest(model_sha256), "client model identity is incomplete")
    require(valid_digest(record.get("binary_sha256")), "client binary hash is missing")
    require(valid_digest(record.get("model_sha256")), "client model hash is missing")
    require(record["binary_sha256"] == binary_sha256, "client receipt binary differs")
    require(record["model_sha256"] == model_sha256, "client receipt model differs")
    if binary is not None:
        packaged_binary = _root_regular_file(binary, "client binary")
        require(digest_file(packaged_binary) == binary_sha256, "client binary changed")
    verifier = record.get("receipt_verifier")
    require(
        isinstance(verifier, dict)
        and verifier.get("command") == "receipt verify-response"
        and verifier.get("binary_sha256") == record["binary_sha256"],
        "client receipt verifier provenance is missing",
    )


def _check_client_v2_workflow(record, receipt_validator):
    records = _client_workflow_records(record)
    _check_client_v2_names(records)
    response_nodes = _client_v2_response_nodes(records)
    _check_client_v2_receipts(record, response_nodes, receipt_validator)
    _check_client_v2_semantics(records)


def _check_client_v2_names(records):
    expected = {
        "client-parent", "client-branch-0", "client-branch-1", "context-growth",
        "client-exact-answer", "reset-isolation", "stop", "tool-roundtrip",
        "strict-tool", "tool-truncation", "disconnect-after-content", "client-recovery",
    }
    require(set(records) == expected, "client workflow records differ from the release contract")


def _client_v2_response_nodes(records):
    paths = (
        (("client-parent",), "client-parent"),
        (("client-branch-0",), "client-branch-0"),
        (("client-branch-1",), "client-branch-1"),
        (("context-growth", "initial"), "client-context-growth"),
        (("context-growth", "grown"), "client-context-growth"),
        (("client-exact-answer",), "client-exact-answer"),
        (("reset-isolation", "survivor"), "client-reset-survivor"),
        (("stop",), "stop"),
        (("tool-roundtrip", "initial_stream"), "tool-stream"),
        (("tool-roundtrip", "assistant_response"), "tool-assistant"),
        (("tool-roundtrip",), "tool-roundtrip"),
        (("client-recovery",), "client-recovery"),
    )
    nodes = []
    for path, name in paths:
        node = records[path[0]]
        for key in path[1:]:
            require(isinstance(node, dict) and key in node, f"client response path is missing: {'.'.join(path)}")
            node = node[key]
        require(isinstance(node, dict) and node.get("name") == name, f"client response path is invalid: {'.'.join(path)}")
        nodes.append(node)
    return nodes


def _check_client_v2_receipts(record, response_nodes, receipt_validator):
    require(receipt_validator is not None, "offline client validation requires a receipt validator")
    response_ids = {id(node) for node in response_nodes}
    for node in walk_dicts(record["workflow"]):
        if is_receipt(node):
            continue
        markers = {"claim", "signature_verified", "leone_receipt", "receipt_sha256", "expected_session"}
        require(not markers.intersection(node) or id(node) in response_ids, "client workflow contains an unsigned response record")
    receipts = []
    for node in response_nodes:
        receipt = _check_client_v2_response(node)
        receipts.append(receipt)
        verify_response(receipt_validator, receipt)
    expected_model = record["model_sha256"]
    require(all(receipt["claim"].get("model_sha256") == expected_model for receipt in receipts), "client model hash differs")


def _check_client_v2_response(node):
    require("leone_receipt" in node and "receipt_sha256" in node, "client response receipt is missing")
    receipt = node.get("leone_receipt")
    require(isinstance(receipt, dict), "client response receipt is not an object")
    require(node.get("receipt_sha256") == value_digest(receipt), "client response receipt hash differs")
    require(node.get("claim") == receipt.get("claim"), "client claim is not copied from its receipt")
    require(node.get("signature_verified") is True, "client response signature failed")
    claim = receipt.get("claim")
    require(isinstance(claim, dict), "client response claim is missing")
    expected_session = node.get("expected_session")
    require(isinstance(expected_session, str), "client response session expectation is missing")
    require(claim.get("session", {}).get("session_id") == expected_session, "client response session differs")
    usage = node.get("usage")
    require(isinstance(usage, dict), "client response usage is missing")
    prompt = claim.get("prompt_tokens")
    generated = claim.get("generated_tokens")
    require(type(prompt) is int and prompt >= 0, "client response prompt count is invalid")
    require(type(generated) is int and generated >= 0, "client response completion count is invalid")
    expected_usage = {"prompt_tokens": prompt, "completion_tokens": generated, "total_tokens": prompt + generated}
    require(all(usage.get(key) == value for key, value in expected_usage.items()), "client response usage differs from its receipt")
    require(valid_digest(claim.get("model_sha256")), "client response model hash is invalid")
    return receipt


def _check_client_v2_semantics(records):
    for name in ("client-parent", "client-branch-0", "client-branch-1", "client-recovery"):
        require(records[name]["usage"]["completion_tokens"] > 0, "client response is empty")
    for name in ("client-branch-0", "client-branch-1"):
        session = records[name]["claim"]["session"]
        require(session["reuse_class"] == "device-fork", "client fork did not preserve device-fork reuse")
        require(type(session["reused_tokens"]) is int and session["reused_tokens"] > 0, "client fork reused no parent tokens")
    require(records["stop"]["claim"]["finish_reason"] == "stop", "matched stop did not finish with stop")
    tool = records["tool-roundtrip"]
    require(tool["streamed_call"] is True and tool["official_template"] is True, "tool workflow was not exercised")
    require(tool["claim"]["session"]["reused_tokens"] > 0, "tool follow-up reused no prompt tokens")
    require(tool["assistant_message"]["role"] == "assistant", "SDK assistant message missing")
    require(tool["assistant_response"]["signature_verified"] is True, "tool assistant signature failed")
    require(records["tool-truncation"]["http_status"] == 400, "truncated tool output was accepted")
    require(records["tool-truncation"]["server_healthy"] is True, "tool failure stopped the server")
    require(records["strict-tool"]["http_status"] == 400, "strict tool decoding was accepted")
    require(records["strict-tool"]["server_healthy"] is True, "strict tool rejection stopped the server")
    _check_client_v2_context(records)
    _check_client_v2_exact_answer(records)
    reset = records.get("reset-isolation")
    require(reset["survivor"]["claim"]["cancelled"] is False, "TCP reset cancelled its neighbor")
    disconnect = records.get("disconnect-after-content")
    require(disconnect["content_observed"] is True, "disconnect lacked content")


def _check_client_v2_context(records):
    initial_claim = records["context-growth"]["initial"]["claim"]
    grown_claim = records["context-growth"]["grown"]["claim"]
    initial = initial_claim["session"]
    grown = grown_claim["session"]
    require(initial["reuse_class"] == "cold", "context growth did not start cold")
    require(grown["reuse_class"] == "append-only", "context growth did not append to the session")
    require(type(grown["reused_tokens"]) is int and grown["reused_tokens"] > 0, "context growth reused no initial tokens")
    require(records["context-growth"]["grown"]["usage"]["prompt_tokens"] > 512, "context growth not exercised")
    require(initial_claim["receipt_id"] != grown_claim["receipt_id"], "context growth reused one receipt")


def _check_client_v2_exact_answer(records):
    exact = records["client-exact-answer"]
    require(exact["exact_match"] is True, "exact-answer client task failed")
    require(exact["prompt_sha256"] == EXACT_PROMPT_SHA256, "exact-answer prompt differs")
    require(exact["expected_answer_sha256"] == EXACT_ANSWER_SHA256, "exact-answer target differs")
    require(exact["normalized_answer_sha256"] == EXACT_ANSWER_SHA256, "exact-answer response differs")
    require(exact["content_sha256"] == EXACT_ANSWER_SHA256, "exact-answer content differs")




def _client_workflow_records(record):
    workflow = record.get("workflow")
    records = workflow.get("records") if isinstance(workflow, dict) else None
    require(isinstance(records, list), "client workflow records are missing")
    result = {}
    for item in records:
        require(isinstance(item, dict) and isinstance(item.get("name"), str), "client workflow record is malformed")
        require(item["name"] not in result, "client workflow record is duplicated")
        result[item["name"]] = item
    return result




def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def _manifest_path(explicit):
    if explicit is not None:
        return _root_regular_file(explicit, "release evidence manifest")
    bundled = ROOT / "release-evidence.json"
    if bundled.exists():
        return _root_regular_file(bundled, "release evidence manifest")
    raise ValueError("release evidence manifest is required")


def _root_regular_file(candidate, label):
    candidate = candidate if candidate.is_absolute() else ROOT / candidate
    try:
        relative = candidate.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{label} is outside its root") from error
    if any(part in {".", ".."} for part in relative.parts):
        raise ValueError(f"{label} is outside its root")
    current = ROOT
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink")
    try:
        resolved_root = ROOT.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{label} is outside its root") from error
    if not candidate.is_file():
        raise ValueError(f"{label} is not a regular file")
    return candidate


def _check_exact_v03_manifest(path, manifest):
    """Require the immutable legacy manifest before using the legacy checker."""
    require(manifest.get("release_line") == "v0.3", "unsupported legacy release manifest")
    trusted_path = Path(__file__).resolve().parents[1] / V03_MANIFEST
    trusted = load_evidence_manifest(trusted_path)
    require(manifest == trusted, "v0.3 evidence manifest differs from the trusted manifest")
    validate_evidence_manifest(manifest, require_complete=True)


def _manifest_record(path, expected):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read evidence record {path}: {error}") from error
    require(isinstance(record, dict), f"evidence record is not an object: {path}")
    require(
        record.get("schema_version") == expected["schema_version"],
        f"evidence schema differs for {path}",
    )
    expected_digest = expected.get("sha256")
    if expected_digest is not None:
        require(valid_digest(expected_digest), f"evidence record has an invalid SHA-256: {path}")
        require(digest_file(path) == expected_digest, f"evidence record changed: {path}")


def check_manifest_records(manifest, receipt_validator=None):
    records = validate_evidence_manifest(manifest, require_complete=True)
    for expected in records:
        path = ROOT / expected["path"]
        require(path.is_file(), f"required evidence record is missing: {expected['path']}")
        _manifest_record(path, expected)
    from release_evidence_validators import validate_records

    validate_records(
        ROOT,
        manifest,
        records,
        Path(__file__).resolve().parents[1],
        receipt_validator,
    )


def check_v04_manifest(manifest_path, offline, receipt_validator=None):
    require(offline, "v0.4 evidence supports offline verification only")
    manifest = load_evidence_manifest(manifest_path)
    require(manifest.get("release_line") == "v0.4", "unexpected v0.4 evidence manifest")
    check_manifest_records(manifest, receipt_validator)


def _argument_parser():
    parser = argparse.ArgumentParser(description="Validate packaged release evidence.")
    parser.add_argument(
        "--root",
        type=Path,
        help="data root to validate; checker code stays in this source tree",
    )
    parser.add_argument(
        "--mode",
        choices=("verify", "reproduce"),
        default="verify",
        help="verify bundled records, or check local models and binaries",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="alias for --mode verify",
    )
    parser.add_argument(
        "--manifest",
        "--release-manifest",
        dest="manifest",
        type=Path,
        help="release evidence manifest to validate",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--quality-receipt", type=Path)
    parser.add_argument("--leone-binary", type=Path)
    parser.add_argument("--llama-binary", type=Path)
    parser.add_argument("--validate-client", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--platform")
    parser.add_argument("--target")
    parser.add_argument("--backend")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument("--model-family")
    parser.add_argument("--model-sha256")
    parser.add_argument(
        "--trusted-receipt-validator",
        type=Path,
        help="trusted CPU-only Rust validator for runtime and quality receipts",
    )
    return parser


def _legacy_check(args, offline):
    check_quality("receipts/quality-concurrent-service.json", "BF16", offline)
    llama_model_sha256 = check_quality("receipts/quality-llama-v03.json", "F16", offline)
    check_client(
        offline,
        args.leone_binary,
        args.model,
        llama_model_sha256,
        args.trusted_receipt_validator,
    )
    checker_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(checker_root / "scripts/study-concurrent-service.py"),
        "--root", str(ROOT),
        "--validate-receipt", "receipts/concurrent-service-study.json",
    ]
    if offline:
        command.append("--offline")
    for option, value in (
        ("--model", args.model),
        ("--plan", args.plan),
        ("--quality-receipt", args.quality_receipt),
        ("--leone-binary", args.leone_binary),
        ("--llama-binary", args.llama_binary),
    ):
        if value is not None:
            command.extend((option, str(value)))
    subprocess.run(command, cwd=checker_root, check=True)
    label = "offline release evidence" if offline else "local release evidence"
    print(f"{label} passed")


def main(arguments=None):
    global ROOT
    parser = _argument_parser()
    args = parser.parse_args(arguments)
    if args.offline and args.mode == "reproduce":
        parser.error("--offline cannot be combined with --mode reproduce")
    if args.root is not None:
        ROOT = args.root.resolve()
    offline = args.offline or args.mode == "verify"
    if args.validate_client is not None:
        _validate_client_command(args, offline)
        return
    manifest_path = _manifest_path(args.manifest)
    manifest = load_evidence_manifest(manifest_path)
    release_line = manifest.get("release_line")
    if release_line == "v0.4":
        check_v04_manifest(manifest_path, offline, args.trusted_receipt_validator)
        print("offline release evidence passed")
        return
    if release_line == "v0.3":
        _check_exact_v03_manifest(manifest_path, manifest)
        _legacy_check(args, offline)
        return
    raise ValueError(f"unsupported release evidence line: {release_line}")


def _validate_client_command(args, offline):
    require(offline, "v0.4 client evidence supports offline verification only")
    trusted = args.trusted_receipt_validator
    require(
        trusted is not None and trusted.is_file() and not trusted.is_symlink(),
        "offline client validation requires a trusted receipt validator",
    )
    for parent in trusted.parents:
        require(not parent.is_symlink(), "trusted receipt validator has a symlink parent")
    client_path = _root_regular_file(args.validate_client, "client evidence record")
    source_manifest = args.source_manifest or Path("receipts/source-inputs-v04.json")
    require_client_identity(args)
    check_client_v04(
        client_path,
        source_manifest,
        args.platform,
        args.target,
        args.backend,
        args.binary,
        args.binary_sha256,
        args.model_family,
        args.model_sha256,
        args.trusted_receipt_validator,
    )
    print("offline client evidence passed")


def require_client_identity(args):
    if any(value is None for value in (
        args.platform,
        args.target,
        args.backend,
        args.binary_sha256,
        args.model_family,
        args.model_sha256,
        args.trusted_receipt_validator,
    )):
        raise ValueError("client release identity is required")


if __name__ == "__main__":
    main()
