#!/usr/bin/env python3
"""Validate the linked quality and client records for the local release gate."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    with (ROOT / path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_source(commit):
    subprocess.run(["python3", "scripts/source_inputs.py", "check", commit], cwd=ROOT, check=True)


def check_build(build, commit):
    require(build["schema_version"] == "leone.build-info.v1", "unknown build schema")
    require(build["source_commit"] == commit, "build and record commits differ")
    require(build["source_tree_dirty"] is False, "evidence uses a dirty source build")
    require(build["profile"] == "release", "evidence uses a non-release build")
    check_source(commit)


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


def check_quality(path, oracle_storage):
    record = load(path)
    require(record["schema_version"] == "leone.quality-comparison.v1", "unknown quality schema")
    check_build(record["build_info"], record["source_commit"])
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


def check_client():
    record = load("receipts/openai-client-check.json")
    require(record["schema_version"] == "leone.openai-client-check.v1", "unknown client schema")
    require(record["passed"] is True, "client workflow failed")
    check_build(record["build"], record["build"]["source_commit"])
    require(record["script_sha256"] == digest("scripts/check-openai-client.py"), "client script changed")
    records = {item["name"]: item for item in record["workflow"]["records"]}
    for name in ("client-parent", "client-branch-0", "client-branch-1", "client-recovery"):
        require(records[name]["signature_verified"] is True, "client response signature failed")
        require(records[name]["usage"]["completion_tokens"] > 0, "client response is empty")
    require(records["unsupported-stop"]["http_status"] == 400, "unsupported stop was accepted")
    require(records["context-growth"]["grown"]["usage"]["prompt_tokens"] > 512, "context growth not exercised")
    require(records["reset-isolation"]["survivor"]["claim"]["cancelled"] is False, "TCP reset cancelled its neighbor")
    require(records["disconnect-after-content"]["content_observed"] is True, "disconnect lacked content")


def main():
    check_quality("receipts/quality-concurrent-service.json", "BF16")
    check_quality("receipts/quality-llama-v03.json", "F16")
    check_client()
    subprocess.run([
        "python3", "scripts/study-concurrent-service.py", "--validate-receipt",
        "receipts/concurrent-service-study.json",
    ], cwd=ROOT, check=True)
    print("linked release evidence passed")


if __name__ == "__main__":
    main()
