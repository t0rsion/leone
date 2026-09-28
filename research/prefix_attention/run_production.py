#!/usr/bin/env python3
"""Build and run the opt-in Leone CUDA prefill differential driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common_oracle import PROVENANCE_SCOPE
from production_oracle import canonicalize_case, case_digests
from runner_support import SOURCE_LAYOUT, public_command, publish_json

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ROOT / "research/prefix_attention"
DRIVER_ROOT = RESEARCH_ROOT / "production_driver"
MANIFEST = RESEARCH_ROOT / "gpu_manifest.json"
BINARY = DRIVER_ROOT / "target/release/prefix-attention-production-driver"
SOURCE_PATHS = [
    DRIVER_ROOT / "Cargo.toml",
    DRIVER_ROOT / "Cargo.lock",
    DRIVER_ROOT / "src/main.rs",
    RESEARCH_ROOT / "gpu_manifest.json",
    RESEARCH_ROOT / "ORACLE_CONTRACT.md",
    RESEARCH_ROOT / "common_oracle.py",
    RESEARCH_ROOT / "runner_support.py",
    RESEARCH_ROOT / "production_oracle.py",
    RESEARCH_ROOT / "run_production.py",
    RESEARCH_ROOT / "production_receipt.py",
    RESEARCH_ROOT / "freeze_production_criteria.py",
    RESEARCH_ROOT / "validate_production_receipt.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--criteria", type=Path)
    parser.add_argument("--build", action="store_true", help="kept for command compatibility")
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}


def build(before: dict[str, str]) -> list[str]:
    command = ["cargo", "+1.92", "build", "--release", "--manifest-path", str(DRIVER_ROOT / "Cargo.toml")]
    subprocess.run(command, cwd=ROOT, check=True)
    after = source_hashes()
    if before != after:
        raise SystemExit("production driver source changed while building")
    if not BINARY.exists():
        raise SystemExit("production driver did not produce a binary")
    return command


def provenance(raw_output: Path, build_command: list[str], run_command: list[str], sources: dict[str, str], artifact: str) -> dict:
    return {
        "source_layout": SOURCE_LAYOUT,
        "scope": PROVENANCE_SCOPE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "manifest_sha256": sha256(MANIFEST),
        "source_sha256": sources,
        "artifact_sha256": artifact,
        "build": {
            "command": public_command(
                build_command,
                {str(DRIVER_ROOT / "Cargo.toml"): "$DRIVER_MANIFEST"},
            ),
            "rustc_version": subprocess.check_output(["rustc", "--version"], text=True),
        },
        "host_system": platform.platform(),
        "command": public_command(
            run_command,
            {str(BINARY): "$PRODUCTION_BINARY", str(raw_output): "$RAW_OUTPUT"},
        ),
    }


def device_identity() -> dict[str, str]:
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader,nounits"],
        text=True,
    ).strip().splitlines()[0]
    name, capability = (part.strip() for part in line.split(",", 1))
    return {"name": name, "compute_capability": capability}


def validate_run_options(args: argparse.Namespace) -> None:
    if args.phase == "evaluation" and args.criteria is None:
        raise SystemExit("evaluation requires --criteria")
    if args.criteria is not None and not args.criteria.exists():
        raise SystemExit(f"criteria file does not exist: {args.criteria}")


def run_phase(args: argparse.Namespace, output: Path) -> None:
    before = source_hashes()
    build_command = build(before)
    sources = source_hashes()
    artifact = sha256(BINARY)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.", suffix=".raw", delete=False) as raw_file:
        raw = Path(raw_file.name)
    run_command = [str(BINARY), "--phase", args.phase, "--output", str(raw)]
    try:
        subprocess.run(run_command, cwd=ROOT, check=True)
        if sources != source_hashes() or artifact != sha256(BINARY):
            raise SystemExit("production driver identity changed while running")
        receipt = json.loads(raw.read_text())
        for case in receipt["cases"]:
            canonicalize_case(case)
            expected_input, expected_oracle = case_digests(case["spec"])
            if case["input_digest"] != expected_input:
                raise SystemExit(f"{case['name']}: backend input digest differs from common oracle")
            case["backend_oracle_digest"] = case["oracle_digest"]
            case["oracle_digest"] = expected_oracle
        receipt["device"] = device_identity()
        receipt["provenance"] = provenance(raw, build_command, run_command, sources, artifact)
        if args.criteria is not None:
            receipt["criteria_sha256"] = sha256(args.criteria)
        publish_json(output, receipt)
    finally:
        raw.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    validate_run_options(args)
    output = args.output or RESEARCH_ROOT / "receipts/tmp" / f"prefix_attention_production_{args.phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_phase(args, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
