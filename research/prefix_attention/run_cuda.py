#!/usr/bin/env python3
"""Build and run the standalone CUDA prefix attention receipt generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from common_oracle import PROVENANCE_SCOPE, canonicalize_receipt, case_digests
from runner_support import SOURCE_LAYOUT, public_command, publish_json

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = ROOT / "research/prefix_attention"
MANIFEST = RESEARCH_ROOT / "gpu_manifest.json"
CUDA_BUILDER = RESEARCH_ROOT / "cuda_fixed_reduction/build_cuda.sh"
SOURCE_PATHS = [
    RESEARCH_ROOT / "cuda_fixed_reduction/operator.cuh",
    RESEARCH_ROOT / "cuda_fixed_reduction/operator.cu",
    RESEARCH_ROOT / "cuda_fixed_reduction/main.cu",
    RESEARCH_ROOT / "cuda_fixed_reduction/manifest_cases.h",
    RESEARCH_ROOT / "cuda_fixed_reduction/build_cuda.sh",
    RESEARCH_ROOT / "generate_manifest.py",
    RESEARCH_ROOT / "common_oracle.py",
    RESEARCH_ROOT / "runner_support.py",
    RESEARCH_ROOT / "ORACLE_CONTRACT.md",
    MANIFEST,
    RESEARCH_ROOT / "run_cuda.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--binary", type=Path, default=Path("/tmp/prefix_attention_cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--build", action="store_true", help="kept for command compatibility")
    parser.add_argument("--criteria", type=Path)
    return parser.parse_args()


def build_binary(binary: Path, source_before: dict[str, str]) -> list[str]:
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [str(CUDA_BUILDER), str(binary)]
    subprocess.run(command, check=True)
    source_after = {str(path): sha256(path) for path in SOURCE_PATHS}
    if source_before != source_after:
        raise SystemExit("source changed while building CUDA harness")
    if not binary.exists():
        raise SystemExit("CUDA builder did not produce the requested binary")
    return command


def run_binary(binary: Path, output: Path, phase: str, warmups: int, repetitions: int) -> tuple[Path, list[str]]:
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".raw", delete=False
    ) as raw_file:
        raw_output = Path(raw_file.name)
    command = [
        str(binary), "--phase", phase, "--output", str(raw_output),
        "--warmups", str(warmups), "--repetitions", str(repetitions),
    ]
    try:
        subprocess.run(command, check=True)
        return raw_output, command
    except BaseException:
        raw_output.unlink(missing_ok=True)
        raise


def provenance(
    raw_output: Path,
    binary: Path,
    build_command: list[str],
    run_command: list[str],
    source_hashes: dict[str, str],
    artifact_hash: str,
) -> dict:
    nvcc = os.environ.get("NVCC", "nvcc")
    return {
        "source_layout": SOURCE_LAYOUT,
        "scope": PROVENANCE_SCOPE,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
        "manifest_sha256": sha256(MANIFEST),
        "source_sha256": source_hashes,
        "artifact_sha256": artifact_hash,
        "build": {
            "command": public_command(
                build_command,
                {str(CUDA_BUILDER): "$CUDA_BUILDER", str(binary): "$CUDA_BINARY"},
            ),
            "nvcc": "$NVCC",
            "cuda_arch": os.environ.get("CUDA_ARCH", "sm_89"),
            "nvcc_version": subprocess.check_output([nvcc, "--version"], text=True),
        },
        "host_system": platform.platform(),
        "command": public_command(
            run_command,
            {str(binary): "$CUDA_BINARY", str(raw_output): "$RAW_OUTPUT"},
        ),
    }


def finish_receipt(
    raw_output: Path,
    output: Path,
    binary: Path,
    build_command: list[str],
    run_command: list[str],
    source_hashes: dict[str, str],
    artifact_hash: str,
    criteria: Path | None,
) -> None:
    receipt = json.loads(raw_output.read_text())
    canonicalize_receipt(receipt)
    for case in receipt["cases"]:
        expected_input, expected_oracle = case_digests(case["spec"])
        if case["input_digest"] != expected_input:
            raise SystemExit(f"{case['name']}: backend input digest differs from common oracle")
        case["backend_oracle_digest"] = case["oracle_digest"]
        case["oracle_digest"] = expected_oracle
    receipt["provenance"] = provenance(
        raw_output, binary, build_command, run_command, source_hashes, artifact_hash
    )
    if criteria is not None:
        receipt["criteria_sha256"] = sha256(criteria)
    publish_json(output, receipt)


def validate_run_options(args: argparse.Namespace, manifest: dict) -> tuple[int, int]:
    warmups = args.warmups if args.warmups is not None else manifest["warmup_runs"]
    repetitions = args.repetitions if args.repetitions is not None else manifest["measured_runs"]
    if warmups < 0 or repetitions <= 0:
        raise SystemExit("warmups must be nonnegative and repetitions must be positive")
    if args.phase == "evaluation" and args.criteria is None:
        raise SystemExit("evaluation requires --criteria")
    if args.criteria is not None and not args.criteria.exists():
        raise SystemExit(f"criteria file does not exist: {args.criteria}")
    return warmups, repetitions


def run_phase(args: argparse.Namespace, output: Path, warmups: int, repetitions: int) -> None:
    source_hashes = {str(path): sha256(path) for path in SOURCE_PATHS}
    build_command = build_binary(args.binary, source_hashes)
    source_hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}
    artifact_hash = sha256(args.binary)
    raw_output, run_command = run_binary(args.binary, output, args.phase, warmups, repetitions)
    try:
        current_sources = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}
        if source_hashes_after != current_sources or artifact_hash != sha256(args.binary):
            raise SystemExit("source or artifact changed while running CUDA harness")
        finish_receipt(
            raw_output, output, args.binary, build_command, run_command,
            source_hashes_after, artifact_hash, args.criteria
        )
    finally:
        raw_output.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    warmups, repetitions = validate_run_options(args, manifest)
    output = args.output or ROOT / "receipts/tmp" / f"prefix_attention_cuda_{args.phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_phase(args, output, warmups, repetitions)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
