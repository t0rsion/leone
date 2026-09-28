#!/usr/bin/env python3
"""Build and run the standalone Metal prefix attention receipt generator."""

from __future__ import annotations

import argparse
import hashlib
import json
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
BUILDER = RESEARCH_ROOT / "metal_fixed_reduction/build_metal.sh"
SOURCE_PATHS = [
    RESEARCH_ROOT / "metal_fixed_reduction/PrefixAttention.metal",
    RESEARCH_ROOT / "metal_fixed_reduction/RunPrefixAttention.swift",
    RESEARCH_ROOT / "metal_fixed_reduction/build_metal.sh",
    RESEARCH_ROOT / "generate_manifest.py",
    RESEARCH_ROOT / "common_oracle.py",
    RESEARCH_ROOT / "runner_support.py",
    RESEARCH_ROOT / "ORACLE_CONTRACT.md",
    MANIFEST,
    RESEARCH_ROOT / "run_metal.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--metallib", type=Path, default=Path("/tmp/prefix_attention.metallib"))
    parser.add_argument("--binary", type=Path, default=Path("/tmp/prefix_attention_metal"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--build", action="store_true", help="kept for command compatibility")
    parser.add_argument("--criteria", type=Path)
    return parser.parse_args()


def build_artifacts(metallib: Path, binary: Path, before: dict[str, str]) -> list[str]:
    metallib.parent.mkdir(parents=True, exist_ok=True)
    binary.parent.mkdir(parents=True, exist_ok=True)
    command = [str(BUILDER), str(metallib), str(binary)]
    subprocess.run(command, check=True)
    after = {str(path): sha256(path) for path in SOURCE_PATHS}
    if before != after:
        raise SystemExit("source changed while building Metal harness")
    if not metallib.exists() or not binary.exists():
        raise SystemExit("Metal builder did not produce both artifacts")
    return command


def run_binary(binary: Path, metallib: Path, output: Path, phase: str,
               warmups: int, repetitions: int) -> tuple[Path, list[str]]:
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".raw", delete=False
    ) as raw_file:
        raw_output = Path(raw_file.name)
    command = [
        str(binary), "--phase", phase, "--metallib", str(metallib),
        "--output", str(raw_output), "--warmups", str(warmups),
        "--repetitions", str(repetitions),
    ]
    try:
        subprocess.run(command, check=True)
        return raw_output, command
    except BaseException:
        raw_output.unlink(missing_ok=True)
        raise


def tool_version(command: str) -> str:
    return subprocess.check_output([command, "--version"], text=True, stderr=subprocess.STDOUT)


def make_provenance(raw_output: Path, metallib: Path, binary: Path, build_command: list[str],
                    run_command: list[str], sources: dict[str, str],
                    artifact_hashes: dict[str, str]) -> dict:
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
        "source_sha256": sources,
        "artifact_sha256": artifact_hashes,
        "build": {
            "command": public_command(
                build_command,
                {
                    str(BUILDER): "$METAL_BUILDER",
                    str(metallib): "$METALLIB",
                    str(binary): "$METAL_BINARY",
                },
            ),
            "xcrun_version": tool_version("xcrun"),
            "swiftc_version": tool_version("swiftc"),
        },
        "host_system": platform.platform(),
        "command": public_command(
            run_command,
            {
                str(binary): "$METAL_BINARY",
                str(metallib): "$METALLIB",
                str(raw_output): "$RAW_OUTPUT",
            },
        ),
    }


def finish_receipt(raw: Path, output: Path, metallib: Path, binary: Path,
                   build_command: list[str], run_command: list[str],
                   sources: dict[str, str], artifact_hashes: dict[str, str],
                   criteria: Path | None) -> None:
    receipt = json.loads(raw.read_text())
    canonicalize_receipt(receipt)
    for case in receipt["cases"]:
        expected_input, expected_oracle = case_digests(case["spec"])
        if case["input_digest"] != expected_input:
            raise SystemExit(f"{case['name']}: backend input digest differs from common oracle")
        case["backend_oracle_digest"] = case["oracle_digest"]
        case["oracle_digest"] = expected_oracle
    receipt["provenance"] = make_provenance(
        raw, metallib, binary, build_command, run_command, sources, artifact_hashes
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
    build_command = build_artifacts(args.metallib, args.binary, source_hashes)
    source_hashes_after = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}
    artifact_hashes = {"runner": sha256(args.binary), "metallib": sha256(args.metallib)}
    raw_output, run_command = run_binary(
        args.binary, args.metallib, output, args.phase, warmups, repetitions
    )
    try:
        current_sources = {str(path.relative_to(ROOT)): sha256(path) for path in SOURCE_PATHS}
        current_artifacts = {"runner": sha256(args.binary), "metallib": sha256(args.metallib)}
        if source_hashes_after != current_sources or artifact_hashes != current_artifacts:
            raise SystemExit("source or artifact changed while running Metal harness")
        finish_receipt(
            raw_output, output, args.metallib, args.binary, build_command,
            run_command, source_hashes_after, artifact_hashes, args.criteria
        )
    finally:
        raw_output.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    manifest = json.loads(MANIFEST.read_text())
    warmups, repetitions = validate_run_options(args, manifest)
    output = args.output or ROOT / "receipts/tmp" / f"prefix_attention_metal_{args.phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    run_phase(args, output, warmups, repetitions)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
