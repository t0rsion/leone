#!/usr/bin/env python3
"""Build and run the opt-in Runtime shared-prefix driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from runner_support import public_command, publish_json
from runtime_build import native_environment

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research/prefix_attention"
DRIVER = RESEARCH / "runtime_driver"
TARGET = Path(os.environ.get("CARGO_TARGET_DIR", str(DRIVER / "target")))
if not TARGET.is_absolute():
    TARGET = ROOT / TARGET
BINARY = TARGET / "release/prefix-attention-runtime-driver"
MANIFEST = RESEARCH / "runtime_manifest.json"
COMMON_SOURCES = [
    DRIVER / "Cargo.toml",
    DRIVER / "build.rs",
    DRIVER / "src/build_info.rs",
    DRIVER / "src/backend_adapter.rs",
    DRIVER / "Cargo.lock",
    ROOT / "Cargo.toml",
    ROOT / "Cargo.lock",
    ROOT / ".cargo/config.toml",
    ROOT / "rust-toolchain.toml",
    ROOT / "crates/leone/Cargo.toml",
    *sorted((ROOT / "crates/leone/src").rglob("*.rs")),
    ROOT / "crates/leone-gguf/Cargo.toml",
    *sorted((ROOT / "crates/leone-gguf/src").rglob("*.rs")),
    ROOT / "crates/leone-receipt/Cargo.toml",
    *sorted((ROOT / "crates/leone-receipt/src").rglob("*.rs")),
    DRIVER / "src/main.rs",
    MANIFEST,
    RESEARCH / "llama_cached_workload.json",
    RESEARCH / "fixtures/qwen-v01.tokens.u32le",
    RESEARCH / "run_runtime.py",
    RESEARCH / "runner_support.py",
    RESEARCH / "runtime_build.py",
    ROOT / "build-support/provenance.rs",
]
BACKEND_SOURCES = {
    "cuda": [
        ROOT / "crates/leone-cuda/Cargo.toml",
        ROOT / "crates/leone-cuda/build.rs",
        *sorted((ROOT / "crates/leone-cuda/src").rglob("*.rs")),
        ROOT / "crates/leone-cuda/cuda/ie_cuda.cu",
    ],
    "metal": [
        ROOT / "crates/leone-metal/Cargo.toml",
        ROOT / "crates/leone-metal/build.rs",
        *sorted((ROOT / "crates/leone-metal/src").rglob("*.rs")),
        ROOT / "crates/leone-metal/metal/leone.metal",
        ROOT / "crates/leone-metal/metal/metal_bridge.m",
        ROOT / "crates/leone-metal/metal/metal_bridge_stub.c",
    ],
}

DIGEST = re.compile(r"[0-9a-f]{64}")
IDENTITY_DIGESTS = (
    "source_tree_sha256", "rustc_sha256", "toolchain_sha256", "native_tools_sha256",
    "build_config_sha256", "build_flags_sha256", "build_flags_raw_sha256",
    "profile_inputs_sha256", "linker_inputs_sha256",
)
DRIVER_MANIFEST = "research/prefix_attention/runtime_driver/Cargo.toml"

AFFINITY_CPUSETS = {
    "build": "16-31",
    "timed_run": "0-3,12-15",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "metal"), default="cuda")
    parser.add_argument("--phase", choices=("calibration", "evaluation"), required=True)
    parser.add_argument("--path", choices=(
        "per_row", "fixed_tile_per_row", "shared_read_unconstrained",
        "shared_read_fixed_reduction",
    ), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--criteria", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    return parser.parse_args()


def source_paths(backend: str) -> list[Path]:
    return COMMON_SOURCES + BACKEND_SOURCES[backend]


def source_hashes(backend: str) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in source_paths(backend)}


def affinity_prefix(role: str, host: str | None = None) -> list[str]:
    if (platform.system() if host is None else host) != "Linux":
        return []
    return ["taskset", "-c", AFFINITY_CPUSETS[role]]


def build_commands(backend: str) -> tuple[list[str], list[str]]:
    manifest = ["--manifest-path", DRIVER_MANIFEST]
    prefix = affinity_prefix("build") + ["cargo", "+1.92"]
    clean = prefix + [
        "clean", "--release", *manifest,
        "-p", "prefix-attention-runtime-driver", "-p", f"leone-{backend}",
    ]
    compile_ = prefix + [
        "build", "--release", "--locked", *manifest,
        "--no-default-features", "--features", backend,
    ]
    return clean, compile_


def build(
    backend: str, before: dict[str, str], environment: dict[str, str]
) -> tuple[list[str], list[str]]:
    clean, compile_ = build_commands(backend)
    subprocess.run(clean, cwd=ROOT, env=environment, check=True)
    subprocess.run(compile_, cwd=ROOT, env=environment, check=True)
    if before != source_hashes(backend):
        raise SystemExit("runtime driver source changed while building")
    if not BINARY.exists():
        raise SystemExit("runtime driver did not produce a binary")
    return clean, compile_


def identity_problem(identity: dict[str, object], backend: str) -> str | None:
    """Name the first reason a build record cannot support a published receipt."""
    if identity.get("features") != backend:
        return f"the driver was built for {identity.get('features')!r}, not {backend!r}"
    if identity.get("profile") != "release":
        return "the driver is not a release build"
    if identity.get("native_provenance") != "complete":
        return "native compiler provenance is incomplete"
    if identity.get("provenance_unknown") is not False:
        return "build provenance is unknown"
    if identity.get("source_tree_dirty") is not False:
        return "source tree is dirty or its state is unknown"
    invalid = [
        name for name in IDENTITY_DIGESTS if not DIGEST.fullmatch(str(identity.get(name)))
    ]
    return f"invalid digests: {', '.join(invalid)}" if invalid else None


def build_identity(backend: str) -> dict[str, object]:
    identity = json.loads(subprocess.check_output([str(BINARY), "--build-info"], text=True))
    problem = identity_problem(identity, backend)
    if problem is not None:
        raise SystemExit(f"runtime driver build identity rejected: {problem}")
    return identity


@dataclass(frozen=True)
class BuildRecord:
    commands: tuple[list[str], list[str]]
    sources: dict[str, str]
    artifact: str
    info: dict[str, object]
    controls: dict[str, object]


def provenance(
    backend: str,
    record: BuildRecord,
    run_command: list[str],
    affinity: dict[str, dict[str, object]],
    manifest: Path,
) -> dict:
    clean, compile_ = (public_command(command, {}) for command in record.commands)
    return {
        "backend": backend,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest),
        "source_sha256": record.sources,
        "artifact_sha256": record.artifact,
        "build": {
            "clean_command": clean,
            "command": compile_,
            **record.info,
            "native_controls": record.controls,
        },
        "host_system": platform.platform(),
        "command": run_command,
        "affinity": affinity,
    }


def public_run_command(command: list[str], args: argparse.Namespace, raw: Path) -> list[str]:
    return public_command(command, {
        str(BINARY): "$RUNTIME_BINARY",
        str(MANIFEST): str(MANIFEST.relative_to(ROOT)),
        str(args.model): "$MODEL",
        str(raw): "$RAW_OUTPUT",
    })


def affinity_records() -> dict[str, dict[str, object]]:
    host = platform.system()
    return {
        role: {
            "role": role,
            "platform": host,
            "command": affinity_prefix(role, host),
        }
        for role in AFFINITY_CPUSETS
    }


def qualify_sidecars(receipt: dict, directory: Path) -> None:
    for case in receipt["cases"]:
        for sidecar in case["logits_sidecars"]:
            relative = Path(sidecar["file"])
            if relative.is_absolute():
                raise ValueError("runtime sidecar path is absolute")
            source = directory / relative
            source.resolve().relative_to(directory.resolve())
            if sha256(source) != sidecar["sha256"]:
                raise ValueError("runtime sidecar hash differs from the driver receipt")
            sidecar["file"] = str(Path(directory.name) / relative)


def finish_receipt(args, raw, record, run_command, criteria):
    if record.sources != source_hashes(args.backend) or record.artifact != sha256(BINARY):
        raise SystemExit("runtime driver identity changed while running")
    if criteria is not None and sha256(args.criteria) != criteria:
        raise SystemExit("criteria file changed while running")
    receipt = json.loads(raw.read_text())
    qualify_sidecars(receipt, raw.parent)
    receipt["backend"] = args.backend
    receipt["provenance"] = provenance(
        args.backend, record, public_run_command(run_command, args, raw),
        affinity_records(), MANIFEST,
    )
    if criteria is not None:
        receipt["criteria_sha256"] = criteria
    publish_json(args.output, receipt)


def check_arguments(args: argparse.Namespace) -> None:
    """Reject unusable arguments before any build or measurement starts."""
    if args.repetitions <= 0 or args.warmups < 0:
        raise SystemExit("repetitions must be positive and warmups cannot be negative")
    if args.phase == "evaluation" and args.criteria is None:
        raise SystemExit("evaluation requires --criteria")
    if os.path.lexists(args.output):
        raise SystemExit(f"refusing to replace existing output: {args.output}")
    if args.criteria is not None:
        require_regular_file(args.criteria)


def require_regular_file(path: Path) -> None:
    try:
        regular = stat.S_ISREG(path.stat().st_mode)
    except OSError as error:
        raise SystemExit(f"criteria file is unusable: {path}: {error.strerror}") from error
    if not regular:
        raise SystemExit(f"criteria must be a regular file: {path}")


def reserve_artifacts(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=output.parent, prefix="runtime-", suffix=".artifacts"))


def run_measurement(command: list[str], raw: Path, run_dir: Path) -> None:
    """Run the driver. Only a run that wrote no receipt is removed."""
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except BaseException:
        if not raw.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        raise


def publish_or_retain(args, raw, run_dir, record, run_command, criteria) -> None:
    """Publish the receipt. A completed measurement is never deleted."""
    try:
        finish_receipt(args, raw, record, run_command, criteria)
    except (Exception, SystemExit) as error:
        raise SystemExit(
            f"runtime receipt not published: {error}\n"
            f"the completed run is kept in {run_dir} (receipt.json and its sidecars)"
        ) from error
    raw.unlink()


def main() -> int:
    args = parse_args()
    check_arguments(args)
    criteria = None if args.criteria is None else sha256(args.criteria)
    before = source_hashes(args.backend)
    environment, native_controls = native_environment(args.backend)
    commands = build(args.backend, before, environment)
    record = BuildRecord(
        commands, source_hashes(args.backend), sha256(BINARY),
        build_identity(args.backend), native_controls,
    )
    run_dir = reserve_artifacts(args.output)
    raw = run_dir / "receipt.json"
    run_command = affinity_prefix("timed_run") + [
        str(BINARY),
        "--phase", args.phase,
        "--path", args.path,
        "--manifest", str(MANIFEST),
        "--model", str(args.model),
        "--output", str(raw),
        "--repetitions", str(args.repetitions),
        "--warmups", str(args.warmups),
    ]
    run_measurement(run_command, raw, run_dir)
    publish_or_retain(args, raw, run_dir, record, run_command, criteria)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
