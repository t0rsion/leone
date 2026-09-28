"""Select native tools and record the Runtime driver's build inputs."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path


def command_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def tool_identity(path: str) -> dict[str, str]:
    binary = Path(path)
    return {"sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def reject_overrides(environment: dict[str, str], names: list[str]) -> None:
    selected = sorted(name for name in names if name in environment)
    if selected:
        raise ValueError(f"unsupported native build overrides: {', '.join(selected)}")


def cuda_controls(environment: dict[str, str]) -> dict[str, object]:
    reject_overrides(environment, ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN"])
    compilers = {}
    for name in ("gcc", "g++"):
        path = shutil.which(name, path=environment.get("PATH"))
        if path is None:
            raise ValueError(f"CUDA host compiler is unavailable: {name}")
        compilers[name] = tool_identity(path)
    return {
        "lineinfo": environment.get("LEONE_CUDA_LINEINFO") == "1",
        "host_compilers": compilers,
    }


def same_native_input(name: str, actual: str, selected: str) -> bool:
    if actual == selected:
        return True
    if name in {"CC", "AR", "SDKROOT"}:
        return Path(actual).resolve(strict=True) == Path(selected).resolve(strict=True)
    return False


def set_native_input(environment: dict[str, str], name: str, selected: str) -> None:
    if name in environment and not same_native_input(name, environment[name], selected):
        raise ValueError(f"unsupported native build override: {name}")
    environment[name] = selected


def metal_controls(environment: dict[str, str]) -> dict[str, object]:
    prefixes = (
        "CC_", "CXX_", "AR_", "HOST_", "TARGET_", "CFLAGS_", "CXXFLAGS_",
        "OBJCFLAGS_", "CPPFLAGS_", "ARFLAGS_", "RANLIBFLAGS_",
    )
    scoped = [name for name in environment if name.startswith(prefixes)]
    reject_overrides(environment, scoped + [
        "DEVELOPER_DIR", "RANLIB", "ARFLAGS", "RANLIBFLAGS", "CRATE_CC_NO_DEFAULTS",
    ])
    xcrun = ["xcrun", "--sdk", "macosx"]
    compiler = command_text(xcrun + ["--find", "clang"])
    archiver = command_text(xcrun + ["--find", "ar"])
    sdk = command_text(xcrun + ["--show-sdk-path"])
    deployment = command_text(["sw_vers", "-productVersion"])
    for name, value in (("CC", compiler), ("AR", archiver), ("SDKROOT", sdk), ("MACOSX_DEPLOYMENT_TARGET", deployment)):
        set_native_input(environment, name, value)
    # Rust's debuginfo stripping can misalign proc-macro dylibs on macOS 27.
    set_native_input(environment, "CARGO_PROFILE_RELEASE_STRIP", "none")
    return {
        "compiler": tool_identity(compiler),
        "archiver": tool_identity(archiver),
        "sdk_version": command_text(xcrun + ["--show-sdk-version"]),
        "sdk_build": command_text(xcrun + ["--show-sdk-build-version"]),
        "deployment_target": deployment,
        "strip": "none",
    }


def native_environment(backend: str) -> tuple[dict[str, str], dict[str, object]]:
    environment = dict(os.environ)
    controls = cuda_controls(environment) if backend == "cuda" else metal_controls(environment)
    return environment, controls
