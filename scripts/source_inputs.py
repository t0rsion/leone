#!/usr/bin/env python3
"""Verify measured source inputs without requiring their development history."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path("receipts/source-inputs.json")
INPUTS = (
    "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo",
    "crates", "fixtures", "plans", "corpus",
    "benchmarks", "research/oracle", "external/PINNED",
    "scripts/quality-concurrent-service.sh", "scripts/run-llama-oracle.sh",
    "scripts/build-llama-oracle.sh", "scripts/check-openai-client.py",
    "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/check-openai-client.sh", "scripts/client-requirements.txt",
    "scripts/study-concurrent-service.py", "scripts/study-concurrent-service.sh",
    "scripts/study-live-server.sh", "scripts/study-batched-service.sh",
)
V04_EXECUTION_INPUTS = tuple(
    path for path in INPUTS if path not in {"plans", "corpus", "benchmarks"}
) + (
    "scripts/source_inputs.py",
    "build-support",
    "scripts/check-public-tree.py", "scripts/check-public-tree.sh",
    "scripts/check-batched-service.py", "scripts/render-release-evidence.sh",
    "scripts/check-release-evidence.py", "scripts/release_evidence_manifest.py",
    "scripts/release_evidence_validators.py",
    "scripts/validate-quality-stage.py",
    "scripts/release-evidence-manifest.py", "scripts/verify-release-archive.py",
    "scripts/compare-metal-quality.sh", "scripts/export-quality-oracle.sh",
    "scripts/import-metal-quality.sh", "scripts/write-quality-samples.py",
    "scripts/write-quality-task.py", "scripts/write-cuda-quality-comparison.py",
    "scripts/freeze-cuda-generation.py", "scripts/write-metal-quality-generation.py",
    "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/generate-openai-chat-template-legacy-fixture.py",
    "scripts/study-branching-service.py", "scripts/study-branching-service.sh",
    "scripts/freeze-branching-manifest.py", "scripts/linked_libraries.py",
    "scripts/produce-history-tokenization.py", "scripts/check-history-tokenization.py",
    "scripts/fetch-llama-cpp.sh",
)
V04_WORKLOAD_INPUTS = ("plans", "corpus", "benchmarks")
V04_EVIDENCE_INPUTS = (
    "receipts/v04-*",
    "receipts/v04-linux-cuda-runtime.json",
    "receipts/v04-linux-cuda-quality-qwen3.json",
    "receipts/v04-linux-cuda-quality-llama.json",
    "receipts/v04-linux-cuda-quality-comparison-qwen3.json",
    "receipts/v04-linux-cuda-quality-comparison-llama.json",
    "receipts/v04-linux-cuda-service.json",
    "receipts/v04-linux-cuda-batched-service.json",
    "receipts/v04-linux-cuda-branching-service.json",
    "receipts/v04-darwin-metal-runtime.json",
    "receipts/v04-darwin-metal-quality-qwen3.json",
    "receipts/v04-darwin-metal-quality-llama.json",
    "receipts/v04-darwin-metal-quality-comparison-qwen3.json",
    "receipts/v04-darwin-metal-quality-comparison-llama.json",
    "receipts/v04-darwin-metal-service.json",
    "receipts/v04-darwin-metal-branching-service.json",
    "receipts/v04-openai-client-check.json",
    "receipts/v04-service-metrics.json",
)
V04_INPUTS = V04_EXECUTION_INPUTS
V04_MANIFEST_PATHS = (
    "receipts/source-inputs-v04.json",
    "receipts/source-inputs-v04-prestudy.json",
)
V04_REQUIRED_FILES = (
    "build-support/provenance.rs",
    "scripts/source_inputs.py",
    "scripts/check-public-tree.py", "scripts/check-public-tree.sh",
    "scripts/check-batched-service.py", "scripts/render-release-evidence.sh",
    "scripts/check-release-evidence.py", "scripts/release_evidence_manifest.py",
    "scripts/release_evidence_validators.py",
    "scripts/validate-quality-stage.py",
    "scripts/compare-metal-quality.sh", "scripts/export-quality-oracle.sh",
    "scripts/import-metal-quality.sh", "scripts/write-quality-samples.py",
    "scripts/write-quality-task.py", "scripts/write-cuda-quality-comparison.py",
    "scripts/freeze-cuda-generation.py", "scripts/write-metal-quality-generation.py",
    "scripts/generate-openai-chat-template-fixtures.py",
    "scripts/generate-openai-chat-template-legacy-fixture.py",
    "scripts/study-branching-service.py", "scripts/study-branching-service.sh",
    "scripts/freeze-branching-manifest.py", "scripts/linked_libraries.py",
    "scripts/produce-history-tokenization.py", "scripts/check-history-tokenization.py",
    "scripts/fetch-llama-cpp.sh",
)


def git(root, *arguments):
    return subprocess.check_output(["git", *arguments], cwd=root, stderr=subprocess.PIPE)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(root, revision, inputs=INPUTS):
    entries = git(root, "ls-tree", "-r", "-z", revision, "--", *inputs)
    files = {}
    for entry in entries.decode().split("\0"):
        if entry:
            metadata, name = entry.split("\t", 1)
            files[name] = {
                "sha256": digest(git(root, "show", f"{revision}:{name}")),
                "executable": metadata.split()[0] == "100755",
            }
    return {
        "source_commit": git(root, "rev-parse", revision).decode().strip(),
        "files": files,
    }


def current_files(root, inputs=INPUTS):
    names = git(
        root,
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *inputs,
    )
    return {name for name in names.decode().split("\0") if name}


def _snapshot_current(root, inputs):
    files = {}
    for name in sorted(current_files(root, inputs)):
        path = _source_file(root, name)
        files[name] = {
            "sha256": digest(path.read_bytes()),
            "executable": bool(path.stat().st_mode & 0o111),
        }
    return files


def _source_file(root: Path, name: str) -> Path:
    if not _safe_source_name(name):
        raise ValueError(f"source input path is unsafe: {name}")
    path = Path(name)
    current = _reject_symlink_parents(root, path, name)
    try:
        current.resolve(strict=True).relative_to(root.resolve(strict=True))
    except FileNotFoundError as error:
        raise ValueError(f"source input is not a regular file: {name}") from error
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"source input escapes the source tree: {name}") from error
    if not current.is_file() or current.is_symlink():
        raise ValueError(f"source input is not a regular file: {name}")
    return current


def _safe_source_name(name):
    if not isinstance(name, str) or not name or "\\" in name:
        return False
    path = Path(name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == name
        and name != "."
    )


def _reject_symlink_parents(root, path, name):
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"source input is not a regular file: {name} (symlink parent)"
            )
    return current


def check_files(root, files, inputs=INPUTS):
    if not isinstance(files, dict) or not files:
        raise ValueError("source input manifest has no files")
    paths = _source_paths(root, files)
    if set(files) != current_files(root, inputs):
        raise ValueError("measured source file set differs")
    for name, expected in files.items():
        path = paths[name]
        _check_source_record(path, name, expected)


def _check_optional_files(root, files, inputs, label):
    if files is None:
        return
    if not isinstance(files, dict):
        raise ValueError(f"{label} source input records are malformed")
    if files:
        try:
            check_files(root, files, inputs)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error


def _source_paths(root, files):
    paths = {}
    for name, expected in files.items():
        _validate_source_record(name, expected)
        paths[name] = _source_file(root, name)
    return paths


def _validate_source_record(name, expected):
    if not isinstance(expected, dict):
        raise ValueError(f"source input record is malformed: {name}")
    executable = expected.get("executable")
    sha256 = expected.get("sha256")
    if not isinstance(executable, bool):
        raise ValueError(f"source executable mode is malformed: {name}")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(f"source input hash is malformed: {name}")
    if any(character not in "0123456789abcdef" for character in sha256.lower()):
        raise ValueError(f"source input hash is malformed: {name}")


def _check_source_record(path, name, expected):
    if bool(path.stat().st_mode & 0o111) != expected["executable"]:
        raise ValueError(f"source executable mode changed: {name}")
    if digest(path.read_bytes()) != expected["sha256"]:
        raise ValueError(f"measured source input changed: {name}")


def _recorded_files(manifest):
    records = {}
    for field in ("files", "workload_files", "evidence_files"):
        group = manifest.get(field, {} if field != "files" else None)
        if not isinstance(group, dict) or (field == "files" and not group):
            raise ValueError(f"source input {field} are malformed")
        for name, expected in group.items():
            _validate_source_record(name, expected)
            if name in records:
                raise ValueError(f"source input path is repeated: {name}")
            records[name] = expected
    return records


def check_packaged_files(root, manifest, required=()):
    """Verify recorded source files below root without Git history.

    Return every recorded name with its record. A recorded file that is absent
    from root passes unless it is in required. Recorded content and mode must
    match for each file that exists.
    """
    records = _recorded_files(manifest)
    missing = sorted(set(required) - set(records))
    if missing:
        raise ValueError("source manifest omits required files: " + ", ".join(missing))
    for name, expected in records.items():
        if not _safe_source_name(name):
            raise ValueError(f"source input path is unsafe: {name}")
        candidate = root / name
        if name in required or candidate.exists() or candidate.is_symlink():
            _check_source_record(_source_file(root, name), name, expected)
    return records


def _manifest_path(root: Path, value: Path) -> Path:
    if value.is_absolute() or ".." in value.parts or "\\" in value.as_posix():
        raise ValueError(f"source manifest path is unsafe: {value}")
    return _source_file(root, value.as_posix())


def check(root, revision, manifest_path=MANIFEST):
    manifest_path = Path(manifest_path)
    if not _safe_source_name(manifest_path.as_posix()):
        raise ValueError(f"source manifest path is unsafe: {manifest_path}")
    inputs = (
        V04_EXECUTION_INPUTS
        if manifest_path.as_posix() in V04_MANIFEST_PATHS
        else INPUTS
    )
    candidate = root / manifest_path
    path = _manifest_path(root, manifest_path) if candidate.exists() or candidate.is_symlink() else candidate
    if path.is_file():
        manifest = json.loads(path.read_text())
        if manifest["source_commit"] == revision:
            check_files(root, manifest["files"], inputs)
            if inputs == V04_EXECUTION_INPUTS:
                _check_optional_files(
                    root,
                    manifest.get("workload_files"),
                    V04_WORKLOAD_INPUTS,
                    "workload",
                )
                _check_optional_files(
                    root,
                    manifest.get("evidence_files"),
                    V04_EVIDENCE_INPUTS,
                    "evidence",
                )
            return
    git(root, "merge-base", "--is-ancestor", revision, "HEAD")
    git(root, "diff", "--exit-code", revision, "--", *inputs)
    if git(root, "ls-files", "--others", "--exclude-standard", "--", *inputs):
        raise ValueError("source inputs include unrecorded files")


def main(arguments):
    if len(arguments) == 3 and arguments[0] in {"record", "record-v04"}:
        _write_record(arguments)
        return
    if arguments and arguments[0] == "check":
        _check_command(arguments)
        return
    raise ValueError("usage: source_inputs.py check SOURCE | record SOURCE OUTPUT | record-v04 SOURCE OUTPUT")


def _write_record(arguments):
    mode, revision, output_path = arguments
    inputs = V04_EXECUTION_INPUTS if mode == "record-v04" else INPUTS
    record = snapshot(ROOT, revision, inputs)
    if mode == "record-v04":
        missing = sorted(set(V04_REQUIRED_FILES) - set(record["files"]))
        if missing:
            raise ValueError("v0.4 source inputs are missing: " + ", ".join(missing))
        record["schema_version"] = "leone.source-inputs.v2"
        record["workload_files"] = _snapshot_current(ROOT, V04_WORKLOAD_INPUTS)
        record["evidence_files"] = _snapshot_current(ROOT, V04_EVIDENCE_INPUTS)
    with Path(output_path).open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")


def _check_command(arguments):
    if len(arguments) == 2:
        check(ROOT, arguments[1])
        return
    if len(arguments) == 4 and arguments[2] == "--manifest":
        check(ROOT, arguments[1], Path(arguments[3]))
        return
    raise ValueError("usage: source_inputs.py check SOURCE [--manifest PATH]")


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"source verification failed: {error}", file=sys.stderr)
        sys.exit(1)
