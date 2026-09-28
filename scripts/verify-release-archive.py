#!/usr/bin/env python3
"""Verify a Leone archive without compiling code or loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CHECKERS = ("scripts/check-public-tree.sh", "scripts/check-public-tree.py")
PACKAGE_IDENTITIES = {
    ("linux-x86_64", "x86_64-unknown-linux-gnu", "cuda"),
    ("darwin-arm64", "aarch64-apple-darwin", "metal"),
}
V03_CHECKERS = (
    "scripts/check-release-evidence.py",
    "scripts/release_evidence_manifest.py",
    "scripts/source_inputs.py",
    "scripts/study-concurrent-service.py",
)


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"archive contains an unsafe path: {name}")
    return path


def _member_top_level(normalized: PurePosixPath, expected: str | None) -> str:
    top_level = normalized.parts[0]
    if expected is not None and top_level != expected:
        raise ValueError("archive must contain one top-level directory")
    return top_level


def _normalize_member(
    member: tarfile.TarInfo, names: set[str], top_level: str | None
) -> tuple[tarfile.TarInfo, PurePosixPath, str]:
    normalized = safe_member(member.name)
    top_level = _member_top_level(normalized, top_level)
    normalized_name = normalized.as_posix()
    if normalized_name in names:
        raise ValueError(f"archive repeats a member: {member.name}")
    names.add(normalized_name)
    if not (member.isdir() or member.isreg()):
        raise ValueError(f"archive contains a non-regular member: {member.name}")
    return member, normalized, top_level


def _normalized_members(
    members: list[tarfile.TarInfo],
) -> tuple[list[tuple[tarfile.TarInfo, PurePosixPath]], str]:
    normalized_members = []
    names: set[str] = set()
    top_level: str | None = None
    for member in members:
        normalized_member = _normalize_member(member, names, top_level)
        top_level = normalized_member[2]
        normalized_members.append(normalized_member[:2])
    if top_level is None:
        raise ValueError("archive is empty")
    return normalized_members, top_level


def _validate_root_member(
    members: list[tuple[tarfile.TarInfo, PurePosixPath]], top_level: str
) -> None:
    root_members = [member for member in members if member[1].parts == (top_level,)]
    if len(root_members) > 1 or (root_members and not root_members[0][0].isdir()):
        raise ValueError("archive must contain one top-level directory")


def _extract_member(
    bundle: tarfile.TarFile,
    destination: Path,
    member: tarfile.TarInfo,
    normalized: PurePosixPath,
) -> None:
    target = destination / normalized
    _safe_destination_path(destination, normalized)
    if member.isdir():
        if target.exists() and not target.is_dir():
            raise ValueError(f"archive directory collides with a file: {member.name}")
        target.mkdir(parents=True, exist_ok=True)
        return
    if target.exists() and target.is_dir():
        raise ValueError(f"archive file collides with a directory: {member.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    source = bundle.extractfile(member)
    if source is None:
        raise ValueError(f"archive member has no data: {member.name}")
    with target.open("wb") as output:
        shutil.copyfileobj(source, output)
    target.chmod(member.mode & 0o777)


def _safe_destination_path(destination: Path, relative: PurePosixPath) -> Path:
    """Reject existing links before writing one extracted member."""
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("archive extraction destination is not a regular directory")
    current = destination
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"archive extraction path contains a symlink: {relative}")
    return current


def _prepare_destination(destination: Path) -> None:
    if _check_existing_destination(destination):
        return
    _create_destination(destination)


def _check_existing_destination(destination: Path) -> bool:
    if not destination.exists() and not destination.is_symlink():
        return False
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("archive extraction destination is not a directory")
    if any(destination.iterdir()):
        raise ValueError("archive extraction destination is not empty")
    return True


def _create_destination(destination: Path) -> None:
    current = destination
    missing: list[Path] = []
    while not current.exists():
        if current.is_symlink():
            raise ValueError("archive extraction destination contains a symlink parent")
        missing.append(current)
        current = current.parent
    if current.is_symlink():
        raise ValueError("archive extraction destination contains a symlink parent")
    if not current.is_dir():
        raise ValueError("archive extraction destination parent is not a directory")
    for path in reversed(missing):
        path.mkdir()
        if path.is_symlink() or not path.is_dir():
            raise ValueError("archive extraction destination contains a symlink parent")


def _extracted_root(destination: Path) -> Path:
    entries = sorted(destination.iterdir())
    if len(entries) != 1 or not entries[0].is_dir():
        raise ValueError("archive must contain one top-level directory")
    return entries[0]


def extract(archive: Path, destination: Path) -> Path:
    """Extract one archive after rejecting links and unsafe member paths."""
    _prepare_destination(destination)
    with tarfile.open(archive, "r:gz") as bundle:
        normalized_members, top_level = _normalized_members(bundle.getmembers())
        _validate_root_member(normalized_members, top_level)
        for member, normalized in normalized_members:
            _extract_member(bundle, destination, member, normalized)
    return _extracted_root(destination)


def manifest_entries(root: Path) -> dict[str, str]:
    """Read and validate the archive's content manifest."""
    path = root / "MANIFEST.sha256"
    if not path.is_file():
        raise ValueError("archive is missing MANIFEST.sha256")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError("archive manifest contains an invalid row")
        relative = fields[1].lstrip(" *")
        if relative.startswith("./"):
            relative = relative[2:]
        safe_member(relative)
        if relative in entries:
            raise ValueError(f"archive manifest repeats {relative}")
        entries[relative] = fields[0].lower()
    if not entries:
        raise ValueError("archive manifest is empty")
    return entries


def verify_manifest(root: Path) -> dict[str, str]:
    """Check every manifest digest and reject staged files outside it."""
    expected = manifest_entries(root)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"archive contains a symlink: {path.relative_to(root)}")
        if path.is_file() and path != root / "MANIFEST.sha256":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("archive files differ from MANIFEST.sha256")
    for relative, expected_digest in expected.items():
        actual_digest = digest(root / relative)
        if actual_digest != expected_digest:
            raise ValueError(f"archive digest differs for {relative}")
    return expected


def verify_sidecar(archive: Path) -> None:
    """Check an adjacent archive checksum file when the package provides one."""
    sidecar = Path(f"{archive}.sha256")
    if not sidecar.is_file():
        raise ValueError(f"archive checksum is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if len(fields) < 2 or fields[0].lower() != digest(archive):
        raise ValueError("archive checksum does not match")
    if Path(fields[-1]).name != archive.name:
        raise ValueError("archive checksum names another file")


def _read_package_record(root: Path, name: str) -> dict[str, object]:
    path = _regular_file(root, name, f"package {name}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"package metadata is invalid: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"package {name} is not an object")
    return record


def _regular_file(root: Path, relative: str, label: str) -> Path:
    """Return one regular file whose parents remain below one trusted root."""
    path = root / relative
    if not path.exists():
        raise ValueError(f"{label} is missing")
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{label} escapes its root") from error
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink parent")
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a regular file")
    return path


def _package_identity(metadata: dict[str, object], environment: dict[str, object]) -> str:
    kind = _package_kind(metadata, environment)
    identity = _package_target_identity(metadata)
    _check_environment_identity(metadata, environment, identity)
    return kind


def _package_kind(metadata: dict[str, object], environment: dict[str, object]) -> str:
    if metadata.get("schema_version") != "leone.package.v1":
        raise ValueError("package metadata has an unknown schema")
    kind = metadata.get("package")
    if kind not in {"runtime", "evidence"}:
        raise ValueError("package metadata has an unknown package kind")
    if environment.get("schema_version") != "leone.package-environment.v1":
        raise ValueError("package environment has an unknown schema")
    if environment.get("package") != kind:
        raise ValueError("package metadata disagrees on package kind")
    return kind


def _package_target_identity(metadata: dict[str, object]) -> tuple[str, str, str]:
    identity = tuple(metadata.get(field) for field in ("platform", "target", "backend"))
    if not all(isinstance(value, str) for value in identity):
        raise ValueError("package metadata has an incomplete target identity")
    if identity not in PACKAGE_IDENTITIES:
        raise ValueError("package metadata has an unsupported platform/backend pair")
    return identity  # type: ignore[return-value]


def _check_environment_identity(
    metadata: dict[str, object], environment: dict[str, object], identity: tuple[str, str, str]
) -> None:
    for field, expected in zip(("platform", "target", "backend"), identity):
        if environment.get(field) != expected:
            raise ValueError(f"package metadata disagrees on {field}")


def _required_package_files(root: Path, kind: str) -> tuple[str, ...]:
    if kind == "runtime":
        return ("bin/leone", "install.sh")
    return (
        "README.md",
        "release-evidence.json",
        "scripts/check-release-evidence.py",
    )


def package_kind(root: Path) -> str:
    """Validate package metadata and return the declared archive kind."""
    metadata = _read_package_record(root, "package-info.json")
    environment = _read_package_record(root, "environment.json")
    kind = _package_identity(metadata, environment)
    required = _required_package_files(root, kind)
    missing = [relative for relative in required if not (root / relative).is_file()]
    if missing:
        raise ValueError(f"{kind} package is missing required files: {', '.join(missing)}")
    if kind == "runtime":
        for relative in ("bin/leone", "install.sh"):
            if not (root / relative).stat().st_mode & 0o111:
                raise ValueError(f"runtime package file is not executable: {relative}")
    return kind


def trusted_root(value: Path | None) -> Path:
    """Return the checker root and reject an implicit unversioned checker root."""
    if value is not None:
        return value.resolve()
    if not (REPO_ROOT / ".git").exists():
        raise ValueError("trusted checker root must be supplied outside a source checkout")
    return REPO_ROOT


def _trusted_manifest(root: Path, checker_root: Path) -> dict[str, object]:
    """Require the archive manifest to equal the trusted release manifest."""
    archive_manifest = _read_package_record(root, "release-evidence.json")
    release_line = archive_manifest.get("release_line")
    if release_line not in {"v0.3", "v0.4"}:
        raise ValueError("evidence archive has an unsupported release line")
    trusted_path = checker_root / "packaging" / f"release-evidence.{release_line}.json"
    trusted = _regular_file(checker_root, trusted_path.relative_to(checker_root).as_posix(), "trusted release manifest")
    archive_path = _regular_file(root, "release-evidence.json", "archive release manifest")
    if digest(archive_path) != digest(trusted):
        raise ValueError("archive release manifest differs from trusted release manifest")
    return archive_manifest


def _trusted_checker_paths(manifest: dict[str, object]) -> tuple[str, ...]:
    """Return the exact checker files whose trusted copies may run."""
    release_line = manifest.get("release_line")
    if release_line == "v0.3":
        return V03_CHECKERS + PUBLIC_CHECKERS
    validators = manifest.get("trusted_validators")
    if not isinstance(validators, list) or any(not isinstance(path, str) for path in validators):
        raise ValueError("v0.4 release manifest has no trusted validator list")
    return tuple(validators) + PUBLIC_CHECKERS


def _verify_trusted_checkers(
    root: Path,
    checker_root: Path,
    manifest: dict[str, object],
    entries: dict[str, str],
) -> None:
    """Match every executed trusted checker to the archive content hash."""
    for relative in _trusted_checker_paths(manifest):
        expected = entries.get(relative)
        if expected is None:
            raise ValueError(f"archive omits trusted checker: {relative}")
        archive_file = _regular_file(root, relative, f"archive checker {relative}")
        trusted_file = _regular_file(checker_root, relative, f"trusted checker {relative}")
        if digest(archive_file) != expected or digest(trusted_file) != expected:
            raise ValueError(f"trusted checker differs from archive: {relative}")


def verify_evidence(
    root: Path,
    checker_root: Path,
    kind: str,
    receipt_validator: Path | None = None,
) -> None:
    """Run trusted offline evidence checks against extracted data."""
    if kind != "evidence":
        return
    checker = checker_root / "scripts/check-release-evidence.py"
    if not checker.is_file():
        raise ValueError(f"trusted evidence checker is missing: {checker}")
    python_dir = str(Path(sys.executable).parent)
    environment = {
        "PATH": os.pathsep.join((python_dir, os.defpath)),
        "PYTHONNOUSERSITE": "1",
    }
    command = [sys.executable, str(checker), "--mode", "verify", "--root", str(root)]
    if receipt_validator is not None:
        command.extend(("--trusted-receipt-validator", str(receipt_validator)))
    subprocess.run(
        command,
        cwd=checker_root,
        env=environment,
        check=True,
    )


def verify_archive(
    archive: Path,
    checker_root: Path | None = None,
    receipt_validator: Path | None = None,
) -> None:
    """Verify sidecar, extracted content, manifest, and bundled evidence."""
    verify_sidecar(archive)
    trusted = trusted_root(checker_root)
    if receipt_validator is not None:
        receipt_validator = _regular_file(
            receipt_validator.parent,
            receipt_validator.name,
            "trusted receipt validator",
        ).resolve()
    with tempfile.TemporaryDirectory(prefix="leone-archive-verify-") as temporary:
        root = extract(archive, Path(temporary))
        entries = verify_manifest(root)
        kind = package_kind(root)
        if kind == "evidence":
            manifest = _trusted_manifest(root, trusted)
            _verify_trusted_checkers(root, trusted, manifest, entries)
        public_gate = _regular_file(
            trusted,
            "scripts/check-public-tree.sh",
            "trusted public checker",
        )
        subprocess.run(
            [str(public_gate), "--archive-kind", kind, str(root)],
            cwd=trusted,
            check=True,
        )
        verify_evidence(root, trusted, kind, receipt_validator)


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify one Leone release archive.")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--trusted-root",
        type=Path,
        help="checker directory; required when the verifier is outside a source checkout",
    )
    parser.add_argument(
        "--trusted-receipt-validator",
        type=Path,
        help="trusted CPU-only Rust validator for runtime and quality receipts",
    )
    args = parser.parse_args(arguments)
    try:
        verify_archive(
            args.archive.resolve(),
            args.trusted_root,
            args.trusted_receipt_validator,
        )
    except (OSError, tarfile.TarError, ValueError) as error:
        parser.error(str(error))
    print(f"verified {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
