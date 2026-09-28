#!/usr/bin/env python3
"""Check source and extracted release trees for public material."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_MARKDOWN_ALLOWLIST = frozenset(
    {
        "README.md",
        "CHANGELOG.md",
        "ROADMAP.md",
        "benchmarks/README.md",
        "docs/branching-service-evidence.md",
        "docs/client-workflow.md",
        "docs/concurrent-service-evidence.md",
        "docs/memory-accounting.md",
        "docs/models.md",
        "docs/openai-api.md",
        "docs/oracle.md",
        "docs/release-candidate.md",
        "docs/release-evidence.md",
        "docs/release.md",
        "docs/speculation.md",
        "packaging/EVIDENCE.md",
        "packaging/README.md",
        "receipts/INDEX.md",
        "research/prefix_attention/ORACLE_CONTRACT.md",
        "research/prefix_attention/README.md",
    }
)
ARCHIVE_MARKDOWN_ALLOWLISTS = {
    "runtime": frozenset(
        {
            "README.md",
            "docs/branching-service-evidence.md",
            "docs/client-workflow.md",
            "docs/concurrent-service-evidence.md",
            "docs/memory-accounting.md",
            "docs/models.md",
            "docs/openai-api.md",
            "docs/oracle.md",
            "docs/release-candidate.md",
            "docs/release-evidence.md",
            "docs/release.md",
            "receipts/INDEX.md",
        }
    ),
    "evidence": frozenset({"README.md", "receipts/INDEX.md"}),
}
SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".cuh",
    ".cxx",
    ".hxx",
    ".h",
    ".hh",
    ".hpp",
    ".m",
    ".mm",
    ".metal",
    ".py",
    ".rs",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".toml",
    ".yaml",
    ".yml",
}
FORBIDDEN_PATH = re.compile(
    r"(^|/)(agents|claude|plan)\.md$|(^|/)\.claude(?:/|$)|"
    r"^grok_writeup\.md$|^research/v[0-9]+\.[0-9]+-(intervention|prior-art)\.md$|"
    r"\.(gguf|f16|pem|key)$",
    re.IGNORECASE,
)
QUALITY_SAMPLE_F32_NAMES = frozenset({"oracle.f32", "llama_cpp.f32", "leone.f32"})
PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")
COMMENT = re.compile(r"(^\s*(?://|#|/\*)|(?<!:)//|/\*)")
VERSION = re.compile(
    r"(?<![A-Za-z0-9_./-])v[0-9]+\.[0-9]+(?:\.[0-9]+)?"
    r"(?:[A-Za-z][0-9A-Za-z]*)?(?:[.+-][0-9A-Za-z]+)*(?![A-Za-z0-9_.])",
    re.IGNORECASE,
)
BARE_VERSION = re.compile(
    r"\bversion\s*(?:[:=]\s*|\s+)[`'\"]?(?:[0-9]+\.){1,2}[0-9]+"
    r"(?:[A-Za-z][0-9A-Za-z]*)?(?:[.+-][0-9A-Za-z]+)*",
    re.IGNORECASE,
)
COMMIT_HASH = re.compile(
    r"\b(?:commit|revision|rev|hash|digest)\b[^\n0-9a-f]{1,16}"
    r"[0-9a-f]{7,64}(?![0-9a-f])",
    re.IGNORECASE,
)
FULL_HASH = re.compile(r"(?<![0-9a-f])[0-9a-f]{32,64}(?![0-9a-f])", re.IGNORECASE)
URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s)>\"]+", re.IGNORECASE)
BRANCH = re.compile(
    r"\bbranch(?:\s+name)?\s*[:=]\s*[`'\"]?[A-Za-z][A-Za-z0-9._/-]{2,}"
    r"|\bbranch\s+[`'\"][A-Za-z][A-Za-z0-9._/-]{2,}[`'\"]"
    r"|\bbranch\s+[A-Za-z][A-Za-z0-9._-]*/[A-Za-z0-9._/-]+",
    re.IGNORECASE,
)
MACHINE_FIELD = re.compile(
    r"\b(?:source_commit|git_commit|engine_commit|oracle_commit|subject_commit|"
    r"schema_version|(?:evidence_)?release_line|build_info|sha256|[A-Za-z0-9_]+_sha256|[A-Za-z0-9_]+_checksum|"
    r"sha-256|manifest)\b[\"'`]?\s*[:=|]",
    re.IGNORECASE,
)
TOOLCHAIN = re.compile(r"\b(?:toolchain|rustc|rustup|cargo)\b", re.IGNORECASE)
API_PATH = re.compile(r"(?:/v[0-9]+(?:\.[0-9]+)?\b|\b(?:api|protocol|endpoint)\b)", re.IGNORECASE)
PII_PATTERNS = (
    (
        "an absolute personal path",
        re.compile(r"/ho" r"me/[^/\s\"<>]+|/Us" r"ers/[^/\s\"<>]+|[A-Za-z]:\\Users\\[^\\\s\"<>]+"),
    ),
    ("an email address", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    (
        "a private key or credential token",
        re.compile(
            r"BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|"
            r"gh[pousr]_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}"
        ),
    ),
    ("an internal agent record", re.compile(r"Subagent" r" id|Grok Build" r" TUI|Produced" r" by.*Grok")),
)
NARRATIVE_PATTERNS = (
    ("a narrative version identifier", VERSION),
    ("a narrative version identifier", BARE_VERSION),
    ("a narrative commit or revision hash", COMMIT_HASH),
    ("a narrative hash", FULL_HASH),
    ("a narrative branch identifier", BRANCH),
)


@dataclass(frozen=True)
class Entry:
    path: Path
    relative: str


@dataclass(frozen=True)
class Content:
    lines: tuple[tuple[int, str], ...]
    binary: bool


def forbidden(relative: str, allowed_f32: frozenset[str] = frozenset()) -> bool:
    if relative.lower().endswith(".f32"):
        return relative not in allowed_f32
    return bool(FORBIDDEN_PATH.search(relative.lower()))


def markdown_path(relative: str) -> bool:
    return Path(relative).suffix.lower() == ".md"


def git_markdown(root: Path, others: bool = False) -> tuple[list[str], list[str]]:
    command = ["git", "ls-files", "-z"]
    if others:
        command.extend(["--others", "--exclude-standard"])
    command.extend(["--", "*.md"])
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return [], [f"cannot enumerate Markdown files: {error}"]
    return [os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw], []


def markdown_set_issues(actual: set[str], expected: frozenset[str], label: str) -> list[str]:
    issues = []
    for relative in sorted(actual - expected):
        issues.append(f"{label} contains unexpected Markdown file: {relative}")
    for relative in sorted(expected - actual):
        issues.append(f"{label} is missing required Markdown file: {relative}")
    return issues


def check_source_markdown(root: Path) -> tuple[list[str], list[str]]:
    tracked, errors = git_markdown(root)
    if errors:
        return [], errors
    issues = markdown_set_issues(set(tracked), SOURCE_MARKDOWN_ALLOWLIST, "source tree")
    untracked, untracked_errors = git_markdown(root, others=True)
    errors.extend(untracked_errors)
    issues.extend(
        f"source tree contains unexpected untracked Markdown file: {relative}"
        for relative in sorted(untracked)
    )
    return issues, errors


def symlink_component(root: Path, relative: str) -> bool:
    current = root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def source_entries(root: Path, allowed_f32: frozenset[str]) -> tuple[list[Entry], list[str]]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return [], [f"cannot enumerate source files: {error}"]

    entries = []
    errors = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        relative = os.fsdecode(raw)
        path = root / relative
        if not os.path.lexists(path):
            continue
        if symlink_component(root, relative):
            errors.append(f"{relative} (symlink)")
            continue
        if forbidden(relative, allowed_f32):
            errors.append(relative)
            continue
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(root)
            mode = path.stat().st_mode
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{relative} ({error})")
            continue
        if not stat.S_ISREG(mode):
            errors.append(f"{relative} (non-regular file)")
            continue
        entries.append(Entry(path, relative))
    return entries, errors


def walk_archive(
    directory: Path,
    prefix: str,
    entries: list[Entry],
    errors: list[str],
    allowed_f32: frozenset[str],
) -> None:
    try:
        with os.scandir(directory) as iterator:
            children = sorted(iterator, key=lambda item: item.name)
    except OSError as error:
        errors.append(f"{prefix or '.'} ({error})")
        return
    for child in children:
        relative = f"{prefix}/{child.name}" if prefix else child.name
        path = Path(child.path)
        if child.is_symlink():
            errors.append(f"{relative} (symlink)")
            continue
        if forbidden(relative, allowed_f32):
            errors.append(relative)
            continue
        if child.is_dir(follow_symlinks=False):
            walk_archive(path, relative, entries, errors, allowed_f32)
            continue
        if not child.is_file(follow_symlinks=False):
            errors.append(f"{relative} (non-regular file)")
            continue
        entries.append(Entry(path, relative))


def archive_entries(root: Path, allowed_f32: frozenset[str]) -> tuple[list[Entry], list[str]]:
    entries = []
    errors = []
    walk_archive(root, "", entries, errors, allowed_f32)
    return entries, errors


def quality_sample_f32_paths(
    root: Path, archive_scan: bool, requested_kind: Optional[str] = None
) -> frozenset[str]:
    if archive_scan and archive_kind(root, requested_kind) != "evidence":
        return frozenset()
    manifest = quality_sample_manifest(root, archive_scan)
    if manifest is None:
        return frozenset()
    destinations = manifest_destinations(manifest["files"])
    allowed = set()
    for backend in manifest["backend_requirements"]:
        if isinstance(backend, dict) and backend.get("backend") == "cuda":
            allowed.update(cuda_sample_f32_paths(backend.get("records"), destinations))
    return frozenset(allowed)


def quality_sample_manifest(root: Path, archive_scan: bool) -> Optional[dict]:
    manifest_path = root / ("release-evidence.json" if archive_scan else "packaging/release-evidence.v0.4.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("release_line") != "v0.4":
        return None
    files = manifest.get("files")
    backends = manifest.get("backend_requirements")
    if not isinstance(files, list) or not isinstance(backends, list):
        return None
    return manifest


def manifest_destinations(files: list) -> set[str]:
    return {
        entry.get("destination")
        for entry in files
        if isinstance(entry, dict) and isinstance(entry.get("destination"), str)
    }


def cuda_sample_f32_paths(records: object, destinations: set) -> set[str]:
    """Return the declared f32 paths under each CUDA quality record's own sample root."""
    if not isinstance(records, list):
        return set()
    allowed = set()
    for record in records:
        if not isinstance(record, dict) or record.get("role") != "quality":
            continue
        allowed.update(cuda_record_f32_paths(record) & destinations)
    return allowed


def cuda_record_f32_paths(record: dict) -> set[str]:
    path = record.get("path")
    artifact_roots = record.get("artifact_roots")
    artifact_files = record.get("artifact_files")
    if not (
        isinstance(path, str)
        and isinstance(artifact_roots, list)
        and isinstance(artifact_files, list)
    ):
        return set()
    sample_root = f"{path.removesuffix('.json')}-cuda-samples"
    if sample_root not in artifact_roots:
        return set()
    return {
        f"{sample_root}/{name}"
        for name in QUALITY_SAMPLE_F32_NAMES
        if f"{sample_root}/{name}" in artifact_files
    }


def archive_kind(root: Path, requested: Optional[str]) -> str:
    if requested is not None:
        return requested
    if root.name == "evidence" or root.name.endswith("-evidence"):
        return "evidence"
    return "runtime"


def check_archive_markdown(entries: list[Entry], kind: str) -> list[str]:
    actual = {entry.relative for entry in entries if markdown_path(entry.relative)}
    expected = ARCHIVE_MARKDOWN_ALLOWLISTS[kind]
    return markdown_set_issues(actual, expected, f"{kind} archive")


def read_content(entry: Entry) -> Content:
    data = entry.path.read_bytes()
    if b"\0" not in data:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return Content(tuple(enumerate(text.splitlines(), 1)), False)
    strings = tuple((0, match.group().decode("ascii")) for match in PRINTABLE.finditer(data))
    return Content(strings, True)


def metadata_line(relative: str, line: str, pattern: re.Pattern[str]) -> bool:
    lower = relative.lower()
    if lower == "changelog.md" or lower.endswith("/changelog.md"):
        return True
    if MACHINE_FIELD.search(line):
        return True
    if pattern in (VERSION, BARE_VERSION) and TOOLCHAIN.search(line):
        return True
    if pattern in (VERSION, BARE_VERSION) and API_PATH.search(line):
        return True
    return False


def python_narrative_lines(lines: tuple[tuple[int, str], ...]) -> tuple[tuple[int, str], ...]:
    result = []
    delimiter = None
    for number, line in lines:
        if delimiter is not None:
            result.append((number, line))
            if line.count(delimiter) % 2:
                delimiter = None
            continue
        if COMMENT.search(line):
            result.append((number, line))
        for marker in ('"""', "'''"):
            if marker not in line:
                continue
            result.append((number, line))
            if line.count(marker) % 2:
                delimiter = marker
            break
    return tuple(result)


def comment_narrative_lines(lines: tuple[tuple[int, str], ...]) -> tuple[tuple[int, str], ...]:
    result = []
    block = False
    for number, line in lines:
        if block:
            result.append((number, line))
            if "*/" in line:
                block = False
            continue
        if COMMENT.search(line):
            result.append((number, line))
        if "/*" in line and "*/" not in line.split("/*", 1)[1]:
            block = True
    return tuple(result)


def narrative_lines(entry: Entry, content: Content) -> tuple[tuple[int, str], ...]:
    if content.binary:
        return ()
    if entry.path.suffix.lower() == ".md":
        return content.lines
    if entry.path.suffix.lower() not in SOURCE_SUFFIXES:
        return ()
    if entry.path.suffix.lower() == ".py":
        return python_narrative_lines(content.lines)
    return comment_narrative_lines(content.lines)


def scan_pii(entry: Entry, content: Content) -> list[str]:
    issues = []
    for number, line in content.lines:
        for description, pattern in PII_PATTERNS:
            if pattern.search(line):
                issues.append(f"{entry.relative}:{number or 'strings'}: {description}")
    return issues


def scan_narrative(entry: Entry, content: Content) -> list[str]:
    issues = []
    for number, line in narrative_lines(entry, content):
        candidate = URL.sub("", line)
        for description, pattern in NARRATIVE_PATTERNS:
            if metadata_line(entry.relative, line, pattern):
                continue
            if pattern.search(candidate):
                issues.append(f"{entry.relative}:{number or 'strings'}: {description}")
    return issues


def scan_entry(entry: Entry) -> tuple[list[str], list[str]]:
    try:
        content = read_content(entry)
    except OSError as error:
        return [], [f"{entry.relative} ({error})"]
    return scan_pii(entry, content) + scan_narrative(entry, content), []


def scan(entries: list[Entry]) -> tuple[list[str], list[str]]:
    issues = []
    errors = []
    for entry in entries:
        entry_issues, entry_errors = scan_entry(entry)
        issues.extend(entry_issues)
        errors.extend(entry_errors)
    return issues, errors


def scan_root(argument: Optional[str]) -> Optional[tuple[Path, bool]]:
    candidate = REPO_ROOT if argument is None else Path(argument)
    try:
        root = candidate.resolve(strict=True)
    except OSError as error:
        print(f"scan directory does not exist: {candidate} ({error})", file=sys.stderr)
        return None
    if not root.is_dir():
        print(f"scan path is not a directory: {candidate}", file=sys.stderr)
        return None
    return root, argument is not None


def report_errors(errors: list[str], message: str) -> bool:
    if not errors:
        return False
    for error in errors:
        print(error, file=sys.stderr)
    print(message, file=sys.stderr)
    return True


def report_scan(
    entries: list[Entry],
    path_errors: list[str],
    markdown_issues: list[str],
    archive_scan: bool,
) -> int:
    path_message = (
        "extracted tree contains a forbidden path or unsafe file"
        if archive_scan
        else "public tree contains a private-work, model, logit, key, or unsafe file"
    )
    if report_errors(path_errors, path_message):
        return 1
    if report_errors(markdown_issues, "public tree Markdown set does not match its allowlist"):
        return 1
    issues, read_errors = scan(entries)
    if report_errors(read_errors, "public tree contains an unreadable file"):
        return 1
    if issues:
        for issue in issues:
            print(issue, file=sys.stderr)
        print("public tree contains private material or narrative metadata", file=sys.stderr)
        return 1
    print("public tree hygiene gate passed")
    return 0


def parse_arguments(arguments: list[str]) -> Optional[tuple[Optional[str], Optional[str]]]:
    if not arguments:
        return None, None
    if len(arguments) == 1:
        return arguments[0], None
    if len(arguments) == 3 and arguments[0] == "--archive-kind":
        return arguments[2], arguments[1]
    print(
        f"usage: {Path(sys.argv[0]).name} [--archive-kind runtime|evidence] [DIRECTORY]",
        file=sys.stderr,
    )
    return None


def markdown_check(
    root: Path,
    entries: list[Entry],
    archive_scan: bool,
    requested_kind: Optional[str],
) -> tuple[list[str], list[str], int]:
    if not archive_scan:
        issues, errors = check_source_markdown(root)
        return issues, errors, 0
    kind = archive_kind(root, requested_kind)
    if kind not in ARCHIVE_MARKDOWN_ALLOWLISTS:
        print(
            f"unknown archive kind: {kind}",
            file=sys.stderr,
        )
        return [], [], 2
    return check_archive_markdown(entries, kind), [], 0


def main(arguments: list[str]) -> int:
    parsed = parse_arguments(arguments)
    if parsed is None:
        return 2
    argument, requested_kind = parsed
    selected = scan_root(argument)
    if selected is None:
        return 2
    root, archive_scan = selected
    allowed_f32 = quality_sample_f32_paths(root, archive_scan, requested_kind)
    entries, path_errors = (
        archive_entries(root, allowed_f32)
        if archive_scan
        else source_entries(root, allowed_f32)
    )
    markdown_issues, markdown_errors, markdown_status = markdown_check(
        root, entries, archive_scan, requested_kind
    )
    path_errors.extend(markdown_errors)
    if markdown_status:
        return markdown_status
    return report_scan(entries, path_errors, markdown_issues, archive_scan)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
