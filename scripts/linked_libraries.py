#!/usr/bin/env python3
"""Record resolved shared libraries by content hash.

Linkage is resolved from `ldd` (Linux) or `otool -L` (Darwin) plus the backend
libraries in the library directory. It is not read from a running process.
`otool -L` lists direct dependencies only, so on Darwin the closure follows each
resolved library in turn.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CORE_FAMILIES = ("libllama", "libggml", "libggml-base", "libggml-cpu")
BACKEND_SUFFIX = {"mtl": "metal", "metal": "metal", "cuda": "cuda"}
SYSTEM_PREFIXES = ("/usr/lib/", "/System/")
LIBRARY_NAME = re.compile(r"^(lib[A-Za-z0-9_+-]+?)(?:\.\d+)*\.(?:so(?:\.\d+)*|dylib)$")


class LinkageError(Exception):
    """Raised when a dependency is unresolved or missing."""


def file_hash(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def family(name):
    """Return the library family for a file name: libggml-cuda.so.0.21.0 gives libggml-cuda."""

    match = LIBRARY_NAME.match(name)
    return match.group(1) if match else None


def backend_family(backend):
    suffix = BACKEND_SUFFIX.get(str(backend).lower())
    return f"libggml-{suffix}" if suffix else None


def core_families(backend):
    """Return the core library families for one backend."""

    active = backend_family(backend)
    return CORE_FAMILIES + ((active,) if active else ())


def _valid_digest(value):
    """Return whether a value is a lowercase SHA-256 digest."""

    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _core_entries(records, backend):
    """Yield core library families and their recorded digests."""

    wanted = set(core_families(backend))
    for record in records:
        name = record.get("name", record.get("path"))
        library = family(os.path.basename(name)) if isinstance(name, str) else None
        if library in wanted:
            yield library, record.get("sha256")


def core_hashes(records, backend):
    """Map each core family to the set of hashes recorded for it.

    Entries use `name` (current) or `path` (older records, base name only).
    Invalid digests are omitted and reported by `invalid_core_families`.
    """

    found = {}
    for library, digest in _core_entries(records, backend):
        if _valid_digest(digest):
            found.setdefault(library, set()).add(digest)
    return found


def invalid_core_families(records, backend):
    """Return core library families with malformed recorded digests."""

    return {library for library, digest in _core_entries(records, backend) if not _valid_digest(digest)}


def linux_references(binary, run):
    lines = run(["ldd", str(binary)], text=True).splitlines()
    if any("not found" in line for line in lines):
        raise LinkageError("oracle has an unresolved shared library")
    references = []
    for line in lines:
        line = line.strip()
        if " => " in line:
            references.append(line.split(" => ", 1)[1].split(" (", 1)[0])
        elif line.startswith("/"):
            references.append(line.split(" (", 1)[0])
    return references


def darwin_references(binary, run):
    lines = run(["otool", "-L", str(binary)], text=True).splitlines()[1:]
    return [line.strip().split(" (", 1)[0] for line in lines]


def resolve_reference(reference, owner, library_dir):
    if reference.startswith("@loader_path/"):
        return owner.parent / reference.split("/", 1)[1]
    if reference.startswith("@rpath/"):
        return library_dir / reference.split("/", 1)[1]
    path = Path(reference)
    return path if path.is_absolute() else library_dir / os.path.basename(reference)


def is_system_only(reference, path):
    return reference.startswith(SYSTEM_PREFIXES) and not path.is_file()


def linux_closure(binary, library_dir, run):
    return [resolve_reference(item, binary, library_dir) for item in sorted(set(linux_references(binary, run)))]


def darwin_closure(binary, library_dir, run):
    """Follow `otool -L` from the binary through every resolved non-system library."""

    seen, queue, paths = set(), [binary], []
    while queue:
        owner = queue.pop()
        for reference in sorted(set(darwin_references(owner, run))):
            path = resolve_reference(reference, owner, library_dir)
            if path in seen or is_system_only(reference, path) or path == owner:
                continue
            seen.add(path)
            paths.append(path)
            queue.append(path)
    return paths


def resolve(binary, library_dir, backend, platform=None, run=subprocess.check_output):
    """Return sorted {name, sha256} records for the resolved closure and the active backend."""

    binary, library_dir = Path(binary).resolve(), Path(library_dir).resolve()
    closure = darwin_closure if (platform or sys.platform) == "darwin" else linux_closure
    records, names = [], set()
    for path in closure(binary, library_dir, run):
        if not path.is_file():
            raise LinkageError(f"oracle dependency is not resolvable: {path}")
        records.append({"name": path.name, "sha256": file_hash(path)})
        names.add(path.name)
    active = BACKEND_SUFFIX.get(str(backend).lower())
    for path in sorted(library_dir.glob(f"libggml-{active}.*") if active else ()):
        if path.is_file() and path.name not in names:
            records.append({"name": path.name, "sha256": file_hash(path)})
    return sorted(records, key=lambda item: item["name"])


def main(arguments):
    if len(arguments) != 3:
        raise SystemExit("usage: linked_libraries.py BINARY LIBRARY_DIR BACKEND")
    try:
        print(json.dumps(resolve(*arguments)))
    except LinkageError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main(sys.argv[1:])
