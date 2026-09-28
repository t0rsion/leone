"""Test the offline source closure of the concurrent-service checker.

An evidence archive omits the source crates. Offline mode checks each recorded
file that the archive holds and requires the checker's own files. Repository
mode still checks the whole source with `source_inputs.py`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "study_concurrent_for_offline_source", ROOT / "scripts/study-concurrent-service.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)
COMMIT = "a" * 40
MANIFEST = Path("receipts/source-inputs-v04.json")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class OfflineSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in HARNESS.OFFLINE_REQUIRED_SOURCE_FILES:
            self.put(name, name.encode())
        self.put("corpus/quality.txt", b"corpus", executable=False)
        self.body = {
            "schema_version": "leone.source-inputs.v2",
            "source_commit": COMMIT,
            "files": {
                **{name: self.record(name) for name in HARNESS.OFFLINE_REQUIRED_SOURCE_FILES},
                "crates/leone/src/lib.rs": {"sha256": digest(b"crate"), "executable": False},
                "Cargo.toml": {"sha256": digest(b"cargo"), "executable": False},
            },
            "workload_files": {"corpus/quality.txt": self.record("corpus/quality.txt")},
            "evidence_files": {},
        }
        self.write()

    def put(self, name: str, data: bytes, executable: bool = True) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(0o755 if executable else 0o644)

    def record(self, name: str) -> dict:
        path = self.root / name
        return {"sha256": digest(path.read_bytes()), "executable": bool(path.stat().st_mode & 0o111)}

    def write(self) -> None:
        path = self.root / MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.body), encoding="utf-8")

    def errors(self) -> list[str]:
        return HARNESS._validate_offline_source(COMMIT, self.root, MANIFEST)

    def test_an_archive_without_the_source_crates_passes(self):
        self.assertFalse((self.root / "crates").exists())
        self.assertEqual(self.errors(), [])

    def test_each_recorded_file_the_archive_holds_must_match(self):
        for name in ("scripts/source_inputs.py", "corpus/quality.txt"):
            with self.subTest(name):
                self.setUp()
                self.put(name, b"changed", executable=name.startswith("scripts/"))
                self.assertTrue(any("changed" in error for error in self.errors()), name)

    def test_a_changed_file_mode_is_rejected(self):
        (self.root / "corpus/quality.txt").chmod(0o755)
        self.assertTrue(any("mode changed" in error for error in self.errors()))

    def test_a_recorded_evidence_file_the_archive_holds_must_match(self):
        self.put("receipts/v04-x.json", b"evidence", executable=False)
        self.body["evidence_files"] = {"receipts/v04-x.json": self.record("receipts/v04-x.json")}
        self.write()
        self.assertEqual(self.errors(), [])
        self.put("receipts/v04-x.json", b"tampered", executable=False)
        self.assertTrue(any("changed" in error for error in self.errors()))

    def test_the_checker_files_are_required(self):
        for name in HARNESS.OFFLINE_REQUIRED_SOURCE_FILES:
            with self.subTest(name, case="absent"):
                self.setUp()
                (self.root / name).unlink()
                self.assertTrue(any("not a regular file" in error for error in self.errors()))
            with self.subTest(name, case="unrecorded"):
                self.setUp()
                del self.body["files"][name]
                self.write()
                self.assertTrue(any("omits required files" in error for error in self.errors()))

    def test_a_symlinked_file_is_rejected(self):
        (self.root / "corpus/quality.txt").unlink()
        (self.root / "corpus/quality.txt").symlink_to(self.root / "scripts/source_inputs.py")
        self.assertTrue(self.errors())

    def test_malformed_groups_and_records_are_rejected(self):
        cases = (
            ("no files", lambda body: body.update(files={})),
            ("files not an object", lambda body: body.update(files=[])),
            ("workload not an object", lambda body: body.update(workload_files=[])),
            ("bad digest", lambda body: body["files"]["Cargo.toml"].update(sha256="x")),
            ("bad mode", lambda body: body["files"]["Cargo.toml"].update(executable="yes")),
            ("unsafe name", lambda body: body["files"].update({"../escape": body["files"]["Cargo.toml"]})),
        )
        for name, mutate in cases:
            with self.subTest(name):
                self.setUp()
                mutate(self.body)
                self.write()
                self.assertTrue(self.errors())

    def test_the_commit_still_binds_the_manifest(self):
        self.body["source_commit"] = "b" * 40
        self.write()
        self.assertEqual(
            self.errors(), ["offline source inputs do not match recorded source commit"]
        )

    def test_repository_mode_still_checks_the_whole_source(self):
        commands = []

        def run(command, timeout_s):
            commands.append(command)
            return 0, "", ""

        source = {"commit": COMMIT}
        with mock.patch.object(HARNESS, "_run_command", side_effect=run):
            self.assertEqual(HARNESS._validate_source(source, self.root, False, MANIFEST), [])
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][2:4], ["check", COMMIT])
        self.assertIn(str(MANIFEST), commands[0])


if __name__ == "__main__":
    unittest.main()
