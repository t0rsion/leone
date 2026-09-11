"""Check source manifests independently of Git ancestry."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "source_inputs", Path(__file__).resolve().parents[1] / "scripts/source_inputs.py"
)
SOURCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SOURCE)


class SourceInputsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        self.source = self.root / "Cargo.toml"
        self.source.write_text("fixture\n")
        SOURCE.git(self.root, "add", "Cargo.toml")
        self.revision = "a" * 40
        self.manifest = {"source_commit": self.revision, "files": {
            "Cargo.toml": {"sha256": SOURCE.digest(self.source.read_bytes()), "executable": False},
        }}
        (self.root / "receipts").mkdir()
        self.write_manifest()

    def write_manifest(self):
        (self.root / SOURCE.MANIFEST).write_text(json.dumps(self.manifest))

    def test_matches_without_recorded_history(self):
        SOURCE.check(self.root, self.revision)

    def test_rejects_changed_input(self):
        self.source.write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_added_input(self):
        (self.root / "Cargo.lock").write_text("new\n")
        with self.assertRaisesRegex(ValueError, "file set"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_deleted_input(self):
        self.source.unlink()
        with self.assertRaisesRegex(ValueError, "regular file"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_symlink_input(self):
        other = self.root / "copy"
        other.write_bytes(self.source.read_bytes())
        self.source.unlink()
        self.source.symlink_to(other)
        with self.assertRaisesRegex(ValueError, "regular file"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_incomplete_manifest(self):
        self.manifest["files"] = {}
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "no files"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_unknown_source(self):
        with self.assertRaises(subprocess.CalledProcessError):
            SOURCE.check(self.root, "b" * 40)

    def test_rejects_executable_mode_change(self):
        self.source.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "executable mode"):
            SOURCE.check(self.root, self.revision)

    def test_rejects_added_build_configuration(self):
        (self.root / ".cargo").mkdir()
        (self.root / ".cargo/config.toml").write_text("[build]\n")
        with self.assertRaisesRegex(ValueError, "file set"):
            SOURCE.check(self.root, self.revision)
