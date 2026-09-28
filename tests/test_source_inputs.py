"""Check source manifests independently of Git ancestry."""

import importlib.util
import json
from pathlib import Path
import shutil
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

    def test_v04_record_includes_packaged_checker_inputs(self):
        inputs = set(SOURCE.V04_EXECUTION_INPUTS)
        source_inputs = set(SOURCE.INPUTS) - set(SOURCE.V04_WORKLOAD_INPUTS)
        self.assertTrue(source_inputs.issubset(inputs))
        self.assertNotIn("receipts/v04-linux-cuda-quality-qwen3.json", inputs)
        self.assertIn(
            "receipts/v04-linux-cuda-quality-qwen3.json",
            set(SOURCE.V04_EVIDENCE_INPUTS),
        )
        self.assertTrue(
            {
                "build-support",
                "scripts/check-public-tree.py",
                "scripts/check-public-tree.sh",
                "scripts/check-release-evidence.py",
                "scripts/release_evidence_manifest.py",
                "scripts/release_evidence_validators.py",
                "scripts/validate-quality-stage.py",
                "scripts/source_inputs.py",
                "scripts/write-cuda-quality-comparison.py",
                "scripts/freeze-cuda-generation.py",
                "scripts/write-metal-quality-generation.py",
                "scripts/generate-openai-chat-template-fixtures.py",
            }.issubset(inputs)
        )
        self.assertTrue(
            all(
                required in inputs
                or any(required.startswith(item + "/") for item in inputs)
                for required in SOURCE.V04_REQUIRED_FILES
            )
        )

    def test_v04_required_and_declared_helpers_exist(self):
        repository = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (repository / "packaging/release-evidence.v0.4.json").read_text()
        )
        declared = {
            entry["source"]
            for entry in manifest["files"]
            if entry["source"].startswith(("scripts/", "packaging/"))
        }
        missing = sorted(
            path
            for path in set(SOURCE.V04_REQUIRED_FILES) | declared | set(manifest["trusted_validators"])
            if not (repository / path).exists()
        )
        self.assertEqual(missing, [])

    def test_v04_optional_records_keep_generated_evidence_out_of_execution_set(self):
        source_record = {
            "source_commit": self.revision,
            "files": self.manifest["files"].copy(),
        }
        source_record["workload_files"] = {}
        evidence = self.root / "receipts/v04-linux-cuda-quality-qwen3.json"
        evidence.write_text("evidence\n")
        SOURCE.git(self.root, "add", str(evidence.relative_to(self.root)))
        source_record["evidence_files"] = SOURCE._snapshot_current(
            self.root, SOURCE.V04_EVIDENCE_INPUTS
        )
        manifest = self.root / "receipts/source-inputs-v04.json"
        manifest.write_text(json.dumps(source_record))
        SOURCE.check(self.root, source_record["source_commit"], manifest.relative_to(self.root))
        relative = evidence.relative_to(self.root).as_posix()
        self.assertNotIn(relative, source_record["files"])
        self.assertIn(relative, source_record["evidence_files"])

    def test_matches_without_recorded_history(self):
        SOURCE.check(self.root, self.revision)

    def test_v04_manifest_uses_v04_input_set(self):
        path = self.root / "receipts/source-inputs-v04.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        SOURCE.check(self.root, self.revision, Path("receipts/source-inputs-v04.json"))

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

    def test_rejects_symlinked_parent_input(self):
        source_dir = self.root / "crates"
        source_dir.mkdir()
        tracked = source_dir / "input.txt"
        tracked.write_text("fixture\n")
        SOURCE.git(self.root, "add", "crates/input.txt")
        self.manifest["files"]["crates/input.txt"] = {
            "sha256": SOURCE.digest(tracked.read_bytes()),
            "executable": False,
        }
        self.write_manifest()
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "input.txt").write_text("fixture\n")
        shutil.rmtree(source_dir)
        source_dir.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink parent"):
            SOURCE.check(self.root, self.revision)
        with self.assertRaisesRegex(ValueError, "symlink parent"):
            SOURCE._source_file(self.root, "crates/input.txt")

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
