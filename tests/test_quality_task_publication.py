"""Check bounded artifact hashing and exclusive quality-task publication."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("quality_task", ROOT / "scripts/write-quality-task.py")
TASK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TASK)


class QualityTaskPublicationTests(unittest.TestCase):
    def test_artifact_hashing_does_not_read_a_whole_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            data = bytes(range(256)) * 8201 + b"tail"
            path.write_bytes(data)
            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole file read")):
                record = TASK.tokenizer_artifacts(path, path)
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(record["model"], {"name": path.name, "sha256": expected})
            self.assertEqual(record["executable"]["sha256"], expected)
            self.assertEqual(record["executable"]["bytes"], len(data))

    def test_publication_keeps_existing_file_directory_and_symlink(self):
        for kind in ("file", "directory", "symlink", "dangling_symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "task.json"
                target = root / "target.json"
                self.make_collision(kind, output, target)
                with self.assertRaises(FileExistsError):
                    TASK.publish_task(output, {"new": True})
                self.assertEqual(sorted(item.name for item in root.iterdir()),
                                 ["target.json", "task.json"] if target.exists() else ["task.json"])
                if kind == "file":
                    self.assertEqual(output.read_bytes(), b"winner")
                if kind == "symlink":
                    self.assertEqual(target.read_bytes(), b"winner")
                if kind.endswith("symlink"):
                    self.assertTrue(output.is_symlink())

    @staticmethod
    def make_collision(kind, output, target):
        if kind == "file":
            output.write_bytes(b"winner")
        elif kind == "directory":
            output.mkdir()
        else:
            if kind == "symlink":
                target.write_bytes(b"winner")
            output.symlink_to(target)

    def test_dangling_output_is_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task.json"
            output.symlink_to(Path(directory) / "absent.json")
            arguments = argparse.Namespace(oracle_stage=directory, output=str(output))
            with mock.patch.object(TASK, "load", side_effect=AssertionError("input read")):
                with self.assertRaisesRegex(ValueError, "refusing to replace"):
                    TASK.write_task(arguments)
            self.assertTrue(output.is_symlink())
            self.assertFalse(output.exists())

    def test_publication_preserves_canonical_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "task.json"
            record = {"z": 3, "a": {"token_ids": [1, 2]}}
            TASK.publish_task(output, record)
            self.assertEqual(output.read_text(), json.dumps(record, indent=2, sort_keys=True) + "\n")
            self.assertEqual(list(Path(directory).iterdir()), [output])


if __name__ == "__main__":
    unittest.main()
