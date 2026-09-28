"""Test that the quality records name the files the canonical producers write.

CUDA writes a generation record next to its artifacts. The trusted external copy
is a release input. Metal writes the embedded shader, the doctor output, and the
eval output into its native stage. A planned manifest lists them without hashes.
A complete manifest must fail when any is missing.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_release_packaging as PACKAGING  # noqa: E402

EVIDENCE = PACKAGING.EVIDENCE
DISPATCH = PACKAGING.DISPATCH
NATIVE = ("leone.metal", "leone-doctor.txt", "leone-eval.stdout")


def planned() -> dict:
    return copy.deepcopy(EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json"))


def quality_records(manifest: dict, backend: str) -> list[dict]:
    return [
        record
        for entry in manifest["backend_requirements"]
        if entry["backend"] == backend
        for record in entry["records"]
        if record["role"] == "quality"
    ]


class PlannedQualityFilesTests(unittest.TestCase):
    def setUp(self):
        self.manifest = planned()
        self.destinations = {item["destination"] for item in self.manifest["files"]}

    def test_cuda_records_name_the_generation_tuple(self):
        records = quality_records(self.manifest, "cuda")
        self.assertEqual(len(records), 2)
        for record in records:
            with self.subTest(record["path"]):
                self.assertEqual(record["statistics_platform"], "linux-x86_64")
                generation = record["generation_record"]
                self.assertIn(generation, record["dependencies"])
                self.assertIn(generation, self.destinations)
                snapshot = record["artifact_root"] + "/generation-record.json"
                self.assertIn(snapshot, record["artifact_files"])
                self.assertIn(snapshot, self.destinations)

    def test_planned_records_carry_no_collected_hashes(self):
        for record in quality_records(self.manifest, "cuda"):
            for key in ("generation_record_sha256", "task_manifest_sha256", "sha256"):
                self.assertNotIn(key, record)

    def test_metal_records_name_the_native_stage_files(self):
        records = quality_records(self.manifest, "metal")
        self.assertEqual(len(records), 2)
        for record in records:
            with self.subTest(record["path"]):
                stage = next(root for root in record["artifact_roots"] if root.endswith("-metal-stage"))
                for name in NATIVE:
                    self.assertIn(f"{stage}/{name}", record["artifact_files"])
                    self.assertIn(f"{stage}/{name}", self.destinations)

    def test_the_planned_manifest_is_structurally_valid(self):
        EVIDENCE.validate(self.manifest)

    def test_generation_records_are_release_inputs_of_the_source_manifest(self):
        source_inputs = importlib.util.spec_from_file_location(
            "source_inputs_for_quality_files", ROOT / "scripts/source_inputs.py"
        )
        module = importlib.util.module_from_spec(source_inputs)
        source_inputs.loader.exec_module(module)
        self.assertIn("receipts/v04-*", module.V04_EVIDENCE_INPUTS)
        for record in quality_records(self.manifest, "cuda"):
            self.assertTrue(record["generation_record"].startswith("receipts/v04-"))


class StructuralTests(unittest.TestCase):
    def complete(self, root: Path) -> dict:
        path = PACKAGING.ReleasePackagingTests._complete_v04_skeleton(root)
        return json.loads(path.read_text(encoding="utf-8"))

    def test_a_complete_manifest_needs_the_packaged_generation_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.complete(Path(temporary))
            EVIDENCE.validate(manifest, require_complete=True)
            for record in quality_records(manifest, "cuda"):
                candidate = copy.deepcopy(manifest)
                target = next(
                    item for item in quality_records(candidate, "cuda") if item["path"] == record["path"]
                )
                target["artifact_files"].remove(record["artifact_root"] + "/generation-record.json")
                with self.assertRaisesRegex(ValueError, "does not package its generation record"):
                    EVIDENCE.validate(candidate, require_complete=True)

    def test_a_complete_manifest_needs_each_native_metal_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.complete(Path(temporary))
            for index, record in enumerate(quality_records(manifest, "metal")):
                stage = next(root for root in record["artifact_roots"] if root.endswith("-metal-stage"))
                for name in NATIVE:
                    with self.subTest(record["path"], name=name):
                        candidate = copy.deepcopy(manifest)
                        target = quality_records(candidate, "metal")[index]
                        target["artifact_files"].remove(f"{stage}/{name}")
                        with self.assertRaisesRegex(ValueError, "native artifacts"):
                            EVIDENCE.validate(candidate, require_complete=True)

    def test_a_planned_manifest_may_omit_the_generation_hash_but_not_the_snapshot(self):
        manifest = planned()
        record = quality_records(manifest, "cuda")[0]
        record["artifact_files"].remove(record["artifact_root"] + "/generation-record.json")
        with self.assertRaisesRegex(ValueError, "does not package its generation record"):
            EVIDENCE.validate(manifest)


class DispatchTests(unittest.TestCase):
    def test_a_missing_recorded_quality_artifact_fails_a_complete_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, verifier = PACKAGING.ReleasePackagingTests._complete_dispatch_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = PACKAGING.ReleasePackagingTests._stub_scope(
                EVIDENCE.validate(manifest, require_complete=True)
            )
            DISPATCH.validate_records(root, manifest, records, root, verifier)
            for backend in ("cuda", "metal"):
                record = quality_records(manifest, backend)[0]
                for relative in record["artifact_files"]:
                    if not relative.endswith(("generation-record.json", *NATIVE)):
                        continue
                    with self.subTest(relative):
                        path = root / relative
                        data = path.read_bytes()
                        path.unlink()
                        try:
                            with self.assertRaisesRegex(ValueError, "artifact .* is missing"):
                                DISPATCH.validate_records(root, manifest, records, root, verifier)
                        finally:
                            path.write_bytes(data)

    def test_the_native_shader_is_referenced_by_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "receipts/stage-metal-stage"
            stage.mkdir(parents=True)
            body = {
                "native": {
                    "manifest": {
                        "body": {
                            "execution": {
                                "shader": {"name": "leone.metal", "sha256": "0" * 64, "bytes": 1},
                                "machine": {"path": "leone-doctor.txt"},
                                "eval_stdout": {"path": "leone-eval.stdout"},
                            }
                        }
                    }
                }
            }
            (stage / "metal-stage.json").write_text(json.dumps(body), encoding="utf-8")
            comparison = root / "receipts/comparison.json"
            comparison.write_text(
                json.dumps({"stage": {"path": "stage-metal-stage/metal-stage.json"}}),
                encoding="utf-8",
            )
            files = {f"receipts/stage-metal-stage/{name}" for name in ("metal-stage.json", *NATIVE)}
            references = DISPATCH._record_artifact_references(
                root,
                files,
                [PurePosixPath("receipts/stage-metal-stage")],
                PurePosixPath("receipts/comparison.json"),
            )
            self.assertEqual(references, files)

    def test_an_unreferenced_native_file_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "receipts/stage-metal-stage"
            stage.mkdir(parents=True)
            (stage / "metal-stage.json").write_text("{}", encoding="utf-8")
            comparison = root / "receipts/comparison.json"
            comparison.write_text(
                json.dumps({"stage": {"path": "stage-metal-stage/metal-stage.json"}}),
                encoding="utf-8",
            )
            files = {"receipts/stage-metal-stage/metal-stage.json", "receipts/stage-metal-stage/leone.metal"}
            references = DISPATCH._record_artifact_references(
                root,
                files,
                [PurePosixPath("receipts/stage-metal-stage")],
                PurePosixPath("receipts/comparison.json"),
            )
            self.assertNotIn("receipts/stage-metal-stage/leone.metal", references)


if __name__ == "__main__":
    unittest.main()
