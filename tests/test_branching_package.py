"""Test the branching study records of the release evidence manifest.

The planned manifest names both backends' branching studies. Each binds one
canonical quality record by path, backend, model, and (in a package) digest. The
structural checks here need no study bytes. `test_branching_dispatch` runs the
dispatcher over real records.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_release_packaging as PACKAGING  # noqa: E402

EVIDENCE = PACKAGING.EVIDENCE
DISPATCH = PACKAGING.DISPATCH
CUDA = "receipts/v04-linux-cuda-branching-service.json"
METAL = "receipts/v04-darwin-metal-branching-service.json"
QUALITY = {
    CUDA: "receipts/v04-linux-cuda-quality-comparison-qwen3.json",
    METAL: "receipts/v04-darwin-metal-quality-comparison-qwen3.json",
}


def planned() -> dict:
    return copy.deepcopy(EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json"))


def find(manifest: dict, path: str) -> dict:
    return next(
        record
        for entry in manifest["backend_requirements"]
        for record in entry["records"]
        if record["path"] == path
    )


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlannedManifestTests(unittest.TestCase):
    def test_both_backends_declare_one_branching_study(self):
        manifest = planned()
        records = EVIDENCE.validate(manifest)
        branching = [record for record in records if record["role"] == "branching"]
        self.assertEqual({record["path"] for record in branching}, {CUDA, METAL})
        self.assertEqual({record["backend"] for record in branching}, {"cuda", "metal"})
        for record in branching:
            self.assertEqual(record["validator"], "branching-service-v1")
            self.assertEqual(record["quality_record"], QUALITY[record["path"]])
            self.assertNotIn("sha256", record)
            self.assertNotIn("binary_sha256", record)

    def test_no_quality_calibration_record_exists(self):
        text = (ROOT / "packaging/release-evidence.v0.4.json").read_text(encoding="utf-8")
        for name in ("quality-calibration", "quality_calibration", "quality_tolerances"):
            self.assertNotIn(name, text)

    def test_the_quality_records_are_the_two_existing_comparison_records(self):
        manifest = planned()
        for path, quality in QUALITY.items():
            linked = find(manifest, quality)
            self.assertEqual(linked["validator"], "quality-stage-v1")
            self.assertEqual(linked["validator_mode"], "comparison")
            self.assertEqual(find(manifest, path)["quality_record"], quality)

    def test_the_quality_sidecars_are_dependencies(self):
        manifest = planned()
        for path, quality in QUALITY.items():
            base = quality.removesuffix(".json")
            record = find(manifest, path)
            for name in EVIDENCE.V04_QUALITY_SIDECARS["cuda" if path == CUDA else "metal"]:
                self.assertIn(f"{base}-{name}.json", record["dependencies"])

    def test_the_harness_and_its_closure_are_trusted_and_packaged(self):
        manifest = planned()
        self.assertIn("scripts/study-branching-service.py", manifest["trusted_validators"])
        destinations = {entry["destination"] for entry in manifest["files"]}
        for path in DISPATCH.BRANCHING_ARCHIVE_INPUTS:
            self.assertIn(path, destinations)
        for record in (find(manifest, CUDA), find(manifest, METAL)):
            for path in DISPATCH.BRANCHING_ARCHIVE_INPUTS:
                self.assertIn(path, record["dependencies"])


class StructuralRejectionTests(unittest.TestCase):
    def reject(self, mutate, message: str, path: str = CUDA, complete: bool = False):
        manifest = planned()
        mutate(find(manifest, path), manifest)
        with self.assertRaisesRegex(ValueError, message):
            EVIDENCE.validate(manifest, require_complete=complete)

    def test_a_branching_record_needs_its_quality_record(self):
        self.reject(lambda record, _: record.update(quality_record=QUALITY[METAL]), "must bind")
        self.reject(lambda record, _: record.pop("quality_record"), "must bind", METAL)

    def test_a_dependency_may_not_be_dropped(self):
        for path in (CUDA, METAL):
            record = find(planned(), path)
            for name in list(record["dependencies"]):
                if name in (path, QUALITY[path]):
                    continue
                with self.subTest(path=path, dependency=name):
                    self.reject(
                        lambda item, _, name=name: item["dependencies"].remove(name),
                        "omits dependencies|does not depend|not declared|unreviewed",
                        path,
                    )

    def test_history_files_are_required_and_distinct(self):
        self.reject(lambda record, _: record.pop("history_result"), "no study or history files")
        self.reject(
            lambda record, _: record.update(history_result=record["history_expected"]),
            "repeats a study or history file",
            METAL,
        )

    def test_a_sidecar_may_not_be_dropped(self):
        for path in (CUDA, METAL):
            base = QUALITY[path].removesuffix(".json")
            backend = "cuda" if path == CUDA else "metal"
            for name in EVIDENCE.V04_QUALITY_SIDECARS[backend]:
                sidecar = f"{base}-{name}.json"
                with self.subTest(sidecar=sidecar):
                    self.reject(
                        lambda item, _, sidecar=sidecar: item["dependencies"].remove(sidecar),
                        "omits dependencies",
                        path,
                    )

    def test_another_schema_or_path_is_unreviewed(self):
        self.reject(lambda record, _: record.update(schema_version="leone.branching-service.v0"), "unreviewed")
        self.reject(
            lambda record, _: record.update(path="receipts/v04-linux-cuda-branching-extra.json"),
            "unreviewed|not declared|files",
        )

    def test_a_backend_link_mismatch_is_rejected(self):
        def wrong_model(record, manifest):
            find(manifest, QUALITY[CUDA])["model_sha256"] = "0" * 64

        self.reject(wrong_model, "unreviewed model|no matching quality comparison")
        self.reject(
            lambda record, manifest: find(manifest, QUALITY[METAL]).update(validator_mode="export"),
            "quality|comparison|artifact",
            METAL,
        )

    def test_a_complete_record_needs_the_same_backend_client_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = PACKAGING.ReleasePackagingTests._complete_v04_skeleton(Path(temporary))
            complete = json.loads(manifest_path.read_text(encoding="utf-8"))
        EVIDENCE.validate(complete, require_complete=True)
        for path in (CUDA, METAL):
            backend = "cuda" if path == CUDA else "metal"
            for label, mutate, message in (
                ("missing", lambda record: record.pop("binary_sha256"), "no native binary SHA-256"),
                (
                    "other",
                    lambda record: record.update(binary_sha256="0" * 64),
                    f"differs from the {backend} client binary",
                ),
            ):
                with self.subTest(path=path, case=label):
                    candidate = copy.deepcopy(complete)
                    mutate(find(candidate, path))
                    with self.assertRaisesRegex(ValueError, message):
                        EVIDENCE.validate(candidate, require_complete=True)


class SourceClosureTests(unittest.TestCase):
    def test_the_dispatcher_and_the_harness_name_the_same_seven_archive_inputs(self):
        harness = load_module("branching_harness", "scripts/study-branching-service.py")
        self.assertEqual(len(harness.ARCHIVE_REQUIRED_SOURCES), 7)
        self.assertEqual(
            tuple(harness.ARCHIVE_REQUIRED_SOURCES), tuple(DISPATCH.BRANCHING_ARCHIVE_INPUTS)
        )

    def test_the_source_manifest_records_the_new_study_files(self):
        source = load_module("branching_source_inputs", "scripts/source_inputs.py")
        for name in (
            "scripts/study-branching-service.py",
            "scripts/study-branching-service.sh",
            "scripts/freeze-branching-manifest.py",
            "scripts/linked_libraries.py",
            "scripts/run-llama-oracle.sh",
            "scripts/fetch-llama-cpp.sh",
            *DISPATCH.BRANCHING_ARCHIVE_INPUTS,
        ):
            with self.subTest(name=name):
                self.assertTrue(covered(name, source.V04_EXECUTION_INPUTS))
                self.assertIn(name, planned_destinations())
        for name in (
            "benchmarks/branching-service-frozen.json",
            "benchmarks/branching-service-frozen-metal.json",
            "benchmarks/branching-service-calibration.json",
            "benchmarks/branching-service-calibration-metal.json",
        ):
            with self.subTest(name=name):
                self.assertTrue(covered(name, source.V04_WORKLOAD_INPUTS))

    def test_every_script_a_shell_wrapper_runs_is_a_pinned_input(self):
        source = load_module("branching_source_inputs_two", "scripts/source_inputs.py")
        pinned = source.V04_EXECUTION_INPUTS
        for wrapper in ("scripts/study-branching-service.sh", "scripts/run-llama-oracle.sh"):
            text = (ROOT / wrapper).read_text(encoding="utf-8")
            for name in sorted(path.name for path in (ROOT / "scripts").iterdir()):
                if name.endswith((".py", ".sh")) and name in text:
                    with self.subTest(wrapper=wrapper, script=name):
                        self.assertTrue(covered(f"scripts/{name}", pinned))

    def test_the_source_check_has_no_bypass(self):
        text = (ROOT / "scripts/release_evidence_validators.py").read_text(encoding="utf-8")
        command = DISPATCH._branching_command(Path("harness.py"), Path("root"), find(planned(), CUDA), "source.json")
        self.assertIn("--source-manifest", command)
        self.assertEqual(command[-2:], ["--source-scope", "archive"])
        self.assertNotIn("--skip-source", text)
        self.assertNotIn("--no-source", text)


def covered(path: str, inputs: tuple[str, ...]) -> bool:
    """A source input is a file or a directory that contains the path."""

    return any(path == item or path.startswith(item + "/") for item in inputs)


def planned_destinations() -> set[str]:
    return {entry["destination"] for entry in planned()["files"]}


if __name__ == "__main__":
    unittest.main()
