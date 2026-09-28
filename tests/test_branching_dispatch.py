"""Run the release dispatcher over real branching packages for both backends.

Each package holds a canonical quality comparison record (the real CUDA writer's
output, the real Metal stage builder's record), the accepted frozen study of
`frozen_control_fixture` bound to that exact record, the history reexecution files
from the checker's own builders, and a source manifest. The dispatcher runs the
real `validate-quality-stage.py` and the real study harness in archive scope.
Nothing here is native evidence: the binaries, models, and logits are fixtures.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import branching_package_fixture as F  # noqa: E402

BACKENDS = ("cuda", "metal")
DISPATCH = F.DISPATCH
BUILT: dict[str, dict] = {}
DIRECTORY = tempfile.TemporaryDirectory()


def setUpModule():
    for backend in BACKENDS:
        root = Path(DIRECTORY.name) / backend
        root.mkdir()
        BUILT[backend] = F.build_package(root, backend)


def tearDownModule():
    for package in BUILT.values():
        package["published"]["case"].doCleanups()
    DIRECTORY.cleanup()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def edit(path: Path, change) -> None:
    value = load(path)
    change(value)
    F.write_json(path, value)


class PackageCase(unittest.TestCase):
    """A disposable copy of one backend's package, so a case may change any file."""

    backend = "cuda"

    def setUp(self):
        original = BUILT[self.backend]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "package"
        shutil.copytree(original["root"], self.root, symlinks=True)
        self.records = copy.deepcopy(original["records"])
        self.package = {**original, "root": self.root, "records": self.records}
        self.quality, self.study = self.records
        self.receipt = self.root / self.study["path"]

    def rehash(self):
        F.write_sources(self.root, self.package["manifest"], self.records)

    def run_dispatch(self, records=None):
        F.dispatch(self.package, records)

    def rejects(self, message: str, records=None, rehash: bool = True):
        if rehash:
            self.rehash()
        with self.assertRaisesRegex(ValueError, message):
            self.run_dispatch(records)

    def frozen_edit(self, change):
        edit(self.root / "frozen.json", change)

    def receipt_edit(self, change):
        edit(self.receipt, change)


class GenuinePackageMixin:
    def test_the_package_passes_the_dispatcher(self):
        self.run_dispatch()

    def test_both_canonical_steps_ran_on_the_exact_record(self):
        commands = []
        with mock.patch.object(DISPATCH, "_run", side_effect=lambda command, _root: commands.append(command)):
            self.run_dispatch()
        stage = [item for item in commands if "validate-quality-stage.py" in item[1]]
        study = [item for item in commands if "study-branching-service.py" in item[1]]
        self.assertEqual([item[2:4] for item in stage], [["comparison", str(self.root / self.quality["path"])]])
        self.assertEqual(len(study), 1)
        self.assertEqual(study[0][study[0].index("--validate-receipt") + 1], str(self.receipt))
        self.assertEqual(study[0][-2:], ["--source-scope", "archive"])
        self.assertIn("--source-manifest", study[0])
        self.assertNotIn("--skip-source", study[0])

    def test_the_study_binds_the_exact_quality_record_digest(self):
        frozen = load(self.root / "frozen.json")
        reference = frozen["evaluation"]["quality_record"]
        self.assertEqual(reference["path"], self.quality["path"])
        self.assertEqual(reference["sha256"], F.file_sha(self.root / self.quality["path"]))
        self.assertEqual(frozen["evaluation"]["quality_policy"]["release_validation"], "quality-stage-v1")

    def test_the_harness_says_it_did_not_recompute_quality(self):
        binding = load(self.receipt)["evaluation"]["quality_binding"]
        self.assertEqual(binding["quality_recomputation_status"], "not_run_by_harness")
        self.receipt_edit(lambda value: value["evaluation"]["quality_binding"].update(quality_recomputation_status="recomputed"))
        self.rejects("quality binding differs from the policy")

    def test_no_quality_calibration_record_is_packaged(self):
        names = {path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()}
        self.assertFalse([name for name in names if "quality-calibration" in name or "quality_calibration" in name])
        self.frozen_edit(lambda value: value["evaluation"].update(quality_tolerances={"kld": 1}))
        self.rejects("retired quality calibration field")

    def test_the_canonical_validator_alone_rejects_a_changed_sample(self):
        sample = next(
            path for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.suffix in {".f32", ".bin"} and "oracle" in path.name
        )
        sample.write_bytes(bytes(reversed(sample.read_bytes())) or b"\x00")
        self.rejects("trusted evidence validator failed")

    def test_the_quality_record_must_run_through_the_canonical_validator(self):
        with self.assertRaisesRegex(ValueError, "canonical quality validation did not run"):
            self.rehash()
            DISPATCH._require_quality_ran([self.study], set())
        self.run_dispatch()

    def test_a_changed_quality_record_no_longer_matches_the_study(self):
        path = self.root / self.quality["path"]
        path.write_bytes(path.read_bytes() + b" ")
        self.rejects("quality record digest differs")

    def study_first(self):
        """Dispatch the study before its quality record, so the study's own checks speak first."""

        return [self.study, self.quality]

    def test_a_record_for_another_backend_is_rejected(self):
        self.quality["backend"] = "metal" if self.backend == "cuda" else "cuda"
        self.rejects("backend or model differs", self.study_first())

    def test_a_record_for_another_model_is_rejected(self):
        self.quality["model_sha256"] = "0" * 64
        self.rejects("backend or model differs", self.study_first())

    def test_a_study_of_another_model_is_rejected(self):
        self.study["model_sha256"] = "0" * 64
        self.rejects("branching receipt model differs", self.study_first())

    def test_a_study_naming_another_quality_record_is_rejected(self):
        self.frozen_edit(lambda value: value["evaluation"]["quality_record"].update(path="other/record.json"))
        self.rejects("names another quality record")

    def test_the_quality_sidecars_are_required_dependencies(self):
        for sidecar in [name for name in self.study["dependencies"] if name.endswith("-quality.json")]:
            with self.subTest(sidecar=sidecar):
                records = copy.deepcopy(self.records)
                records[1]["dependencies"].remove(sidecar)
                self.rejects("sidecar is not a dependency", records)

    def test_a_native_binary_other_than_the_release_binary_is_rejected(self):
        self.study["binary_sha256"] = "0" * 64
        self.rejects("Leone binary differs from the release binary")

    def test_the_policy_must_be_the_canonical_common_oracle(self):
        self.frozen_edit(lambda value: value["evaluation"]["quality_policy"].update(name="calibrated"))
        self.rejects("not canonical_v2_common_oracle")

    def test_every_leone_gate_must_pass(self):
        def fail(value):
            gate = next(item for item in value["evaluation"]["threshold_results"] if item["engine"] == "leone")
            gate["status"] = "fail"

        self.receipt_edit(fail)
        self.rejects("Leone branching gate did not pass")

    def test_an_unsupported_leone_branch_method_is_rejected(self):
        def unsupported(value):
            row = next(item for item in value["engines"] if item["kind"] == "leone")
            row["branch_method"]["status"] = "unsupported"

        self.receipt_edit(unsupported)
        self.rejects("branch method is not supported")

    def test_a_source_commit_other_than_the_release_commit_is_rejected(self):
        self.receipt_edit(lambda value: value["source"].update(commit="e" * 40))
        self.rejects("source commit differs")

    def test_a_dirty_source_tree_is_rejected(self):
        self.receipt_edit(lambda value: value["source"].update(tracked_tree_clean=False))
        self.rejects("source tree is dirty")

    def test_the_seven_archive_inputs_are_pinned_by_the_source_manifest(self):
        for name in DISPATCH.BRANCHING_ARCHIVE_INPUTS:
            with self.subTest(name=name):
                source = load(self.root / F.SOURCE_MANIFEST)
                del source["files"][name]
                F.write_json(self.root / F.SOURCE_MANIFEST, source)
                self.rejects("source input hash is missing", rehash=False)
                self.rehash()

    def test_a_changed_archive_input_is_rejected(self):
        for name in DISPATCH.BRANCHING_ARCHIVE_INPUTS:
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n# changed\n")
                self.rejects("packaged source input changed", rehash=False)
                path.write_bytes(original)


class HistoryMixin:
    def history_edit(self, name: str, change):
        edit(self.root / self.study[name], change)

    def test_the_history_result_must_verify(self):
        self.history_edit("history_result", lambda value: value.update(status="incomplete"))
        self.rejects("history reexecution did not verify")

    def test_the_result_may_not_claim_an_offline_recompute(self):
        self.history_edit("history_result", lambda value: value.update(offline_recompute="verified"))
        self.rejects("claims an offline recompute")

    def test_the_result_must_state_its_limits(self):
        self.history_edit("history_result", lambda value: value.update(not_covered=[]))
        self.rejects("omits its stated limits")

    def test_the_result_must_bind_this_receipt(self):
        self.history_edit("history_result", lambda value: value["input"].update(sha256="0" * 64))
        self.rejects("not the packaged branching receipt")

    def test_the_result_must_bind_the_expected_pins(self):
        self.history_edit("history_result", lambda value: value.update(expected_sha256="0" * 64))
        self.rejects("does not bind the packaged expected pins")

    def test_the_expected_pins_must_match_the_packaged_helpers(self):
        for name in ("producer_sha256", "template_generator_sha256", "template_config_sha256", "model_sha256"):
            with self.subTest(name=name):
                self.history_edit("history_expected", lambda value, name=name: value.update({name: "0" * 64}))
                self.rejects(f"history expected {name} differs from the packaged pin")
                self.setUp()

    def test_the_peer_executable_pin_must_be_the_studied_binary(self):
        self.history_edit("history_expected", lambda value: value.update(executable_sha256="0" * 64))
        self.rejects("history expected executable_sha256 differs from the packaged pin")

    def test_the_result_must_check_every_retained_branch(self):
        def drop(value):
            value["checked_branches"].pop()
            value["checked_branch_count"] -= 1

        self.history_edit("history_result", drop)
        self.rejects("checked branch count differs")

    def test_a_fresh_server_launch_may_not_use_a_gpu(self):
        def gpu(value):
            value["fresh_server_launches"][0]["argv"] = ["<llama-server>"]

        self.history_edit("history_result", gpu)
        self.rejects("may use a GPU")

    def test_the_helper_and_template_bytes_are_source_bound(self):
        for name in (F.PRODUCER, F.GENERATOR, F.TEMPLATE):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                self.rejects("packaged source input changed", rehash=False)
                path.write_bytes(original)


class HistoryIdentityMixin:
    """The expected pins and the result identity join to every retained oracle.

    Each case rehashes the expected file, the result, and the source manifest,
    so only the identity join can object. None of it reruns a tokenizer.
    """

    def pins(self, expected=None, identity=None, result=None):
        """Change the expected file and the result together, then bind the new bytes."""

        if expected is not None:
            edit(self.root / self.study["history_expected"], expected)
        path = self.root / self.study["history_result"]

        def rebind(value):
            value["expected_sha256"] = F.file_sha(self.root / self.study["history_expected"])
            if identity is not None:
                identity(value["identity"])
            if result is not None:
                result(value)

        edit(path, rebind)

    def peer_oracles(self):
        receipt = load(self.receipt)
        peer = next(item["id"] for item in receipt["engines"] if item["kind"] in ("llama.cpp", "llama_cpp"))
        return [
            branch["history_reuse"]["tokenization"]["oracle"]
            for run in receipt["runs"] if run["engine"] == peer
            for branch in run["branches"]
        ]

    def test_the_positive_pins_are_the_retained_oracle_values(self):
        expected = load(self.root / self.study["history_expected"])
        identity = load(self.root / self.study["history_result"])["identity"]
        for oracle in self.peer_oracles():
            self.assertEqual(oracle["loaded_library_sha256"], expected["loaded_library_sha256"])
            self.assertEqual(oracle["vocab_size"], expected["vocab_size"])
            self.assertEqual(oracle["gguf_sha256"], expected["model_sha256"])
            self.assertEqual({name: oracle[name] for name in DISPATCH.HISTORY_ORACLE_FIELDS}, identity["oracle"])

    def test_a_library_pin_that_disagrees_with_the_retained_oracle_is_rejected(self):
        for value in ("c" * 64, "a" * 64):
            with self.subTest(value=value):
                self.pins(
                    lambda item, value=value: item.update(loaded_library_sha256=value),
                    lambda item, value=value: item.update(loaded_library_sha256=value),
                )
                self.rejects("retained oracle loaded_library_sha256 differs from the expected")
                self.setUp()

    def test_a_vocabulary_or_commit_that_disagrees_is_rejected(self):
        for field, value in (("vocab_size", 999), ("source_commit", "1" * 40)):
            with self.subTest(field=field):
                self.pins(
                    lambda item, field=field, value=value: item.update({field: value}),
                    lambda item, field=field, value=value: item.update({field: value}),
                )
                self.rejects(f"retained oracle {field} differs from the expected|differs from the packaged pin")
                self.setUp()

    def test_a_policy_that_disagrees_is_rejected(self):
        policy = {"prompt": {"add_special": True, "parse_special": True}}
        digest = DISPATCH._canonical_sha256(policy)
        self.pins(
            lambda item: item.update(special_tokens_policy=policy, special_tokens_policy_sha256=digest),
            lambda item: item.update(special_tokens_policy_sha256=digest),
        )
        self.rejects("study special-token policy differs from the expected pins")

    def test_malformed_expected_fields_are_rejected(self):
        for field, value in (
            ("vocab_size", -1), ("vocab_size", 0), ("vocab_size", True), ("vocab_size", "1000"),
            ("loaded_library_sha256", "invalid"), ("loaded_library_sha256", "A" * 64),
            ("executable_sha256", None), ("source_commit", "d" * 39), ("source_commit", "D" * 40),
            ("prompt_prefix", None), ("special_tokens_policy", []),
        ):
            with self.subTest(field=field, value=value):
                self.pins(
                    lambda item, field=field, value=value: item.update({field: value}),
                    lambda item, field=field, value=value: item.update({field: value}),
                )
                self.rejects(f"history expected {field} is invalid|history expected fields differ|differs from the packaged pin")
                self.setUp()

    def test_the_expected_fields_are_exactly_the_checker_fields(self):
        self.pins(lambda item: item.pop("prompt_prefix"))
        self.rejects("history expected fields differ from the checker contract")
        self.setUp()
        self.pins(lambda item: item.update(embedded_template_sha256="1" * 64))
        self.rejects("history expected fields differ from the checker contract")

    def test_the_forged_expected_and_cleared_oracle_is_rejected(self):
        self.pins(
            lambda item: item.update(vocab_size=-1, loaded_library_sha256="invalid"),
            lambda item: item.update(vocab_size=-1, loaded_library_sha256="invalid", oracle=None),
        )
        self.rejects("history expected .* is invalid")

    def test_the_result_oracle_identity_is_required(self):
        self.pins(identity=lambda item: item.update(oracle=None))
        self.rejects("history result oracle identity is missing")
        self.setUp()
        self.pins(identity=lambda item: item["oracle"].pop("vocab_size"))
        self.rejects("history result oracle identity fields differ")
        self.setUp()
        self.pins(identity=lambda item: item["oracle"].update(extra=1))
        self.rejects("history result oracle identity fields differ")

    def test_each_result_oracle_field_must_equal_the_retained_oracle(self):
        for field in DISPATCH.HISTORY_ORACLE_FIELDS:
            with self.subTest(field=field):
                self.pins(identity=lambda item, field=field: item["oracle"].update({field: "changed"}))
                self.rejects("history result oracle identity differs from the retained oracle")
                self.setUp()

    def test_the_result_prompt_prefix_must_match_the_expected_prefix(self):
        self.pins(expected=lambda item: item.update(prompt_prefix="prefix"))
        self.rejects("prompt prefix differs")

    def test_every_retained_oracle_is_joined_not_only_the_first(self):
        oracles = self.peer_oracles()
        self.assertGreater(len(oracles), 1)
        for field, value in (("loaded_library_sha256", "a" * 64), ("vocab_size", 999), ("source_commit", "1" * 40)):
            with self.subTest(field=field):
                def change(receipt, field=field, value=value):
                    peer = next(item["id"] for item in receipt["engines"] if item["kind"] in ("llama.cpp", "llama_cpp"))
                    run = [item for item in receipt["runs"] if item["engine"] == peer][-1]
                    run["branches"][-1]["history_reuse"]["tokenization"]["oracle"][field] = value

                self.receipt_edit(change)
                F.write_history(self.root, self.study, self.package["control"])
                self.rejects("history result oracle identity differs from the retained oracle")
                self.setUp()


class CudaPackageTests(GenuinePackageMixin, HistoryMixin, HistoryIdentityMixin, PackageCase):
    backend = "cuda"

    def test_the_native_and_statistics_executables_differ(self):
        record = load(self.root / self.quality["path"])
        native = record["producers"]["leone"]["body"]["executable"]["sha256"]
        self.assertEqual(native, self.study["binary_sha256"])
        self.assertNotEqual(record["executable"]["sha256"], native)

    def test_the_peer_linkage_is_the_adapter_linkage(self):
        record = load(self.root / self.quality["path"])
        adapter = record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"]
        self.assertEqual({item["name"].split(".")[0] for item in adapter}, set(F.CUDA_FAMILIES))

    def test_the_real_writer_layout_is_the_planned_manifest_layout(self):
        planned = next(
            record
            for entry in json.loads((ROOT / "packaging/release-evidence.v0.4.json").read_text())["backend_requirements"]
            if entry["backend"] == "cuda"
            for record in entry["records"]
            if record["path"] == F.CUDA_RECORD
        )
        self.assertEqual(set(planned["artifact_files"]), set(self.quality["artifact_files"]))
        self.assertEqual(set(planned["artifact_roots"]), set(self.quality["artifact_roots"]))
        self.assertEqual(planned["trusted_inputs"], self.quality["trusted_inputs"])


class MetalPackageTests(GenuinePackageMixin, HistoryMixin, HistoryIdentityMixin, PackageCase):
    backend = "metal"

    def test_the_native_and_statistics_executables_differ(self):
        record = load(self.root / self.quality["path"])
        native = record["stages"]["metal_subject"]["body"]["native"]["manifest"]["body"]["executable"]["sha256"]
        self.assertEqual(native, self.study["binary_sha256"])
        self.assertNotEqual(record["executable"]["sha256"], native)

    def test_the_peer_linkage_is_the_adapter_linkage(self):
        record = load(self.root / self.quality["path"])
        body = record["stages"]["metal_subject"]["body"]["subject"]["manifest"]["body"]
        names = {item["name"].split(".")[0] for item in body["executable"]["linked_libraries"]}
        self.assertEqual(names, set(F.METAL_FAMILIES))

    def test_the_metal_stage_carries_its_native_artifacts(self):
        stage = self.root / "metal-quality/packaged-metal-stage"
        for name in ("leone.metal", "leone-doctor.txt", "leone-eval.stdout"):
            self.assertTrue((stage / name).is_file())


if __name__ == "__main__":
    unittest.main()
