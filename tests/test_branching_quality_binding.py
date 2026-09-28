"""Bind the branching study to canonical v2 quality comparison records.

The records come from the builders of the quality stage tests, which the
canonical validator accepts. The study binds to them. It does not validate
retained logits: `validate-quality-stage.py comparison` does.
"""

import copy
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_quality_cuda as QC  # noqa: E402
import test_quality_stage as QS  # noqa: E402
import test_study_branching_service as T  # noqa: E402

HARNESS = T.HARNESS
LIBRARIES = HARNESS._load_sibling_module("linked_libraries.py")
FAMILIES = ("libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-cuda")


def libraries(suffix, extra=()):
    return [{"name": f"{name}.so.{suffix}", "sha256": HARNESS.sha256_bytes(name.encode())} for name in (*FAMILIES, *extra)]


STATISTICS_HASH = "d" * 64
STATISTICS_PATHS = (
    ("executable",), ("validation", "statistics_executable"),
    ("validation", "trusted", "statistics_identity", "executable"), ("source_identities", "statistics", "executable"),
)


def distinct_statistics(record, trusted=None):
    """Give the statistics producer its own executable hash, as a collection on another target has."""

    for keys in STATISTICS_PATHS:
        if HARNESS._dig(record, *keys):
            HARNESS._dig(record, *keys)["sha256"] = STATISTICS_HASH
    if trusted is not None:
        trusted["statistics_identity"]["executable"]["sha256"] = STATISTICS_HASH


def served_row(executable, commit):
    return {"provenance": {"executable_sha256": executable, "build_info": {"source_id": f"leone:git:{commit}"}}}


def cuda_case(test):
    case = QC.CudaQualityTests("test_cuda_comparison_accepts_samples_only")
    case.setUp()
    test.addCleanup(case.doCleanups)
    record = json.loads(case.manifest.read_text())
    distinct_statistics(record, case.trusted)
    return case, record


class CudaRecordTests(unittest.TestCase):
    def setUp(self):
        self.case, self.record = cuda_case(self)
        self.directory = self.case.manifest.parent
        self.model = self.record["models"]["subject"]["sha256"]

    def errors(self, record=None, model=None):
        return HARNESS._quality_record_errors(record or self.record, self.directory, "cuda", model or self.model, "evaluation")

    def test_the_record_the_study_binds_is_canonical(self):
        self.case.manifest.write_text(json.dumps(self.record))
        QC.VALIDATE.validate_comparison(self.case.manifest, self.case.verifier, self.case.trusted, self.case.generation)
        self.assertEqual(self.errors(), [])

    def test_the_canonical_validator_binds_the_native_executable_the_study_reads(self):
        record = copy.deepcopy(self.record)
        record["producers"]["leone"]["body"]["executable"]["sha256"] = "e" * 64
        self.case.manifest.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            QC.VALIDATE.validate_comparison(self.case.manifest, self.case.verifier, self.case.trusted, self.case.generation)
        identity = self.case.trusted["producer_identities"]["leone_cuda"]["executable"]["sha256"]
        self.assertEqual(HARNESS._quality_native_executable(self.record), identity)
        self.assertNotEqual(HARNESS._quality_native_executable(self.record), self.record["executable"]["sha256"])

    def test_the_v1_receipt_is_rejected_by_schema(self):
        v1 = json.loads((ROOT / "receipts/quality-concurrent-service.json").read_text())
        self.assertEqual(v1["schema_version"], "leone.quality-comparison.v1")
        self.assertIn("evaluation quality record schema does not match the cuda backend", self.errors(v1))

    def test_the_metal_schema_is_rejected_for_a_cuda_study(self):
        record = {**self.record, "schema_version": HARNESS.QUALITY_RECORD_SCHEMAS["metal"]}
        self.assertIn("evaluation quality record schema does not match the cuda backend", self.errors(record))

    def test_a_subject_model_that_differs_from_the_served_model_is_rejected(self):
        self.assertIn(
            "evaluation quality record subject model differs from the served model", self.errors(model="0" * 64)
        )

    def test_a_sidecar_that_differs_on_disk_is_rejected(self):
        (self.directory / self.record["quality"]["leone"]["path"]).write_text("{}")
        self.assertIn("evaluation quality record leone sidecar SHA-256 does not match", self.errors())

    def test_an_inline_receipt_that_differs_from_its_sidecar_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["quality"]["leone"]["receipt"]["sample_count"] += 1
        self.assertIn("evaluation quality record leone inline receipt differs from its sidecar", self.errors(record))

    def test_a_receipt_id_that_differs_between_the_sidecar_and_the_inline_receipt_is_rejected(self):
        for label in ("leone", "llama_cpp"):
            record = copy.deepcopy(self.record)
            entry = record["quality"][label]
            sidecar = json.loads((self.directory / entry["path"]).read_text())
            self.assertEqual(sidecar["receipt_id"], entry["receipt"]["receipt_id"])
            sidecar["receipt_id"] = "00000000-0000-4000-8000-00000000abcd"
            (self.directory / entry["path"]).write_text(json.dumps(sidecar))
            entry["sha256"] = HARNESS.sha256_file(self.directory / entry["path"])
            self.assertIn(f"evaluation quality record {label} inline receipt differs from its sidecar", self.errors(record))

    def test_a_receipt_id_changed_only_in_the_inline_receipt_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["quality"]["llama_cpp"]["receipt"]["receipt_id"] = "00000000-0000-4000-8000-00000000abcd"
        self.assertIn("evaluation quality record llama_cpp inline receipt differs from its sidecar", self.errors(record))

    def test_a_sample_contract_other_than_the_declared_policy_is_rejected(self):
        record = copy.deepcopy(self.record)
        record["validation"]["trusted"]["sample_contract"] = "linspace-inclusive-v1:64"
        self.assertIn("evaluation quality record sample contract differs from the declared policy", self.errors(record))

    def test_rebound_quality_rows_must_identify_one_common_oracle(self):
        record = copy.deepcopy(self.record)
        entry = record["quality"]["llama_cpp"]
        entry["receipt"]["oracle"]["artifact_sha256"] = "c" * 64
        path = self.directory / entry["path"]
        path.write_text(json.dumps(entry["receipt"]))
        entry["sha256"] = HARNESS.sha256_file(path)
        self.assertIn("evaluation quality receipts do not share a common oracle", self.errors(record))

    def test_a_missing_sidecar_is_rejected(self):
        (self.directory / self.record["quality"]["llama_cpp"]["path"]).unlink()
        self.assertIn("evaluation quality record llama_cpp sidecar SHA-256 does not match", self.errors())

    def test_the_oracle_is_not_a_selectable_subject(self):
        self.assertEqual(set(HARNESS.QUALITY_PRODUCERS), {"leone", "llama_cpp"})
        record = copy.deepcopy(self.record)
        record["quality"]["oracle"] = record["quality"]["llama_cpp"]
        self.assertEqual(self.errors(record), [])
        del record["quality"]["llama_cpp"]
        self.assertIn("evaluation quality record llama_cpp row is missing", self.errors(record))

    def test_a_receipt_that_is_not_schema_3_or_names_another_corpus_is_rejected(self):
        for change, message in (
            (lambda receipt: receipt.update(schema_version=2), "receipt schema is not 3"),
            (lambda receipt: receipt["corpus"].update(sha256="9" * 64), "receipt corpus differs from the record"),
            (lambda receipt: receipt["subject"]["model_artifact"].update(sha256="9" * 64), "receipt subject model differs"),
            (lambda receipt: receipt["metrics"]["kld"].pop("max"), "receipt lacks a finite KLD max"),
        ):
            record = copy.deepcopy(self.record)
            change(record["quality"]["leone"]["receipt"])
            self.directory_with(record)
            self.assertTrue(any(message in error for error in self.errors(record)), message)

    def directory_with(self, record):
        """Write the mutated receipt over its sidecar so only the receipt check can object."""

        entry = record["quality"]["leone"]
        path = self.directory / entry["path"]
        path.write_text(json.dumps(entry["receipt"]))
        entry["sha256"] = HARNESS.sha256_file(path)

    def test_cuda_task_rows_must_follow_the_answer_tokens(self):
        record = copy.deepcopy(self.record)
        record["tasks"]["long_context"]["rows"] = [1]
        self.assertIn("evaluation quality record task rows differ from the answer tokens", self.errors(record))


class MetalRecordTests(unittest.TestCase):
    def setUp(self):
        self.case = QS.QualityStageTests("test_comparison_receipt_is_linked_by_hash")
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.manifest, self.record, self.verifier, self.trusted = self.case.write_comparison_fixture()
        distinct_statistics(self.record)
        self.manifest.write_text(json.dumps(self.record))
        self.model = self.record["models"]["subject"]["sha256"]

    def errors(self, record=None):
        return HARNESS._quality_record_errors(record or self.record, self.manifest.parent, "metal", self.model, "evaluation")

    def test_the_cross_device_record_binds_through_the_same_entry_point(self):
        QS.VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted)
        self.assertEqual(self.errors(), [])
        self.assertNotEqual(self.record["executable"]["sha256"], HARNESS._quality_native_executable(self.record))

    def test_the_task_result_must_pass_with_every_argmax_equal_to_the_answer(self):
        for change in (
            lambda result: result.update(passed=False),
            lambda result: result.update(schema_version="leone.quality-task-result.v1"),
            lambda result: result["argmax"].update(leone_metal=[99]),
            lambda result: result["argmax"].update(expected=[]),
        ):
            record = copy.deepcopy(self.record)
            change(record["tasks"]["long_context"]["result"])
            self.assertIn(
                "evaluation quality record task result is not a passed leone.quality-task-result.v2", self.errors(record)
            )

    def test_the_adapter_and_native_identities_come_from_the_stage_bodies(self):
        self.assertEqual(HARNESS._quality_adapter_libraries(self.record), [])
        native = self.record["stages"]["metal_subject"]["body"]["native"]["manifest"]["body"]["executable"]["sha256"]
        self.assertEqual(HARNESS._quality_native_executable(self.record), native)
        self.assertNotEqual(native, self.record["executable"]["sha256"])

    def test_the_canonical_validator_binds_the_native_executable_the_study_reads(self):
        record = copy.deepcopy(self.record)
        record["stages"]["metal_subject"]["body"]["native"]["manifest"]["body"]["executable"]["sha256"] = "e" * 64
        self.manifest.write_text(json.dumps(record))
        with self.assertRaises(ValueError):
            QS.VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted)

    def test_the_native_executable_binds_the_served_process_and_the_statistics_executable_does_not(self):
        commit = self.record["subjects"]["leone"]["engine"]["git_commit"]
        native = HARNESS._quality_native_executable(self.record)
        self.assertEqual(HARNESS._quality_same_executable_errors(self.record, served_row(native, commit), "leone"), [])
        for served in (self.record["executable"]["sha256"], "f" * 64):
            self.assertEqual(HARNESS._quality_same_executable_errors(self.record, served_row(served, commit), "leone"), [
                "quality native executable differs from the serving process: leone"])

    def test_the_metal_backend_family_is_a_core_library(self):
        self.assertIn("libggml-metal", LIBRARIES.core_families("metal"))
        self.assertNotIn("libggml-cuda", LIBRARIES.core_families("metal"))


class PeerAndExecutableBindingTests(unittest.TestCase):
    def setUp(self):
        _, self.record = cuda_case(self)
        self.record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"] = libraries("0.21.0")
        self.engine = {"id": "llama_cpp", "backend": "cuda", "quality_producer": "peer_adapter"}
        self.served = {"status": "observed", "loaded_library_status": HARNESS.LOADED_LIBRARY_STATUS,
                       "libraries": libraries("0", ("libllama-server-impl", "libllama-common", "libmtmd"))}

    def errors(self, served=None):
        row = {"provenance": {"linked_libraries": served or self.served}}
        return HARNESS._quality_peer_library_errors(self.record, self.engine, row, ROOT)

    def test_matching_core_libraries_bind_the_server_to_the_adapter(self):
        self.assertEqual(self.errors(), [])

    def test_server_only_libraries_are_not_compared(self):
        self.served["libraries"].append({"name": "libmtmd.so.0", "sha256": "f" * 64})
        self.assertEqual(self.errors(), [])

    def test_null_or_garbage_core_library_digests_are_rejected(self):
        for invalid in (None, "garbage"):
            record = copy.deepcopy(self.record)
            adapter = record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"]
            served = copy.deepcopy(self.served)
            for libraries in (adapter, served["libraries"]):
                next(item for item in libraries if item["name"].startswith("libggml-cuda"))["sha256"] = invalid
            errors = HARNESS._quality_peer_library_errors(
                record, self.engine, {"provenance": {"linked_libraries": served}}, ROOT
            )
            self.assertIn("quality peer library digest is invalid: llama_cpp libggml-cuda", errors)

    def test_a_different_backend_library_is_rejected(self):
        for item in self.served["libraries"]:
            if item["name"].startswith("libggml-cuda"):
                item["sha256"] = "f" * 64
        self.assertEqual(self.errors(), ["quality peer library differs from the adapter: llama_cpp libggml-cuda"])

    def test_a_missing_core_family_is_rejected_on_either_side(self):
        self.served["libraries"] = [item for item in self.served["libraries"] if not item["name"].startswith("libggml-cpu")]
        self.assertEqual(self.errors(), ["quality peer library differs from the adapter: llama_cpp libggml-cpu"])
        self.record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"] = []
        self.assertEqual(len(self.errors()), 5)

    def test_darwin_dylib_names_bind_through_the_metal_stage_body(self):
        names = ("libllama.0.dylib", "libggml.0.dylib", "libggml-base.0.dylib", "libggml-cpu.0.dylib", "libggml-metal.0.dylib")
        adapter = [{"name": name, "sha256": HARNESS.sha256_bytes(name.encode())} for name in names]
        record = {
            "subjects": {"llama_cpp": {"engine": {"git_commit": HARNESS._pinned_llama_commit(ROOT)}}},
            "stages": {"metal_subject": {"body": {"subject": {"manifest": {"body": {"executable": {"linked_libraries": adapter}}}}}}},
        }
        engine = {"id": "llama_cpp", "backend": "metal", "quality_producer": "peer_adapter"}
        served = {"status": "observed", "loaded_library_status": HARNESS.LOADED_LIBRARY_STATUS,
                  "libraries": [*copy.deepcopy(adapter), {"name": "libmtmd.0.dylib", "sha256": "f" * 64}]}
        check = lambda: HARNESS._quality_peer_library_errors(record, engine, {"provenance": {"linked_libraries": served}}, ROOT)
        self.assertEqual(check(), [])
        served["libraries"][4]["sha256"] = "f" * 64
        self.assertEqual(check(), ["quality peer library differs from the adapter: llama_cpp libggml-metal"])

    def test_older_records_that_name_a_path_are_read(self):
        adapter = self.record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"]
        self.record["producers"]["llama_cpp"]["body"]["executable"]["linked_libraries"] = [
            {"path": item["name"], "sha256": item["sha256"]} for item in adapter
        ]
        self.assertEqual(self.errors(), [])

    def test_linkage_must_be_recorded_as_resolved_not_observed(self):
        self.assertEqual(self.errors({**self.served, "loaded_library_status": "observed_process_map"}), [
            "peer serving libraries are not recorded as resolved linkage: llama_cpp"])
        self.assertEqual(self.errors({"status": "unavailable"}), [
            "peer serving libraries are not recorded as resolved linkage: llama_cpp"])

    def test_the_pinned_commit_must_match_external_pinned(self):
        self.record["subjects"]["llama_cpp"]["engine"]["git_commit"] = "0" * 40
        self.assertIn("quality peer commit differs from external/PINNED: llama_cpp", self.errors())

    def test_leone_needs_the_native_executable_bytes_and_source_of_the_record(self):
        native = HARNESS._quality_native_executable(self.record)
        self.assertNotEqual(native, self.record["executable"]["sha256"])
        commit = self.record["subjects"]["leone"]["engine"]["git_commit"]
        self.assertEqual(HARNESS._quality_same_executable_errors(self.record, served_row(native, commit), "leone"), [])
        for served in (self.record["executable"]["sha256"], "f" * 64):
            self.assertEqual(HARNESS._quality_same_executable_errors(self.record, served_row(served, commit), "leone"), [
                "quality native executable differs from the serving process: leone"])
        self.assertEqual(HARNESS._quality_same_executable_errors(self.record, served_row(native, "0" * 40), "leone"), [
            "quality source commit differs from the serving process: leone"])

    def test_a_record_without_a_native_executable_is_rejected(self):
        del self.record["producers"]["leone"]["body"]["executable"]
        commit = self.record["subjects"]["leone"]["engine"]["git_commit"]
        self.assertIn("quality native executable differs from the serving process: leone",
                      HARNESS._quality_same_executable_errors(self.record, served_row("f" * 64, commit), "leone"))

    def test_the_peer_executable_is_never_compared_with_the_adapter(self):
        row = {"provenance": {"executable_sha256": "f" * 64, "linked_libraries": self.served},
               "running_identity": {"start": {"identity": {"model_sha256": self.record["models"]["subject"]["sha256"]}}}}
        self.assertEqual(HARNESS._quality_engine_binding_errors(self.record, self.engine, row, ROOT), [])
        self.assertNotEqual(row["provenance"]["executable_sha256"], self.record["producers"]["llama_cpp"]["body"]["executable"]["sha256"])

    def test_a_serving_model_that_differs_from_the_record_is_rejected(self):
        row = {"provenance": {"linked_libraries": self.served},
               "running_identity": {"start": {"identity": {"model_sha256": "0" * 64}}}}
        self.assertEqual(HARNESS._quality_engine_binding_errors(self.record, self.engine, row, ROOT), [
            "quality model differs from the serving process: llama_cpp"])

    def test_the_binding_declaration_names_the_limits(self):
        manifest = T.calibration_manifest()
        declaration = HARNESS._quality_binding_declaration(manifest)
        self.assertEqual(declaration["quality_policy"], "canonical_v2_common_oracle")
        self.assertEqual(declaration["quality_recomputation_status"], "not_run_by_harness")
        self.assertEqual(declaration["quality_path_status"], "eval_path_not_served_path")
        self.assertEqual(declaration["engines"]["llama_cpp"]["loaded_library_status"], "resolved_linkage_not_process_map")
        self.assertEqual(declaration["engines"]["leone"], {"quality_label": "leone", "quality_producer": "same_executable"})


class ManifestDeclarationTests(unittest.TestCase):
    NAMES = ("calibration", "frozen", "calibration-metal", "frozen-metal")

    def manifest(self, name):
        return json.loads((ROOT / f"benchmarks/branching-service-{name}.json").read_text())

    def test_every_prospective_manifest_declares_labels_and_validates(self):
        for name in self.NAMES:
            manifest = self.manifest(name)
            self.assertEqual(HARNESS.validate_manifest(manifest), [], name)
            labels = {engine["id"]: (engine["quality_label"], engine["quality_producer"]) for engine in manifest["engines"]}
            self.assertEqual(labels, {"leone": ("leone", "same_executable"), "llama_cpp": ("llama_cpp", "peer_adapter")})

    def test_the_metal_pair_differs_from_the_cuda_pair_only_in_backend_names_and_plan(self):
        for phase in ("calibration", "frozen"):
            cuda, metal = self.manifest(phase), self.manifest(f"{phase}-metal")
            self.assertEqual({e["backend"] for e in metal["engines"]}, {"metal"})
            self.assertNotIn("plan", metal["artifacts"])
            self.assertEqual(metal["prompts"], cuda["prompts"])
            self.assertEqual(metal["schedule"], cuda["schedule"])
            self.assertEqual(metal["budgets"], cuda["budgets"])
            self.assertNotEqual(metal["workload_id"], cuda["workload_id"])
            argv = [e for e in metal["engines"] if e["id"] == "llama_cpp"][0]["spawn"]["argv"]
            self.assertEqual(argv[argv.index("--device") + 1], "MTL0")

    def test_a_missing_or_mismatched_label_is_rejected(self):
        manifest = self.manifest("calibration")
        manifest["engines"][0]["quality_label"] = "llama_cpp"
        self.assertIn("leone: quality_label differs from the engine kind", HARNESS.validate_manifest(manifest))
        del manifest["engines"][0]["quality_label"]
        self.assertIn("leone: quality_label must be leone or llama_cpp", HARNESS.validate_manifest(manifest))
        manifest = self.manifest("calibration")
        manifest["engines"][1]["quality_producer"] = "same_executable"
        self.assertIn("llama_cpp: quality_producer must be peer_adapter", HARNESS.validate_manifest(manifest))

    def test_the_prospective_templates_declare_the_common_oracle_policy(self):
        for name in ("frozen", "frozen-metal"):
            self.assertEqual(self.manifest(name)["evaluation"]["quality_policy"], HARNESS.QUALITY_POLICY)
            self.assertEqual(self.manifest(name)["evaluation"]["quality_record"], None)

    def test_a_missing_or_changed_policy_is_rejected_before_and_after_freezing(self):
        manifest = self.manifest("frozen")
        message = "evaluation.quality_policy must be the declared canonical_v2_common_oracle policy"
        for change in (lambda value: value.pop("quality_policy"), lambda value: value["quality_policy"].update(name="other")):
            changed = copy.deepcopy(manifest)
            change(changed["evaluation"])
            self.assertIn(message, HARNESS.validate_manifest(changed))

    def test_frozen_engines_must_share_one_backend(self):
        manifest = self.manifest("frozen")
        manifest["engines"][1]["backend"] = "metal"
        self.assertIn("frozen engines must share one quality backend: cuda or metal", HARNESS._quality_declaration_errors(manifest))

    def test_the_removed_receipt_fields_are_no_longer_declared(self):
        for name in self.NAMES:
            manifest = self.manifest(name)
            self.assertNotIn("quality", manifest["artifacts"])
            self.assertNotIn("quality_receipt", manifest["evaluation"])
            self.assertNotIn("quality_calibration_receipt", manifest["evaluation"])
            self.assertNotIn("quality_tolerances", manifest["evaluation"])


class ControlBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import frozen_control_fixture

        cls.directory = tempfile.TemporaryDirectory()
        cls.control = frozen_control_fixture.build_control(pathlib.Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def mutated(self, change):
        """Return the validator errors for a mutated copy, then restore the accepted receipt."""

        receipt = copy.deepcopy(self.control["receipt"])
        change(receipt)
        try:
            self.control["receipt_path"].write_text(json.dumps(receipt, sort_keys=True))
            return HARNESS.validate_receipt(self.control["receipt_path"], self.control["root"])
        finally:
            self.control["receipt_path"].write_text(json.dumps(self.control["receipt"], sort_keys=True))

    def row(self, receipt, engine):
        return next(item for item in receipt["engines"] if item["id"] == engine)

    def test_the_accepted_control_binds_both_engines_to_the_record(self):
        self.assertEqual(HARNESS.validate_receipt(self.control["receipt_path"], self.control["root"]), [])

    def test_a_peer_backend_library_that_differs_from_the_adapter_is_rejected(self):
        def change(receipt):
            for item in self.row(receipt, "llama_cpp")["provenance"]["linked_libraries"]["libraries"]:
                if item["name"].startswith("libggml-cuda"):
                    item["sha256"] = "f" * 64

        self.assertIn("quality peer library differs from the adapter: llama_cpp libggml-cuda", self.mutated(change))

    def test_a_receipt_without_the_quality_binding_declaration_is_rejected(self):
        errors = self.mutated(lambda receipt: receipt["evaluation"].pop("quality_binding"))
        self.assertIn("receipt quality binding differs from the manifest declaration", errors)

    def test_the_declaration_states_the_eval_path_limit(self):
        binding = self.control["receipt"]["evaluation"]["quality_binding"]
        self.assertEqual(binding["quality_path_status"], "eval_path_not_served_path")
        self.assertEqual(binding["engines"]["llama_cpp"]["loaded_library_status"], "resolved_linkage_not_process_map")

    def test_one_evaluation_record_binds_both_labels_and_no_bound_is_typed(self):
        evaluation = self.control["frozen"]["evaluation"]
        self.assertEqual(evaluation["quality_policy"], HARNESS.QUALITY_POLICY)
        self.assertEqual(set(evaluation), {
            "thresholds", "quality", "quality_record", "quality_policy", "calibration_receipt_sha256",
        })
        self.assertEqual(self.control["receipt"]["evaluation"]["quality_binding"]["quality_recomputation_status"], "not_run_by_harness")

    def test_the_leone_row_binds_the_native_executable_and_not_the_statistics_executable(self):
        record = json.loads((self.control["root"] / self.control["frozen"]["evaluation"]["quality_record"]["path"]).read_text())
        served = self.row(self.control["receipt"], "leone")["provenance"]["executable_sha256"]
        self.assertEqual(served, HARNESS._quality_native_executable(record))
        self.assertNotEqual(served, record["executable"]["sha256"])


if __name__ == "__main__":
    unittest.main()
