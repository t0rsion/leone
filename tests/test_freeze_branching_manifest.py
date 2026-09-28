"""Check that the freeze command derives every service bound from the calibration receipt and types no quality bound."""

import copy
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_study_branching_service as T  # noqa: E402

HARNESS = T.HARNESS
SPEC = importlib.util.spec_from_file_location("freeze_branching_manifest", ROOT / "scripts/freeze-branching-manifest.py")
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


class FreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import frozen_control_fixture

        cls.directory = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.directory.name)
        cls.control = frozen_control_fixture.build_control(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def template(self):
        template = copy.deepcopy(self.control["frozen"])
        for key in ("calibration_receipt",):
            template[key] = None
        template.update({"phase": "pending", "freeze_status": "pending", "workload_id": "fixture-frozen-pending-v1",
                         "name": "fixture-template"})
        template["evaluation"] = {"thresholds": None, "quality": "pending", "quality_record": None,
                                "quality_policy": copy.deepcopy(HARNESS.QUALITY_POLICY)}
        for engine in template["engines"]:
            engine.get("history_tokenization", {}).pop("special_tokens_policy_sha256", None)
        path = self.root / "template.json"
        path.write_text(json.dumps(template))
        return template

    def arguments(self, **overrides):
        values = {
            "root": self.root, "template": "template.json", "calibration_receipt": "calibration-receipt.json",
            "quality_record": self.control["frozen"]["evaluation"]["quality_record"]["path"], "offline": False,
            "output": self.root / f"frozen-{len(list(self.root.glob('frozen-*.json')))}.json",
        }
        return type("Arguments", (), {**values, **overrides})

    def freeze(self, **overrides):
        self.template()
        arguments = self.arguments(**overrides)
        FREEZE.freeze(arguments)
        return json.loads(arguments.output.read_text())

    def test_the_manifest_derives_from_the_receipts_and_validates(self):
        manifest = self.freeze()
        self.assertEqual(HARNESS.validate_manifest(manifest), [])
        self.assertEqual(manifest["phase"], "frozen")
        self.assertEqual(manifest["workload_id"], "fixture-frozen-v1")
        self.assertEqual(manifest["calibration_receipt"]["sha256"], HARNESS.sha256_file(self.root / "calibration-receipt.json"))
        model = HARNESS.sha256_file(self.root / "models/model.gguf")
        self.assertEqual(HARNESS._frozen_manifest_reference(manifest, self.root, model)[0], [])
        for engine in manifest["engines"]:
            declaration = engine["history_tokenization"]
            for field in FREEZE.TOKENIZER_FIELDS:
                self.assertIn(field, declaration)
            for field in FREEZE.TOKENIZER_IDENTITY_FIELDS:
                self.assertIn(field, declaration)

    def test_each_threshold_is_its_observation_times_its_declared_rule(self):
        manifest = self.freeze()
        for threshold in manifest["evaluation"]["thresholds"]:
            decision = threshold["calibration"]
            rule = FREEZE.RULES[threshold["metric"]][0]
            self.assertEqual(decision["rule"], rule)
            expected = True if rule == "boolean_true" else decision["observation"] * FREEZE.FACTORS[rule]
            self.assertEqual(threshold["value"], expected)
            self.assertEqual(threshold["operator"], FREEZE.RULES[threshold["metric"]][1])

    def losing_record(self, change):
        """Copy the accepted record directory, apply `change` to the llama_cpp receipt, and return its path."""

        source = (self.root / self.control["frozen"]["evaluation"]["quality_record"]["path"]).parent
        losing = self.root / f"quality-losing-{len(list(self.root.glob('quality-losing-*')))}"
        shutil.copytree(source, losing)
        record = json.loads((losing / "record.json").read_text())
        entry = record["quality"]["llama_cpp"]
        change(entry["receipt"])
        (losing / entry["path"]).write_text(json.dumps(entry["receipt"]))
        entry["sha256"] = HARNESS.sha256_file(losing / entry["path"])
        (losing / "record.json").write_text(json.dumps(record))
        return str((losing / "record.json").relative_to(self.root))

    def test_one_quality_record_is_bound_under_the_declared_policy_with_no_tolerance(self):
        manifest = self.freeze()
        evaluation = manifest["evaluation"]
        self.assertEqual(evaluation["quality_record"], self.control["frozen"]["evaluation"]["quality_record"])
        self.assertEqual(evaluation["quality_policy"], HARNESS.QUALITY_POLICY)
        self.assertNotIn("quality_tolerances", evaluation)
        self.assertEqual(manifest["engines"][0]["quality_label"], "leone")

    def test_a_peer_row_that_differs_from_the_leone_row_is_bound_and_retained(self):
        path = self.losing_record(lambda receipt: receipt["metrics"]["kld"].update(max=receipt["metrics"]["kld"]["max"] * 1000))
        manifest = self.freeze(quality_record=path)
        self.assertEqual(manifest["evaluation"]["quality_record"]["path"], path)
        record = json.loads((self.root / path).read_text())
        self.assertGreater(record["quality"]["llama_cpp"]["receipt"]["metrics"]["kld"]["max"], record["quality"]["leone"]["receipt"]["metrics"]["kld"]["max"])

    def test_a_record_that_fails_the_structural_join_is_refused(self):
        path = self.losing_record(lambda receipt: receipt["corpus"].update(sha256="9" * 64))
        with self.assertRaises(FREEZE.FreezeError) as error:
            self.freeze(quality_record=path)
        self.assertIn("receipt corpus differs from the record", str(error.exception))

    def test_a_template_without_the_policy_is_refused(self):
        template = self.template()
        del template["evaluation"]["quality_policy"]
        (self.root / "template.json").write_text(json.dumps(template))
        with self.assertRaises(FREEZE.FreezeError) as error:
            FREEZE.freeze(self.arguments())
        self.assertIn("evaluation.quality_policy must be the declared canonical_v2_common_oracle policy", str(error.exception))

    def test_a_missing_observation_is_refused_without_a_typed_bound(self):
        receipt = json.loads((self.root / "calibration-receipt.json").read_text())
        template = self.template()
        for run in receipt["runs"]:
            run["probes"] = []
        with self.assertRaises(FREEZE.FreezeError) as error:
            FREEZE.derive_thresholds(HARNESS, template, receipt, "a" * 64)
        self.assertIn("the calibration receipt does not support", str(error.exception))

    def test_the_tokenizer_pins_must_come_from_agreeing_runs(self):
        receipt = copy.deepcopy(self.control["calibration_receipt"])
        pins = FREEZE.derive_tokenizer(receipt, "llama_cpp")
        self.assertEqual(set(pins), set(FREEZE.TOKENIZER_FIELDS))
        first = next(run for run in receipt["runs"] if run["engine"] == "llama_cpp")
        first["branches"][0]["history_reuse"]["tokenization"]["oracle"]["vocab_size"] = 1
        with self.assertRaises(FREEZE.FreezeError):
            FREEZE.derive_tokenizer(receipt, "llama_cpp")

    def test_the_output_is_written_exclusively(self):
        manifest = self.freeze()
        self.assertTrue(manifest)
        existing = self.root / "existing.json"
        existing.write_text("kept")
        self.template()
        with self.assertRaises(FileExistsError):
            FREEZE.freeze(self.arguments(output=existing))
        self.assertEqual(existing.read_text(), "kept")

    def test_no_bound_is_typed_in_the_script(self):
        source = (ROOT / "scripts/freeze-branching-manifest.py").read_text()
        for literal in ("1e300", "0.11", "0.891", "max_kld", "min_top1", "quality_tolerances"):
            self.assertNotIn(literal, source)


if __name__ == "__main__":
    unittest.main()
