#!/usr/bin/env python3
"""Pass receipts from the real study scripts through the real release checker.

The real `study-live-server.sh` and `study-batched-service.sh` run in a
temporary Git checkout against a fake executable and a fake `nvidia-smi`, as in
`test_study_live_server.py`. The checker then validates the receipt they wrote.
The quality sidecar, the quality comparison record, and the executable are
synthetic fixtures. No model loads and no GPU is used. Nothing here is a
measurement.
"""

import importlib.util
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


LIVE = load("batched_pipeline_live_tests", ROOT / "tests" / "test_study_live_server.py")
CHECK = load("batched_pipeline_checker", ROOT / "scripts" / "check-batched-service.py")
DISPATCH = load("batched_pipeline_validators", ROOT / "scripts" / "release_evidence_validators.py")

STUDY = "receipts/v04-linux-cuda-batched-service.json"
SOURCE = "receipts/source-inputs-v04.json"
PRESTUDY = "receipts/source-inputs-v04-prestudy.json"
SCRATCH = "receipts/study-scratch.json"
COMPARISON = "receipts/v04-linux-cuda-quality-comparison-qwen3.json"
SIDECAR = "receipts/v04-linux-cuda-quality-comparison-qwen3-leone-quality.json"
PLAN = "plans/qwen3-8b-sm89.json"
REPETITIONS = str(CHECK.REPETITIONS)


def wide_base():
    """Return a port whose next 12 ports are free. Five repetitions use 12."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base < 60000 and all(LIVE._free(base + offset) for offset in range(1, 13)):
            return base


class Pipeline(LIVE.Fixture):
    """The live-server fixture with the files the release checker reads."""

    def __init__(self, directory):
        super().__init__(directory)
        self.port = wide_base()
        self.model_sha256 = LIVE.SHARED.sha256(self.root / LIVE.MODEL)
        self.write_comparison()
        self.write_source_manifest()

    def build_checkout(self):
        super().build_checkout()
        for name in CHECK.source_inputs.V04_REQUIRED_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, destination)
        (self.root / PLAN).write_text("{}\n")
        sidecar = {
            "receipt_id": "synthetic-quality-sidecar",
            "subject": {"model_artifact": {"sha256": LIVE.SHARED.sha256(self.root / LIVE.MODEL)}},
            "metrics": {"kld": {"mean": 0.25}, "top1_agreement": 0.5},
        }
        (self.root / SIDECAR).write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        self.commit("pipeline inputs")

    def inputs(self, output, *counts):
        return [LIVE.MODEL, PLAN, SIDECAR, str(output), *counts]

    def write_comparison(self):
        sidecar = self.root / SIDECAR
        body = {
            "schema_version": CHECK.QUALITY_SCHEMA,
            "model_family": "qwen3",
            "models": {"subject": {"sha256": self.model_sha256}},
            "quality": {"leone": {
                "path": sidecar.name,
                "sha256": LIVE.SHARED.sha256(sidecar),
                "receipt": {"receipt_id": json.loads(sidecar.read_text())["receipt_id"]},
            }},
        }
        (self.root / COMPARISON).write_text(json.dumps(body) + "\n")

    def write_source_manifest(self, evidence=False):
        inputs = CHECK.source_inputs
        body = inputs.snapshot(self.root, "HEAD", inputs.V04_EXECUTION_INPUTS)
        body["schema_version"] = "leone.source-inputs.v2"
        listed = subprocess.run(
            ["git", "ls-files", "plans"], cwd=self.root, check=True, capture_output=True, text=True,
        ).stdout.split()
        body["workload_files"] = {
            name: {"sha256": inputs.digest((self.root / name).read_bytes()), "executable": False}
            for name in listed
        }
        body["evidence_files"] = (
            inputs._snapshot_current(self.root, inputs.V04_EVIDENCE_INPUTS) if evidence else {}
        )
        (self.root / SOURCE).write_text(json.dumps(body) + "\n")

    def record_manifest(self, name):
        """Run the documented `record-v04` command at HEAD. It creates the file."""
        subprocess.run(
            [sys.executable, "scripts/source_inputs.py", "record-v04", "HEAD", name],
            cwd=self.root, check=True, capture_output=True, text=True,
        )

    def study(self, repetitions=REPETITIONS, clients="4", **env):
        return self.outer(
            self.root / STUDY, clients, "64",
            LEONE_STUDY_REPETITIONS=repetitions, LEONE_STUDY_SOURCE_MANIFEST=SOURCE, **env,
        )

    def expected(self, **changes):
        values = {
            "platform": "linux-x86_64", "target": "x86_64-unknown-linux-gnu", "backend": "cuda",
            "model_sha256": self.model_sha256,
            "binary_sha256": LIVE.SHARED.sha256(self.binary),
            "quality_record": COMPARISON, "quality_receipt": SIDECAR, "source_manifest": SOURCE,
        }
        return CHECK.Expected(**{**values, **changes})


def validate(tree, expected, offline=True):
    CHECK.validate(tree, tree / STUDY, expected, offline=offline)


class GeneratedReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._temporary.cleanup)
        cls.fixture = Pipeline(cls._temporary.name)
        cls.result = cls.fixture.study()
        if cls.result.status == 0:
            cls.study = json.loads((cls.fixture.root / STUDY).read_text())

    def setUp(self):
        self.assertEqual(self.result.status, 0, self.result.stderr)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tree = pathlib.Path(temporary.name) / "tree"
        shutil.copytree(self.fixture.root, self.tree, symlinks=True)
        self.expected = self.fixture.expected()

    def edit_study(self, change):
        path = self.tree / STUDY
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value) + "\n")

    def test_generated_receipt_carries_the_appended_interface(self):
        study = self.study
        self.assertEqual(len(study["runs"]), CHECK.REPETITIONS)
        self.assertEqual(study["workload"]["prompt_sha256"], CHECK.PROMPT_SHA256)
        self.assertEqual(study["binary"]["sha256"], LIVE.SHARED.sha256(self.fixture.binary))
        for run in study["runs"]:
            self.assertEqual(run["binary"], study["binary"])
            self.assertEqual(run["hardware"], study["hardware"])
            self.assertEqual(run["workload"]["prompt_sha256"], CHECK.PROMPT_SHA256)
            self.assertEqual(run["scheduled"]["batch_size"], CHECK.WORKLOAD["concurrent_clients"])
            self.assertEqual(run["serial"]["batch_size"], 1)
        self.assertIn(CHECK.DIGEST_LIMIT, study["limits"])
        self.assertIn(CHECK.BATCH_LIMIT, study["limits"])
        for run in study["runs"]:
            self.assertIn(CHECK.BATCH_LIMIT, run["limits"])

    def test_checker_accepts_the_generated_receipt_offline_and_against_git(self):
        validate(self.tree, self.expected, offline=True)
        validate(self.tree, self.expected, offline=False)

    def test_dispatcher_command_accepts_the_generated_receipt(self):
        record = {
            "validator": "batched-service-v1", "path": STUDY, "platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu", "backend": "cuda",
            "model_sha256": self.fixture.model_sha256,
            "binary_sha256": self.expected.binary_sha256,
            "quality_record": COMPARISON, "quality_receipt": SIDECAR,
        }
        command = DISPATCH._command(self.tree, ROOT, record, None, "0" * 40, SOURCE, "0" * 64)
        DISPATCH._run(command, ROOT)

    def test_source_change_after_the_study_is_rejected(self):
        script = self.tree / "scripts" / "study-live-server.sh"
        script.write_text(script.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(ValueError, "measured source input changed"):
            validate(self.tree, self.expected, offline=True)
        with self.assertRaisesRegex(ValueError, "measured source input changed"):
            validate(self.tree, self.expected, offline=False)

    def test_generated_receipt_fails_after_each_trusted_binding_changes(self):
        cases = (
            ("binary", {"binary_sha256": "0" * 64}, "study binary differs from the release binary"),
            ("model", {"model_sha256": "0" * 64}, "not the gated Qwen3 artifact"),
            ("quality sidecar", {"quality_receipt": "receipts/other.json"}, "not the comparison sidecar"),
        )
        for label, changes, message in cases:
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, message):
                    validate(self.tree, self.fixture.expected(**changes))

    def test_generated_receipt_fails_after_each_edit(self):
        cases = (
            ("batch limit", lambda s: s["runs"][3]["scheduled"].update(batch_size=1), "batch limit is not 4"),
            ("baseline limit", lambda s: s["runs"][0]["serial"].update(batch_size=4), "batch limit is not 1"),
            ("summary", lambda s: s["summary"]["p95_completion_latency_ratio"].update(median=0.5), "summary"),
            ("elapsed time", lambda s: s["runs"][1]["scheduled"].update(wall_ms=1.0), "wall"),
            ("no runs", lambda s: s.pop("runs"), "retained runs is missing"),
            ("no digest limit", lambda s: s["limits"].remove(CHECK.DIGEST_LIMIT), "digest limit"),
            ("no batch size limit", lambda s: s["limits"].remove(CHECK.BATCH_LIMIT), "batch size limit"),
            ("binary commit", lambda s: s["binary"]["build_info"].update(source_commit="1" * 40), "build source differs"),
        )
        for label, change, message in cases:
            with self.subTest(label):
                shutil.copyfile(self.fixture.root / STUDY, self.tree / STUDY)
                self.edit_study(change)
                with self.assertRaisesRegex(ValueError, message):
                    validate(self.tree, self.expected)

    def render(self, output):
        return subprocess.run(
            [str(self.tree / "scripts/render-release-evidence.sh"), output, STUDY],
            cwd=self.tree, capture_output=True, text=True,
        )

    def test_renderer_reads_the_generated_receipt_and_states_both_limits(self):
        rendered = self.render("out/evidence.md")
        self.assertEqual(rendered.returncode, 0, rendered.stderr)
        text = (self.tree / "out/evidence.md").read_text()
        self.assertIn(f"- {CHECK.DIGEST_LIMIT}\n- {CHECK.BATCH_LIMIT}", text)
        self.assertIn(self.fixture.model_sha256, text)
        self.assertIn(f"| Repetitions | {CHECK.REPETITIONS} |", text)

    def test_renderer_rejects_retained_runs_without_either_limit(self):
        for limit in (CHECK.DIGEST_LIMIT, CHECK.BATCH_LIMIT):
            with self.subTest(limit):
                shutil.copyfile(self.fixture.root / STUDY, self.tree / STUDY)
                self.edit_study(lambda s: s["limits"].remove(limit))
                rendered = self.render("out/evidence.md")
                self.assertEqual(rendered.returncode, 2)
                self.assertIn("lacks the digest or batch size limit", rendered.stderr)
                self.assertFalse((self.tree / "out/evidence.md").exists())


class ProducerContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Pipeline(temporary.name)

    def check_written_study(self, message):
        with self.assertRaisesRegex(ValueError, message):
            validate(self.fixture.root, self.fixture.expected())

    def test_losing_five_repetition_study_keeps_its_receipt_and_fails_the_checker(self):
        self.fixture.configure(delay_scheduled=0.4, delay_serial=0.05)
        result = self.fixture.study()
        self.assertEqual(result.status, 1, result.stderr)
        self.assertIn("performance gate", result.stderr)
        self.check_written_study("aggregate throughput ratio is not greater than 1")

    def test_other_client_count_and_prompt_fail_the_frozen_workload(self):
        result = self.fixture.study(repetitions="1", clients="3")
        self.assertEqual(result.status, 0, result.stderr)
        self.check_written_study("study workload concurrent_clients")
        (self.fixture.root / STUDY).unlink()
        result = self.fixture.study(repetitions="1", LEONE_STUDY_PROMPT="Another prompt.")
        self.assertEqual(result.status, 0, result.stderr)
        self.check_written_study("study workload prompt is not the frozen prompt")

    def test_occupied_port_in_a_later_repetition_publishes_no_study(self):
        holder = socket.socket()
        self.addCleanup(holder.close)
        holder.bind(("127.0.0.1", self.fixture.port + 4))
        holder.listen()
        result = self.fixture.study(repetitions="2")
        self.assertNotEqual(result.status, 0, result.stderr)
        self.assertIn(f"scheduled port {self.fixture.port + 4} already has a listener", result.stderr)
        self.assertFalse((self.fixture.root / STUDY).exists())
        (kept,) = self.fixture.beside(self.fixture.root / STUDY, "runs")
        self.assertTrue((kept / "1.json").is_file())
        self.assertEqual(len(self.fixture.serves()), 2)
        self.assertIn(str(kept), result.stderr)

    def test_manifest_with_evidence_needs_a_study_path_outside_the_evidence_glob(self):
        self.fixture.write_source_manifest(evidence=True)
        result = self.fixture.study(repetitions="2")
        self.assertNotEqual(result.status, 0, result.stderr)
        self.assertIn("evidence:", result.stderr)
        self.assertFalse((self.fixture.root / STUDY).exists())

    def test_study_moved_into_place_passes_against_a_new_final_manifest(self):
        (self.fixture.root / SOURCE).unlink()
        self.fixture.record_manifest(PRESTUDY)
        result = self.fixture.outer(
            self.fixture.root / SCRATCH, "4", "64",
            LEONE_STUDY_REPETITIONS=REPETITIONS, LEONE_STUDY_SOURCE_MANIFEST=PRESTUDY)
        self.assertEqual(result.status, 0, result.stderr)
        shutil.move(self.fixture.root / SCRATCH, self.fixture.root / STUDY)
        with self.assertRaisesRegex(ValueError, "evidence:"):
            validate(self.fixture.root, self.fixture.expected(source_manifest=PRESTUDY), offline=False)
        self.assertFalse((self.fixture.root / SOURCE).exists())
        prestudy = (self.fixture.root / PRESTUDY).read_bytes()
        self.fixture.record_manifest(SOURCE)
        self.assertEqual((self.fixture.root / PRESTUDY).read_bytes(), prestudy)
        validate(self.fixture.root, self.fixture.expected(), offline=False)

    def test_record_v04_refuses_to_replace_a_recorded_manifest(self):
        self.fixture.record_manifest(PRESTUDY)
        with self.assertRaises(subprocess.CalledProcessError):
            self.fixture.record_manifest(PRESTUDY)

    def test_other_repetition_count_fails_the_checker(self):
        result = self.fixture.study(repetitions="2")
        self.assertEqual(result.status, 0, result.stderr)
        self.check_written_study("study workload repetitions is not 5")


if __name__ == "__main__":
    unittest.main()
