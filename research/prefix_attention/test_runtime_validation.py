"""Check the Runtime freeze and validation workflow on small synthetic receipts."""

from __future__ import annotations

import array
import ast
import hashlib
import importlib.util
import json
import math
import random
import struct
import sys
import tempfile
import unittest
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_runtime as V

VOCAB = 8
MANIFEST = json.loads(V.MANIFEST.read_text())
TOKENS = V.read_tokens(V.RESEARCH / MANIFEST["fixture"])
SOURCE = V.RESEARCH / "llama_cached_workload.json"
REAL_DIGESTS = {
    "calibration_shared": "1d5143a952e0e6398e5b7045bac3bc58fdab9b92f05ee40c86f70d5d8d2681c1",
    "calibration_owned_equal": "dfcce879ec4063c8612fb1c6538bc2fbf4c4d4121ba2eacafe13c4a1fa74c336",
    "calibration_unrelated": "12cec630dd191607b580d48881f2ac56f6afbb7b7be5f6cd5db4ffb173feb92c",
    "calibration_nested": "e903729ec961fab1b01c22e12d231451cb2d08273a4f217c3545e637414bc168",
}
FROZEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
SUBJECT = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
BUILD = {field: f"{field}-fixture" for field in V.BUILD_FIELDS}
BUILD.update({"source_tree_dirty": False, "provenance_unknown": False, "native_controls": {}})


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def values(seed: int, shift: float = 0.0) -> list[float]:
    generator = random.Random(seed)
    return [struct.unpack("<f", struct.pack("<f", generator.uniform(-3, 3) + shift * (i == 0)))[0]
            for i in range(VOCAB)]


def sidecar_bytes(keys: list[tuple], shift: float, order: list[int] | None = None) -> bytes:
    rows = []
    for number, (_, sequence, index, predicted) in enumerate(keys):
        header = V.ROW_HEADER.pack(sequence, index, predicted, VOCAB)
        rows.append(header + array.array("f", values(number, shift)).tobytes())
    return b"".join(rows[n] for n in (order or range(len(rows))))


def linked_digests(data: bytes) -> tuple[str, str]:
    """Recompute the driver digests from sidecar bytes. Branch equals sequence in the fixtures."""
    logits, positions, stride = hashlib.sha256(), hashlib.sha256(), V.ROW_HEADER.size + 4 * VOCAB
    for start in range(0, len(data), stride):
        sequence, index, position, _ = V.ROW_HEADER.unpack(data[start:start + V.ROW_HEADER.size])
        logits.update(struct.pack("<Q", position) + data[start + V.ROW_HEADER.size:start + stride])
        positions.update(struct.pack("<QQQ", sequence, index, position))
    return logits.hexdigest(), positions.hexdigest()


def stats(path: str, topology: str) -> dict:
    grouped = path in V.SHARED_PATHS and topology in ("shared", "fork_of_fork", "other")
    return {"dispatches": 4, "rows": 16, "groups": 4, "multi_row_groups": 2 if grouped else 0}


def case_record(name, topology, keys, path, digest, directory, shift, repeat=0.0):
    files, digests = [], []
    for repetition in range(2):
        data = sidecar_bytes(keys, shift + (repeat if repetition else 0.0))
        relative = Path("runtime-t.artifacts/receipt.json.sidecars") / f"{name}-{path}.{repetition}.logits.bin"
        (directory / relative).parent.mkdir(parents=True, exist_ok=True)
        (directory / relative).write_bytes(data)
        files.append({"file": str(relative), "sha256": sha(directory / relative)})
        digests.append(linked_digests(data))
    bindings = [
        {"sequence": s, "event": e, "token_index": i, "input_position": p - 1, "predicted_position": p, "vocab": VOCAB}
        for e, s, i, p in keys
    ]
    memory = {"live_bytes": 100, "peak_live_bytes": 120}
    return {
        "name": name, "source_case": name, "topology": topology, "input_digest": digest,
        "decode_steps": 4, "branch_count": 4, "retained_branch_count": 0,
        "raw_logit_vocab": VOCAB, "raw_logit_rows": len(keys), "sidecar_logit_rows": len(keys),
        "logit_bindings": bindings, "batch_schedule": [4, 4], "setup_median_ms": 1.1, "decode_median_ms": 2.2,
        "setup_ms": [1.0, 1.1], "decode_ms": [2.0, 2.2], "logits_digest": [d[0] for d in digests],
        "raw_logit_position_digest": [d[1] for d in digests], "token_digest": ["c", "c"], "logits_sidecars": files,
        "setup_stats": [stats(path, topology)] * 2, "stats": [stats(path, topology)] * 2,
        "memory_before": [memory] * 2, "memory_after_setup": [memory] * 2,
        "memory_after_decode": [memory] * 2, "memory_after_retire": [memory] * 2,
    }


def receipt(phase, path, cases, generated):
    return {
        "schema": V.RECEIPT_SCHEMA, "phase": phase, "path": path, "backend": "cuda",
        "device": {"name": "fixture"}, "quality": "unverified", "claim_scope": "fixture",
        "model_name": "fixture.gguf", "model_sha256": SUBJECT, "fixture_sha256": sha(V.RESEARCH / MANIFEST["fixture"]),
        "source_manifest_sha256": sha(SOURCE), "source_plan_sha256": MANIFEST["source_plan_sha256"],
        "source_subject_sha256": SUBJECT, "source_phase": "fixture", "cases": cases,
        "provenance": {
            "manifest_sha256": sha(V.MANIFEST), "source_sha256": {"a": "0" * 64}, "artifact_sha256": path,
            "build": BUILD, "affinity": {}, "host_system": "fixture", "generated_utc": generated.isoformat(),
            "command": ["driver", "--repetitions", "2", "--warmups", "0"],
        },
    }


def calibration_specs():
    for spec in MANIFEST["calibration"]:
        keys = V.calibration_bindings(spec, TOKENS)
        yield spec, keys, [key for key in keys if key[2] in spec["capture_steps"]]


def write_calibration(directory: Path, mutate=None) -> list[Path]:
    files = []
    for path in V.PATHS:
        cases = [case_record(spec["name"], spec["topology"], keys, path, V.calibration_digest(spec, TOKENS),
                             directory, 0.0 if path == V.CONTROL or path == "per_row" else 1e-3)
                 for spec, keys, _ in calibration_specs()]
        data = receipt("calibration", path, cases, FROZEN - timedelta(days=1))
        if mutate:
            mutate(path, data)
        files.append(directory / f"{path}.json")
        files[-1].write_text(json.dumps(data))
    return files


def evaluation_topology(name: str) -> str:
    return "unrelated" if name == "unrelated" else "shared" if name.endswith("shared") else "other"


def scale_timing(record: dict, factor: float) -> None:
    for field in ("setup", "decode"):
        record[f"{field}_ms"] = [value * factor for value in record[f"{field}_ms"]]
        record[f"{field}_median_ms"] = V.median(record[f"{field}_ms"])


def write_evaluation(directory: Path, criteria_sha: str, rows: dict, shift=1e-3, repeat=0.0, timing=None) -> list[Path]:
    files = []
    for path in V.PATHS:
        cases = [case_record(name, evaluation_topology(name), keys, path, "d" * 64, directory,
                             0.0 if path in (V.CONTROL, "per_row") else shift,
                             repeat if path == V.CONTROL else 0.0) for name, keys in rows.items()]
        for record in cases:
            scale_timing(record, (timing or {}).get(path, 1.0))
        data = receipt("evaluation", path, cases, FROZEN + timedelta(days=1))
        data["criteria_sha256"] = criteria_sha
        files.append(directory / f"{path}.json")
        files[-1].write_text(json.dumps(data))
    return files


def source_rows() -> dict:
    return V.source_rows(V.cached.load_workload(SOURCE, V.ROOT), json.loads(SOURCE.read_text()))


def freeze(files: list[Path], directory: Path, oracles=None, monitoring=None) -> Path:
    study, manifest, source = V.load_calibration(files, "cuda")
    criteria = V.derive_criteria(study, manifest, source, oracles, monitoring)
    criteria["frozen_utc"] = FROZEN.isoformat()
    output = directory / "criteria.json"
    output.write_text(json.dumps(criteria))
    return output


def validator_args(criteria, calibration, evaluation, **overrides) -> Namespace:
    values = {
        "backend": "cuda", "criteria": criteria, "calibration": calibration, "evaluation": evaluation,
        "bf16_calibration": None, "bf16_evaluation": None, "oracle_receipt": None, "oracle_logits": None,
        "monitoring": None, "evaluation_monitoring": None,
    }
    return Namespace(**{**values, **overrides})


SAMPLES_HEADER = ("timestamp, name, driver_version, pstate, clocks.current.sm [MHz], "
                  "clocks.current.memory [MHz], power.draw [W], power.limit [W], temperature.gpu, "
                  "utilization.gpu [%], memory.used [MiB]\n")


def write_monitoring(directory: Path, study: "V.Study") -> tuple[Path, Path]:
    """Write a synthetic sample file and a collection record that names the study's receipts."""
    samples, collection = directory / "samples.csv", directory / "collection.json"
    samples.write_text(
        SAMPLES_HEADER
        + "2026/09/20 18:55:28.705, GPU, 1, P8, 210 MHz, 405 MHz, 12.9 W, 500.00 W, 40, 0 %, 140 MiB\n"
        + "2026/09/20 18:55:29.705, GPU, 1, P0, 2715 MHz, 10501 MHz, 306.1 W, 500.00 W, 57, 99 %, 9000 MiB\n")
    collection.write_text(json.dumps({
        "scope": "fixture", "gpu_clock_policy": "default dynamic clocks", "gpu_samples_sha256": sha(samples),
        "records": [{"path": run.file.name, "sha256": run.file_sha256,
                     "driver_sha256": run.receipt["provenance"]["artifact_sha256"]} for run in study.runs.values()],
    }))
    return samples, collection


def imported_modules(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class ScratchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calibration = self.root / "calibration"
        self.calibration.mkdir()


class CalibrationIdentityTests(ScratchTests):
    def test_input_digests_match_the_driver_receipt(self) -> None:
        for spec in MANIFEST["calibration"]:
            self.assertEqual(V.calibration_digest(spec, TOKENS), REAL_DIGESTS[spec["name"]])

    def test_valid_calibration_loads_and_freezes(self) -> None:
        files = write_calibration(self.calibration)
        criteria = json.loads(freeze(files, self.root).read_text())
        self.assertEqual(sorted(criteria["numerical"]), sorted(V.PATHS))
        self.assertEqual(criteria["numerical"][V.CONTROL]["max_abs_logit_diff"], math.nextafter(0.0, math.inf))
        self.assertEqual(criteria["numerical"]["per_row"]["max_abs_logit_diff"], math.nextafter(0.0, math.inf))
        self.assertEqual(criteria["quality"]["gate"], "unverified")
        self.assertFalse(criteria["gpu_monitoring"]["bound"])
        self.assertTrue(criteria["timing_rule"].startswith("new for the runtime study"))

    def fails(self, mutate, message: str) -> None:
        files = write_calibration(self.calibration, mutate)
        with self.assertRaisesRegex(ValueError, message):
            V.load_calibration(files, "cuda")

    def test_wrong_input_digest_is_rejected(self) -> None:
        def mutate(path, data):
            data["cases"][0]["input_digest"] = "0" * 64
        self.fails(mutate, "identity differs|input digest")

    def test_reordered_bindings_are_rejected(self) -> None:
        def mutate(path, data):
            bindings = data["cases"][0]["logit_bindings"]
            bindings[0], bindings[1] = bindings[1], bindings[0]
        self.fails(mutate, "bindings differ|absent, repeated, or reordered")

    def test_shifted_position_is_rejected(self) -> None:
        def mutate(path, data):
            for binding in data["cases"][1]["logit_bindings"]:
                binding["predicted_position"] += 1
                binding["input_position"] += 1
        self.fails(mutate, "bindings differ|absent, repeated, or reordered")

    def test_zero_scored_rows_are_rejected(self) -> None:
        def mutate(path, data):
            data["cases"][0]["sidecar_logit_rows"] = 0
        self.fails(mutate, "no scored rows|identity differs")

    def test_repeated_receipt_path_is_rejected(self) -> None:
        files = write_calibration(self.calibration)
        with self.assertRaisesRegex(ValueError, "each path once"):
            V.load_runs(files[:3] + files[:1], "calibration", "cuda")

    def test_model_hash_difference_between_paths_is_rejected(self) -> None:
        def mutate(path, data):
            if path == "per_row":
                data["model_sha256"] = "2" * 64
        self.fails(mutate, "identity differs")

    def test_dirty_source_is_rejected(self) -> None:
        def mutate(path, data):
            data["provenance"]["build"] = {**BUILD, "source_tree_dirty": True}
        self.fails(mutate, "dirty tree or unknown build provenance")

    def test_quality_must_stay_unverified(self) -> None:
        def mutate(path, data):
            data["quality"] = "verified"
        self.fails(mutate, "unverified")

    def test_shared_reads_on_unrelated_case_are_rejected(self) -> None:
        def mutate(path, data):
            if path in V.SHARED_PATHS:
                for case in data["cases"]:
                    if case["topology"] == "unrelated":
                        case["stats"] = [{**case["stats"][0], "multi_row_groups": 1}] * 2
        self.fails(mutate, "reuse rule")


class SidecarTests(ScratchTests):
    def load(self, edit) -> None:
        files = write_calibration(self.calibration)
        target = self.calibration / "runtime-t.artifacts/receipt.json.sidecars/calibration_shared-per_row.0.logits.bin"
        target.write_bytes(edit(target.read_bytes()))
        V.load_runs(files, "calibration", "cuda")
        V.load_study(files, "calibration", "cuda")

    def test_truncated_sidecar_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "size differs"):
            self.load(lambda data: data[:-4])

    def test_edited_sidecar_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "hash differs"):
            self.load(lambda data: data[:-1] + bytes([data[-1] ^ 1]))

    def test_path_escape_is_rejected(self) -> None:
        files = write_calibration(self.calibration)
        data = json.loads(files[0].read_text())
        data["cases"][0]["logits_sidecars"][0]["file"] = "../outside.bin"
        files[0].write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "leaves the receipt directory"):
            V.load_study(files, "calibration", "cuda")

    def test_reordered_repeated_and_missing_rows_are_rejected(self) -> None:
        keys = [("e", 0, n, 10 + n) for n in range(3)]
        bindings = [{"sequence": s, "event": e, "token_index": i, "predicted_position": p} for e, s, i, p in keys]
        for name, order in (("reordered", [1, 0, 2]), ("repeated", [0, 0, 2]), ("missing", [0, 2, 2])):
            file = self.root / f"{name}.bin"
            file.write_bytes(sidecar_bytes(keys, 0.0, order))
            with self.assertRaisesRegex(ValueError, "absent, repeated, or reordered"):
                V.scored_keys(V.Sidecar(file, "", 3, VOCAB), bindings)

    def test_nonfinite_logit_is_rejected(self) -> None:
        file = self.root / "nan.bin"
        file.write_bytes(V.ROW_HEADER.pack(0, 0, 1, VOCAB) + array.array("f", [math.nan] * VOCAB).tobytes())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            list(V.read_rows(V.Sidecar(file, "", 1, VOCAB)))


class MetricTests(unittest.TestCase):
    def test_kld_matches_the_quality_stage_definition(self) -> None:
        spec = importlib.util.spec_from_file_location("quality_stage", V.ROOT / "scripts/validate-quality-stage.py")
        stage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage)
        for seed in range(5):
            reference, other = values(seed), values(seed + 50)
            expected, top1 = stage.row_stats(tuple(reference), tuple(other), "row")
            kld, largest, agree = V.row_metrics(array.array("f", reference), array.array("f", other))
            self.assertAlmostEqual(kld, expected, places=12)
            self.assertEqual(agree, top1)
            self.assertEqual(largest, max(abs(a - b) for a, b in zip(reference, other)))

    def test_uniform_shift_keeps_the_distribution(self) -> None:
        row = array.array("f", values(1))
        shifted = array.array("f", [value + 2.0 for value in row])
        kld, _, agree = V.row_metrics(row, shifted)
        self.assertLess(kld, 1e-6)
        self.assertTrue(agree)

    def test_bitwise_flag_follows_the_bytes(self) -> None:
        row = array.array("f", values(2))
        self.assertTrue(V.score_rows(iter([(row, array.array("f", row))]))["bitwise_equal"])
        other = array.array("f", row)
        other[0] += 1e-6
        self.assertFalse(V.score_rows(iter([(row, other)]))["bitwise_equal"])


class EvaluationTests(ScratchTests):
    def setUp(self) -> None:
        super().setUp()
        self.rows = source_rows()
        self.criteria = freeze(write_calibration(self.calibration), self.root)
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir()

    def args(self, files, **overrides):
        return validator_args(self.criteria, sorted(self.calibration.glob("*.json")), files, **overrides)

    def test_source_rows_cover_setup_and_captured_steps(self) -> None:
        self.assertEqual(self.rows["long_shared"][0], ("prefix", 0, 512, 513))
        self.assertEqual(len(self.rows["long_shared"]), 21)
        self.assertEqual(sum(len(keys) for keys in self.rows.values()), 168)

    def test_negative_result_is_reported_not_hidden(self) -> None:
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows, shift=1e-4)
        report = V.validate(self.args(files))
        self.assertEqual(report["quality"]["gate"], "unverified")
        self.assertEqual(report["verdict"]["numerical_vs_control"], "pass")
        self.assertEqual(report["verdict"]["bf16_quality"], "unverified")
        self.assertEqual(report["verdict"]["timing"], "within_review_threshold_unmonitored")
        self.assertTrue(report["speed_claim"].startswith("none"))
        self.assertIsNone(report["evidence"]["traffic"]["measured"])
        self.assertTrue(report["claim"].startswith("none"))

    def test_error_beyond_the_frozen_bound_fails_numerically(self) -> None:
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows, shift=5.0)
        report = V.validate(self.args(files))
        self.assertEqual(report["verdict"]["numerical_vs_control"], "fail")

    def test_receipt_naming_other_criteria_is_rejected(self) -> None:
        files = write_evaluation(self.evaluation, "0" * 64, self.rows)
        with self.assertRaisesRegex(ValueError, "other criteria"):
            V.validate(self.args(files))

    def test_receipt_older_than_the_freeze_is_rejected(self) -> None:
        criteria = json.loads(self.criteria.read_text())
        criteria["frozen_utc"] = (FROZEN + timedelta(days=9)).isoformat()
        self.criteria.write_text(json.dumps(criteria))
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows)
        with self.assertRaisesRegex(ValueError, "predates the freeze"):
            V.validate(self.args(files))

    def test_missing_source_row_is_rejected(self) -> None:
        rows = {name: keys[:-1] if name == "unrelated" else keys for name, keys in self.rows.items()}
        files = write_evaluation(self.evaluation, sha(self.criteria), rows)
        with self.assertRaisesRegex(ValueError, "differ from the (source )?workload"):
            V.validate(self.args(files))

    def test_reordered_source_rows_are_rejected(self) -> None:
        rows = {name: keys[1:] + keys[:1] if name == "long_cold" else keys for name, keys in self.rows.items()}
        files = write_evaluation(self.evaluation, sha(self.criteria), rows)
        with self.assertRaisesRegex(ValueError, "differ from the (source )?workload"):
            V.validate(self.args(files))

    def test_criteria_edit_is_rejected(self) -> None:
        criteria = json.loads(self.criteria.read_text())
        criteria["numerical"]["per_row"]["max_kld"] = 1.0
        self.criteria.write_text(json.dumps(criteria))
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows)
        with self.assertRaisesRegex(ValueError, "criteria differ from the calibration"):
            V.validate(self.args(files))

    def test_tool_source_change_after_the_freeze_is_rejected(self) -> None:
        criteria = json.loads(self.criteria.read_text())
        criteria["tools"]["validate_runtime.py"] = "0" * 64
        self.criteria.write_text(json.dumps(criteria))
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows)
        with self.assertRaisesRegex(ValueError, r"criteria differ from the calibration in \['tools'\]"):
            V.validate(self.args(files))

    def test_validation_leaves_every_input_unchanged(self) -> None:
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows)
        watched = [*files, self.criteria, *(p for p in self.calibration.rglob("*") if p.is_file())]
        before = {path: sha(path) for path in watched}
        V.validate(self.args(files))
        self.assertEqual(before, {path: sha(path) for path in watched})


class TimingTests(ScratchTests):
    """Timing labels are signed, and a run without bound GPU samples says so."""

    def setUp(self) -> None:
        super().setUp()
        self.rows = source_rows()
        self.criteria = freeze(write_calibration(self.calibration), self.root)
        self.evaluation = self.root / "evaluation"
        self.evaluation.mkdir()

    def report(self, candidate: float, monitored: bool = False) -> dict:
        files = write_evaluation(self.evaluation, sha(self.criteria), self.rows, timing={V.CANDIDATE: candidate})
        monitoring = write_monitoring(self.root, V.load_study(files, "evaluation", "cuda")) if monitored else None
        args = validator_args(self.criteria, sorted(self.calibration.glob("*.json")), files, evaluation_monitoring=monitoring)
        return V.validate(args)

    def test_a_slower_candidate_is_named_slower(self) -> None:
        report = self.report(3.0)
        self.assertEqual(report["verdict"]["timing"], V.SLOWER + "_unmonitored")
        self.assertTrue(all(item["relative_effect"] < 0 for item in report["timing"]["cases"].values()))

    def test_a_faster_candidate_and_a_mixed_result_are_named(self) -> None:
        self.assertEqual(self.report(0.5)["verdict"]["timing"], V.FASTER + "_unmonitored")
        self.assertEqual(V.overall_label([V.FASTER, V.SLOWER, V.WITHIN]), V.MIXED)
        self.assertEqual(V.overall_label([V.WITHIN, V.WITHIN]), V.WITHIN)

    def test_an_effect_inside_the_threshold_is_within(self) -> None:
        threshold = 0.0909
        self.assertEqual(V.effect_label(threshold * 0.9, threshold), V.WITHIN)
        self.assertEqual(V.effect_label(-threshold * 0.9, threshold), V.WITHIN)
        self.assertEqual(V.effect_label(-threshold, threshold), V.SLOWER)
        self.assertEqual(V.effect_label(threshold, threshold), V.FASTER)

    def test_bound_samples_remove_the_unmonitored_suffix(self) -> None:
        report = self.report(0.5, monitored=True)
        self.assertEqual(report["verdict"]["timing"], V.FASTER)
        self.assertTrue(report["timing"]["monitored"])

    def test_require_positive_rejects_unmonitored_slower_and_mixed_labels(self) -> None:
        verdict = {"numerical_vs_control": "pass", "bf16_quality": "unverified"}
        for label, expected in ((V.FASTER, 0), (V.FASTER + "_unmonitored", 1), (V.SLOWER, 1), (V.MIXED, 1), (V.WITHIN, 1)):
            self.assertEqual(V.exit_code({**verdict, "timing": label}, True), expected, label)
        self.assertEqual(V.exit_code({**verdict, "timing": V.SLOWER}, False), 0)
        self.assertEqual(V.exit_code({**verdict, "timing": V.FASTER, "bf16_quality": "fail"}, True), 1)

    def test_report_lists_executables_and_run_order(self) -> None:
        builds = self.report(1.0)["timing"]["builds"]
        self.assertEqual(builds["distinct_executables"], builds["receipts"])
        self.assertEqual(sorted(builds["artifact_sha256"]), sorted(V.PATHS))
        self.assertIn("run order is not randomized", builds["limit"])

    def test_run_order_follows_the_recorded_times(self) -> None:
        def mutate(path, data):
            data["provenance"]["generated_utc"] = (FROZEN - timedelta(minutes=V.PATHS.index(path))).isoformat()
        study = V.load_study(write_calibration(self.calibration, mutate), "calibration", "cuda")
        self.assertEqual(V.build_disclosure(study)["run_order"], list(reversed(V.PATHS)))


class ControlAndLinkageTests(ScratchTests):
    def test_receipts_must_run_the_source_subject(self) -> None:
        def mutate(path, data):
            data["model_sha256"] = "0" * 64
        files = write_calibration(self.calibration, mutate)
        with self.assertRaisesRegex(ValueError, "model other than the source subject"):
            V.load_study(files, "calibration", "cuda")

    def test_control_repeatability_is_scored_and_gated(self) -> None:
        rows = source_rows()
        criteria = freeze(write_calibration(self.calibration), self.root)
        directory = self.root / "evaluation"
        directory.mkdir()
        files = write_evaluation(directory, sha(criteria), rows, repeat=0.25)
        report = V.validate(validator_args(criteria, sorted(self.calibration.glob("*.json")), files))
        self.assertEqual(report["numerical_by_path"][V.CONTROL]["verdict"], "fail")
        self.assertFalse(report["numerical_vs_control"]["long_shared"][V.CONTROL]["bitwise_equal"])
        self.assertEqual(report["verdict"]["numerical_vs_control"], "fail")

    def test_each_path_reports_a_role_and_a_verdict(self) -> None:
        by_path = V.numerical_by_path({"case": [f"{V.PATHS[2]}:max_kld"]})
        self.assertEqual(by_path[V.PATHS[2]]["verdict"], "fail")
        self.assertEqual(by_path[V.CANDIDATE]["verdict"], "pass")
        self.assertIn("expected to differ", by_path["shared_read_unconstrained"]["role"])

    def tamper(self, field: str) -> None:
        def mutate(path, data):
            data["cases"][0][field][1] = "0" * 64
        files = write_calibration(self.calibration, mutate)
        with self.assertRaisesRegex(ValueError, "digests differ from its sidecar"):
            V.load_calibration(files, "cuda")

    def test_logit_digest_must_match_the_sidecar_of_its_repetition(self) -> None:
        self.tamper("logits_digest")

    def test_position_digest_must_match_the_sidecar_of_its_repetition(self) -> None:
        self.tamper("raw_logit_position_digest")

    def test_digests_recompute_from_independent_fixture_code(self) -> None:
        files = write_calibration(self.calibration)
        study = V.load_study(files, "calibration", "cuda")
        sidecar = study.sidecars[V.CONTROL]["calibration_shared"][0]
        self.assertEqual(V.sidecar_digests(sidecar, [0, 1, 2, 3]), linked_digests(sidecar.file.read_bytes()))

    def test_a_sequence_without_a_branch_is_rejected(self) -> None:
        study = V.load_study(write_calibration(self.calibration), "calibration", "cuda")
        with self.assertRaisesRegex(ValueError, "has no branch"):
            V.sidecar_digests(study.sidecars[V.CONTROL]["calibration_shared"][0], [0, 1])

    def test_delta_limit_is_one_sided_and_floored_at_zero(self) -> None:
        deltas = {"c": {path: {metric: -0.5 for metric in V.METRICS} for path in V.PATHS if path != V.CONTROL}}
        self.assertEqual(V.delta_bounds(deltas)["per_row"]["max_kld"], math.nextafter(0.0, math.inf))
        deltas["c"]["per_row"]["max_kld"] = 0.25
        self.assertEqual(V.delta_bounds(deltas)["per_row"]["max_kld"], math.nextafter(0.25, math.inf))
        self.assertNotIn(V.CONTROL, V.delta_bounds(deltas))

    def test_tool_pins_cover_every_local_import(self) -> None:
        local = {path.stem for path in V.RESEARCH.glob("*.py")}
        needed = set().union(*(imported_modules(V.RESEARCH / name) for name in V.TOOLS)) & local
        self.assertLessEqual(needed, {name.removesuffix(".py") for name in V.TOOLS})
        self.assertLessEqual({"gpu_receipt.py", "generate_manifest.py"}, set(V.tool_hashes()))


class PublicationTests(ScratchTests):
    def test_criteria_and_report_refuse_replacement(self) -> None:
        output = self.root / "existing.json"
        output.write_text("kept")
        with patch.object(sys, "argv", ["freeze", "--backend", "cuda", "--calibration", "a", "--quality-unverified", "--output", str(output)]):
            import freeze_runtime_criteria as F
            with self.assertRaisesRegex(SystemExit, "refusing to replace"):
                F.main()
        with patch.object(sys, "argv", ["validate", "--backend", "cuda", "--criteria", "c", "--calibration", "a",
                                        "--evaluation", "e", "--report", str(output)]):
            with self.assertRaisesRegex(SystemExit, "refusing to replace"):
                V.main()
        self.assertEqual(output.read_text(), "kept")

    def test_a_rejected_freeze_creates_no_directory(self) -> None:
        import freeze_runtime_criteria as F
        target = self.root / "made" / "criteria.json"
        argv = ["freeze", "--backend", "cuda", "--calibration", str(self.root / "missing.json"),
                "--quality-unverified", "--output", str(target)]
        with patch.object(sys, "argv", argv), self.assertRaisesRegex(SystemExit, "criteria not frozen"):
            F.main()
        self.assertFalse(target.parent.exists())

    def test_tool_set_names_both_new_scripts(self) -> None:
        self.assertLessEqual({"freeze_runtime_criteria.py", "validate_runtime.py"}, set(V.tool_hashes()))


if __name__ == "__main__":
    unittest.main()
