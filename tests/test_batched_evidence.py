"""Test the batched-service release evidence gate.

The checker runs against a disposable evidence tree. Every value in that tree
is synthetic and none is a measurement. Numbers are chosen so that the
expected ratios can be computed by hand.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load("check_batched_service", ROOT / "scripts/check-batched-service.py")
EVIDENCE = load("batched_gate_evidence_manifest", ROOT / "scripts/release_evidence_manifest.py")
DISPATCH = load("batched_gate_evidence_validators", ROOT / "scripts/release_evidence_validators.py")

MODEL = EVIDENCE.V04_SUBJECT_MODEL_SHA256["qwen3"]
BINARY = "b1" * 32
COMMIT = "c1" * 20
STUDY_PATH = "receipts/v04-linux-cuda-batched-service.json"
SOURCE_MANIFEST = "receipts/source-inputs-v04.json"
COMPARISON_PATH = "receipts/v04-linux-cuda-quality-comparison-qwen3.json"
QUALITY_PATH = "receipts/v04-linux-cuda-quality-comparison-qwen3-leone-quality.json"
PLAN_BYTES = b"synthetic plan\n"
QUALITY_ID = "synthetic-quality-receipt"
QUALITY_BYTES = (
    json.dumps({"receipt_id": QUALITY_ID, "subject": {"model_artifact": {"sha256": MODEL}}}) + "\n"
).encode()
HARDWARE = {"gpu_name": "Synthetic RTX 4090", "driver_version": "0.0", "memory_total_mib": 1, "compute_capability": "8.9"}
IDENTITY = {
    "platform": "linux-x86_64",
    "target": "x86_64-unknown-linux-gnu",
    "backend": "cuda",
}
TOKENS = 64
CLIENTS = 4


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stats(values: list[float]) -> dict[str, float]:
    """Percentiles of four sorted values. Indexes floor(3p) are 1, 2, and 2."""
    ordered = sorted(values)
    return {"p50": ordered[1], "p95": ordered[2], "p99": ordered[2]}


def make_mode(name: str, batch: int, wall: float, totals: list[float], tokens: int = TOKENS) -> dict:
    return {
        "mode": name,
        "batch_size": batch,
        "wall_ms": wall,
        "aggregate_completion_tok_s": CLIENTS * tokens / (wall / 1000),
        "ttft_ms": stats([total / 2 for total in totals]),
        "total_ms": stats(totals),
        "latency_fairness_ratio": max(totals) / min(totals),
        "server_log_sha256": "5e" * 32,
        "requests": [
            {
                "client": client + 1,
                "http_code": 200,
                "ttft_ms": total / 2,
                "total_ms": total,
                "completion_tokens": tokens,
                "transcript_sha256": "7a" * 32,
                "response_sha256": f"{client + 1:02x}" * 32,
            }
            for client, total in enumerate(totals)
        ],
    }


def make_run(index: int, scheduled=(1000.0, (900.0, 920.0, 940.0, 960.0)), serial=(2000.0, (500.0, 1000.0, 1500.0, 1950.0)), serial_tokens: int = TOKENS) -> dict:
    """Return one synthetic repetition. Repetition k adds 10k or 20k milliseconds."""
    step = float(index)
    scheduled_wall, scheduled_totals = scheduled[0] + 10 * step, [t + 10 * step for t in scheduled[1]]
    serial_wall, serial_totals = serial[0] + 20 * step, [t + 20 * step for t in serial[1]]
    scheduled_mode = make_mode("scheduled", CLIENTS, scheduled_wall, scheduled_totals)
    serial_mode = make_mode("serial", 1, serial_wall, serial_totals, serial_tokens)
    return {
        "schema_version": "leone.server-study.v2",
        "created_utc": "2000-01-01T00:00:00Z",
        "source_commit": COMMIT,
        "model": {"path": "synthetic/model.gguf", "sha256": MODEL},
        "plan": {"path": CHECK.PLAN_PATH, "sha256": sha256(PLAN_BYTES)},
        "quality": {"path": QUALITY_PATH, "sha256": sha256(QUALITY_BYTES), "receipt_id": QUALITY_ID},
        "workload": {**CHECK.WORKLOAD, "prompt_sha256": CHECK.PROMPT_SHA256},
        "binary": make_binary(),
        "hardware": dict(HARDWARE),
        "scheduled": scheduled_mode,
        "serial": serial_mode,
        "checks": {
            "scheduled_transcripts_agree": True,
            "scheduled_matches_serial": True,
            "aggregate_throughput_ratio": scheduled_mode["aggregate_completion_tok_s"] / serial_mode["aggregate_completion_tok_s"],
            "disconnect": {
                "curl_exit": 28,
                "client_disconnected": True,
                "recovery_http_code": 200,
                "recovery_transcript_sha256": "7a" * 32,
            },
        },
    }


def make_binary() -> dict:
    return {
        "sha256": BINARY,
        "build_info": {
            "schema_version": "leone.build-info.v1",
            "source_commit": COMMIT,
            "source_tree_dirty": False,
            "provenance_unknown": False,
            "profile": "release",
            "target": IDENTITY["target"],
            "features": "cuda,server",
        },
    }


def derive(study: dict) -> None:
    """Rewrite samples, summary, and checks from the retained runs."""
    runs = study["runs"]
    throughput = [run["checks"]["aggregate_throughput_ratio"] for run in runs]
    latency = [run["scheduled"]["total_ms"]["p95"] / run["serial"]["total_ms"]["p95"] for run in runs]

    def sample(run: dict, ratio: float, p95_ratio: float) -> dict:
        def mode(record: dict) -> dict:
            return {
                "wall_ms": record["wall_ms"],
                "aggregate_completion_tok_s": record["aggregate_completion_tok_s"],
                "p95_completion_ms": record["total_ms"]["p95"],
                "transcript_sha256": record["requests"][0]["transcript_sha256"],
            }

        return {
            "scheduled": mode(run["scheduled"]),
            "serial": mode(run["serial"]),
            "aggregate_throughput_ratio": ratio,
            "p95_completion_latency_ratio": p95_ratio,
            "transcript_matches": True,
            "disconnect_recovers": True,
        }

    def summary(values: list[float]) -> dict:
        return {"minimum": min(values), "median": sorted(values)[len(values) // 2], "maximum": max(values)}

    study["samples"] = [sample(run, t, l) for run, t, l in zip(runs, throughput, latency)]
    study["summary"] = {
        "aggregate_throughput_ratio": summary(throughput),
        "p95_completion_latency_ratio": summary(latency),
    }
    study["checks"] = {
        "every_transcript_matches": True,
        "every_disconnect_recovers": True,
        "every_throughput_sample_wins": all(value > 1 for value in throughput),
        "every_p95_completion_sample_wins": all(value < 1 for value in latency),
    }


def make_study() -> dict:
    study = {
        "schema_version": CHECK.STUDY_SCHEMA,
        "created_utc": "2000-01-01T00:00:00Z",
        "source_commit": COMMIT,
        "model": {"path": "synthetic/model.gguf", "sha256": MODEL},
        "plan": {"path": CHECK.PLAN_PATH, "sha256": sha256(PLAN_BYTES)},
        "quality": {"path": QUALITY_PATH, "sha256": sha256(QUALITY_BYTES), "receipt_id": QUALITY_ID},
        "workload": {**CHECK.WORKLOAD, "prompt_sha256": CHECK.PROMPT_SHA256, "repetitions": CHECK.REPETITIONS},
        "binary": make_binary(),
        "hardware": dict(HARDWARE),
        "runs": [make_run(index) for index in range(CHECK.REPETITIONS)],
        "limits": ["Synthetic fixture. It is not a measurement.", CHECK.DIGEST_LIMIT, CHECK.BATCH_LIMIT],
    }
    derive(study)
    return study


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_tree(root: Path, study: dict) -> None:
    """Write the study and every input the checker reads below root."""
    records = {}
    for name in CHECK.REQUIRED_SOURCE_FILES:
        data = f"synthetic {name}\n".encode()
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
        records[name] = {"sha256": sha256(data), "executable": False}
    (root / CHECK.PLAN_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / CHECK.PLAN_PATH).write_bytes(PLAN_BYTES)
    write_json(root / SOURCE_MANIFEST, {
        "schema_version": "leone.source-inputs.v2",
        "source_commit": COMMIT,
        "files": records,
        "workload_files": {CHECK.PLAN_PATH: {"sha256": sha256(PLAN_BYTES), "executable": False}},
        "evidence_files": {},
    })
    (root / QUALITY_PATH).write_bytes(QUALITY_BYTES)
    write_json(root / COMPARISON_PATH, {
        "schema_version": "leone.quality-comparison.v2",
        "model_family": "qwen3",
        "models": {"subject": {"sha256": MODEL}},
        "quality": {"leone": {
            "path": Path(QUALITY_PATH).name,
            "sha256": sha256(QUALITY_BYTES),
            "receipt": {
                "receipt_id": QUALITY_ID,
                "subject": {"model_artifact": {"sha256": MODEL}},
            },
        }},
    })
    write_json(root / STUDY_PATH, study)


def expected(**changes) -> "CHECK.Expected":
    values = {
        **IDENTITY,
        "model_sha256": MODEL,
        "binary_sha256": BINARY,
        "quality_record": COMPARISON_PATH,
        "quality_receipt": QUALITY_PATH,
        "source_manifest": SOURCE_MANIFEST,
    }
    return CHECK.Expected(**{**values, **changes})


class CheckerCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.reset()

    def reset(self) -> None:
        for child in self.root.iterdir():
            shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
        self.study = make_study()
        write_tree(self.root, self.study)

    def check(self, **changes) -> None:
        CHECK.validate(self.root, self.root / STUDY_PATH, expected(**changes), offline=True)

    def rewrite(self, rederive: bool = False) -> None:
        if rederive:
            derive(self.study)
        write_json(self.root / STUDY_PATH, self.study)

    def edit(self, relative: str, change) -> None:
        path = self.root / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        change(value)
        write_json(path, value)

    def assert_rejects(self, message: str, rederive: bool = False, **changes) -> None:
        self.rewrite(rederive)
        with self.assertRaisesRegex(ValueError, message):
            self.check(**changes)

    def run_table(self, cases, rederive: bool = False) -> None:
        for label, mutate, message in cases:
            with self.subTest(label):
                self.reset()
                mutate(self.study)
                self.assert_rejects(message, rederive)


def scheduled(study: dict, run: int = 0) -> dict:
    return study["runs"][run]["scheduled"]


def serial(study: dict, run: int = 0) -> dict:
    return study["runs"][run]["serial"]


def disconnect(study: dict, run: int = 0) -> dict:
    return study["runs"][run]["checks"]["disconnect"]


class RecomputationTests(CheckerCase):
    def test_complete_study_passes(self) -> None:
        self.check()

    def test_recomputed_values_match_hand_computation(self) -> None:
        result = CHECK.check_run(self.study["runs"][0], self.study, 0)
        self.assertEqual(result.scheduled.tokens, 256)
        self.assertAlmostEqual(result.scheduled.tok_s, 256.0)
        self.assertAlmostEqual(result.serial.tok_s, 128.0)
        self.assertAlmostEqual(result.throughput_ratio, 2.0)
        self.assertAlmostEqual(result.p95_ratio, 940.0 / 1500.0)
        self.assertEqual(result.scheduled.p95_ms, 940.0)

    def test_appended_fields_stay_compatible(self) -> None:
        self.study["appended"] = {"note": "later field"}
        self.study["samples"][0]["appended"] = 1
        self.study["runs"][0]["appended"] = 1
        self.study["runs"][0]["scheduled"]["requests"][0]["appended"] = 1
        self.rewrite()
        self.check()

    def test_recorded_derived_values_must_match_retained_runs(self) -> None:
        cases = (
            ("summary median", lambda s: s["summary"]["aggregate_throughput_ratio"].update(median=9.0), r"summary\.aggregate_throughput_ratio\.median differs"),
            ("summary latency", lambda s: s["summary"]["p95_completion_latency_ratio"].update(maximum=0.1), r"summary\.p95_completion_latency_ratio\.maximum differs"),
            ("sample ratio", lambda s: s["samples"][0].update(aggregate_throughput_ratio=5.0), r"sample 1\.aggregate_throughput_ratio differs"),
            ("sample transcript", lambda s: s["samples"][1]["scheduled"].update(transcript_sha256="0" * 64), r"sample 2\.scheduled\.transcript_sha256 differs"),
            ("sample flag", lambda s: s["samples"][2].update(disconnect_recovers=False), r"sample 3\.disconnect_recovers differs"),
            ("study check", lambda s: s["checks"].update(every_p95_completion_sample_wins=False), r"checks\.every_p95_completion_sample_wins differs"),
            ("run throughput", lambda s: scheduled(s).update(aggregate_completion_tok_s=999.0), r"aggregate_completion_tok_s differs"),
            ("run p95", lambda s: scheduled(s)["total_ms"].update(p95=1.0), r"total_ms\.p95 differs"),
            ("run first byte", lambda s: serial(s)["ttft_ms"].update(p50=1.0), r"ttft_ms\.p50 differs"),
            ("run fairness", lambda s: serial(s).update(latency_fairness_ratio=1.0), r"latency_fairness_ratio differs"),
            ("run ratio", lambda s: s["runs"][3]["checks"].update(aggregate_throughput_ratio=9.0), r"run 4 aggregate_throughput_ratio differs"),
            ("run boolean", lambda s: s["runs"][3]["checks"].update(scheduled_matches_serial=False), r"scheduled_matches_serial differs"),
        )
        self.run_table(cases)

    def test_run_that_loses_the_gate_fails_even_when_consistent(self) -> None:
        cases = (
            ("equal throughput", make_run(0, scheduled=(2000.0, (900.0, 920.0, 940.0, 960.0))), "aggregate throughput ratio is not greater than 1"),
            ("slower throughput", make_run(0, scheduled=(3000.0, (900.0, 920.0, 940.0, 960.0))), "aggregate throughput ratio is not greater than 1"),
            ("equal p95", make_run(0, scheduled=(1700.0, (1400.0, 1450.0, 1500.0, 1600.0))), "p95 completion latency ratio is not less than 1"),
        )
        for label, run, message in cases:
            with self.subTest(label):
                self.reset()
                self.study["runs"][0] = run
                self.assert_rejects(message, rederive=True)

    def test_changed_run_with_stale_samples_fails(self) -> None:
        self.study["runs"][0] = make_run(0, scheduled=(1100.0, (900.0, 920.0, 940.0, 960.0)))
        self.assert_rejects(r"sample 1\.scheduled\.wall_ms differs")


class CountAndWorkloadTests(CheckerCase):
    def test_wrong_counts_fail(self) -> None:
        cases = (
            ("four runs", lambda s: s["runs"].pop(), "retains 4 runs, not 5"),
            ("six runs", lambda s: s["runs"].append(copy.deepcopy(s["runs"][0])), "retains 6 runs, not 5"),
            ("missing runs", lambda s: s.pop("runs"), "retained runs is missing"),
            ("fewer samples", lambda s: s["samples"].pop(), "sample count differs"),
            ("three requests", lambda s: scheduled(s)["requests"].pop(), "wrong request count"),
            ("five requests", lambda s: serial(s)["requests"].append(copy.deepcopy(serial(s)["requests"][0])), "wrong request count"),
            ("repeated client", lambda s: scheduled(s)["requests"][1].update(client=1), "client numbers are not 1 to 4"),
        )
        self.run_table(cases)

    def test_repeated_run_records_fail_with_consistent_summaries(self) -> None:
        cases = (
            ("five copies", lambda s: s.update(runs=[copy.deepcopy(s["runs"][0]) for _ in s["runs"]]), "run 2 repeats an earlier run record"),
            ("one copy", lambda s: s["runs"].__setitem__(3, copy.deepcopy(s["runs"][1])), "run 4 repeats an earlier run record"),
        )
        self.run_table(cases, rederive=True)

    def test_run_timestamps_are_not_ordered(self) -> None:
        for index, run in enumerate(self.study["runs"]):
            run["created_utc"] = f"2000-01-01T00:00:0{CHECK.REPETITIONS - index}Z"
        self.rewrite(rederive=True)
        self.check()

    def test_workload_must_be_the_frozen_default(self) -> None:
        cases = tuple(
            (key, lambda s, k=key, v=value: s["workload"].update({k: v}), f"study workload {key}")
            for key, value in (
                ("concurrent_clients", 3), ("max_tokens", 32), ("temperature", 1),
                ("seed", 1), ("repetitions", 4),
            )
        ) + (
            ("prompt", lambda s: s["workload"].update(prompt_sha256="0" * 64), "study workload prompt"),
            ("no prompt", lambda s: s["workload"].pop("prompt_sha256"), "study workload prompt"),
            ("float temperature", lambda s: s["workload"].update(temperature=0.0), "study workload temperature"),
            ("run workload", lambda s: s["runs"][2]["workload"].update(max_tokens=32), "run 3 workload max_tokens"),
            ("run prompt", lambda s: s["runs"][2]["workload"].update(prompt_sha256="0" * 64), "run 3 workload prompt"),
        )
        self.run_table(cases)

    def test_frozen_defaults_match_the_study_scripts(self) -> None:
        live = (ROOT / "scripts/study-live-server.sh").read_text(encoding="utf-8")
        batched = (ROOT / "scripts/study-batched-service.sh").read_text(encoding="utf-8")
        self.assertIn(CHECK.PROMPT, live)
        self.assertIn(f"clients=${{5:-{CHECK.WORKLOAD['concurrent_clients']}}}", batched)
        self.assertIn(f"max_tokens=${{6:-{CHECK.WORKLOAD['max_tokens']}}}", batched)
        self.assertIn(f"LEONE_STUDY_REPETITIONS:-{CHECK.REPETITIONS}}}", batched)


class NumberTests(CheckerCase):
    def test_elapsed_times_and_token_counts_must_be_positive_and_finite(self) -> None:
        cases = (
            ("zero wall", lambda s: scheduled(s).update(wall_ms=0), "wall time is not positive"),
            ("negative wall", lambda s: serial(s).update(wall_ms=-5.0), "wall time is not positive"),
            ("boolean wall", lambda s: scheduled(s).update(wall_ms=True), "wall time is not a number"),
            ("text wall", lambda s: scheduled(s).update(wall_ms="1000"), "wall time is not a number"),
            ("zero total", lambda s: scheduled(s)["requests"][0].update(total_ms=0), "total time is not positive"),
            ("first byte after total", lambda s: scheduled(s)["requests"][0].update(ttft_ms=5000.0), "times are not ordered"),
            ("total after wall", lambda s: scheduled(s)["requests"][0].update(total_ms=5000.0), "times are not ordered"),
            ("zero tokens", lambda s: scheduled(s)["requests"][0].update(completion_tokens=0), "completion tokens is outside 1 to 64"),
            ("too many tokens", lambda s: scheduled(s)["requests"][0].update(completion_tokens=65), "completion tokens is outside 1 to 64"),
            ("float tokens", lambda s: scheduled(s)["requests"][0].update(completion_tokens=64.0), "completion tokens is outside 1 to 64"),
            ("http error", lambda s: scheduled(s)["requests"][0].update(http_code=500), "did not return HTTP 200"),
        )
        self.run_table(cases)

    def test_non_finite_values_are_rejected_when_read(self) -> None:
        for label, value in (("nan", float("nan")), ("infinity", float("inf"))):
            with self.subTest(label):
                self.reset()
                scheduled(self.study)["wall_ms"] = value
                self.assert_rejects("non-finite number")
        self.reset()
        text = (self.root / STUDY_PATH).read_text(encoding="utf-8").replace('"wall_ms": 1000.0', '"wall_ms": 1e999', 1)
        (self.root / STUDY_PATH).write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "wall time is not positive and finite"):
            self.check()

    def test_repeated_object_keys_are_rejected(self) -> None:
        (self.root / STUDY_PATH).write_text('{"schema_version": "a", "schema_version": "b"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "repeats an object key"):
            self.check()

    def test_modes_must_generate_the_same_token_count(self) -> None:
        self.study["runs"][0] = make_run(0, serial_tokens=63)
        self.assert_rejects("generated different token counts", rederive=True)


class TranscriptAndDisconnectTests(CheckerCase):
    def test_transcripts_must_agree_and_match_the_baseline(self) -> None:
        cases = (
            ("batched disagree", lambda s: scheduled(s)["requests"][1].update(transcript_sha256="0" * 64), "batched transcripts disagree"),
            ("baseline differs", lambda s: [r.update(transcript_sha256="0" * 64) for r in scheduled(s)["requests"]], "differs from the baseline"),
            ("bad digest", lambda s: scheduled(s)["requests"][0].update(transcript_sha256="xyz"), "transcript is not a SHA-256"),
            ("bad response digest", lambda s: scheduled(s)["requests"][0].update(response_sha256="A" * 64), "response is not a SHA-256"),
        )
        self.run_table(cases)

    def test_disconnect_recovery_is_required_in_every_run(self) -> None:
        cases = (
            ("no recovery", lambda s: disconnect(s, 4).update(recovery_http_code=500), "server did not recover"),
            ("no disconnect", lambda s: disconnect(s).update(curl_exit=0), "did not time out and disconnect"),
            ("flag", lambda s: disconnect(s).update(client_disconnected=False), "disconnect flag differs"),
            ("recovery transcript", lambda s: disconnect(s, 2).update(recovery_transcript_sha256="0" * 64), "recovery transcript differs"),
            ("missing probe", lambda s: s["runs"][1]["checks"].pop("disconnect"), "disconnect is missing"),
        )
        self.run_table(cases)


class ProvenanceTests(CheckerCase):
    def test_actual_binary_and_final_native_build_are_required(self) -> None:
        def build(**changes):
            return lambda s: s["binary"]["build_info"].update(changes)

        cases = (
            ("no binary", lambda s: s.pop("binary"), "study binary is missing"),
            ("no build", lambda s: s["binary"].pop("build_info"), "study build information is missing"),
            ("stale commit", build(source_commit="d1" * 20), "build source differs from the final source"),
            ("dirty", build(source_tree_dirty=True), "source tree is dirty"),
            ("dirty missing", lambda s: s["binary"]["build_info"].pop("source_tree_dirty"), "source tree is dirty"),
            ("unknown provenance", build(provenance_unknown=True), "provenance is unknown"),
            ("provenance missing", lambda s: s["binary"]["build_info"].pop("provenance_unknown"), "provenance is unknown"),
            ("debug", build(profile="debug"), "not a release build"),
            ("other target", build(target="aarch64-apple-darwin"), "target differs from the release target"),
            ("no cuda", build(features="cpu,server"), "lacks the cuda feature"),
            ("cuda prefix", build(features="cudax"), "lacks the cuda feature"),
            ("schema", build(schema_version="leone.build-info.v0"), "unknown schema"),
        )
        self.run_table(cases)

    def test_gpu_record_and_scope_limits_are_required(self) -> None:
        cases = (
            ("no hardware", lambda s: s.pop("hardware"), "study hardware is missing"),
            ("other gpu", lambda s: s["hardware"].update(gpu_name="Synthetic RTX 3090"), "study hardware is not an RTX 4090"),
            ("run hardware", lambda s: s["runs"][1]["hardware"].update(driver_version="1.0"), "run 2 hardware differs from the study"),
            ("no limits", lambda s: s.pop("limits"), "study limits is missing"),
            ("no digest limit", lambda s: s.update(limits=[CHECK.BATCH_LIMIT]), "omit the response digest limit"),
            ("no batch size limit", lambda s: s.update(limits=[CHECK.DIGEST_LIMIT]), "omit the batch size limit"),
            ("malformed limit", lambda s: s.update(limits=[1, CHECK.DIGEST_LIMIT, CHECK.BATCH_LIMIT]), "limits are malformed"),
        )
        self.run_table(cases)

    def test_binary_must_be_the_release_binary_in_every_run(self) -> None:
        self.assert_rejects("study binary differs from the release binary", binary_sha256="0" * 64)
        self.reset()
        self.study["runs"][2]["binary"]["sha256"] = "0" * 64
        self.assert_rejects("run 3 binary differs from the study")
        self.reset()
        del self.study["runs"][2]["binary"]
        self.assert_rejects("run 3 binary differs from the study")

    def test_runs_must_share_the_study_identity(self) -> None:
        for key, value in (
            ("source_commit", "d1" * 20), ("model", {"path": "x", "sha256": MODEL}),
            ("plan", {"path": CHECK.PLAN_PATH, "sha256": "0" * 64}),
            ("quality", {"path": QUALITY_PATH, "sha256": "0" * 64, "receipt_id": "other"}),
        ):
            with self.subTest(key):
                self.reset()
                self.study["runs"][1][key] = value
                self.assert_rejects(f"run 2 {key} differs from the study")

    def test_study_source_commit_must_match_the_source_manifest(self) -> None:
        self.study["source_commit"] = "d1" * 20
        self.assert_rejects("study source commit differs from the source manifest")

    def test_release_identity_is_linux_cuda_only(self) -> None:
        for changes in (
            {"platform": "darwin-arm64", "target": "aarch64-apple-darwin", "backend": "metal"},
            {"backend": "metal"},
        ):
            with self.subTest(changes):
                with self.assertRaisesRegex(ValueError, "only the Linux CUDA release identity"):
                    self.check(**changes)


class BindingTests(CheckerCase):
    def test_model_plan_and_quality_bind_to_the_gated_inputs(self) -> None:
        cases = (
            ("study model", lambda s: s["model"].update(sha256="0" * 64), "not the gated Qwen3 artifact"),
            ("plan path", lambda s: s["plan"].update(path="plans/other.json"), "not the gated Qwen3 plan"),
            ("plan hash", lambda s: s["plan"].update(sha256="0" * 64), "plan differs from the source manifest"),
            ("quality id", lambda s: s["quality"].update(receipt_id="other"), "study quality receipt id differs"),
            ("quality hash", lambda s: s["quality"].update(sha256="0" * 64), "study quality receipt hash differs"),
            ("quality path", lambda s: s["quality"].update(path="receipts/other.json"), "path is not the comparison sidecar"),
            ("full row receipt", lambda s: s["quality"].update(path="receipts/v04-linux-cuda-quality-qwen3.json"), "path is not the comparison sidecar"),
        )
        self.run_table(cases)

    def test_expected_model_must_be_the_study_model(self) -> None:
        self.assert_rejects("not the gated Qwen3 artifact", model_sha256="0" * 64)

    def test_linked_quality_comparison_must_agree(self) -> None:
        cases = (
            ("schema", lambda v: v.update(schema_version="leone.quality-comparison.v1"), "unknown schema"),
            ("family", lambda v: v.update(model_family="llama"), "not for Qwen3"),
            ("subject", lambda v: v["models"]["subject"].update(sha256="0" * 64), "subject differs from the study model"),
            ("receipt id", lambda v: v["quality"]["leone"]["receipt"].update(receipt_id="other"), "receipt id differs from the comparison"),
            ("sidecar hash", lambda v: v["quality"]["leone"].update(sha256="0" * 64), "differs from the comparison record"),
            ("sidecar name", lambda v: v["quality"]["leone"].update(path="other.json"), "not the comparison sidecar"),
            ("row", lambda v: v["quality"].pop("leone"), "Leone quality row is missing"),
        )
        for label, mutate, message in cases:
            with self.subTest(label):
                self.reset()
                self.edit(COMPARISON_PATH, mutate)
                with self.assertRaisesRegex(ValueError, message):
                    self.check()

    def test_quality_sidecar_must_exist_and_match_the_study_and_comparison(self) -> None:
        sidecar = self.root / QUALITY_PATH
        sidecar.write_bytes(b'{"receipt_id": "other", "subject": {}}\n')
        with self.assertRaisesRegex(ValueError, "differs from the comparison record"):
            self.check()
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, "quality receipt is missing"):
            self.check()

    def test_sidecar_content_must_name_the_model_and_receipt(self) -> None:
        cases = (
            ("model", {"receipt_id": QUALITY_ID, "subject": {"model_artifact": {"sha256": "0" * 64}}}, "receipt subject differs"),
            ("id", {"receipt_id": "other", "subject": {"model_artifact": {"sha256": MODEL}}}, "receipt id differs"),
            ("no subject", {"receipt_id": QUALITY_ID}, "quality subject is missing"),
        )
        for label, body, message in cases:
            with self.subTest(label):
                self.reset()
                data = (json.dumps(body) + "\n").encode()
                (self.root / QUALITY_PATH).write_bytes(data)
                self.edit(COMPARISON_PATH, lambda v: v["quality"]["leone"].update(sha256=sha256(data)))
                self.study["quality"]["sha256"] = sha256(data)
                for run in self.study["runs"]:
                    run["quality"]["sha256"] = sha256(data)
                self.rewrite()
                with self.assertRaisesRegex(ValueError, message):
                    self.check()

    def test_v03_study_cannot_satisfy_the_gate(self) -> None:
        shutil.copyfile(ROOT / "receipts/batched-service-study.json", self.root / STUDY_PATH)
        with self.assertRaises(ValueError):
            self.check()


class SourceClosureTests(CheckerCase):
    def test_required_source_files_must_be_recorded_and_unchanged(self) -> None:
        script = "scripts/study-live-server.sh"
        (self.root / script).write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "measured source input changed"):
            self.check()
        self.reset()
        self.edit(SOURCE_MANIFEST, lambda v: v["files"].pop(script))
        with self.assertRaisesRegex(ValueError, "omits required files"):
            self.check()

    def test_every_required_source_file_must_be_recorded(self) -> None:
        for name in CHECK.REQUIRED_SOURCE_FILES:
            with self.subTest(name):
                self.reset()
                self.edit(SOURCE_MANIFEST, lambda v, n=name: v["files"].pop(n))
                with self.assertRaisesRegex(ValueError, "omits required files: " + name):
                    self.check()
        self.assertIn("scripts/render-release-evidence.sh", CHECK.REQUIRED_SOURCE_FILES)

    def test_required_source_files_must_exist_and_stay_regular(self) -> None:
        script = self.root / "scripts/check-batched-service.py"
        script.unlink()
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            self.check()
        script.symlink_to(self.root / "scripts/source_inputs.py")
        with self.assertRaisesRegex(ValueError, "not a regular file|symlink"):
            self.check()

    def test_source_manifest_must_be_a_v2_record(self) -> None:
        self.edit(SOURCE_MANIFEST, lambda v: v.update(schema_version="leone.source-inputs.v1"))
        with self.assertRaisesRegex(ValueError, "unknown schema"):
            self.check()
        self.reset()
        self.edit(SOURCE_MANIFEST, lambda v: v.update(source_commit="d1" * 20))
        with self.assertRaisesRegex(ValueError, "study source commit differs"):
            self.check()

    def test_workload_file_hash_must_match_when_the_plan_is_packaged(self) -> None:
        (self.root / CHECK.PLAN_PATH).write_bytes(b"changed\n")
        with self.assertRaisesRegex(ValueError, "measured source input changed"):
            self.check()

    def test_symlinked_inputs_are_rejected(self) -> None:
        real = self.root / "real-study.json"
        (self.root / STUDY_PATH).rename(real)
        (self.root / STUDY_PATH).symlink_to(real)
        with self.assertRaisesRegex(ValueError, "study receipt contains a symlink"):
            self.check()
        self.reset()
        (self.root / SOURCE_MANIFEST).rename(self.root / "real-source.json")
        (self.root / SOURCE_MANIFEST).symlink_to(self.root / "real-source.json")
        with self.assertRaisesRegex(ValueError, "source input manifest contains a symlink"):
            self.check()

    def test_receipt_outside_the_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            outside = Path(other) / "study.json"
            write_json(outside, self.study)
            with self.assertRaisesRegex(ValueError, "outside its root"):
                CHECK.validate(self.root, outside, expected(), offline=True)


class CommandLineTests(CheckerCase):
    def command(self, **changes) -> list[str]:
        record = {
            "path": STUDY_PATH,
            **IDENTITY,
            "model_sha256": MODEL,
            "binary_sha256": BINARY,
            "quality_record": COMPARISON_PATH,
            "quality_receipt": QUALITY_PATH,
            **changes,
        }
        return DISPATCH._command(self.root, ROOT, {"validator": "batched-service-v1", **record}, None, COMMIT, SOURCE_MANIFEST, "0" * 64)

    def test_dispatcher_runs_the_real_checker(self) -> None:
        DISPATCH._run(self.command(), ROOT)

    def test_dispatcher_rejects_a_tampered_study(self) -> None:
        self.study["summary"]["aggregate_throughput_ratio"]["median"] = 9.0
        self.rewrite()
        with self.assertRaisesRegex(ValueError, "trusted evidence validator failed"):
            DISPATCH._run(self.command(), ROOT)

    def test_dispatcher_requires_the_release_bindings(self) -> None:
        for key in ("model_sha256", "binary_sha256", "quality_record", "quality_receipt"):
            with self.subTest(key):
                record = {
                    "validator": "batched-service-v1",
                    "path": STUDY_PATH,
                    **IDENTITY,
                    "model_sha256": MODEL,
                    "binary_sha256": BINARY,
                    "quality_record": COMPARISON_PATH,
                    "quality_receipt": QUALITY_PATH,
                }
                del record[key]
                with self.assertRaisesRegex(ValueError, f"has no {key}"):
                    DISPATCH._command(self.root, ROOT, record, None, COMMIT, SOURCE_MANIFEST, "0" * 64)

    def test_command_line_reports_pass_and_failure(self) -> None:
        script = str(ROOT / "scripts/check-batched-service.py")
        command = [sys.executable, script, *self.command()[2:]]
        passed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertIn("validated", passed.stdout)
        self.study["workload"]["max_tokens"] = 32
        self.rewrite()
        failed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 1)
        self.assertIn("batched-service validation failed", failed.stderr)


def manifest() -> dict:
    return json.loads(json.dumps(EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")))


def batching_record(candidate: dict) -> dict:
    records = [
        record
        for backend in candidate["backend_requirements"]
        for record in backend["records"]
        if record["role"] == "batching"
    ]
    return records[0]


class ManifestGateTests(unittest.TestCase):
    def test_manifest_declares_one_linux_cuda_batching_record(self) -> None:
        candidate = manifest()
        EVIDENCE.validate(candidate)
        batching = [
            (backend["backend"], record)
            for backend in candidate["backend_requirements"]
            for record in backend["records"]
            if record["role"] == "batching"
        ]
        self.assertEqual([backend for backend, _ in batching], ["cuda"])
        record = batching[0][1]
        self.assertEqual(record["path"], STUDY_PATH)
        self.assertEqual(record["schema_version"], "leone.batched-service-study.v1")
        self.assertEqual(record["validator"], "batched-service-v1")
        self.assertEqual(record["model_sha256"], MODEL)
        self.assertIn("scripts/check-batched-service.py", candidate["trusted_validators"])
        destinations = {entry["destination"] for entry in candidate["files"]}
        self.assertIn(STUDY_PATH, destinations)
        self.assertIn(QUALITY_PATH, destinations)
        self.assertEqual(record["quality_receipt"], QUALITY_PATH)
        self.assertIn(QUALITY_PATH, record["dependencies"])
        self.assertIn("scripts/render-release-evidence.sh", record["dependencies"])
        runtime_quality = "receipts/v04-linux-cuda-quality-qwen3.json"
        self.assertIn(runtime_quality, destinations)
        self.assertNotEqual(record["quality_receipt"], runtime_quality)
        self.assertIn("receipts/batched-service-study.json", destinations)

    def test_both_existing_service_records_remain(self) -> None:
        candidate = manifest()
        paths = {
            record["path"]
            for backend in candidate["backend_requirements"]
            for record in backend["records"]
            if record["role"] == "service"
        }
        self.assertEqual(paths, {"receipts/v04-linux-cuda-service.json", "receipts/v04-darwin-metal-service.json"})
        self.assertEqual([record["role"] for record in candidate["shared_records"]].count("metrics"), 1)

    def test_batching_slot_cannot_be_omitted_or_moved(self) -> None:
        omitted = manifest()
        omitted["backend_requirements"][0]["records"].remove(batching_record(omitted))
        with self.assertRaisesRegex(ValueError, "exactly these roles"):
            EVIDENCE.validate(omitted)
        metal = manifest()
        moved = copy.deepcopy(batching_record(metal))
        moved["path"] = "receipts/v04-darwin-metal-batched-service.json"
        metal["backend_requirements"][1]["records"].append(moved)
        metal["files"].append({"source": moved["path"], "destination": moved["path"]})
        with self.assertRaisesRegex(ValueError, "batching record must be receipts/v04-linux-cuda"):
            EVIDENCE.validate(metal)
        duplicate = manifest()
        copied = copy.deepcopy(batching_record(duplicate))
        copied["path"] = "receipts/v04-linux-cuda-batched-service-copy.json"
        duplicate["backend_requirements"][0]["records"].append(copied)
        duplicate["files"].append({"source": copied["path"], "destination": copied["path"]})
        with self.assertRaisesRegex(ValueError, "batching record must be receipts/v04-linux-cuda"):
            EVIDENCE.validate(duplicate)

    def test_batching_record_binds_its_contract(self) -> None:
        cases = (
            ("validator", lambda r, m: r.update(validator="concurrent-service-v1"), "does not match its role"),
            ("unknown validator", lambda r, m: r.update(validator="batched-service-v2"), "unknown validator"),
            ("schema", lambda r, m: r.update(schema_version="leone.batched-service-study.v2"), "must be receipts/v04-linux-cuda-batched-service.json"),
            ("path", lambda r, m: (r.update(path="receipts/other.json"), m["files"].append({"source": "receipts/other.json", "destination": "receipts/other.json"})), "must be receipts/v04-linux-cuda-batched-service.json"),
            ("quality receipt", lambda r, m: r.pop("quality_receipt"), "must bind"),
            ("full row receipt", lambda r, m: r.update(quality_receipt="receipts/v04-linux-cuda-quality-qwen3.json"), "must bind"),
            ("quality record", lambda r, m: r.pop("quality_record"), "must bind"),
            ("other quality record", lambda r, m: r.update(quality_record="receipts/v04-linux-cuda-quality-comparison-llama.json"), "must bind"),
            ("family", lambda r, m: r.update(model_family="llama"), "unexpected model family"),
            ("model", lambda r, m: r.update(model_sha256="0" * 64), "unreviewed model SHA-256"),
            ("checker not trusted", lambda r, m: m["trusted_validators"].remove("scripts/check-batched-service.py"), "omits trusted validators"),
        )
        for label, mutate, message in cases:
            with self.subTest(label):
                candidate = manifest()
                mutate(batching_record(candidate), candidate)
                with self.assertRaisesRegex(ValueError, message):
                    EVIDENCE.validate(candidate)

    def test_batching_dependencies_are_all_required(self) -> None:
        for dependency in sorted(EVIDENCE.V04_BATCHING_DEPENDENCIES):
            with self.subTest(dependency):
                candidate = manifest()
                record = batching_record(candidate)
                record["dependencies"].remove(dependency)
                with self.assertRaises(ValueError):
                    EVIDENCE.validate(candidate)

    def test_source_inputs_cover_the_checker_and_the_study_record(self) -> None:
        inputs = CHECK.source_inputs
        self.assertIn("scripts/check-batched-service.py", inputs.V04_EXECUTION_INPUTS)
        self.assertIn("scripts/check-batched-service.py", inputs.V04_REQUIRED_FILES)
        self.assertIn(STUDY_PATH, inputs.V04_EVIDENCE_INPUTS)
        self.assertIn("scripts/render-release-evidence.sh", inputs.V04_EXECUTION_INPUTS)
        self.assertIn("scripts/render-release-evidence.sh", inputs.V04_REQUIRED_FILES)
        for name in CHECK.REQUIRED_SOURCE_FILES:
            self.assertIn(name, inputs.V04_EXECUTION_INPUTS)

    def test_release_check_selects_the_manifest_batching_record(self) -> None:
        if shutil.which("jq") is None:
            self.skipTest("jq is not installed")
        script = (ROOT / "scripts/check-release.sh").read_text(encoding="utf-8")
        start = script.index("batching_study=$(jq -er '") + len("batching_study=$(jq -er '")
        program = script[start:script.index("' \"$release_manifest\")", start)]
        candidate = manifest()
        selected = subprocess.run(["jq", "-er", program], input=json.dumps(candidate), capture_output=True, text=True)
        self.assertEqual(selected.stdout.strip(), STUDY_PATH)
        candidate["backend_requirements"][0]["records"].remove(batching_record(candidate))
        missing = subprocess.run(["jq", "-er", program], input=json.dumps(candidate), capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("needs one CUDA batching record", missing.stderr)


if __name__ == "__main__":
    unittest.main()
