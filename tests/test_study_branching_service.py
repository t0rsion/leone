"""Tests for bounded timing and branching-study provenance."""

import contextlib
import copy
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_quality_cuda as QC  # noqa: E402
SPEC = importlib.util.spec_from_file_location(
    "study_branching_service", ROOT / "scripts" / "study-branching-service.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)


def calibration_manifest():
    return json.loads((ROOT / "benchmarks/branching-service-calibration.json").read_text())


def _token_sha256(values):
    return HARNESS.sha256_bytes(b"".join(value.to_bytes(4, "little") for value in values))


def _receipt(claim):
    return {
        "claim": claim,
        "public_key_ed25519": "0" * 64,
        "signature_ed25519": "0" * 128,
    }


class LatencyTests(unittest.TestCase):
    def test_ttft_and_p95_itl_use_client_token_boundaries(self):
        metrics = HARNESS.latency_metrics(
            1_000_000,
            [1_250_000, 1_500_000, 1_900_000],
            True,
            token_boundary_ns=[1_250_000, 1_500_000, 1_900_000],
        )
        self.assertEqual(metrics["ttft_ms"]["value"], 0.25)
        self.assertEqual(metrics["content_receive_ns"], [1_250_000, 1_500_000, 1_900_000])
        self.assertEqual(metrics["token_boundary_receive_ns"], [1_250_000, 1_500_000, 1_900_000])
        self.assertEqual(metrics["inter_token_latency_ms"]["status"], "observed")
        self.assertEqual(metrics["inter_token_latency_p95_ms"]["value"], 0.3925)
        self.assertEqual(metrics["inter_token_latency_samples_ms"], [0.25, 0.4])

    def test_server_timestamp_does_not_enter_client_itl(self):
        events = [
            ({"leone_telemetry": {"token_index": 0, "token_timestamp_ns": 7}}, 2_000_000),
            ({"leone_telemetry": {"token_index": 1, "token_timestamp_ns": 8}}, 2_400_000),
        ]
        received, verified = HARNESS._token_timing(
            events, {"token_boundary_field": "token_index"}, {"completion_tokens": 2}
        )
        metrics = HARNESS.latency_metrics(1_000_000, received, verified, received)
        self.assertEqual(received, [2_000_000, 2_400_000])
        self.assertEqual(metrics["inter_token_latency_samples_ms"], [0.4])

    def test_token_index_must_be_integer_consecutive_and_counted(self):
        method = {"token_boundary_field": "token_index"}
        duplicate = [({"leone_telemetry": {"token_index": 0}}, 1), ({"leone_telemetry": {"token_index": 0}}, 2)]
        boolean = [({"leone_telemetry": {"token_index": True}}, 1)]
        decreasing = [({"leone_telemetry": {"token_index": 2}}, 1), ({"leone_telemetry": {"token_index": 1}}, 2)]
        gap = [({"leone_telemetry": {"token_index": 0}}, 1), ({"leone_telemetry": {"token_index": 2}}, 2)]
        self.assertEqual(HARNESS._token_timing(duplicate, method, {"completion_tokens": 2}), ([], False))
        self.assertEqual(HARNESS._token_timing(boolean, method, {"completion_tokens": 1}), ([], False))
        self.assertEqual(HARNESS._token_timing(decreasing, method, {"completion_tokens": 2}), ([], False))
        self.assertEqual(HARNESS._token_timing(gap, method, {"completion_tokens": 2}), ([], False))

    def test_frozen_token_usage_rejects_missing_index_provenance(self):
        metric = {
            "token_boundaries_verified": True,
            "token_boundary_receive_ns": [10, 20],
            "token_boundary_indexes": [],
        }
        errors = HARNESS._strict_token_usage_errors(metric, metric["token_boundary_receive_ns"], 2)
        self.assertTrue(any("indexes are missing" in error for error in errors))

    def test_reuse_count_comes_from_declared_receipt_path(self):
        events = [
            ({"leone_receipt": {"session": {"reused_tokens": 17}}}, 2_000_000)
        ]
        result = HARNESS._cache_reuse(
            events,
            {"branch_method": {"reuse_count_path": ["leone_receipt", "session", "reused_tokens"]}},
        )
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["values"], [17])

    def test_reuse_count_accepts_declared_nested_cache_path(self):
        events = [
            ({"usage": {"prompt_tokens_details": {"cached_tokens": 9}}}, 2_000_000)
        ]
        result = HARNESS._cache_reuse(
            events,
            {
                "cache_method": {
                    "reuse_count_path": ["usage", "prompt_tokens_details", "cached_tokens"]
                },
                "branch_method": {"status": "unsupported"},
            },
        )
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["values"], [9])

    def test_unverified_boundaries_do_not_become_itl(self):
        metrics = HARNESS.latency_metrics(1_000_000, [1_100_000, 1_300_000], False)
        self.assertIsNone(metrics["inter_token_latency_ms"]["value"])
        self.assertEqual(metrics["inter_token_latency_ms"]["reason"], "token_boundaries_unverified")

    def test_summary_recomputes_latency_from_raw_receive_times(self):
        metrics = HARNESS.latency_metrics(
            1_000_000,
            [1_250_000, 1_500_000],
            True,
            token_boundary_ns=[1_250_000, 1_500_000],
        )
        metrics["ttft_ms"]["value"] = 99.0
        summary = HARNESS.outcome_summary(
            [{"status": "completed", "metrics": metrics}], 1
        )
        self.assertEqual(summary["ttft_ms"]["values"]["p50"], 0.25)


class StreamTests(unittest.TestCase):
    class Response:
        def __init__(self):
            self.chunks = [
                b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                b'data: [DONE]\n\n',
            ]
            self.read1_calls = []

        def read1(self, size):
            self.read1_calls.append(size)
            return self.chunks.pop(0) if self.chunks else b""

        def read(self, _size):
            raise AssertionError("buffering read was used")

    def test_stream_reader_uses_incremental_read1(self):
        response = self.Response()
        events, terminal = HARNESS._read_stream_events(response)
        self.assertTrue(terminal)
        self.assertEqual(len(events), 1)
        self.assertTrue(len(response.read1_calls) >= 2)
        self.assertTrue(all(size == HARNESS.STREAM_READ_BYTES for size in response.read1_calls))

    def test_stream_bound_has_explicit_outcome(self):
        self.assertEqual(
            HARNESS._stream_error_status(HARNESS.StudyError("SSE event history exceeds the configured bound")),
            "stream_limit_exceeded",
        )
        self.assertEqual(
            HARNESS._stream_error_status(HARNESS.StudyError("study wall-time bound exceeded")),
            "deadline_expired",
        )

    def test_stream_bound_counts_non_content_event_bytes(self):
        event = {"telemetry": "x" * 128}
        payload = f"data: {json.dumps(event)}\n\n".encode()

        class Response:
            def __init__(self):
                self.chunks = [payload]

            def read1(self, _size):
                return self.chunks.pop(0) if self.chunks else b""

        limit = len(HARNESS.canonical_json(event)) - 1
        with self.assertRaisesRegex(HARNESS.StudyError, "retained event history"):
            HARNESS._read_stream_events(Response(), max_retained_bytes=limit)

    def test_stream_serializes_each_event_once_across_chunks(self):
        values = [{"telemetry": "first"}, {"telemetry": "second"}]

        class Response:
            def __init__(self):
                self.chunks = [
                    *(f"data: {json.dumps(value)}\n\n".encode() for value in values),
                    b"data: [DONE]\n\n",
                ]

            def read1(self, _size):
                return self.chunks.pop(0) if self.chunks else b""

        with mock.patch.object(HARNESS, "canonical_json", wraps=HARNESS.canonical_json) as encode:
            events, terminal = HARNESS._read_stream_events(Response())
        self.assertTrue(terminal)
        self.assertEqual(len(events), len(values))
        self.assertEqual(encode.call_count, len(values))

        limit = sum(len(HARNESS.canonical_json(value)) for value in values) - 1
        with mock.patch.object(HARNESS, "canonical_json", wraps=HARNESS.canonical_json) as encode:
            with self.assertRaisesRegex(HARNESS.StudyError, "retained event history"):
                HARNESS._read_stream_events(Response(), max_retained_bytes=limit)
        self.assertEqual(encode.call_count, len(values))


class JsonBoundTests(unittest.TestCase):
    class Response:
        status = 200

        def __init__(self, body):
            self.body = body
            self.read_size = None

        def getheaders(self):
            return []

        def read(self, size):
            self.read_size = size
            return self.body

    class Connection:
        def __init__(self, response):
            self.response = response
            self.closed = False

        def request(self, *_args, **_kwargs):
            pass

        def getresponse(self):
            return self.response

        def close(self):
            self.closed = True

    def test_json_endpoint_reads_one_byte_past_the_limit(self):
        exact = self.Response(b"x" * HARNESS.DEFAULT_READ_BYTES)
        connection = self.Connection(exact)
        with mock.patch.object(HARNESS.http.client, "HTTPConnection", return_value=connection):
            _, _, body = HARNESS._request_json("http://127.0.0.1/status", "GET", None, 1)
        self.assertEqual(len(body), HARNESS.DEFAULT_READ_BYTES)
        self.assertEqual(exact.read_size, HARNESS.DEFAULT_READ_BYTES + 1)

        oversized = self.Response(b"x" * (HARNESS.DEFAULT_READ_BYTES + 1))
        connection = self.Connection(oversized)
        with mock.patch.object(HARNESS.http.client, "HTTPConnection", return_value=connection):
            with self.assertRaisesRegex(HARNESS.StudyError, "JSON response"):
                HARNESS._request_json("http://127.0.0.1/status", "GET", None, 1)
        self.assertEqual(oversized.read_size, HARNESS.DEFAULT_READ_BYTES + 1)


class DeadlineTests(unittest.TestCase):
    def test_run_that_finishes_after_deadline_is_rejected(self):
        manifest = {"budgets": {"repetitions": 1}, "engines": [{"id": "engine"}]}
        with mock.patch.object(HARNESS, "run_engine", return_value={"status": "completed"}), mock.patch.object(
            HARNESS.time, "monotonic", side_effect=[0.0, 2.0]
        ):
            with self.assertRaisesRegex(HARNESS.StudyError, "after run"):
                HARNESS._study_runs(manifest, 1.0, 1.0)

    def test_deadline_is_checked_before_final_receipt(self):
        manifest = {"phase": "calibration", "budgets": {"wall_time_limit_s": 1}, "engines": []}
        with mock.patch.object(HARNESS, "_engine_rows", return_value=[]), mock.patch.object(
            HARNESS, "artifact_provenance", return_value={}
        ), mock.patch.object(
            HARNESS, "_study_engines", return_value=contextlib.nullcontext({"engines": []})
        ), mock.patch.object(
            HARNESS, "_load_history_tokenization_producer", return_value=lambda *_args: {}
        ), mock.patch.object(HARNESS, "_study_runs", return_value=[]), mock.patch.object(
            HARNESS, "_study_evaluation", return_value={}
        ), mock.patch.object(HARNESS, "_study_receipt", return_value={}) as receipt, mock.patch.object(
            HARNESS.time, "monotonic", side_effect=[0.0, 2.0]
        ):
            with self.assertRaisesRegex(HARNESS.StudyError, "before final receipt"):
                HARNESS.run_study(manifest, ROOT, ROOT / "benchmarks/branching-service-calibration.json")
        receipt.assert_not_called()


class ProducerLoaderTests(unittest.TestCase):
    def test_loader_ignores_legacy_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            harness_path = root / "study-branching-service.py"
            shutil.copy2(ROOT / "scripts/study-branching-service.py", harness_path)
            official = root / "produce-history-tokenization.py"
            shutil.copy2(ROOT / "scripts/produce-history-tokenization.py", official)
            for alias in ("history_tokenizer.py", "history-tokenizer.py"):
                (root / alias).write_text("raise AssertionError('legacy producer executed')\n")
            spec = importlib.util.spec_from_file_location("copied_study_branching_service", harness_path)
            assert spec and spec.loader
            harness = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(harness)
            loaded = harness._load_history_producer_module()
            self.assertEqual(pathlib.Path(loaded.__file__).resolve(), official.resolve())
            official.unlink()
            harness._HISTORY_PRODUCER_MODULE.clear()
            with self.assertRaisesRegex(harness.StudyError, "producer is missing"):
                harness._load_history_producer_module()


class ReceiptWriterTests(unittest.TestCase):
    def test_existing_and_dangling_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            existing = root / "existing.json"
            existing.write_text("old")
            with self.assertRaises(HARNESS.StudyError):
                HARNESS._write_append_only(existing, {"new": True})
            self.assertEqual(existing.read_text(), "old")

            dangling = root / "dangling.json"
            dangling.symlink_to(root / "missing.json")
            with self.assertRaises(HARNESS.StudyError):
                HARNESS._write_append_only(dangling, {"new": True})
            self.assertTrue(dangling.is_symlink())

    def test_creation_race_does_not_remove_the_winner(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "race.json"

            def race(candidate, _flags, _mode):
                pathlib.Path(candidate).write_text("winner")
                raise FileExistsError

            with mock.patch.object(HARNESS.os, "open", side_effect=race):
                with self.assertRaises(HARNESS.StudyError):
                    HARNESS._write_append_only(path, {"new": True})
            self.assertEqual(path.read_text(), "winner")

    def test_oversized_receipt_is_rejected_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            oversized = root / "oversized.json"
            with mock.patch.object(HARNESS, "MAX_RECEIPT_BYTES", 32):
                with self.assertRaisesRegex(HARNESS.StudyError, "64 MiB"):
                    HARNESS._write_append_only(oversized, {"payload": "x" * 64})
            self.assertFalse(oversized.exists())

    def test_short_write_removes_partial_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            short = pathlib.Path(directory) / "short.json"

            class ShortWriter:
                def __init__(self, descriptor, _mode):
                    self.descriptor = descriptor

                def __enter__(self):
                    return self

                def write(self, payload):
                    return len(payload) - 1

                def __exit__(self, *_args):
                    os.close(self.descriptor)

            with mock.patch.object(HARNESS.os, "fdopen", side_effect=ShortWriter):
                with self.assertRaisesRegex(HARNESS.StudyError, "write was short"):
                    HARNESS._write_append_only(short, {"status": "completed"})
            self.assertFalse(short.exists())


class SampleTests(unittest.TestCase):
    def test_minimum_sample_count_qualifies_only_at_declared_bound(self):
        result = HARNESS.empirical_quantiles([1.0, 2.0], 2)
        self.assertTrue(result["qualified"])
        self.assertFalse(HARNESS.empirical_quantiles([1.0], 2)["qualified"])

    def test_truncation_is_reported_and_values_are_not_claimed_exact(self):
        result = HARNESS.bounded_history([1, 2, 3], 2)
        self.assertEqual(result["values"], [2, 3])
        self.assertEqual(result["dropped_count"], 1)
        self.assertEqual(result["sample_count"], 3)
        quantiles = HARNESS.empirical_quantiles([2.0, 3.0], 1, truncated=True)
        self.assertFalse(quantiles["qualified"])
        self.assertEqual(quantiles["reason"], "sample_history_truncated")

    def test_threshold_measurements_need_minimum_observed_values(self):
        records = [
            {
                "status": "completed",
                "history_complete": True,
                "request_start_ns": index,
                "request_end_ns": index + 1,
                "metrics": {"request_start_ns": index},
            }
            for index in range(12)
        ]
        records[0]["metrics"]["content_receive_ns"] = [1]
        self.assertIsNone(HARNESS._quantile_metric(records, HARNESS._record_ttft, 12))


class OutcomeTests(unittest.TestCase):
    def test_all_outcomes_are_counted(self):
        records = [
            {"status": "completed", "metrics": {"ttft_ms": {"value": 1.0}, "inter_token_latency_ms": {"status": "unavailable"}}},
            {"status": "cancelled", "metrics": {}},
            {"status": "slow_client", "metrics": {}},
            {"status": "unsupported", "metrics": {}},
        ]
        summary = HARNESS.outcome_summary(records, 1)
        self.assertEqual(summary["request_count"], 4)
        self.assertEqual(summary["outcomes"], {"cancelled": 1, "completed": 1, "slow_client": 1, "unsupported": 1})

    def test_missing_service_measurements_are_not_zero(self):
        metrics = HARNESS._service_metrics([])
        self.assertIsNone(metrics["physical_bytes_by_class"]["value"])
        self.assertEqual(metrics["physical_bytes_by_class"]["reason"], "physical_bytes_unreported")

    def test_service_measurements_keep_reported_values_and_classes_separate(self):
        events = [
            (
                {"leone_telemetry": {"fork_latency_ns": 12, "physical_bytes_by_class": {"kv_cache": 64}, "scheduler_reservation_bytes": 128}},
                1,
            ),
            ({"leone_telemetry": {}, "choices": [{"delta": {}}]}, 2),
        ]
        metrics = HARNESS._service_metrics(events)
        self.assertEqual(metrics["fork_latency_ns"]["value"], 12)
        self.assertEqual(metrics["physical_bytes_by_class"]["value"], {"kv_cache": 64})
        self.assertEqual(metrics["scheduler_reservation_bytes"]["value"], 128)


class ManifestTests(unittest.TestCase):
    def test_calibration_and_pending_template_validate(self):
        self.assertEqual(HARNESS.validate_manifest(calibration_manifest()), [])
        frozen = json.loads((ROOT / "benchmarks/branching-service-frozen.json").read_text())
        self.assertEqual(HARNESS.validate_manifest(frozen), [])
        self.assertEqual(frozen["phase"], "pending")
        calibration_hashes = {item["sha256"] for item in HARNESS.prompt_provenance(calibration_manifest())}
        frozen_hashes = {item["sha256"] for item in HARNESS.prompt_provenance(frozen)}
        self.assertTrue(calibration_hashes.isdisjoint(frozen_hashes))

    def test_invalid_frozen_manifest_requires_real_gate(self):
        body = calibration_manifest()
        body["phase"] = "frozen"
        body["freeze_status"] = "frozen"
        errors = HARNESS.validate_manifest(body)
        self.assertIn("frozen evaluation.thresholds must be a nonempty list", errors)
        self.assertIn("frozen manifests require calibration_receipt path and sha256", errors)
        self.assertIn("frozen manifests require quality_record path and sha256", errors)

    def test_pending_template_cannot_run(self):
        body = json.loads((ROOT / "benchmarks/branching-service-frozen.json").read_text())
        with self.assertRaises(HARNESS.StudyError):
            HARNESS.run_study(body, ROOT, ROOT / "benchmarks/branching-service-frozen.json")

    def test_engine_provenance_uses_executable_and_build_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            executable = root / "engine"
            executable.write_bytes(b"actual executable")
            engine = {
                "id": "test",
                "kind": "test",
                "executable_path": "engine",
                "build_info_command": [sys.executable, "-c", "import json; print(json.dumps({'git_commit':'abc'}))"],
            }
            provenance = HARNESS.engine_provenance(engine, root)
        self.assertEqual(provenance["status"], "observed")
        self.assertEqual(provenance["build_info"]["source_id"], "test:git:abc")
        self.assertEqual(provenance["executable_sha256"], HARNESS.sha256_bytes(b"actual executable"))

    def test_plan_does_not_supply_a_claimed_source_id(self):
        manifest = calibration_manifest()
        manifest["engines"][0]["source_id_hint"] = "claimed-static-id"
        self.assertTrue(any(
            "source identity must come from build_info_command" in error
            for error in HARNESS.validate_manifest(manifest)
        ))
        manifest["engines"][0].pop("source_id_hint")
        plan = HARNESS.build_plan(manifest, ROOT / "benchmarks/branching-service-calibration.json")
        self.assertNotIn("source_id", plan["engines"][0])
        self.assertNotIn("source_id_hint", plan["engines"][0])

    def test_threshold_schema_rejects_unrelated_fields_and_requires_scope(self):
        manifest = calibration_manifest()
        manifest["phase"] = "frozen"
        manifest["freeze_status"] = "frozen"
        manifest["evaluation"]["thresholds"] = {"unrelated_no_op": "passes"}
        errors = HARNESS.validate_manifest(manifest)
        self.assertIn("evaluation.thresholds must be a list", errors)
        manifest["evaluation"]["thresholds"] = [
            {
                "id": "ttft",
                "engine": "leone",
                "role": "parent",
                "scenario": "parent",
                "metric": "ttft_p95_ms",
                "model": "Qwen3-8B-Q4_K_M.gguf",
                "backend": "cuda",
                "device": "SM89",
                "operator": "<=",
                "value": 10,
                "unit": "tokens",
            }
        ]
        errors = HARNESS.validate_manifest(manifest)
        self.assertIn("evaluation.thresholds[0].unit does not match metric", errors)

    def test_manifest_paths_cannot_escape_study_root(self):
        manifest = calibration_manifest()
        manifest["artifacts"]["model"]["path"] = "../outside/model.gguf"
        self.assertTrue(any("stay within the study root" in error for error in HARNESS.validate_manifest(manifest)))
        manifest = calibration_manifest()
        manifest["engines"][0]["executable_path"] = "/etc/hosts"
        self.assertTrue(any("stay within the study root" in error for error in HARNESS.validate_manifest(manifest)))

    def test_frozen_thresholds_cover_each_engine_and_schedule_scope(self):
        manifest = calibration_manifest()
        manifest["phase"] = "frozen"
        manifest["freeze_status"] = "frozen"
        manifest["evaluation"]["thresholds"] = [{
            "id": "parent-ttft",
            "engine": "leone",
            "role": "parent",
            "scenario": "parent",
            "metric": "ttft_p95_ms",
            "model": "Qwen3-8B-Q4_K_M.gguf",
            "backend": "cuda",
            "device": "SM89",
            "operator": "<=",
            "value": 10,
            "unit": "ms",
        }]
        self.assertTrue(any("frozen thresholds miss scopes" in error for error in HARNESS.validate_manifest(manifest)))

    def test_frozen_threshold_matrix_requires_memory_and_latency_metrics(self):
        manifest = calibration_manifest()
        manifest["phase"] = "frozen"
        manifest["freeze_status"] = "frozen"
        manifest["evaluation"]["thresholds"] = [
            {
                "id": "success",
                "engine": "leone",
                "role": "parent",
                "scenario": "parent",
                "metric": "request_success_rate",
                "model": "Qwen3-8B-Q4_K_M.gguf",
                "backend": "cuda",
                "device": "SM89",
                "operator": ">=",
                "value": 0,
                "unit": "ratio",
            }
        ]
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("branch/branch/inter_token_latency_p95_ms" in error for error in errors))
        self.assertTrue(any("schedule/schedule/physical_memory_peak_bytes" in error for error in errors))

    def test_calibration_requires_two_samples_for_quantiles(self):
        manifest = calibration_manifest()
        manifest["budgets"]["repetitions"] = 1
        manifest["budgets"]["minimum_samples_for_quantiles"] = 1
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("at least 2" in error for error in errors))

    def test_boolean_thresholds_must_require_true(self):
        manifest = calibration_manifest()
        manifest["evaluation"]["thresholds"] = [{
            "id": "pressure",
            "engine": "leone",
            "role": "slow_reader",
            "scenario": "slow_reader",
            "metric": "backpressure_observed",
            "model": manifest["engines"][0]["model"],
            "backend": manifest["engines"][0]["backend"],
            "device": manifest["engines"][0]["device"],
            "operator": "==",
            "value": False,
            "unit": "boolean",
        }]
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("value true" in error for error in errors))

    def test_numeric_thresholds_reject_vacuous_directions(self):
        cases = [
            ("ttft_p95_ms", ">=", 0, "ms"),
            ("physical_memory_peak_bytes", ">=", 0, "bytes"),
            ("history_reuse_min_tokens", "<=", 10**30, "tokens"),
        ]
        for metric, operator, value, unit in cases:
            errors = HARNESS._threshold_value_errors(
                {"metric": metric, "operator": operator, "value": value, "unit": unit},
                "threshold",
            )
            self.assertTrue(errors, metric)

    def test_missing_metric_observation_does_not_fall_back_to_overlap(self):
        threshold = {"metric": "ttft_p95_ms", "engine": "leone"}
        self.assertIsNone(HARNESS._threshold_observation(threshold, [], [], 1))

    def test_calibration_reference_rejects_minimal_inert_receipt(self):
        manifest = json.loads((ROOT / "benchmarks/branching-service-frozen.json").read_text())
        calibration = {"phase": "calibration", "workload_id": "old", "prompts": []}
        errors = HARNESS._frozen_workload_errors(calibration, manifest)
        self.assertTrue(any("calibration receipt has no runs" in error for error in errors))

    def test_prompt_similarity_rejects_near_copy_frozen_workload(self):
        calibration = calibration_manifest()
        frozen = json.loads((ROOT / "benchmarks/branching-service-frozen.json").read_text())
        errors = HARNESS._prompt_disjointness_errors(calibration, frozen)
        self.assertTrue(any("not meaningfully disjoint" in error for error in errors))

    def test_selected_threshold_requires_calibration_observation(self):
        calibration = {"budgets": {"minimum_samples_for_quantiles": 1}, "runs": []}
        frozen = {"phase": "frozen", "evaluation": {"thresholds": [{
            "engine": "leone", "role": "schedule", "metric": "physical_memory_peak_bytes",
        }]}}
        errors = HARNESS._calibration_metric_support_errors(calibration, frozen)
        self.assertTrue(any("physical_memory_peak_bytes" in error for error in errors))

    def test_engine_rows_retain_every_run_identity(self):
        manifest = calibration_manifest()
        identity = {"start": {"status": "observed"}, "end": {"status": "observed"}}
        runs = [
            {"engine": "leone", "running_identity": {**identity, "run": index}}
            for index in range(2)
        ]
        row = next(item for item in HARNESS._engine_rows(manifest, runs=runs) if item["id"] == "leone")
        self.assertEqual(row["running_identities"], [run["running_identity"] for run in runs])

    def test_frozen_identity_rejects_process_swap_between_runs(self):
        def identity(source):
            payload = {
                "source_id": source,
                "executable_sha256": "a" * 64,
                "model_sha256": "b" * 64,
                "process_start_ns": 1,
            }
            return {
                "start": {"status": "observed", "identity": payload},
                "end": {"status": "observed", "identity": payload},
            }

        row = {
            "provenance": {"build_info": {}},
            "running_identities": [identity("one"), identity("two")],
        }
        engine = {"id": "test", "executable_path": "missing"}
        self.assertIn("changed between runs", HARNESS._frozen_identity_error(row, engine, ROOT, 2))

    def test_identity_fingerprint_rejects_process_restart(self):
        first = {"start": {"identity": {"source_id": "same", "process_start_ns": 1}}}
        second = {"start": {"identity": {"source_id": "same", "process_start_ns": 2}}}
        self.assertFalse(HARNESS._identities_match([first, second]))

    def test_physical_memory_observation_rejects_missing_sample(self):
        threshold = {"engine": "leone"}
        run = {"engine": "leone", "service_metrics": {"end": {"body": {
            "physical_bytes_peak_by_class": {"weights": 10, "kv": 20}
        }}}}
        self.assertEqual(HARNESS._physical_memory_peak_observation(threshold, [run], 2), None)
        self.assertEqual(HARNESS._physical_memory_peak_observation(threshold, [run, run], 2), None)
        run["service_metrics"]["end"]["body"] = {
            "memory_topology": "cpu_parent",
            "physical_tracker_ledger": "parent_memory_tracker_root",
            "physical_peak_definition": HARNESS.PHYSICAL_PEAK_DEFINITION,
            "physical_tracker_peak_bytes": {"status": "observed", "value": 9999},
            "physical_bytes_by_class": {"weights": 1},
            "collection_errors": 0,
            "collection_losses": {field: 0 for field in HARNESS.MEMORY_COLLECTION_LOSS_FIELDS - {"overflowed"}} | {"overflowed": False},
            "counter_overflowed": False,
            "degraded": False,
            "memory_topology_conflict": False,
            "physical_bytes": {"capacity": 1, "sample_count": 0, "dropped_count": 0, "values": []},
        }
        self.assertEqual(HARNESS._physical_memory_peak_observation(threshold, [run, run], 2), 9999)
        run["service_metrics"]["end"]["body"]["physical_tracker_peak_bytes"] = {
            "status": "observed", "value": {"cpu": 9999}
        }
        self.assertIsNone(HARNESS._physical_memory_peak_observation(threshold, [run, run], 2))
        run["service_metrics"]["end"]["body"]["physical_tracker_peak_bytes"] = {
            "status": "observed", "value": 9999
        }
        run["service_metrics"]["end"]["body"]["physical_tracker_ledger"] = "forged-ledger"
        self.assertIsNone(HARNESS._physical_memory_peak_observation(threshold, [run, run], 2))

    def test_physical_memory_observation_rejects_collection_loss(self):
        body = {
            "memory_topology": "cpu_parent",
            "physical_tracker_ledger": "parent_memory_tracker_root",
            "physical_peak_definition": HARNESS.PHYSICAL_PEAK_DEFINITION,
            "physical_tracker_peak_bytes": {"status": "observed", "value": 123},
            "collection_errors": 1,
            "physical_bytes": {"capacity": 1, "sample_count": 1, "dropped_count": 1, "values": []},
        }
        run = {"engine": "leone", "service_metrics": {"end": {"body": body}}}
        self.assertIsNone(HARNESS._run_physical_memory_peak(run))

    def test_complete_prefix_rejects_prompt_only_reuse(self):
        self.assertIn(
            "tokenized prefix",
            " ".join(HARNESS._history_token_reuse_value_errors([512], 576, 576)),
        )

    def test_all_error_receipt_is_rejected(self):
        manifest = calibration_manifest()
        run = {
            "engine": manifest["engines"][0]["id"],
            "repetition": 0,
            "parent": {"status": "connection_error", "metrics": {}},
            "branches": [{"status": "connection_error", "metrics": {}} for _ in manifest["prompts"]["branches"]],
            "probes": [{"role": role, "status": "connection_error", "metrics": {}} for role in ("new_prompt", "cancel", "slow_reader")],
        }
        errors = HARNESS._receipt_run_errors({"runs": [run]}, {**manifest, "budgets": {**manifest["budgets"], "repetitions": 1}, "engines": [manifest["engines"][0]]})
        self.assertIn("receipt contains no completed request evidence", errors)

    def test_receipt_validation_recomputes_summary_and_schedule(self):
        manifest_path = ROOT / "benchmarks/branching-service-calibration.json"
        manifest = calibration_manifest()
        runs = []
        metrics = HARNESS.latency_metrics(1, [2], False)

        def completed(role=None, request_id=None):
            item = {
                "status": "completed",
                "metrics": metrics,
                "request_start_ns": 1,
                "request_end_ns": 2,
                "history_complete": True,
            }
            if role is not None:
                item["role"] = role
            if request_id is not None:
                item["request_id"] = request_id
            return item

        for repetition in range(manifest["budgets"]["repetitions"]):
            for engine in manifest["engines"]:
                parent_id = f"branch-parent-{engine['id']}-{repetition}"
                branches = [
                    {
                        **completed("branch", f"{parent_id}-{branch['id']}"),
                        "branch_id": f"{parent_id}-{branch['id']}",
                    }
                    for branch in manifest["prompts"]["branches"]
                ]
                probes = [
                    completed(role, f"{parent_id}-{manifest['schedule'][role]['id']}")
                    for role in ("new_prompt", "cancel", "slow_reader")
                ]
                records = [
                    {"role": "parent", **completed("parent", parent_id)},
                    *branches,
                    *probes,
                ]
                runs.append(
                    {
                        "engine": engine["id"],
                        "engine_kind": HARNESS.engine_kind(engine),
                        "comparison_support": HARNESS.capability_record(HARNESS.engine_kind(engine)),
                        "repetition": repetition,
                        "parent": {
                            **completed("parent", parent_id),
                            "usage": {"prompt_tokens": manifest["prompts"]["parent"]["minimum_prompt_tokens"]},
                        },
                        "branches": branches,
                        "probes": probes,
                        "overlap": HARNESS._schedule_overlap(records),
                        "slot_copy": {"parent_slot": 0, "save": {}, "restore": {}, "cleanup": {"status": "not_owned"}}
                        if engine["id"] == "llama_cpp" else None,
                    }
                )
        receipt = {
            "schema_version": HARNESS.SCHEMA_VERSION,
            "phase": manifest["phase"],
            "workload_id": manifest["workload_id"],
            "manifest": {"path": str(manifest_path.relative_to(ROOT)), "canonical_sha256": HARNESS.sha256_bytes(HARNESS.canonical_json(manifest))},
            "prompts": HARNESS.prompt_provenance(manifest),
            "artifacts": HARNESS.artifact_provenance(manifest, ROOT),
            "engines": HARNESS._engine_rows(manifest, ROOT),
            "runs": runs,
            "summary": HARNESS.outcome_summary(
                HARNESS._all_records({"runs": runs}),
                manifest["budgets"]["minimum_samples_for_quantiles"],
                manifest["budgets"]["max_history"],
            ),
        }
        receipt["summary_by_engine_role"] = HARNESS.summary_by_engine_role(
            receipt,
            manifest["budgets"]["minimum_samples_for_quantiles"],
            manifest["budgets"]["max_history"],
        )
        receipt["end_to_end"] = HARNESS.receipt_end_to_end(receipt)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            path.write_text(json.dumps(receipt))
            self.assertEqual(HARNESS.validate_receipt(path, ROOT), [])
            receipt["summary"]["request_count"] = 1
            path.write_text(json.dumps(receipt))
            self.assertIn("summary does not recompute from outcomes", HARNESS.validate_receipt(path, ROOT))


class ScheduleTests(unittest.TestCase):
    def test_leone_session_header_binds_service_request_identity(self):
        self.assertEqual(
            HARNESS._service_request_header_id({"x-leone-session": "session-1"}),
            "session-1",
        )
        self.assertEqual(
            HARNESS._service_request_header_id({"x-request-id": "request-1", "x-leone-session": "session-1"}),
            "request-1",
        )

    def test_history_producer_receives_exact_request_and_raw_events(self):
        parent = {
            "service_request_id": "parent-service", "_request_bytes_hex": b"parent".hex(),
            "_raw_events": [{"event": {"id": "parent"}, "received_ns": 1}],
        }
        branch = {
            "branch_mode": "fork", "history_reuse": {},
            "service_request_id": "branch-service", "_request_bytes_hex": b"branch".hex(),
            "_raw_events": [{"event": {"id": "branch"}, "received_ns": 2}],
        }
        calls = []

        def producer(engine, parent_bytes, parent_events, branch_bytes, branch_events):
            calls.append((engine, parent_bytes, parent_events, branch_bytes, branch_events))
            return {"status": "unavailable", "reason": "test"}

        HARNESS._attach_history_tokenization({"id": "leone"}, parent, branch, producer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], b"parent")
        self.assertEqual(calls[0][3], b"branch")
        self.assertEqual(calls[0][2]["events"], parent["_raw_events"])
        self.assertEqual(calls[0][4]["events"], branch["_raw_events"])
        self.assertEqual(calls[0][2]["service_request_id"], "parent-service")
        self.assertEqual(calls[0][4]["parent_service_request_id"], "parent-service")
        self.assertEqual(branch["history_reuse"]["tokenization"]["reason"], "test")

    def test_history_producer_does_not_claim_cache_only_branch(self):
        branch = {"branch_mode": "cached_history", "history_reuse": {}}
        HARNESS._attach_history_tokenization({}, {}, branch, lambda *_: {"status": "observed"})
        self.assertEqual(
            branch["history_reuse"]["tokenization"]["reason"],
            "history_tokenization_scope_requires_explicit_fork",
        )

    def test_run_engine_carries_parent_history_and_runs_bounded_probes(self):
        manifest = calibration_manifest()
        engine = next(item for item in manifest["engines"] if item["id"] == "leone")
        calls = []
        result = _run_stubbed_leone(manifest, engine, calls)
        self.assertEqual(len(result["branches"]), 2)
        self.assertEqual({probe["role"] for probe in result["probes"]}, {"new_prompt", "cancel", "slow_reader"})
        self.assertEqual(result["overlap"], HARNESS._schedule_overlap(HARNESS._run_records(result)))
        self.assertEqual(result["sibling_progress"], HARNESS._sibling_progress(HARNESS._run_records(result)))
        branch_call = next(call for call in calls if call[1].get("leone_fork_session"))
        self.assertEqual(len(branch_call[0]["messages"]), 3)
        self.assertTrue(any(call[2].get("cancel_after_first_content") for call in calls))
        self.assertTrue(any(call[2].get("read_delay_s", 0) > 0 for call in calls))

    def test_llama_uses_same_history_cache_when_fork_is_unsupported(self):
        manifest = calibration_manifest()
        engine = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
        calls = []

        def fake_stream(_engine, body, fields, _timeout, **_options):
            calls.append((body, fields))
            return {"status": "completed", "metrics": HARNESS.latency_metrics(1, [], False), "_content_text": "answer"}

        with mock.patch.object(HARNESS, "stream_request", side_effect=fake_stream), mock.patch.object(
            HARNESS, "inspect_branch_method", return_value={"status": "unsupported", "reason": "no_fork"}
        ), mock.patch.object(HARNESS, "fetch_metrics_snapshot", return_value={"status": "unavailable"}):
            result = HARNESS.run_engine(manifest, engine, 0, 1)
        self.assertTrue(all(branch["branch_mode"] == "cached_history" for branch in result["branches"]))
        self.assertTrue(any(fields.get("cache_prompt") is True for _, fields in calls))
        self.assertEqual(len(result["branches"][0]["history_reuse"]["prefix_sha256"]), 64)

    def test_overlap_requires_recorded_request_intervals(self):
        records = [
            {"role": "branch", "request_start_ns": 10, "request_end_ns": 30},
            {"role": "new_prompt", "request_start_ns": 20, "request_end_ns": 40},
        ]
        result = HARNESS._schedule_overlap(records)
        self.assertEqual(result["status"], "observed")
        self.assertTrue(result["overlap"])
        self.assertEqual(HARNESS._schedule_overlap([{"request_start_ns": 1}])["status"], "unavailable")

    def test_overlap_requires_each_probe_to_overlap_a_branch(self):
        records = [
            {"role": "branch", "request_start_ns": 0, "request_end_ns": 100},
            {"role": "cancel", "request_start_ns": 10, "request_end_ns": 90},
            {"role": "new_prompt", "request_start_ns": 200, "request_end_ns": 300},
            {"role": "slow_reader", "request_start_ns": 400, "request_end_ns": 500},
        ]
        result = HARNESS._schedule_overlap(records)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["required_overlap"])
        self.assertFalse(HARNESS._observed_overlap({"engine": "e"}, [{"engine": "e", "overlap": result}], 1))

    def test_client_close_without_server_ack_is_not_cancelled(self):
        result = {
            "status": "cancel_acknowledgement_unavailable",
            "cancel_ack": {"status": "unavailable"},
            "backpressure": {"status": "unavailable"},
        }
        errors = HARNESS._frozen_probe_errors(
            {
                "new_prompt": {"status": "completed"},
                "cancel": result,
                "slow_reader": {"backpressure": {"status": "unavailable"}},
            }
        )
        self.assertTrue(any("cancellation acknowledgement" in error for error in errors))

    def test_frozen_probe_requires_sibling_progress_and_metrics_epoch(self):
        probes = {
            "new_prompt": {"status": "completed"},
            "cancel": {"status": "cancelled", "cancel_ack": {"status": "observed"}},
            "slow_reader": {"backpressure": {"status": "observed"}},
        }
        errors = HARNESS._frozen_probe_errors(probes, {"status": "unavailable"})
        self.assertTrue(any("sibling progress" in error for error in errors))
        run = {
            "service_metrics": {
                "start": {
                    "status": "observed",
                    "body": {
                        "workload_epoch": 1,
                        "request_count": 1,
                        "outcomes": {},
                        "requests": {"values": [], "capacity": 1, "dropped_count": 0, "sample_count": 0},
                    },
                },
                "end": {
                    "status": "observed",
                    "body": {
                        "workload_epoch": 2,
                        "request_count": 2,
                        "outcomes": {"completed": 1},
                        "requests": {"values": [], "capacity": 1, "dropped_count": 0, "sample_count": 0},
                    },
                },
            }
        }
        self.assertTrue(any("different workload epochs" in error for error in HARNESS._frozen_metrics_errors(run)))

    def test_history_reuse_binds_hashes_to_canonical_request_material(self):
        manifest = calibration_manifest()
        manifest["engines"][0]["history_tokenization"] = {
            "tokenizer_sha256": "a" * 64,
            "chat_template_sha256": "b" * 64,
            "special_tokens_policy_sha256": "c" * 64,
            "vocab_size": 100,
        }
        parent_messages = HARNESS._prompt_messages(manifest["prompts"]["parent"])
        branch_prompt = manifest["prompts"]["branches"][0]
        branch_messages = HARNESS._prompt_messages(branch_prompt)
        parent_id = "branch-parent-leone-0"
        parent = {
            "status": "completed",
            "request_id": parent_id,
            "service_request_id": parent_id,
            "content_sha256": HARNESS.sha256_bytes(b"parent answer"),
            "_content_text": "parent answer",
        }
        branch_id = "branch-parent-leone-0-memory"
        parent_body = HARNESS.request_body(manifest, parent_messages, parent_id, "leone_session")
        parent_bytes = HARNESS.canonical_json(parent_body)
        parent["request_body_sha256"] = HARNESS.sha256_bytes(parent_bytes)
        parent["_request_bytes_hex"] = parent_bytes.hex()
        body = HARNESS.request_body(
            manifest,
            HARNESS._history_messages(parent_messages, parent) + branch_messages,
            branch_id,
            "leone_session",
        )
        body.update({"leone_fork_session": "branch-parent-leone-0", "leone_session": branch_id})
        prefix = HARNESS._history_messages(parent_messages, parent)
        request = prefix + branch_messages
        prompt_ids, generated_ids = list(range(4)), [10, 11, 12]
        parent["usage"] = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated_ids)}
        evaluated_ids = [*prompt_ids, *generated_ids][:-1]
        request_ids = [*evaluated_ids, 20, 21]
        oracle = {
            "engine": "leone", "source_commit": "d" * 40,
            "executable_sha256": "e" * 64, "loaded_library_sha256": "f" * 64,
            "gguf_sha256": "1" * 64, "template_config_sha256": "b" * 64,
            "template_bytes_sha256": "2" * 64, "special_tokens_policy_sha256": "c" * 64,
            "vocab_size": 100, "apply_template_request_sha256": "3" * 64,
            "apply_template_response_sha256": "4" * 64, "tokenize_request_sha256": "5" * 64,
            "tokenize_response_sha256": "6" * 64, "tokenizer_metadata_sha256": "a" * 64,
        }
        parent_claim = {
            "request_sha256": parent["request_body_sha256"],
            "prompt_tokens_sha256": _token_sha256(prompt_ids),
            "response_tokens_sha256": _token_sha256(generated_ids),
            "transcript_sha256": _token_sha256([*prompt_ids, *generated_ids]),
            "prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
            "finish_reason": "length", "cancelled": False,
            "session": {"session_id": parent_id},
        }
        parent_receipt = _receipt(parent_claim)
        parent["response_receipt"] = parent_receipt
        parent["_response_receipt_sha256"] = HARNESS.sha256_bytes(HARNESS.canonical_json(parent_receipt))
        tokenization = {
            "schema_version": HARNESS.HISTORY_TOKEN_SCHEMA,
            "status": "observed",
            "scope": HARNESS.HISTORY_TOKEN_SCOPE,
            "parent_service_request_id": parent_id,
            "service_request_id": branch_id,
            "process_instance_id": "process-1", "workload_epoch": "epoch-1",
            "parent_receipt_sha256": parent["_response_receipt_sha256"],
            "branch_receipt_sha256": "0" * 64,
            "parent_request_sha256": parent["request_body_sha256"],
            "request_sha256": "0" * 64,
            "canonical_prefix_sha256": HARNESS.sha256_bytes(HARNESS.canonical_json(prefix)),
            "canonical_request_sha256": HARNESS.sha256_bytes(HARNESS.canonical_json(request)),
            "parent_prompt_token_ids": prompt_ids,
            "parent_generated_token_ids": generated_ids,
            "parent_evaluated_token_ids": evaluated_ids,
            "request_token_ids": request_ids,
            "parent_evaluated_token_count": len(evaluated_ids),
            "expected_reused_token_count": len(evaluated_ids),
            "observed_reused_token_count": len(evaluated_ids),
            "oracle": oracle,
        }
        branch_bytes = HARNESS.canonical_json(body)
        response = {
            "request_body_sha256": HARNESS.sha256_bytes(branch_bytes),
            "cache_reuse": {"status": "observed", "values": [len(evaluated_ids)]},
            "service_request_id": branch_id,
            "history_tokenization": tokenization,
        }
        branch = {
            "status": "completed", "request_id": branch_id, "service_request_id": branch_id,
            "branch_mode": "fork", "request_body_sha256": response["request_body_sha256"],
            "_request_bytes_hex": branch_bytes.hex(),
        }
        branch_claim = {
            "request_sha256": response["request_body_sha256"],
            "prompt_tokens_sha256": _token_sha256(request_ids),
            "prompt_tokens": len(request_ids), "generated_tokens": 1,
            "finish_reason": "length", "cancelled": False,
            "session": {"session_id": branch_id, "cached_tokens": len(evaluated_ids),
                        "reused_tokens": len(evaluated_ids)},
        }
        branch_receipt = _receipt(branch_claim)
        branch["response_receipt"] = branch_receipt
        branch["_response_receipt_sha256"] = HARNESS.sha256_bytes(HARNESS.canonical_json(branch_receipt))
        tokenization["branch_receipt_sha256"] = branch["_response_receipt_sha256"]
        tokenization["request_sha256"] = response["request_body_sha256"]
        history = HARNESS._history_check(
            parent_messages,
            branch_messages,
            parent,
            response,
            body,
            branch_prompt["id"],
        )
        branch["history_reuse"] = history
        self.assertEqual(HARNESS._history_reuse_errors(parent, branch, manifest, "leone"), [])
        tokenization["parent_generated_token_ids"][0] = 99
        self.assertTrue(any(
            "signed claim differs: response_tokens_sha256" in error
            for error in HARNESS._history_reuse_errors(parent, branch, manifest, "leone")
        ))
        tokenization["parent_generated_token_ids"][0] = 10
        history["reuse_count"]["values"] = [len(evaluated_ids) - 1]
        self.assertTrue(any(
            "does not cover tokenized prefix" in error
            for error in HARNESS._history_reuse_errors(parent, branch, manifest, "leone")
        ))
        history["reuse_count"]["values"] = [len(evaluated_ids)]
        tokenization["scope"] = "tool_call"
        self.assertTrue(any(
            "scope is unsupported" in error
            for error in HARNESS._history_reuse_errors(parent, branch, manifest, "leone")
        ))
        tokenization["scope"] = HARNESS.HISTORY_TOKEN_SCOPE
        parent["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
        usage_errors = HARNESS._history_reuse_errors(parent, branch, manifest, "leone")
        self.assertEqual(len(usage_errors), 2)
        self.assertTrue(all("parent usage" in error for error in usage_errors))
        del parent["usage"]
        self.assertEqual(
            HARNESS._history_reuse_errors(parent, branch, manifest, "leone"),
            ["frozen run parent usage or token IDs are missing"],
        )
        parent["usage"] = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(generated_ids)}
        history["request_material"]["request_messages"][0]["content"] = "mutated"
        self.assertTrue(any("canonical material" in error for error in HARNESS._history_reuse_errors(parent, branch, manifest, "leone")))

    def test_history_reuse_binds_fork_fields_and_tokenized_prefix(self):
        manifest = calibration_manifest()
        parent = {
            "status": "completed",
            "request_id": "branch-parent-leone-0",
            "usage": {"prompt_tokens": 512},
            "content_sha256": HARNESS.sha256_bytes(b"answer"),
        }
        branch = {
            "request_id": "branch-parent-leone-0-memory",
            "branch_id": "branch-parent-leone-0-memory",
            "branch_mode": "fork",
            "history_reuse": {
                "status": "observed",
                "reuse_count": {"status": "observed", "values": [1]},
                "request_material": {
                    "branch_prompt_id": "memory",
                    "prefix_messages": [],
                    "request_messages": [],
                    "request_body": {
                        "messages": [],
                        "leone_fork_session": "wrong-parent",
                        "leone_session": "wrong-branch",
                    },
                },
            },
        }
        errors = HARNESS._history_reuse_errors(parent, branch, manifest, "leone")
        self.assertTrue(any("canonical tokenization is missing" in error for error in errors))
        self.assertTrue(any("fork parent identity" in error for error in errors))
        self.assertTrue(any("fork session identity" in error for error in errors))

    def test_frozen_quality_requires_the_record_and_types_no_numeric_bound(self):
        evaluation = {"quality": "observed", "quality_record": {"path": "q/record.json", "sha256": "e" * 64},
                      "quality_policy": HARNESS.QUALITY_POLICY, "calibration_receipt_sha256": "f" * 64}
        manifest = {"evaluation": evaluation, "engines": []}
        self.assertEqual(HARNESS._frozen_quality_requirements(manifest), [
            "frozen engines must share one quality backend: cuda or metal",
            "frozen engines must declare quality labels leone and llama_cpp once each",
        ])
        evaluation["quality_record"] = None
        self.assertIn("frozen manifests require quality_record path and sha256", HARNESS._frozen_quality_requirements(manifest))

    def test_threshold_requires_recomputed_calibration_decision(self):
        threshold = {
            "id": "memory",
            "engine": "leone",
            "role": "schedule",
            "scenario": "schedule",
            "metric": "physical_memory_peak_bytes",
            "model": "model",
            "backend": "cuda",
            "device": "SM89",
            "operator": "<=",
            "value": 135.3,
            "unit": "bytes",
            "calibration": {
                "receipt_sha256": "f" * 64,
                "observation": 123.0,
                "rule": "upper_10_percent",
                "derived_value": 135.3,
            },
        }
        threshold["calibration"]["decision_sha256"] = HARNESS._threshold_decision_digest(
            threshold, threshold["calibration"]
        )
        self.assertEqual(HARNESS._threshold_calibration_shape_errors(threshold, "threshold"), [])
        threshold["value"] = 1e300
        self.assertTrue(HARNESS._threshold_calibration_shape_errors(threshold, "threshold"))
        threshold["value"] = 135.3
        threshold["calibration"]["rule"] = "lower_10_percent"
        threshold["calibration"]["derived_value"] = 110.7
        threshold["calibration"]["decision_sha256"] = HARNESS._threshold_decision_digest(
            threshold, threshold["calibration"]
        )
        self.assertTrue(HARNESS._threshold_calibration_shape_errors(threshold, "threshold"))

    def test_runtime_source_parses_documented_dirty_source_id(self):
        row = {
            "engine": "leone",
            "provenance": {
                "executable_sha256": "b" * 64,
                "build_info": {"source_id": "leone-cli:" + "a" * 40 + ":false"},
            },
            "running_identity": {"start": {"identity": {"model_sha256": "c" * 64}}},
        }
        self.assertEqual(HARNESS._quality_build_source_commit(row["provenance"]["build_info"]), "a" * 40)

    def test_sibling_progress_uses_server_pressure_interval(self):
        records = [
            {"role": "slow_reader", "request_start_ns": 0, "request_end_ns": 1_000},
            {"role": "new_prompt", "first_content_ns": 50, "metrics": {"clock_domain": "client_monotonic_ns"}},
            {"role": "cancel", "first_content_ns": 150, "metrics": {"clock_domain": "caller_monotonic_ns"}},
        ]
        pressure = {
            "status": "observed",
            "request_id": "slow",
            "interval_start_ns": 100,
            "interval_end_ns": 200,
            "clock_domain": "caller_monotonic_ns",
        }
        self.assertEqual(HARNESS._sibling_progress(records, pressure)["roles"], ["cancel"])
        self.assertEqual(
            HARNESS._sibling_progress(records, {"status": "observed"})["reason"],
            "server_backpressure_interval_missing",
        )

    def test_sibling_progress_rejects_cross_clock_intervals(self):
        records = [{
            "role": "new_prompt", "first_content_ns": 150,
            "metrics": {"clock_domain": "client_monotonic_ns"},
        }, {
            "role": "slow_reader", "request_start_ns": 1, "request_end_ns": 300,
        }]
        pressure = {
            "status": "observed", "interval_start_ns": 100, "interval_end_ns": 200,
            "clock_domain": "caller_monotonic_ns",
        }
        self.assertEqual(HARNESS._sibling_progress(records, pressure)["status"], "unavailable")

    def test_trace_sibling_progress_joins_server_ids_and_interval(self):
        records = [
            {"role": "slow_reader", "service_request_id": "slow-wire"},
            {"role": "new_prompt", "service_request_id": "new-wire"},
        ]
        trace = {
            "status": "observed",
            "body": {
                "schema_version": "leone.service-trace.v1",
                "clock_domain": "caller_monotonic_ns",
                "source_id": "source-1",
                "workload_epoch": "epoch-1",
                "process_start_ns": 1,
                "dropped_events": 0,
                "dropped_memory_samples": 0,
                "events": [
                    {"kind": "prefill_chunk", "request_id": 1, "at_ns": 100, "processed_tokens": 4},
                    {
                        "kind": "resident_decode_progress", "request_id": 2, "at_ns": 150,
                        "emitted_tokens": 1, "during_prefill_request_id": 1,
                    },
                ],
                "terminal_requests": {"values": [
                    {"request_id": "1", "wire_request_id": "slow-wire", "numeric_request_id": 1},
                    {"request_id": "2", "wire_request_id": "new-wire", "numeric_request_id": 2},
                ]},
            },
        }
        pressure = {
            "status": "observed", "request_id": "slow-wire", "workload_epoch": "epoch-1",
            "interval_start_ns": 100, "interval_end_ns": 200, "clock_domain": "caller_monotonic_ns",
        }
        metrics = {"end": {"body": {
            "workload_epoch": "epoch-1", "process_start_ns": 1, "source_id": "source-1",
        }}}
        trace["body"]["process_start_ns"] = 1
        result = HARNESS._sibling_progress(records, pressure, trace, metrics)
        self.assertEqual(result["roles"], ["new_prompt"])
        trace["body"]["events"][1]["during_prefill_request_id"] = None
        trace["body"]["events"][1]["during_prefill_request_ids"] = []
        trace["body"]["events"][1]["resident_request_ids"] = [1, 2]
        self.assertEqual(HARNESS._sibling_progress(records, pressure, trace, metrics)["roles"], ["new_prompt"])
        trace["body"]["workload_epoch"] = "foreign-epoch"
        self.assertEqual(
            HARNESS._sibling_progress(records, pressure, trace, metrics)["reason"],
            "server_trace_epoch_mismatch",
        )
        trace["body"]["workload_epoch"] = "epoch-1"
        trace["body"]["source_id"] = "foreign-source"
        self.assertEqual(
            HARNESS._sibling_progress(records, pressure, trace, metrics)["reason"],
            "server_trace_source_identity_mismatch",
        )
        trace["body"]["source_id"] = "source-1"
        mismatched_identity = {"start": {"identity": {"process_start_ns": 2}}}
        result = HARNESS._sibling_progress(records, pressure, trace, metrics, mismatched_identity)
        self.assertEqual(result["reason"], "server_trace_process_identity_mismatch")
        trace["body"]["events"][1]["at_ns"] = 250
        self.assertEqual(HARNESS._sibling_progress(records, pressure, trace, metrics)["status"], "unavailable")

    def test_trace_partial_history_cannot_prove_sibling_progress(self):
        trace = {
            "status": "observed",
            "body": {
                "schema_version": "leone.service-trace.v1", "clock_domain": "clock",
                "dropped_events": 1, "dropped_memory_samples": 0, "events": [],
            },
        }
        pressure = {
            "status": "observed", "request_id": "slow", "workload_epoch": "epoch",
            "interval_start_ns": 1, "interval_end_ns": 2, "clock_domain": "clock",
        }
        metrics = {"end": {"body": {"workload_epoch": "epoch"}}}
        self.assertEqual(
            HARNESS._sibling_progress([{"role": "slow_reader"}], pressure, trace, metrics)["reason"],
            "server_trace_partial",
        )

    def test_mixed_pressure_rows_cannot_lend_identity_or_interval(self):
        slow = {
            "role": "slow_reader",
            "status": "completed",
            "request_id": "slow-client",
            "service_request_id": "slow-service",
            "service_metrics": {"slow_client": {"value": {
                "status": "observed", "value": True, "request_id": "other-service",
                "interval_start_ns": 100, "interval_end_ns": 200,
                "clock_domain": "server-clock", "reason": "write_blocked",
            }}},
        }
        slow["backpressure"] = HARNESS._backpressure_evidence(slow)
        records = [
            {"role": "slow_reader", "metrics": {"clock_domain": "server-clock", "workload_epoch": "epoch-1"}},
            {"role": "new_prompt", "first_content_ns": 150, "metrics": {"clock_domain": "server-clock", "workload_epoch": "epoch-1"}},
        ]
        snapshot = {"end": {"body": {
            "clock_domain": "server-clock", "workload_epoch": "epoch-1",
            "slow_client_intervals": {"values": [{
                "request_id": "slow-service", "reason": "write_blocked", "start_ns": 500, "end_ns": 600,
            }]},
        }}}
        sibling = HARNESS._sibling_progress(records, slow["backpressure"])
        errors = HARNESS._frozen_pressure_errors({"slow_reader": slow}, sibling, snapshot, records)
        self.assertTrue(any("selected service evidence" in error for error in errors))
        self.assertTrue(any("sibling progress" in error for error in errors))

    def test_backpressure_requires_a_measured_interval(self):
        result = {
            "service_metrics": {
                "slow_client": {
                    "value": {"status": "observed", "value": True, "request_id": "slow"}
                }
            }
        }
        self.assertEqual(HARNESS._backpressure_evidence(result)["status"], "unavailable")
        result["service_metrics"]["slow_client"]["value"].update(
            {
                "interval_start_ns": 10,
                "interval_end_ns": 20,
                "clock_domain": "caller_monotonic_ns",
                "reason": "write_blocked",
            }
        )
        evidence = HARNESS._backpressure_evidence(result)
        self.assertEqual(evidence["status"], "observed")
        self.assertEqual(evidence["interval_end_ns"], 20)

    def test_claimed_overlap_and_progress_are_recomputed(self):
        manifest = calibration_manifest()
        run = {
            "engine": "leone",
            "repetition": 0,
            "parent": {"request_id": "branch-parent-leone-0", "status": "completed"},
            "branches": [{"branch_id": "branch-parent-leone-0-memory", "request_id": "branch-parent-leone-0-memory"},
                         {"branch_id": "branch-parent-leone-0-scheduling", "request_id": "branch-parent-leone-0-scheduling"}],
            "probes": [
                {"role": "new_prompt", "request_id": "branch-parent-leone-0-new-prompt", "request_start_ns": 30, "request_end_ns": 40},
                {"role": "cancel", "request_id": "branch-parent-leone-0-cancel-probe", "request_start_ns": 40, "request_end_ns": 50},
                {"role": "slow_reader", "request_id": "branch-parent-leone-0-slow-reader-probe", "request_start_ns": 50, "request_end_ns": 60},
            ],
            "overlap": {"status": "observed"},
        }
        errors = HARNESS._run_shape_errors(
            run, manifest, ("leone", 0), len(manifest["prompts"]["branches"]), HARNESS.REQUIRED_SCHEDULE_ROLES
        )
        self.assertTrue(any("overlap does not recompute" in error for error in errors))

    def test_backpressure_event_must_name_slow_reader(self):
        probes = {
            "new_prompt": {"status": "completed", "request_id": "new"},
            "cancel": {"status": "cancelled", "service_request_id": "cancel", "cancel_ack": {"status": "observed", "request_id": "cancel"}},
            "slow_reader": {
                "status": "completed",
                "request_id": "slow",
                "service_request_id": "slow",
                "backpressure": {"status": "observed", "request_id": "other"},
            },
        }
        records = [
            {"role": "new_prompt", "request_start_ns": 10, "request_end_ns": 20, "first_content_ns": 15},
            {"role": "cancel", "request_start_ns": 10, "request_end_ns": 20, "first_content_ns": 15},
            {"role": "slow_reader", "request_start_ns": 10, "request_end_ns": 20, "first_content_ns": 15},
        ]
        errors = HARNESS._frozen_probe_errors(probes, HARNESS._sibling_progress(records), {}, records)
        self.assertTrue(any("another request" in error for error in errors))

    def test_snapshot_cancellation_uses_terminal_service_history(self):
        snapshot = {"body": {
            "terminal_requests": {"values": [{"request_id": "chatcmpl-7", "numeric_request_id": 7}]},
            "cancellation": {"values": [{
                "request_id": "7",
                "latency_ns": {"status": "observed", "value": 42},
            }]},
        }}
        self.assertEqual(HARNESS._snapshot_cancellation(snapshot, "chatcmpl-7")["value"], 42)
        self.assertEqual(
            HARNESS._snapshot_cancellation({"body": {}}, "chatcmpl-7")["status"],
            "unavailable",
        )

    def test_snapshot_pressure_uses_transport_interval_fields(self):
        snapshot = {"body": {
            "clock_domain": "caller_monotonic_ns",
            "workload_epoch": "epoch-1",
            "slow_client_intervals": {"values": [{
                "numeric_request_id": 7,
                "wire_request_id": "chatcmpl-7",
                "start_ns": 100,
                "end_ns": 140,
                "reason": "write_blocked",
            }]},
        }}
        evidence = HARNESS._snapshot_backpressure(snapshot, "chatcmpl-7")
        self.assertEqual(evidence["status"], "observed")
        self.assertEqual(evidence["clock_domain"], "caller_monotonic_ns")
        self.assertEqual(evidence["interval_start_ns"], 100)

    def test_slow_client_interval_loss_invalidates_history(self):
        self.assertTrue(HARNESS._snapshot_history_truncated({
            "slow_client_intervals": {"dropped_count": 1},
        }))

    def test_pressure_requires_an_allowlisted_reason(self):
        result = {"service_metrics": {"slow_client": {"value": {
            "status": "observed", "value": True, "request_id": "slow",
            "interval_start_ns": 10, "interval_end_ns": 20,
            "clock_domain": "server", "reason": "client_sleep",
        }}}}
        self.assertEqual(HARNESS._backpressure_evidence(result)["status"], "unavailable")

    def test_cancel_acknowledgement_binds_response_request_id(self):
        with mock.patch.object(
            HARNESS,
            "_request_json",
            return_value=(200, {}, json.dumps({"request_id": "other", "outcome": "cancelled", "reclaimed": True})),
        ):
            result = HARNESS._cancel_acknowledgement(
                {"cancel_observation_endpoint": "/debug/{request_id}", "base_url": "http://127.0.0.1"},
                "expected",
                1,
            )
        self.assertEqual(result["status"], "unavailable")

    def test_frozen_token_history_binds_boundaries_to_usage_and_start(self):
        streamed = {
            "status": "completed",
            "request_start_ns": 1_000,
            "request_end_ns": 2_000,
            "history_complete": True,
            "token_boundary_complete": True,
            "usage": {"completion_tokens": 2},
            "metrics": {
                "request_start_ns": 1_000,
                "content_receive_ns": [1_100],
                "token_boundary_receive_ns": [900, 950],
                "token_boundaries_verified": True,
            },
        }
        run = {"parent": {"status": "completed", "usage": {"completion_tokens": 2}}, "branches": [streamed], "probes": []}
        self.assertTrue(any("token boundary timestamps are invalid" in error for error in HARNESS._frozen_token_errors(run)))
        self.assertEqual(HARNESS._frozen_token_errors({**run, "branches": []}), [])
        unmeasured = {**run, "parent": {"status": "completed", "usage": {}}, "branches": []}
        self.assertEqual(HARNESS._frozen_token_errors(unmeasured), ["frozen run token usage count is missing"])

    def test_reuse_floor_includes_parent_assistant_completion(self):
        parent = {"usage": {"prompt_tokens": 512, "completion_tokens": 64}}
        self.assertEqual(HARNESS._history_required_reuse_tokens(parent, None), 576)


class FrozenGateTests(unittest.TestCase):
    def test_frozen_receipt_keeps_quality_runtime_binding_errors(self):
        manifest = {
            "phase": "frozen",
            "calibration_receipt": {"sha256": "a" * 64},
            "evaluation": {"calibration_receipt_sha256": "a" * 64, "thresholds": []},
            "engines": [],
        }
        receipt = {"evaluation": {"thresholds": []}}
        with mock.patch.object(HARNESS, "_frozen_manifest_reference", return_value=([], {})), mock.patch.object(
            HARNESS,
            "_quality_runtime_binding_errors",
            return_value=["quality runtime identity differs"],
        ):
            errors = HARNESS._receipt_frozen_errors(receipt, manifest, ROOT)
        self.assertIn("quality runtime identity differs", errors)

    def test_bound_metrics_delta_accepts_matching_request_ids_and_outcomes(self):
        run = {
            "parent": {"status": "completed", "service_request_id": "srv-1"},
            "branches": [],
            "probes": [],
            "running_identity": {
                "start": {"identity": {"process_instance_id": "process", "source_id": "source"}},
                "end": {"identity": {"process_instance_id": "process", "source_id": "source"}},
            },
            "service_metrics": {
                "start": {
                    "status": "observed",
                    "body": {
                        "workload_epoch": "epoch",
                        "process_instance_id": "process",
                        "source_id": "source",
                        "request_count": 0,
                        "outcomes": {},
                        "requests": {"values": [], "capacity": 2, "dropped_count": 0, "sample_count": 0},
                    },
                },
                "end": {
                    "status": "observed",
                    "body": {
                        "workload_epoch": "epoch",
                        "process_instance_id": "process",
                        "source_id": "source",
                        "request_count": 1,
                        "outcomes": {"finished": 1},
                        "requests": {
                            "values": [{"request_id": "srv-1", "outcome": "finished"}],
                            "capacity": 2,
                            "dropped_count": 0,
                            "sample_count": 1,
                        },
                    },
                },
            },
        }
        memory_fields = {
            "memory_topology": "cpu_parent",
            "physical_tracker_ledger": "parent_memory_tracker_root",
            "physical_peak_definition": HARNESS.PHYSICAL_PEAK_DEFINITION,
            "physical_tracker_peak_bytes": {"status": "observed", "value": 1},
            "collection_errors": 0,
            "collection_losses": {field: 0 for field in HARNESS.MEMORY_COLLECTION_LOSS_FIELDS - {"overflowed"}} | {"overflowed": False},
            "counter_overflowed": False,
            "degraded": False,
            "memory_topology_conflict": False,
            "physical_bytes": {"capacity": 1, "sample_count": 0, "dropped_count": 0, "values": []},
        }
        for snapshot in (run["service_metrics"]["start"], run["service_metrics"]["end"]):
            snapshot["body"].update(memory_fields)
        self.assertEqual(HARNESS._frozen_metrics_errors(run), [])
        run["service_metrics"]["start"]["body"]["process_instance_id"] = "other-process"
        self.assertTrue(any("change identity: process_instance_id" in error for error in HARNESS._frozen_metrics_errors(run)))
        run["service_metrics"]["start"]["body"]["process_instance_id"] = "process"
        run["service_metrics"]["end"]["body"]["source_id"] = "other-source"
        self.assertTrue(any("change identity: source_id" in error for error in HARNESS._frozen_metrics_errors(run)))

    def test_metrics_reject_conflicting_terminal_history_spellings(self):
        body = {
            "workload_epoch": 1,
            "request_count": 0,
            "outcomes": {},
            "requests": {"values": [], "capacity": 2, "dropped_count": 0, "sample_count": 0},
            "terminal_requests": {"values": [{"request_id": "other", "outcome": "finished"}], "capacity": 2, "dropped_count": 0, "sample_count": 1},
        }
        snapshot = {"status": "observed", "body": body}
        run = {"service_metrics": {"start": snapshot, "end": snapshot}}
        self.assertTrue(any("history is missing" in error for error in HARNESS._frozen_metrics_errors(run)))

    def test_old_false_frozen_evidence_claims_fail_closed(self):
        run = {
            "engine": "leone",
            "repetition": 0,
            "parent": {"status": "completed", "token_boundary_complete": True},
            "branches": [{"status": "connection_error", "history_reuse": {"status": "observed"}}],
            "probes": [
                {"role": "new_prompt", "status": "completed", "token_boundary_complete": True},
                {"role": "cancel", "status": "cancelled", "cancel_ack": {"status": "observed"}},
                {"role": "slow_reader", "status": "connection_error", "backpressure": {"status": "observed"}},
            ],
            "overlap": {"status": "observed"},
            "sibling_progress": {"status": "observed"},
            "service_metrics": {
                "start": {"status": "observed", "body": {"workload_epoch": 1, "request_count": 0, "outcomes": {}, "requests": {"values": [], "capacity": 1, "dropped_count": 100, "sample_count": 100}}},
                "end": {"status": "observed", "body": {"workload_epoch": 1, "request_count": 1, "outcomes": {"completed": 1}, "requests": {"values": [], "capacity": 1, "dropped_count": 101, "sample_count": 101}}},
            },
        }
        errors = HARNESS._frozen_run_evidence_errors(run)
        self.assertTrue(any("branch did not complete" in error for error in errors))
        self.assertTrue(any("service metric request history is truncated" in error for error in errors))
        self.assertTrue(any("complete token boundary history" in error for error in errors))

    def test_accepted_frozen_control_reaches_whole_validator(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "tests/frozen_control_fixture.py")],
            capture_output=True, text=True, check=True,
            env={**os.environ, "TMPDIR": "/dev/shm"},
        )
        self.assertIn("frozen manifest []", result.stdout)
        self.assertIn("quality []", result.stdout)
        self.assertIn("frozen receipt []", result.stdout)

    def test_calibration_requires_canonical_manifest_context(self):
        calibration = {"phase": "calibration", "workload_id": "old", "prompts": []}
        errors = HARNESS._frozen_workload_errors(calibration, calibration_manifest(), ROOT)
        self.assertTrue(any("manifest reference is missing" in error for error in errors))

    def test_calibration_requires_minimum_successful_samples_per_role(self):
        calibration = {
            "engines": [{"id": "leone"}],
            "budgets": {"minimum_samples_for_quantiles": 12},
            "runs": [{
                "engine": "leone",
                "parent": {"status": "completed", "history_complete": True},
                "branches": [{"status": "completed", "history_complete": True}],
                "probes": [
                    {"role": "new_prompt", "status": "completed", "history_complete": True},
                    {"role": "cancel", "status": "cancelled", "history_complete": True},
                    {"role": "slow_reader", "status": "completed", "history_complete": True},
                ],
            }],
        }
        errors = HARNESS._calibration_sample_errors(calibration, 12)
        self.assertTrue(any("fewer than minimum samples" in error for error in errors))


class LlamaSlotCopyTests(unittest.TestCase):
    """Concurrent llama.cpp branches start warm through a slot copy, never through serialization."""

    @staticmethod
    def _engine():
        return next(item for item in calibration_manifest()["engines"] if item["id"] == "llama_cpp")

    @staticmethod
    def _tasks(slots):
        return [(role, f"id-{role}-{slot}", {"id_slot": slot}, {}, False, 0.0, {}, 1) for role, slot in slots]

    def _run_llama(self, save_status=200):
        manifest = calibration_manifest()
        engine = self._engine()
        events = []

        def fake_action(_engine, slot, action, filename, _timeout):
            events.append(("action", action, slot, filename))
            status = save_status if action == "save" else 200
            return {"action": action, "slot": slot, "http_status": status, "request_start_ns": 1,
                    "request_end_ns": 2, "request_hex": "7b7d", "response_hex": "7b7d"}

        def fake_stream(_engine, body, fields, _timeout, **_options):
            events.append(("stream", body.get("id_slot")))
            return {"status": "completed", "metrics": HARNESS.latency_metrics(1, [], False),
                    "_content_text": "answer"}

        def fake_parent(_engine, body, fields, _timeout, **_options):
            events.append(("parent", body.get("id_slot")))
            return {"status": "completed", "_content_text": "parent answer",
                    "content_sha256": HARNESS.sha256_bytes(b"parent answer"),
                    "metrics": {"status": "unavailable"}, "history_complete": True,
                    "cache_reuse": {"status": "unavailable", "reason": "test"}}

        with mock.patch.object(HARNESS, "nonstream_request", side_effect=fake_parent), mock.patch.object(
            HARNESS, "stream_request", side_effect=fake_stream), mock.patch.object(
            HARNESS, "_slot_action", side_effect=fake_action), mock.patch.object(
            HARNESS, "inspect_branch_method", return_value={"status": "unsupported", "reason": "no_fork"}
        ), mock.patch.object(HARNESS, "fetch_metrics_snapshot", return_value={"status": "unavailable"}):
            result = HARNESS.run_engine(manifest, engine, 0, 1)
        return result, events

    def test_history_slots_are_restored_before_the_concurrent_barrier(self):
        result, events = self._run_llama()
        actions = [event for event in events if event[0] == "action"]
        first_stream = next(index for index, event in enumerate(events) if event[0] == "stream")
        self.assertEqual(
            [event[1:3] for event in actions],
            [("save", 0), ("restore", 1), ("restore", 2), ("restore", 4)],
        )
        self.assertTrue(all(events.index(action) < first_stream for action in actions))
        self.assertEqual({event[1] for event in events if event[0] == "stream"}, {1, 2, 3, 4, 5})
        self.assertEqual(result["schedule"]["barrier"]["party_count"], 5)
        self.assertEqual(sorted(result["slot_copy"]["restore"]), ["1", "2", "4"])
        self.assertEqual(result["slot_copy"]["filename"], "branch-parent-llama_cpp-0.slot")

    def test_independent_and_slow_reader_slots_are_not_restored(self):
        tasks = self._tasks(
            [("branch", 1), ("branch", 2), ("new_prompt", 3), ("cancel", 4), ("slow_reader", 5)]
        )
        with mock.patch.object(HARNESS, "_slot_action", side_effect=lambda _e, slot, action, *_: {
            "action": action, "slot": slot, "http_status": 200,
        }):
            plan = HARNESS._prepare_slot_copies(self._engine(), "parent", {"status": "completed"}, tasks, 1)
        self.assertEqual(sorted(plan["restore"]), ["1", "2", "4"])

    def test_failed_save_skips_restores_and_leaves_the_branches_without_a_restore(self):
        result, events = self._run_llama(save_status=501)
        self.assertEqual([event[1] for event in events if event[0] == "action"], ["save"])
        self.assertEqual(result["slot_copy"]["restore"], {})
        engine = {"_slot_copy": result["slot_copy"]}
        branch = {"_request_bytes_hex": json.dumps({"id_slot": 1}).encode().hex()}
        self.assertIsNone(HARNESS._branch_slot_copy(engine, branch)["restore"])

    def test_no_copy_without_completed_parent_or_for_leone_or_single_slot(self):
        tasks = self._tasks([("branch", 1)])
        with mock.patch.object(HARNESS, "_slot_action") as action:
            self.assertIsNone(HARNESS._prepare_slot_copies(self._engine(), "p", {"status": "failed"}, tasks, 1))
            self.assertIsNone(HARNESS._prepare_slot_copies({"kind": "leone"}, "p", {"status": "completed"}, tasks, 1))
            self.assertIsNone(HARNESS._prepare_slot_copies(
                self._engine(), "p", {"status": "completed"}, self._tasks([("branch", 0)]), 1
            ))
        action.assert_not_called()

    def test_producer_receives_the_slot_copy_only_for_a_branch_outside_the_parent_slot(self):
        engine = {**self._engine(), "_slot_copy": {
            "parent_slot": 0, "save": {"action": "save"}, "restore": {"1": {"action": "restore"}},
        }}
        warm = {"branch_mode": "cached_history", "history_reuse": {}, "service_request_id": "b",
                "request_start_ns": 77, "_request_bytes_hex": json.dumps({"id_slot": 1}).encode().hex(),
                "_raw_events": [{"event": {"id": "b"}, "received_ns": 2}]}
        shared = {**warm, "history_reuse": {}, "_request_bytes_hex": json.dumps({"id_slot": 0}).encode().hex()}
        parent = {"service_request_id": "p", "_request_bytes_hex": b"parent".hex(),
                  "_raw_events": [{"event": {"id": "p"}, "received_ns": 1}]}
        calls = []

        def producer(_engine, _pb, _pe, _bb, branch_events):
            calls.append(branch_events)
            return {"status": "unavailable", "reason": "test"}

        HARNESS._attach_history_tokenization(engine, parent, warm, producer)
        HARNESS._attach_history_tokenization(engine, parent, shared, producer)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["slot_copy"]["restore"], {"action": "restore"})
        self.assertEqual(calls[0]["slot_copy"]["branch_request_start_ns"], 77)
        self.assertEqual(
            shared["history_reuse"]["tokenization"]["reason"],
            "history_tokenization_scope_requires_explicit_fork",
        )

    def test_harness_slot_copy_reaches_the_llama_producer_and_validator(self):
        producer = HARNESS._load_history_tokenization_producer()
        parent_request = {"messages": [{"role": "user", "content": "question"}], "stream": False,
                          "verbose": True, "return_tokens": True, "cache_prompt": True, "id_slot": 0}
        branch_request = {"messages": [{"role": "user", "content": "question"},
                                       {"role": "assistant", "content": "answer"},
                                       {"role": "user", "content": "follow-up"}],
                          "stream": True, "verbose": True, "cache_prompt": True, "id_slot": 1,
                          "stream_options": {"include_usage": True}}
        identity = {"process_instance_id": "process-1", "workload_epoch": "epoch-1", "model": "model.gguf"}
        parent_response = {**identity, "service_request_id": "parent-service",
                           "choices": [{"message": {"content": "answer"}}],
                           "__verbose": {"tokens": [3, 4], "prompt": "parent-rendered", "id_slot": 0,
                                         "tokens_evaluated": 2, "tokens_predicted": 2, "tokens_cached": 3,
                                         "stop_type": "limit", "truncated": False,
                                         "generation_settings": {"speculative.types": "none"}}}
        branch_response = {**identity, "service_request_id": "branch-service",
                           "choices": [{"finish_reason": "length", "delta": {}}],
                           "usage": {"prompt_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}},
                           "__verbose": {"tokens": [], "prompt": "branch-rendered", "id_slot": 1}}

        def exchange(action, slot, tokens, start, end):
            fields = ("n_saved", "n_written") if action == "save" else ("n_restored", "n_read")
            reply = {"id_slot": slot, "filename": "p.slot", fields[0]: tokens, fields[1]: 64}
            return {"action": action, "slot": slot, "http_status": 200,
                    "request_hex": HARNESS.canonical_json({"filename": "p.slot"}).hex(),
                    "response_hex": HARNESS.canonical_json(reply).hex(),
                    "request_start_ns": start, "request_end_ns": end}

        engine = {**self._engine(), "_slot_copy": {
            "parent_slot": 0, "filename": "p.slot", "save": exchange("save", 0, 3, 10, 20),
            "restore": {"1": exchange("restore", 1, 3, 30, 40)}}}
        parent = {"service_request_id": "parent-service", "request_id": "parent",
                  "_request_bytes_hex": HARNESS.canonical_json(parent_request).hex(),
                  "_raw_events": [{"event": parent_response, "received_ns": 1}]}
        branch = {"branch_mode": "cached_history", "history_reuse": {}, "request_start_ns": 50,
                  "service_request_id": "branch-service",
                  "_request_bytes_hex": HARNESS.canonical_json(branch_request).hex(),
                  "_raw_events": [{"event": branch_response, "received_ns": 60}]}
        prepared = {
            "source_commit": "d" * 40, "executable_sha256": "b" * 64, "loaded_library_sha256": "c" * 64,
            "model_sha256": "a" * 64, "tokenizer_metadata_sha256": "e" * 64,
            "tokenizer_metadata_hash_scheme": "tokenizer_config_json_bytes",
            "template_config_sha256": "f" * 64, "template_config_hash_scheme": "tokenizer_config_json_bytes",
            "special_tokens_policy": {"prompt": {"add_special": False, "parse_special": True}},
            "template_bytes_sha256": "1" * 64, "special_tokens_policy_sha256": "2" * 64,
            "vocab_size": 100, "engine": "llama.cpp", "model": "model.gguf",
        }
        tokenizer = {"parent_tokens": [1, 2], "generated_tokens": [3, 4], "branch_tokens": [1, 2, 3, 9],
                     "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {}}
        with mock.patch.dict(producer.__globals__, {
            "_prepare_engine": lambda *_a, **_k: prepared, "_run_tokenizer": lambda *_a, **_k: tokenizer,
        }):
            HARNESS._attach_history_tokenization(engine, parent, branch, producer)
        tokenization = branch["history_reuse"]["tokenization"]
        self.assertEqual(tokenization["status"], "observed", tokenization)
        copy = tokenization["oracle"]["slot_copy"]
        self.assertEqual((copy["parent_slot"], copy["branch_slot"]), (0, 1))
        self.assertEqual(HARNESS._history_slot_copy_errors(tokenization, parent, branch), [])
        tokenization["oracle"]["slot_copy"]["saved_token_count"] = 2
        self.assertTrue(HARNESS._history_slot_copy_errors(tokenization, parent, branch))

    def test_slot_action_keeps_the_query_and_retains_exchange_bytes(self):
        import http.server
        import threading

        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
                body = b'{"id_slot":1,"filename":"f.slot","n_restored":3,"n_read":9}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            engine = {"base_url": f"http://127.0.0.1:{server.server_address[1]}"}
            record = HARNESS._slot_action(engine, 1, "restore", "f.slot", 5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(seen, [("/slots/1?action=restore", b'{"filename":"f.slot"}')])
        self.assertEqual(record["http_status"], 200)
        self.assertEqual(json.loads(bytes.fromhex(record["response_hex"]))["n_restored"], 3)
        self.assertLess(record["request_start_ns"], record["request_end_ns"])

    def test_unreachable_slot_endpoint_is_a_typed_record(self):
        record = HARNESS._slot_action({"base_url": "http://127.0.0.1:9"}, 1, "save", "f.slot", 1)
        self.assertIsNone(record["http_status"])
        self.assertIn("error", record)


class HistorySlotCopyValidationTests(unittest.TestCase):
    """The frozen validator ties a slot change to a retained slot copy."""

    @staticmethod
    def _record(slot):
        body = {} if slot is None else {"id_slot": slot}
        return {"_request_bytes_hex": json.dumps(body).encode().hex()}

    @staticmethod
    def _tokenization(copy):
        oracle = {} if copy is None else {"slot_copy": copy}
        return {"parent_evaluated_token_count": 3, "oracle": oracle}

    @staticmethod
    def _copy(**changes):
        copy = {
            "parent_slot": 0, "branch_slot": 1, "filename": "f.slot", "saved_token_count": 3,
            "restored_token_count": 3, "byte_count": 9, "restore_elapsed_ns": 5,
            **{name: "a" * 64 for name in (
                "save_request_sha256", "save_response_sha256",
                "restore_request_sha256", "restore_response_sha256")},
        }
        copy.update(changes)
        return copy

    def _errors(self, parent_slot, branch_slot, copy):
        return HARNESS._history_slot_copy_errors(
            self._tokenization(copy), self._record(parent_slot), self._record(branch_slot)
        )

    def test_accepts_shared_slot_without_copy_and_changed_slot_with_copy(self):
        self.assertEqual(self._errors(None, None, None), [])
        self.assertEqual(self._errors(0, 0, None), [])
        self.assertEqual(self._errors(0, 1, self._copy()), [])

    def test_rejects_changed_slot_without_a_matching_copy(self):
        self.assertTrue(self._errors(0, 1, None))
        self.assertTrue(self._errors(0, 0, self._copy()))
        self.assertTrue(self._errors(None, 1, self._copy()))
        for changes in (
            {"branch_slot": 2}, {"parent_slot": 3}, {"saved_token_count": 2},
            {"restored_token_count": 2}, {"byte_count": 0}, {"filename": ""},
            {"restore_response_sha256": "short"},
        ):
            with self.subTest(changes=changes):
                self.assertTrue(self._errors(0, 1, self._copy(**changes)))


class ControlCase(unittest.TestCase):
    """Builds the accepted frozen control once and validates mutated copies of its receipt."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(ROOT / "tests"))
        import frozen_control_fixture

        cls.fixture = frozen_control_fixture
        cls.directory = tempfile.TemporaryDirectory()
        cls.control = frozen_control_fixture.build_control(pathlib.Path(cls.directory.name))

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def receipt(self):
        return json.loads(json.dumps(self.control["receipt"]))

    def errors(self, receipt):
        return self.fixture.rewrite_receipt(self.control, receipt)

    @staticmethod
    def llama_runs(receipt):
        return [run for run in receipt["runs"] if run["engine"] == "llama_cpp"]

    @staticmethod
    def probe(run, role):
        return next(item for item in run["probes"] if item["role"] == role)

    def assertRejected(self, receipt, text):
        errors = self.errors(receipt)
        self.assertTrue(any(text in error for error in errors), errors[:3])


WIRE_SOURCE_ROOT = pathlib.Path(os.environ.get("LEONE_WIRE_SOURCE_ROOT", ROOT))
RECEIPT_SRC = "crates/leone-receipt/src/response.rs"
METRICS_SRC = "crates/leone-cli/src/server_metrics.rs"
SERVICE_METRICS_SRC = "crates/leone/src/service_metrics.rs"
SERVER_SRC = "crates/leone-cli/src/server.rs"


def rust_wire_fields(path, name):
    """Serialized field names of one Rust struct or enum variant, from the checked-in source."""

    text = (WIRE_SOURCE_ROOT / path).read_text()
    found = re.search(r"(?:\bstruct |\n\s+)" + name + r"\b[^{;]*\{(.*?)\n\s*\}", text, re.S)
    if found is None:
        raise AssertionError(f"{name} not found in {path}")
    fields, attributes = set(), []
    for line in found.group(1).splitlines():
        line = line.strip()
        if line.startswith("#["):
            attributes.append(line)
            continue
        field = re.match(r"(?:pub(?:\(crate\))? )?(\w+):", line)
        if field and not any("serde(skip)" in item for item in attributes):
            rename = re.search(r'rename = "(\w+)"', " ".join(attributes))
            fields.add(rename.group(1) if rename else field.group(1))
        if field or not line.startswith("///"):
            attributes = []
    return fields


def signed_envelope(reused, cached):
    """One `leone_receipt` value with every field the receipt types serialize, at production types."""

    session = {"session_id": "branch-1", "reuse_class": "prefix", "cached_tokens": cached,
               "reused_tokens": reused, "replayed_tokens": 0, "computed_tokens": 2}
    claim = {
        "schema_version": 1, "receipt_id": "00000000-0000-0000-0000-000000000000",
        "created_utc": "2026-09-01T00:00:00Z", "engine_version": "0.4.0", "model_sha256": "a" * 64,
        "request_sha256": "b" * 64, "prompt_tokens_sha256": "c" * 64, "response_tokens_sha256": "d" * 64,
        "transcript_sha256": "e" * 64, "seed": 0, "prompt_tokens": 30, "generated_tokens": 1,
        "finish_reason": "length", "cancelled": False, "session": session,
    }
    return {"claim": claim, "public_key_ed25519": "0" * 64, "signature_ed25519": "0" * 128}


def production_stream(envelope):
    """The chunks `write_stream_terminal`, `emit_token_boundary`, and the usage chunk emit, in order."""

    head = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "leone"}
    return [
        {**head, "choices": [{"index": 0, "delta": {"content": "a"}, "finish_reason": None}], "usage": None},
        {**head, "choices": [], "usage": None,
         "leone_telemetry": {"token_index": 0, "engine_token_boundary_ns": 100}},
        {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}], "usage": None,
         "leone_telemetry": {"prefill_chunks": 1, "prefill_tokens": 30, "decode_quanta": 1,
                             "phase_trace": ["prefill", "decode"]},
         "leone_receipt": envelope},
        {**head, "choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 1, "total_tokens": 31}},
    ]


class ProductionSignedEnvelopeTests(unittest.TestCase):
    """Consume the signed envelope and token boundary chunks as the server serializes them."""

    def leone(self, name="frozen"):
        manifest = json.loads((ROOT / f"benchmarks/branching-service-{name}.json").read_text())
        return next(item for item in manifest["engines"] if item["id"] == "leone")

    def events(self):
        return [(event, 10 * (index + 1)) for index, event in enumerate(production_stream(signed_envelope(27, 27)))]

    def test_envelope_fields_equal_the_receipt_types(self):
        envelope = signed_envelope(27, 27)
        self.assertEqual(set(envelope), rust_wire_fields(RECEIPT_SRC, "ResponseReceipt"))
        self.assertEqual(set(envelope["claim"]), rust_wire_fields(RECEIPT_SRC, "ResponseClaim"))
        self.assertEqual(set(envelope["claim"]["session"]), rust_wire_fields(RECEIPT_SRC, "SessionReplayRecord"))
        source = (WIRE_SOURCE_ROOT / SERVER_SRC).read_text()
        self.assertIn('"leone_receipt": receipt', source)
        self.assertIn('"engine_token_boundary_ns": engine_boundary_ns', source)

    def test_both_manifests_read_the_reused_count_from_the_claim(self):
        for name in ("calibration", "frozen"):
            path = self.leone(name)["branch_method"]["reuse_count_path"]
            self.assertEqual(path, ["leone_receipt", "claim", "session", "reused_tokens"])
            engine = self.leone(name)
            self.assertEqual(HARNESS._cache_reuse(self.events(), engine)["values"], [27])

    def test_the_flattened_path_finds_nothing_in_the_signed_envelope(self):
        engine = json.loads(json.dumps(self.leone()))
        engine["branch_method"]["reuse_count_path"] = ["leone_receipt", "session", "reused_tokens"]
        self.assertEqual(HARNESS._cache_reuse(self.events(), engine)["status"], "unavailable")

    def test_token_boundary_and_engine_timing_come_from_the_production_chunks(self):
        method = self.leone()["branch_method"]
        usage = {"completion_tokens": 1}
        self.assertEqual(HARNESS._token_indexes(self.events(), method, usage), [0])
        timing = HARNESS._engine_timing(self.events(), method)
        self.assertEqual((timing["status"], timing["values"], timing["clock"]), ("observed", [100.0], "caller_monotonic_ns"))
        self.assertEqual(HARNESS._response_receipt(self.events())["claim"]["session"]["reused_tokens"], 27)


class ProductionMetricsShapeTests(ControlCase):
    """The Leone snapshots of the accepted control use only fields the server serializes."""

    def leone_run(self):
        return next(run for run in self.receipt()["runs"] if run["engine"] == "leone")

    def test_metrics_snapshot_rows_have_exactly_the_serialized_fields(self):
        body = self.leone_run()["service_metrics"]["end"]["body"]
        self.assertLessEqual(set(body), rust_wire_fields(METRICS_SRC, "MetricsWire"))
        self.assertEqual(body["clock_domain"], "caller_monotonic_ns")
        checks = (
            (body["requests"]["values"][0], METRICS_SRC, "TerminalStatus"),
            (body["slow_client_intervals"]["values"][0], METRICS_SRC, "SlowClientInterval"),
            (body["collection_losses"], METRICS_SRC, "CollectionLosses"),
            (body["cancellation"]["values"][0], SERVICE_METRICS_SRC, "RequestControlSample"),
            (body["physical_bytes"]["values"][0], SERVICE_METRICS_SRC, "PhysicalBytesSample"),
        )
        for value, path, name in checks:
            self.assertEqual(set(value), rust_wire_fields(path, name), name)
        self.assertEqual(set(body["physical_tracker_peak_bytes"]), {"value", "status", "reason"})

    def test_identity_has_exactly_the_serialized_fields(self):
        identity = self.leone_run()["running_identity"]["start"]["identity"]
        self.assertEqual(set(identity), rust_wire_fields(METRICS_SRC, "ServiceIdentity"))

    def test_trace_uses_the_serialized_fields_and_clock(self):
        trace = self.leone_run()["service_trace"]["body"]
        self.assertLessEqual(set(trace), rust_wire_fields(SERVER_SRC, "ServiceTraceResponse"))
        self.assertEqual(trace["clock_domain"], "caller_monotonic_ns")
        for event, name in zip(trace["events"], ("PrefillChunk", "ResidentDecodeProgress")):
            self.assertEqual(set(event) - {"kind"}, rust_wire_fields(SERVER_SRC, name))


class ParentUsageBindingTests(ControlCase):
    """The parent usage counts must equal the proof token IDs for every engine."""

    def test_forged_parent_usage_is_rejected_for_each_engine(self):
        for engine in ("llama_cpp", "leone"):
            for field in ("prompt_tokens", "completion_tokens"):
                receipt = self.receipt()
                run = next(item for item in receipt["runs"] if item["engine"] == engine)
                run["parent"]["usage"][field] += 1
                self.assertRejected(receipt, f"parent usage {field} differs from the proof token IDs")

    def test_parent_without_usage_is_rejected(self):
        receipt = self.receipt()
        del self.llama_runs(receipt)[0]["parent"]["usage"]
        self.assertRejected(receipt, "parent usage or token IDs are missing")


class LlamaAcceptanceTests(ControlCase):
    """The accepted control uses the declared llama.cpp engine. Each break must be rejected."""

    def test_declared_llama_engine_is_not_a_fork_and_carries_no_signed_receipt(self):
        engine = next(item for item in self.control["frozen"]["engines"] if item["id"] == "llama_cpp")
        self.assertEqual(engine["branch_method"]["status"], "unsupported")
        self.assertEqual(engine["slot_copy"], HARNESS.SLOT_COPY_DECLARATION)
        branch = self.llama_runs(self.receipt())[0]["branches"][0]
        self.assertEqual(branch["branch_mode"], "cached_history")
        self.assertNotIn("response_receipt", branch)
        self.assertEqual(self.errors(self.receipt()), [])

    def test_forged_raw_cache_count_is_rejected(self):
        receipt = self.receipt()
        branch = self.llama_runs(receipt)[0]["branches"][0]
        branch["_raw_events"][-1]["event"]["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
        self.assertRejected(receipt, "llama.cpp evidence")

    def test_forged_verbose_boundary_is_rejected(self):
        receipt = self.receipt()
        run = self.llama_runs(receipt)[0]
        run["parent"]["_raw_events"][0]["event"]["__verbose"]["tokens_cached"] += 1
        self.assertRejected(receipt, "llama.cpp evidence")

    def test_forged_tokenizer_exchange_with_matching_digest_is_rejected(self):
        receipt = self.receipt()
        branch = self.llama_runs(receipt)[0]["branches"][0]
        exchange = branch["history_reuse"]["tokenization"]["oracle"]["apply_template_tokenize"]["branch"]["tokenize"]
        forged = json.dumps({"tokens": [1, 2, 3]}).encode()
        exchange["response_hex"] = forged.hex()
        exchange["response_sha256"] = HARNESS.sha256_bytes(forged)
        self.assertRejected(receipt, "llama.cpp evidence")

    def test_forged_slot_copy_bytes_are_rejected(self):
        receipt = self.receipt()
        run = self.llama_runs(receipt)[0]
        restore = run["slot_copy"]["restore"]["1"]
        reply = json.loads(bytes.fromhex(restore["response_hex"]))
        reply["n_read"] = 1
        restore["response_hex"] = json.dumps(reply, separators=(",", ":")).encode().hex()
        self.assertRejected(receipt, "llama.cpp evidence")

    def test_missing_slot_copy_is_rejected_for_a_declared_engine(self):
        receipt = self.receipt()
        for run in self.llama_runs(receipt):
            run["slot_copy"] = None
        self.assertRejected(receipt, "run slot copy does not match the engine declaration")

    def test_serving_process_must_match_the_tokenizer_binary_and_model(self):
        receipt = self.receipt()
        for run in self.llama_runs(receipt):
            for side in ("start", "end"):
                run["running_identity"][side]["identity"]["model_sha256"] = "9" * 64
        self.assertRejected(receipt, "tokenizer model differs from the serving process")

    def test_served_alias_must_equal_the_model_artifact_and_the_wire_model(self):
        receipt = self.receipt()
        for run in self.llama_runs(receipt):
            for side in ("start", "end"):
                run["running_identity"][side]["identity"]["flags"]["alias"] = "other.gguf"
        self.assertRejected(receipt, "served alias differs from the model artifact")
        receipt = self.receipt()
        for run in self.llama_runs(receipt):
            for side in ("start", "end"):
                del run["running_identity"][side]["identity"]["flags"]["alias"]
        self.assertRejected(receipt, "served alias differs from the model artifact")
        receipt = self.receipt()
        for event in self.llama_runs(receipt)[0]["parent"]["_raw_events"]:
            event["event"]["model"] = "/private/tmp/dir/" + event["event"]["model"]
        self.assertRejected(receipt, "llama_response_model_mismatch")

    def test_served_template_must_match_the_tokenizer_template(self):
        engine = next(item for item in self.control["frozen"]["engines"] if item["id"] == "llama_cpp")
        self.assertEqual(engine["identity_provider"], "spawned_process")
        for change in ("9" * 64, None):
            receipt = self.receipt()
            for run in self.llama_runs(receipt):
                for side in ("start", "end"):
                    identity = run["running_identity"][side]["identity"]
                    identity.pop("template_sha256")
                    if change:
                        identity["template_sha256"] = change
            self.assertRejected(receipt, "served template differs from the tokenizer template")

    def test_signed_leone_receipts_are_still_required_for_leone(self):
        receipt = self.receipt()
        leone = next(run for run in receipt["runs"] if run["engine"] == "leone")
        del leone["branches"][0]["response_receipt"]
        self.assertRejected(receipt, "signed response claims are missing")

    def test_receipt_must_carry_end_to_end_accounting(self):
        receipt = self.receipt()
        del receipt["end_to_end"]
        self.assertRejected(receipt, "end-to-end accounting")

    def test_hiding_setup_from_the_accounting_is_rejected(self):
        receipt = self.receipt()
        llama = receipt["end_to_end"]["engines"]["llama_cpp"]
        llama["preparation_ns"] = 0
        llama["end_to_end_window_ns"] = llama["ready_window_ns"]
        llama["end_to_end_tokens_per_s"] = llama["ready_tokens_per_s"]
        self.assertRejected(receipt, "end-to-end accounting")

    def test_unearned_speed_claim_is_rejected(self):
        receipt = self.receipt()
        receipt["end_to_end"]["comparison"]["speed_claim"] = "llama_cpp"
        self.assertRejected(receipt, "end-to-end accounting")

    def test_control_accounting_charges_setup_to_llama_only(self):
        accounting = self.receipt()["end_to_end"]["engines"]
        self.assertEqual(accounting["leone"]["preparation_ns"], 0)
        self.assertGreater(accounting["llama_cpp"]["preparation_ns"], 0)
        self.assertLess(
            accounting["llama_cpp"]["end_to_end_tokens_per_s"], accounting["llama_cpp"]["ready_tokens_per_s"]
        )
        self.assertGreater(
            accounting["llama_cpp"]["inclusive_first_token_p95_ms"],
            accounting["llama_cpp"]["ready_first_token_p95_ms"],
        )


class SetupAccountingTests(unittest.TestCase):
    @staticmethod
    def rows(tokens, ready_ns, setup_ns, ready_ms, definition=None):
        return [{
            "status": "observed", "tokens": tokens, "ready_window_ns": ready_ns,
            "token_count_definition": definition or HARNESS.TOKEN_COUNT_DEFINITION,
            "preparation_ns": setup_ns, "ready_first_token_ms": [ready_ms, ready_ms],
            "inclusive_first_token_ms": [ready_ms + setup_ns / 1e6] * 2,
        }]

    def compare(self, leone, llama, llama_definition=None):
        engines = {
            "leone": HARNESS._engine_accounting(self.rows(*leone)),
            "llama_cpp": HARNESS._engine_accounting(self.rows(*llama, llama_definition)),
        }
        return engines, HARNESS.compare_accounting(engines)

    def test_a_win_needs_equal_verified_token_count_definitions(self):
        wins = ((100, 10**10, 0, 50.0), (200, 10**10, 10**6, 20.0))
        _, comparison = self.compare(*wins)
        self.assertEqual(comparison["speed_claim"], "llama_cpp")
        engines, comparison = self.compare(*wins, llama_definition="unverified_stop_mix")
        self.assertEqual(engines["llama_cpp"]["token_count_definition"], "unverified_stop_mix")
        self.assertIsNone(comparison["speed_claim"])
        self.assertEqual(comparison["claim_withheld"], "token_count_definitions_differ_or_unverified")
        self.assertEqual(comparison["winners"]["ready_tokens_per_s"], "llama_cpp")

    def test_natural_stops_make_a_run_definition_unverified(self):
        branch = {"status": "completed", "request_start_ns": 1_000_000, "request_end_ns": 1_100_000,
                  "finish_reason": "length", "usage": {"completion_tokens": 4},
                  "metrics": {"request_start_ns": 1_000_000, "content_receive_ns": [1_020_000]}}
        run = {"branches": [branch], "probes": [], "slot_copy": None}
        self.assertEqual(HARNESS.run_accounting(run)["token_count_definition"], HARNESS.TOKEN_COUNT_DEFINITION)
        branch["finish_reason"] = "stop"
        self.assertEqual(HARNESS.run_accounting(run)["token_count_definition"], "unverified_stop_mix")

    def test_setup_cost_reverses_a_ready_only_win(self):
        engines, comparison = self.compare((100, 10**10, 0, 50.0), (120, 10**10, 5 * 10**9, 40.0))
        self.assertGreater(engines["llama_cpp"]["ready_tokens_per_s"], engines["leone"]["ready_tokens_per_s"])
        self.assertLess(engines["llama_cpp"]["end_to_end_tokens_per_s"], engines["leone"]["end_to_end_tokens_per_s"])
        self.assertIsNone(comparison["speed_claim"])
        self.assertTrue(comparison["setup_reverses_ready_winner"])

    def test_claim_needs_every_ready_and_inclusive_win(self):
        _, comparison = self.compare((100, 10**10, 0, 50.0), (200, 10**10, 10**6, 20.0))
        self.assertEqual(comparison["speed_claim"], "llama_cpp")
        self.assertFalse(comparison["setup_reverses_ready_winner"])
        _, comparison = self.compare((200, 10**10, 0, 20.0), (100, 10**10, 10**6, 50.0))
        self.assertEqual(comparison["speed_claim"], "leone")

    def test_a_tie_or_missing_setup_names_no_faster_engine(self):
        _, comparison = self.compare((100, 10**10, 0, 50.0), (100, 10**10, 0, 50.0))
        self.assertIsNone(comparison["speed_claim"])
        engines = {"leone": {"status": "unavailable"}, "llama_cpp": {"status": "observed"}}
        self.assertIsNone(HARNESS.compare_accounting(engines)["speed_claim"])

    def test_run_accounting_starts_the_clock_at_the_slot_copy(self):
        metrics = {"request_start_ns": 1_000_000, "content_receive_ns": [1_020_000, 1_030_000]}
        branch = {"status": "completed", "request_start_ns": 1_000_000, "request_end_ns": 1_100_000,
                  "metrics": metrics, "usage": {"completion_tokens": 4}}
        exchange = {"http_status": 200, "request_start_ns": 100_000, "request_end_ns": 300_000}
        run = {"branches": [branch], "probes": [],
               "slot_copy": {"save": exchange, "restore": {"1": {**exchange, "request_start_ns": 400_000, "request_end_ns": 500_000}}}}
        result = HARNESS.run_accounting(run)
        self.assertEqual(result["preparation_ns"], 900_000)
        self.assertEqual(result["ready_first_token_ms"], [0.02])
        self.assertEqual(result["inclusive_first_token_ms"], [0.92])
        self.assertEqual(result["ready_window_ns"], 100_000)
        run["slot_copy"]["restore"]["1"]["http_status"] = 500
        self.assertEqual(HARNESS.run_accounting(run)["reason"], "slot_copy_incomplete")

    def test_engine_declaration_requires_slot_copy_for_llama_only(self):
        manifest = calibration_manifest()
        self.assertEqual(HARNESS.validate_manifest(manifest), [])
        llama = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
        leone = next(item for item in manifest["engines"] if item["id"] == "leone")
        del llama["slot_copy"]
        leone["slot_copy"] = dict(HARNESS.SLOT_COPY_DECLARATION)
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("llama.cpp needs the slot_copy declaration" in error for error in errors))
        self.assertTrue(any("apply to llama.cpp only" in error for error in errors))

    def test_frozen_template_declares_the_same_llama_setup(self):
        for name in ("calibration", "frozen"):
            manifest = json.loads((ROOT / f"benchmarks/branching-service-{name}.json").read_text())
            llama = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
            self.assertEqual(llama["slot_copy"], HARNESS.SLOT_COPY_DECLARATION)
            self.assertEqual(llama["identity_provider"], "spawned_process")
            self.assertNotIn("identity_endpoint", llama)
            self.assertEqual(HARNESS._thinking_policy_errors(llama, "llama_cpp"), [])
            self.assertEqual(HARNESS._spawn_declaration_errors(llama, "llama_cpp"), [])
            self.assertEqual(llama["thinking_policy"]["mode"], "explicit_legacy_chatml")


LEGACY_CHATML_BYTES = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + "
    "'<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def _template_llama(name):
    manifest = json.loads((ROOT / f"benchmarks/branching-service-{name}.json").read_text())
    return next(item for item in manifest["engines"] if item["id"] == "llama_cpp")


class ProspectiveTemplateTests(unittest.TestCase):
    def test_each_engine_declares_the_independent_tokenizer_inputs(self):
        for name in ("calibration", "frozen", "calibration-metal", "frozen-metal"):
            manifest = json.loads((ROOT / f"benchmarks/branching-service-{name}.json").read_text())
            for engine in manifest["engines"]:
                declaration = engine["history_tokenization"]
                self.assertEqual(declaration["producer_engine"], "llama.cpp")
                if engine["id"] == "leone":
                    self.assertNotEqual(declaration["producer_executable_path"], engine["executable_path"])
                self.assertEqual(declaration["producer_template_file"], declaration["template_file"])

    def test_frozen_leone_tokenizer_identity_pins_bind_to_the_oracle(self):
        manifest = calibration_manifest()
        declaration = next(item for item in manifest["engines"] if item["id"] == "leone")[
            "history_tokenization"
        ]
        declaration.update({
            "tokenizer_metadata_sha256": "a" * 64,
            "special_tokens_policy_sha256": "b" * 64,
            "vocab_size": 100,
            "producer_source_commit": "c" * 40,
            "producer_executable_sha256": "d" * 64,
            "producer_model_sha256": "e" * 64,
            "producer_loaded_library_sha256": "f" * 64,
        })
        oracle = {
            "tokenizer_metadata_sha256": "a" * 64,
            "special_tokens_policy_sha256": "b" * 64,
            "vocab_size": 100,
            "source_commit": "c" * 40,
            "executable_sha256": "d" * 64,
            "gguf_sha256": "e" * 64,
            "loaded_library_sha256": "f" * 64,
            "template_config_sha256": declaration["template_config_sha256"],
            "template_bytes_sha256": declaration["template_bytes_sha256"],
        }
        tokenization = {"oracle": oracle}
        self.assertEqual(
            HARNESS._history_token_provenance_errors(tokenization, manifest, "leone"), []
        )
        for declaration_field, oracle_field in (
            ("producer_source_commit", "source_commit"),
            ("producer_executable_sha256", "executable_sha256"),
            ("producer_model_sha256", "gguf_sha256"),
            ("producer_loaded_library_sha256", "loaded_library_sha256"),
            ("template_config_sha256", "template_config_sha256"),
            ("template_bytes_sha256", "template_bytes_sha256"),
        ):
            changed = copy.deepcopy(tokenization)
            changed["oracle"][oracle_field] = "0" * len(oracle[oracle_field])
            errors = HARNESS._history_token_provenance_errors(changed, manifest, "leone")
            self.assertIn(f"frozen run branch tokenizer provenance differs: {declaration_field}", errors)

    def test_missing_leone_producer_inputs_are_rejected_before_a_run(self):
        manifest = calibration_manifest()
        del next(item for item in manifest["engines"] if item["id"] == "leone")["history_tokenization"]
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("history_tokenization producer declaration is required" in error for error in errors))

    def test_missing_producer_files_are_rejected_before_engine_provenance(self):
        manifest = calibration_manifest()
        with tempfile.TemporaryDirectory() as directory:
            key = pathlib.Path(directory) / "response.key"
            key.write_bytes(bytes(range(32)))
            with mock.patch.dict(os.environ, {"LEONE_BRANCHING_SIGNING_KEY": str(key)}, clear=False):
                with self.assertRaisesRegex(HARNESS.StudyError, "history tokenizer inputs are missing"):
                    HARNESS.run_study(manifest, pathlib.Path(directory), pathlib.Path(directory) / "calibration.json")

    def test_trusted_public_key_must_derive_from_the_run_key(self):
        manifest = calibration_manifest()
        leone = next(item for item in manifest["engines"] if item["id"] == "leone")
        with tempfile.TemporaryDirectory() as directory:
            key = pathlib.Path(directory) / "response.key"
            key.write_bytes(bytes(range(32)))
            with mock.patch.dict(os.environ, {
                "LEONE_BRANCHING_SIGNING_KEY": str(key),
                "LEONE_BRANCHING_TRUSTED_PUBLIC_KEY": "0" * 64,
            }, clear=False):
                with self.assertRaises(HARNESS.StudyError):
                    HARNESS._producer_paths(pathlib.Path("/study"), leone)

    def test_template_file_is_the_exact_legacy_chatml_literal(self):
        data = (ROOT / HARNESS.LEGACY_CHATML_TEMPLATE).read_bytes()
        self.assertEqual(data, LEGACY_CHATML_BYTES.encode())
        for name in ("calibration", "frozen"):
            declaration = _template_llama(name)["history_tokenization"]
            self.assertEqual(declaration["template_bytes_sha256"], HARNESS.sha256_bytes(data))
            self.assertEqual(declaration["template_config_sha256"], HARNESS.sha256_bytes(data))
            self.assertEqual(declaration["template_mode"], "legacy")
            self.assertEqual(declaration["template_file"], HARNESS.LEGACY_CHATML_TEMPLATE)

    def test_templates_carry_no_measured_pin_before_freeze(self):
        for name in ("calibration", "frozen", "calibration-metal", "frozen-metal"):
            declaration = _template_llama(name)["history_tokenization"]
            for field in (
                "producer_executable_sha256", "producer_model_sha256",
                "producer_loaded_library_sha256", "tokenizer_metadata_sha256",
                "special_tokens_policy_sha256", "vocab_size",
            ):
                self.assertNotIn(field, declaration)

    def test_both_templates_declare_the_same_spawn_argv(self):
        argv = _template_llama("calibration")["spawn"]["argv"]
        self.assertEqual(argv, _template_llama("frozen")["spawn"]["argv"])
        pairs = set(zip(argv, argv[1:]))
        for flag in (("--parallel", "{slot_count}"), ("--slot-save-path", "{slot_save_path}"),
                     ("--cache-ram", "0"), ("--flash-attn", "on"), ("--ctx-size", "12288")):
            self.assertIn(flag, pairs)
        self.assertNotIn("--no-cache-prompt", argv)

    def test_spawn_alias_is_the_model_artifact_basename(self):
        for name in ("calibration", "frozen"):
            engine = _template_llama(name)
            argv = engine["spawn"]["argv"]
            self.assertEqual(argv[argv.index("--alias") + 1], "Qwen3-8B-Q4_K_M.gguf")
            self.assertEqual(HARNESS._spawn_declaration_errors(engine, "llama_cpp"), [])
            for change in (["--alias", "other.gguf"], []):
                changed = copy.deepcopy(engine)
                changed_argv = changed["spawn"]["argv"]
                index = changed_argv.index("--alias")
                changed_argv[index:index + 2] = change
                errors = HARNESS._spawn_declaration_errors(changed, "llama_cpp")
                self.assertTrue(any("--alias" in error for error in errors), change)
        self.assertEqual(HARNESS.argv_flags(["x", "-a", "m.gguf"])["alias"], "m.gguf")
        self.assertIsNone(HARNESS.argv_flags(["x"])["alias"])

    def test_spawn_argv_fills_every_placeholder(self):
        engine = {**_template_llama("frozen"), "slot_count": 6}
        argv = HARNESS._spawn_argv(engine, pathlib.Path("/x/llama-server"), pathlib.Path("/x/m.gguf"), pathlib.Path("/tmp/slots"))
        self.assertEqual(argv[:3], ["/x/llama-server", "--model", "/x/m.gguf"])
        self.assertFalse(any("{" in item for item in argv))
        self.assertEqual(argv[argv.index("--slot-save-path") + 1], "/tmp/slots/")

    def test_other_templates_and_reasoning_settings_are_rejected(self):
        engine = copy.deepcopy(_template_llama("frozen"))
        engine["thinking_policy"]["mode"] = "template_default"
        self.assertTrue(HARNESS._thinking_policy_errors(engine, "llama_cpp"))
        engine = copy.deepcopy(_template_llama("frozen"))
        engine["history_tokenization"]["template_mode"] = "official"
        self.assertTrue(any("legacy template" in e for e in HARNESS._thinking_policy_errors(engine, "llama_cpp")))
        for dropped in ("--jinja", "--reasoning", "--chat-template-file"):
            engine = copy.deepcopy(_template_llama("frozen"))
            argv = engine["spawn"]["argv"]
            index = argv.index(dropped)
            del argv[index:index + (1 if dropped == "--jinja" else 2)]
            self.assertTrue(any("spawn argv must pass" in e for e in HARNESS._thinking_policy_errors(engine, "llama_cpp")))

    def test_tokenizer_paths_follow_the_declared_independent_binary_and_model(self):
        engine = _template_llama("frozen")
        paths = HARNESS._producer_paths(pathlib.Path("/study"), engine)
        self.assertEqual(paths["llama_server"], "/study/" + engine["history_tokenization"]["producer_executable_path"])
        self.assertEqual(paths["model"], "/study/" + engine["history_tokenization"]["producer_model_artifact"])
        self.assertEqual(paths["template_file"], "/study/" + HARNESS.LEGACY_CHATML_TEMPLATE)
        merged = HARNESS._producer_engine({**engine, "_producer_paths": paths})
        self.assertEqual(merged["model"], paths["model"])
        changed = {**engine, "executable_path": "other/leone", "model_artifact": "other/model.gguf"}
        self.assertEqual(HARNESS._producer_paths(pathlib.Path("/study"), changed), paths)
        leone = next(item for item in json.loads((ROOT / "benchmarks/branching-service-frozen.json").read_text())["engines"]
                     if item["id"] == "leone")
        with tempfile.TemporaryDirectory() as directory:
            key = pathlib.Path(directory) / "response.key"
            key.write_bytes(bytes(range(32)))
            with mock.patch.dict(os.environ, {"LEONE_BRANCHING_SIGNING_KEY": str(key)}, clear=False):
                leone_paths = HARNESS._producer_paths(pathlib.Path("/study"), leone)
        self.assertEqual(leone_paths["llama_server"], "/study/" + leone["history_tokenization"]["producer_executable_path"])
        self.assertEqual(len(leone_paths["trusted_public_key_ed25519"]), 64)
        with self.assertRaises(HARNESS.StudyError):
            HARNESS._producer_paths(pathlib.Path("/study"), {**leone, "history_tokenization": {}})
        with self.assertRaises(ValueError):
            broken = copy.deepcopy(engine)
            broken["history_tokenization"]["producer_model_artifact"] = "../outside.gguf"
            HARNESS._producer_paths(pathlib.Path("/study"), broken)


def _probe(run, role):
    return next(item for item in run["probes"] if item["role"] == role)


def _synthetic_cancel_ack(run):
    _probe(run, "cancel")["cancel_ack"] = {"status": "observed", "request_id": "x"}


def _synthetic_cancel_latency(run):
    _probe(run, "cancel")["service_metrics"] = {"cancel_latency_ns": {"status": "observed", "value": 1}}


def _synthetic_pressure(run):
    _probe(run, "slow_reader")["backpressure"] = {"status": "observed", "request_id": "x"}


def _synthetic_sibling(run):
    run["sibling_progress"] = {"status": "observed"}


def _synthetic_snapshots(run):
    run["service_metrics"]["end"] = {"status": "observed", "body": {}}


def _synthetic_fork(run):
    run["branches"][0]["service_metrics"] = {"fork_latency_ns": {"status": "observed", "value": 1}}


def _synthetic_boundaries(run):
    run["branches"][0]["token_boundary_complete"] = True


SYNTHETIC_SHAPES = {
    "cancel_ack": _synthetic_cancel_ack, "cancel_latency": _synthetic_cancel_latency,
    "pressure": _synthetic_pressure, "sibling": _synthetic_sibling, "snapshots": _synthetic_snapshots,
    "fork": _synthetic_fork, "boundaries": _synthetic_boundaries,
}


class EvidenceContractTests(ControlCase):
    """Each engine kind has a fixed evidence contract. Unsupported means absent, never synthetic."""

    UNSUPPORTED = {
        "inter_token_latency_p95_ms", "fork_latency_p95_ms", "cancel_latency_p95_ms",
        "backpressure_observed", "physical_memory_peak_bytes",
    }

    def test_llama_contract_lists_five_unsupported_metrics_with_reasons(self):
        contract = HARNESS.capability_record("llama.cpp")
        unsupported = {name for name, entry in contract.items() if entry["status"] == "unsupported"}
        self.assertEqual(unsupported, self.UNSUPPORTED)
        self.assertTrue(all(contract[name]["reason"] for name in unsupported))
        measured = set(contract) - unsupported
        self.assertEqual(measured, {
            "ttft_p95_ms", "completion_token_count", "history_reuse_min_tokens",
            "schedule_overlap_observed", "branch_setup_ms",
        })
        self.assertTrue(all(item["status"] == "measured" for item in HARNESS.capability_record("leone").values()))

    def test_a_manifest_cannot_waive_a_leone_gate(self):
        manifest = calibration_manifest()
        leone = next(item for item in manifest["engines"] if item["id"] == "leone")
        leone["comparison_support"]["cancel_latency_p95_ms"] = {"status": "unsupported", "reason": "waived"}
        self.assertTrue(any("comparison_support differs" in error for error in HARNESS.validate_manifest(manifest)))

    def test_a_manifest_cannot_claim_llama_measures_cancel(self):
        manifest = calibration_manifest()
        llama = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
        llama["comparison_support"]["cancel_latency_p95_ms"] = {"status": "measured", "source": "invented"}
        self.assertTrue(any("comparison_support differs" in error for error in HARNESS.validate_manifest(manifest)))

    def test_a_run_cannot_downgrade_leone(self):
        receipt = self.receipt()
        leone = next(run for run in receipt["runs"] if run["engine"] == "leone")
        leone["comparison_support"]["fork_latency_p95_ms"] = {"status": "unsupported", "reason": "waived"}
        self.assertRejected(receipt, "run comparison_support differs from the fixed evidence contract")

    def test_thresholds_may_not_name_an_unsupported_metric_and_leone_keeps_all(self):
        frozen = json.loads(json.dumps(self.control["frozen"]))
        thresholds = frozen["evaluation"]["thresholds"]
        cancel = next(t for t in thresholds if t["engine"] == "leone" and t["metric"] == "cancel_latency_p95_ms")
        thresholds.append({**cancel, "id": "llama-cancel", "engine": "llama_cpp"})
        self.assertTrue(any("name unsupported metrics" in e for e in HARNESS._frozen_threshold_scope_errors(frozen)))
        frozen["evaluation"]["thresholds"] = [t for t in thresholds if t is not cancel and t["id"] != "llama-cancel"]
        self.assertTrue(any("miss scopes: leone/cancel/cancel/cancel_latency_p95_ms" in e
                            for e in HARNESS._frozen_threshold_scope_errors(frozen)))

    def test_llama_thresholds_cover_only_the_measured_metrics(self):
        metrics = {t["metric"] for t in self.control["frozen"]["evaluation"]["thresholds"] if t["engine"] == "llama_cpp"}
        self.assertEqual(metrics, {"ttft_p95_ms", "history_reuse_min_tokens", "schedule_overlap_observed"})

    SYNTHETIC = (
        ("cancel_ack", "unsupported cancel acknowledgement"),
        ("cancel_latency", "unsupported cancel latency"),
        ("pressure", "unsupported backpressure is observed"),
        ("sibling", "unsupported sibling progress"),
        ("snapshots", "unsupported service telemetry"),
        ("fork", "unsupported service telemetry"),
        ("boundaries", "token boundaries without a token marker"),
    )

    def test_synthetic_leone_shapes_are_rejected_when_declared_unsupported(self):
        for name, text in self.SYNTHETIC:
            with self.subTest(mutation=name):
                receipt = self.receipt()
                for run in self.llama_runs(receipt):
                    SYNTHETIC_SHAPES[name](run)
                self.assertRejected(receipt, text)

    def test_shared_metrics_must_recompute_from_the_retained_wire_bytes(self):
        def faster_first_token(run):
            metric = self.probe(run, "new_prompt")["metrics"]
            metric["content_receive_ns"] = [value - 1 for value in metric["content_receive_ns"]]

        def more_tokens(run):
            self.probe(run, "new_prompt")["usage"]["completion_tokens"] += 1

        def no_request_bytes(run):
            del self.probe(run, "slow_reader")["_request_bytes_hex"]

        def no_events(run):
            del self.probe(run, "new_prompt")["_raw_events"]

        cases = (
            (faster_first_token, "content receive times does not recompute"),
            (more_tokens, "usage does not recompute"),
            (no_request_bytes, "request bytes are missing"),
            (no_events, "raw stream events are missing"),
        )
        for mutate, text in cases:
            with self.subTest(mutation=mutate.__name__):
                receipt = self.receipt()
                for run in self.llama_runs(receipt):
                    mutate(run)
                self.assertRejected(receipt, text)

    def test_a_natural_stop_is_rejected_for_both_engines(self):
        for engine in ("llama_cpp", "leone"):
            receipt = self.receipt()
            run = next(item for item in receipt["runs"] if item["engine"] == engine)
            self.probe(run, "new_prompt")["finish_reason"] = "stop"
            self.assertRejected(receipt, "stopped before the token limit")

    def test_missing_fields_and_zero_counts_are_errors_not_skips(self):
        streamed = {"status": "completed", "finish_reason": "length", "usage": {"completion_tokens": 0}}
        errors = HARNESS._strict_token_record_errors(streamed, True)
        self.assertIn("frozen run lacks client content timing", errors)
        streamed["metrics"] = {"request_start_ns": 1, "content_receive_ns": [2]}
        self.assertIn("frozen run token usage count is missing", HARNESS._strict_token_record_errors(streamed, True))
        self.assertIn("frozen run token usage count is missing", HARNESS._parent_count_errors({"usage": {}}))

    def test_sse_chunks_never_become_inter_token_latency(self):
        metric = HARNESS.latency_metrics(1_000, [1_010, 1_020, 1_030], False)
        self.assertEqual(metric["inter_token_latency_p95_ms"]["status"], "unavailable")
        self.assertEqual(metric["inter_token_latency_p95_ms"]["reason"], "token_boundaries_unverified")
        forged = {"status": "completed", "usage": {"completion_tokens": 3},
                  "metrics": {**metric, "inter_token_latency_p95_ms": {"status": "observed", "value": 1.0}}}
        errors = HARNESS._strict_token_record_errors(forged, True)
        self.assertIn("frozen run claims inter-token latency without a token marker", errors)

    def test_verbose_output_is_limited_to_the_history_proof_requests(self):
        engine = next(item for item in calibration_manifest()["engines"] if item["id"] == "llama_cpp")
        timed = HARNESS._engine_request_body(engine, {"stream": True}, 1)
        proof = HARNESS._engine_request_body(engine, {"stream": True}, 1, proof=True)
        self.assertNotIn("verbose", timed)
        self.assertIs(proof["verbose"], True)
        self.assertEqual(timed["reasoning_format"], "none")
        self.assertNotIn("chat_template_kwargs", timed)

    def test_llama_manifest_needs_the_pinned_thinking_policy(self):
        manifest = calibration_manifest()
        llama = next(item for item in manifest["engines"] if item["id"] == "llama_cpp")
        self.assertEqual(llama["prompt_prefix"], "")
        llama["thinking_policy"]["reasoning_format"] = "auto"
        del llama["thinking_policy"]["generation_prompt_suffix"]
        errors = HARNESS.validate_manifest(manifest)
        self.assertTrue(any("thinking_policy" in error for error in errors))

    def test_recorded_receipt_evidence_class_is_retained_and_recomputed(self):
        receipt = self.receipt()
        branch = self.llama_runs(receipt)[0]["branches"][0]
        oracle = branch["history_reuse"]["tokenization"]["oracle"]
        self.assertEqual(oracle["evidence_class"], "retained_and_recomputed")
        oracle["evidence_class"] = "independently_reexecuted"
        self.assertRejected(receipt, "evidence class must be retained_and_recomputed")


class BuildSourceTests(unittest.TestCase):
    PINNED = "d7fa69b7de21" + "0" * 28
    STDERR = b"version: 1 (build 9, commit d7fa69b)\nbuilt with cc for x86_64\n"

    def test_llama_version_line_on_stderr_binds_to_the_pinned_commit(self):
        source = HARNESS._build_info_source_id("llama.cpp", b"", self.STDERR, self.PINNED)
        self.assertEqual(source, f"llama.cpp:git:{self.PINNED}")

    def test_a_commit_outside_the_pinned_prefix_or_no_pin_gives_no_source(self):
        other = b"version: 1 (build 9, commit abcdef0)\n"
        self.assertIsNone(HARNESS._build_info_source_id("llama.cpp", b"", other, self.PINNED))
        self.assertIsNone(HARNESS._build_info_source_id("llama.cpp", b"", self.STDERR, None))
        self.assertIsNone(HARNESS._build_info_source_id("llama.cpp", b"", b"", self.PINNED))

    def test_build_result_reports_the_observed_source(self):
        completed = subprocess.CompletedProcess([], 0, stdout=b"", stderr=self.STDERR)
        result = HARNESS._build_info_result({"kind": "llama.cpp"}, completed, self.PINNED)
        self.assertEqual(result["source_id_status"], "observed")
        self.assertEqual(result["source_id"], f"llama.cpp:git:{self.PINNED}")
        self.assertEqual(HARNESS._build_info_result({"kind": "llama.cpp"}, completed)["source_id_status"], "unavailable")

    def test_the_source_id_form_is_accepted_by_the_quality_binding(self):
        self.assertEqual(HARNESS._quality_source_id_commit(f"llama.cpp:git:{self.PINNED}"), self.PINNED)


class LocalProcessIdentityTests(unittest.TestCase):
    """A llama.cpp server has no identity endpoint, so the harness reads the OS."""

    SOURCE = "llama.cpp:git:" + "d" * 40
    SCRIPT = (
        "import socket,sys,time\n"
        "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(1)\n"
        "print(s.getsockname()[1], flush=True); time.sleep(30)\n"
    )

    def identify(self, arguments, cwd=None):
        child = subprocess.Popen(
            [sys.executable, "-c", self.SCRIPT, *arguments], stdout=subprocess.PIPE, text=True, cwd=cwd
        )
        try:
            port = int(child.stdout.readline())
            engine = {"id": "llama_cpp", "base_url": f"http://127.0.0.1:{port}",
                      "identity_provider": "local_process", "_build_source_id": self.SOURCE}
            return child.pid, HARNESS.fetch_server_identity(engine, 1)
        finally:
            child.kill()
            child.wait()
            child.stdout.close()

    def test_listening_process_is_identified_with_its_binary_model_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            model = pathlib.Path(directory) / "tiny.gguf"
            model.write_bytes(b"tiny model bytes")
            pid, result = self.identify(["-m", str(model), "--parallel", "6", "--slot-save-path", directory])
        identity = result["identity"]
        self.assertEqual(result["status"], "observed")
        self.assertEqual(identity["model_sha256"], HARNESS.sha256_bytes(b"tiny model bytes"))
        self.assertEqual(identity["executable_sha256"], HARNESS.sha256_file(pathlib.Path(sys.executable).resolve()))
        self.assertEqual(identity["workload_epoch"], identity["process_instance_id"])
        self.assertIn(f":{pid}:", identity["process_instance_id"])
        self.assertEqual(identity["source_id"], self.SOURCE)
        self.assertEqual(identity["flags"]["parallel"], 6)
        self.assertIs(identity["flags"]["slot_save_path"], True)
        self.assertTrue(HARNESS._valid_digest(identity["argv_sha256"]))
        self.assertTrue(HARNESS._identity_fields_valid(identity))

    def test_a_relative_model_path_resolves_against_the_process_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            (pathlib.Path(directory) / "rel.gguf").write_bytes(b"relative model")
            _, result = self.identify(["-m", "rel.gguf"], cwd=directory)
        self.assertEqual(result["identity"]["model_sha256"], HARNESS.sha256_bytes(b"relative model"))

    def test_a_host_without_proc_reports_a_typed_unavailable_state(self):
        engine = {"id": "llama_cpp", "base_url": "http://127.0.0.1:9", "identity_provider": "local_process",
                  "_build_source_id": self.SOURCE}
        with mock.patch.object(HARNESS, "_proc_available", return_value=False):
            self.assertEqual(HARNESS.fetch_server_identity(engine, 1)["reason"], "proc_filesystem_unavailable")
        with mock.patch.object(HARNESS.Path, "iterdir", side_effect=FileNotFoundError):
            self.assertIsNone(HARNESS._socket_owner_pid({"1"}))

    def test_missing_listener_and_missing_build_source_are_unavailable(self):
        engine = {"id": "llama_cpp", "base_url": "http://127.0.0.1:9", "identity_provider": "local_process",
                  "_build_source_id": self.SOURCE}
        self.assertEqual(HARNESS.fetch_server_identity(engine, 1)["reason"], "local_process_not_found")
        del engine["_build_source_id"]
        self.assertEqual(HARNESS.fetch_server_identity(engine, 1)["reason"], "build_source_unavailable")

    def test_argv_hashes_bind_the_exact_argv_and_a_portable_role_form(self):
        first = ["/a/llama-server", "-m", "/a/m.gguf", "--port", "1", "--parallel", "6", "--slot-save-path=/s1/"]
        second = ["/b/llama-server", "-m", "/b/m.gguf", "--port", "2", "--parallel", "6", "--slot-save-path=/s2/"]
        one, two = HARNESS.argv_hashes(first), HARNESS.argv_hashes(second)
        self.assertNotEqual(one["argv_sha256"], two["argv_sha256"])
        self.assertEqual(one["argv_roles_sha256"], two["argv_roles_sha256"])
        other = HARNESS.argv_hashes([*second[:5], "--parallel", "4", *second[7:]])
        self.assertNotEqual(one["argv_roles_sha256"], other["argv_roles_sha256"])
        flags = HARNESS.argv_flags(first)
        self.assertEqual((flags["parallel"], flags["slot_save_path"], flags["jinja"]), (6, True, False))
        self.assertIsNone(HARNESS.argv_flags(["x"])["parallel"])

    def test_missing_or_short_flags_are_rejected_by_the_process_binding(self):
        engine = {"slot_count": 6}
        identity = {"argv_sha256": "a" * 64, "argv_roles_sha256": "b" * 64, "workload_epoch": "p", "process_instance_id": "p",
                    "flags": {"parallel": 6, "slot_save_path": True}}
        self.assertEqual(HARNESS._llama_flag_errors(identity, engine), [])
        for change, text in (
            ({"flags": {"parallel": 4, "slot_save_path": True}}, "fewer slots"),
            ({"flags": {"parallel": None, "slot_save_path": True}}, "fewer slots"),
            ({"flags": {"parallel": 6, "slot_save_path": False}}, "slot save path"),
            ({"argv_sha256": None}, "argv hashes"),
            ({"workload_epoch": "q"}, "workload epoch"),
        ):
            errors = HARNESS._llama_flag_errors({**identity, **change}, engine)
            self.assertTrue(any(text in error for error in errors), (change, errors))

    def test_port_owner_falls_back_to_lsof_and_never_reports_zero_for_a_failed_tool(self):
        done = subprocess.CompletedProcess([], 0, stdout=b"p123\nf5\np456\n", stderr=b"")
        with mock.patch.object(HARNESS, "_proc_available", return_value=False), mock.patch.object(
            HARNESS.subprocess, "run", return_value=done
        ):
            self.assertEqual(HARNESS.port_owner_pids(80), {123, 456})
        with mock.patch.object(HARNESS, "_proc_available", return_value=False), mock.patch.object(
            HARNESS.subprocess, "run", side_effect=FileNotFoundError
        ):
            self.assertIsNone(HARNESS.port_owner_pids(80))

    def test_boot_session_falls_back_to_the_macos_sysctl(self):
        done = subprocess.CompletedProcess([], 0, stdout=b"UUID-1\n", stderr=b"")
        with mock.patch.object(HARNESS.Path, "read_text", side_effect=OSError), mock.patch.object(
            HARNESS.subprocess, "run", return_value=done
        ) as run:
            self.assertEqual(HARNESS.boot_session_id(), "UUID-1")
        self.assertEqual(run.call_args.args[0][:2], ["sysctl", "-n"])


class FileHashCacheTests(unittest.TestCase):
    def setUp(self):
        HARNESS._FILE_HASHES.clear()

    def test_an_unchanged_file_is_hashed_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "m.gguf"
            path.write_bytes(b"model")
            with mock.patch.object(HARNESS, "sha256_file", wraps=HARNESS.sha256_file) as hasher:
                first, second = HARNESS.cached_file_sha256(path), HARNESS.cached_file_sha256(path)
        self.assertEqual((first, hasher.call_count), (HARNESS.sha256_bytes(b"model"), 1))
        self.assertEqual(second, first)

    def test_any_mutation_rehashes_instead_of_relabeling_old_readiness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "m.gguf"
            path.write_bytes(b"model")
            before = HARNESS.cached_file_sha256(path)
            stamp = path.stat()
            path.write_bytes(b"MODEL")
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
            self.assertEqual(path.stat().st_size, stamp.st_size)
            self.assertNotEqual(HARNESS.cached_file_sha256(path), before)

    def test_a_file_that_changes_while_hashed_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "m.gguf"
            path.write_bytes(b"model")

            def rewrite(target):
                target.write_bytes(b"other bytes")
                return "0" * 64

            with mock.patch.object(HARNESS, "sha256_file", side_effect=rewrite):
                with self.assertRaises(OSError):
                    HARNESS.cached_file_sha256(path)
        self.assertEqual(len(HARNESS._FILE_HASHES), 0)

    def test_the_cache_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            for index in range(HARNESS.FILE_HASH_CACHE_LIMIT + 4):
                path = pathlib.Path(directory) / f"f{index}"
                path.write_bytes(bytes([index]))
                HARNESS.cached_file_sha256(path)
        self.assertEqual(len(HARNESS._FILE_HASHES), HARNESS.FILE_HASH_CACHE_LIMIT)


SERVER_SCRIPT = """import http.server, sys
port = int(sys.argv[sys.argv.index("--port") + 1])
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *args): pass
http.server.HTTPServer(("127.0.0.1", port), Handler).serve_forever()
"""


def _free_port():
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class SpawnedProcessTests(unittest.TestCase):
    """The harness starts the server itself, so pid, argv, files, and port owner are first-hand."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        (self.root / "server.py").write_text(SERVER_SCRIPT)
        binary = self.root / "bin" / "server"
        binary.parent.mkdir()
        binary.write_text(f'#!/bin/sh\nexec {sys.executable} {self.root}/server.py "$@"\n')
        binary.chmod(0o755)
        (self.root / "model.gguf").write_bytes(b"spawn model")
        self.engine = {
            "id": "llama_cpp", "kind": "llama.cpp", "base_url": f"http://127.0.0.1:{_free_port()}",
            "executable_path": "bin/server", "model_artifact": "model.gguf", "slot_count": 6,
            "identity_provider": "spawned_process", "_build_source_id": "llama.cpp:git:" + "d" * 40,
            "spawn": {"startup_timeout_s": 30, "argv": [
                "{executable}", "--model", "{model}", "--host", "{host}", "--port", "{port}",
                "--alias", "model.gguf", "--parallel", "{slot_count}", "--slot-save-path", "{slot_save_path}"]},
        }

    def tearDown(self):
        self.directory.cleanup()

    def test_identity_is_first_hand_and_the_private_directory_is_removed_on_stop(self):
        with HARNESS.SpawnedServer(self.engine, self.root) as server:
            slot_dir = server.slot_dir
            self.assertTrue(slot_dir.is_dir())
            result = server.identity()
            identity = result["identity"]
            self.assertEqual(result["status"], "observed")
            self.assertEqual(identity["model_sha256"], HARNESS.sha256_bytes(b"spawn model"))
            self.assertEqual(identity["executable_sha256"], HARNESS.sha256_file(self.root / "bin" / "server"))
            self.assertEqual(identity["flags"], {**identity["flags"], "parallel": 6, "slot_save_path": True})
            self.assertIn(f":{server.process.pid}:", identity["process_instance_id"])
            self.assertEqual(identity["source_id"], self.engine["_build_source_id"])
            engine = {**self.engine, "_spawned": server}
            self.assertEqual(HARNESS.fetch_server_identity(engine, 1), result)
        self.assertFalse(slot_dir.exists())
        self.assertEqual(server.identity()["reason"], "spawned_process_not_running")

    def test_a_replaced_port_owner_is_rejected(self):
        with HARNESS.SpawnedServer(self.engine, self.root) as server:
            with mock.patch.object(HARNESS, "port_owner_pids", return_value={server.process.pid + 1}):
                self.assertEqual(server.identity()["reason"], "port_owner_is_not_the_spawned_process")
            with mock.patch.object(HARNESS, "port_owner_pids", return_value=None):
                self.assertEqual(server.identity()["reason"], "port_owner_unavailable")

    def test_a_mutated_model_or_binary_is_rejected_not_relabeled(self):
        with HARNESS.SpawnedServer(self.engine, self.root) as server:
            self.assertEqual(server.identity()["status"], "observed")
            (self.root / "model.gguf").write_bytes(b"swapped model")
            self.assertEqual(server.identity()["reason"], "spawned_model_changed")
            (self.root / "model.gguf").write_bytes(b"spawn model")
            with (self.root / "bin" / "server").open("a") as handle:
                handle.write("# changed\n")
            self.assertEqual(server.identity()["reason"], "spawned_executable_changed")

    def test_served_template_bytes_enter_the_identity_and_a_mutation_is_rejected(self):
        template = self.root / "template.jinja"
        template.write_bytes(b"template one")
        engine = {**self.engine, "spawn": {**self.engine["spawn"], "argv": [
            *self.engine["spawn"]["argv"], "--chat-template-file", "template.jinja"]}}
        with HARNESS.SpawnedServer(engine, self.root) as server:
            identity = server.identity()["identity"]
            self.assertEqual(identity["template_sha256"], HARNESS.sha256_bytes(b"template one"))
            template.write_bytes(b"template two")
            self.assertEqual(server.identity()["reason"], "spawned_template_changed")
            template.write_bytes(b"template one")
            self.assertEqual(server.identity()["status"], "observed")

    def test_a_launch_without_a_template_flag_records_no_template_hash(self):
        with HARNESS.SpawnedServer(self.engine, self.root) as server:
            self.assertNotIn("template_sha256", server.identity()["identity"])

    def test_a_port_in_use_or_an_unknown_placeholder_stops_before_a_process_starts(self):
        with HARNESS.SpawnedServer(self.engine, self.root) as server:
            with self.assertRaises(HARNESS.StudyError):
                HARNESS.SpawnedServer(self.engine, self.root).start()
        bad = {**self.engine, "base_url": f"http://127.0.0.1:{_free_port()}",
               "spawn": {**self.engine["spawn"], "argv": ["{executable}", "{surprise}"]}}
        with self.assertRaises(HARNESS.StudyError):
            with HARNESS.SpawnedServer(bad, self.root):
                pass
        self.assertIsNone(server.process)

    def test_an_exiting_server_fails_startup_and_cleans_up(self):
        (self.root / "bin" / "server").write_text("#!/bin/sh\nexit 3\n")
        server = HARNESS.SpawnedServer(self.engine, self.root)
        with self.assertRaises(HARNESS.StudyError):
            with server:
                pass
        self.assertIsNone(server.slot_dir)

    def test_study_engines_bind_the_build_source_and_stop_every_spawned_server(self):
        manifest = {"engines": [self.engine, {"id": "leone"}]}
        rows = [{"provenance": {"build_info": {"source_id": "llama.cpp:git:" + "e" * 40}}}, {}]
        with HARNESS._study_engines(manifest, self.root, rows) as bound:
            first = bound["engines"][0]
            self.assertEqual(first["_build_source_id"], "llama.cpp:git:" + "e" * 40)
            self.assertIsInstance(first["_spawned"], HARNESS.SpawnedServer)
            self.assertNotIn("_spawned", bound["engines"][1])
            slot_dir = first["_slot_dir"]
            self.assertTrue(slot_dir.is_dir())
        self.assertFalse(slot_dir.exists())

    def test_the_spawn_declaration_is_validated(self):
        base = {"id": "llama_cpp", "kind": "llama.cpp", "slot_count": 6, "identity_provider": "spawned_process",
                "model_artifact": "model.gguf",
                "slot_copy": dict(HARNESS.SLOT_COPY_DECLARATION), "spawn": self.engine["spawn"]}
        self.assertEqual(HARNESS._slot_copy_declaration_errors(base, "e"), [])
        errors = HARNESS._slot_copy_declaration_errors({**base, "spawn": {"argv": ["{executable}"]}}, "e")
        self.assertTrue(any("{model}" in error for error in errors) and any("startup_timeout_s" in error for error in errors))
        local = {**base, "identity_provider": "local_process"}
        self.assertTrue(any("spawn needs identity_provider" in e for e in HARNESS._slot_copy_declaration_errors(local, "e")))


class SlotFileCleanupTests(unittest.TestCase):
    def test_only_a_file_in_the_owned_directory_is_removed(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as foreign:
            owned, other = pathlib.Path(directory), pathlib.Path(foreign)
            (owned / "a.slot").write_bytes(b"kv")
            (other / "keep.slot").write_bytes(b"kv")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, "a.slot"), "removed")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, "a.slot"), "missing")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, f"../{other.name}/keep.slot"), "refused")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, str(other / "keep.slot")), "refused")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, "notes.txt"), "refused")
            (owned / "link.slot").symlink_to(other / "keep.slot")
            self.assertEqual(HARNESS._remove_owned_slot_file(owned, "link.slot"), "missing")
            self.assertTrue((other / "keep.slot").exists())

    def test_a_server_the_harness_did_not_start_never_loses_a_file(self):
        plan = {"filename": "p.slot", "save": {}, "restore": {}}
        self.assertEqual(HARNESS._release_slot_files({}, plan)["cleanup"], {"status": "not_owned"})
        self.assertIsNone(HARNESS._release_slot_files({}, None))
        with tempfile.TemporaryDirectory() as directory:
            (pathlib.Path(directory) / "p.slot").write_bytes(b"kv")
            released = HARNESS._release_slot_files({"_slot_dir": pathlib.Path(directory)}, plan)
            self.assertEqual(released["cleanup"], {"status": "removed"})

    def test_a_slot_copy_without_a_typed_cleanup_status_is_rejected(self):
        self.assertEqual(HARNESS._slot_cleanup_errors({"slot_copy": None}), [])
        self.assertTrue(HARNESS._slot_cleanup_errors({"slot_copy": {"filename": "x"}}))
        self.assertTrue(HARNESS._slot_cleanup_errors({"slot_copy": {"cleanup": {"status": "deleted"}}}))
        self.assertEqual(HARNESS._slot_cleanup_errors({"slot_copy": {"cleanup": {"status": "removed"}}}), [])


class OfflineValidationTests(ControlCase):
    """Offline validation reads recorded digests and never local models or binaries."""

    def test_offline_accepts_the_control_without_binaries_or_models_and_online_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            copy = pathlib.Path(directory)
            for path in self.control["root"].rglob("*"):
                if path.is_file() and path.parts[-2] not in {"bin", "models"}:
                    target = copy / path.relative_to(self.control["root"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
            receipt = self.control["receipt_path"].relative_to(self.control["root"])
            self.assertEqual(HARNESS.validate_receipt(copy / receipt, copy, offline=True), [])
            self.assertNotEqual(HARNESS.validate_receipt(copy / receipt, copy), [])
            self.assertFalse(HARNESS._offline())

    def test_offline_still_rejects_a_malformed_recorded_digest(self):
        receipt = self.receipt()
        receipt["engines"][0]["provenance"]["executable_sha256"] = "not a digest"
        self.control["receipt_path"].write_text(json.dumps(receipt, sort_keys=True))
        errors = HARNESS.validate_receipt(self.control["receipt_path"], self.control["root"], offline=True)
        self.assertTrue(any("executable SHA-256 does not match" in error for error in errors), errors[:3])

    def test_offline_never_claims_reexecution_of_the_tokenizer(self):
        receipt = self.receipt()
        branch = self.llama_runs(receipt)[0]["branches"][0]
        branch["history_reuse"]["tokenization"]["oracle"]["evidence_class"] = "reexecuted"
        self.control["receipt_path"].write_text(json.dumps(receipt, sort_keys=True))
        errors = HARNESS.validate_receipt(self.control["receipt_path"], self.control["root"], offline=True)
        self.assertTrue(any("retained_and_recomputed" in error for error in errors))

    def test_the_flags_reach_the_command_line(self):
        calls = []
        with mock.patch.object(HARNESS, "validate_receipt", side_effect=lambda *a: calls.append(a) or []), \
                contextlib.redirect_stdout(io.StringIO()):
            code = HARNESS.main(["--validate-receipt", "r.json", "--root", "/x", "--offline", "--source-manifest", "s.json"])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0][2:], (True, pathlib.Path("s.json"), "repository"))

    def test_the_source_scope_reaches_the_validator(self):
        calls = []
        with mock.patch.object(HARNESS, "validate_receipt", side_effect=lambda *a: calls.append(a) or []), \
                contextlib.redirect_stdout(io.StringIO()):
            HARNESS.main(["--validate-receipt", "r.json", "--source-manifest", "s.json", "--source-scope", "archive"])
        self.assertEqual(calls[0][4], "archive")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            HARNESS.main(["--validate-receipt", "r.json", "--source-scope", "anywhere"])

    def test_archive_scope_requires_a_source_manifest(self):
        errors = HARNESS.validate_receipt(
            self.control["receipt_path"], self.control["root"], offline=True, source_scope="archive"
        )
        self.assertEqual(errors, ["archive source scope requires --source-manifest"])


class SourceBindingTests(unittest.TestCase):
    COMMIT = "a" * 40

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)
        self.files = {}
        for name in HARNESS.ARCHIVE_REQUIRED_SOURCES:
            self.put(name, b"print(1)\n", 0o755)
        self.files["crates/lib.rs"] = {"sha256": HARNESS.sha256_bytes(b"fn main() {}\n"), "executable": False}
        self.source = {"status": "observed", "commit": self.COMMIT, "tracked_tree_clean": True}
        self.write()

    def tearDown(self):
        self.directory.cleanup()

    def put(self, name, content, mode):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        path.chmod(mode)
        self.files[name] = {"sha256": HARNESS.sha256_bytes(content), "executable": bool(mode & 0o111)}

    def write(self):
        body = {"schema_version": "leone.source-inputs.v2", "source_commit": self.COMMIT, "files": self.files}
        (self.root / "source-inputs.json").write_text(json.dumps(body))

    def errors(self, source=None, scope="archive"):
        return HARNESS._source_binding_errors(
            source or self.source, self.root, pathlib.Path("source-inputs.json"), scope
        )

    def test_an_archive_without_crates_binds_the_receipt(self):
        self.assertEqual(self.errors(), [])

    def test_an_archive_needs_the_files_the_harness_loads(self):
        del self.files["scripts/linked_libraries.py"]
        self.write()
        self.assertTrue(any("omits required files" in error for error in self.errors()))
        self.put("scripts/linked_libraries.py", b"print(1)\n", 0o755)
        self.write()
        (self.root / "scripts/linked_libraries.py").unlink()
        self.assertTrue(any("not a regular file" in error for error in self.errors()))

    def test_a_changed_file_mode_or_commit_is_rejected(self):
        run = self.root / "scripts/produce-history-tokenization.py"
        run.write_bytes(b"print(2)\n")
        self.assertTrue(any("measured source input changed" in error for error in self.errors()))
        run.write_bytes(b"print(1)\n")
        run.chmod(0o644)
        self.assertTrue(any("source executable mode changed" in error for error in self.errors()))
        self.assertTrue(any("source commit" in error for error in self.errors({**self.source, "commit": "b" * 40})))

    def test_a_missing_symlinked_or_escaping_path_is_rejected(self):
        self.assertTrue(self.errors({"status": "unavailable"}))
        real = self.root / "real.json"
        (self.root / "source-inputs.json").rename(real)
        (self.root / "source-inputs.json").symlink_to(real)
        self.assertEqual(self.errors(), ["source input manifest path is unsafe"])
        self.assertEqual(
            HARNESS._source_binding_errors(self.source, self.root, pathlib.Path("../x.json"), "archive"),
            ["source input manifest path is unsafe"],
        )

    def test_an_unknown_scope_is_rejected(self):
        self.assertEqual(self.errors(scope="everything"), ["source scope is not supported: everything"])

    def test_the_repository_scope_checks_the_whole_recorded_set(self):
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        for directory in ("scripts", "fixtures", "crates"):
            shutil.rmtree(self.root / directory, ignore_errors=True)
        self.files = {}
        self.put("fixtures/a.json", b"{}\n", 0o644)
        self.write()
        self.assertEqual(self.errors(scope="repository"), [])
        self.put("fixtures/b.json", b"{}\n", 0o644)
        del self.files["fixtures/b.json"]
        self.write()
        self.assertTrue(any("file set differs" in error for error in self.errors(scope="repository")))

    def test_the_receipt_records_its_source(self):
        with mock.patch.object(HARNESS, "_git_source", return_value=self.source) as source:
            self.assertEqual(HARNESS._git_source(self.root), self.source)
        source.assert_called_once()
        self.assertEqual(HARNESS._git_source(self.root)["status"], "unavailable")


def _run_stubbed_leone(manifest, engine, calls):
    """Run one schedule with a deterministic transport stub."""

    def fake_stream(_engine, body, fields, _timeout, **options):
        calls.append((body, fields, options))
        return {
            "status": "cancelled" if options.get("cancel_after_first_content") else "completed",
            "metrics": HARNESS.latency_metrics(1, [], False),
            "cache_reuse": {"status": "unavailable", "reason": "test"},
            "_content_text": "parent answer",
        }

    def fake_parent(_engine, body, fields, _timeout, **options):
        calls.append((body, fields, options))
        return {
            "status": "completed",
            "metrics": {"status": "unavailable", "reason": "nonstream_parent_unmeasured"},
            "cache_reuse": {"status": "unavailable", "reason": "nonstream_parent_unmeasured"},
            "_content_text": "parent answer",
            "content_sha256": HARNESS.sha256_bytes(b"parent answer"),
            "request_start_ns": 1,
            "request_end_ns": 2,
            "history_complete": True,
        }

    with mock.patch.object(HARNESS, "nonstream_request", side_effect=fake_parent), mock.patch.object(
        HARNESS, "stream_request", side_effect=fake_stream), mock.patch.object(
        HARNESS, "inspect_branch_method", return_value={"status": "supported"}
    ), mock.patch.object(HARNESS, "fetch_metrics_snapshot", return_value={"status": "unavailable"}):
        return HARNESS.run_engine(manifest, engine, 0, 1)


if __name__ == "__main__":
    unittest.main()
