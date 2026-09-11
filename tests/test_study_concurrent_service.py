#!/usr/bin/env python3
"""Unit tests for the non-GPU concurrent service harness."""

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "study_concurrent_service", ROOT / "scripts" / "study-concurrent-service.py"
)
assert SPEC and SPEC.loader
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


class VariantFailureTests(unittest.TestCase):
    def run_variant(self, failure, required=True):
        manifest = json.loads((ROOT / "benchmarks/concurrent-service-frozen.json").read_text())
        manifest["trace"]["required"] = required
        server = Mock()
        server.start.return_value = {"ready": True}
        server.stop.return_value = {"returncode": 0}
        result = Mock()
        result.to_dict.return_value = {"status": "completed"}
        replacements = {
            "ManagedServer": Mock(return_value=server),
            "build_launch_argv": Mock(return_value=[]),
            "nvidia_metadata": Mock(return_value={}),
            "run_offered_load": Mock(return_value=([result], {"completed_count": 1})),
            "_run_variant_disconnect": Mock(return_value={"passed": True}),
            "get_json": Mock(return_value={}),
            "resident_progress_trace": Mock(return_value={}),
        }
        replacements[failure].side_effect = OSError("late failure")
        with tempfile.TemporaryDirectory() as directory:
            with patch.multiple(HARNESS, **replacements), patch.object(HARNESS.threading, "Thread"):
                record = HARNESS._run_variant(
                    ROOT, manifest, {"id": "leone_batch1", "kind": "leone"},
                    [{"id": "request"}], [0], 0, 0, ROOT / "model.gguf", None,
                    ROOT / "leone", ROOT / "llama-server", 18000, pathlib.Path(directory), None,
                )
        server.stop.assert_called_once()
        return record

    def assert_completed_work(self, record):
        self.assertEqual(record["health"], {"ready": True})
        self.assertEqual(record["requests"], [{"status": "completed"}])
        self.assertEqual(record["summary"], {"completed_count": 1})

    def test_required_trace_failure_retains_completed_work(self):
        for failure in ("get_json", "resident_progress_trace"):
            with self.subTest(failure=failure):
                record = self.run_variant(failure)
                self.assert_completed_work(record)
                self.assertEqual(record["disconnect_recovery"], {"passed": True})
                self.assertEqual(record["resident_trace"], {"status": "unavailable", "error": "late failure"})
                self.assertEqual(record["startup_error"], "OSError: late failure")

    def test_optional_trace_failure_keeps_run_successful(self):
        record = self.run_variant("get_json", required=False)
        self.assert_completed_work(record)
        self.assertEqual(record["resident_trace"]["status"], "unavailable")
        self.assertIsNone(record["startup_error"])

    def test_disconnect_failure_retains_completed_work(self):
        record = self.run_variant("_run_variant_disconnect")
        self.assert_completed_work(record)
        self.assertEqual(record["disconnect_recovery"]["status"], "not_run")
        self.assertEqual(record["resident_trace"]["status"], "not_requested")
        self.assertEqual(record["startup_error"], "OSError: late failure")

    def test_invalid_usage_retains_earlier_consistency_errors(self):
        manifest = json.loads((ROOT / "benchmarks/concurrent-service-frozen.json").read_text())
        requests = HARNESS._expand_requests(manifest)
        run = {
            "repetition": 0, "order_position": 0, "variant": "leone_batch1",
            "requests": [{"request_id": requests[0]["id"], "usage": None}],
        }
        errors = HARNESS._validate_manifest_consistency({}, manifest, [run], requests)
        self.assertIn("run ordering differs from balanced manifest schedule", errors)
        self.assertIn("request body hash differs from manifest", errors)
        self.assertTrue(errors[-1].startswith("manifest consistency failed:"))


class SseParserTests(unittest.TestCase):
    def test_parser_handles_split_crlf_event_and_timestamp(self):
        parser = HARNESS.SseEventParser()
        self.assertEqual(parser.feed(b"data: {\"a\":", 100), [])
        events = parser.feed(b"1}\r\n\r\n", 200)
        parser.finish()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data, '{"a":1}')
        self.assertEqual(events[0].received_ns, 200)

    def test_parser_rejects_malformed_field(self):
        parser = HARNESS.SseEventParser()
        with self.assertRaises(HARNESS.MalformedSseError):
            parser.feed(b"event: message\n\n", 100)

    def test_parser_rejects_incomplete_event(self):
        parser = HARNESS.SseEventParser()
        parser.feed(b"data: {\"a\":1}\n", 100)
        with self.assertRaises(HARNESS.IncompleteStreamError):
            parser.finish()

    def test_parser_accepts_comments_without_data(self):
        parser = HARNESS.SseEventParser()
        self.assertEqual(parser.feed(b": keepalive\n\n", 100), [])
        parser.finish()


class StreamMetricsTests(unittest.TestCase):
    def test_content_events_are_not_token_intervals(self):
        parsed = HARNESS.ParsedStream()
        parsed.consume(
            HARNESS.SseEvent(
                '{"id":"x","choices":[{"delta":{"role":"assistant"}}]}',
                1_100_000,
            )
        )
        parsed.consume(
            HARNESS.SseEvent(
                '{"id":"x","choices":[{"delta":{"content":"a"}}]}',
                1_250_000,
            )
        )
        parsed.consume(
            HARNESS.SseEvent(
                '{"id":"x","choices":[{"delta":{"content":"bc"}}]}',
                1_500_000,
            )
        )
        metrics = HARNESS._stream_metrics(parsed, 1_000_000, 1_050_000, 2_000_000)
        self.assertEqual(metrics["first_nonempty_content_ttft_ms"], 0.25)
        self.assertEqual(metrics["content_event_intervals_ms"], [None, 0.25])
        self.assertFalse(metrics["token_boundaries_verified"])
        self.assertIsNone(metrics["token_itl_ms"])

    def test_usage_only_event_after_terminal_choice(self):
        parsed = HARNESS.ParsedStream()
        parsed.consume(HARNESS.SseEvent('{"choices":[{"delta":{},"finish_reason":"length"}]}', 1))
        parsed.consume(HARNESS.SseEvent('{"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}', 2))
        parsed.consume(HARNESS.SseEvent("[DONE]", 3))
        parsed.finish()
        self.assertEqual(parsed.usage["completion_tokens"], 1)
        self.assertEqual(parsed.content_events, [])

    def test_terminal_usage_and_done_are_required(self):
        parsed = HARNESS.ParsedStream()
        parsed.consume(
            HARNESS.SseEvent(
                '{"choices":[{"delta":{"content":"x"}}]}', 1
            )
        )
        with self.assertRaises(HARNESS.IncompleteStreamError):
            parsed.finish()
        parsed.consume(
            HARNESS.SseEvent(
                '{"choices":[{"delta":{},"finish_reason":"stop"}],'
                '"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}',
                2,
            )
        )
        parsed.consume(HARNESS.SseEvent("[DONE]", 3))
        parsed.finish()


class SummaryTests(unittest.TestCase):
    def test_linear_empirical_quantiles_and_qualification(self):
        self.assertEqual(HARNESS.quantile([1, 2, 4, 8], 0.5), 3.0)
        summary = HARNESS.empirical_quantiles([1, 2, 4, 8], 4)
        self.assertTrue(summary["qualified"])
        self.assertAlmostEqual(summary["values"]["p95"], 7.4)
        unqualified = HARNESS.empirical_quantiles([1, 2, 4], 4)
        self.assertFalse(unqualified["qualified"])
        self.assertEqual(unqualified["values"], {})

    def test_summary_keeps_losses_and_does_not_infer_token_itl(self):
        completed = HARNESS.RequestResult(
            request_id="a",
            prompt_id="short",
            planned_offset_ms=0,
            status="completed",
            prompt_tokens=3,
            completion_tokens=4,
            total_tokens=7,
            metrics={
                "first_nonempty_content_ttft_ms": 5.0,
                "completion_latency_ms": 20.0,
                "content_event_intervals_ms": [None, 2.0],
            },
        )
        timeout = HARNESS.RequestResult(
            request_id="b",
            prompt_id="long",
            planned_offset_ms=100,
            status="timeout",
            error="deadline",
        )
        summary = HARNESS.summarize_requests([completed, timeout], 1)
        self.assertEqual(summary["request_count"], 2)
        self.assertEqual(summary["loss_count"], 1)
        self.assertEqual(summary["outcomes"], {"completed": 1, "timeout": 1})
        self.assertIsNone(summary["token_itl_ms"])


class TraceAndOrderingTests(unittest.TestCase):
    def test_resident_trace_requires_progress_for_distinct_request(self):
        trace = {
            "schema_version": "leone.service-trace.v1",
            "events": [
                {"kind": "prefill_chunk", "request_id": "old", "processed_tokens": 4},
                {"kind": "prefill_chunk", "request_id": "new", "processed_tokens": 4},
                {
                    "kind": "resident_decode_progress", "emitted_tokens": 1,
                    "request_id": "old",
                    "during_prefill_request_id": "new",
                },
                {"kind": "prefill_chunk", "request_id": "new", "processed_tokens": 4},
            ],
        }
        self.assertTrue(HARNESS.resident_progress_trace(trace)["resident_progress_proven"])
        trace["events"].pop()
        self.assertFalse(HARNESS.resident_progress_trace(trace)["resident_progress_proven"])
        trace["events"].append({"kind": "prefill_chunk", "request_id": "new", "processed_tokens": 4})
        trace["events"][2]["request_id"] = "new"
        self.assertFalse(HARNESS.resident_progress_trace(trace)["resident_progress_proven"])

    def test_ordering_balances_three_variants(self):
        orders = HARNESS._balanced_orders(HARNESS.VARIANT_IDS, 6)
        counts = HARNESS._position_counts(orders, HARNESS.VARIANT_IDS)
        self.assertTrue(all(set(values) == {2} for values in counts.values()))


class EvidenceContractTests(unittest.TestCase):
    def manifest(self):
        import json
        return json.loads((ROOT / "benchmarks/concurrent-service-exploratory.json").read_text())

    def test_checked_in_manifests_and_controlled_flags(self):
        import json
        for name in ("exploratory", "frozen"):
            body = json.loads((ROOT / f"benchmarks/concurrent-service-{name}.json").read_text())
            HARNESS._validate_manifest(body, True)
        body = self.manifest()
        body["variants"][2]["batch_size"] = 4
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS._validate_manifest(body, True)
        body = self.manifest()
        body["server"]["llama_launch_args"].remove("--no-cache-prompt")
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS._validate_manifest(body, True)

    def test_request_ids_change_wire_session_and_hash(self):
        import json
        request = HARNESS._expand_requests(self.manifest())[0]
        first = HARNESS._request_body(request, "leone", True)
        second = HARNESS._request_body(dict(request, id="disconnect-probe"), "leone", True)
        self.assertNotEqual(first, second)
        self.assertNotEqual(json.loads(first)["leone_session"], json.loads(second)["leone_session"])

    def test_usage_total_and_quality_build_are_checked(self):
        with self.assertRaises(HARNESS.MalformedSseError):
            HARNESS._parse_usage({"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 4})
        build = {"source_commit": "a", "source_tree_dirty": False}
        quality = {"comparison_schema": True, "build_info": build, "executable": {"sha256": "binary"},
                   "metrics": {"leone_q4": {"receipt_sha256": "receipt", "kld": {"definition": "KL"}}}}
        HARNESS.validate_quality_build({"common": quality}, build, "binary")
        with self.assertRaises(HARNESS.HarnessError):
            HARNESS.validate_quality_build({"common": quality}, build, "different")

    def test_trace_hash_is_verified(self):
        body = {"schema_version": "leone.service-trace.v1", "events": [], "memory_samples": [{}]}
        run = {"variant": "leone_batch1", "requests": [], "disconnect_recovery": {"passed": True},
               "resident_trace": {"body": body, "body_sha256": "wrong"}}
        errors = HARNESS.release_outcome_errors({"runs": [run]})
        self.assertTrue(any("hash" in error for error in errors))

    def test_public_metadata_preserves_other_strings(self):
        value = {"file": str(ROOT / "model.gguf"), "artifact": str(ROOT / "target/run/log"), "claim": "content"}
        public = HARNESS.public_metadata(value, ROOT, ROOT / "target/run")
        self.assertEqual(public["file"], "./model.gguf")
        self.assertEqual(public["artifact"], "<run-artifacts>/log")
        self.assertEqual(public["claim"], "content")


class ReceiptRoundTripTests(unittest.TestCase):
    def test_sorted_requests_round_trip_and_bad_usage_returns_errors(self):
        import copy
        import json
        import tempfile
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            manifest = json.loads((ROOT / "benchmarks/concurrent-service-frozen.json").read_text())
            manifest["freeze_status"] = "frozen"
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            raw = manifest_path.read_bytes()
            (root / "model").write_bytes(b"synthetic-model")
            (root / "binary").write_bytes(b"synthetic-binary")
            (root / "quality").write_text("{}")
            build = {"source_commit": "a" * 40, "source_tree_dirty": False}
            requests = HARNESS._expand_requests(manifest)
            orders = HARNESS._balanced_orders(HARNESS.VARIANT_IDS, manifest["workload"]["repetitions"])
            body = {"schema_version": "leone.service-trace.v1", "memory_samples": [{}], "events": [
                {"kind": "prefill_chunk", "request_id": 1, "processed_tokens": 1, "ready": False},
                {"kind": "resident_decode_progress", "request_id": 2, "during_prefill_request_id": 1, "emitted_tokens": 1},
                {"kind": "prefill_chunk", "request_id": 1, "processed_tokens": 1, "ready": True}]}
            runs = []
            for repetition, order in enumerate(orders):
                for position, variant in enumerate(order):
                    outcomes = []
                    for request, offset in zip(requests, manifest["workload"]["stagger_ms"]):
                        outcome = HARNESS.RequestResult(request["id"], request["prompt_id"], offset, "completed",
                            prompt_tokens=1, completion_tokens=1, total_tokens=2,
                            request_body_sha256=HARNESS.sha256_bytes(HARNESS._request_body(request, "leone", True)),
                            metrics={"first_nonempty_content_ttft_ms": 1, "completion_latency_ms": 2,
                                     "content_event_intervals_ms": [None, 1]})
                        outcomes.append(outcome.to_dict())
                    runs.append({"variant": variant, "repetition": repetition, "order_position": position,
                        "requests": sorted(outcomes, key=lambda item: item["request_id"]), "summary": {},
                        "disconnect_recovery": {"passed": True}, "resident_trace": {"body": body,
                        "body_sha256": HARNESS.sha256_bytes(HARNESS.canonical_json(body)),
                        "validation": HARNESS.resident_progress_trace(body)}})
            quality = {"path": "quality", "sha256": HARNESS.sha256_file(root / "quality"),
                "comparison_schema": True, "comparison_engines": ["leone_q4", "llama_q4"],
                "build_info": build, "executable": {"sha256": HARNESS.sha256_file(root / "binary")},
                "metrics": {"leone_q4": {"receipt_sha256": "synthetic", "kld": {"definition": "synthetic"}}}}
            record = {"schema_version": HARNESS.SCHEMA_VERSION, "phase": "frozen",
                "manifest": {"path": "manifest.json", "body": manifest,
                    "file_sha256": HARNESS.sha256_bytes(raw), "canonical_sha256": HARNESS.sha256_bytes(HARNESS.canonical_json(manifest))},
                "model": {"path": "model", "sha256": HARNESS.sha256_file(root / "model")},
                "quality": {"common": quality}, "source": {"commit": "a" * 40},
                "llama_cpp": {"commit_matches_pinned": True},
                "binaries": {"leone": {"build_info": {"value": build}, "binary_hashes": [
                    {"path": "binary", "sha256": HARNESS.sha256_file(root / "binary")}] }},
                "ordering": {"orders": orders}, "runs": runs,
                "workload": {"repetitions": len(orders), "request_count_per_repetition": len(requests),
                    "requests": requests, "quantiles": {"minimum_samples": 12}},
                "summary": HARNESS._aggregate_runs(runs, 12), "comparability": {
                    "prompt_tokens": HARNESS._prompt_token_comparison(runs),
                    "context_limits": HARNESS._context_limit_check(runs, requests, 2048)}}
            path = root / "record.json"
            path.write_text(json.dumps(record))
            with patch.object(HARNESS, "_run_command", return_value=(0, "", "")):
                self.assertEqual(HARNESS.validate_recorded_receipt(path, root), [])
                broken = copy.deepcopy(record)
                broken["runs"][0]["requests"][0]["usage"]["total_tokens"] = None
                path.write_text(json.dumps(broken))
                self.assertTrue(HARNESS.validate_recorded_receipt(path, root))


if __name__ == "__main__":
    unittest.main()
