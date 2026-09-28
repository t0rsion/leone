"""Tests the cached llama.cpp workload and receipt boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from run_llama_cached_comparator import (
    ENGINE_PROTOCOL,
    MANIFEST_SCHEMA,
    ValidationError,
    backend_module_identity,
    context_json,
    encode_plan,
    event_json,
    expected_logits,
    file_identity,
    load_workload,
    normalize_engine_header,
    range_json,
    refuse_output,
    resolve_backend_module,
    resolve_linked_library,
    validate_engine_records,
    validate_subject_model,
)


def fixture_bytes() -> bytes:
    return struct.pack("<12I", *range(1, 13))


def manifest_value(data: bytes) -> dict:
    return {
        "schema": MANIFEST_SCHEMA,
        "phase": "exploratory_calibration",
        "model_contract": {
            "architecture": "test",
            "tokenizer": "test",
            "vocab": 16,
            "subject_sha256": "3" * 64,
        },
        "token_fixture": {
            "path": "fixture.tokens.u32le",
            "encoding": "u32le",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "tokenization": {
                "token_limit": len(data) // 4,
                "add_bos": False,
                "parse_special": True,
                "llama_revision": "1" * 40,
                "reference_tokenizer_model_sha256": "2" * 64,
            },
            "source_corpus": {
                "path": "corpus.txt",
                "bytes": 7,
                "sha256": hashlib.sha256(b"fixture").hexdigest(),
            },
        },
        "context": {
            "n_ctx": 16,
            "n_batch": 8,
            "n_ubatch": 4,
            "n_seq_max": 4,
            "threads": 1,
            "kv_type": "f16",
            "flash_attention": "off",
            "kv_unified": True,
            "swa_full": True,
        },
        "slices": {
            "prefix": {"offset": 0, "count": 3},
            "suffix": {"offset": 3, "count": 1},
            "continuation": {"offset": 4, "count": 2},
        },
        "cases": [
            {
                "name": "shared",
                "class": "shared_prefix",
                "operations": [
                    {
                        "kind": "decode",
                        "name": "prefix",
                        "streams": [
                            {
                                "seq": 0,
                                "position": 0,
                                "tokens": ["prefix"],
                                "outputs": "last",
                            }
                        ],
                    },
                    {
                        "kind": "copy",
                        "name": "fork",
                        "source": 0,
                        "target": 1,
                        "p0": 0,
                        "p1": -1,
                    },
                    {
                        "kind": "decode",
                        "name": "suffix",
                        "streams": [
                            {
                                "seq": 1,
                                "position": 3,
                                "tokens": ["suffix"],
                                "outputs": "last",
                            }
                        ],
                    },
                    {
                        "kind": "decode_stepwise",
                        "name": "continuation",
                        "streams": [
                            {"seq": 1, "position": 4, "tokens": ["continuation"]}
                        ],
                        "capture_steps": [0, 1],
                    },
                    {
                        "kind": "remove",
                        "name": "remove",
                        "sequence": 1,
                        "p0": 0,
                        "p1": -1,
                    },
                ],
            }
        ],
    }


class DefaultWorkloadTests(unittest.TestCase):
    def load_cases(self):
        root = Path(__file__).resolve().parents[2]
        workload = load_workload(
            root / "research/prefix_attention/llama_cached_workload.json", root
        )
        return {case.name: case for case in workload.cases}

    def test_preserves_tail_growth_case(self) -> None:
        growth = self.load_cases()["tail_growth"]
        growth_suffix = next(
            event for event in growth.events if event.name == "growth_suffix"
        )
        self.assertEqual(
            [len(stream.tokens) for stream in growth_suffix.streams], [33, 36, 39, 42]
        )
        growth_steps = [
            event
            for event in growth.events
            if event.name.startswith("growth_continuation_")
        ]
        self.assertEqual(len(growth_steps), 40)
        self.assertEqual(
            [
                index
                for index, event in enumerate(growth_steps)
                if event.streams[0].outputs
            ],
            [0, 31, 32, 39],
        )

    def test_preserves_shrinking_membership_case(self) -> None:
        membership = self.load_cases()["shrinking_membership"]
        active = [event for event in membership.events if event.kind == "decode"]
        self.assertEqual(
            [tuple(stream.seq for stream in event.streams) for event in active],
            [(0,), (0, 1, 2, 3), (0, 1, 2, 3), (0, 2, 3), (0, 3), (3,)],
        )
        self.assertEqual(
            membership.expected_ranges[-1][:4],
            ((-1, -1), (-1, -1), (-1, -1), (0, 52)),
        )
        self.assertEqual(
            [event.name for event in membership.events if event.kind == "remove"],
            ["remove_seq_1", "remove_seq_2", "remove_seq_0"],
        )


class WorkloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.data = fixture_bytes()
        (self.root / "fixture.tokens.u32le").write_bytes(self.data)
        (self.root / "corpus.txt").write_bytes(b"fixture")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def load(self, value: dict):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(value))
        return load_workload(path, self.root)

    def assert_rejected(self, mutate) -> None:
        value = manifest_value(self.data)
        mutate(value)
        with self.assertRaises(ValidationError):
            self.load(value)

    def test_expands_stepwise_tokens_and_binary_plan(self) -> None:
        workload = self.load(manifest_value(self.data))
        case = workload.cases[0]
        self.assertEqual(
            [event.name for event in case.events],
            [
                "prefix",
                "fork",
                "suffix",
                "continuation_0",
                "continuation_1",
                "remove",
            ],
        )
        self.assertEqual(case.expected_ranges[-1][1], (-1, -1))
        self.assertTrue(encode_plan(workload).startswith(b"LCMPPLN1"))
        self.assertEqual(event_json(case.events[2])["streams"][0]["tokens"], [4])
        self.assertEqual(event_json(case.events[-1])["sequence"], 1)

    def test_rejects_fixture_identity_and_path_escape(self) -> None:
        mutations = [
            lambda value: value["token_fixture"].update(sha256="0" * 64),
            lambda value: value["token_fixture"].update(path="../fixture.tokens.u32le"),
            lambda value: value["token_fixture"].update(bytes=4),
            lambda value: value["token_fixture"]["tokenization"].update(token_limit=1),
            lambda value: value["token_fixture"]["tokenization"].update(
                llama_revision="0" * 39
            ),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_rejects_position_and_copy_mutations(self) -> None:
        mutations = [
            lambda value: value["cases"][0]["operations"][2]["streams"][0].update(
                position=4
            ),
            lambda value: value["cases"][0]["operations"][1].update(p0=3),
            lambda value: value["cases"][0]["operations"][1].update(target=0),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_rejects_invalid_remove_mutations(self) -> None:
        mutations = [
            lambda value: value["cases"][0]["operations"][-1].update(p0=6),
            lambda value: value["cases"][0]["operations"][-1].update(sequence=2),
            lambda value: value["cases"][0]["operations"][-1].update(p1=0),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_rejects_batch_output_and_name_mutations(self) -> None:
        mutations = [
            lambda value: value["context"].update(n_batch=2, n_ubatch=2),
            lambda value: value["cases"][0]["operations"][0]["streams"][0].update(
                outputs=[3]
            ),
            lambda value: value["cases"][0]["operations"][2].update(name="prefix"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_rejected(mutation)

    def test_resolves_macos_rpath_inside_pinned_checkout(self) -> None:
        library = self.root / "llama.cpp/build/bin/libllama.0.dylib"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"library")
        binary = self.root / "bin/comparator"
        self.assertEqual(
            resolve_linked_library(
                "@rpath/libllama.0.dylib", binary, self.root / "llama.cpp"
            ),
            library,
        )
        self.assertIsNone(
            resolve_linked_library(
                "/usr/lib/libc.dylib", binary, self.root / "llama.cpp"
            )
        )

    def test_resolves_actual_backend_module_path(self) -> None:
        library = self.root / "backend/libggml-cpu.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"backend")
        self.assertEqual(
            resolve_backend_module(str(library), self.root / "llama.cpp"), library
        )

    def test_rejects_wrong_subject_model(self) -> None:
        model = {"sha256": "4" * 64}
        contract = {"subject_sha256": "5" * 64}
        with self.assertRaises(ValidationError):
            validate_subject_model(model, contract)

    def test_rejects_unresolved_or_untrusted_backend_module(self) -> None:
        llama_dir = self.root / "llama.cpp"
        trusted = llama_dir / "build/bin/libggml-cpu.so"
        trusted.parent.mkdir(parents=True)
        versioned = llama_dir / "build/bin/libggml-cpu.so.0"
        target = llama_dir / "build/bin/libggml-cpu.so.0.21.0"
        target.write_bytes(b"trusted")
        versioned.symlink_to(target.name)
        trusted.symlink_to(versioned.name)
        outside = self.root / "outside/libggml-cpu.so"
        outside.parent.mkdir()
        outside.write_bytes(b"outside")
        paths = [trusted]
        before = [file_identity(trusted)]
        identity = backend_module_identity(
            {"backend_module_path": str(versioned)}, llama_dir, paths, before
        )
        self.assertEqual(identity["name"], versioned.name)
        with self.assertRaises(ValidationError):
            backend_module_identity(
                {"backend_module_path": ""}, llama_dir, paths, before
            )
        with self.assertRaises(ValidationError):
            backend_module_identity(
                {"backend_module_path": str(outside)}, llama_dir, paths, before
            )
        target.write_bytes(b"changed")
        with self.assertRaises(ValidationError):
            backend_module_identity(
                {"backend_module_path": str(trusted)}, llama_dir, paths, before
            )

    def test_refuses_broken_output_symlink(self) -> None:
        output = self.root / "output.json"
        os.symlink(self.root / "missing-target", output)
        with self.assertRaises(ValidationError):
            refuse_output(output, "test output")


def engine_records(workload) -> tuple[list[dict], int]:
    records = [
        {
            "kind": "header",
            "protocol": ENGINE_PROTOCOL,
            "requested_backend": "cpu",
            "device": {
                "backend": "cpu",
                "backend_module_path": "",
                "backend_module_dynamic": False,
                "registry": "CPU",
                "name": "CPU",
                "description": "test CPU",
                "type": 0,
                "memory_free_before_model": 100,
                "memory_total_before_model": 100,
                "memory_free_after_context": 90,
                "memory_total_after_context": 100,
            },
            "model": {
                "vocab": 16,
                "context_train": 16,
                "size_bytes": 1,
                "ftype": 1,
                "ftype_name": "test",
                "architecture": "test",
                "tokenizer": "test",
            },
            "context": {
                "requested": context_json(workload.context),
                "effective": {
                    "n_ctx": 16,
                    "n_ctx_seq": 16,
                    "n_batch": 8,
                    "n_ubatch": 4,
                    "n_seq_max": 4,
                },
            },
            "model_load_ns": 1,
            "context_create_ns": 1,
        }
    ]
    offset = 0
    for case_index, case in enumerate(workload.cases):
        for event_index, event in enumerate(case.events):
            rows = expected_logits(case_index, event_index, event)
            records.append(
                {
                    "kind": "event",
                    "case": case_index,
                    "event": event_index,
                    "name": event.name,
                    "operation": event.kind,
                    "duration_ns": 1,
                    "submitted_tokens": sum(
                        len(stream.tokens) for stream in event.streams
                    )
                    if event.kind == "decode"
                    else 0,
                    "output_rows": len(rows),
                    "ranges": range_json(case.expected_ranges[event_index]),
                }
            )
            for row in rows:
                records.append(
                    {"kind": "logit", **row, "offset_bytes": offset, "float_count": 16}
                )
                offset += 64
    return records, offset


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        data = fixture_bytes()
        (self.root / "fixture.tokens.u32le").write_bytes(data)
        (self.root / "corpus.txt").write_bytes(b"fixture")
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest_value(data)))
        self.workload = load_workload(path, self.root)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def assert_record_rejected(self, mutate) -> None:
        records, size = engine_records(self.workload)
        mutate(records)
        with self.assertRaises(ValidationError):
            validate_engine_records(records, self.workload, "cpu", size)

    def test_accepts_complete_engine_transcript(self) -> None:
        records, size = engine_records(self.workload)
        header, cases = validate_engine_records(records, self.workload, "cpu", size)
        self.assertEqual(header["protocol"], ENGINE_PROTOCOL)
        self.assertEqual(cases[0]["name"], "shared")

    def test_normalizes_backend_module_path_before_publication(self) -> None:
        records, _ = engine_records(self.workload)
        identity = {
            "name": "libggml-cpu.so.0",
            "bytes": 7,
            "sha256": hashlib.sha256(b"backend").hexdigest(),
        }
        normalized = normalize_engine_header(records[0], identity)
        self.assertNotIn("backend_module_path", normalized["device"])
        self.assertEqual(normalized["device"]["backend_module_name"], identity["name"])
        self.assertEqual(normalized["device"]["backend_module_bytes"], 7)
        self.assertEqual(
            normalized["device"]["backend_module_sha256"], identity["sha256"]
        )

    def test_rejects_event_and_position_mutations(self) -> None:
        mutations = [
            lambda records: records[1].update(duration_ns=-1),
            lambda records: records[1]["ranges"][0].update(max=1),
            lambda records: records[1].update(submitted_tokens=2),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_record_rejected(mutation)

    def test_rejects_logit_omission_offset_and_backend_mutations(self) -> None:
        mutations = [
            lambda records: records.pop(2),
            lambda records: records[2].update(offset_bytes=4),
            lambda records: records[0]["device"].update(backend="cuda"),
            lambda records: records[0]["model"].update(architecture="other"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.assert_record_rejected(mutation)

    def test_rejects_unbound_sidecar_size(self) -> None:
        records, size = engine_records(self.workload)
        with self.assertRaises(ValidationError):
            validate_engine_records(
                copy.deepcopy(records), self.workload, "cpu", size + 4
            )


if __name__ == "__main__":
    unittest.main()
