"""Check the BF16 oracle adapter, the calibration workload, and the quality gate."""

from __future__ import annotations

import array
import math
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common_oracle as C
import freeze_runtime_criteria as F
import run_llama_cached_comparator as cached
import validate_runtime as V
from test_runtime_validation import (
    FROZEN,
    MANIFEST,
    SOURCE,
    SUBJECT,
    TOKENS,
    VOCAB,
    ScratchTests,
    sha,
    source_rows,
    validator_args,
    values,
    write_calibration,
    write_evaluation,
    write_monitoring,
)

ORACLE = C.LOGIT_ORACLE_PAIRS[SUBJECT]["oracle_sha256"]
REAL_LOAD = cached.load_workload
GGUF = {
    "architecture": "qwen3", "tokenizer": "gpt2", "architecture_sha256": "a" * 64,
    "tokenizer_keys": list(C.GGUF_TOKENIZER_KEYS), "tokenizer_sha256": "b" * 64,
}
PIN = {key: GGUF[key] for key in ("architecture_sha256", "tokenizer_sha256", "tokenizer_keys")}


def small_load(path: Path, root: Path):
    workload = REAL_LOAD(path, root)
    return dataclasses.replace(workload, model_contract={**workload.model_contract, "vocab": VOCAB})


def gguf_string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def gguf_pair(key: str, value: object) -> bytes:
    if isinstance(value, bytes):
        return gguf_string(key) + value
    if isinstance(value, str):
        return gguf_string(key) + struct.pack("<I", 8) + gguf_string(value)
    if isinstance(value, int):
        return gguf_string(key) + struct.pack("<II", 4, value)
    kind, items = value
    body = b"".join(gguf_string(item) if kind == 8 else struct.pack("<i", item) for item in items)
    return gguf_string(key) + struct.pack("<IIQ", 9, kind, len(items)) + body


def write_gguf(path: Path, repeat: bytes = b"", **overrides) -> Path:
    """Write a small GGUF v3 file. `repeat` appends one raw entry, and bytes values are written as given."""
    entries = {
        "general.architecture": "qwen3", "qwen3.block_count": 36, "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2", "tokenizer.ggml.tokens": (8, ["a", "b", "c"]),
        "tokenizer.ggml.token_type": (5, [1, 1, 3]), "tokenizer.ggml.merges": (8, ["a b"]),
        "tokenizer.ggml.eos_token_id": 2, "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.padding_token_id": 0, "general.file_type": 15,
    }
    entries.update(overrides)
    body = b"".join(gguf_pair(key, value) for key, value in entries.items()) + repeat
    count = len(entries) + (1 if repeat else 0)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, count) + body)
    return path


class GgufIdentityTests(ScratchTests):
    def identity(self, **overrides) -> dict:
        return C.gguf_identity(write_gguf(self.root / "model.gguf", **overrides))

    def test_ids_that_forced_tokens_never_use_do_not_change_the_identity(self) -> None:
        base = self.identity()
        other = self.identity(**{"tokenizer.ggml.bos_token_id": 9, "tokenizer.ggml.padding_token_id": 7, "general.file_type": 32})
        self.assertEqual(base, other)

    def test_same_size_vocabulary_with_other_tokens_differs(self) -> None:
        base = self.identity()
        self.assertNotEqual(base["tokenizer_sha256"], self.identity(**{"tokenizer.ggml.tokens": (8, ["a", "b", "z"])})["tokenizer_sha256"])
        self.assertNotEqual(base["tokenizer_sha256"], self.identity(**{"tokenizer.ggml.merges": (8, ["b a"])})["tokenizer_sha256"])

    def test_layer_shape_and_architecture_changes_differ(self) -> None:
        base = self.identity()
        self.assertNotEqual(base["architecture_sha256"], self.identity(**{"qwen3.block_count": 40})["architecture_sha256"])
        self.assertEqual(self.identity(**{"general.architecture": "llama"})["architecture"], "llama")

    def test_recorded_digest_scheme_is_unchanged(self) -> None:
        """Re-derive the scheme in existing identity records with no shared code."""
        def text_hash(text: str) -> bytes:
            return hashlib.sha256(struct.pack("<Q", len(text.encode())) + text.encode()).hexdigest().encode()

        def strings(items: list[str]) -> str:
            return hashlib.sha256(struct.pack("<IQ", 8, len(items)) + b"".join(
                struct.pack("<Q", len(i.encode())) + i.encode() for i in items)).hexdigest()

        def digest(lines: dict[str, str]) -> str:
            return hashlib.sha256("".join(f"{k}={lines[k]}\n" for k in sorted(lines)).encode()).hexdigest()

        ints = lambda kind, items: hashlib.sha256(struct.pack("<IQ", kind, len(items)) + struct.pack(f"<{len(items)}i", *items)).hexdigest()
        u32 = lambda value: hashlib.sha256(struct.pack("<I", value)).hexdigest()
        tokenizer = {
            "tokenizer.ggml.model": text_hash("gpt2").decode(), "tokenizer.ggml.pre": text_hash("qwen2").decode(),
            "tokenizer.ggml.tokens": strings(["a", "b", "c"]), "tokenizer.ggml.token_type": ints(5, [1, 1, 3]),
            "tokenizer.ggml.merges": strings(["a b"]), "tokenizer.ggml.eos_token_id": u32(2),
        }
        architecture = {"general.architecture": text_hash("qwen3").decode(), "qwen3.block_count": u32(36)}
        identity = self.identity()
        self.assertEqual(identity["tokenizer_sha256"], digest(tokenizer))
        self.assertEqual(identity["architecture_sha256"], digest(architecture))
        self.assertEqual(sorted(identity), sorted(C.GGUF_IDENTITY_FIELDS))

    def test_top_level_scalar_type_code_is_outside_the_digest_scope(self) -> None:
        """Documents the recorded scheme: u32 36 and i32 36 hash alike. The file hash covers the type."""
        signed = struct.pack("<Ii", 5, 36)
        self.assertEqual(self.identity()["architecture_sha256"], self.identity(**{"qwen3.block_count": signed})["architecture_sha256"])

    def rejects(self, message: str, **overrides) -> None:
        with self.assertRaisesRegex(ValueError, message):
            self.identity(**overrides)

    def test_duplicate_keys_are_rejected(self) -> None:
        file = write_gguf(self.root / "dup.gguf", repeat=gguf_pair("qwen3.block_count", 40))
        with self.assertRaisesRegex(ValueError, "repeats the metadata key qwen3.block_count"):
            C.gguf_identity(file)

    def test_invalid_utf8_is_rejected_in_values_and_array_elements(self) -> None:
        bad = struct.pack("<IQ", 8, 2) + b"\xff\xfe"
        self.rejects("not valid UTF-8", **{"tokenizer.ggml.pre": bad})
        element = struct.pack("<IIQ", 9, 8, 1) + struct.pack("<Q", 1) + b"\xff"
        self.rejects("not valid UTF-8", **{"tokenizer.ggml.merges": element})

    def test_malformed_values_are_rejected(self) -> None:
        self.rejects("bool byte", **{"qwen3.flag": struct.pack("<I", 7) + b"\x02"})
        self.rejects("unsupported GGUF array element type 9", **{"qwen3.nested": struct.pack("<IIQ", 9, 9, 0)})
        self.rejects("unsupported GGUF value type 99", **{"qwen3.odd": struct.pack("<I", 99)})
        self.rejects("string length|reader bound", **{"tokenizer.ggml.pre": struct.pack("<IQ", 8, 1 << 40)})
        self.rejects("array length", **{"tokenizer.ggml.merges": struct.pack("<IIQ", 9, 8, 1 << 40)})

    def test_truncated_and_foreign_files_are_rejected(self) -> None:
        file = write_gguf(self.root / "model.gguf")
        file.write_bytes(file.read_bytes()[:-3])
        with self.assertRaisesRegex(ValueError, "ends early"):
            C.gguf_identity(file)
        file.write_bytes(b"NOPE" + b"\0" * 40)
        with self.assertRaisesRegex(ValueError, "not GGUF"):
            C.gguf_identity(file)


class CalibrationWorkloadTests(ScratchTests):
    def setUp(self) -> None:
        super().setUp()
        self.source = json.loads(SOURCE.read_text())
        self.workload = C.calibration_workload(MANIFEST, self.source)
        self.file = self.root / "workload.json"
        self.file.write_bytes(C.workload_bytes(self.workload))

    def test_comparator_parses_it_and_rows_match_the_manifest_bindings(self) -> None:
        loaded = REAL_LOAD(self.file, V.ROOT)
        rows = V.source_rows(loaded, self.workload)
        for spec in MANIFEST["calibration"]:
            self.assertEqual(rows[spec["name"]], V.calibration_bindings(spec, TOKENS))
        self.assertGreater(len(cached.encode_plan(loaded)), 0)

    def test_first_rows_match_the_driver_receipt(self) -> None:
        loaded = REAL_LOAD(self.file, V.ROOT)
        rows = V.source_rows(loaded, self.workload)
        first = {name: keys[0] for name, keys in rows.items()}
        self.assertEqual(first["calibration_shared"], ("decode_stepwise", 0, 0, 73))
        self.assertEqual([keys[0][3] for keys in rows.values()], [73, 73, 41, 61])

    def test_teacher_tokens_and_prompts_come_from_the_fixture(self) -> None:
        loaded = REAL_LOAD(self.file, V.ROOT)
        for spec, case in zip(MANIFEST["calibration"], loaded.cases):
            prefill, steps = case.events[0], case.events[1:]
            for branch, stream in enumerate(prefill.streams):
                self.assertEqual(list(stream.tokens), V.logical_prompt(spec, branch, TOKENS))
                self.assertEqual(stream.outputs, ())
            for step, event in enumerate(steps):
                for branch, stream in enumerate(event.streams):
                    self.assertEqual(list(stream.tokens), [TOKENS[spec["teacher_offsets"][branch][step]]])
                    self.assertEqual(stream.position, len(prefill.streams[branch].tokens) + step)

    def test_generation_is_deterministic_and_keeps_the_source_contract(self) -> None:
        self.assertEqual(C.workload_bytes(C.calibration_workload(MANIFEST, self.source)), self.file.read_bytes())
        for key in ("model_contract", "token_fixture", "context"):
            self.assertEqual(self.workload[key], self.source[key])

    def test_writer_refuses_to_replace_a_file(self) -> None:
        args = Namespace(write_calibration_workload=self.root / "new.json", runtime_manifest=V.MANIFEST, manifest=SOURCE)
        self.assertEqual(cached.write_calibration_workload(args).read_bytes(), self.file.read_bytes())
        with self.assertRaises(FileExistsError):
            cached.write_calibration_workload(args)

    def test_manifest_edit_changes_the_workload(self) -> None:
        edited = json.loads(json.dumps(MANIFEST))
        edited["calibration"][0]["teacher_offsets"][0][0] += 1
        self.assertNotEqual(C.workload_bytes(C.calibration_workload(edited, self.source)), self.file.read_bytes())


class ComparatorModeTests(ScratchTests):
    def setUp(self) -> None:
        super().setUp()
        self.subject = write_gguf(self.root / "subject.gguf", **{"general.file_type": 15})
        self.oracle = write_gguf(self.root / "oracle.gguf", **{"general.file_type": 32, "tokenizer.ggml.padding_token_id": 5})
        self.pin = {**C.LOGIT_ORACLE_PAIRS[SUBJECT], "oracle_sha256": sha(self.oracle)}
        contract = {"architecture": "qwen3", "tokenizer": "gpt2", "vocab": 3, "subject_sha256": sha(self.subject)}
        self.workload = dataclasses.replace(REAL_LOAD(SOURCE, V.ROOT), model_contract=contract)
        self.args = Namespace(subject_model=self.subject, model=self.oracle)

    def check(self, pins=None, model=None, metadata=None):
        pins = {sha(self.subject): self.pin} if pins is None else pins
        gguf = C.gguf_identity(self.subject)
        metadata = {sha(self.subject): {key: gguf[key] for key in PIN}} if metadata is None else metadata
        with patch.dict(C.LOGIT_ORACLE_PAIRS, pins, clear=True), patch.dict(C.LOGIT_ORACLE_METADATA, metadata, clear=True):
            return cached.check_oracle_models(self.args, self.workload, cached.file_identity(model or self.oracle))

    def test_metadata_pin_is_required_and_compared(self) -> None:
        with self.assertRaisesRegex(ValueError, "no independent metadata pin"):
            self.check(metadata={})
        gguf = C.gguf_identity(self.subject)
        wrong = {sha(self.subject): {**{key: gguf[key] for key in PIN}, "tokenizer_sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "differ from the independent pin"):
            self.check(metadata=wrong)

    def test_pinned_pair_with_equal_metadata_is_accepted(self) -> None:
        record = self.check()
        self.assertEqual(record["subject"]["gguf"], record["oracle"]["gguf"])
        self.assertEqual(record["pair"]["oracle_dtype"], "bf16")

    def test_unpinned_subject_and_oracle_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pinned oracle"):
            self.check(pins={"0" * 64: self.pin})
        with self.assertRaisesRegex(ValueError, "not the pinned oracle"):
            self.check(model=self.subject)

    def test_unrelated_model_with_the_same_vocabulary_is_rejected(self) -> None:
        write_gguf(self.oracle, **{"tokenizer.ggml.tokens": (8, ["a", "b", "z"])})
        self.pin["oracle_sha256"] = sha(self.oracle)
        with self.assertRaisesRegex(ValueError, "metadata differ"):
            self.check()

    def test_subject_must_match_the_workload_contract(self) -> None:
        self.args.subject_model = self.oracle
        with self.assertRaisesRegex(ValueError, "workload subject"):
            self.check()

    def test_engine_file_type_must_match_the_pin(self) -> None:
        record = self.check()
        cached.check_oracle_ftype({"model": {"ftype": 32}}, record)
        with self.assertRaisesRegex(ValueError, "file type"):
            cached.check_oracle_ftype({"model": {"ftype": 15}}, record)

    def test_default_output_is_byte_identical_to_the_recorded_contract(self) -> None:
        receipt = {"artifacts": {"logits": {"sha256": "1" * 64}}}
        logits, temporary, output = self.root / "plain.f32", self.root / "plain.tmp", self.root / "plain.json"
        temporary.write_bytes(b"x")
        cached.write_outputs(receipt, temporary, output, logits)
        self.assertEqual(output.read_text(), json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        self.assertEqual(sorted(path.name for path in self.root.glob("plain*")), ["plain.f32", "plain.json"])

    def oracle_publication(self, before=None, sidecar=None):
        """Run write_oracle_outputs on a fresh directory and return the paths it used."""
        directory = self.root / f"out{len(list(self.root.glob('out*')))}"
        directory.mkdir()
        paths = {name: directory / name for name in ("receipt.json", "logits.f32", "identity.json")}
        temporary = directory / "temporary.logits"
        temporary.write_bytes(b"logits")
        receipt = {"artifacts": {"logits": {"sha256": "1" * 64}}, "workload": {"manifest_sha256": "2" * 64},
                   "provenance": {"plan_sha256": "3" * 64}}
        for name in before or ():
            paths[name].write_text("existing")
        args = (receipt, temporary, paths["receipt.json"], paths["logits.f32"], {"pair": {}}, paths["identity.json"])
        return directory, paths, args

    def test_oracle_mode_publishes_logits_and_identity_before_the_receipt(self) -> None:
        directory, paths, args = self.oracle_publication()
        order = []
        real_link = os.link
        with patch.object(os, "link", side_effect=lambda src, dst: (order.append(Path(dst).name), real_link(src, dst))[1]):
            cached.write_oracle_outputs(*args)
        self.assertEqual(order, ["logits.f32", "identity.json", "receipt.json"])
        linked = json.loads(paths["identity.json"].read_text())
        self.assertEqual(linked["comparator_receipt_sha256"], sha(paths["receipt.json"]))
        self.assertEqual(linked["schema"], cached.ORACLE_IDENTITY_SCHEMA)
        self.assertEqual(sorted(path.name for path in directory.iterdir()), sorted(path.name for path in paths.values()))

    def test_a_file_that_appears_during_the_run_is_kept_and_nothing_else_stays(self) -> None:
        for taken in ("logits.f32", "identity.json", "receipt.json"):
            directory, paths, args = self.oracle_publication(before=(taken,))
            with self.assertRaises(FileExistsError):
                cached.write_oracle_outputs(*args)
            self.assertEqual([path.name for path in directory.iterdir()], [taken], taken)
            self.assertEqual(paths[taken].read_text(), "existing")

    def test_a_temporary_file_failure_leaves_no_file(self) -> None:
        directory, paths, args = self.oracle_publication()
        real, calls = cached.make_temporary_sidecar, []

        def flaky(path):
            calls.append(path)
            if len(calls) == 2:
                raise OSError("no space")
            return real(path)

        with patch.object(cached, "make_temporary_sidecar", side_effect=flaky), self.assertRaisesRegex(OSError, "no space"):
            cached.write_oracle_outputs(*args)
        self.assertEqual(list(directory.iterdir()), [])

    def test_a_failure_that_is_not_an_os_error_also_cleans_up(self) -> None:
        directory, paths, args = self.oracle_publication()
        with self.assertRaises(KeyError):
            cached.write_oracle_outputs({"artifacts": {}}, *args[1:])
        self.assertEqual(list(directory.iterdir()), [])

    def test_oracle_options_go_together(self) -> None:
        argv = ["run", "--llama-dir", "l", "--model", "m", "--output", "o", "--logits", "g", "--backend", "cpu", "--subject-model", "s"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            cached.parse_args()

    def test_output_paths_must_differ_and_not_exist(self) -> None:
        args = Namespace(output=self.root / "a", logits=self.root / "b", oracle_identity=None)
        with self.assertRaisesRegex(ValueError, "receipt, logit, and binary outputs must differ"):
            cached.refuse_outputs(args, self.root / "a")
        args.oracle_identity = self.root / "a"
        with self.assertRaisesRegex(ValueError, "identity outputs must differ"):
            cached.refuse_outputs(args, self.root / "bin")
        (self.root / "taken").write_text("x")
        args.oracle_identity = self.root / "taken"
        with self.assertRaisesRegex(ValueError, "already exists"):
            cached.refuse_outputs(args, self.root / "bin")


def bundle(root: Path, tag: str, workload_file: Path, *, model_sha: str = ORACLE, identity: bool = True,
           ftype: int = 32, shift: float = 0.0, edit=None):
    """Write a synthetic comparator receipt, logits, and identity record for one workload.

    `shift` moves the first logit of every row. `edit` changes the receipt before it is written.
    """
    workload = small_load(workload_file, V.ROOT)
    measurements, blob, cursor = [], bytearray(), 0
    for index, case in enumerate(workload.cases):
        events, number = [], 0
        for event_index, event in enumerate(case.events):
            rows = []
            for row in cached.expected_logits(index, event_index, event):
                rows.append({"kind": "logit", **row, "offset_bytes": cursor, "float_count": VOCAB})
                blob += array.array("f", values(number, shift)).tobytes()
                cursor, number = cursor + VOCAB * 4, number + 1
            events.append({"event": {"name": event.name}, "logit_rows": rows})
        measurements.append({"name": case.name, "events": events})
    logits = root / f"{tag}.f32"
    logits.write_bytes(bytes(blob))
    receipt = {
        "schema": cached.RECEIPT_SCHEMA, "quality": "unverified", "workload": {"manifest_sha256": sha(workload_file)},
        "engine": {"requested_backend": "cpu",
                   "device": {"backend": "cpu", "description": "fixture cpu", "backend_module_sha256": "1" * 64},
                   "model": {"vocab": VOCAB, "ftype": ftype, "ftype_name": "BF16"}},
        "provenance": {"plan_sha256": hashlib.sha256(cached.encode_plan(workload)).hexdigest(),
                       "model": {"sha256": model_sha}, "llama_revision": "r", "pin_file_sha256": "6" * 64,
                       "host": {"system": "Linux", "machine": "x86_64", "release": "7.2"},
                       "compiler": {"name": "c++", "version": "16"},
                       "comparator_binary": {"name": f"{tag}.driver", "sha256": "2" * 64},
                       "linked_llama_libraries": [{"name": "libllama.so.0", "sha256": "3" * 64}],
                       "source_inputs": [
                           {"name": "research/prefix_attention/run_llama_cached_comparator.py", "sha256": "4" * 64},
                           {"name": "$LLAMA_CPP/include/llama.h", "sha256": "5" * 64}]},
        "artifacts": {"logits": {"bytes": len(blob), "sha256": sha(logits)}}, "measurements": measurements,
    }
    if edit:
        edit(receipt)
    receipt_file = root / f"{tag}.json"
    receipt_file.write_text(json.dumps(receipt))
    record = {
        "schema": cached.ORACLE_IDENTITY_SCHEMA, "comparator_receipt_sha256": sha(receipt_file),
        "logits_sha256": sha(logits), "workload_sha256": receipt["workload"]["manifest_sha256"],
        "plan_sha256": receipt["provenance"]["plan_sha256"],
        "subject": {"sha256": SUBJECT, "gguf": GGUF}, "oracle": {"sha256": ORACLE, "gguf": GGUF},
        "pair": C.LOGIT_ORACLE_PAIRS[SUBJECT],
    }
    identity_file = root / f"{tag}.identity.json"
    identity_file.write_text(json.dumps(record))
    return [receipt_file, logits, identity_file] if identity else [receipt_file, logits]


class Bf16GateTests(ScratchTests):
    def setUp(self) -> None:
        super().setUp()
        for patcher in (patch.object(cached, "load_workload", side_effect=small_load),
                        patch.dict(C.LOGIT_ORACLE_METADATA, {SUBJECT: PIN}, clear=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.generated = self.root / "generated.json"
        self.generated.write_bytes(C.workload_bytes(C.calibration_workload(MANIFEST, json.loads(SOURCE.read_text()))))
        self.calibration_files = write_calibration(self.calibration)

    def oracles(self, **overrides) -> V.Oracles:
        if "bf16_calibration" not in overrides:
            overrides["bf16_calibration"] = bundle(self.root, "cal", self.generated)
        if "bf16_evaluation" not in overrides:
            overrides["bf16_evaluation"] = bundle(self.root, "ev", SOURCE)
        return V.load_oracles(validator_args(None, None, None, **overrides))

    def freeze(self, oracles=None) -> Path:
        study, manifest, source = V.load_calibration(self.calibration_files, "cuda")
        criteria = V.derive_criteria(study, manifest, source, oracles or self.oracles())
        criteria["frozen_utc"] = FROZEN.isoformat()
        output = self.root / "criteria.json"
        output.write_text(json.dumps(criteria))
        return output

    def evaluate(self, criteria: Path, shift: float, **overrides) -> dict:
        directory = self.root / "evaluation"
        directory.mkdir(exist_ok=True)
        files = write_evaluation(directory, sha(criteria), source_rows(), shift=shift)
        args = validator_args(criteria, self.calibration_files, files, **{
            "bf16_calibration": [self.root / "cal.json", self.root / "cal.f32", self.root / "cal.identity.json"],
            "bf16_evaluation": [self.root / "ev.json", self.root / "ev.f32", self.root / "ev.identity.json"], **overrides})
        return V.validate(args)

    def test_limits_freeze_from_bf16_calibration_before_evaluation(self) -> None:
        criteria = json.loads(self.freeze().read_text())
        quality = criteria["quality"]
        self.assertEqual(quality["gate"], "bf16")
        self.assertEqual(sorted(quality["delta_bounds"]), sorted(set(V.PATHS) - {V.CONTROL}))
        self.assertEqual(quality["oracles"]["calibration"]["kind"], "bf16_oracle")
        self.assertNotEqual(quality["oracles"]["calibration"]["plan_sha256"], quality["oracles"]["evaluation"]["plan_sha256"])
        self.assertEqual(quality["delta_bounds"][V.CANDIDATE]["max_kld"], math.nextafter(quality["calibration_deltas"]["calibration_shared"][V.CANDIDATE]["max_kld"], math.inf))
        self.assertIn("calibration_absolute_report_only", quality)
        self.assertIn("checkpoint", quality["limit"])

    def test_evaluation_inside_the_bounds_passes_and_outside_fails(self) -> None:
        criteria = self.freeze()
        passed = self.evaluate(criteria, 1e-4)
        self.assertEqual(passed["verdict"]["bf16_quality"], "pass")
        self.assertEqual(passed["quality"]["bf16"]["binding"]["kind"], "bf16_oracle")
        self.assertEqual(passed["quality"]["bf16"]["absolute_report_only"]["unrelated"][V.CONTROL]["max_kld"], 0.0)
        (self.root / "evaluation").rename(self.root / "first")
        failed = self.evaluate(criteria, 5.0)
        self.assertEqual(failed["verdict"]["bf16_quality"], "fail")

    def test_validation_needs_the_frozen_oracles(self) -> None:
        criteria = self.freeze()
        with self.assertRaisesRegex(ValueError, "criteria differ from the calibration"):
            self.evaluate(criteria, 1e-4, bf16_calibration=None, bf16_evaluation=None)

    def rejects(self, message: str, **overrides) -> None:
        with self.assertRaisesRegex(ValueError, message):
            self.oracles(**overrides)

    def test_identity_record_must_link_the_receipt(self) -> None:
        files = bundle(self.root, "cal", self.generated)
        files[0].write_text(files[0].read_text() + " ")
        self.rejects("not linked", bf16_calibration=files)

    def test_run_model_must_be_the_pinned_oracle(self) -> None:
        self.rejects("not the pinned oracle", bf16_calibration=bundle(self.root, "cal", self.generated, model_sha="0" * 64))

    def test_engine_type_must_be_bf16(self) -> None:
        self.rejects("file type", bf16_calibration=bundle(self.root, "cal", self.generated, ftype=15))

    def test_differing_gguf_metadata_is_rejected(self) -> None:
        files = bundle(self.root, "cal", self.generated)
        record = json.loads(files[2].read_text())
        record["oracle"]["gguf"] = {**GGUF, "tokenizer_sha256": "c" * 64}
        files[2].write_text(json.dumps(record))
        self.rejects("GGUF metadata differ", bf16_calibration=files)

    def test_phases_cannot_swap_workloads(self) -> None:
        self.rejects("workload hash differs", bf16_calibration=bundle(self.root, "cal", SOURCE))

    def test_bf16_oracle_must_describe_the_receipts_model(self) -> None:
        study, *_ = V.load_calibration(self.calibration_files, "cuda")
        oracle = self.oracles().calibration
        other = dataclasses.replace(oracle, binding={**oracle.binding, "subject_model_sha256": "0" * 64})
        with self.assertRaisesRegex(ValueError, "does not describe"):
            V.oracle_numerics(study, other)

    def test_same_weights_comparison_is_labelled_and_never_gates(self) -> None:
        reference = bundle(self.root, "q4", SOURCE, model_sha=SUBJECT, identity=False)
        args = validator_args(None, None, None, oracle_receipt=reference[0], oracle_logits=reference[1])
        oracles = V.load_oracles(args)
        self.assertEqual(oracles.q4_reference.binding["kind"], "q4_subject_cpu_comparison")
        study, manifest, source = V.load_calibration(self.calibration_files, "cuda")
        criteria = V.derive_criteria(study, manifest, source, oracles)
        self.assertEqual(criteria["quality"]["gate"], "unverified")
        self.assertEqual(criteria["q4_reference"]["kind"], "q4_subject_cpu_comparison")
        wrong = bundle(self.root, "q4b", SOURCE, model_sha=ORACLE, identity=False)
        with self.assertRaisesRegex(ValueError, "must run the subject model"):
            V.load_oracles(validator_args(None, None, None, oracle_receipt=wrong[0], oracle_logits=wrong[1]))

    def edit_identity(self, files: list[Path], edit) -> list[Path]:
        record = json.loads(files[2].read_text())
        edit(record)
        files[2].write_text(json.dumps(record))
        return files

    def test_the_record_is_checked_against_the_independent_pin(self) -> None:
        def forged(record):
            for side in ("subject", "oracle"):
                record[side]["gguf"] = {**GGUF, "tokenizer_keys": ["tokenizer.ggml.eos_token_id"], "tokenizer_sha256": "c" * 64}
        files = self.edit_identity(bundle(self.root, "cal", self.generated), forged)
        self.rejects("differ from the independent pin", bf16_calibration=files)

    def test_equal_but_unpinned_digests_are_rejected(self) -> None:
        def other_digest(record):
            for side in ("subject", "oracle"):
                record[side]["gguf"] = {**GGUF, "architecture_sha256": "d" * 64}
        self.rejects("differ from the independent pin", bf16_calibration=self.edit_identity(bundle(self.root, "cal", self.generated), other_digest))

    def test_a_record_without_a_pin_or_with_extra_fields_is_rejected(self) -> None:
        def extra(record):
            for side in ("subject", "oracle"):
                record[side]["gguf"] = {**GGUF, "note": "free text"}
        self.rejects("other fields", bf16_calibration=self.edit_identity(bundle(self.root, "cal", self.generated), extra))
        with patch.dict(C.LOGIT_ORACLE_METADATA, {}, clear=True):
            self.rejects("no independent metadata pin")

    def test_phases_from_different_cpu_origins_are_rejected(self) -> None:
        edits = {
            "host": lambda r: r["provenance"]["host"].update(system="Darwin", machine="arm64"),
            "binary": lambda r: r["provenance"]["comparator_binary"].update(sha256="9" * 64),
            "library": lambda r: r["provenance"]["linked_llama_libraries"][0].update(sha256="9" * 64),
            "compiler": lambda r: r["provenance"]["compiler"].update(version="17"),
            "collector": lambda r: r["provenance"]["source_inputs"][0].update(sha256="9" * 64),
            "cpu": lambda r: r["engine"]["device"].update(description="other cpu"),
        }
        for name, edit in edits.items():
            with self.subTest(name):
                self.rejects("CPU origin", bf16_evaluation=bundle(self.root, "ev", SOURCE, edit=edit))

    def test_binary_name_and_kernel_release_do_not_split_the_origin(self) -> None:
        def rename(receipt):
            receipt["provenance"]["comparator_binary"]["name"] = "evaluation.driver"
            receipt["provenance"]["host"]["release"] = "7.3"
        oracles = self.oracles(bf16_evaluation=bundle(self.root, "ev", SOURCE, edit=rename))
        self.assertEqual(oracles.calibration.binding["origin"], oracles.evaluation.binding["origin"])
        self.assertNotEqual(oracles.calibration.binding["host_release"], oracles.evaluation.binding["host_release"])

    def test_origin_label_names_a_common_bundle(self) -> None:
        linux, mac = {"host_system": "Linux", "host_machine": "x86_64"}, {"host_system": "Darwin", "host_machine": "arm64"}
        self.assertEqual(V.origin_label(linux, "Linux-7.2.6-arch2-1-x86_64-with-glibc2.44")["kind"], "same_host")
        self.assertEqual(V.origin_label(linux, "macOS-15.1-arm64-arm-64bit")["kind"], "common_bundle")
        self.assertEqual(V.origin_label(mac, "macOS-15.1-arm64-arm-64bit")["kind"], "same_host")
        report = self.evaluate(self.freeze(), 1e-4)
        self.assertEqual(report["quality"]["bf16"]["origin"]["kind"], "common_bundle")
        self.assertEqual(report["quality"]["bf16"]["origin"]["oracle_host"], "Linux x86_64")

    def test_the_control_is_not_blamed_for_a_shifted_oracle(self) -> None:
        """All four paths agree exactly. The oracle sits away from every path, so only absolute distances grow."""
        criteria = self.freeze(self.oracles(bf16_evaluation=bundle(self.root, "ev", SOURCE, shift=0.5)))
        report = self.evaluate(criteria, 0.0)
        self.assertEqual(report["verdict"]["numerical_vs_control"], "pass")
        self.assertEqual(report["verdict"]["bf16_quality"], "pass")
        absolute = report["quality"]["bf16"]["absolute_report_only"]["unrelated"]
        self.assertGreater(absolute[V.CONTROL]["max_abs_logit_diff"], 0.4)
        self.assertEqual(report["quality"]["bf16"]["deltas_vs_control"]["unrelated"]["per_row"]["max_kld"], 0.0)

    def test_a_path_worse_than_the_control_fails_the_delta_gate(self) -> None:
        report = self.evaluate(self.freeze(), 5.0)
        self.assertEqual(report["verdict"]["bf16_quality"], "fail")
        self.assertNotIn(V.CONTROL, {entry.split(":")[0] for entries in report["quality"]["failures"].values() for entry in entries})

    def test_freeze_refuses_to_omit_the_bf16_gate_silently(self) -> None:
        argv = ["freeze", "--backend", "cuda", "--calibration", "a", "--output", str(self.root / "c.json")]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            F.parse_args()
        self.assertEqual(raised.exception.code, 2)
        with patch.object(sys, "argv", argv + ["--quality-unverified"]):
            self.assertTrue(F.parse_args().quality_unverified)


class CatalogTests(unittest.TestCase):
    """The three model hash catalogs agree with each other and with the pinned pair."""

    def catalogs(self) -> dict[str, tuple[str, str]]:
        fetch = (V.ROOT / "scripts/fetch-model.sh").read_text()
        spec = importlib.util.spec_from_file_location("release_evidence_manifest", V.ROOT / "scripts/release_evidence_manifest.py")
        manifest = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manifest)
        subject, oracle = next(iter(C.LOGIT_ORACLE_PAIRS.items()))
        return {
            "common_oracle": (subject, oracle["oracle_sha256"]),
            "fetch-model.sh": tuple(re.search(rf"^{name}_sha256=([0-9a-f]{{64}})$", fetch, re.M).group(1) for name in ("q4", "bf16")),
            "release_evidence_manifest.py": (manifest.V04_SUBJECT_MODEL_SHA256["qwen3"], manifest.V04_ORACLE_MODEL_SHA256["qwen3"]),
        }

    def test_all_catalogs_name_the_same_qwen3_pair(self) -> None:
        catalogs = self.catalogs()
        self.assertEqual(len(set(catalogs.values())), 1, catalogs)

    def test_the_checksum_file_agrees_when_present(self) -> None:
        sums = V.ROOT / "models/SHA256SUMS"
        if not sums.is_file():
            self.skipTest("models/SHA256SUMS is not in this checkout")
        listed = dict(line.split()[::-1] for line in sums.read_text().splitlines() if line.strip())
        subject, oracle = self.catalogs()["common_oracle"]
        self.assertEqual((listed["models/Qwen3-8B-Q4_K_M.gguf"], listed["models/Qwen3-8B-BF16.gguf"]), (subject, oracle))

    def test_metadata_pins_name_pinned_pairs_with_exact_fields(self) -> None:
        for subject, pin in C.LOGIT_ORACLE_METADATA.items():
            self.assertIn(subject, C.LOGIT_ORACLE_PAIRS)
            self.assertEqual(sorted(pin), ["architecture_sha256", "tokenizer_keys", "tokenizer_sha256"])
            self.assertEqual(pin["tokenizer_keys"], list(C.GGUF_TOKENIZER_KEYS))
            self.assertTrue(all(len(pin[key]) == 64 for key in ("architecture_sha256", "tokenizer_sha256")))

    def test_the_qwen3_pair_has_a_metadata_pin(self) -> None:
        self.assertEqual(sorted(C.LOGIT_ORACLE_METADATA), sorted(C.LOGIT_ORACLE_PAIRS))


class MonitoringTests(ScratchTests):
    def setUp(self) -> None:
        super().setUp()
        files = write_calibration(self.calibration)
        self.study = V.load_study(files, "calibration", "cuda")
        self.samples, self.collection = write_monitoring(self.root, self.study)

    def write_collection(self, **overrides) -> None:
        record = json.loads(self.collection.read_text())
        self.collection.write_text(json.dumps({**record, **overrides}))

    def test_unbound_monitoring_is_stated_as_a_limit(self) -> None:
        binding = V.monitoring_binding(None, self.study)
        self.assertFalse(binding["bound"])
        self.assertIn("unrecorded", binding["limit"])

    def test_bound_monitoring_records_content_and_range(self) -> None:
        binding = V.monitoring_binding((self.samples, self.collection), self.study)
        self.assertEqual(binding["samples_sha256"], sha(self.samples))
        self.assertEqual(binding["sm_clock_mhz"], [210.0, 2715.0])
        self.assertEqual(binding["median_interval_s"], 1.0)
        self.assertEqual(binding["pstates"], ["P0", "P8"])
        self.assertIn("not attributed", binding["limit"])

    def test_changed_samples_or_receipts_are_rejected(self) -> None:
        self.samples.write_text(self.samples.read_text() + "x")
        with self.assertRaisesRegex(ValueError, "samples differ"):
            V.monitoring_binding((self.samples, self.collection), self.study)
        self.write_collection(gpu_samples_sha256=sha(self.samples))
        record = json.loads(self.collection.read_text())
        record["records"][0]["sha256"] = "0" * 64
        self.collection.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "differs from the receipt"):
            V.monitoring_binding((self.samples, self.collection), self.study)

    def test_collection_must_name_every_receipt(self) -> None:
        record = json.loads(self.collection.read_text())
        record["records"].pop()
        self.collection.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "names other receipts"):
            V.monitoring_binding((self.samples, self.collection), self.study)


if __name__ == "__main__":
    unittest.main()
