"""Tests for the independent retained-history tokenization producer."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
except ImportError:  # pragma: no cover - the producer reports unavailable without it.
    Ed25519PrivateKey = None


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "history_tokenization", ROOT / "scripts" / "produce-history-tokenization.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _receipt(
    private_key,
    model_hash,
    request_bytes,
    prompt_tokens,
    generated_tokens,
    session_id,
    reuse_class,
    cached_tokens,
    reused_tokens,
):
    transcript = [*prompt_tokens, *generated_tokens]
    claim = {
        "schema_version": 1,
        "receipt_id": str(uuid.uuid4()),
        "created_utc": "2026-09-20T00:00:00Z",
        "engine_version": "0.4.0",
        "model_sha256": model_hash,
        "request_sha256": MODULE.sha256_bytes(request_bytes),
        "prompt_tokens_sha256": MODULE.uint32_le_sha256(prompt_tokens),
        "response_tokens_sha256": MODULE.uint32_le_sha256(generated_tokens),
        "transcript_sha256": MODULE.uint32_le_sha256(transcript),
        "seed": 0,
        "prompt_tokens": len(prompt_tokens),
        "generated_tokens": len(generated_tokens),
        "finish_reason": "length",
        "cancelled": False,
        "session": {
            "session_id": session_id,
            "reuse_class": reuse_class,
            "cached_tokens": cached_tokens,
            "reused_tokens": reused_tokens,
            "replayed_tokens": 0,
            "computed_tokens": len(prompt_tokens) - reused_tokens,
        },
    }
    ordered = {field: claim[field] for field in MODULE.RESPONSE_FIELDS if field != "session"}
    ordered["session"] = {
        field: claim["session"][field] for field in MODULE.SESSION_FIELDS
    }
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "claim": claim,
        "public_key_ed25519": private_key.public_key().public_bytes_raw().hex(),
        "signature_ed25519": private_key.sign(payload).hex(),
    }


def _resign(receipt, private_key):
    claim = receipt["claim"]
    ordered = {field: claim[field] for field in MODULE.RESPONSE_FIELDS if field != "session"}
    ordered["session"] = {field: claim["session"][field] for field in MODULE.SESSION_FIELDS}
    payload = json.dumps(ordered, ensure_ascii=False, separators=(",", ":")).encode()
    receipt["signature_ed25519"] = private_key.sign(payload).hex()


def _checked_leone_post(prompt, generated, branch):
    """Return the tokenizer transport stub used by the checked producer path."""

    def post(_endpoint, route, body):
        if route == "/apply-template":
            name = "branch" if len(body["messages"]) > 1 else "parent"
            return {"prompt": name}, {
                "request_sha256": "a" * 64, "response_sha256": "b" * 64,
                "request_hex": "7b7d", "response_hex": "7b7d",
            }
        tokens = {"parent": prompt, "branch": branch, "answer": generated}[body["content"]]
        return {"tokens": tokens}, {
            "request_sha256": "a" * 64, "response_sha256": "b" * 64,
            "request_hex": "7b7d", "response_hex": "7b7d",
        }

    return post


def _write_checked_leone_artifacts(root, engine, parent_events, branch_events, private_key):
    """Write the checked producer inputs and bind their measured fixture hashes."""

    model = root / engine["history_tokenization"]["producer_model_artifact"]
    model.parent.mkdir(parents=True)
    model.write_bytes(
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
        + struct.pack("<Q", len(b"tokenizer.ggml.tokens"))
        + b"tokenizer.ggml.tokens" + struct.pack("<I", 9)
        + struct.pack("<IQ", 8, 1000)
    )
    model_hash = MODULE._file_digest(model)
    for events in (parent_events, branch_events):
        events["leone_receipt"]["claim"]["model_sha256"] = model_hash
        _resign(events["leone_receipt"], private_key)
    binary = root / engine["history_tokenization"]["producer_executable_path"]
    binary.parent.mkdir(parents=True)
    shutil.copyfile(sys.executable, binary)
    binary.chmod(0o755)
    template = root / engine["history_tokenization"]["producer_template_file"]
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_bytes((ROOT / engine["history_tokenization"]["template_file"]).read_bytes())
    key = root / "run.key"
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    key.write_bytes(private_key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()))
    linked_spec = importlib.util.spec_from_file_location(
        "history_linked_libraries_fixture", ROOT / "scripts" / "linked_libraries.py"
    )
    linked = importlib.util.module_from_spec(linked_spec)
    assert linked_spec.loader is not None
    linked_spec.loader.exec_module(linked)
    declaration = engine["history_tokenization"]
    declaration["producer_loaded_library_sha256"] = MODULE.sha256_bytes(
        MODULE.canonical_json(linked.resolve(binary, binary.parent, engine["backend"]))
    )
    return declaration, binary, key


def _run_checked_leone_producer(
    harness, engine, root, key, source_commit, parent_request, parent_events, branch_request, branch_events,
    prompt, generated, branch,
):
    """Run the real producer twice, with the second run using a bad closure pin."""

    generator = MODULE._template_generator()
    post = _checked_leone_post(prompt, generated, branch)
    patches = (
        mock.patch.dict(os.environ, {"LEONE_BRANCHING_SIGNING_KEY": str(key)}, clear=False),
        mock.patch.object(generator, "verify_engine", return_value=source_commit),
        mock.patch.object(generator, "start_server", return_value=(mock.Mock(), "http://tokenizer")),
        mock.patch.object(generator, "stop_server"),
        mock.patch.object(MODULE, "_post_json", side_effect=post),
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        paths = harness._producer_paths(root, engine)
        producer_engine = harness._producer_engine({**engine, "_producer_paths": paths})
        result = MODULE.produce_history_tokenization(
            producer_engine, parent_request, parent_events, branch_request, branch_events
        )
        engine["history_tokenization"]["producer_loaded_library_sha256"] = "0" * 64
        mismatch_paths = harness._producer_paths(root, engine)
        mismatch_engine = harness._producer_engine({**engine, "_producer_paths": mismatch_paths})
        mismatch = MODULE.produce_history_tokenization(
            mismatch_engine, parent_request, parent_events, branch_request, branch_events
        )
    return result, mismatch


def _fixture():
    private_key = Ed25519PrivateKey.generate()
    model_hash = "a" * 64
    parent_request = json.dumps(
        {"messages": [{"role": "user", "content": "question"}]},
        separators=(",", ":"),
    ).encode()
    branch_request = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
            "leone_fork_session": "parent-session",
            "leone_session": "branch-session",
        },
        separators=(",", ":"),
    ).encode()
    parent_prompt = [1, 2]
    generated = [3, 4]
    branch_prompt = [1, 2, 3, 9]
    parent_receipt = _receipt(
        private_key,
        model_hash,
        parent_request,
        parent_prompt,
        generated,
        "parent-session",
        "cold",
        2,
        0,
    )
    branch_receipt = _receipt(
        private_key,
        model_hash,
        branch_request,
        branch_prompt,
        [8],
        "branch-session",
        "device-fork",
        3,
        3,
    )
    parent_events = {
        "service_request_id": "parent-request",
        "process_instance_id": "process-1",
        "workload_epoch": "epoch-1",
        "choices": [{"message": {"content": "answer"}}],
        "leone_receipt": parent_receipt,
    }
    branch_events = {
        "service_request_id": "branch-request",
        "parent_service_request_id": "parent-request",
        "process_instance_id": "process-1",
        "workload_epoch": "epoch-1",
        "choices": [{"message": {"content": "branch answer"}}],
        "leone_receipt": branch_receipt,
    }
    prepared = {
        "trusted_public_key": parent_receipt["public_key_ed25519"],
        "model_sha256": model_hash,
        "source_commit": "d" * 40,
        "executable_sha256": "b" * 64,
        "loaded_library_sha256": "c" * 64,
        "tokenizer_metadata_sha256": "e" * 64,
        "tokenizer_metadata_hash_scheme": "tokenizer_config_json_bytes",
        "template_config_sha256": "f" * 64,
        "template_config_hash_scheme": "tokenizer_config_json_bytes",
        "template_bytes_sha256": "1" * 64,
        "special_tokens_policy_sha256": "2" * 64,
        "special_tokens_policy": {"prompt": {"add_special": False, "parse_special": True}},
        "vocab_size": 100,
    }
    return (
        prepared,
        parent_request,
        parent_events,
        branch_request,
        branch_events,
        parent_prompt,
        generated,
        branch_prompt,
        private_key,
    )


class TokenizerResponseBoundTests(unittest.TestCase):
    class Response:
        def __init__(self, body):
            self.body = body
            self.read_size = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, size):
            self.read_size = size
            return self.body

    def test_tokenizer_response_reads_one_byte_past_limit(self):
        exact = self.Response(b"{}")
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=exact):
            MODULE._post_json("http://127.0.0.1:1", "/tokenize", {})
        self.assertEqual(exact.read_size, MODULE.MAX_TOKENIZER_RESPONSE_BYTES + 1)

        oversized = self.Response(b"x" * (MODULE.MAX_TOKENIZER_RESPONSE_BYTES + 1))
        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=oversized):
            with self.assertRaises(MODULE._Unavailable) as raised:
                MODULE._post_json("http://127.0.0.1:1", "/tokenize", {})
        self.assertEqual(raised.exception.reason, "tokenizer_response_too_large")


@unittest.skipUnless(Ed25519PrivateKey is not None, "cryptography is required for signed fixtures")
class HistoryTokenizationTests(unittest.TestCase):
    def test_checked_calibration_declaration_reaches_observed_leone_producer(self):
        harness_spec = importlib.util.spec_from_file_location(
            "branching_history_harness", ROOT / "scripts" / "study-branching-service.py"
        )
        harness = importlib.util.module_from_spec(harness_spec)
        assert harness_spec.loader is not None
        harness_spec.loader.exec_module(harness)
        manifest = json.loads((ROOT / "benchmarks" / "branching-service-calibration.json").read_text())
        engine = next(item for item in manifest["engines"] if item["id"] == "leone")
        _, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, private_key = _fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, key = _write_checked_leone_artifacts(
                root, engine, parent_events, branch_events, private_key
            )
            result, mismatch = _run_checked_leone_producer(
                harness, engine, root, key,
                manifest["engines"][1]["history_tokenization"]["producer_source_commit"],
                parent_request, parent_events, branch_request, branch_events,
                prompt, generated, branch,
            )
        self.assertEqual(result["status"], "observed", result)
        self.assertEqual(result["parent_evaluated_token_ids"], [1, 2, 3])
        self.assertEqual(result["request_token_ids"], branch)
        self.assertEqual(mismatch, {
            "schema_version": MODULE.SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "loaded_library_hash_mismatch",
        })

    def test_success_requires_full_evaluated_prefix_and_signed_counts(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, _ = fixture
        fake_oracle = {
            "parent_tokens": prompt,
            "generated_tokens": generated,
            "branch_tokens": branch,
            "records": {"parent": {}, "branch": {}, "generated": {}},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["parent_evaluated_token_ids"], [1, 2, 3])
        self.assertEqual(result["expected_reused_token_count"], 3)
        self.assertEqual(result["observed_reused_token_count"], 3)
        self.assertNotIn("retained_history_lcp", result)

    def test_positive_short_lcp_is_unavailable(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, _, private_key = fixture
        altered_branch = [1, 2, 8, 9]
        claim = branch_events["leone_receipt"]["claim"]
        claim["prompt_tokens_sha256"] = MODULE.uint32_le_sha256(altered_branch)
        _resign(branch_events["leone_receipt"], private_key)
        fake_oracle = {
            "parent_tokens": prompt,
            "generated_tokens": generated,
            "branch_tokens": altered_branch,
            "records": {},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result, {
            "schema_version": MODULE.SCHEMA_VERSION,
            "status": "unavailable",
            "reason": "retained_history_is_not_full_branch_prefix",
        })

    def test_signed_transcript_hash_is_checked_in_uint32_little_endian(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, _ = fixture
        parent_events = json.loads(json.dumps(parent_events))
        parent_events["leone_receipt"]["claim"]["transcript_sha256"] = "0" * 64
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value={
                "parent_tokens": prompt,
                "generated_tokens": generated,
                "branch_tokens": branch,
                "records": {},
            }
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "signed_receipt_signature_invalid")

    def test_tools_and_speculation_are_typed_unavailable(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, *_ = fixture
        body = json.loads(parent_request)
        body["tools"] = [{"type": "function"}]
        request = json.dumps(body, separators=(",", ":")).encode()
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared):
            result = MODULE.produce_history_tokenization(
                {}, request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "unsupported_tools")

    def test_cached_count_binds_to_plain_evaluated_boundary(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, private_key = fixture
        branch_events["leone_receipt"]["claim"]["session"]["cached_tokens"] = 4
        _resign(branch_events["leone_receipt"], private_key)
        fake_oracle = {"parent_tokens": prompt, "generated_tokens": generated, "branch_tokens": branch, "records": {}}
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "cached_count_does_not_match_evaluated_boundary")

    def test_reused_count_must_equal_full_evaluated_boundary(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, private_key = fixture
        branch_events["leone_receipt"]["claim"]["session"]["reused_tokens"] = 2
        _resign(branch_events["leone_receipt"], private_key)
        fake_oracle = {"parent_tokens": prompt, "generated_tokens": generated, "branch_tokens": branch, "records": {}}
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "reused_count_does_not_match_evaluated_boundary")

    def test_branch_cannot_end_at_the_evaluated_boundary(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, _, private_key = fixture
        branch_tokens = [1, 2, 3]
        claim = branch_events["leone_receipt"]["claim"]
        claim["prompt_tokens"] = len(branch_tokens)
        claim["prompt_tokens_sha256"] = MODULE.uint32_le_sha256(branch_tokens)
        _resign(branch_events["leone_receipt"], private_key)
        fake_oracle = {"parent_tokens": prompt, "generated_tokens": generated, "branch_tokens": branch_tokens, "records": {}}
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "retained_history_is_not_full_branch_prefix")

    def test_branch_request_hash_and_session_identity_are_bound(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, *_ = fixture
        changed = json.loads(branch_request)
        changed["leone_fork_session"] = "other-parent"
        changed_request = json.dumps(changed, separators=(",", ":")).encode()
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, changed_request, branch_events
            )
        self.assertEqual(result["reason"], "signed_request_hash_mismatch")

    def test_declared_running_process_identity_is_required(self):
        fixture = _fixture()
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, _ = fixture
        prepared["running_identity"] = {
            "start": {"identity": {"process_instance_id": "different-process"}}
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value={
                "parent_tokens": prompt,
                "generated_tokens": generated,
                "branch_tokens": branch,
                "records": {},
            }
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "service_process_identity_mismatch")

    def test_request_bytes_are_bound_before_json_normalization(self):
        fixture = _fixture()
        prepared, _, parent_events, branch_request, branch_events, *_ = fixture
        request = {"messages": [{"role": "user", "content": "question"}]}
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared):
            result = MODULE.produce_history_tokenization(
                {}, request, parent_events, branch_request, branch_events
            )
        self.assertEqual(result["reason"], "request_bytes_required")


def _llama_fixture():
    parent_request = json.dumps(
        {
            "messages": [{"role": "user", "content": "question"}],
            "stream": False,
            "verbose": True,
            "return_tokens": True,
            "cache_prompt": True,
            "id_slot": 7,
        },
        separators=(",", ":"),
    ).encode()
    branch_request = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "follow-up"},
            ],
            "stream": True,
            "verbose": True,
            "cache_prompt": True,
            "id_slot": 7,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()
    parent = {
        "service_request_id": "parent-request",
        "process_instance_id": "process-1",
        "workload_epoch": "epoch-1",
        "model": "model.gguf",
        "choices": [{"message": {"content": "answer"}}],
        "__verbose": {
            "tokens": [3, 4],
            "prompt": "parent-rendered",
            "id_slot": 7,
            "tokens_evaluated": 2,
            "tokens_predicted": 2,
            "tokens_cached": 3,
            "stop_type": "limit",
            "truncated": False,
            "generation_settings": {"speculative.types": "none"},
        },
    }
    branch = {
        "service_request_id": "branch-request",
        "parent_service_request_id": "parent-request",
        "process_instance_id": "process-1",
        "workload_epoch": "epoch-1",
        "model": "model.gguf",
        "choices": [{"finish_reason": "length", "delta": {}}],
        "usage": {"prompt_tokens": 4, "prompt_tokens_details": {"cached_tokens": 3}},
        "__verbose": {"tokens": [], "prompt": "branch-rendered", "id_slot": 7},
    }
    prepared = {
        "source_commit": "d" * 40,
        "executable_sha256": "b" * 64,
        "loaded_library_sha256": "c" * 64,
        "model_sha256": "a" * 64,
        "tokenizer_metadata_sha256": "e" * 64,
        "tokenizer_metadata_hash_scheme": "tokenizer_config_json_bytes",
        "template_config_sha256": "f" * 64,
        "template_config_hash_scheme": "tokenizer_config_json_bytes",
        "template_bytes_sha256": "1" * 64,
        "special_tokens_policy_sha256": "2" * 64,
        "special_tokens_policy": {"prompt": {"add_special": False, "parse_special": True}},
        "vocab_size": 100,
        "engine": "llama.cpp",
        "model": "model.gguf",
    }
    return prepared, parent_request, parent, branch_request, branch


def _exchange(action, slot, filename, tokens, size, start_ns, end_ns, status=200):
    """Build one retained slot save or restore exchange as the harness records it."""

    count_fields = ("n_saved", "n_written") if action == "save" else ("n_restored", "n_read")
    reply = {"id_slot": slot, "filename": filename, count_fields[0]: tokens, count_fields[1]: size}
    return {
        "action": action, "slot": slot, "http_status": status,
        "request_hex": json.dumps({"filename": filename}, separators=(",", ":")).encode().hex(),
        "response_hex": json.dumps(reply, separators=(",", ":")).encode().hex(),
        "request_start_ns": start_ns, "request_end_ns": end_ns,
    }


def _concurrent_llama_fixture(branch_slot=8):
    """Build a llama.cpp parent in slot 7 and a warm branch in another slot."""

    prepared, parent_request, parent, branch_request, branch = _llama_fixture()
    request = json.loads(branch_request)
    request["id_slot"] = branch_slot
    branch_request = json.dumps(request, separators=(",", ":")).encode()
    branch["__verbose"]["id_slot"] = branch_slot
    branch["slot_copy"] = {
        "save": _exchange("save", 7, "parent.slot", 3, 4096, 10, 20),
        "restore": _exchange("restore", branch_slot, "parent.slot", 3, 4096, 30, 45),
        "branch_request_start_ns": 100,
    }
    return prepared, parent_request, parent, branch_request, branch


def _produce_llama(prepared, parent_request, parent, branch_request, branch):
    fake_oracle = {
        "parent_tokens": [1, 2], "generated_tokens": [3, 4], "branch_tokens": [1, 2, 3, 9],
        "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {},
    }
    with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
        MODULE, "_run_tokenizer", return_value=fake_oracle
    ):
        return MODULE.produce_history_tokenization(
            {"protocol": "llama.cpp", "id_slot": 7}, parent_request, parent, branch_request, branch
        )


def _set_exchange_reply(exchange, **fields):
    reply = json.loads(bytes.fromhex(exchange["response_hex"]))
    reply.update(fields)
    exchange["response_hex"] = json.dumps(reply, separators=(",", ":")).encode().hex()



def _wrong_restore_slot(branch):
    branch["slot_copy"]["restore"] = _exchange("restore", 9, "parent.slot", 3, 4096, 30, 45)


def _wrong_file(branch):
    branch["slot_copy"]["restore"] = _exchange("restore", 8, "other.slot", 3, 4096, 30, 45)


def _short_save(branch):
    branch["slot_copy"]["save"] = _exchange("save", 7, "parent.slot", 2, 4096, 10, 20)
    branch["slot_copy"]["restore"] = _exchange("restore", 8, "parent.slot", 2, 4096, 30, 45)


def _byte_mismatch(branch):
    _set_exchange_reply(branch["slot_copy"]["restore"], n_read=1)


def _restore_before_save(branch):
    branch["slot_copy"]["restore"] = _exchange("restore", 8, "parent.slot", 3, 4096, 5, 8)


def _restore_after_branch(branch):
    branch["slot_copy"]["branch_request_start_ns"] = 40


def _failed_restore(branch):
    branch["slot_copy"]["restore"] = _exchange("restore", 8, "parent.slot", 3, 4096, 30, 45, status=501)


def _save_reply_slot(branch):
    _set_exchange_reply(branch["slot_copy"]["save"], id_slot=3)


def _missing_restore(branch):
    branch["slot_copy"]["restore"] = None


class LlamaCppHistoryTokenizationTests(unittest.TestCase):
    def test_nonstream_parent_and_stream_branch_are_checked(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        fake_oracle = {
            "parent_tokens": [1, 2],
            "generated_tokens": [3, 4],
            "branch_tokens": [1, 2, 3, 9],
            "parent_rendered": "parent-rendered",
            "branch_rendered": "branch-rendered",
            "records": {},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request,
                parent,
                branch_request,
                branch,
            )
        self.assertEqual(result["status"], "observed")
        self.assertEqual(result["oracle"]["proof_adapter"], "llama.cpp.verbose")
        self.assertIsNone(result["parent_receipt_sha256"])
        record = MODULE.harness_tokenization_record(result)
        self.assertEqual(record["prefix_token_ids"], [1, 2, 3])

    def test_evidence_retains_template_kwargs_and_is_labelled_retained_and_recomputed(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        prepared["special_tokens_policy"]["chat_template_kwargs"] = {"enable_thinking": True}
        result = _produce_llama(prepared, parent_request, parent, branch_request, branch)
        self.assertEqual(result["status"], "observed")
        oracle = result["oracle"]
        self.assertEqual(oracle["apply_template_kwargs"], {
            "parent": {"enable_thinking": True}, "branch": {"enable_thinking": True},
        })
        self.assertEqual(oracle["evidence_class"], "retained_and_recomputed")
        self.assertEqual(oracle["special_tokens_policy"]["chat_template_kwargs"], {"enable_thinking": True})

    def test_template_kwargs_in_the_request_override_the_policy_and_are_retained(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        prepared["special_tokens_policy"]["chat_template_kwargs"] = {"enable_thinking": True}
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        parent_request = json.dumps({**json.loads(parent_request), **body}, separators=(",", ":")).encode()
        branch_request = json.dumps({**json.loads(branch_request), **body}, separators=(",", ":")).encode()
        oracle = _produce_llama(prepared, parent_request, parent, branch_request, branch)["oracle"]
        self.assertEqual(oracle["apply_template_kwargs"]["parent"], {"enable_thinking": False})

    def test_rendered_prompts_must_end_with_the_declared_generation_prompt(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        prepared["special_tokens_policy"]["generation_prompt_suffix"] = "-rendered"
        self.assertEqual(_produce_llama(prepared, parent_request, parent, branch_request, branch)["status"], "observed")
        prepared["special_tokens_policy"]["generation_prompt_suffix"] = "<|im_start|>assistant\n"
        result = _produce_llama(prepared, parent_request, parent, branch_request, branch)
        self.assertEqual(result["reason"], "llama_generation_prompt_incompatible")
        prepared["special_tokens_policy"]["generation_prompt_suffix"] = ""
        self.assertEqual(
            _produce_llama(prepared, parent_request, parent, branch_request, branch)["reason"],
            "generation_prompt_suffix_invalid",
        )

    def test_fresh_generated_text_ids_must_equal_the_verbose_sampled_ids(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        oracle = {
            "parent_tokens": [1, 2], "generated_tokens": [3, 5], "branch_tokens": [1, 2, 3, 9],
            "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=oracle
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7}, parent_request, parent, branch_request, branch
            )
        self.assertEqual(result["reason"], "generated_token_ids_differ_from_fresh_tokenizer")

    def test_a_natural_stop_is_typed_unavailable_never_a_forged_warm_match(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        branch["choices"][0]["finish_reason"] = "stop"
        result = _produce_llama(prepared, parent_request, parent, branch_request, branch)
        self.assertEqual(result["reason"], "unsupported_llama_branch_stop")
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        parent["__verbose"]["stop_type"] = "eos"
        result = _produce_llama(prepared, parent_request, parent, branch_request, branch)
        self.assertEqual(result["reason"], "unsupported_llama_parent_stop")

    def test_llama_parent_link_is_required(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        del branch["parent_service_request_id"]
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value={
                "parent_tokens": [1, 2], "generated_tokens": [3, 4], "branch_tokens": [1, 2, 3, 9],
                "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {},
            }
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request, parent, branch_request, branch,
            )
        self.assertEqual(result["reason"], "parent_service_request_identity_missing")

    def test_llama_response_reports_the_model_file_name(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        prepared["model"] = "/models/dir/model.gguf"
        result = _produce_llama(prepared, parent_request, parent, branch_request, branch)
        self.assertEqual(result["status"], "observed")
        parent["model"] = "other.gguf"
        self.assertEqual(
            _produce_llama(prepared, parent_request, parent, branch_request, branch)["reason"],
            "llama_response_model_mismatch",
        )

    def test_llama_response_model_is_required(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        del parent["model"]
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value={
                "parent_tokens": [1, 2], "generated_tokens": [3, 4], "branch_tokens": [1, 2, 3, 9],
                "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {},
            }
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request, parent, branch_request, branch,
            )
        self.assertEqual(result["reason"], "llama_response_model_missing")

    def test_llama_native_speculation_is_unavailable_for_plain_parent(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        parent["__verbose"]["generation_settings"]["speculative.types"] = "draft"
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value={
                "parent_tokens": [1, 2], "generated_tokens": [3, 4], "branch_tokens": [1, 2, 3, 9],
                "parent_rendered": "parent-rendered", "branch_rendered": "branch-rendered", "records": {},
            }
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request, parent, branch_request, branch,
            )
        self.assertEqual(result["reason"], "unsupported_llama_parent_speculation")

    def test_streaming_parent_without_sampled_tokens_is_unavailable(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        parent["__verbose"]["tokens"] = []
        fake_oracle = {
            "parent_tokens": [1, 2],
            "generated_tokens": [],
            "branch_tokens": [1, 2, 3, 9],
            "parent_rendered": "parent-rendered",
            "branch_rendered": "branch-rendered",
            "records": {},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request,
                parent,
                branch_request,
                branch,
            )
        self.assertEqual(result["reason"], "llama_parent_generated_tokens_missing")

    def test_llama_slot_change_is_unavailable(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        branch_request = json.loads(branch_request)
        branch_request["id_slot"] = 8
        branch_request = json.dumps(branch_request, separators=(",", ":")).encode()
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared):
            result = MODULE.produce_history_tokenization(
                {"protocol": "llama.cpp", "id_slot": 7},
                parent_request,
                parent,
                branch_request,
                branch,
            )
        self.assertEqual(result["reason"], "llama_slot_copy_missing")

    def test_concurrent_branch_in_another_slot_needs_and_records_a_slot_copy(self):
        result = _produce_llama(*_concurrent_llama_fixture())
        self.assertEqual(result["status"], "observed")
        copy = result["oracle"]["slot_copy"]
        self.assertEqual((copy["parent_slot"], copy["branch_slot"]), (7, 8))
        self.assertEqual((copy["saved_token_count"], copy["restored_token_count"]), (3, 3))
        self.assertEqual(copy["restore_elapsed_ns"], 15)
        self.assertEqual(result["observed_reused_token_count"], 3)
        self.assertNotIn("slot_copy", _produce_llama(*_llama_fixture())["oracle"])

    def test_cold_branch_is_rejected_even_when_a_slot_copy_is_retained(self):
        fixture = _concurrent_llama_fixture()
        fixture[4]["usage"] = {"prompt_tokens": 4, "prompt_tokens_details": {"cached_tokens": 0}}
        self.assertEqual(
            _produce_llama(*fixture)["reason"], "reused_count_does_not_match_evaluated_boundary"
        )

    def test_slot_copy_lineage_breaks_are_rejected(self):
        cases = (
            (_wrong_restore_slot, "llama_slot_copy_action_mismatch"),
            (_missing_restore, "llama_slot_copy_action_mismatch"),
            (_wrong_file, "llama_slot_copy_file_mismatch"),
            (_short_save, "llama_slot_copy_boundary_mismatch"),
            (_byte_mismatch, "llama_slot_copy_count_mismatch"),
            (_restore_before_save, "llama_slot_copy_order_invalid"),
            (_restore_after_branch, "llama_slot_copy_order_invalid"),
            (_failed_restore, "llama_slot_copy_failed"),
            (_save_reply_slot, "llama_slot_copy_response_mismatch"),
        )
        for mutate, reason in cases:
            with self.subTest(mutation=mutate.__name__):
                fixture = _concurrent_llama_fixture()
                mutate(fixture[4])
                self.assertEqual(_produce_llama(*fixture)["reason"], reason)

    def test_branch_must_run_in_the_slot_it_requested(self):
        fixture = _concurrent_llama_fixture()
        fixture[4]["__verbose"]["id_slot"] = 7
        self.assertEqual(_produce_llama(*fixture)["reason"], "llama_branch_slot_mismatch")

    def test_slot_copy_is_rejected_when_branch_shares_the_parent_slot(self):
        prepared, parent_request, parent, branch_request, branch = _llama_fixture()
        branch["slot_copy"] = _concurrent_llama_fixture()[4]["slot_copy"]
        self.assertEqual(
            _produce_llama(prepared, parent_request, parent, branch_request, branch)["reason"],
            "llama_slot_copy_unexpected",
        )


class TokenizerServerLaunchTests(unittest.TestCase):
    def _generator(self):
        return MODULE._template_generator()

    def test_tokenizer_server_launches_on_two_cpu_threads_with_no_device(self):
        generator = self._generator()
        process = mock.Mock()
        health = mock.MagicMock()
        health.__enter__.return_value.status = 200
        with mock.patch.object(generator.subprocess, "Popen", return_value=process) as popen, mock.patch.object(
            generator.urllib.request, "urlopen", return_value=health
        ):
            generator.start_server("llama-server", "model.gguf", "template.jinja")
        argv = popen.call_args.args[0]
        pairs = set(zip(argv, argv[1:]))
        for flag in (("--n-gpu-layers", "0"), ("--device", "none"), ("--threads", "2"),
                     ("--threads-batch", "2"), ("--parallel", "1"), ("--ctx-size", "8192"),
                     ("--chat-template-file", "template.jinja")):
            self.assertIn(flag, pairs)
        self.assertIn("--no-op-offload", argv)

    def test_a_server_that_never_becomes_healthy_is_terminated(self):
        generator = self._generator()
        process = mock.Mock()
        with mock.patch.object(generator.subprocess, "Popen", return_value=process), mock.patch.object(
            generator, "HEALTH_TIMEOUT_SECONDS", 0
        ):
            with self.assertRaises(RuntimeError):
                generator.start_server("llama-server", "model.gguf", "template.jinja")
        process.terminate.assert_called_once_with()
        process.wait.assert_called_once_with(timeout=10)

    def test_producer_hides_cuda_devices_only_during_the_launch(self):
        generator = self._generator()
        seen = {}

        def launch(*_):
            seen.update(cuda=os.environ.get("CUDA_VISIBLE_DEVICES"), disable=os.environ.get("GGML_CUDA_DISABLE"))
            raise RuntimeError("stop")

        prepared = {"template_bytes": b"t", "binary": "llama-server", "model": "m.gguf"}
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), mock.patch.object(
            generator, "start_server", launch
        ):
            os.environ.pop("GGML_CUDA_DISABLE", None)
            with self.assertRaises(RuntimeError):
                MODULE._run_tokenizer(prepared, {}, {}, "text")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(seen, {"cuda": "", "disable": None})


class HarnessAdapterTests(unittest.TestCase):
    def test_adapter_keeps_harness_shape(self):
        evidence = {
            "status": "observed",
            "parent_evaluated_token_ids": [1, 2, 3],
            "request_token_ids": [1, 2, 3, 9],
            "service_request_id": "branch-request",
            "canonical_prefix_sha256": "a" * 64,
            "canonical_request_sha256": "b" * 64,
            "oracle": {
                "tokenizer_metadata_sha256": "c" * 64,
                "template_config_sha256": "d" * 64,
                "special_tokens_policy_sha256": "e" * 64,
                "vocab_size": 100,
            },
        }
        record = MODULE.harness_tokenization_record(evidence)
        self.assertEqual(record["status"], "observed")
        self.assertEqual(record["prefix_token_count"], 3)
        self.assertEqual(record["request_token_count"], 4)
        self.assertEqual(record["service_request_id"], "branch-request")

    def test_producer_record_passes_frozen_harness_checker(self):
        harness_path = os.environ.get("LEONE_HISTORY_HARNESS")
        if harness_path is None:
            harness_path = str(ROOT / "scripts" / "study-branching-service.py")
        if not Path(harness_path).is_file():
            self.skipTest("set LEONE_HISTORY_HARNESS to the public-audit harness")
        spec = importlib.util.spec_from_file_location("history_harness", harness_path)
        harness = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(harness)
        prepared, parent_request, parent_events, branch_request, branch_events, prompt, generated, branch, private_key = _fixture()
        branch_body = json.loads(branch_request)
        branch_body["leone_session"] = "branch-request"
        branch_request = json.dumps(branch_body, separators=(",", ":")).encode()
        branch_events["leone_receipt"]["claim"]["request_sha256"] = MODULE.sha256_bytes(branch_request)
        branch_events["leone_receipt"]["claim"]["session"]["session_id"] = "branch-request"
        _resign(branch_events["leone_receipt"], private_key)
        fake_oracle = {
            "parent_tokens": prompt,
            "generated_tokens": generated,
            "branch_tokens": branch,
            "records": {},
        }
        with mock.patch.object(MODULE, "_prepare_engine", return_value=prepared), mock.patch.object(
            MODULE, "_run_tokenizer", return_value=fake_oracle
        ):
            result = MODULE.produce_history_tokenization(
                {}, parent_request, parent_events, branch_request, branch_events
            )
        prefix = json.loads(parent_request)["messages"] + [
            {"role": "assistant", "content": "answer"}
        ]
        request = json.loads(branch_request)["messages"]
        parent_record = _harness_record(parent_events, parent_request, "answer")
        branch_record = _harness_record(branch_events, branch_request, "branch answer")
        history = {
            "status": "observed",
            "tokenization": MODULE.harness_tokenization_evidence(result),
            "request_material": {
                "prefix_messages": prefix,
                "request_messages": request,
                "request_body": json.loads(branch_request),
            },
            "reuse_count": {"status": "observed", "values": [3]},
        }
        errors = harness._history_tokenization_errors(history, parent_record, branch_record)
        self.assertEqual(errors, [])


def _harness_record(events, request_bytes, content):
    receipt = events["leone_receipt"]
    return {
        "service_request_id": events["service_request_id"],
        "request_id": events["service_request_id"],
        "content_sha256": MODULE.sha256_bytes(content.encode()),
        "_request_bytes_hex": request_bytes.hex(),
        "response_receipt": receipt,
        "_response_receipt_sha256": MODULE.sha256_bytes(MODULE.canonical_json(receipt)),
        "usage": {"prompt_tokens": receipt["claim"]["prompt_tokens"],
                  "completion_tokens": receipt["claim"]["generated_tokens"]},
    }


if __name__ == "__main__":
    unittest.main()
