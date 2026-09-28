"""Tests for the fresh tokenizer re-execution check of retained llama.cpp history.

The fresh server is a small Python executable that answers `/apply-template` and
`/tokenize` and logs the argv and environment it started with. The producer and
the comparator under test run for real. Only the tokenizer binary is a stand-in.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
NATIVE_RUN = Path(os.environ["LEONE_HISTORY_NATIVE_RUN"]) if os.environ.get("LEONE_HISTORY_NATIVE_RUN") else None
PINNED = (ROOT / "external" / "PINNED").read_text().strip()
LIBRARY_SHA256 = hashlib.sha256(b"[]").hexdigest()
VOCABULARY = 1000
ANSWER = "answer"


def _load(name, file_name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECK = _load("history_reexecution_check", "check-history-tokenization.py")
PRODUCER = _load("history_reexecution_producer", "produce-history-tokenization.py")

FAKE_SERVER = r'''#!__PYTHON__
import http.server, json, os, sys

OFFSET = __OFFSET__
ANSWER_OFFSET = __ANSWER_OFFSET__
argv = sys.argv[1:]
if "--version" in argv:
    sys.stderr.write("version: 1 (build 1, commit __COMMIT__)\n")
    raise SystemExit(0)


def flag(name):
    return argv[argv.index(name) + 1]


TAG = open(flag("--chat-template-file")).read().strip()
if os.environ.get("FAKE_LLAMA_LOG"):
    with open(os.environ["FAKE_LLAMA_LOG"], "a") as sink:
        sink.write(json.dumps({
            "argv": argv,
            "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": os.environ.get("LLAMA_ARG_DEVICE"),
        }) + "\n")
if os.environ.get("FAKE_LLAMA_MUTATE"):
    with open(os.environ["FAKE_LLAMA_MUTATE"], "ab") as sink:
        sink.write(b"x")


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_value(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.send_value({"status": "ok"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            text = TAG + "".join("<%s>%s" % (m["role"], m["content"]) for m in body["messages"])
            self.send_value({"prompt": text + "<assistant>"})
        else:
            shift = OFFSET + (ANSWER_OFFSET if body["content"] == "answer" else 0)
            tokens = [ord(c) % 997 + shift for c in body["content"]]
            self.send_value({"tokens": tokens})


http.server.HTTPServer(("127.0.0.1", int(flag("--port"))), Handler).serve_forever()
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def toy_tokens(text):
    return [ord(character) % 997 for character in text]


def write_server(path, offset=0, answer_offset=0):
    source = FAKE_SERVER.replace("__PYTHON__", sys.executable)
    source = source.replace("__OFFSET__", str(offset)).replace("__COMMIT__", PINNED[:12])
    source = source.replace("__ANSWER_OFFSET__", str(answer_offset))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o755)
    return path


def compact(value):
    return json.dumps(value, separators=(",", ":")).encode()


def exchange(action, slot, tokens, start_ns, end_ns):
    names = ("n_saved", "n_written") if action == "save" else ("n_restored", "n_read")
    reply = {"id_slot": slot, "filename": "p.slot", names[0]: tokens, names[1]: 4096}
    return {
        "action": action, "slot": slot, "http_status": 200,
        "request_hex": compact({"filename": "p.slot"}).hex(),
        "response_hex": compact(reply).hex(),
        "request_start_ns": start_ns, "request_end_ns": end_ns,
    }


class Workspace:
    """Holds the pinned files and builds retained records with the real producer."""

    def __init__(self, directory, template="tpl-a", policy=None, wire_model=None):
        self.dir = Path(directory)
        self.wire_model = wire_model
        self.binary = write_server(self.dir / "bin" / "llama-server")
        self.tools = self.dir / "tools"
        self.tools.mkdir()
        self.ldd = self.tools / "ldd"
        self.ldd.write_text("#!/bin/sh\nexit 0\n")
        self.ldd.chmod(0o755)
        self.model = self.dir / "models" / "toy-model.gguf"
        self.model.parent.mkdir()
        self.model.write_bytes(b"toy model bytes")
        self.template = self.dir / "template.jinja"
        self.template.write_text(template)
        self.policy = policy or {"prompt": {"add_special": False, "parse_special": True}}
        self.log = self.dir / "server.log"
        self.identity = {
            "process_instance_id": "p1", "workload_epoch": "e1", "source_id": "llama_cpp:local_process",
        }

    def engine(self):
        return {
            "protocol": "llama.cpp", "engine": "llama.cpp",
            "llama_server_path": str(self.binary), "model_path": str(self.model),
            "llama_cpp_commit": PINNED, "executable_sha256": sha(self.binary.read_bytes()),
            "model_sha256": sha(self.model.read_bytes()), "loaded_library_sha256": LIBRARY_SHA256,
            "template_mode": "legacy", "template_file": str(self.template),
            "special_tokens_policy": self.policy, "vocab_size": VOCABULARY, "prompt_prefix": "",
            "_running_identity": {"start": {"identity": self.identity}},
        }

    def requests(self, branch_slot):
        parent = {
            "messages": [{"role": "user", "content": "question"}], "stream": False,
            "verbose": True, "return_tokens": True, "cache_prompt": True, "id_slot": 7,
        }
        branch = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": ANSWER},
                {"role": "user", "content": "follow-up"},
            ],
            "stream": True, "verbose": True, "cache_prompt": True, "id_slot": branch_slot,
            "stream_options": {"include_usage": True},
        }
        return compact(parent), compact(branch)

    def render(self, messages):
        text = self.template.read_text().strip()
        return text + "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def events(self, branch_slot):
        question = [{"role": "user", "content": "question"}]
        parent_prompt = self.render(question)
        branch_messages = question + [
            {"role": "assistant", "content": ANSWER}, {"role": "user", "content": "follow-up"},
        ]
        parent_ids, generated = toy_tokens(parent_prompt), toy_tokens(ANSWER)
        cached = len(parent_ids) + len(generated) - 1
        parent = {
            "model": self.wire_model or "toy-model.gguf", "choices": [{"message": {"content": ANSWER}}],
            "__verbose": {
                "tokens": generated, "prompt": parent_prompt, "id_slot": 7,
                "tokens_evaluated": len(parent_ids), "tokens_predicted": len(generated),
                "tokens_cached": cached, "stop_type": "limit", "truncated": False,
                "generation_settings": {"speculative.types": "none"},
            },
        }
        branch = {
            "model": self.wire_model or "toy-model.gguf", "choices": [{"finish_reason": "length", "delta": {}}],
            "usage": {
                "prompt_tokens": len(toy_tokens(self.render(branch_messages))),
                "prompt_tokens_details": {"cached_tokens": cached},
            },
            "__verbose": {"prompt": self.render(branch_messages), "id_slot": branch_slot},
        }
        return [{"event": parent, "received_ns": 1}], [{"event": branch, "received_ns": 2}], cached

    def wrapper(self, events, request_id, parent_id=None, copy_claim=None):
        wrapper = {"events": events, "service_request_id": request_id, **self.identity}
        if parent_id is not None:
            wrapper["parent_service_request_id"] = parent_id
        if copy_claim is not None:
            wrapper["slot_copy"] = copy_claim
        return wrapper

    def collect(self, branch_slot=7):
        """Run the real collection producer and return one run record."""

        parent_request, branch_request = self.requests(branch_slot)
        parent_events, branch_events, cached = self.events(branch_slot)
        plan = self.slot_plan(branch_slot, cached)
        claim = plan and {"save": plan["save"], "restore": plan["restore"][str(branch_slot)],
                          "branch_request_start_ns": 100}
        with self.path_environment():
            evidence = PRODUCER.produce_history_tokenization(
                self.engine(), parent_request, self.wrapper(parent_events, "parent-1"),
                branch_request, self.wrapper(branch_events, "branch-1", "parent-1", claim),
            )
        record = PRODUCER.harness_tokenization_evidence(evidence)
        assert record["status"] == "observed", record
        return self.run_record(parent_request, parent_events, branch_request, branch_events, record, plan)

    def slot_plan(self, branch_slot, cached):
        if branch_slot == 7:
            return None
        return {
            "parent_slot": 7, "filename": "p.slot",
            "save": exchange("save", 7, cached, 10, 20),
            "restore": {str(branch_slot): exchange("restore", branch_slot, cached, 30, 45)},
        }

    def run_record(self, parent_request, parent_events, branch_request, branch_events, record, plan):
        return {
            "engine": "llama_cpp",
            "parent": {
                "_request_bytes_hex": parent_request.hex(), "_raw_events": parent_events,
                "service_request_id": "parent-1", "request_start_ns": 10,
            },
            "branches": [{
                "_request_bytes_hex": branch_request.hex(), "_raw_events": branch_events,
                "service_request_id": "branch-1", "request_start_ns": 100,
                "history_reuse": {"tokenization": record},
            }],
            "running_identity": {"start": {"identity": self.identity}},
            "slot_copy": plan,
        }

    def expected(self, input_bytes, binary=None, model=None, template=None):
        binary, model, template = binary or self.binary, model or self.model, template or self.template
        producer_dir = ROOT / "scripts"
        template_bytes = template.read_bytes()
        return {
            "schema_version": CHECK.EXPECTED_SCHEMA, "source_commit": PINNED,
            "template_mode": "legacy", "vocab_size": VOCABULARY, "prompt_prefix": "",
            "special_tokens_policy": self.policy,
            "special_tokens_policy_sha256": sha(PRODUCER.canonical_json(self.policy)),
            "input_sha256": sha(input_bytes),
            "producer_sha256": sha((producer_dir / "produce-history-tokenization.py").read_bytes()),
            "template_generator_sha256": sha(
                (producer_dir / "generate-openai-chat-template-fixtures.py").read_bytes()
            ),
            "executable_sha256": sha(binary.read_bytes()), "loaded_library_sha256": LIBRARY_SHA256,
            "model_sha256": sha(model.read_bytes()), "template_config_sha256": sha(template_bytes),
            "template_bytes_sha256": sha(template_bytes),
        }

    def check(self, document, input_format="run", override=None, drop=(), via_process=False, **paths):
        """Write inputs, run the command in process, and return (exit code, result, output path)."""

        name = f"case-{len(list(self.dir.glob('case-*.input.json')))}"
        input_path, expected_path = self.dir / f"{name}.input.json", self.dir / f"{name}.expected.json"
        input_bytes = json.dumps(document).encode()
        input_path.write_bytes(input_bytes)
        expected = {**self.expected(input_bytes, **paths), **(override or {})}
        for field in drop:
            del expected[field]
        expected_path.write_text(json.dumps(expected))
        output = self.dir / f"{name}.result.json"
        argv = [
            "--input", str(input_path), "--input-format", input_format, "--expected", str(expected_path),
            "--llama-server", str(paths.get("binary") or self.binary),
            "--model", str(paths.get("model") or self.model),
            "--template", str(paths.get("template") or self.template), "--output", str(output),
        ]
        with self.path_environment():
            code = self.invoke(argv, via_process)
        result = json.loads(output.read_text()) if output.exists() else None
        return code, result, output

    def invoke(self, argv, via_process):
        if via_process:
            done = subprocess.run(
                [sys.executable, str(ROOT / "scripts/check-history-tokenization.py"), *argv],
                capture_output=True, text=True, check=False,
            )
            self.stdout = done.stdout
            return done.returncode
        with contextlib.redirect_stdout(io.StringIO()):
            return CHECK.main(argv)

    @contextlib.contextmanager
    def path_environment(self):
        """Expose a successful linkage command while the real resolver runs."""

        path = os.environ.get("PATH", "")
        with mock.patch.dict(os.environ, {"PATH": f"{self.tools}{os.pathsep}{path}"}):
            yield

    def launches(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


def forge_prompt_ids(record, delta=7):
    """Raise the first parent and branch prompt IDs and rewrite every retained digest."""

    forged = copy.deepcopy(record)
    oracle = forged["oracle"]
    exchanges = oracle["apply_template_tokenize"]
    for name in ("parent", "branch"):
        item = exchanges[name]["tokenize"]
        body = json.loads(bytes.fromhex(item["response_hex"]))
        body["tokens"][0] += delta
        data = compact(body)
        item["response_hex"], item["response_sha256"] = data.hex(), sha(data)
    digests = [exchanges[name]["tokenize"]["response_sha256"] for name in ("parent", "branch", "generated")]
    oracle["tokenize_response_sha256"] = sha(PRODUCER.canonical_json(digests))
    for field in ("parent_prompt_token_ids", "parent_evaluated_token_ids", "request_token_ids"):
        forged[field][0] += delta
    return forged


def with_record(run, record):
    changed = copy.deepcopy(run)
    changed["branches"][0]["history_reuse"]["tokenization"] = record
    return changed


class ReexecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="leone-reexec-test-")
        cls.addClassCleanup(directory.cleanup)
        cls.fx = Workspace(directory.name)
        environment = mock.patch.dict(os.environ, {"FAKE_LLAMA_LOG": str(cls.fx.log)})
        environment.start()
        cls.addClassCleanup(environment.stop)
        cls.run_record = cls.fx.collect()

    def setUp(self):
        self.fx.log.write_text("")

    def record(self):
        return self.run_record["branches"][0]["history_reuse"]["tokenization"]

    def test_verified_result_binds_input_and_identities(self):
        code, result, output = self.fx.check(self.run_record)
        self.assertEqual((code, result["status"], result["reason"]), (0, "verified", None))
        self.assertEqual(result["checked_branch_count"], 1)
        self.assertEqual(result["input"]["sha256"], sha(output.with_name("case-0.input.json").read_bytes()))
        self.assertEqual(result["identity"]["executable_sha256"], sha(self.fx.binary.read_bytes()))
        self.assertEqual(result["identity"]["model_sha256"], sha(self.fx.model.read_bytes()))
        self.assertEqual(result["identity"]["oracle"]["gguf_sha256"], result["identity"]["model_sha256"])
        self.assertEqual(result["identity"]["producer_sha256"], sha(
            (ROOT / "scripts/produce-history-tokenization.py").read_bytes()))
        self.assertEqual(result["offline_recompute"], "not_evaluated")
        self.assertIn("gpu_kv_contents", result["not_covered"])
        self.assertEqual(result["checked_branches"][0]["request_sha256"], self.record()["request_sha256"])

    def test_script_process_exit_codes_and_summary_line(self):
        code, _, _ = self.fx.check(self.run_record, via_process=True)
        self.assertEqual((code, self.fx.stdout), (0, "history reexecution: verified\n"))
        code, _, _ = self.fx.check(self.run_record, override={"input_sha256": "0" * 64}, via_process=True)
        self.assertEqual((code, self.fx.stdout), (1, "history reexecution: rejected input_digest_mismatch\n"))
        code, _, _ = self.fx.check(self.run_record, drop=("model_sha256",), via_process=True)
        self.assertEqual(code, 2)

    def test_result_has_no_paths_and_no_timing_fields(self):
        _, result, output = self.fx.check(self.run_record)
        text = output.read_text()
        for forbidden in (str(self.fx.dir), str(ROOT), str(Path.home()), "/tmp"):
            self.assertNotIn(forbidden, text)
        self.assertFalse([key for key in result if "ns" in key.split("_") or "ms" in key.split("_")])

    def test_fresh_server_received_cpu_flags_and_environment(self):
        code, result, _ = self.fx.check(self.run_record)
        self.assertEqual(code, 0)
        (launch,) = self.fx.launches()
        self.assertEqual(launch["argv"][launch["argv"].index("--n-gpu-layers") + 1], "0")
        self.assertEqual((launch["cuda"], launch["device"]), ("", "none"))
        recorded = result["fresh_server_launches"][0]
        self.assertEqual(recorded["env"], {"CUDA_VISIBLE_DEVICES": "", "LLAMA_ARG_DEVICE": "none"})
        self.assertIn("<model>", recorded["argv"])

    def test_branch_in_another_slot_replays_with_slot_copy(self):
        run = self.fx.collect(branch_slot=8)
        code, result, _ = self.fx.check(run)
        self.assertEqual((code, result["status"]), (0, "verified"))
        self.assertIsNotNone(run["slot_copy"])

    def test_receipt_selects_llama_runs_and_never_runs_receipt_paths(self):
        marker = self.fx.dir / "marker"
        poison = self.fx.dir / "poison"
        poison.write_text(f"#!/bin/sh\ntouch {marker}\n")
        poison.chmod(0o755)
        rows = [
            {"id": "leone", "kind": "leone", "executable_path": str(poison),
             "build_info_command": [str(poison), "--version"]},
            {"id": "llama_cpp", "kind": "llama_cpp", "executable_path": str(poison),
             "build_info_command": [str(poison), "--version"]},
        ]
        receipt = {"schema_version": CHECK.RECEIPT_SCHEMA, "engines": rows,
                   "runs": [{"engine": "leone"}, self.run_record]}
        code, result, _ = self.fx.check(receipt, input_format="receipt")
        self.assertEqual((code, result["status"]), (0, "verified"))
        self.assertEqual(result["input"]["runs_not_llama_cpp"], 1)
        self.assertFalse(marker.exists())

    def test_equal_length_forged_ids_pass_offline_recompute_and_fail_reexecution(self):
        forged = forge_prompt_ids(self.record())
        run = with_record(self.run_record, forged)
        self.assertEqual(self.offline_recompute(run, forged)["status"], "observed")
        code, result, _ = self.fx.check(run)
        self.assertEqual((code, result["status"], result["reason"]), (1, "rejected", "token_ids_differ"))
        self.assertEqual(result["failure"]["field"], "parent_prompt_token_ids")
        self.assertEqual(result["checked_branch_count"], 0)

    def offline_recompute(self, run, record):
        parent, branch = run["parent"], run["branches"][0]
        parent_wrapper = self.fx.wrapper(parent["_raw_events"], "parent-1")
        branch_wrapper = self.fx.wrapper(branch["_raw_events"], "branch-1", "parent-1")
        return PRODUCER.recompute_llama_history(
            self.fx.engine(), record["oracle"], bytes.fromhex(parent["_request_bytes_hex"]),
            parent_wrapper, bytes.fromhex(branch["_request_bytes_hex"]), branch_wrapper,
        )

    def test_forged_id_fields_with_intact_records_are_rejected(self):
        record = copy.deepcopy(self.record())
        record["request_token_ids"][-1] += 1
        code, result, _ = self.fx.check(with_record(self.run_record, record))
        self.assertEqual((code, result["reason"]), (1, "token_ids_differ"))
        self.assertEqual(result["failure"]["field"], "request_token_ids")

    def test_retained_tokenizer_request_bytes_must_equal_fresh_bytes(self):
        record = copy.deepcopy(self.record())
        item = record["oracle"]["apply_template_tokenize"]["parent"]["apply_template"]
        body = json.loads(bytes.fromhex(item["request_hex"]))
        body["add_generation_prompt"] = False
        data = compact(body)
        item["request_hex"], item["request_sha256"] = data.hex(), sha(data)
        code, result, _ = self.fx.check(with_record(self.run_record, record))
        self.assertEqual((code, result["reason"]), (1, "retained_record_differs_from_fresh_run"))
        self.assertEqual(result["failure"]["oracle_key"], "apply_template_tokenize")

    def test_binary_with_other_tokenization_is_rejected_even_when_pinned(self):
        other = write_server(self.fx.dir / "other" / "llama-server", offset=5)
        code, result, _ = self.fx.check(self.run_record, binary=other)
        self.assertEqual((code, result["status"]), (1, "rejected"))
        self.assertEqual(result["checked_branch_count"], 0)

    def test_fresh_generated_ids_must_equal_the_retained_verbose_ids(self):
        other = write_server(self.fx.dir / "shift" / "llama-server", answer_offset=5)
        code, result, _ = self.fx.check(self.run_record, binary=other)
        self.assertEqual(code, 1)
        self.assertEqual(result["reason"], "generated_token_ids_differ_from_fresh_tokenizer")
        self.assertEqual(result["failure"], {"run_index": 0, "branch_index": 0})

    def test_wrong_binary_model_and_template_are_rejected_before_any_launch(self):
        other = write_server(self.fx.dir / "wrong" / "llama-server", offset=1)
        model = self.fx.dir / "wrong" / "toy-model.gguf"
        model.write_bytes(b"other model bytes")
        template = self.fx.dir / "wrong" / "template.jinja"
        template.write_text("tpl-b")
        good = self.fx.expected(b"")
        cases = (
            ("llama_server", "executable_sha256", other),
            ("model", "model_sha256", model),
            ("template", "template_config_sha256", template),
        )
        for artifact, field, path in cases:
            paths = {"binary": other} if artifact == "llama_server" else {artifact: path}
            override = {name: good[name] for name in (field,)}
            code, result, _ = self.fx.check(self.run_record, override=override, **paths)
            self.assertEqual((code, result["reason"]), (1, "artifact_digest_mismatch"))
            self.assertEqual(result["failure"]["artifact"], artifact)
        self.assertEqual(self.fx.launches(), [])

    def test_pinned_producer_and_generator_digests_are_checked(self):
        for field, artifact in (("producer_sha256", "producer"), ("template_generator_sha256", "template_generator")):
            code, result, _ = self.fx.check(self.run_record, override={field: "0" * 64})
            self.assertEqual((code, result["failure"]["artifact"]), (1, artifact))

    def test_wrong_source_commit_pin_is_rejected(self):
        code, result, _ = self.fx.check(self.run_record, override={"source_commit": "0" * 40})
        self.assertEqual((code, result["reason"]), (1, "pinned_source_commit_mismatch"))

    def test_input_digest_pin_is_checked_before_any_launch(self):
        code, result, _ = self.fx.check(self.run_record, override={"input_sha256": "0" * 64})
        self.assertEqual((code, result["reason"]), (1, "input_digest_mismatch"))
        self.assertEqual(self.fx.launches(), [])
        self.assertIsNone(result["identity"])

    def test_artifact_changed_by_the_run_is_rejected_after_the_comparison(self):
        alt = self.fx.dir / "mutable" / "toy-model.gguf"
        alt.parent.mkdir()
        alt.write_bytes(b"toy model bytes")
        with mock.patch.dict(os.environ, {"FAKE_LLAMA_MUTATE": str(alt)}):
            code, result, _ = self.fx.check(self.run_record, model=alt)
        self.assertEqual((code, result["reason"]), (1, "artifact_mutated_during_run"))
        self.assertEqual(result["failure"]["artifact"], "model")

    def test_pin_file_gaps_are_incomplete(self):
        code, result, _ = self.fx.check(self.run_record, drop=("model_sha256",))
        self.assertEqual((code, result["reason"]), (2, "expected_field_missing"))
        self.assertEqual(result["failure"]["field"], "model_sha256")
        code, result, _ = self.fx.check(self.run_record, override={"extra_field": 1})
        self.assertEqual((code, result["reason"]), (2, "expected_field_unsupported"))
        code, result, _ = self.fx.check(self.run_record, override={"loaded_library_sha256": "ABC"})
        self.assertEqual((code, result["reason"]), (2, "expected_value_invalid"))
        code, result, _ = self.fx.check(self.run_record, override={"template_mode": "guess"})
        self.assertEqual((code, result["reason"]), (2, "expected_template_mode_unsupported"))

    def test_partial_and_unsupported_retained_records_are_typed(self):
        partial = copy.deepcopy(self.record())
        del partial["oracle"]["apply_template_tokenize"]["branch"]
        cases = {
            "retained_tokenization_missing": {"status": "unavailable", "reason": "x"},
            "proof_adapter_unsupported": {**self.record(), "oracle": {"proof_adapter": "leone.signed-receipt"}},
        }
        for reason, record in cases.items():
            code, result, _ = self.fx.check(with_record(self.run_record, record))
            self.assertEqual((code, result["reason"]), (2, reason))
        code, result, _ = self.fx.check(with_record(self.run_record, partial))
        self.assertEqual((code, result["reason"]), (1, "retained_record_differs_from_fresh_run"))
        self.assertEqual(result["checked_branch_count"], 0)

    def test_missing_wire_inputs_are_incomplete(self):
        for holder, field, reason in (
            ("parent", "_raw_events", "raw_events_missing"),
            ("parent", "_request_bytes_hex", "request_bytes_invalid"),
        ):
            run = copy.deepcopy(self.run_record)
            del run[holder][field]
            code, result, _ = self.fx.check(run)
            self.assertEqual((code, result["reason"]), (2, reason))
        run = copy.deepcopy(self.run_record)
        run["branches"] = []
        self.assertEqual(self.fx.check(run)[1]["reason"], "run_record_unsupported")

    def test_bounds_make_the_result_incomplete_instead_of_truncating(self):
        run = copy.deepcopy(self.run_record)
        run["branches"].append(copy.deepcopy(run["branches"][0]))
        with mock.patch.object(CHECK, "MAX_CHECKED_BRANCHES", 1):
            code, result, _ = self.fx.check(run)
        self.assertEqual((code, result["reason"]), (2, "too_many_branches"))
        with mock.patch.object(CHECK, "MAX_INPUT_BYTES", 10):
            code, result, _ = self.fx.check(self.run_record)
        self.assertEqual((code, result["reason"]), (2, "input_too_large"))
        self.assertEqual(self.fx.launches(), [])

    def test_unpinned_clock_in_the_template_is_incomplete(self):
        template = self.fx.dir / "clock" / "template.jinja"
        template.parent.mkdir()
        template.write_text("tpl-a {{ strftime_now('%d') }}")
        code, result, _ = self.fx.check(self.run_record, template=template)
        self.assertEqual((code, result["reason"]), (2, "template_date_not_pinned"))

    def test_existing_output_is_never_replaced(self):
        input_path = self.fx.dir / "keep.input.json"
        input_path.write_text("{}")
        output = self.fx.dir / "keep.result.json"
        output.write_text("original")
        argv = ["--input", str(input_path), "--input-format", "run", "--expected", str(input_path),
                "--llama-server", "x", "--model", "x", "--template", "x", "--output", str(output)]
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CHECK.main(argv), 2)
        self.assertEqual(output.read_text(), "original")

    def test_output_created_during_the_run_is_not_replaced(self):
        output = self.fx.dir / "race.result.json"
        real = CHECK.execute

        def racing(args):
            result = real(args)
            output.write_text("original")
            return result

        argv = self.fx.dir / "race.input.json"
        argv.write_text("{}")
        args = ["--input", str(argv), "--input-format", "run", "--expected", str(argv),
                "--llama-server", "x", "--model", "x", "--template", "x", "--output", str(output)]
        with mock.patch.object(CHECK, "execute", racing), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(CHECK.main(args), 2)
        self.assertEqual(output.read_text(), "original")
        self.assertEqual(list(self.fx.dir.glob(".reexecution-*")), [])

    def test_absent_values_are_null_not_zero(self):
        _, result, _ = self.fx.check(self.run_record, override={"input_sha256": "0" * 64})
        self.assertIsNone(result["identity"])
        self.assertEqual(result["checked_branches"], [])
        self.assertEqual(result["fresh_server_launches"], [])


class LaunchGuardTests(unittest.TestCase):
    def guard(self):
        return CHECK.LaunchGuard(mock.Mock(return_value="process"))

    def test_gpu_layers_must_be_zero(self):
        guard = self.guard()
        env = {"CUDA_VISIBLE_DEVICES": "", "LLAMA_ARG_DEVICE": "none"}
        with self.assertRaises(RuntimeError):
            guard(["s", "--port", "1", "--n-gpu-layers", "99"], env=env)
        self.assertEqual(guard.refusals, ["fresh_server_gpu_layers_not_zero"])
        guard.real.assert_not_called()

    def test_environment_must_disable_gpus(self):
        guard = self.guard()
        env = {"CUDA_VISIBLE_DEVICES": "0", "LLAMA_ARG_DEVICE": "none"}
        with self.assertRaises(RuntimeError):
            guard(["s", "--port", "1", "--n-gpu-layers", "0"], env=env)
        self.assertEqual(guard.refusals, ["fresh_server_gpu_environment_not_disabled"])

    def test_non_server_commands_pass_through(self):
        guard = self.guard()
        self.assertEqual(guard(["s", "--version"]), "process")
        self.assertEqual(guard.launches, [])

    def test_context_restores_popen_and_environment(self):
        real, before = subprocess.Popen, dict(os.environ)
        with self.assertRaises(ValueError):
            with CHECK.cpu_only_launches():
                self.assertEqual(os.environ["LLAMA_ARG_DEVICE"], "none")
                raise ValueError("stop")
        self.assertIs(subprocess.Popen, real)
        self.assertEqual(dict(os.environ), before)


class ComparatorTests(unittest.TestCase):
    """Drive the real comparator with a fresh record from a stubbed tokenizer call."""

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="leone-reexec-cmp-")
        cls.addClassCleanup(directory.cleanup)
        run = Workspace(directory.name).collect()
        cls.retained = run["branches"][0]["history_reuse"]["tokenization"]

    def compare(self, retained, fresh):
        return CHECK.compare_reexecution(
            retained, fresh, PRODUCER.harness_tokenization_evidence, PRODUCER.HARNESS_EVIDENCE_FIELDS
        )

    def reason(self, retained, fresh):
        with self.assertRaises(CHECK.Stop) as caught:
            self.compare(retained, fresh)
        return caught.exception.status, caught.exception.reason

    def test_identical_records_pass(self):
        self.assertEqual(self.compare(self.retained, copy.deepcopy(self.retained))["status"], "observed")

    def test_equal_length_forgery_is_rejected(self):
        self.assertEqual(
            self.reason(forge_prompt_ids(self.retained), copy.deepcopy(self.retained)),
            ("rejected", "token_ids_differ"),
        )

    def test_float_ids_do_not_equal_integer_ids(self):
        forged = copy.deepcopy(self.retained)
        forged["parent_prompt_token_ids"] = [float(value) for value in forged["parent_prompt_token_ids"]]
        self.assertEqual(self.reason(forged, copy.deepcopy(self.retained)), ("rejected", "token_ids_differ"))

    def test_every_changed_frozen_field_is_rejected(self):
        skipped = {"schema_version", "status", "scope", "parent_receipt_sha256", "branch_receipt_sha256"}
        for field in sorted(set(PRODUCER.HARNESS_EVIDENCE_FIELDS) - skipped):
            fresh = copy.deepcopy(self.retained)
            target = fresh["oracle"] if field == "oracle" else fresh
            target["engine" if field == "oracle" else field] = "changed"
            self.assertEqual(self.reason(self.retained, fresh)[0], "rejected", field)

    def test_verbose_ids_must_equal_fresh_generated_ids(self):
        fresh = copy.deepcopy(self.retained)
        fresh["parent_generated_token_ids"][0] += 1
        self.assertEqual(
            self.reason(self.retained, fresh), ("rejected", "generated_token_ids_differ_from_fresh_tokenizer")
        )

    def test_partial_retained_records_are_incomplete(self):
        for field in ("oracle", "request_token_ids"):
            retained = copy.deepcopy(self.retained)
            del retained[field]
            self.assertEqual(self.reason(retained, copy.deepcopy(self.retained)), ("incomplete", "retained_record_partial"))
        self.assertEqual(self.reason(None, copy.deepcopy(self.retained)), ("incomplete", "retained_record_partial"))

    def test_fresh_unavailable_reasons_are_typed(self):
        cases = {
            "tokenizer_request_failed": "incomplete",
            "tokenizer_response_too_large": "incomplete",
            "producer_error": "incomplete",
            "llama_parent_prompt_mismatch": "rejected",
            "pinned_artifact_hash_mismatch": "rejected",
            "Not A Code /tmp/x": "incomplete",
        }
        for reason, status in cases.items():
            fresh = {"status": "unavailable", "reason": reason}
            self.assertEqual(self.reason(self.retained, fresh)[0], status, reason)

    def test_missing_fresh_records_are_incomplete(self):
        fresh = copy.deepcopy(self.retained)
        del fresh["oracle"]["apply_template_tokenize"]["generated"]
        self.assertEqual(self.reason(self.retained, fresh), ("incomplete", "fresh_record_invalid"))


class PortableNameTests(unittest.TestCase):
    """A path spelling must not decide the result. A different model or alias must."""

    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory(prefix="leone-portable-test-")
        cls.addClassCleanup(directory.cleanup)
        cls.dir = Path(os.path.realpath(directory.name))
        environment = mock.patch.dict(os.environ, {"FAKE_LLAMA_LOG": str(cls.dir / "server.log")})
        environment.start()
        cls.addClassCleanup(environment.stop)
        cls.alias = Workspace(cls.dir / "alias")
        cls.alias_run = cls.alias.collect()
        cls.full = Workspace(cls.dir / "full")
        cls.full.wire_model = str(cls.full.model)
        cls.full_run = cls.full.collect()

    def test_a_symlinked_or_relative_model_path_equals_the_canonical_one(self):
        link = self.dir / "link"
        link.symlink_to(self.full.model.parent, target_is_directory=True)
        relative = Path(os.path.relpath(self.full.model))
        for spelling in (link / "toy-model.gguf", relative):
            code, result, _ = self.full.check(self.full_run, model=spelling)
            self.assertEqual((code, result["reason"]), (0, None), str(spelling))
        self.assertEqual(CHECK.canonical_path(str(link / "toy-model.gguf")), self.full.model)

    def test_a_full_wire_path_does_not_accept_another_directory_with_the_same_name(self):
        moved = self.dir / "moved" / "toy-model.gguf"
        moved.parent.mkdir()
        moved.write_bytes(self.full.model.read_bytes())
        code, result, _ = self.full.check(self.full_run, model=moved)
        self.assertEqual((code, result["reason"]), (1, "llama_response_model_mismatch"))

    def test_an_alias_wire_model_verifies_after_the_model_is_relocated(self):
        moved = self.dir / "elsewhere" / "toy-model.gguf"
        moved.parent.mkdir()
        moved.write_bytes(self.alias.model.read_bytes())
        code, result, _ = self.alias.check(self.alias_run, model=moved)
        self.assertEqual((code, result["reason"]), (0, None))

    def test_a_wrong_alias_or_a_wrong_model_is_rejected(self):
        wrong = copy.deepcopy(self.alias_run)
        wrong["parent"]["_raw_events"][0]["event"]["model"] = "other-model.gguf"
        code, result, _ = self.alias.check(wrong)
        self.assertEqual((code, result["reason"]), (1, "llama_response_model_mismatch"))
        other = self.dir / "other" / "toy-model.gguf"
        other.parent.mkdir()
        other.write_bytes(b"other model bytes")
        good = self.alias.expected(b"")
        code, result, _ = self.alias.check(
            self.alias_run, model=other, override={"model_sha256": good["model_sha256"]}
        )
        self.assertEqual((code, result["reason"]), (1, "artifact_digest_mismatch"))

    def test_a_missing_model_stays_a_typed_incomplete_result(self):
        self.alias.check(self.alias_run)
        argv = ["--input", str(self.alias.dir / "case-0.input.json"), "--input-format", "run",
                "--expected", str(self.alias.dir / "case-0.expected.json"),
                "--llama-server", str(self.alias.binary), "--model", str(self.dir / "absent.gguf"),
                "--template", str(self.alias.template), "--output", str(self.dir / "absent.result.json")]
        self.assertEqual(self.alias.invoke(argv, False), 2)
        result = json.loads((self.dir / "absent.result.json").read_text())
        self.assertEqual((result["reason"], result["failure"]), ("artifact_unreadable", {"artifact": "model"}))


@unittest.skipUnless(
    NATIVE_RUN is not None and NATIVE_RUN.is_file(),
    "set LEONE_HISTORY_NATIVE_RUN to a retained native llama.cpp run record",
)
class NativeAdapterTests(unittest.TestCase):
    """Check the input adapter against the retained native llama.cpp run. No model runs."""

    @classmethod
    def setUpClass(cls):
        cls.native = json.loads(NATIVE_RUN.read_text())
        cls.items, cls.skipped = CHECK.collect_inputs(cls.native, "run")

    def test_adapter_reproduces_the_recorded_raw_event_digests(self):
        for item in self.items:
            oracle = item["retained"]["oracle"]
            _, parent, _, branch = item["inputs"]
            self.assertEqual(sha(CHECK.canonical_json(parent)), oracle["raw_parent_events_sha256"])
            self.assertEqual(sha(CHECK.canonical_json(branch)), oracle["raw_branch_events_sha256"])

    def test_adapter_reproduces_the_recorded_request_hashes(self):
        for item in self.items:
            parent_request, _, request, _ = item["inputs"]
            self.assertEqual(sha(parent_request), item["retained"]["parent_request_sha256"])
            self.assertEqual(sha(request), item["retained"]["request_sha256"])

    def test_retained_native_record_is_complete_for_the_comparator(self):
        for item in self.items:
            retained = item["retained"]
            self.assertEqual(set(retained), set(PRODUCER.HARNESS_EVIDENCE_FIELDS))
            projected = CHECK.compare_reexecution(
                retained, copy.deepcopy(retained), PRODUCER.harness_tokenization_evidence,
                PRODUCER.HARNESS_EVIDENCE_FIELDS,
            )
            self.assertEqual(projected["request_sha256"], retained["request_sha256"])

    def test_native_branches_in_other_slots_carry_their_slot_copy(self):
        claims = [item["inputs"][3].get("slot_copy") for item in self.items]
        self.assertTrue(all(isinstance(claim, dict) and claim["restore"] for claim in claims))


if __name__ == "__main__":
    unittest.main()
