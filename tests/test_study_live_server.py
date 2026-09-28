#!/usr/bin/env python3
"""Tests for the real per-repetition study script and its outer script.

The scripts run in a temporary Git checkout. A fake executable answers
`--build-info` and `serve` with a local HTTP server on 127.0.0.1, and a fake
`nvidia-smi` reports one GPU. Real `curl`, `ss`, `jq`, and `git` run unchanged.
No model loads and no GPU is used. Every receipt and identity here is a fixture
and never enters the source tree.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "study_batched_service_tests", ROOT / "tests" / "test_study_batched_service.py"
)
assert SPEC and SPEC.loader
SHARED = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SHARED
SPEC.loader.exec_module(SHARED)

MODEL = "models/Qwen3-8B-Q4_K_M.gguf"
SHARED_PROMPT = (
    "Write a detailed explanation of why deterministic scheduling matters for "
    "language model inference. Do not use a list."
)
GPU = "NVIDIA GeForce RTX 4090, 580.00, 24564, 8.9\n"
BUILD_INFO = {
    "schema_version": "leone.build-info.v1", "version": "0.4.0",
    "source_tree_dirty": False, "provenance_unknown": False, "profile": "release",
    "target": "x86_64-unknown-linux-gnu", "features": "cuda",
}

FAKE_BINARY = r'''#!/usr/bin/env python3
import http.server
import itertools
import json
import os
import pathlib
import signal
import sys
import time

config = json.loads(pathlib.Path(os.environ["FAKE_BINARY_CONFIG"]).read_text())
args = sys.argv[1:]
with open(config["log"], "a") as log:
    log.write(json.dumps({"argv": args}) + "\n")
if args[0] == "--build-info":
    print(json.dumps(config["build_info"]))
    sys.exit(0)


def option(name):
    return args[args.index(name) + 1]


port = int(option("--bind").rsplit(":", 1)[1])
serial = int(option("--batch-size")) == 1
receipts = pathlib.Path(option("--receipt-dir"))
mode = "serial" if serial else "scheduled"
if config.get("mutate_binary") and not serial:
    with open(sys.argv[0], "a") as self_file:
        self_file.write("\n# changed while serving\n")
if config.get("collide_path") and serial:
    pathlib.Path(config["collide_path"]).write_text("winner\n")
signal.alarm(60)
counter = itertools.count(1)
if config.get("squat_scheduled") and not serial and os.fork() == 0:
    # A foreign listener that takes the port after the study's own check.
    import socket
    squatter = socket.socket()
    squatter.bind(("127.0.0.1", port))
    squatter.listen()
    pathlib.Path(config["squat_pid"]).write_text(str(os.getpid()))
    signal.alarm(30)
    time.sleep(30)
    os._exit(0)
if config.get("squat_scheduled") and not serial:
    time.sleep(0.5)


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *arguments):
        sys.stderr.write("fake server: " + format % arguments + "\n")

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["content-length"])))
        if request.get("stream"):
            time.sleep(3)
            return
        time.sleep(config["delay_" + mode])
        if serial and config.get("serial_status"):
            return self.reply(config["serial_status"], {"error": "fixture"})
        transcript = config["transcript_serial"] if serial else config["transcript"]
        body = {"usage": {"completion_tokens": request["max_tokens"]},
                "leone_receipt": {"claim": {"transcript_sha256": transcript}}}
        number = next(counter)
        (receipts / f"request-{number}.json").write_text(json.dumps(body))
        self.reply(200, body)
        if config.get("exit_after_request") == number and not serial:
            self.wfile.flush()
            os._exit(0)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


sys.stderr.write(f"fake server ready: {mode}\n")
Server(("127.0.0.1", port), Handler).serve_forever()
'''

FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

config = json.loads(pathlib.Path(os.environ["FAKE_BINARY_CONFIG"]).read_text())
sys.stdout.write(config["gpu"])
sys.exit(config.get("gpu_status", 0))
'''


def free_base():
    """Returns a port with its next five ports free on 127.0.0.1."""
    while True:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            base = probe.getsockname()[1]
        if base < 60000 and all(_free(base + offset) for offset in range(1, 6)):
            return base


def _free(port):
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def write_executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


class Fixture:
    """A Git checkout with the real study scripts, a fake binary, and a fake GPU tool."""

    def __init__(self, directory):
        self.base = pathlib.Path(directory)
        self.root = self.base / "checkout"
        self.config_path = self.base / "config.json"
        self.log = self.base / "invocations.log"
        self.binary = self.base / "bin" / "leone"
        self.port = free_base()
        self.build_checkout()
        self.config = {
            "log": str(self.log), "gpu": GPU, "delay_scheduled": 0.05, "delay_serial": 0.4,
            "transcript": "aa" * 32, "transcript_serial": "aa" * 32,
            "build_info": {**BUILD_INFO, "source_commit": self.head},
        }
        self.write_config()
        (self.base / "bin").mkdir(exist_ok=True)
        write_executable(self.binary, FAKE_BINARY)
        write_executable(self.base / "bin" / "nvidia-smi", FAKE_NVIDIA_SMI)

    def build_checkout(self):
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        for name in ("study-live-server.sh", "study-batched-service.sh", "source_inputs.py"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        (self.root / "models").mkdir()
        (self.root / "models" / "Qwen3-8B-Q4_K_M.gguf").write_text("model")
        (self.root / "plans").mkdir()
        (self.root / "plans" / "plan.json").write_text("{}")
        (self.root / "Cargo.toml").write_text("first\n")
        (self.root / "receipts").mkdir()
        digest = SHARED.sha256(self.root / MODEL)
        quality = {"receipt_id": "fixture-quality", "subject": {"model_artifact": {"sha256": digest}}}
        (self.root / "receipts" / "quality.json").write_text(json.dumps(quality))
        self.git("init", "-q", "-b", "main")
        self.commit("first")
        self.first = self.head
        (self.root / "Cargo.toml").write_text("second\n")
        self.commit("second")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def commit(self, message):
        self.git("add", ".")
        self.git("-c", "user.name=fixture", "-c", "user.email=fixture",
                 "commit", "-q", "-m", message)

    @property
    def head(self):
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.root, check=True,
                                capture_output=True, text=True)
        return result.stdout.strip()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def configure(self, **changes):
        self.config.update(changes)
        self.write_config()

    def build_info(self, **changes):
        self.configure(build_info={**self.config["build_info"], **changes})

    def run(self, script, *args, **env):
        environment = {
            **os.environ, "PATH": f"{self.base / 'bin'}:{os.environ['PATH']}",
            "FAKE_BINARY_CONFIG": str(self.config_path), "LEONE_BINARY": str(self.binary),
            "LEONE_STUDY_PORT": str(self.port), **env,
        }
        completed = subprocess.run(
            [str(self.root / "scripts" / script), *args],
            cwd=self.root, env=environment, text=True, capture_output=True, timeout=120,
        )
        return SHARED.Result(completed)

    def inputs(self, output, *counts):
        return [MODEL, "plans/plan.json", "receipts/quality.json", str(output), *counts]

    def inner(self, output, *counts, **env):
        return self.run("study-live-server.sh", *self.inputs(output, *counts), **env)

    def outer(self, output, *counts, **env):
        return self.run("study-batched-service.sh", *self.inputs(output, *counts), **env)

    def invocations(self):
        if not self.log.exists():
            return []
        return [json.loads(line)["argv"] for line in self.log.read_text().splitlines()]

    def serves(self):
        return [argv for argv in self.invocations() if argv[0] == "serve"]

    def beside(self, output, infix):
        output = pathlib.Path(output)
        return sorted(output.parent.glob(f"{output.name}.{infix}.*"))


class Base(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(temporary.name)
        self.output = self.fixture.root / "receipts" / "study.json"

    def load(self, path):
        return json.loads(pathlib.Path(path).read_text())


class InnerPublicationTests(Base):
    def test_passing_study_records_identity_and_leaves_no_raw_directory(self):
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 0, result.stderr)
        receipt = self.load(self.output)
        self.assertEqual(receipt["source_commit"], self.fixture.head)
        self.assertGreater(receipt["checks"]["aggregate_throughput_ratio"], 1)
        self.assertEqual(receipt["binary"]["sha256"], SHARED.sha256(self.fixture.binary))
        self.assertEqual(receipt["binary"]["role"], "leone-cli")
        self.assertEqual(receipt["binary"]["build_info"], self.fixture.config["build_info"])
        self.assertEqual(receipt["hardware"], {
            "gpu_name": "NVIDIA GeForce RTX 4090", "driver_version": "580.00",
            "memory_total_mib": 24564, "compute_capability": "8.9"})
        self.assertEqual(receipt["workload"]["prompt_sha256"], hashlib.sha256(
            SHARED_PROMPT.encode()).hexdigest())
        self.assertEqual(receipt["scheduled"]["batch_size"], 2)
        self.assertEqual(receipt["serial"]["batch_size"], 1)
        self.assertIn("does not retain signed responses", receipt["limits"][-2])
        self.assertEqual(receipt["limits"][-1],
                         "Batch sizes record command-line limits. Dispatch widths are unmeasured.")
        self.assertNotIn(str(self.fixture.base), self.output.read_text())
        self.assertEqual(self.fixture.beside(self.output, "raw"), [])

    def test_prompt_override_changes_the_recorded_prompt_digest(self):
        result = self.fixture.inner(self.output, "2", "8", LEONE_STUDY_PROMPT="Another prompt.")
        self.assertEqual(result.status, 0, result.stderr)
        digest = hashlib.sha256(b"Another prompt.").hexdigest()
        self.assertEqual(self.load(self.output)["workload"]["prompt_sha256"], digest)

    def test_recorded_batch_limits_are_the_values_passed_to_the_servers(self):
        result = self.fixture.inner(self.output, "3", "8")
        self.assertEqual(result.status, 0, result.stderr)
        limits = {}
        for argv in self.fixture.serves():
            limits[int(argv[argv.index("--bind") + 1].rsplit(":", 1)[1])] = int(
                argv[argv.index("--batch-size") + 1])
        receipt = self.load(self.output)
        self.assertEqual(sorted(limits.values()), [1, 3])
        self.assertEqual(sorted([receipt["scheduled"]["batch_size"], receipt["serial"]["batch_size"]]),
                         [1, 3])
        self.assertEqual(limits[self.fixture.port], receipt["scheduled"]["batch_size"])
        self.assertEqual(limits[self.fixture.port + 1], receipt["serial"]["batch_size"])

    def test_throughput_loss_exits_nonzero_and_keeps_receipt_and_raw_records(self):
        self.fixture.configure(delay_scheduled=0.4, delay_serial=0.05)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 1, result.stderr)
        self.assertLess(self.load(self.output)["checks"]["aggregate_throughput_ratio"], 1)
        (raw,) = self.fixture.beside(self.output, "raw")
        names = {path.name for path in raw.iterdir()}
        for expected in ("receipt.json", "scheduled-1.json", "serial-2.json", "scheduled-server.log",
                         "serial-server.log", "scheduled-receipts", "serial-receipts"):
            self.assertIn(expected, names)
        self.assertEqual(len(list((raw / "scheduled-receipts").iterdir())), 3)
        self.assertIn(str(raw), result.stderr)

    def test_collect_mode_ignores_only_the_throughput_condition(self):
        self.fixture.configure(delay_scheduled=0.4, delay_serial=0.05)
        raw = self.fixture.base / "raw"
        result = self.fixture.inner(
            self.output, "2", "8", LEONE_STUDY_GATE="collect", LEONE_STUDY_RAW_DIR=str(raw))
        self.assertEqual(result.status, 0, result.stderr)
        self.assertLess(self.load(self.output)["checks"]["aggregate_throughput_ratio"], 1)
        self.assertTrue((raw / "scheduled-server.log").is_file())

    def test_collect_mode_still_fails_on_transcript_mismatch(self):
        self.fixture.configure(transcript_serial="bb" * 32)
        result = self.fixture.inner(self.output, "2", "8", LEONE_STUDY_GATE="collect")
        self.assertEqual(result.status, 1, result.stderr)
        self.assertFalse(self.load(self.output)["checks"]["scheduled_matches_serial"])
        self.assertEqual(len(self.fixture.beside(self.output, "raw")), 1)

    def test_server_error_keeps_raw_records_and_publishes_nothing(self):
        self.fixture.configure(serial_status=500)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertNotEqual(result.status, 0)
        self.assertFalse(os.path.lexists(self.output))
        (raw,) = self.fixture.beside(self.output, "raw")
        self.assertTrue((raw / "scheduled-1.json").is_file())
        self.assertTrue((raw / "serial-server.log").is_file())
        self.assertIn(str(raw), result.stderr)

    def test_binary_changed_during_study_is_rejected(self):
        self.fixture.configure(mutate_binary=True)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 1, result.stderr)
        self.assertIn("changed during the study", result.stderr)
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual(len(self.fixture.beside(self.output, "raw")), 1)


class InnerCollisionTests(Base):
    def test_existing_output_is_rejected_before_the_executable_runs(self):
        cases = {
            "file": lambda: self.output.write_text("recorded\n"),
            "symlink": lambda: self.output.symlink_to("Cargo.toml"),
            "dangling symlink": lambda: self.output.symlink_to("missing"),
            "directory": lambda: self.output.mkdir(),
        }
        for name, create in cases.items():
            with self.subTest(name):
                create()
                result = self.fixture.inner(self.output)
                self.assertEqual(result.status, 2, result.stderr)
                self.assertEqual(self.fixture.invocations(), [])
                self.assertEqual(self.fixture.beside(self.output, "raw"), [])
                if self.output.is_dir() and not self.output.is_symlink():
                    self.output.rmdir()
                else:
                    self.output.unlink()

    def test_late_collision_keeps_the_winner_and_the_complete_receipt(self):
        self.fixture.configure(collide_path=str(self.output))
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 2, result.stderr)
        self.assertEqual(self.output.read_text(), "winner\n")
        (raw,) = self.fixture.beside(self.output, "raw")
        self.assertEqual(self.load(raw / "receipt.json")["workload"]["concurrent_clients"], 2)
        self.assertIn(str(raw), result.stderr)

    def test_raw_directory_must_be_new(self):
        raw = self.fixture.base / "raw"
        raw.mkdir()
        result = self.fixture.inner(self.output, LEONE_STUDY_RAW_DIR=str(raw))
        self.assertEqual(result.status, 2, result.stderr)
        self.assertEqual(self.fixture.serves(), [])
        self.assertEqual(list(raw.iterdir()), [])


class InnerListenerTests(Base):
    """A listener the study did not start is never measured as the study server."""

    def occupy(self, port):
        holder = socket.socket()
        self.addCleanup(holder.close)
        holder.bind(("127.0.0.1", port))
        holder.listen()

    def assert_nothing_measured(self, result):
        self.assertFalse(os.path.lexists(self.output))

    def test_occupied_scheduled_port_is_rejected_before_any_server_starts(self):
        self.occupy(self.fixture.port)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 2, result.stderr)
        self.assertIn(f"scheduled port {self.fixture.port} already has a listener", result.stderr)
        self.assertEqual(self.fixture.serves(), [])
        self.assert_nothing_measured(result)
        self.assertEqual(self.fixture.beside(self.output, "raw"), [])

    def test_occupied_serial_port_is_rejected_before_any_server_starts(self):
        self.occupy(self.fixture.port + 1)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 2, result.stderr)
        self.assertIn(f"serial port {self.fixture.port + 1} already has a listener", result.stderr)
        self.assertEqual(self.fixture.serves(), [])
        self.assert_nothing_measured(result)

    def test_listener_taking_the_port_after_the_check_is_not_measured(self):
        pid_file = self.fixture.base / "squatter.pid"
        self.fixture.configure(squat_scheduled=True, squat_pid=str(pid_file))
        self.addCleanup(self.kill_squatter, pid_file)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 1, result.stderr)
        self.assertIn("exited before it listened", result.stderr)
        self.assertIn("Address already in use", result.stderr)
        self.assert_nothing_measured(result)
        (raw,) = self.fixture.beside(self.output, "raw")
        self.assertIn("Address already in use", (raw / "scheduled-server.log").read_text())
        self.assertEqual(list((raw / "scheduled-receipts").iterdir()), [])
        self.assertEqual(len(self.fixture.serves()), 1)
        self.assertIn(str(raw), result.stderr)

    @staticmethod
    def kill_squatter(pid_file):
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_server_that_exits_before_the_measurements_end_is_rejected(self):
        self.fixture.configure(exit_after_request=3)
        result = self.fixture.inner(self.output, "2", "8")
        self.assertEqual(result.status, 1, result.stderr)
        self.assertIn("exited before its measurements ended", result.stderr)
        self.assert_nothing_measured(result)
        (raw,) = self.fixture.beside(self.output, "raw")
        self.assertEqual(len(list((raw / "scheduled-receipts").iterdir())), 3)
        self.assertEqual(len(self.fixture.serves()), 1)


class InnerRejectionTests(Base):
    def assert_rejected(self, result, message):
        self.assertEqual(result.status, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertEqual(self.fixture.serves(), [])
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual(self.fixture.beside(self.output, "raw"), [])

    def test_stale_executable_is_rejected_before_any_server_starts(self):
        self.fixture.build_info(source_commit=self.fixture.first)
        result = self.fixture.inner(self.output)
        self.assert_rejected(result, "does not match the current source inputs")

    def test_executable_without_release_cuda_provenance_is_rejected(self):
        cases = {
            "dirty": {"source_tree_dirty": True}, "unknown": {"provenance_unknown": True},
            "no unknown field": {"provenance_unknown": None}, "debug": {"profile": "debug"},
            "cpu only": {"features": "avx2"}, "metal": {"target": "aarch64-apple-darwin"},
            "short commit": {"source_commit": "abc123"}, "schema": {"schema_version": "x"},
        }
        for name, change in cases.items():
            with self.subTest(name):
                self.fixture.configure(build_info={
                    **BUILD_INFO, "source_commit": self.fixture.head, **change})
                result = self.fixture.inner(self.output)
                self.assert_rejected(result, "clean release CUDA build")

    def test_missing_executable_and_unsafe_manifest_are_rejected(self):
        result = self.fixture.inner(self.output, LEONE_BINARY=str(self.fixture.base / "absent"))
        self.assert_rejected(result, "executable is missing")
        result = self.fixture.inner(self.output, LEONE_STUDY_SOURCE_MANIFEST="../outside.json")
        self.assert_rejected(result, "does not match the current source inputs")

    def test_gpu_identity_must_name_exactly_one_device(self):
        for name, change in {
            "two GPUs": {"gpu": GPU + GPU}, "short record": {"gpu": "NVIDIA, 580.00\n"},
            "tool failure": {"gpu": "", "gpu_status": 9},
        }.items():
            with self.subTest(name):
                self.fixture.configure(**change)
                self.assert_rejected(self.fixture.inner(self.output), "GPU")

    def test_arguments_are_checked_before_the_executable_runs(self):
        cases = [
            (["/abs/model.gguf", "plans/plan.json", "receipts/quality.json", str(self.output)], {},
             "repository-relative"),
            (self.fixture.inputs(self.output, "0"), {}, "CLIENTS"),
            (self.fixture.inputs(self.output, "2", "4294967296"), {}, "MAX_TOKENS"),
            (self.fixture.inputs(self.output), {"LEONE_STUDY_PORT": "65535"}, "LEONE_STUDY_PORT"),
            (self.fixture.inputs(self.output), {"LEONE_STUDY_GATE": "skip"}, "LEONE_STUDY_GATE"),
            (self.fixture.inputs(""), {}, "new file path"),
        ]
        for arguments, env, message in cases:
            with self.subTest(message=message):
                result = self.fixture.run("study-live-server.sh", *arguments, **env)
                self.assertEqual(result.status, 2, result.stderr)
                self.assertIn(message, result.stderr)
                self.assertEqual(self.fixture.invocations(), [])

    def test_dirty_tracked_tree_is_rejected(self):
        (self.fixture.root / "Cargo.toml").write_text("changed\n")
        result = self.fixture.inner(self.output)
        self.assertEqual(result.status, 2)
        self.assertIn("clean candidate commit", result.stderr)
        self.assertEqual(self.fixture.invocations(), [])

    def test_quality_record_for_another_model_leaves_no_directory(self):
        (self.fixture.root / MODEL).write_text("other")
        self.fixture.commit("other model")
        self.fixture.build_info(source_commit=self.fixture.head)
        result = self.fixture.inner(self.output)
        self.assert_rejected(result, "does not match the served model")


class OuterWithRealInnerTests(Base):
    def outer(self, *counts, **env):
        return self.fixture.outer(self.output, "2", "8", *counts, LEONE_STUDY_REPETITIONS="2", **env)

    def test_losing_repetitions_all_run_and_the_receipt_is_kept(self):
        self.fixture.configure(delay_scheduled=0.4, delay_serial=0.05)
        result = self.outer()
        self.assertEqual(result.status, 1, result.stderr)
        receipt = self.load(self.output)
        self.assertEqual(len(receipt["samples"]), 2)
        self.assertFalse(receipt["checks"]["every_throughput_sample_wins"])
        self.assertTrue(receipt["checks"]["every_transcript_matches"])
        self.assertEqual(len(self.fixture.serves()), 4)
        (work,) = self.fixture.beside(self.output, "runs")
        for name in ("1.json", "2.json", "receipt.json", "1.raw", "2.raw"):
            self.assertTrue((work / name).exists(), name)
        self.assertTrue((work / "2.raw" / "serial-server.log").is_file())
        self.assertIn("performance gate", result.stderr)

    def test_passing_study_publishes_identity_and_removes_all_records(self):
        result = self.outer()
        self.assertEqual(result.status, 0, result.stderr)
        receipt = self.load(self.output)
        self.assertEqual(receipt["binary"]["sha256"], SHARED.sha256(self.fixture.binary))
        self.assertEqual(receipt["hardware"]["gpu_name"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(self.fixture.beside(self.output, "runs"), [])
        self.assertTrue(all(receipt["checks"].values()))

    def test_transcript_mismatch_stops_the_study_and_keeps_raw_records(self):
        self.fixture.configure(transcript_serial="bb" * 32)
        result = self.outer()
        self.assertEqual(result.status, 1, result.stderr)
        self.assertFalse(os.path.lexists(self.output))
        self.assertEqual(len(self.fixture.serves()), 2)
        (work,) = self.fixture.beside(self.output, "runs")
        self.assertTrue((work / "1.raw" / "serial-server.log").is_file())
        self.assertIn("repetition 1 of 2 failed", result.stderr)

    def test_stale_executable_stops_before_any_server_and_leaves_no_directory(self):
        self.fixture.build_info(source_commit=self.fixture.first)
        result = self.outer()
        self.assertEqual(result.status, 2, result.stderr)
        self.assertEqual(self.fixture.serves(), [])
        self.assertEqual(self.fixture.beside(self.output, "runs"), [])

    def test_late_collision_keeps_the_winner_and_every_measured_record(self):
        self.fixture.configure(collide_path=str(self.output))
        result = self.outer()
        self.assertEqual(result.status, 2, result.stderr)
        self.assertEqual(self.output.read_text(), "winner\n")
        (work,) = self.fixture.beside(self.output, "runs")
        self.assertEqual(self.load(work / "receipt.json")["workload"]["repetitions"], 2)
        self.assertTrue((work / "1.raw" / "scheduled-server.log").is_file())


if __name__ == "__main__":
    unittest.main()
