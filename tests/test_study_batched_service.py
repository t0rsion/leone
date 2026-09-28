#!/usr/bin/env python3
"""Tests for receipt publication in the batched-service study and its renderer.

The tests run the real outer and renderer scripts in a temporary Git checkout.
A fake `scripts/study-live-server.sh` replaces the per-repetition study, so
these tests cover only the outer script. `test_study_live_server.py` runs the
real inner script. Receipts written here are fixtures and never enter the
source tree.
"""

import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_COMMIT = "fixture-commit"
MODEL = "models/Qwen3-8B-Q4_K_M.gguf"

FAKE_LIVE_SERVER = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import sys

config = json.loads(pathlib.Path(os.environ["FAKE_STUDY_CONFIG"]).read_text())
model, plan, quality, output, clients, max_tokens = sys.argv[1:7]
with open(config["log"], "a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "port": os.environ["LEONE_STUDY_PORT"]}) + "\n")
repetition = int(pathlib.Path(output).stem)
if repetition == config.get("fail_at"):
    sys.exit(7)
if repetition == config.get("collide_at"):
    collide = pathlib.Path(config["collide_path"])
    if config["collide_kind"] == "file":
        collide.write_text("winner\n")
    elif config["collide_kind"] == "directory":
        collide.mkdir()
    else:
        collide.symlink_to("missing-target")
digest = hashlib.sha256(pathlib.Path(quality).read_bytes()).hexdigest()
ratio = config["throughput_ratio"] + repetition / 100
mode = lambda name, tok_s, p95: {
    "wall_ms": 100.0 * repetition,
    "aggregate_completion_tok_s": tok_s,
    "total_ms": {"p95": p95},
    "requests": [{"transcript_sha256": "aa" * 32}],
}
run = {
    "schema_version": "leone.server-study.v2",
    "created_utc": "2026-01-01T00:00:00Z",
    "source_commit": config["commit"],
    "model": {"path": model, "sha256": "bb" * 32},
    "plan": {"path": plan, "sha256": "cc" * 32},
    "quality": {"path": quality, "sha256": digest, "receipt_id": "fixture-quality"},
    "workload": {"concurrent_clients": int(clients), "max_tokens": int(max_tokens),
                 "temperature": 0, "seed": 0},
    "scheduled": mode("scheduled", 100.0 * ratio, config["scheduled_p95"]),
    "serial": mode("serial", 100.0, 200.0),
    "checks": {
        "scheduled_transcripts_agree": True,
        "scheduled_matches_serial": True,
        "aggregate_throughput_ratio": ratio,
        "disconnect": {"client_disconnected": True, "recovery_http_code": 200},
    },
    "binary": {"role": "leone-cli", "sha256": "dd" * 32, "build_info": {}},
    "hardware": {"gpu_name": config.get("gpu_name", "NVIDIA GeForce RTX 4090")},
}
pathlib.Path(output).write_text(json.dumps(run))
'''

QUALITY = {
    "receipt_id": "fixture-quality",
    "metrics": {"kld": {"mean": 0.25}, "top1_agreement": 0.5},
}


class Result:
    def __init__(self, completed):
        self.status = completed.returncode
        self.stdout = completed.stdout
        self.stderr = completed.stderr


class Checkout:
    """A temporary Git checkout with the real scripts and a fake study server."""

    def __init__(self, directory, **overrides):
        self.root = pathlib.Path(directory) / "checkout"
        self.config_path = pathlib.Path(directory) / "config.json"
        self.log = pathlib.Path(directory) / "runs.log"
        self.config = {
            "log": str(self.log), "commit": FIXTURE_COMMIT,
            "throughput_ratio": 1.5, "scheduled_p95": 100.0,
        }
        self.config.update(overrides)
        self.write_config()
        self.build()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def build(self):
        scripts = self.root / "scripts"
        scripts.mkdir(parents=True)
        for name in ("study-batched-service.sh", "render-release-evidence.sh"):
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
        fake = scripts / "study-live-server.sh"
        fake.write_text(FAKE_LIVE_SERVER)
        fake.chmod(0o755)
        (self.root / "receipts").mkdir()
        (self.root / "receipts" / "quality.json").write_text(json.dumps(QUALITY))
        (self.root / "docs").mkdir()
        (self.root / "docs" / "sentinel.md").write_text("sentinel\n")
        (self.root / "models").mkdir()
        for name in (MODEL, "plan.json"):
            (self.root / name).write_text(name)
        self.git("init", "-q", "-b", "main")
        self.git("add", ".")
        self.git("-c", "user.name=fixture", "-c", "user.email=fixture",
                 "commit", "-q", "-m", "fixture")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True)

    def run_script(self, name, *args, **env):
        environment = {**os.environ, "FAKE_STUDY_CONFIG": str(self.config_path), **env}
        completed = subprocess.run(
            [str(self.root / "scripts" / name), *args],
            cwd=self.root, env=environment, text=True, capture_output=True, timeout=120,
        )
        return Result(completed)

    def study(self, output, *counts, **env):
        return self.run_script(
            "study-batched-service.sh", MODEL, "plan.json", "receipts/quality.json",
            str(output), *counts, **env,
        )

    def render(self, *args):
        return self.run_script("render-release-evidence.sh", *args)

    def invocations(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def retained(self, output):
        return sorted(pathlib.Path(output).parent.glob(pathlib.Path(output).name + ".runs.*"))


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = self.temporary.name

    def checkout(self, **overrides):
        checkout = Checkout(self.directory, **overrides)
        self.output = checkout.root / "receipts" / "study-a.json"
        return checkout


class EarlyRejectionTests(Base):
    def assert_rejected_without_runs(self, checkout, result):
        self.assertEqual(result.status, 2, result.stderr)
        self.assertEqual(checkout.invocations(), [])
        self.assertEqual(checkout.retained(self.output), [])

    def test_existing_output_is_rejected_before_any_run(self):
        checkout = self.checkout()
        (checkout.root / "target-file").write_text("recorded\n")
        cases = {
            "file": lambda: self.output.write_text("recorded\n"),
            "symlink": lambda: self.output.symlink_to("target-file"),
            "dangling symlink": lambda: self.output.symlink_to("missing-target"),
            "directory": lambda: self.output.mkdir(),
        }
        for name, create in cases.items():
            with self.subTest(name):
                create()
                result = checkout.study(self.output)
                self.assert_rejected_without_runs(checkout, result)
                self.assertTrue(os.path.lexists(self.output))
                if self.output.is_dir() and not self.output.is_symlink():
                    self.output.rmdir()
                else:
                    self.output.unlink()
        self.assertEqual((checkout.root / "target-file").read_text(), "recorded\n")

    def test_unusable_output_names_are_rejected_before_any_run(self):
        checkout = self.checkout()
        for name in ("", f"{self.output}/"):
            with self.subTest(name=name):
                self.assert_rejected_without_runs(checkout, checkout.study(name))

    def test_counts_are_checked_before_any_run(self):
        checkout = self.checkout()
        cases = [
            (("0",), {}), (("abc",), {}), (("4", "0"), {}), (("4294967296",), {}),
            (("4", "007"), {}), ((), {"LEONE_STUDY_REPETITIONS": "0"}),
            ((), {"LEONE_STUDY_REPETITIONS": "x"}), ((), {"LEONE_STUDY_PORT": "65535"}),
            ((), {"LEONE_STUDY_REPETITIONS": "99999999999"}),
            ((), {"LEONE_STUDY_PORT": "65000", "LEONE_STUDY_REPETITIONS": "400"}),
        ]
        for counts, env in cases:
            with self.subTest(counts=counts, env=env):
                result = checkout.study(self.output, *counts, **env)
                self.assert_rejected_without_runs(checkout, result)
                self.assertFalse(os.path.lexists(self.output))

    def test_absolute_inputs_and_dirty_tree_keep_their_status(self):
        checkout = self.checkout()
        result = checkout.run_script(
            "study-batched-service.sh", "/abs/model.gguf", "plan.json", "receipts/quality.json",
            str(self.output),
        )
        self.assert_rejected_without_runs(checkout, result)
        self.assertIn("repository-relative", result.stderr)
        (checkout.root / "plan.json").write_text("changed")
        result = checkout.study(self.output)
        self.assert_rejected_without_runs(checkout, result)
        self.assertIn("clean candidate commit", result.stderr)


class PublicationTests(Base):
    def test_passing_study_publishes_one_valid_receipt(self):
        checkout = self.checkout()
        result = checkout.study(self.output, "3", "16", LEONE_STUDY_REPETITIONS="12", LEONE_STUDY_PORT="20000")
        self.assertEqual(result.status, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.output))
        receipt = json.loads(self.output.read_text())
        self.assertEqual(receipt["schema_version"], "leone.batched-service-study.v1")
        self.assertEqual(receipt["workload"]["repetitions"], 12)
        self.assertEqual(receipt["workload"]["concurrent_clients"], 3)
        self.assertEqual(receipt["workload"]["max_tokens"], 16)
        self.assertEqual([s["scheduled"]["wall_ms"] for s in receipt["samples"]],
                         [100.0 * n for n in range(1, 13)])
        self.assertTrue(all(receipt["checks"].values()))
        self.assertEqual([run["scheduled"]["wall_ms"] for run in receipt["runs"]],
                         [100.0 * n for n in range(1, 13)])
        self.assertIn("It does not retain signed responses", receipt["limits"][-2])
        self.assertEqual(receipt["limits"][-1],
                         "Batch sizes record command-line limits. Dispatch widths are unmeasured.")
        self.assertEqual(checkout.retained(self.output), [])
        self.assertEqual([run["port"] for run in checkout.invocations()],
                         [str(20000 + 2 * n) for n in range(1, 13)])

    def test_default_counts_run_five_repetitions(self):
        checkout = self.checkout()
        self.assertEqual(checkout.study(self.output).status, 0)
        runs = checkout.invocations()
        self.assertEqual(len(runs), 5)
        self.assertEqual(runs[0]["argv"][4:], ["4", "64"])

    def test_run_failure_keeps_completed_runs_and_publishes_nothing(self):
        checkout = self.checkout(fail_at=3)
        result = checkout.study(self.output)
        self.assertEqual(result.status, 7)
        self.assertFalse(os.path.lexists(self.output))
        (kept,) = checkout.retained(self.output)
        self.assertEqual(sorted(p.name for p in kept.iterdir()), ["1.json", "2.json"])
        self.assertIn("repetition 3 of 5 failed with status 7", result.stderr)
        self.assertIn(str(kept), result.stderr)

    def test_failed_first_run_leaves_no_directory(self):
        checkout = self.checkout(fail_at=1)
        self.assertEqual(checkout.study(self.output).status, 7)
        self.assertEqual(checkout.retained(self.output), [])

    def test_losing_receipt_is_kept_and_gate_fails(self):
        checkout = self.checkout(throughput_ratio=-0.5)
        result = checkout.study(self.output)
        self.assertEqual(result.status, 1)
        receipt = json.loads(self.output.read_text())
        self.assertEqual(len(receipt["samples"]), 5)
        self.assertFalse(receipt["checks"]["every_throughput_sample_wins"])
        self.assertIn("performance gate", result.stderr)
        (kept,) = checkout.retained(self.output)
        self.assertEqual(sha256(kept / "receipt.json"), sha256(self.output))
        self.assertEqual(len(list(kept.glob("[0-9]*.json"))), 5)

    def test_slower_latency_fails_the_gate_with_a_kept_receipt(self):
        checkout = self.checkout(scheduled_p95=300.0)
        self.assertEqual(checkout.study(self.output).status, 1)
        receipt = json.loads(self.output.read_text())
        self.assertFalse(receipt["checks"]["every_p95_completion_sample_wins"])


class LateCollisionTests(Base):
    def test_winner_is_preserved_and_measurements_are_retained(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                self.late_collision(kind)

    def late_collision(self, kind):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Checkout(directory)
            output = checkout.root / "receipts" / "study-a.json"
            checkout.config.update(collide_at=5, collide_path=str(output), collide_kind=kind)
            checkout.write_config()
            result = checkout.study(output)
            self.assertEqual(result.status, 2, result.stderr)
            self.assert_winner_intact(kind, output)
            (kept,) = checkout.retained(output)
            self.assertEqual(len(list(kept.glob("[0-9]*.json"))), 5)
            receipt = json.loads((kept / "receipt.json").read_text())
            self.assertEqual(receipt["workload"]["repetitions"], 5)
            self.assertIn(str(kept), result.stderr)

    def assert_winner_intact(self, kind, output):
        if kind == "file":
            self.assertEqual(output.read_text(), "winner\n")
        elif kind == "directory":
            self.assertEqual(list(output.iterdir()), [])
        else:
            self.assertTrue(output.is_symlink())
            self.assertEqual(os.readlink(output), "missing-target")


class RendererTests(Base):
    def published(self, checkout, *counts, **overrides):
        checkout.config.update(overrides)
        checkout.write_config()
        self.study = checkout.root / "receipts" / "2026-01-01T00-00-00Z-batched-service-study.json"
        checkout.study(self.study, *counts)
        return self.study

    def render_to(self, checkout, *args):
        target = checkout.root / "docs" / "evidence.md"
        target.write_text("previous\n")
        return target, checkout.render(str(target.relative_to(checkout.root)), *args)

    def test_chosen_study_is_linked_and_numbers_come_from_json(self):
        checkout = self.checkout()
        study = self.published(checkout)
        relative = str(study.relative_to(checkout.root))
        target, result = self.render_to(checkout, relative)
        self.assertEqual(result.status, 0, result.stderr)
        text = target.read_text()
        record = json.loads(study.read_text())
        self.assertIn(f"[study receipt](../{relative})", text)
        for key in ("minimum", "median", "maximum"):
            self.assertIn(str(record["summary"]["aggregate_throughput_ratio"][key]), text)
            self.assertIn(str(record["summary"]["p95_completion_latency_ratio"][key]), text)
        self.assertIn("`fixture-quality`", text)
        self.assertIn("mean KLD is 0.25", text)
        self.assertIn(record["model"]["sha256"], text)

    def test_default_study_renders_like_the_same_study_chosen(self):
        checkout = self.checkout()
        study = self.published(checkout)
        default = checkout.root / "receipts" / "batched-service-study.json"
        shutil.copy2(study, default)
        first, result = self.render_to(checkout)
        self.assertEqual(result.status, 0, result.stderr)
        self.assertIn("(../receipts/batched-service-study.json)", first.read_text())
        default_bytes = first.read_bytes()
        _, result = self.render_to(checkout, "receipts/batched-service-study.json")
        self.assertEqual(first.read_bytes(), default_bytes)

    def test_checked_in_document_matches_the_default_render(self):
        rendered = pathlib.Path(self.directory) / "rendered.md"
        completed = subprocess.run(
            [str(ROOT / "scripts" / "render-release-evidence.sh"), str(rendered)],
            cwd=ROOT, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(rendered.read_bytes(), (ROOT / "docs" / "release-evidence.md").read_bytes())

    def test_unusable_input_leaves_the_output_untouched(self):
        checkout = self.checkout()
        study = self.published(checkout)
        losing = self.published_losing(checkout)
        relative = str(study.relative_to(checkout.root))
        (checkout.root / "receipts" / "link.json").symlink_to(study.name)
        for name, args in {
            "absolute": [str(study)], "dot segment": ["receipts/../" + relative],
            "space": ["receipts/a b.json"], "missing": ["receipts/none.json"],
            "symlink": ["receipts/link.json"], "losing study": [losing],
            "extra argument": [relative, "x"],
        }.items():
            with self.subTest(name):
                target, result = self.render_to(checkout, *args)
                self.assertEqual(result.status, 2, result.stderr)
                self.assertEqual(target.read_text(), "previous\n")

    def test_output_cannot_replace_a_study_or_quality_receipt(self):
        checkout = self.checkout()
        study = self.published(checkout)
        relative = str(study.relative_to(checkout.root))
        before = sha256(study)
        for output in (relative, "receipts/quality.json"):
            with self.subTest(output=output):
                result = checkout.render(output, relative)
                self.assertEqual(result.status, 2, result.stderr)
                self.assertIn("would replace an input", result.stderr)
        self.assertEqual(sha256(study), before)

    def test_study_must_bind_the_model_client_count_and_gpu_of_the_text(self):
        cases = {
            "three clients": ((("3",)), {}),
            "other GPU": ((), {"gpu_name": "NVIDIA A100"}),
        }
        for name, (counts, overrides) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                checkout = Checkout(directory)
                study = self.published(checkout, *counts, **overrides)
                target, result = self.render_to(checkout, str(study.relative_to(checkout.root)))
                self.assertEqual(result.status, 2, result.stderr)
                self.assertEqual(target.read_text(), "previous\n")

    def test_hardware_may_be_absent_only_from_the_historical_default_receipt(self):
        checkout = self.checkout()
        study = self.published(checkout)
        record = json.loads(study.read_text())
        del record["hardware"]
        study.write_text(json.dumps(record))
        target, result = self.render_to(checkout, str(study.relative_to(checkout.root)))
        self.assertEqual(result.status, 2, result.stderr)
        default = checkout.root / "receipts" / "batched-service-study.json"
        default.write_text(json.dumps(record))
        target, result = self.render_to(checkout)
        self.assertEqual(result.status, 0, result.stderr)

    def published_losing(self, checkout):
        with tempfile.TemporaryDirectory() as directory:
            losing = Checkout(directory, throughput_ratio=-0.5)
            output = losing.root / "receipts" / "losing.json"
            self.assertEqual(losing.study(output).status, 1)
            shutil.copy2(output, checkout.root / "receipts" / "losing.json")
        return "receipts/losing.json"

    def test_quality_digest_mismatch_is_rejected(self):
        checkout = self.checkout()
        study = self.published(checkout)
        (checkout.root / "receipts" / "quality.json").write_text(json.dumps({**QUALITY, "extra": 1}))
        target, result = self.render_to(checkout, str(study.relative_to(checkout.root)))
        self.assertEqual(result.status, 2)
        self.assertIn("quality digest", result.stderr)
        self.assertEqual(target.read_text(), "previous\n")


if __name__ == "__main__":
    unittest.main()
