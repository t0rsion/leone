"""Check staged llama.cpp quality provenance before numerical comparison."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import struct
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_quality_stage", ROOT / "scripts/validate-quality-stage.py"
)
VALIDATE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(VALIDATE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QualityStageTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.write_fixture()

    def write_fixture(self):
        pin = (ROOT / "external/PINNED").read_text().strip()
        adapter = digest(ROOT / "research/oracle/llama_logits.cpp")
        adapter_bytes = (ROOT / "research/oracle/llama_logits.cpp").stat().st_size
        adapter_record = {"path": "research/oracle/llama_logits.cpp", "sha256": adapter, "artifact": {"path": "llama_logits.cpp", "sha256": adapter, "bytes": adapter_bytes}}
        tokens = self.root / "tokens.u32le"
        token_values = [0] * 4097
        token_values[4096] = 1
        tokens.write_bytes(struct.pack("<4097I", *token_values))
        logits = b"".join(struct.pack("<2f", 0.0, 1.0) if index == 4095 else struct.pack("<2f", 1.0, 0.0) for index in range(4096))
        (self.root / "oracle.f32").write_bytes(logits)
        (self.root / "metal.f32").write_bytes(logits)
        (self.root / "leone-metal.f32").write_bytes(logits)
        oracle_model = self.root / "oracle-model.gguf"
        subject_model = self.root / "subject-model.gguf"
        corpus = self.root / "corpus.txt"
        oracle_model.write_bytes(b"oracle-model")
        subject_model.write_bytes(b"subject-model")
        corpus.write_text("corpus\n4")
        oracle_run = {
            "schema_version": "leone.llama-oracle.v2",
            "source_commit": "a" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": adapter_record,
            "executable": {"path": "llama-logits-oracle", "sha256": "c" * 64, "linked_libraries": []},
            "model": {"path": "oracle-model.gguf", "sha256": digest(oracle_model), "storage_type": "BF16", "vocab_size": 2, "architecture": "test", "tokenizer": "gpt2"},
            "corpus": {"name": corpus.name, "sha256": digest(corpus)},
            "input": {"tokens": {"path": "tokens.u32le", "sha256": digest(tokens), "bytes": tokens.stat().st_size, "count": 4097, "encoding": "u32le"}, "window_tokens": 4096, "stride_tokens": 4095, "rows": 4096, "vocab_size": 2},
            "execution": {
                "device": "cpu", "backend_registry": "CPU", "device_type": "cpu",
                "device_name": "CPU", "device_description": "CPU", "device_id": None,
            },
            "logits": {"path": "oracle.f32", "sha256": digest(self.root / "oracle.f32"), "bytes": len(logits), "encoding": "row-major-f32-le"},
        }
        (self.root / "oracle.json").write_text(json.dumps(oracle_run))
        exported = {
            "schema_version": "leone.quality-stage.v1",
            "stage": "oracle-export",
            "source_commit": "a" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": adapter_record,
            "model": {
                "oracle": {"name": oracle_model.name, "sha256": digest(oracle_model), "storage_type": "BF16", "vocab_size": 2, "architecture": "test", "tokenizer": "gpt2"},
                "subject": {"name": subject_model.name, "sha256": digest(subject_model), "storage_type": "Q4_K - Medium"},
            },
            "corpus": {"name": corpus.name, "sha256": digest(corpus)},
            "input": {
                "tokens": {"path": tokens.name, "sha256": digest(tokens), "bytes": tokens.stat().st_size, "encoding": "u32le", "count": 4097},
                "window_tokens": 4096,
                "stride_tokens": 4095,
                "rows": 4096,
                "vocab_size": 2,
            },
            "oracle": {
                "manifest": {"path": "oracle.json", "sha256": digest(self.root / "oracle.json"), "body": oracle_run},
                "logits": {"path": "oracle.f32", "sha256": digest(self.root / "oracle.f32"), "bytes": len(logits), "encoding": "row-major-f32-le"},
            },
        }
        (self.root / "oracle-stage.json").write_text(json.dumps(exported))
        metal = self.root / "metal"
        metal.mkdir()
        (self.root / "llama_logits.cpp").write_bytes((ROOT / "research/oracle/llama_logits.cpp").read_bytes())
        for name in ("tokens.u32le", "oracle.f32", "oracle-stage.json", "llama_logits.cpp"):
            (metal / name).write_bytes((self.root / name).read_bytes())
        (metal / "metal.f32").write_bytes(logits)
        (metal / "leone-metal.f32").write_bytes(logits)
        (metal / "leone.metal").write_bytes(b"shader\n")
        (metal / "leone-doctor.txt").write_text("platform: darwin arm64\nselected backend: metal\nmetal device: detected\n")
        (metal / "leone-eval.stdout").write_text("logits: leone-metal.f32 (4096 positions, vocab 2)\n")
        metal_run = dict(oracle_run)
        metal_run["model"] = {
            "path": subject_model.name,
            "sha256": digest(subject_model),
            "storage_type": "Q4_K - Medium",
            "vocab_size": 2,
            "architecture": "test",
            "tokenizer": "gpt2",
        }
        metal_run["execution"] = {
            "device": "metal", "backend_registry": "MTL", "device_type": "gpu",
            "device_name": "Apple M4", "device_description": "Apple M4", "device_id": None,
        }
        metal_run["logits"] = {"path": "metal.f32", "sha256": digest(metal / "metal.f32"), "bytes": len(logits), "encoding": "row-major-f32-le"}
        (metal / "metal.json").write_text(json.dumps(metal_run))
        native_source = "b" * 40
        native_run = {
            "schema_version": "leone.native-metal-eval.v1",
            "source_commit": native_source,
            "executable": {
                "name": "leone",
                "sha256": "a" * 64,
                "build_info": {
                    "schema_version": "leone.build-info.v1",
                    "source_commit": native_source,
                    "source_tree_dirty": False,
                    "profile": "release",
                    "target": "aarch64-apple-darwin",
                    "metal_shader": {"name": "leone.metal", "sha256": digest(metal / "leone.metal"), "bytes": 7},
                },
            },
            "model": {
                "path": subject_model.name,
                "sha256": digest(subject_model),
                "storage_type": "Q4_K - Medium",
                "vocab_size": 2,
                "architecture": "test",
                "tokenizer": "gpt2",
            },
            "input": {
                "tokens": {"path": "tokens.u32le", "sha256": digest(tokens), "bytes": tokens.stat().st_size, "count": 4097, "encoding": "u32le"},
                "window_tokens": 4096,
                "stride_tokens": 4095,
                "rows": 4096,
                "vocab_size": 2,
            },
            "execution": {
                "engine": "leone",
                "backend": "metal",
                "backend_registry": "metal",
                "device_type": "gpu",
                "device": "metal",
                "kv_cache_dtype": "F16",
                "prefill": {"path": "sequential", "chunk_tokens": None},
                "shader": {"name": "leone.metal", "sha256": digest(metal / "leone.metal"), "bytes": 7},
                "machine": {
                    "path": "leone-doctor.txt",
                    "sha256": digest(metal / "leone-doctor.txt"),
                    "body": "platform: darwin arm64\nselected backend: metal\nmetal device: detected\n",
                },
                "eval_stdout": {
                    "path": "leone-eval.stdout",
                    "sha256": digest(metal / "leone-eval.stdout"),
                    "body": "logits: leone-metal.f32 (4096 positions, vocab 2)\n",
                },
            },
            "logits": {
                "path": "leone-metal.f32",
                "sha256": digest(metal / "leone-metal.f32"),
                "bytes": len(logits),
                "encoding": "row-major-f32-le",
            },
        }
        (metal / "leone-metal.json").write_text(json.dumps(native_run))
        metal_model = json.loads(json.dumps(exported["model"]))
        metal_model["subject"].update({"storage_type": "Q4_K - Medium", "vocab_size": 2, "architecture": "test", "tokenizer": "gpt2"})
        metal_record = {
            "schema_version": "leone.quality-stage.v1",
            "stage": "metal-subject",
            "source_commit": "a" * 40,
            "native_source_commit": native_source,
            "expected_native_source_commit": native_source,
            "engine": exported["engine"],
            "adapter": exported["adapter"],
            "parent": {
                "path": "oracle-stage.json",
                "manifest_sha256": digest(self.root / "oracle-stage.json"),
                "manifest": exported,
            },
            "model": metal_model,
            "corpus": exported["corpus"],
            "input": exported["input"],
            "oracle": {"logits": {"path": "oracle.f32", "sha256": digest(metal / "oracle.f32"), "bytes": len(logits), "encoding": "row-major-f32-le"}},
            "subject": {
                "manifest": {"path": "metal.json", "sha256": digest(metal / "metal.json"), "body": metal_run},
                "logits": {"path": "metal.f32", "sha256": digest(metal / "metal.f32"), "bytes": len(logits), "encoding": "row-major-f32-le"},
            },
            "execution": metal_run["execution"],
            "native": {
                "manifest": {
                    "path": "leone-metal.json",
                    "sha256": digest(metal / "leone-metal.json"),
                    "body": native_run,
                },
                "logits": {
                    "path": "leone-metal.f32",
                    "sha256": digest(metal / "leone-metal.f32"),
                    "bytes": len(logits),
                    "encoding": "row-major-f32-le",
                },
            },
            "native_execution": native_run["execution"],
        }
        (metal / "metal-stage.json").write_text(json.dumps(metal_record))
        task = {
            "schema_version": "leone.quality-task.v2",
            "task": {
                "name": "needle-exact-answer",
                "context_tokens": 4096,
                "minimum_context_tokens": 4096,
                "prompt": {"encoding": "utf-8", "text": "corpus\n", "bytes": 7, "sha256": hashlib.sha256(b"corpus\n").hexdigest()},
                "needle": {"encoding": "utf-8", "text": "4", "bytes": 1, "sha256": hashlib.sha256(b"4").hexdigest()},
                "answer": {"row": 4095, "text": "4", "sha256": hashlib.sha256(b"4").hexdigest(), "token_ids": [1]},
            },
            "model": {
                "family": "test",
                "oracle": {"name": oracle_model.name, "sha256": digest(oracle_model), "storage_type": "BF16", "vocab_size": 2, "architecture": "test", "tokenizer": "gpt2"},
                "subject": {"name": subject_model.name, "sha256": digest(subject_model), "storage_type": "Q4_K - Medium"},
            },
            "tokenizer": {"name": "gpt2"},
            "tokenization": {"source": {"name": "llama.cpp", "git_commit": pin, "pin_file": "external/PINNED", "provenance": "declared", "trust": "operator-attested"}, "executable": {"name": "llama-tokenize", "sha256": "d" * 64, "bytes": 1}, "model": {"name": oracle_model.name, "sha256": digest(oracle_model)}, "method": "llama-tokenize", "prompt_sha256": hashlib.sha256(b"corpus\n").hexdigest(), "needle_sha256": hashlib.sha256(b"4").hexdigest(), "combined_sha256": digest(corpus), "answer_tokens_sha256": hashlib.sha256(struct.pack("<I", 1)).hexdigest(), "input_tokens_sha256": digest(tokens)},
            "corpus": exported["corpus"],
            "input": exported["input"],
        }
        task_path = self.root / "long-context-task.json"
        task_path.write_text(json.dumps(task))
        self.task_path = task_path
        oracle_package = self.root / "packaged-oracle-stage"
        metal_package = self.root / "packaged-metal-stage"
        oracle_package.mkdir()
        shutil.copy2(self.root / "oracle-stage.json", oracle_package / "oracle-stage.json")
        shutil.copy2(self.root / "oracle.json", oracle_package / "oracle.json")
        shutil.copy2(self.root / "oracle.f32", oracle_package / "oracle.f32")
        shutil.copy2(self.root / "tokens.u32le", oracle_package / "tokens.u32le")
        shutil.copy2(self.root / "llama_logits.cpp", oracle_package / "llama_logits.cpp")
        shutil.copytree(metal, metal_package)
        self.task_result = VALIDATE.validate_task(task_path, self.root, metal)

    def test_matching_stages_pass(self):
        exported, metal = VALIDATE.validate_compare(self.root, self.root / "metal")
        self.assertEqual(exported["input"], metal["input"])
        self.assertEqual(metal["execution"]["device"], "metal")
        self.assertEqual(metal["native_execution"]["backend"], "metal")

    def test_sample_writer_rejects_changed_source_logits(self):
        source = self.root / "oracle.f32"
        data = bytearray(source.read_bytes())
        data[0] ^= 1
        source.write_bytes(data)
        result = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts/write-quality-samples.py"),
                "--oracle-stage",
                str(self.root),
                "--metal-stage",
                str(self.root / "metal"),
                "--task-manifest",
                str(self.task_path),
                "--output-dir",
                str(self.root / "changed-samples"),
                "--path-prefix",
                "changed-samples",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("logits source hash differs", result.stderr)

    def test_argmax_uses_lowest_index_and_positive_zero(self):
        self.assertEqual(VALIDATE.canonical_argmax((2.0, 2.0), "tie"), 0)
        self.assertEqual(VALIDATE.canonical_argmax((-0.0, 0.0), "signed zero"), 1)
        self.assertEqual(VALIDATE.canonical_argmax((0.0, -0.0), "signed zero"), 0)

    def test_logsumexp_matches_rust_negative_infinity_policy(self):
        value = VALIDATE.logsumexp((float("-inf"), 0.0), "negative infinity")
        self.assertEqual(value, 0.0)
        kld, top_match = VALIDATE.row_stats((float("-inf"), 0.0), (0.0, 1.0), "negative infinity")
        self.assertGreaterEqual(kld, 0.0)
        self.assertTrue(top_match)

    def test_native_text_requires_the_recorded_final_newline(self):
        path = self.root / "metal/leone-metal.json"
        record = json.loads(path.read_text())
        record["execution"]["machine"]["body"] = record["execution"]["machine"]["body"].rstrip("\n")
        path.write_text(json.dumps(record))
        stage_path = self.root / "metal/metal-stage.json"
        stage = json.loads(stage_path.read_text())
        stage["native"]["manifest"]["sha256"] = digest(path)
        stage["native"]["manifest"]["body"] = record
        stage_path.write_text(json.dumps(stage))
        with self.assertRaisesRegex(ValueError, "machine record body differs"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_native_text_rejects_duplicate_machine_identity_fields(self):
        native_path = self.root / "metal/leone-metal.json"
        doctor_path = self.root / "metal/leone-doctor.txt"
        stage_path = self.root / "metal/metal-stage.json"
        base_native = json.loads(native_path.read_text())
        base_stage = json.loads(stage_path.read_text())
        for duplicate in (
            "platform: linux x86_64",
            "selected backend: cpu",
            "metal device: unavailable",
            "platform:\tlinux x86_64",
            "selected backend:\tcpu",
            "metal device:\tunavailable",
        ):
            native = json.loads(json.dumps(base_native))
            machine = native["execution"]["machine"]
            machine["body"] += duplicate + "\n"
            doctor_path.write_text(machine["body"])
            machine["sha256"] = digest(doctor_path)
            native_path.write_text(json.dumps(native))
            stage = json.loads(json.dumps(base_stage))
            stage["native"]["manifest"]["sha256"] = digest(native_path)
            stage["native"]["manifest"]["body"] = native
            stage["native_execution"] = native["execution"]
            stage_path.write_text(json.dumps(stage))
            with self.assertRaisesRegex(ValueError, "field is missing, duplicated, or noncanonical"):
                VALIDATE.validate_metal(self.root / "metal")

    def test_changed_input_is_rejected(self):
        path = self.root / "metal/metal-stage.json"
        record = json.loads(path.read_text())
        record["input"]["window_tokens"] = 3
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "stage stride differs from window minus one"):
            VALIDATE.validate_compare(self.root, self.root / "metal")

    def test_changed_parent_snapshot_is_rejected(self):
        path = self.root / "metal/oracle-stage.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "parent manifest snapshot changed"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_non_metal_backend_is_rejected(self):
        path = self.root / "metal/metal.json"
        record = json.loads(path.read_text())
        record["execution"]["backend_registry"] = "CPU"
        path.write_text(json.dumps(record))
        stage_path = self.root / "metal/metal-stage.json"
        stage = json.loads(stage_path.read_text())
        stage["subject"]["manifest"]["sha256"] = digest(path)
        stage["subject"]["manifest"]["body"] = record
        stage_path.write_text(json.dumps(stage))
        with self.assertRaisesRegex(ValueError, "not Metal"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_native_backend_is_rejected(self):
        path = self.root / "metal/leone-metal.json"
        record = json.loads(path.read_text())
        record["execution"]["backend"] = "cpu"
        path.write_text(json.dumps(record))
        stage_path = self.root / "metal/metal-stage.json"
        stage = json.loads(stage_path.read_text())
        stage["native"]["manifest"]["sha256"] = digest(path)
        stage["native"]["manifest"]["body"] = record
        stage_path.write_text(json.dumps(stage))
        with self.assertRaisesRegex(ValueError, "native execution backend"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_native_source_is_bound_to_build(self):
        path = self.root / "metal/metal-stage.json"
        record = json.loads(path.read_text())
        record["native_source_commit"] = "c" * 40
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "native source commit differs"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_subject_storage_type_is_required(self):
        path = self.root / "metal/metal-stage.json"
        record = json.loads(path.read_text())
        del record["model"]["subject"]["storage_type"]
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Metal subject storage type is missing"):
            VALIDATE.validate_metal(self.root / "metal")

    def test_long_context_task_requires_the_declared_depth(self):
        task = json.loads(self.task_path.read_text())
        task["task"]["context_tokens"] = 2048
        self.task_path.write_text(json.dumps(task))
        with self.assertRaisesRegex(ValueError, "context_tokens is invalid"):
            VALIDATE.validate_task(self.task_path, self.root, self.root / "metal")

    def test_long_context_task_binds_the_answer_to_all_subjects(self):
        path = self.root / "metal/leone-metal.f32"
        data = bytearray(path.read_bytes())
        data[4095 * 8:4095 * 8 + 8] = struct.pack("<2f", 1.0, 0.0)
        path.write_bytes(data)
        record = json.loads((self.root / "metal/leone-metal.json").read_text())
        record["logits"]["sha256"] = digest(path)
        (self.root / "metal/leone-metal.json").write_text(json.dumps(record))
        stage = json.loads((self.root / "metal/metal-stage.json").read_text())
        stage["native"]["manifest"]["sha256"] = digest(self.root / "metal/leone-metal.json")
        stage["native"]["manifest"]["body"] = record
        stage["native"]["logits"]["sha256"] = digest(path)
        (self.root / "metal/metal-stage.json").write_text(json.dumps(stage))
        with self.assertRaisesRegex(ValueError, "quality task leone_metal answer differs"):
            VALIDATE.validate_task(self.task_path, self.root, self.root / "metal")

    def test_absolute_artifact_path_is_rejected(self):
        path = self.root / "oracle-stage.json"
        record = json.loads(path.read_text())
        record["model"]["oracle"]["name"] = "/tmp/oracle-model.gguf"
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "absolute path"):
            VALIDATE.validate_export(self.root)

    def write_comparison_fixture(self):
        exported = json.loads((self.root / "oracle-stage.json").read_text())
        metal = json.loads((self.root / "metal/metal-stage.json").read_text())
        sample_directory = self.root / "packaged-samples"
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts/write-quality-samples.py"),
                "--oracle-stage",
                str(self.root),
                "--metal-stage",
                str(self.root / "metal"),
                "--task-manifest",
                str(self.task_path),
                "--output-dir",
                str(sample_directory),
                "--path-prefix",
                "packaged-samples",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        sample_path = sample_directory / "sampling.json"
        sample_body = json.loads(sample_path.read_text())
        self.task_result = VALIDATE.validate_task(self.task_path, self.root, self.root / "metal", sample_path)

        def make_receipt(engine, commit, logits):
            return {
                "schema_version": 3,
                "receipt_id": "00000000-0000-4000-8000-000000000000" if engine == "llama.cpp-metal" else "00000000-0000-4000-8000-000000000001",
                "created_utc": "2026-09-19T00:00:00Z",
                "corpus": {"name": "test", "sha256": exported["corpus"]["sha256"], "n_prompts": 1, "n_tokens_scored": sample_body["sampling"]["sample_count"]},
                "oracle": {
                    "description": "llama.cpp BF16 execution",
                    "artifact_sha256": sample_body["artifacts"]["logits"]["oracle"]["sha256"],
                    "engine": {"name": "llama.cpp", "git_commit": exported["engine"]["git_commit"]},
                    "dtype": "bf16",
                },
                "subject": {
                    "model_artifact": {"sha256": exported["model"]["subject"]["sha256"], "path": "subject-model.gguf"},
                    "logits_artifact": {"sha256": sample_body["artifacts"]["logits"][logits]["sha256"], "path": "not-distributed"},
                    "engine": {"name": engine, "git_commit": commit},
                },
                "sample_count": sample_body["sampling"]["sample_count"],
                "metrics": {
                    "kld": {
                        "mean": 0.0,
                        "p50": 0.0,
                        "p99": 0.0,
                        "max": 0.0,
                        "definition": VALIDATE.KLD_DEFINITION,
                    },
                    "top1_agreement": 1.0,
                },
            }

        llama_receipt = make_receipt("llama.cpp-metal", exported["engine"]["git_commit"], "llama_cpp")
        leone_receipt = make_receipt("leone-metal", metal["native_source_commit"], "leone")
        llama_quality_path = self.root / "llama-quality.json"
        leone_quality_path = self.root / "leone-quality.json"
        llama_quality_path.write_text(json.dumps(llama_receipt))
        leone_quality_path.write_text(json.dumps(leone_receipt))
        verifier = self.root / "leone-receipt-verify"
        verifier.write_text("#!/bin/sh\n[ \"$1\" = quality ] && [ -f \"$2\" ]\n")
        verifier.chmod(0o755)
        comparison = {
            "schema_version": "leone.quality-cross-device.v2",
            "source_commit": metal["native_source_commit"],
            "executable": {"name": "leone", "sha256": "a" * 64},
            "build_info": {
                "schema_version": "leone.build-info.v1",
                "source_commit": metal["native_source_commit"],
                "source_tree_dirty": False,
                "profile": "release",
                "target": "x86_64-unknown-linux-gnu",
            },
            "source_identities": {
                "oracle": {"source_commit": "a" * 40},
                "llama_cpp_metal": {"source_commit": "a" * 40},
                "leone_metal": {"source_commit": metal["native_source_commit"], "expected_source_commit": metal["native_source_commit"]},
                "statistics": {"source_commit": metal["native_source_commit"]},
            },
            "models": metal["model"],
            "corpus": exported["corpus"],
            "input": exported["input"],
            "oracle": {
                "engine": exported["engine"],
                "execution": exported["oracle"]["manifest"]["body"]["execution"],
                "logits": exported["oracle"]["logits"],
            },
            "subjects": {
                "llama_cpp": {
                    "engine": {"name": "llama.cpp-metal", "git_commit": exported["engine"]["git_commit"]},
                    "execution": metal["execution"],
                    "logits": metal["subject"]["logits"],
                },
                "leone": {
                    "engine": {"name": "leone-metal", "git_commit": metal["native_source_commit"]},
                    "execution": metal["native_execution"],
                    "logits": metal["native"]["logits"],
                },
            },
            "stages": {
                "oracle_export": {"path": "packaged-oracle-stage/oracle-stage.json", "sha256": digest(self.root / "packaged-oracle-stage/oracle-stage.json"), "body": exported},
                "metal_subject": {"path": "packaged-metal-stage/metal-stage.json", "sha256": digest(self.root / "packaged-metal-stage/metal-stage.json"), "body": metal},
            },
            "tasks": {
                "long_context": {
                    "manifest": {"path": "long-context-task.json", "sha256": digest(self.task_path), "body": json.loads(self.task_path.read_text())},
                    "result": self.task_result,
                },
            },
            "samples": {
                "manifest": {
                    "path": "packaged-samples/sampling.json",
                    "sha256": digest(sample_path),
                    "body": sample_body,
                },
            },
            "quality": {
                "llama_cpp": {"path": llama_quality_path.name, "sha256": digest(llama_quality_path), "receipt": llama_receipt},
                "leone": {"path": leone_quality_path.name, "sha256": digest(leone_quality_path), "receipt": leone_receipt},
            },
            "validation": {
                "mode": "offline",
                "packaged_stage_manifests": True,
                "generation_rehashed_original_artifacts": True,
                "statistics_executable": {"name": "leone", "sha256": "a" * 64},
                "receipt_parser": {"name": "leone-receipt-verify", "sha256": digest(verifier), "interface": "quality"},
            },
        }
        trusted = {
            "source_commit": metal["native_source_commit"],
            "platform": "darwin-arm64",
            "target": "aarch64-apple-darwin",
            "statistics_target": "x86_64-unknown-linux-gnu",
            "backend": "metal",
            "adapter_path": "research/oracle/llama_logits.cpp",
            "adapter_sha256": exported["adapter"]["sha256"],
            "model_family": "test",
            "model_sha256": metal["model"]["subject"]["sha256"],
            "oracle_model_sha256": exported["model"]["oracle"]["sha256"],
            "corpus_sha256": exported["corpus"]["sha256"],
            "sample_manifest_sha256": digest(sample_path),
            "sample_contract": "linspace-inclusive-v1:128+task-rows",
            "metric_family": "kld",
        }
        comparison["validation"]["trusted"] = trusted
        manifest = self.root / "comparison.json"
        manifest.write_text(json.dumps(comparison))
        for path in (
            self.root / "packaged-oracle-stage/oracle.f32",
            self.root / "packaged-metal-stage/oracle.f32",
            self.root / "packaged-metal-stage/metal.f32",
            self.root / "packaged-metal-stage/leone-metal.f32",
        ):
            path.unlink()
        return manifest, comparison, verifier, trusted

    def test_comparison_receipt_is_linked_by_hash(self):
        manifest, _, verifier, trusted = self.write_comparison_fixture()
        self.assertEqual(VALIDATE.validate_comparison(manifest, verifier, trusted)["subjects"]["leone"]["execution"]["backend"], "metal")
        VALIDATE.run_stage_command(
            "comparison",
            [
                str(manifest),
                str(verifier),
                "--source-commit", trusted["source_commit"],
                "--platform", trusted["platform"],
                "--target", trusted["target"],
                "--statistics-target", trusted["statistics_target"],
                "--backend", trusted["backend"],
                "--adapter-path", trusted["adapter_path"],
                "--adapter-sha256", trusted["adapter_sha256"],
                "--model-family", trusted["model_family"],
                "--model-sha256", trusted["model_sha256"],
                "--oracle-model-sha256", trusted["oracle_model_sha256"],
                "--corpus-sha256", trusted["corpus_sha256"],
                "--sample-manifest-sha256", trusted["sample_manifest_sha256"],
                "--sample-contract", trusted["sample_contract"],
                "--metric-family", trusted["metric_family"],
            ],
        )
    def test_comparison_rejects_changed_trusted_inputs(self):
        manifest, comparison, verifier, trusted = self.write_comparison_fixture()
        for name, value, message in (
            ("source_commit", "c" * 40, "comparison trusted source_commit differs"),
            ("statistics_target", "aarch64-apple-darwin", "comparison trusted statistics_target differs"),
            ("backend", "cuda", "comparison trusted backend differs"),
            ("adapter_sha256", "f" * 64, "comparison trusted adapter_sha256 differs"),
            ("model_sha256", "e" * 64, "comparison trusted model_sha256 differs"),
            ("oracle_model_sha256", "d" * 64, "comparison trusted oracle_model_sha256 differs"),
            ("corpus_sha256", "b" * 64, "comparison trusted corpus_sha256 differs"),
            ("sample_contract", "other", "comparison trusted sample_contract differs"),
        ):
            bad_trusted = {**trusted, name: value}
            with self.assertRaisesRegex(ValueError, message):
                VALIDATE.validate_comparison(manifest, verifier, bad_trusted)
        comparison["build_info"]["target"] = trusted["target"]
        manifest.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, "statistics target differs from trusted input"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)

    def test_comparison_rejects_changed_artifacts(self):
        manifest, comparison, verifier, trusted = self.write_comparison_fixture()
        leone_receipt = comparison["quality"]["leone"]["receipt"]
        leone_quality_path = self.root / comparison["quality"]["leone"]["path"]
        comparison["stages"]["metal_subject"]["body"]["input"]["window_tokens"] = 3
        manifest.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, "packaged Metal stage body differs"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)
        comparison["stages"]["metal_subject"]["body"]["input"]["window_tokens"] = 4096
        manifest.write_text(json.dumps(comparison))
        leone_receipt["metrics"]["top1_agreement"] = 1.1
        leone_quality_path.write_text(json.dumps(leone_receipt))
        comparison["quality"]["leone"]["sha256"] = digest(leone_quality_path)
        comparison["quality"]["leone"]["receipt"] = leone_receipt
        manifest.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, r"outside \[0, 1\]"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)
        leone_receipt["metrics"]["top1_agreement"] = 1.0
        leone_quality_path.write_text(json.dumps(leone_receipt))
        comparison["quality"]["leone"]["sha256"] = digest(leone_quality_path)
        comparison["quality"]["leone"]["receipt"] = leone_receipt
        manifest.write_text(json.dumps(comparison))
        packaged_stage = self.root / "packaged-metal-stage/metal-stage.json"
        packaged_bytes = packaged_stage.read_bytes()
        packaged_stage.write_bytes(packaged_bytes + b"\n")
        with self.assertRaisesRegex(ValueError, "packaged Metal stage SHA-256 differs"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)
        packaged_stage.write_bytes(packaged_bytes)
        leone_receipt["metrics"]["kld"]["mean"] = 1.0
        leone_quality_path.write_text(json.dumps(leone_receipt))
        with self.assertRaisesRegex(ValueError, "quality receipt SHA-256 differs"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)

    def test_metal_generation_anchor_is_written_after_sampling(self):
        manifest, _, _, trusted = self.write_comparison_fixture()
        generation = self.root / "metal-generation.json"
        subprocess.run(
            [
                "python3",
                str(ROOT / "scripts/write-metal-quality-generation.py"),
                str(manifest),
                str(generation),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        body = json.loads(generation.read_text())
        self.assertEqual(body["schema_version"], "leone.quality-metal-generation.v1")
        self.assertEqual(body["sample_manifest"]["sha256"], trusted["sample_manifest_sha256"])
        self.assertEqual(body["backend"], "metal")

    def test_external_anchor_accepts_legacy_embedded_record(self):
        manifest, comparison, verifier, trusted = self.write_comparison_fixture()
        comparison["validation"]["trusted"] = {
            key: value for key, value in trusted.items() if key != "sample_manifest_sha256"
        }
        manifest.write_text(json.dumps(comparison))
        VALIDATE.validate_comparison(manifest, verifier, trusted)

    def test_embedded_anchor_matches_packaged_sample_manifest(self):
        manifest, comparison, verifier, trusted = self.write_comparison_fixture()
        expected = dict(trusted)
        comparison["validation"]["trusted"]["sample_manifest_sha256"] = "0" * 64
        manifest.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, "embedded sample manifest differs"):
            VALIDATE.validate_comparison(manifest, verifier, expected)

    def test_generation_anchor_refuses_replacement(self):
        manifest, _, _, _ = self.write_comparison_fixture()
        generation = self.root / "metal-generation.json"
        command = ["python3", str(ROOT / "scripts/write-metal-quality-generation.py"), str(manifest), str(generation)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        original = generation.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(generation.read_bytes(), original)

    def test_generation_anchor_reports_published_inode(self):
        manifest, _, _, _ = self.write_comparison_fixture()
        generation = self.root / "metal-generation.json"
        result = subprocess.run(
            [
                "python3",
                str(ROOT / "scripts/write-metal-quality-generation.py"),
                str(manifest),
                str(generation),
                "--identity",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        device, inode = (int(value) for value in result.stdout.strip().split(":", 1))
        record = generation.stat()
        self.assertEqual((device, inode), (record.st_dev, record.st_ino))

    def test_fixed_trust_sample_shift_is_rejected_by_anchor(self):
        manifest, comparison, verifier, trusted = self.write_comparison_fixture()
        sample_directory = self.root / "packaged-samples"
        sample_body = comparison["samples"]["manifest"]["body"]
        labels = {"oracle": "oracle.logits.bin", "llama_cpp": "llama-cpp-metal.logits.bin", "leone": "leone-metal.logits.bin"}
        for label, filename in labels.items():
            path = sample_directory / filename
            values = struct.unpack(f"<{path.stat().st_size // 4}f", path.read_bytes())
            path.write_bytes(struct.pack(f"<{len(values)}f", *(value + 1.0 for value in values)))
            artifact = sample_body["artifacts"]["logits"][label]
            data = path.read_bytes()
            row_bytes = 2 * 4
            artifact["sha256"] = digest(path)
            artifact["source_row_sha256"] = [
                hashlib.sha256(data[offset:offset + row_bytes]).hexdigest()
                for offset in range(0, len(data), row_bytes)
            ]
            comparison["quality"]["llama_cpp"]["receipt"]["oracle"]["artifact_sha256"] = sample_body["artifacts"]["logits"]["oracle"]["sha256"]
            comparison["quality"]["leone"]["receipt"]["oracle"]["artifact_sha256"] = sample_body["artifacts"]["logits"]["oracle"]["sha256"]
            comparison["quality"]["llama_cpp"]["receipt"]["subject"]["logits_artifact"]["sha256"] = sample_body["artifacts"]["logits"]["llama_cpp"]["sha256"]
            comparison["quality"]["leone"]["receipt"]["subject"]["logits_artifact"]["sha256"] = sample_body["artifacts"]["logits"]["leone"]["sha256"]
        sample_path = sample_directory / "sampling.json"
        sample_path.write_text(json.dumps(sample_body))
        comparison["samples"]["manifest"]["sha256"] = digest(sample_path)
        for name in ("llama_cpp", "leone"):
            quality_path = self.root / comparison["quality"][name]["path"]
            quality = comparison["quality"][name]["receipt"]
            quality_path.write_text(json.dumps(quality))
            comparison["quality"][name]["sha256"] = digest(quality_path)
        comparison["tasks"]["long_context"]["result"] = VALIDATE.validate_task(
            self.task_path, self.root, self.root / "metal", sample_path
        )
        manifest.write_text(json.dumps(comparison))
        with self.assertRaisesRegex(ValueError, "quality sample manifest differs from the trusted anchor"):
            VALIDATE.validate_comparison(manifest, verifier, trusted)


if __name__ == "__main__":
    unittest.main()
