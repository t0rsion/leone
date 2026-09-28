"""Check the packaged CUDA quality comparison contract."""

import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_quality_stage", ROOT / "scripts/validate-quality-stage.py"
)
VALIDATE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(VALIDATE)
WRITER = ROOT / "scripts/write-cuda-quality-comparison.py"
FREEZE = ROOT / "scripts/freeze-cuda-generation.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_collection(source, destination):
    script = (ROOT / "scripts/quality-concurrent-service.sh").read_text()
    body = script.split("publish_collection_file() {", 1)[1]
    body = body.split("\npublish_collection_manifest()", 1)[0]
    command = 'publish_collection_file() {' + body + '\npublish_collection_file "$1" "$2"\n'
    return subprocess.run(
        ["bash", "-c", command, "collection-test", str(source), str(destination)],
        capture_output=True, text=True,
    )


def logits(path, offset, rows=4):
    path.write_bytes(
        b"".join(struct.pack("<3f", 1.0 + offset, 0.0, -1.0) for _ in range(rows))
    )


def artifact_record(path, name, encoding="row-major-f32-le"):
    return {
        "path": name,
        "sha256": digest(path),
        "bytes": path.stat().st_size,
        "encoding": encoding,
    }


def execution(device, backend, device_type):
    return {
        "device": device,
        "backend": backend,
        "backend_registry": backend.upper(),
        "device_type": device_type,
        "device_name": "fixture device",
        "device_description": "fixture",
        "logits_dtype": "f32",
    }


def quality_receipt(corpus, models, sample_records, metrics, engine, commit, subject_key, pin, rows):
    return {
        "schema_version": 3,
        "receipt_id": f"00000000-0000-4000-8000-{subject_key:0>12}",
        "created_utc": "2026-09-20T00:00:00Z",
        "corpus": {
            "name": corpus.name,
            "sha256": digest(corpus),
            "n_prompts": 1,
            "n_tokens_scored": rows,
        },
        "oracle": {
            "description": "llama.cpp BF16 execution",
            "artifact_sha256": sample_records["oracle"]["sha256"],
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "dtype": "bf16",
        },
        "subject": {
            "model_artifact": {
                "sha256": models["subject"]["sha256"],
                "path": models["subject"]["name"],
            },
            "logits_artifact": {
                "sha256": sample_records[subject_key]["sha256"],
                "path": sample_records[subject_key]["path"],
            },
            "engine": {"name": engine, "git_commit": commit},
        },
        "sample_count": rows,
        "metrics": {
            "kld": {**metrics, "definition": VALIDATE.KLD_DEFINITION},
            "top1_agreement": metrics["top1_agreement"],
        },
    }


def write_samples(root, full, token_record):
    sample_dir = root / "samples"
    sample_dir.mkdir()
    rows = full["oracle"]["bytes"] // 12
    requested = min(VALIDATE.PUBLIC_SAMPLE_ROWS, rows)
    indices = [index * (rows - 1) // (requested - 1) for index in range(requested)]
    values = struct.unpack(f"<{rows + 1}I", (root / "tokens.u32le").read_bytes())
    sample_tokens = sample_dir / "sample.tokens.u32le"
    sample_tokens.write_bytes(struct.pack(f"<{len(indices) + 1}I", values[0], *[values[row + 1] for row in indices]))
    sample_records = {}
    for label, source_name in (("oracle", "oracle"), ("llama_cpp", "llama"), ("leone", "leone")):
        source = root / f"full-{source_name}.f32"
        target = sample_dir / f"{label}.f32"
        row_hashes = []
        with source.open("rb") as source_file, target.open("wb") as output:
            for _ in indices:
                row = source_file.read(12)
                output.write(row)
                row_hashes.append(hashlib.sha256(row).hexdigest())
        sample_records[label] = {
            "path": f"samples/{label}.f32",
            "sha256": digest(target),
            "bytes": target.stat().st_size,
            "encoding": "row-major-f32-le",
            "source_row_sha256": row_hashes,
        }
    body = {
        "schema_version": "leone.quality-sample.v1",
        "provenance": {
            "full_logits": "generation-only",
            "row_hashes": "sha256 of each packaged row in source order",
            "offline_replay": "sampled rows only",
        },
        "source": {
            "rows": rows,
            "vocab_size": 3,
            "tokens": token_record,
            "logits": full,
        },
        "originals": {
            "mode": "generation-only",
            "retention": "caller-owned immutable cache",
            "logits": full,
        },
        "sampling": {
            "algorithm": "linspace-inclusive-v1",
            "requested_rows": requested,
            "source_rows": rows,
            "sample_count": len(indices),
            "indices": indices,
            "task_rows": [rows - 1],
        },
        "artifacts": {
            "tokens": {
                "path": "samples/sample.tokens.u32le",
                "sha256": digest(sample_tokens),
                "bytes": sample_tokens.stat().st_size,
                "count": len(indices) + 1,
                "encoding": "u32le",
            },
            "logits": sample_records,
        },
    }
    manifest = sample_dir / "sampling.json"
    manifest.write_text(json.dumps(body))
    return manifest, body, sample_records


def generation_record(full_paths, full, input_record, models, corpus, task_path, manifests, task_rows):
    rows = input_record["rows"]
    vocab = input_record["vocab_size"]
    requested = min(VALIDATE.PUBLIC_SAMPLE_ROWS, rows)
    base = [index * (rows - 1) // (requested - 1) for index in range(requested)]
    indices = sorted(set(base + task_rows))
    sampled = {}
    for label, path in full_paths.items():
        row_hashes = []
        sample_bytes = bytearray()
        with path.open("rb") as source:
            for row in indices:
                source.seek(row * vocab * 4)
                data = source.read(vocab * 4)
                sample_bytes.extend(data)
                row_hashes.append(hashlib.sha256(data).hexdigest())
        sampled[label] = {
            "sha256": hashlib.sha256(sample_bytes).hexdigest(),
            "bytes": len(sample_bytes),
            "encoding": "row-major-f32-le",
            "source_row_sha256": row_hashes,
        }
    return {
        "schema_version": "leone.quality-generation.v1",
        "inputs": {
            "model_sha256": models["subject"]["sha256"],
            "oracle_model_sha256": models["oracle"]["sha256"],
            "corpus_sha256": digest(corpus),
            "task_manifest_sha256": digest(task_path),
            "tokens_sha256": input_record["tokens"]["sha256"],
            "token_count": input_record["tokens"]["count"],
            "rows": rows,
            "vocab_size": vocab,
            "sample_contract": VALIDATE.SAMPLE_CONTRACT,
        },
        "sampling": {
            "algorithm": "linspace-inclusive-v1",
            "requested_rows": requested,
            "indices": indices,
            "task_rows": task_rows,
        },
        "producer_manifests": {label: digest(path) for label, path in manifests.items()},
        "full_logits": {
            label: {key: full[label][key] for key in ("sha256", "bytes", "encoding")}
            for label in full_paths
        },
        "sampled_logits": sampled,
    }


class CudaQualityTests(unittest.TestCase):
    def test_collection_retains_files_without_duplicate_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "temporary.logits"
            destination = Path(directory) / "retained.logits"
            contents = struct.pack("<3f", 1.0, 0.0, -1.0)
            source.write_bytes(contents)
            result = publish_collection(source, destination)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.stat().st_ino, destination.stat().st_ino)
            source.unlink()
            self.assertEqual(destination.read_bytes(), contents)

    def test_collection_preserves_an_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "temporary.logits"
            destination = Path(directory) / "retained.logits"
            source.write_bytes(b"new collection")
            destination.write_bytes(b"earlier collection")
            result = publish_collection(source, destination)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(destination.read_bytes(), b"earlier collection")
            self.assertEqual(source.read_bytes(), b"new collection")

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest, self.trusted, self.verifier, self.generation = self.write_fixture()

    def write_fixture(self):
        pin = (ROOT / "external/PINNED").read_text().strip()
        source = "c" * 40
        adapter_path = self.root / "adapter.llama_logits.cpp"
        adapter_path.write_bytes((ROOT / "research/oracle/llama_logits.cpp").read_bytes())
        oracle_model = self.root / "oracle.gguf"
        subject_model = self.root / "subject.gguf"
        corpus = self.root / "corpus.txt"
        tokens = self.root / "tokens.u32le"
        oracle_model.write_bytes(b"oracle")
        subject_model.write_bytes(b"subject")
        corpus.write_text("cuda fixture\n")
        token_values = list(range(4097))
        token_values[-1] = 0
        tokens.write_bytes(struct.pack("<4097I", *token_values))
        full_paths = {}
        for name, offset in (("oracle", 0.0), ("llama", 0.01), ("leone", 0.02)):
            path = self.root / f"full-{name}.f32"
            logits(path, offset, 4096)
            full_paths[name] = path
        token_record = {
            "path": "tokens.u32le",
            "sha256": digest(tokens),
            "bytes": tokens.stat().st_size,
            "count": 4097,
            "encoding": "u32le",
        }
        input_record = {
            "tokens": token_record,
            "window_tokens": 4096,
            "stride_tokens": 4095,
            "rows": 4096,
            "vocab_size": 3,
        }
        models = {
            "oracle": {
                "name": oracle_model.name,
                "sha256": digest(oracle_model),
                "storage_type": "BF16",
                "vocab_size": 3,
                "architecture": "fixture",
                "tokenizer": "fixture",
            },
            "subject": {
                "name": subject_model.name,
                "sha256": digest(subject_model),
                "storage_type": "Q4_K - Medium",
                "vocab_size": 3,
                "architecture": "fixture",
                "tokenizer": "fixture",
            },
        }
        full = {
            "oracle": artifact_record(full_paths["oracle"], "generated/oracle.f32"),
            "llama_cpp": artifact_record(full_paths["llama"], "generated/llama-cpp-cuda.f32"),
            "leone": artifact_record(full_paths["leone"], "generated/leone-cuda.f32"),
        }
        sample_manifest, sample_body, sample_records = write_samples(self.root, full, token_record)
        sample_count = len(sample_records["oracle"]["source_row_sha256"])
        oracle_run = {
            "schema_version": "leone.llama-oracle.v2",
            "source_commit": "a" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": {"path": "research/oracle/llama_logits.cpp", "sha256": digest(adapter_path)},
            "executable": {"path": "llama-oracle", "sha256": "b" * 64},
            "model": {**models["oracle"], "path": "oracle.gguf"},
            "input": input_record,
            "execution": execution("cpu", "cpu", "cpu"),
            "logits": artifact_record(full_paths["oracle"], "full-oracle.f32"),
        }
        llama_run = {
            "schema_version": "leone.llama-oracle.v2",
            "source_commit": "b" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": {"path": "research/oracle/llama_logits.cpp", "sha256": digest(adapter_path)},
            "executable": {"path": "llama-cuda", "sha256": "c" * 64},
            "model": {**models["subject"], "path": "subject.gguf"},
            "input": input_record,
            "execution": execution("cuda", "cuda", "gpu"),
            "logits": artifact_record(full_paths["llama"], "full-llama.f32"),
        }
        build_info = {
            "schema_version": "leone.build-info.v1",
            "source_commit": source,
            "source_tree_dirty": False,
            "profile": "release",
            "target": "x86_64-unknown-linux-gnu",
        }
        native_run = {
            "schema_version": "leone.native-eval.v1",
            "source_commit": source,
            "executable": {
                "name": "leone",
                "sha256": "a" * 64,
                "build_info": build_info,
            },
            "model": {**models["subject"], "path": "subject.gguf"},
            "input": input_record,
            "execution": {
                **execution("cuda", "cuda", "gpu"),
                "engine": "leone",
                "kv_cache_dtype": "F16",
                "prefill": {"path": "chunked", "chunk_tokens": 2},
            },
            "logits": artifact_record(full_paths["leone"], "full-leone.f32"),
        }
        producer_dir = self.root / "producers"
        producer_dir.mkdir()
        producer_records = {}
        for label, body in (("oracle", oracle_run), ("llama_cpp", llama_run), ("leone", native_run)):
            path = producer_dir / f"{label}.json"
            path.write_text(json.dumps(body))
            producer_records[label] = {
                "path": f"producers/{path.name}",
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "encoding": "json",
                "body": body,
            }
        task_body = {
            "schema_version": "leone.quality-task.v2",
            "task": {
                "name": "needle-exact-answer",
                "context_tokens": 4096,
                "minimum_context_tokens": 4096,
                "prompt": {"encoding": "utf-8", "text": "cuda ", "bytes": 5, "sha256": hashlib.sha256(b"cuda ").hexdigest()},
                "needle": {"encoding": "utf-8", "text": "fixture\n", "bytes": 8, "sha256": hashlib.sha256(b"fixture\n").hexdigest()},
                "answer": {"row": 4095, "text": "fixture\n", "sha256": hashlib.sha256(b"fixture\n").hexdigest(), "token_ids": [0]},
            },
            "tokenizer": {"name": "fixture"},
            "tokenization": {
                "source": {"name": "llama.cpp", "git_commit": pin},
                "model": {"sha256": models["oracle"]["sha256"]},
                "method": "llama-tokenize",
                "combined_sha256": digest(corpus),
                "input_tokens_sha256": token_record["sha256"],
                "answer_tokens_sha256": hashlib.sha256(struct.pack("<I", 0)).hexdigest(),
            },
            "model": {
                "family": "fixture",
                "oracle": {"sha256": models["oracle"]["sha256"]},
                "subject": {"sha256": models["subject"]["sha256"]},
            },
            "corpus": {"sha256": digest(corpus)},
            "input": input_record,
        }
        task_path = self.root / "task.json"
        task_path.write_text(json.dumps(task_body))
        task_record = {
            "path": task_path.name,
            "sha256": digest(task_path),
            "bytes": task_path.stat().st_size,
            "encoding": "json",
            "body": task_body,
            "rows": [4095],
        }
        generation = generation_record(
            {"oracle": full_paths["oracle"], "llama_cpp": full_paths["llama"], "leone": full_paths["leone"]},
            full,
            input_record,
            models,
            corpus,
            task_path,
            {"oracle": producer_dir / "oracle.json", "llama_cpp": producer_dir / "llama_cpp.json", "leone": producer_dir / "leone.json"},
            [4095],
        )
        llama_receipt = quality_receipt(
            corpus,
            models,
            sample_records,
            VALIDATE.recompute_quality(self.root / "samples/oracle.f32", self.root / "samples/llama_cpp.f32", sample_count, 3),
            "llama.cpp-cuda",
            pin,
            "llama_cpp",
            pin,
            sample_count,
        )
        leone_receipt = quality_receipt(
            corpus,
            models,
            sample_records,
            VALIDATE.recompute_quality(self.root / "samples/oracle.f32", self.root / "samples/leone.f32", sample_count, 3),
            "leone-cuda-eval",
            source,
            "leone",
            pin,
            sample_count,
        )
        quality_paths = {}
        for label, body in (("llama_cpp", llama_receipt), ("leone", leone_receipt)):
            path = self.root / f"{label}-quality.json"
            path.write_text(json.dumps(body))
            quality_paths[label] = {
                "path": path.name,
                "sha256": digest(path),
                "receipt": body,
            }
        verifier = self.root / "leone-receipt-verify"
        verifier.write_text("#!/bin/sh\n[ \"$1\" = quality ] && [ -f \"$2\" ]\n")
        verifier.chmod(0o755)
        adapter_sha = digest(adapter_path)
        trusted = {
            "source_commit": source,
            "platform": "linux-x86_64",
            "statistics_platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "statistics_target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
            "adapter_path": "research/oracle/llama_logits.cpp",
            "adapter_sha256": adapter_sha,
            "model_family": "fixture",
            "model_sha256": models["subject"]["sha256"],
            "oracle_model_sha256": models["oracle"]["sha256"],
            "corpus_sha256": digest(corpus),
            "task_manifest_sha256": digest(task_path),
            "sample_contract": VALIDATE.SAMPLE_CONTRACT,
            "metric_family": "kld",
            "producer_identities": {
                "oracle_stage": {
                    "source_commit": oracle_run["source_commit"],
                    "manifest_sha256": producer_records["oracle"]["sha256"],
                    "executable": {"name": Path(oracle_run["executable"]["path"]).name, "sha256": oracle_run["executable"]["sha256"]},
                },
                "llama_cpp_cuda": {
                    "source_commit": llama_run["source_commit"],
                    "manifest_sha256": producer_records["llama_cpp"]["sha256"],
                    "executable": {"name": Path(llama_run["executable"]["path"]).name, "sha256": llama_run["executable"]["sha256"]},
                    "external_pin": pin,
                },
                "leone_cuda": {
                    "source_commit": native_run["source_commit"],
                    "manifest_sha256": producer_records["leone"]["sha256"],
                    "executable": {"name": native_run["executable"]["name"], "sha256": native_run["executable"]["sha256"]},
                    "build_info_sha256": VALIDATE.json_digest(build_info),
                    "target": "x86_64-unknown-linux-gnu",
                },
            },
            "statistics_identity": {
                "source_commit": source,
                "executable": {"name": "leone", "sha256": "a" * 64},
                "build_info_sha256": VALIDATE.json_digest(build_info),
                "target": "x86_64-unknown-linux-gnu",
            },
            "native_workflow": {
                "kv_cache_dtype": "F16",
                "prefill": {"path": "chunked", "chunk_tokens": 2},
                "token_count": input_record["tokens"]["count"],
            },
        }
        body = {
            "schema_version": "leone.quality-comparison.v2",
            "source_commit": source,
            "statistics_platform": "linux-x86_64",
            "statistics_target": "x86_64-unknown-linux-gnu",
            "executable": {"name": "leone", "sha256": "a" * 64},
            "build_info": build_info,
            "source_identities": {
                "oracle": trusted["producer_identities"]["oracle_stage"],
                "llama_cpp_cuda": trusted["producer_identities"]["llama_cpp_cuda"],
                "leone_cuda": trusted["producer_identities"]["leone_cuda"],
                "statistics": trusted["statistics_identity"],
            },
            "model_family": "fixture",
            "models": models,
            "corpus": {"name": corpus.name, "sha256": digest(corpus)},
            "input": input_record,
            "adapter": {
                "path": "research/oracle/llama_logits.cpp",
                "sha256": adapter_sha,
                "artifact": {
                    "path": adapter_path.name,
                    "sha256": adapter_sha,
                    "bytes": adapter_path.stat().st_size,
                },
            },
            "producers": producer_records,
            "tasks": {"long_context": {"manifest": task_record, "rows": [4095]}},
            "oracle": {
                "engine": oracle_run["engine"],
                "model_sha256": models["oracle"]["sha256"],
                "input_tokens_sha256": token_record["sha256"],
                "execution": oracle_run["execution"],
                "logits": full["oracle"],
            },
            "subjects": {
                "llama_cpp": {
                    "engine": {"name": "llama.cpp-cuda", "git_commit": pin},
                    "model_sha256": models["subject"]["sha256"],
                    "input_tokens_sha256": token_record["sha256"],
                    "execution": llama_run["execution"],
                    "logits": full["llama_cpp"],
                },
                "leone": {
                    "engine": {"name": "leone-cuda-eval", "git_commit": source},
                    "model_sha256": models["subject"]["sha256"],
                    "input_tokens_sha256": token_record["sha256"],
                    "execution": native_run["execution"],
                    "logits": full["leone"],
                },
            },
            "samples": {
                "manifest": {
                    "path": "samples/sampling.json",
                    "sha256": digest(sample_manifest),
                    "body": sample_body,
                }
            },
            "quality": quality_paths,
            "validation": {
                "mode": "offline",
                "packaged_logits": False,
                "generation_rehashed_original_artifacts": True,
                "generation_record": {
                    "path": "generation-record.json",
                    "sha256": "",
                    "bytes": 0,
                    "encoding": "json",
                    "body": generation,
                },
                "statistics_executable": {"name": "leone", "sha256": "a" * 64},
                "receipt_parser": {
                    "name": "leone-receipt-verify",
                    "sha256": digest(verifier),
                    "interface": "quality",
                },
                "trusted": trusted,
            },
        }
        generation_path = self.root / "generation-record.json"
        generation_path.write_text(json.dumps(generation))
        body["validation"]["generation_record"].update(
            {
                "sha256": digest(generation_path),
                "bytes": generation_path.stat().st_size,
            }
        )
        manifest = self.root / "quality-comparison.json"
        manifest.write_text(json.dumps(body))
        return manifest, trusted, verifier, generation

    def test_cuda_comparison_accepts_samples_only(self):
        result = VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)
        self.assertEqual(result["schema_version"], "leone.quality-comparison.v2")
        self.assertFalse(result["validation"]["packaged_logits"])

    def test_cuda_comparison_rejects_full_logits_reintroduced(self):
        record = json.loads(self.manifest.read_text())
        record["validation"]["packaged_logits"] = True
        self.manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "generation-only"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_producer_hash_mutation(self):
        record = json.loads(self.manifest.read_text())
        record["producers"]["leone"]["body"]["logits"]["sha256"] = "f" * 64
        self.manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "Leone producer manifest changed"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_unbound_backend(self):
        record = json.loads(self.manifest.read_text())
        record["subjects"]["leone"]["execution"]["device"] = "cpu"
        self.manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "CUDA Leone subject is not CUDA"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_sample_mutation(self):
        sample = self.root / "samples/leone.f32"
        sample.write_bytes(sample.read_bytes() + b"\x00")
        with self.assertRaisesRegex(ValueError, "quality sample leone logits SHA-256 differs"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_unbound_adapter_source(self):
        adapter = self.root / "adapter.llama_logits.cpp"
        adapter.write_bytes(adapter.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "CUDA adapter source SHA-256 differs"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_untrusted_task_rows(self):
        record = json.loads(self.manifest.read_text())
        record["tasks"]["long_context"]["rows"] = [0]
        self.manifest.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "CUDA long-context task rows differ"):
            VALIDATE.validate_comparison(self.manifest, self.verifier, self.trusted, self.generation)

    def test_cuda_comparison_rejects_each_external_identity_mutation(self):
        mutations = {
            "statistics source": lambda value: value.update(source_commit="e" * 40),
            "statistics platform": lambda value: value.update(statistics_platform="darwin-arm64"),
            "statistics target": lambda value: value.update(statistics_target="other-target"),
            "subject model": lambda value: value.update(model_sha256="e" * 64),
            "oracle source": lambda value: value["producer_identities"]["oracle_stage"].update(source_commit="e" * 40),
            "oracle manifest": lambda value: value["producer_identities"]["oracle_stage"].update(manifest_sha256="e" * 64),
            "oracle executable": lambda value: value["producer_identities"]["oracle_stage"]["executable"].update(sha256="e" * 64),
            "native source": lambda value: value["producer_identities"]["leone_cuda"].update(source_commit="e" * 40),
            "native executable": lambda value: value["producer_identities"]["leone_cuda"]["executable"].update(sha256="e" * 64),
            "native build": lambda value: value["producer_identities"]["leone_cuda"].update(build_info_sha256="e" * 64),
            "statistics executable": lambda value: value["statistics_identity"]["executable"].update(sha256="e" * 64),
            "statistics build": lambda value: value["statistics_identity"].update(build_info_sha256="e" * 64),
            "native KV": lambda value: value["native_workflow"].update(kv_cache_dtype="Q8_0"),
            "native prefill": lambda value: value["native_workflow"].update(prefill={"path": "sequential", "chunk_tokens": None}),
            "native token count": lambda value: value["native_workflow"].update(token_count=2),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                trusted = copy.deepcopy(self.trusted)
                mutate(trusted)
                with self.assertRaises(ValueError):
                    VALIDATE.validate_comparison(self.manifest, self.verifier, trusted, self.generation)

    def test_writer_invocation_packages_only_sampled_rows(self):
        fixture = self.write_writer_fixture()
        command = self.writer_command(fixture)
        subprocess.run(command, check=True, capture_output=True, text=True)
        record = json.loads(fixture["output"].read_text())
        self.assertFalse(record["validation"]["packaged_logits"])
        artifact_dir = fixture["output"].parent / "quality-comparison-cuda-artifacts"
        self.assertFalse((artifact_dir / "oracle.f32").exists())
        generation = json.loads(fixture["generation_record"].read_text())
        VALIDATE.validate_comparison(fixture["output"], fixture["verifier"], record["validation"]["trusted"], generation)
        for path in (fixture["oracle_logits"], fixture["llama_logits"], fixture["leone_logits"]):
            path.unlink()
        VALIDATE.validate_comparison(fixture["output"], fixture["verifier"], record["validation"]["trusted"], generation)

    def test_freeze_generation_from_retained_collection(self):
        fixture = self.write_writer_fixture()
        collection = fixture["output"].parent / "collection"
        collection.mkdir()
        source_names = {
            "tokens": ("tokens.u32le", fixture["tokens"]),
            "oracle_logits": ("oracle.f32", fixture["oracle_logits"]),
            "llama_logits": ("llama-cpp-cuda.f32", fixture["llama_logits"]),
            "leone_logits": ("leone-cuda.f32", fixture["leone_logits"]),
            "oracle_manifest": ("oracle-manifest.json", fixture["oracle_manifest"]),
            "llama_manifest": ("llama-manifest.json", fixture["llama_manifest"]),
            "leone_manifest": ("leone-manifest.json", fixture["leone_manifest"]),
            "native_build_info": ("native-build-info.json", fixture["build_info"]),
        }
        artifacts = {}
        for label, (name, source) in source_names.items():
            target = collection / name
            shutil.copyfile(source, target)
            record = {"path": name, "sha256": digest(target)}
            if label == "tokens":
                record.update(bytes=target.stat().st_size, count=4097, encoding="u32le")
            elif label.endswith("_logits"):
                record.update(bytes=target.stat().st_size, encoding="row-major-f32-le")
            artifacts[label] = record
        trusted = json.loads(fixture["trusted_inputs"].read_text())["trusted"]
        input_record = json.loads(fixture["llama_manifest"].read_text())["input"]
        input_record["tokens"]["path"] = "tokens.u32le"
        body = {
            "schema_version": "leone.quality-collection.v1",
            "input": input_record,
            "models": {
                "subject_sha256": trusted["model_sha256"],
                "oracle_sha256": trusted["oracle_model_sha256"],
                "corpus_sha256": trusted["corpus_sha256"],
            },
            "artifacts": artifacts,
            "producer_manifests": {
                "oracle": artifacts["oracle_manifest"]["sha256"],
                "llama_cpp": artifacts["llama_manifest"]["sha256"],
                "leone": artifacts["leone_manifest"]["sha256"],
            },
        }
        collection_manifest = collection / "collection.json"
        collection_manifest.write_text(json.dumps(body, sort_keys=True))
        output = fixture["output"].parent / "frozen-generation.json"
        command = [
            sys.executable,
            str(FREEZE),
            "--collection", str(collection_manifest),
            "--trusted-inputs", str(fixture["trusted_inputs"]),
            "--subject-model", str(fixture["subject_model"]),
            "--oracle-model", str(fixture["oracle_model"]),
            "--corpus", str(fixture["corpus"]),
            "--task-manifest", str(fixture["task_manifest"]),
            "--output", str(output),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(output.read_text()), json.loads(fixture["generation_record"].read_text()))

    def test_writer_accepts_distinct_native_and_statistics_identities(self):
        fixture = self.write_writer_fixture()
        native_source = "e" * 40
        native_manifest = json.loads(fixture["leone_manifest"].read_text())
        native_build = dict(native_manifest["executable"]["build_info"])
        native_build["source_commit"] = native_source
        native_manifest["source_commit"] = native_source
        native_manifest["executable"]["build_info"] = native_build
        fixture["leone_manifest"].write_text(json.dumps(native_manifest))
        trusted_body = json.loads(fixture["trusted_inputs"].read_text())
        trusted = trusted_body["trusted"]
        native_identity = trusted["producer_identities"]["leone_cuda"]
        native_identity["source_commit"] = native_source
        native_identity["manifest_sha256"] = digest(fixture["leone_manifest"])
        native_identity["build_info_sha256"] = VALIDATE.json_digest(native_build)
        statistics_target = "aarch64-unknown-linux-gnu"
        build_info = json.loads(fixture["build_info"].read_text())
        build_info["target"] = statistics_target
        fixture["build_info"].write_text(json.dumps(build_info))
        trusted["statistics_target"] = statistics_target
        trusted["statistics_identity"]["target"] = statistics_target
        trusted["statistics_platform"] = "darwin-arm64"
        fixture["statistics_platform"] = "darwin-arm64"
        trusted["statistics_identity"]["build_info_sha256"] = VALIDATE.json_digest(build_info)
        generation = json.loads(fixture["generation_record"].read_text())
        generation["producer_manifests"]["leone"] = digest(fixture["leone_manifest"])
        fixture["generation_record"].write_text(json.dumps(generation))
        trusted_body["trusted"] = trusted
        fixture["trusted_inputs"].write_text(json.dumps(trusted_body))
        fixture["statistics_target"] = statistics_target
        subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        record = json.loads(fixture["output"].read_text())
        VALIDATE.validate_comparison(fixture["output"], fixture["verifier"], record["validation"]["trusted"], generation)

    def test_writer_rejects_self_consistent_sample_shift(self):
        fixture = self.write_writer_fixture()
        subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        output = fixture["output"]
        record = json.loads(output.read_text())
        sample_dir = output.parent / "quality-comparison-cuda-samples"
        sample_body = record["samples"]["manifest"]["body"]
        for label in ("oracle", "llama_cpp", "leone"):
            sample = sample_body["artifacts"]["logits"][label]
            path = output.parent / sample["path"]
            values = struct.unpack(f"<{sample['bytes'] // 4}f", path.read_bytes())
            path.write_bytes(b"".join(struct.pack("<f", value + 1.0) for value in values))
            rows = []
            with path.open("rb") as source:
                for _ in range(sample["bytes"] // 12):
                    row = source.read(12)
                    rows.append(hashlib.sha256(row).hexdigest())
            sample["sha256"] = digest(path)
            sample["source_row_sha256"] = rows
        sample_manifest = sample_dir / "sampling.json"
        sample_manifest.write_text(json.dumps(sample_body, indent=2, sort_keys=True) + "\n")
        sample_record = record["samples"]["manifest"]
        sample_record["sha256"] = digest(sample_manifest)
        sample_record["bytes"] = sample_manifest.stat().st_size
        sample_record["body"] = sample_body
        for key in ("llama_cpp", "leone"):
            quality_record = record["quality"][key]
            quality_path = output.parent / quality_record["path"]
            quality = quality_record["receipt"]
            subject_path = sample_dir / f"{key}.f32"
            metrics = VALIDATE.recompute_quality(sample_dir / "oracle.f32", subject_path, len(sample_body["sampling"]["indices"]), 3)
            quality["oracle"]["artifact_sha256"] = digest(sample_dir / "oracle.f32")
            quality["subject"]["logits_artifact"]["sha256"] = digest(subject_path)
            quality["metrics"] = {
                "kld": {**{name: metrics[name] for name in ("mean", "p50", "p99", "max")}, "definition": VALIDATE.KLD_DEFINITION},
                "top1_agreement": metrics["top1_agreement"],
            }
            quality_path.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
            quality_record["sha256"] = digest(quality_path)
            quality_record["receipt"] = quality
        output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(ValueError, "CUDA generation .* sampled sha256 differs"):
            VALIDATE.validate_comparison(output, fixture["verifier"], record["validation"]["trusted"], json.loads(fixture["generation_record"].read_text()))

    def test_validator_rejects_changed_external_generation_record(self):
        fixture = self.write_writer_fixture()
        subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        record = json.loads(fixture["output"].read_text())
        generation = json.loads(fixture["generation_record"].read_text())
        generation["sampled_logits"]["leone"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "CUDA generation record differs from external input"):
            VALIDATE.validate_comparison(fixture["output"], fixture["verifier"], record["validation"]["trusted"], generation)

    def test_validator_cli_loads_external_trusted_tuple(self):
        fixture = self.write_writer_fixture()
        subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        trusted = json.loads(fixture["trusted_inputs"].read_text())["trusted"]
        command = [
            sys.executable,
            str(ROOT / "scripts/validate-quality-stage.py"),
            "comparison",
            str(fixture["output"]),
            str(fixture["verifier"]),
        ]
        for option, key in (
            ("--source-commit", "source_commit"),
            ("--platform", "platform"),
            ("--statistics-platform", "statistics_platform"),
            ("--target", "target"),
            ("--statistics-target", "statistics_target"),
            ("--backend", "backend"),
            ("--adapter-path", "adapter_path"),
            ("--adapter-sha256", "adapter_sha256"),
            ("--model-family", "model_family"),
            ("--model-sha256", "model_sha256"),
            ("--oracle-model-sha256", "oracle_model_sha256"),
            ("--corpus-sha256", "corpus_sha256"),
            ("--task-manifest-sha256", "task_manifest_sha256"),
            ("--sample-contract", "sample_contract"),
            ("--metric-family", "metric_family"),
        ):
            command.extend((option, trusted[key]))
        command.extend(("--trusted-inputs", str(fixture["trusted_inputs"]), "--generation-record", str(fixture["generation_record"])))
        subprocess.run(command, check=True, capture_output=True, text=True)
        wrong_anchor = subprocess.run(
            [*command, "--sample-manifest-sha256", "a" * 64],
            check=False, capture_output=True, text=True,
        )
        self.assertNotEqual(wrong_anchor.returncode, 0)
        self.assertIn("CUDA sample anchors require --generation-record", wrong_anchor.stderr)

    def test_writer_rejects_broken_output_symlink_before_mutation(self):
        fixture = self.write_writer_fixture()
        fixture["output"].symlink_to(fixture["output"].with_name("missing-output"))
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        self.assertFalse(fixture["output"].with_name("quality-comparison-cuda-artifacts").exists())

    def test_writer_rejects_quality_receipt_collision_before_mutation(self):
        fixture = self.write_writer_fixture()
        collision = fixture["output"].with_name("quality-comparison-llama-cpp-quality.json")
        collision.symlink_to(collision.with_name("missing-quality"))
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        self.assertFalse(fixture["output"].with_name("quality-comparison-cuda-artifacts").exists())

    def test_writer_rejects_logits_replaced_after_producer_run(self):
        fixture = self.write_writer_fixture()
        fixture["leone_logits"].write_bytes(fixture["leone_logits"].read_bytes() + b"changed")
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        self.assertFalse(fixture["output"].with_name("quality-comparison-cuda-artifacts").exists())

    def test_writer_does_not_replace_completed_outputs(self):
        fixture = self.write_writer_fixture()
        subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        before = fixture["output"].read_bytes()
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.run(self.writer_command(fixture), check=True, capture_output=True, text=True)
        self.assertEqual(fixture["output"].read_bytes(), before)

    @staticmethod
    def writer_command(fixture):
        return [
            sys.executable,
            str(WRITER),
            "--output",
            str(fixture["output"]),
            "--binary",
            str(fixture["binary"]),
            "--receipt-verifier",
            str(fixture["verifier"]),
            "--source-commit",
            fixture["source"],
            "--build-info",
            str(fixture["build_info"]),
            "--platform",
            "linux-x86_64",
            "--statistics-platform",
            fixture["statistics_platform"],
            "--target",
            "x86_64-unknown-linux-gnu",
            "--statistics-target",
            fixture["statistics_target"],
            "--trusted-inputs",
            str(fixture["trusted_inputs"]),
            "--generation-record",
            str(fixture["generation_record"]),
            "--model-family",
            "fixture",
            "--subject-model",
            str(fixture["subject_model"]),
            "--oracle-model",
            str(fixture["oracle_model"]),
            "--corpus",
            str(fixture["corpus"]),
            "--tokens",
            str(fixture["tokens"]),
            "--oracle-logits",
            str(fixture["oracle_logits"]),
            "--llama-logits",
            str(fixture["llama_logits"]),
            "--leone-logits",
            str(fixture["leone_logits"]),
            "--oracle-manifest",
            str(fixture["oracle_manifest"]),
            "--llama-manifest",
            str(fixture["llama_manifest"]),
            "--leone-manifest",
            str(fixture["leone_manifest"]),
            "--task-manifest",
            str(fixture["task_manifest"]),
            "--leone-backend",
            "cuda",
        ]

    def write_writer_fixture(self):
        root = self.root / "writer"
        root.mkdir()
        source = "d" * 40
        oracle_model = root / "oracle.gguf"
        subject_model = root / "subject.gguf"
        corpus = root / "corpus.txt"
        tokens = root / "tokens.u32le"
        oracle_model.write_bytes(b"oracle")
        subject_model.write_bytes(b"subject")
        corpus.write_text("fixture\n")
        token_values = list(range(4097))
        token_values[-1] = 0
        tokens.write_bytes(struct.pack("<4097I", *token_values))
        oracle_logits = root / "oracle.f32"
        llama_logits = root / "llama.f32"
        leone_logits = root / "leone.f32"
        for path, offset in ((oracle_logits, 0.0), (llama_logits, 0.01), (leone_logits, 0.02)):
            logits(path, offset, 4096)
        token_record = {
            "path": "tokens.u32le",
            "sha256": digest(tokens),
            "bytes": tokens.stat().st_size,
            "count": 4097,
            "encoding": "u32le",
        }
        input_record = {
            "tokens": token_record,
            "window_tokens": 4096,
            "stride_tokens": 4095,
            "rows": 4096,
            "vocab_size": 3,
        }
        oracle_model_record = {
            "path": "oracle.gguf",
            "sha256": digest(oracle_model),
            "storage_type": "BF16",
            "vocab_size": 3,
            "architecture": "fixture",
            "tokenizer": "fixture",
        }
        subject_model_record = {
            "path": "subject.gguf",
            "sha256": digest(subject_model),
            "storage_type": "Q4_K - Medium",
            "vocab_size": 3,
            "architecture": "fixture",
            "tokenizer": "fixture",
        }
        build_info = {
            "schema_version": "leone.build-info.v1",
            "source_commit": source,
            "source_tree_dirty": False,
            "profile": "release",
            "target": "x86_64-unknown-linux-gnu",
        }
        pin = (ROOT / "external/PINNED").read_text().strip()
        adapter_sha = digest(ROOT / "research/oracle/llama_logits.cpp")
        adapter = {"path": "research/oracle/llama_logits.cpp", "sha256": adapter_sha}
        oracle_run = {
            "schema_version": "leone.llama-oracle.v2",
            "source_commit": "a" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": adapter,
            "executable": {"path": "llama-oracle", "sha256": "b" * 64, "linked_libraries": []},
            "model": oracle_model_record,
            "input": input_record,
            "execution": execution("cpu", "cpu", "cpu"),
            "logits": artifact_record(oracle_logits, "oracle.f32"),
        }
        llama_run = {
            "schema_version": "leone.llama-oracle.v2",
            "source_commit": "b" * 40,
            "engine": {"name": "llama.cpp", "git_commit": pin},
            "adapter": adapter,
            "executable": {"path": "llama-cuda", "sha256": "c" * 64, "linked_libraries": []},
            "model": subject_model_record,
            "input": input_record,
            "execution": execution("cuda", "cuda", "gpu"),
            "logits": artifact_record(llama_logits, "llama.f32"),
        }
        executable = root / "fake-leone"
        executable.write_text(
            "#!/usr/bin/env python3\n"
            "import hashlib,json,math,struct,sys\n"
            "from pathlib import Path\n"
            "def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()\n"
            "a=Path(sys.argv[sys.argv.index('--oracle')+1]); b=Path(sys.argv[sys.argv.index('--subject')+1])\n"
            "rows=a.stat().st_size//12\n"
            "values=[]; matches=0\n"
            "with a.open('rb') as x, b.open('rb') as y:\n"
            "  for _ in range(rows):\n"
            "    av=struct.unpack('<3f',x.read(12)); bv=struct.unpack('<3f',y.read(12));\n"
            "    ma=max(av); mb=max(bv); ap=[math.exp(v-ma) for v in av]; bp=[math.exp(v-mb) for v in bv]; sa=sum(ap); sb=sum(bp); values.append(sum((p/sa)*math.log((p/sa)/(q/sb)) for p,q in zip(ap,bp))); matches += int(av.index(max(av)) == bv.index(max(bv)))\n"
            "  values.sort()\n"
            "  metric={'mean':sum(values)/rows,'p50':values[(rows-1)//2],'p99':values[-1],'max':values[-1],'top1_agreement':matches/rows}\n"
            "  model=Path(sys.argv[sys.argv.index('--subject-model')+1]); corpus=Path(sys.argv[sys.argv.index('--corpus')+1]);\n"
            "  body={'schema_version':3,'receipt_id':'00000000-0000-4000-8000-000000000001','created_utc':'2026-09-20T00:00:00Z','corpus':{'name':corpus.name,'sha256':h(corpus),'n_prompts':1,'n_tokens_scored':rows},'oracle':{'description':'llama.cpp BF16 execution','artifact_sha256':h(a),'engine':{'name':'llama.cpp','git_commit':sys.argv[sys.argv.index('--oracle-commit')+1]},'dtype':'bf16'},'subject':{'model_artifact':{'sha256':h(model),'path':model.name},'logits_artifact':{'sha256':h(b),'path':b.name},'engine':{'name':sys.argv[sys.argv.index('--subject-engine')+1],'git_commit':sys.argv[sys.argv.index('--subject-commit')+1]}},'sample_count':rows,'metrics':{'kld':{k:metric[k] for k in ('mean','p50','p99','max')} | {'definition':'mean over scored positions of KL(P_oracle || P_subject) in nats, full softmax over full vocab'},'top1_agreement':metric['top1_agreement']}}\n"
            "  Path('receipt.json').write_text(json.dumps(body)); print('receipt: receipt.json')\n"
        )
        executable.chmod(0o755)
        native_run = {
            "schema_version": "leone.native-eval.v1",
            "source_commit": source,
            "executable": {
                "name": executable.name,
                "sha256": digest(executable),
                "build_info": build_info,
            },
            "model": subject_model_record,
            "input": input_record,
            "execution": {
                **execution("cuda", "cuda", "gpu"),
                "engine": "leone",
                "kv_cache_dtype": "F16",
                "prefill": {"path": "chunked", "chunk_tokens": 2},
            },
            "logits": artifact_record(leone_logits, "leone.f32"),
        }
        task_body = {
            "schema_version": "leone.quality-task.v2",
            "task": {
                "name": "needle-exact-answer",
                "context_tokens": 4096,
                "minimum_context_tokens": 4096,
                "prompt": {"encoding": "utf-8", "text": "fix", "bytes": 3, "sha256": hashlib.sha256(b"fix").hexdigest()},
                "needle": {"encoding": "utf-8", "text": "ture\n", "bytes": 5, "sha256": hashlib.sha256(b"ture\n").hexdigest()},
                "answer": {"row": 4095, "text": "ture\n", "sha256": hashlib.sha256(b"ture\n").hexdigest(), "token_ids": [0]},
            },
            "tokenizer": {"name": "fixture"},
            "tokenization": {
                "source": {"name": "llama.cpp", "git_commit": pin},
                "model": {"sha256": oracle_model_record["sha256"]},
                "method": "llama-tokenize",
                "combined_sha256": digest(corpus),
                "input_tokens_sha256": token_record["sha256"],
                "answer_tokens_sha256": hashlib.sha256(struct.pack("<I", 0)).hexdigest(),
            },
            "model": {
                "family": "fixture",
                "oracle": {"sha256": oracle_model_record["sha256"]},
                "subject": {"sha256": subject_model_record["sha256"]},
            },
            "corpus": {"sha256": digest(corpus)},
            "input": input_record,
        }
        task_path = root / "task.json"
        task_path.write_text(json.dumps(task_body))
        paths = {}
        for name, body in (("oracle", oracle_run), ("llama", llama_run), ("leone", native_run)):
            path = root / f"{name}.json"
            path.write_text(json.dumps(body))
            paths[name] = path
        build_path = root / "build-info.json"
        build_path.write_text(json.dumps(build_info))
        verifier = root / "leone-receipt-verify"
        verifier.write_text("#!/bin/sh\n[ \"$1\" = quality ] && [ -f \"$2\" ]\n")
        verifier.chmod(0o755)
        producer_records = {}
        for name, path, run in (
            ("oracle_stage", paths["oracle"], oracle_run),
            ("llama_cpp_cuda", paths["llama"], llama_run),
            ("leone_cuda", paths["leone"], native_run),
        ):
            executable_record = run["executable"]
            executable_name = executable_record.get("name") or Path(executable_record["path"]).name
            identity = {
                "source_commit": run["source_commit"],
                "manifest_sha256": digest(path),
                "executable": {"name": executable_name, "sha256": executable_record["sha256"]},
            }
            if name == "leone_cuda":
                identity["build_info_sha256"] = VALIDATE.json_digest(build_info)
                identity["target"] = "x86_64-unknown-linux-gnu"
            if name == "llama_cpp_cuda":
                identity["external_pin"] = pin
            producer_records[name] = identity
        full = {
            "oracle": artifact_record(oracle_logits, "generated/oracle.f32"),
            "llama_cpp": artifact_record(llama_logits, "generated/llama-cpp-cuda.f32"),
            "leone": artifact_record(leone_logits, "generated/leone-cuda.f32"),
        }
        generation = generation_record(
            {"oracle": oracle_logits, "llama_cpp": llama_logits, "leone": leone_logits},
            full,
            input_record,
            {"oracle": oracle_model_record, "subject": subject_model_record},
            corpus,
            task_path,
            {"oracle": paths["oracle"], "llama_cpp": paths["llama"], "leone": paths["leone"]},
            [4095],
        )
        generation_path = root / "generation-record.json"
        generation_path.write_text(json.dumps(generation))
        trusted = {
            "source_commit": source,
            "platform": "linux-x86_64",
            "statistics_platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "statistics_target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
            "adapter_path": "research/oracle/llama_logits.cpp",
            "adapter_sha256": adapter_sha,
            "model_family": "fixture",
            "model_sha256": subject_model_record["sha256"],
            "oracle_model_sha256": oracle_model_record["sha256"],
            "corpus_sha256": digest(corpus),
            "task_manifest_sha256": digest(task_path),
            "sample_contract": VALIDATE.SAMPLE_CONTRACT,
            "metric_family": "kld",
            "producer_identities": producer_records,
            "statistics_identity": {
                "source_commit": source,
                "executable": {"name": executable.name, "sha256": digest(executable)},
                "build_info_sha256": VALIDATE.json_digest(build_info),
                "target": "x86_64-unknown-linux-gnu",
            },
            "native_workflow": {
                "kv_cache_dtype": "F16",
                "prefill": {"path": "chunked", "chunk_tokens": 2},
                "token_count": input_record["tokens"]["count"],
            },
        }
        trusted_path = root / "trusted-inputs.json"
        trusted_path.write_text(json.dumps({"schema_version": "leone.quality-trusted-cuda.v1", "trusted": trusted}))
        return {
            "output": root / "quality-comparison.json",
            "binary": executable,
            "verifier": verifier,
            "source": source,
            "build_info": build_path,
            "subject_model": subject_model,
            "oracle_model": oracle_model,
            "corpus": corpus,
            "tokens": tokens,
            "oracle_logits": oracle_logits,
            "llama_logits": llama_logits,
            "leone_logits": leone_logits,
            "oracle_manifest": paths["oracle"],
            "llama_manifest": paths["llama"],
            "leone_manifest": paths["leone"],
            "task_manifest": task_path,
            "trusted_inputs": trusted_path,
            "generation_record": generation_path,
            "statistics_platform": "linux-x86_64",
            "statistics_target": "x86_64-unknown-linux-gnu",
        }


if __name__ == "__main__":
    unittest.main()
