"""Test archive verification and offline evidence boundaries."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
V03_CHECKERS = (
    "scripts/check-release-evidence.py",
    "scripts/release_evidence_manifest.py",
    "scripts/source_inputs.py",
    "scripts/study-concurrent-service.py",
    "scripts/check-public-tree.sh",
    "scripts/check-public-tree.py",
)


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VERIFY = load("verify_release_archive", ROOT / "scripts/verify-release-archive.py")
PUBLIC = load("check_public_tree", ROOT / "scripts/check-public-tree.py")
STUDY = load("study_concurrent_service", ROOT / "scripts/study-concurrent-service.py")
EVIDENCE = load("release_evidence_manifest", ROOT / "scripts/release_evidence_manifest.py")
CHECK = load("check_release_evidence", ROOT / "scripts/check-release-evidence.py")
DISPATCH = load("release_evidence_validators", ROOT / "scripts/release_evidence_validators.py")
WRITE_MANIFEST = load("write_manifest", ROOT / "packaging/write-manifest.py")
MAKE_ARCHIVE = load("make_archive", ROOT / "packaging/make-archive.py")


class ReleasePackagingTests(unittest.TestCase):
    @staticmethod
    def _write_package_fixture(root: Path, kind: str = "runtime") -> None:
        metadata = {
            "schema_version": "leone.package.v1",
            "package": kind,
            "version": "fixture",
            "platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
        }
        environment = {
            "schema_version": "leone.package-environment.v1",
            "package": kind,
            "platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
        }
        root.mkdir(parents=True, exist_ok=True)
        (root / "package-info.json").write_text(
            json.dumps(metadata) + "\n", encoding="utf-8"
        )
        (root / "environment.json").write_text(
            json.dumps(environment) + "\n", encoding="utf-8"
        )
        if kind == "runtime":
            (root / "bin").mkdir(exist_ok=True)
            (root / "bin/leone").write_text("#!/bin/sh\n", encoding="utf-8")
            (root / "install.sh").write_text("#!/bin/sh\n", encoding="utf-8")
            (root / "bin/leone").chmod(0o755)
            (root / "install.sh").chmod(0o755)
            markdown = (
                "README.md",
                "docs/branching-service-evidence.md",
                "docs/client-workflow.md",
                "docs/concurrent-service-evidence.md",
                "docs/memory-accounting.md",
                "docs/models.md",
                "docs/openai-api.md",
                "docs/oracle.md",
                "docs/release-candidate.md",
                "docs/release-evidence.md",
                "docs/release.md",
                "receipts/INDEX.md",
            )
            for relative in markdown:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture\n", encoding="utf-8")
            return
        (root / "README.md").write_text("evidence fixture\n", encoding="utf-8")
        (root / "release-evidence.json").write_bytes(
            (ROOT / "packaging/release-evidence.v0.3.json").read_bytes()
        )
        (root / "receipts").mkdir(exist_ok=True)
        (root / "receipts/INDEX.md").write_text("fixture\n", encoding="utf-8")
        (root / "external").mkdir(exist_ok=True)
        (root / "external/PINNED").write_text("fixture\n", encoding="utf-8")
        required_json = (
            "receipts/concurrent-service-study.json",
            "receipts/quality-concurrent-service.json",
            "receipts/quality-llama-v03.json",
            "receipts/openai-client-check.json",
            "receipts/source-inputs.json",
        )
        for relative in required_json:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        scripts = root / "scripts"
        scripts.mkdir(exist_ok=True)
        for relative in V03_CHECKERS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".sh":
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)
            else:
                path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")

    @staticmethod
    def _repack(directory: Path, root: Path, name: str = "fixture") -> Path:
        rows = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != root / "MANIFEST.sha256":
                relative = path.relative_to(root).as_posix()
                rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{relative}")
        (root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
        return ReleasePackagingTests._archive_existing_manifest(directory, root, name)

    @staticmethod
    def _archive_existing_manifest(directory: Path, root: Path, name: str) -> Path:
        archive = directory / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(root, arcname=root.name)
        archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (directory / f"{name}.tar.gz.sha256").write_text(
            f"{archive_digest}  {archive.name}\n", encoding="utf-8"
        )
        return archive

    @classmethod
    def _archive(
        cls, directory: Path, root: Path, name: str = "fixture", kind: str = "runtime"
    ) -> Path:
        cls._write_package_fixture(root, kind)
        return cls._repack(directory, root, name)

    @staticmethod
    def _trusted_stub(directory: Path, evidence: bool = True) -> Path:
        packaging = directory / "packaging"
        packaging.mkdir(parents=True, exist_ok=True)
        (packaging / "release-evidence.v0.3.json").write_bytes(
            (ROOT / "packaging/release-evidence.v0.3.json").read_bytes()
        )
        scripts = directory / "scripts"
        scripts.mkdir(parents=True)
        for relative in V03_CHECKERS:
            if not evidence and relative not in {"scripts/check-public-tree.sh", "scripts/check-public-tree.py"}:
                continue
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".sh":
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)
            else:
                path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
        return directory

    @staticmethod
    def _complete_v04_skeleton(root: Path) -> Path:
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["status"] = "complete"
        (root / "scripts").mkdir(parents=True)
        (root / "receipts").mkdir()
        binary = root / "bin/leone"
        binary.parent.mkdir(parents=True)
        binary.write_text("fixture binary\n", encoding="utf-8")
        binary.chmod(0o755)
        ReleasePackagingTests._prepare_complete_quality_records(manifest)
        binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        for record in ReleasePackagingTests._v04_records(manifest):
            if record.get("role") in {"client", "batching", "branching"}:
                record["binary_sha256"] = binary_sha256
        records = ReleasePackagingTests._v04_records(manifest)
        ReleasePackagingTests._write_generation_records(root, records)
        ReleasePackagingTests._write_v04_sources(root, manifest, records)
        for record in records:
            for relative in record.get("artifact_files", []):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("artifact\n", encoding="utf-8")
        ReleasePackagingTests._write_v04_records(root, records)
        manifest_path = root / "release-evidence.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return manifest_path

    @staticmethod
    def _prepare_complete_quality_records(manifest: dict) -> None:
        for backend in manifest["backend_requirements"]:
            for record in backend["records"]:
                if record.get("role") != "quality":
                    continue
                record["task_manifest_sha256"] = "f" * 64
                if backend["backend"] == "metal":
                    record["sample_manifest_sha256"] = "e" * 64
                    record["native_source_commit"] = "a" * 40

    @staticmethod
    def _write_generation_records(root: Path, records: list[dict]) -> None:
        for record in records:
            generation = record.get("generation_record")
            if generation is None:
                continue
            path = root / generation
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"schema_version": "leone.quality-generation.v1"})
                + "\n",
                encoding="utf-8",
            )
            record["generation_record_sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    @staticmethod
    def _v04_records(manifest: dict) -> list[dict]:
        records = [
            record
            for backend in manifest["backend_requirements"]
            for record in backend["records"]
        ]
        records.extend(manifest["shared_records"])
        return records

    @staticmethod
    def _write_v04_sources(root: Path, manifest: dict, records: list[dict]) -> None:
        for relative in manifest["trusted_validators"]:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("trusted validator\n", encoding="utf-8")
        record_paths = {record["path"] for record in records}
        source_names = set(manifest["trusted_validators"])
        source_names.update(("scripts/check-public-tree.sh", "scripts/check-public-tree.py"))
        for record in records:
            source_names.update(
                dependency
                for dependency in record["dependencies"]
                if dependency != manifest["source_manifest"] and dependency not in record_paths
            )
        source_files = {}
        for relative in sorted(source_names):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text("dependency\n", encoding="utf-8")
            source_files[relative] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        source_body = {"source_commit": "a" * 40, "files": source_files}
        (root / manifest["source_manifest"]).write_text(
            json.dumps(source_body) + "\n", encoding="utf-8"
        )

    @staticmethod
    def _write_v04_records(root: Path, records: list[dict]) -> None:
        for record in records:
            path = root / record["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"schema_version": record["schema_version"]}) + "\n",
                encoding="utf-8",
            )
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _dispatch_records(manifest: dict) -> tuple[dict[str, dict], list[dict]]:
        raw_records = {
            record["path"]: record
            for backend in manifest["backend_requirements"]
            for record in backend["records"]
        }
        raw_records.update(
            {record["path"]: record for record in manifest["shared_records"]}
        )
        records = []
        for backend in manifest["backend_requirements"]:
            for raw in backend["records"]:
                record = dict(raw)
                record.update(
                    {
                        "platform": backend["platform"],
                        "target": backend["target"],
                        "backend": backend["backend"],
                    }
                )
                records.append(record)
        records.extend(dict(record) for record in manifest["shared_records"])
        return raw_records, records

    @staticmethod
    def _write_dispatch_services(
        root: Path, records: list[dict], raw_records: dict[str, dict], source_commit: str
    ) -> None:
        for record in records:
            if record.get("role") not in {"service", "metrics"}:
                continue
            path = root / record["path"]
            path.write_text(
                json.dumps(
                    {
                        "model": {"sha256": record["model_sha256"]},
                        "source": {
                            "commit": source_commit,
                            "tracked_tree_clean": True,
                        },
                        "binaries": {
                            "leone": {
                                "build_info": {
                                    "value": {
                                        "source_commit": source_commit,
                                        "target": record["target"],
                                        "source_tree_dirty": False,
                                        "profile": "release",
                                    }
                                }
                            }
                        },
                        "runs": [
                            {
                                "launch_argv": [
                                    "leone",
                                    "--backend",
                                    record["backend"],
                                ]
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            raw_records[record["path"]]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    @staticmethod
    def _write_dispatch_trusted_inputs(
        root: Path, records: list[dict], adapter_sha: str, source_commit: str
    ) -> None:
        for record in records:
            if record.get("role") != "quality" or record["backend"] != "cuda":
                continue
            trusted = {
                "source_commit": source_commit,
                "platform": record["platform"],
                "target": record["target"],
                "statistics_target": record["statistics_target"],
                "statistics_platform": record["statistics_platform"],
                "backend": record["backend"],
                "adapter_path": "research/oracle/llama_logits.cpp",
                "adapter_sha256": adapter_sha,
                "model_family": record["model_family"],
                "model_sha256": record["model_sha256"],
                "oracle_model_sha256": record["oracle_model_sha256"],
                "corpus_sha256": record["corpus_sha256"],
                "task_manifest_sha256": record["task_manifest_sha256"],
                "sample_contract": record["sample_contract"],
                "metric_family": record["metric_family"],
            }
            path = root / record["trusted_inputs"]
            path.write_text(
                json.dumps(
                    {"schema_version": "leone.quality-trusted-cuda.v1", "trusted": trusted}
                )
                + "\n",
                encoding="utf-8",
            )

    @staticmethod
    def _write_dispatch_checkers(root: Path) -> Path:
        stage = root / "scripts/validate-quality-stage.py"
        stage.write_text(
            """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
if not args or args[0] != "comparison":
    raise SystemExit(2)
values = dict(zip(args[3::2], args[4::2]))
if "--trusted-inputs" not in values:
    raise SystemExit(0)
trusted = json.loads(pathlib.Path(values["--trusted-inputs"]).read_text())["trusted"]
for key, option in {
    "source_commit": "--source-commit",
    "platform": "--platform",
    "target": "--target",
    "statistics_target": "--statistics-target",
    "backend": "--backend",
    "adapter_path": "--adapter-path",
    "adapter_sha256": "--adapter-sha256",
    "model_family": "--model-family",
    "model_sha256": "--model-sha256",
    "oracle_model_sha256": "--oracle-model-sha256",
    "corpus_sha256": "--corpus-sha256",
    "task_manifest_sha256": "--task-manifest-sha256",
    "sample_contract": "--sample-contract",
    "metric_family": "--metric-family",
}.items():
    if values.get(option) != trusted.get(key):
        raise SystemExit(3)
""",
            encoding="utf-8",
        )
        for relative in (
            "scripts/study-concurrent-service.py",
            "scripts/check-release-evidence.py",
            "scripts/check-batched-service.py",
        ):
            (root / relative).write_text(
                "#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8"
            )
        verifier = root / "leone-receipt-verify"
        verifier.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        verifier.chmod(0o755)
        return verifier

    @staticmethod
    def _refresh_dispatch_sources(root: Path, manifest: dict) -> None:
        source_path = root / manifest["source_manifest"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        for relative, value in source["files"].items():
            path = root / relative
            value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            value["executable"] = bool(path.stat().st_mode & 0o111)
        source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")

    @staticmethod
    def _stub_scope(records: list[dict]) -> list[dict]:
        """The stub checkers stand in for the quality and service validators only.

        A branching record binds one exact canonical quality record and reruns the
        study harness, so it is dispatched with real records in
        `test_branching_package`, never against these stub checkers.
        """

        return [record for record in records if record.get("role") != "branching"]

    @classmethod
    def _complete_dispatch_fixture(cls, root: Path) -> tuple[Path, Path]:
        manifest_path = cls._complete_v04_skeleton(root)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_records, records = cls._dispatch_records(manifest)
        source_commit = "a" * 40
        adapter = root / "research/oracle/llama_logits.cpp"
        adapter_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
        cls._write_dispatch_services(root, records, raw_records, source_commit)
        cls._write_dispatch_trusted_inputs(root, records, adapter_sha, source_commit)
        verifier = cls._write_dispatch_checkers(root)
        cls._refresh_dispatch_sources(root, manifest)
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        return manifest_path, verifier

    def test_offline_receipt_check_does_not_need_local_inputs(self):
        receipt = ROOT / "receipts/concurrent-service-study.json"
        errors = STUDY.validate_recorded_receipt(receipt, ROOT, offline=True)
        self.assertEqual(errors, [])

    def test_v04_manifest_requires_both_backend_records(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        records = EVIDENCE.validate(manifest)
        self.assertEqual(len(records), 14)
        self.assertFalse(
            any(entry["destination"] == "bin/leone" for entry in manifest["files"])
        )
        self.assertEqual(
            {
                (record["platform"], record["target"], record["backend"])
                for record in manifest["shared_records"]
                if record["role"] == "client"
            },
            {
                ("linux-x86_64", "x86_64-unknown-linux-gnu", "cuda"),
                ("darwin-arm64", "aarch64-apple-darwin", "metal"),
            },
        )
        broken = json.loads(json.dumps(manifest))
        broken["backend_requirements"].pop()
        with self.assertRaisesRegex(ValueError, "Linux CUDA and Darwin Metal"):
            EVIDENCE.validate(broken)

    def test_v04_manifest_rejects_f32_outside_quality_samples(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        extra = {
            "source": "receipts/v04-service-metrics.json",
            "destination": "receipts/unrelated-full-matrix.f32",
        }
        candidate = json.loads(json.dumps(manifest))
        candidate["files"].append(extra)
        with self.assertRaisesRegex(ValueError, "outside a declared quality sample"):
            EVIDENCE.validate(candidate)

        candidate = json.loads(json.dumps(manifest))
        quality = candidate["backend_requirements"][0]["records"][1]
        extra["destination"] = quality["artifact_roots"][0] + "/full-matrix.f32"
        quality["artifact_files"].append(extra["destination"])
        candidate["files"].append(extra)
        with self.assertRaisesRegex(ValueError, "bounded contract"):
            EVIDENCE.validate(candidate)

    def test_v04_manifest_requires_the_exact_cuda_sample_root(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        foreign = "receipts/foreign-samples"
        names = ("oracle.f32", "llama_cpp.f32", "leone.f32")

        def declare(candidate: dict, record: dict, root: str, files: list[str]) -> None:
            record["artifact_roots"].append(root)
            record["artifact_files"].extend(files)
            candidate["files"].extend({"source": path, "destination": path} for path in files)

        for backend, index in (("cuda", 1), ("metal", 1)):
            candidate = json.loads(json.dumps(manifest))
            record = candidate["backend_requirements"][
                0 if backend == "cuda" else 1
            ]["records"][index]
            declare(candidate, record, foreign, [f"{foreign}/{name}" for name in names])
            with self.subTest(backend):
                with self.assertRaisesRegex(ValueError, "bounded contract"):
                    EVIDENCE.validate(candidate)

    def test_v04_manifest_requires_every_cuda_sample_file(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        candidate = json.loads(json.dumps(manifest))
        record = candidate["backend_requirements"][0]["records"][1]
        dropped = next(path for path in record["artifact_files"] if path.endswith("/leone.f32"))
        record["artifact_files"].remove(dropped)
        candidate["files"] = [entry for entry in candidate["files"] if entry["destination"] != dropped]
        with self.assertRaisesRegex(ValueError, "bounded contract"):
            EVIDENCE.validate(candidate)

    def test_public_tree_allows_only_declared_quality_sample_f32(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        expected = {
            entry["destination"]
            for entry in manifest["files"]
            if entry["destination"].endswith(".f32")
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "release-evidence.json"
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            allowed = PUBLIC.quality_sample_f32_paths(root, True, "evidence")
            self.assertEqual(allowed, expected)
            self.assertEqual(len(allowed), 6)
            for sample in allowed:
                self.assertFalse(PUBLIC.forbidden(sample, allowed))
                self.assertTrue(PUBLIC.forbidden(sample.upper(), allowed))
            self.assertTrue(PUBLIC.forbidden("receipts/unrelated-full-matrix.f32", allowed))
            self.assertTrue(PUBLIC.forbidden(str(PurePosixPath(sample).parent / "subject.f32"), allowed))

            foreign = "receipts/foreign-cuda-samples"
            widened = json.loads(json.dumps(manifest))
            record = widened["backend_requirements"][0]["records"][1]
            record["artifact_roots"].append(foreign)
            record["artifact_files"].append(f"{foreign}/oracle.f32")
            widened["files"].append(
                {"source": f"{foreign}/oracle.f32", "destination": f"{foreign}/oracle.f32"}
            )
            metal = widened["backend_requirements"][1]["records"][1]
            metal_root = next(item for item in metal["artifact_roots"] if item.endswith("-samples"))
            metal["artifact_files"].append(f"{metal_root}/oracle.f32")
            widened["files"].append(
                {"source": f"{metal_root}/oracle.f32", "destination": f"{metal_root}/oracle.f32"}
            )
            manifest_path.write_text(json.dumps(widened) + "\n", encoding="utf-8")
            self.assertEqual(PUBLIC.quality_sample_f32_paths(root, True, "evidence"), expected)

    def test_public_tree_rejects_f32_in_runtime_archive(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runtime"
            root.mkdir()
            (root / "release-evidence.json").write_text(
                json.dumps(manifest) + "\n", encoding="utf-8"
            )
            (root / "unexpected.F32").write_bytes(b"fixture")
            allowed = PUBLIC.quality_sample_f32_paths(root, True, "runtime")
            _, errors = PUBLIC.archive_entries(root, allowed)
        self.assertEqual(allowed, frozenset())
        self.assertEqual(errors, ["unexpected.F32"])

    def test_manifest_rejects_uppercase_f32_outside_quality_samples(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        candidate = json.loads(json.dumps(manifest))
        candidate["files"].append(
            {
                "source": "receipts/v04-service-metrics.json",
                "destination": "receipts/unrelated-full-matrix.F32",
            }
        )
        with self.assertRaisesRegex(ValueError, "outside a declared quality sample"):
            EVIDENCE.validate(candidate)

    def test_v04_manifest_stays_incomplete_until_records_exist(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        with self.assertRaisesRegex(ValueError, "not complete"):
            EVIDENCE.validate(manifest, require_complete=True)

    def test_v04_complete_requires_reviewed_model_digest(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest["status"] = "complete"
        del manifest["backend_requirements"][0]["records"][0]["model_sha256"]
        with self.assertRaisesRegex(ValueError, "has no model SHA-256"):
            EVIDENCE.validate(manifest, require_complete=True)

    def test_v04_complete_cuda_quality_requires_task_digest(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest["status"] = "complete"
        with self.assertRaisesRegex(ValueError, "task_manifest_sha256"):
            EVIDENCE.validate(manifest, require_complete=True)

    def test_complete_cuda_quality_requires_collection_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = self._complete_dispatch_fixture(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for field, message in (
                ("statistics_platform", "statistics_platform"),
                ("generation_record", "generation_record"),
                ("generation_record_sha256", "generation_record_sha256"),
            ):
                candidate = json.loads(json.dumps(manifest))
                del candidate["backend_requirements"][0]["records"][1][field]
                with self.assertRaisesRegex(ValueError, message):
                    EVIDENCE.validate(candidate, require_complete=True)

    def test_complete_dispatch_binds_external_cuda_tuple(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, verifier = self._complete_dispatch_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = self._stub_scope(EVIDENCE.validate(manifest, require_complete=True))
            DISPATCH.validate_records(root, manifest, records, root, verifier)

            changed_tuple = json.loads(json.dumps(manifest))
            changed_tuple["backend_requirements"][0]["records"][1][
                "task_manifest_sha256"
            ] = "0" * 64
            changed_records = self._stub_scope(EVIDENCE.validate(changed_tuple, require_complete=True))
            with self.assertRaisesRegex(ValueError, "trusted evidence validator failed"):
                DISPATCH.validate_records(root, changed_tuple, changed_records, root, verifier)

            changed_model = json.loads(json.dumps(manifest))
            changed_model["backend_requirements"][0]["records"][1]["model_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "unreviewed model SHA-256"):
                EVIDENCE.validate(changed_model, require_complete=True)

            changed_generation = json.loads(json.dumps(manifest))
            changed_generation["backend_requirements"][0]["records"][1][
                "generation_record_sha256"
            ] = "0" * 64
            changed_records = self._stub_scope(EVIDENCE.validate(changed_generation, require_complete=True))
            with self.assertRaisesRegex(ValueError, "generation record differs"):
                DISPATCH.validate_records(
                    root, changed_generation, changed_records, root, verifier
                )

    def test_complete_batching_record_binds_the_cuda_client_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._complete_v04_skeleton(Path(temporary))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            EVIDENCE.validate(manifest, require_complete=True)
            for label, mutate, message in (
                ("no binary", lambda record: record.pop("binary_sha256"), "no native binary SHA-256"),
                ("other binary", lambda record: record.update(binary_sha256="0" * 64), "differs from the CUDA client binary"),
            ):
                with self.subTest(label):
                    candidate = json.loads(json.dumps(manifest))
                    mutate(next(
                        record for record in candidate["backend_requirements"][0]["records"]
                        if record["role"] == "batching"
                    ))
                    with self.assertRaisesRegex(ValueError, message):
                        EVIDENCE.validate(candidate, require_complete=True)

    def _check_extracted_public_gate(self, extracted: Path, scratch: Path) -> None:
        for name in ("check-public-tree.sh", "check-public-tree.py"):
            self.assertEqual(
                (extracted / "scripts" / name).read_bytes(),
                (ROOT / "scripts" / name).read_bytes(),
            )
        gate = extracted / "scripts/check-public-tree.sh"

        def run(tree: Path) -> subprocess.CompletedProcess:
            return subprocess.run(
                [str(gate), "--archive-kind", "evidence", str(tree)],
                capture_output=True,
                text=True,
            )

        self.assertEqual(run(extracted).returncode, 0)
        sample = next(extracted.rglob("oracle.f32")).parent.relative_to(extracted)

        def add(relative: str):
            def mutate(tree: Path) -> None:
                (tree / relative).parent.mkdir(parents=True, exist_ok=True)
                (tree / relative).write_bytes(b"x")

            return relative, mutate

        def link(tree: Path) -> None:
            (tree / sample / "leone.f32").unlink()
            (tree / sample / "leone.f32").symlink_to("oracle.f32")

        cases = {
            "full matrix": add("receipts/full-matrix.f32"),
            "uppercase sample": add(f"{sample}/ORACLE.F32"),
            "unlisted sample name": add(f"{sample}/subject.f32"),
            "foreign sample root": add("receipts/other-cuda-samples/oracle.f32"),
            "symlinked sample": (f"{sample}/leone.f32", link),
        }
        for label, (relative, mutate) in cases.items():
            with self.subTest(label):
                tree = scratch / label.replace(" ", "-")
                shutil.copytree(extracted, tree, symlinks=True)
                mutate(tree)
                result = run(tree)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(relative, result.stderr + result.stdout)

    def test_fixture_archive_boundary_and_dispatch_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            manifest_path, verifier = self._complete_dispatch_fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for entry in manifest["files"]:
                source = root / entry["source"]
                if not source.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    if source.name == "release-evidence.v0.4.json":
                        source.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
                    else:
                        source.write_text("fixture source\n", encoding="utf-8")
            for name in ("check-public-tree.sh", "check-public-tree.py"):
                shutil.copy2(ROOT / "scripts" / name, root / "scripts" / name)
            self._refresh_dispatch_sources(root, manifest)
            package = Path(temporary) / "evidence"
            EVIDENCE.stage(manifest, root, package)
            self.assertFalse((package / "bin/leone").exists())
            self.assertEqual(len(list(package.rglob("*.f32"))), 6)
            (package / "README.md").write_text("fixture evidence\n", encoding="utf-8")
            (package / "package-info.json").write_text(
                json.dumps({
                    "schema_version": "leone.package.v1",
                    "package": "evidence",
                    "version": "fixture",
                    "platform": "linux-x86_64",
                    "target": "x86_64-unknown-linux-gnu",
                    "backend": "cuda",
                })
                + "\n",
                encoding="utf-8",
            )
            (package / "environment.json").write_text(
                json.dumps({
                    "schema_version": "leone.package-environment.v1",
                    "package": "evidence",
                    "platform": "linux-x86_64",
                    "target": "x86_64-unknown-linux-gnu",
                    "backend": "cuda",
                })
                + "\n",
                encoding="utf-8",
            )
            shutil.copy2(verifier, package / verifier.name)
            WRITE_MANIFEST.write_manifest(package, package / "MANIFEST.sha256")
            archive = Path(temporary) / "evidence.tar.gz"
            MAKE_ARCHIVE.write_archive(Path(temporary), "evidence", archive, 1_700_000_000)
            archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
            archive.with_name("evidence.tar.gz.sha256").write_text(
                f"{archive_hash}  evidence.tar.gz\n", encoding="utf-8"
            )
            VERIFY.verify_archive(
                archive,
                checker_root=root,
                receipt_validator=package / verifier.name,
            )
            extracted = VERIFY.extract(archive, Path(temporary) / "extract")
            self._check_extracted_public_gate(extracted, Path(temporary) / "mutations")
            extracted_manifest = json.loads((extracted / "release-evidence.json").read_text())
            records = self._stub_scope(EVIDENCE.validate(extracted_manifest, require_complete=True))
            DISPATCH.validate_records(
                extracted,
                extracted_manifest,
                records,
                extracted,
                extracted / verifier.name,
            )

    def test_v04_quality_requires_the_kld_metric_family(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        del manifest["backend_requirements"][0]["records"][1]["metric_family"]
        with self.assertRaisesRegex(ValueError, "must bind kld"):
            EVIDENCE.validate(manifest)

    def test_v04_client_slot_requires_api_v2_evidence(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest["shared_records"][0]["schema_version"] = "leone.openai-client-check.v1"
        with self.assertRaisesRegex(ValueError, "v2 API evidence schema"):
            EVIDENCE.validate(manifest)

    def test_manifest_stages_only_declared_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            destination = directory / "destination"
            source.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            (source / "record.json").write_text(
                '{"schema_version": "fixture"}\n', encoding="utf-8"
            )
            (source / "extra.txt").write_text("unlisted\n", encoding="utf-8")
            manifest = {
                "schema_version": "leone.release-evidence.v1",
                "release_line": "v0.3",
                "package_kind": "evidence",
                "status": "complete",
                "backend_requirements": [{
                    "id": "linux-x86_64-cuda",
                    "platform": "linux-x86_64",
                    "target": "x86_64-unknown-linux-gnu",
                    "backend": "cuda",
                    "records": [{
                        "path": "receipts/record.json",
                        "schema_version": "fixture",
                    }],
                }],
                "files": [
                    {"source": "manifest.json", "destination": "release-evidence.json"},
                    {"source": "record.json", "destination": "receipts/record.json"},
                ],
            }
            EVIDENCE.stage(manifest, source, destination)
            self.assertTrue((destination / "release-evidence.json").is_file())
            self.assertTrue((destination / "receipts/record.json").is_file())
            self.assertFalse((destination / "extra.txt").exists())

    def test_manifest_rejects_symlinked_source_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            destination = directory / "destination"
            source.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            (source / "real").mkdir()
            (source / "real/record.json").write_text("{}\n", encoding="utf-8")
            (source / "alias").symlink_to(source / "real", target_is_directory=True)
            manifest = {
                "schema_version": "leone.release-evidence.v1",
                "release_line": "v0.3",
                "package_kind": "evidence",
                "status": "complete",
                "backend_requirements": [{
                    "id": "linux-x86_64-cuda",
                    "platform": "linux-x86_64",
                    "target": "x86_64-unknown-linux-gnu",
                    "backend": "cuda",
                    "records": [{"path": "receipts/record.json", "schema_version": "fixture"}],
                }],
                "files": [
                    {"source": "manifest.json", "destination": "release-evidence.json"},
                    {
                        "source": "alias/record.json",
                        "destination": "receipts/record.json",
                    },
                ],
            }
            with self.assertRaisesRegex(ValueError, "contains a symlink"):
                EVIDENCE.stage(manifest, source, destination)

    def test_manifest_stages_the_frozen_binary_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "source"
            destination = directory / "destination"
            source.mkdir()
            (source / "manifest.json").write_text("{}\n", encoding="utf-8")
            (source / "record.json").write_text(
                '{"schema_version": "fixture"}\n', encoding="utf-8"
            )
            frozen = directory / "frozen-leone"
            frozen.write_text("measured binary\n", encoding="utf-8")
            manifest = {
                "schema_version": "leone.release-evidence.v1",
                "release_line": "v0.3",
                "package_kind": "evidence",
                "status": "complete",
                "backend_requirements": [{
                    "id": "linux-x86_64-cuda",
                    "platform": "linux-x86_64",
                    "target": "x86_64-unknown-linux-gnu",
                    "backend": "cuda",
                    "records": [{"path": "receipts/record.json", "schema_version": "fixture"}],
                }],
                "files": [
                    {"source": "manifest.json", "destination": "release-evidence.json"},
                    {"source": "record.json", "destination": "receipts/record.json"},
                    {"source": "target/release/leone", "destination": "bin/leone"},
                ],
            }
            EVIDENCE.stage(
                manifest,
                source,
                destination,
                {"target/release/leone": frozen},
            )
            self.assertEqual((destination / "bin/leone").read_text(), "measured binary\n")

    def test_v04_offline_checker_rejects_unavailable_canonical_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "canonical receipt validator is required"):
                    CHECK.check_v04_manifest(manifest_path, offline=True)
            finally:
                CHECK.ROOT = previous

    def test_v04_complete_manifest_rejects_missing_validator(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["status"] = "complete"
        del manifest["backend_requirements"][0]["records"][0]["validator"]
        with self.assertRaisesRegex(ValueError, "semantic validator"):
            EVIDENCE.validate(manifest, require_complete=True)

    def test_v04_manifest_rejects_unknown_validator_role(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["backend_requirements"][0]["records"][0]["validator"] = "untrusted"
        with self.assertRaisesRegex(ValueError, "unknown validator"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_requires_each_gated_model(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["backend_requirements"][1]["records"] = [
            record
            for record in manifest["backend_requirements"][1]["records"]
            if record.get("model_family") != "llama"
        ]
        with self.assertRaisesRegex(ValueError, "every gated model"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_rejects_duplicate_quality_slot(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        duplicate = json.loads(json.dumps(manifest["backend_requirements"][0]["records"][1]))
        duplicate["path"] = "receipts/v04-linux-cuda-quality-comparison-qwen3-copy.json"
        manifest["backend_requirements"][0]["records"].append(duplicate)
        manifest["files"].append(
            {"source": duplicate["path"], "destination": duplicate["path"]}
        )
        with self.assertRaisesRegex(ValueError, "one quality record"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_rejects_missing_quality_sidecar(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        sidecar = manifest["backend_requirements"][0]["records"][1]["artifact_files"][0]
        manifest["files"] = [
            item for item in manifest["files"] if item["destination"] != sidecar
        ]
        with self.assertRaisesRegex(ValueError, "artifact sidecar"):
            EVIDENCE.validate(manifest)

    def test_quality_sidecar_must_be_referenced_by_canonical_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = root / "receipts/comparison.json"
            stage = root / "receipts/stage"
            stage.mkdir(parents=True)
            comparison.write_text(
                json.dumps(
                    {
                        "stages": {"oracle_export": {"path": "stage/oracle.json"}},
                        "logits": {"path": "oracle.f32"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (stage / "oracle.json").write_text(
                json.dumps({"execution": {"machine": {"path": "machine.txt"}}, "logits": {"path": "oracle.f32"}})
                + "\n",
                encoding="utf-8",
            )
            (stage / "machine.txt").write_text("machine\n", encoding="utf-8")
            record = {
                "path": "receipts/comparison.json",
                "validator": "quality-stage-v1",
                "validator_mode": "comparison",
                "artifact_root": "receipts/stage",
                "artifact_roots": ["receipts/stage"],
                "artifact_files": [
                    "receipts/stage/oracle.json",
                    "receipts/stage/machine.txt",
                ],
            }
            DISPATCH._check_quality_artifact_references(root, record)
            record["artifact_files"].append("receipts/stage/stale.json")
            (stage / "stale.json").write_text("stale\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact is unreferenced"):
                DISPATCH._check_quality_artifact_references(root, record)

    def test_v04_manifest_rejects_duplicate_shared_role(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        duplicate = json.loads(json.dumps(manifest["shared_records"][0]))
        duplicate["path"] = "receipts/v04-openai-client-copy.json"
        manifest["shared_records"].append(duplicate)
        manifest["files"].append(
            {"source": duplicate["path"], "destination": duplicate["path"]}
        )
        with self.assertRaisesRegex(ValueError, "one CUDA client, one Metal client"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_rejects_extra_backend_role(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        extra = json.loads(json.dumps(manifest["backend_requirements"][0]["records"][0]))
        extra["path"] = "receipts/v04-linux-cuda-extra.json"
        extra["role"] = "oracle"
        extra["validator"] = "quality-stage-v1"
        extra["dependencies"].append("scripts/validate-quality-stage.py")
        extra["validator_mode"] = "export"
        extra["artifact_root"] = "receipts/v04-linux-cuda-extra-stage"
        extra["artifact_roots"] = [extra["artifact_root"]]
        extra["artifact_files"] = [extra["artifact_root"] + "/stage.json"]
        manifest["backend_requirements"][0]["records"].append(extra)
        manifest["files"].extend(
            {"source": path, "destination": path}
            for path in (extra["path"], extra["artifact_files"][0])
        )
        with self.assertRaisesRegex(ValueError, "exactly these roles"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_rejects_self_declared_model_scope(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["required_model_families"] = ["qwen3"]
        with self.assertRaisesRegex(ValueError, "reviewed model families"):
            EVIDENCE.validate(manifest)

    def test_v04_manifest_rejects_unknown_trusted_validator(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        manifest = json.loads(json.dumps(manifest))
        manifest["trusted_validators"].append("scripts/extra-validator.py")
        with self.assertRaisesRegex(ValueError, "unknown trusted validators"):
            EVIDENCE.validate(manifest)

    def test_v04_checker_requires_packaged_validator_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["files"] = [
                item
                for item in manifest["files"]
                if item["destination"] != "scripts/source_inputs.py"
            ]
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "planned"
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "trusted validator is absent"):
                    DISPATCH.validate_records(
                        root,
                        manifest,
                        EVIDENCE.validate(manifest),
                        ROOT,
                    )
            finally:
                CHECK.ROOT = previous

    def test_v04_checker_rejects_missing_source_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            (root / "scripts/check-release-evidence.py").unlink()
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "planned"
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "packaged source input .* is missing"):
                    DISPATCH.validate_records(
                        root,
                        manifest,
                        EVIDENCE.validate(manifest),
                        ROOT,
                    )
            finally:
                CHECK.ROOT = previous

    def test_v04_source_manifest_records_can_remain_external(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "receipts").mkdir()
            source = {
                "source_commit": "a" * 40,
                "files": {
                    "crates/private/src/lib.rs": {
                        "sha256": "b" * 64,
                        "executable": False,
                    }
                },
            }
            (root / "receipts/source-inputs.json").write_text(
                json.dumps(source) + "\n", encoding="utf-8"
            )
            record = DISPATCH._source_manifest(
                root, {"source_manifest": "receipts/source-inputs.json"}
            )
            self.assertEqual(record["source_commit"], "a" * 40)
            DISPATCH._check_source_file_if_present(
                root, "crates/private/src/lib.rs", source["files"]["crates/private/src/lib.rs"]
            )

    def test_v04_checker_rejects_changed_source_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            (root / "scripts/check-release-evidence.py").write_text(
                "changed\n", encoding="utf-8"
            )
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "planned"
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "packaged source input changed"):
                    DISPATCH.validate_records(
                        root,
                        manifest,
                        EVIDENCE.validate(manifest),
                        ROOT,
                    )
            finally:
                CHECK.ROOT = previous

    def test_v04_checker_requires_source_hash_for_trusted_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            manifest = json.loads(manifest_path.read_text())
            source_path = root / manifest["source_manifest"]
            source = json.loads(source_path.read_text())
            del source["files"]["scripts/check-release-evidence.py"]
            source_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
            manifest["status"] = "planned"
            with self.assertRaisesRegex(ValueError, "source input hash is missing"):
                DISPATCH.validate_records(
                    root,
                    manifest,
                    EVIDENCE.validate(manifest),
                    ROOT,
                )

    def test_receipt_dispatch_requires_trusted_rust_tool(self):
        record = {
            "validator": "runtime-receipt-v1",
            "path": "receipts/runtime.json",
            "platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
            "model_family": "qwen3",
        }
        with self.assertRaisesRegex(ValueError, "canonical receipt validator is required"):
            DISPATCH._receipt_command(
                Path("/data"), record, None, "a" * 40, "receipts/source-inputs-v04.json"
            )

    def test_receipt_dispatch_binds_record_identity(self):
        record = {
            "validator": "quality-receipt-v1",
            "path": "receipts/quality.json",
            "platform": "darwin-arm64",
            "target": "aarch64-apple-darwin",
            "backend": "metal",
            "model_family": "llama",
            "model_sha256": "c" * 64,
            "metric_family": "kld",
        }
        command = DISPATCH._receipt_command(
            Path("/data"),
            record,
            Path("/trusted/leone-receipt-verify"),
            "b" * 40,
            "receipts/source-inputs-v04.json",
        )
        self.assertEqual(command[0], "/trusted/leone-receipt-verify")
        self.assertEqual(command[command.index("--kind") + 1], "quality")
        self.assertEqual(
            command[command.index("--source-manifest") + 1],
            "/data/receipts/source-inputs-v04.json",
        )
        for option, expected in (
            ("--source-commit", "b" * 40),
            ("--platform", "darwin-arm64"),
            ("--target", "aarch64-apple-darwin"),
            ("--backend", "metal"),
            ("--model-family", "llama"),
            ("--model-sha256", "c" * 64),
            ("--metric-family", "kld"),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)

    def test_service_dispatch_forwards_v04_source_manifest(self):
        record = {
            "validator": "service-metrics-v1",
            "path": "receipts/service.json",
        }
        command = DISPATCH._command(
            ROOT,
            ROOT,
            record,
            None,
            "a" * 40,
            "receipts/source-inputs-v04.json",
            "b" * 64,
        )
        self.assertEqual(
            command[command.index("--source-manifest") + 1],
            "receipts/source-inputs-v04.json",
        )

    def test_service_identity_binds_model_backend_and_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "receipts/service.json"
            path.parent.mkdir()
            receipt = {
                "model": {"sha256": "a" * 64},
                "source": {"commit": "b" * 40, "tracked_tree_clean": True},
                "binaries": {
                    "leone": {
                        "build_info": {
                            "value": {
                                "source_commit": "b" * 40,
                                "target": "x86_64-unknown-linux-gnu",
                                "source_tree_dirty": False,
                                "profile": "release",
                            }
                        }
                    }
                },
                "runs": [{"launch_argv": ["leone", "--backend", "cuda"]}],
            }
            path.write_text(json.dumps(receipt), encoding="utf-8")
            record = {
                "path": "receipts/service.json",
                "model_sha256": "a" * 64,
                "target": "x86_64-unknown-linux-gnu",
                "backend": "metal",
            }
            with self.assertRaisesRegex(ValueError, "backend differs"):
                DISPATCH._service_identity(root, record, "b" * 40)
            record["backend"] = "cuda"
            record["model_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "model differs"):
                DISPATCH._service_identity(root, record, "b" * 40)

    def test_client_dispatch_binds_v04_release_identity(self):
        record = {
            "validator": "openai-client-v1",
            "path": "receipts/client.json",
            "platform": "linux-x86_64",
            "target": "x86_64-unknown-linux-gnu",
            "backend": "cuda",
            "binary_sha256": "c" * 64,
            "model_family": "llama",
            "model_sha256": "d" * 64,
        }
        command = DISPATCH._command(
            ROOT,
            ROOT,
            record,
            Path("/trusted/leone-receipt-verify"),
            "a" * 40,
            "receipts/source-inputs-v04.json",
            "b" * 64,
        )
        for option, expected in (
            ("--source-manifest", "receipts/source-inputs-v04.json"),
            ("--platform", "linux-x86_64"),
            ("--target", "x86_64-unknown-linux-gnu"),
            ("--backend", "cuda"),
            ("--binary-sha256", "c" * 64),
            ("--model-family", "llama"),
            ("--model-sha256", "d" * 64),
        ):
            self.assertEqual(command[command.index(option) + 1], expected)

    @staticmethod
    def _client_v2_response(name, session, model_sha256, reuse_class="cold", reused_tokens=0):
        claim = {
            "model_sha256": model_sha256,
            "prompt_tokens": 2,
            "generated_tokens": 1,
            "finish_reason": "stop",
            "cancelled": False,
            "receipt_id": f"receipt-{session}",
            "session": {
                "session_id": session,
                "reuse_class": reuse_class,
                "reused_tokens": reused_tokens,
            },
        }
        receipt = {
            "claim": claim,
            "public_key_ed25519": "a" * 64,
            "signature_ed25519": "b" * 128,
        }
        return {
            "name": name,
            "claim": claim,
            "leone_receipt": receipt,
            "receipt_sha256": CHECK.value_digest(receipt),
            "signature_verified": True,
            "expected_session": session,
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            "content_sha256": "c" * 64,
        }

    @classmethod
    def _client_v2_fixture(cls, root):
        model_sha256 = "d" * 64
        script = root / "scripts/check-openai-client.py"
        script.parent.mkdir(parents=True)
        script.write_text("client checker\n", encoding="utf-8")
        binary = root / "bin/leone"
        binary.parent.mkdir(parents=True)
        log = root / "verify.log"
        binary.write_text(f"#!/bin/sh\nprintf '%s\\n' \"$*\" >> {log}\n", encoding="utf-8")
        binary.chmod(0o755)
        verifier = root / "leone-receipt-verify"
        verifier.write_text(
            "#!/usr/bin/env python3\n"
            "import json, pathlib, sys\n"
            "assert sys.argv[1:3] == ['receipt', 'verify-response']\n"
            "receipt = json.loads(pathlib.Path(sys.argv[3]).read_text())\n"
            "assert {'claim', 'public_key_ed25519', 'signature_ed25519'} <= receipt.keys()\n"
            f"with open({str(log)!r}, 'a', encoding='utf-8') as output: output.write('trusted\\n')\n",
            encoding="utf-8",
        )
        verifier.chmod(0o755)

        parent = cls._client_v2_response("client-parent", "client-parent", model_sha256)
        branch0 = cls._client_v2_response("client-branch-0", "client-branch-0", model_sha256, "device-fork", 1)
        branch1 = cls._client_v2_response("client-branch-1", "client-branch-1", model_sha256, "device-fork", 1)
        initial = cls._client_v2_response("client-context-growth", "client-context", model_sha256)
        grown = cls._client_v2_response("client-context-growth", "client-context", model_sha256, "append-only", 1)
        grown["claim"]["prompt_tokens"] = 513
        grown["claim"]["receipt_id"] = "receipt-client-context-grown"
        grown["usage"] = {"prompt_tokens": 513, "completion_tokens": 1, "total_tokens": 514}
        grown["receipt_sha256"] = CHECK.value_digest(grown["leone_receipt"])
        exact = cls._client_v2_response("client-exact-answer", "client-exact-answer", model_sha256)
        exact.update({
            "exact_match": True,
            "prompt_sha256": CHECK.EXACT_PROMPT_SHA256,
            "expected_answer_sha256": CHECK.EXACT_ANSWER_SHA256,
            "normalized_answer_sha256": CHECK.EXACT_ANSWER_SHA256,
            "content_sha256": CHECK.EXACT_ANSWER_SHA256,
        })
        survivor = cls._client_v2_response("client-reset-survivor", "client-reset-survivor", model_sha256)
        stop = cls._client_v2_response("stop", "client-stop", model_sha256)
        tool_stream = cls._client_v2_response("tool-stream", "client-tool-stream", model_sha256)
        tool_assistant = cls._client_v2_response("tool-assistant", "client-tool-roundtrip", model_sha256)
        tool = cls._client_v2_response("tool-roundtrip", "client-tool-roundtrip", model_sha256, "append-only", 1)
        tool.update({
            "streamed_call": True,
            "official_template": True,
            "assistant_message": {"role": "assistant"},
            "assistant_response": tool_assistant,
            "initial_stream": tool_stream,
        })
        recovery = cls._client_v2_response("client-recovery", "client-recovery", model_sha256)
        records = [
            parent,
            branch0,
            branch1,
            {"name": "context-growth", "initial": initial, "grown": grown},
            exact,
            {"name": "reset-isolation", "survivor": survivor},
            stop,
            tool,
            {"name": "strict-tool", "http_status": 400, "server_healthy": True},
            {"name": "tool-truncation", "http_status": 400, "server_healthy": True},
            {"name": "disconnect-after-content", "content_observed": True},
            recovery,
        ]
        source = root / "receipts/source-inputs-v04.json"
        source.parent.mkdir(parents=True)
        source.write_text(
            json.dumps({
                "source_commit": "a" * 40,
                "files": {
                    "scripts/check-openai-client.py": {
                        "sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                        "executable": False,
                    }
                },
            }) + "\n",
            encoding="utf-8",
        )
        record = {
            "schema_version": "leone.openai-client-check.v2",
            "passed": True,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "model_sha256": model_sha256,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "receipt_verifier": {
                "command": "receipt verify-response",
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            },
            "runtime": {"backend": "cuda", "kv": "f16", "batch_size": 1},
            "build": {
                "schema_version": "leone.build-info.v1",
                "source_commit": "a" * 40,
                "source_tree_dirty": False,
                "profile": "release",
                "target": "x86_64-unknown-linux-gnu",
            },
            "workflow": {"records": records},
        }
        path = root / "receipts/client.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path, binary, verifier, model_sha256

    def test_client_v2_verifies_retained_receipts_with_trusted_verifier(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, binary, verifier, model_sha256 = self._client_v2_fixture(root)
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                CHECK.check_client_v04(
                    path,
                    Path("receipts/source-inputs-v04.json"),
                    "linux-x86_64",
                    "x86_64-unknown-linux-gnu",
                    "cuda",
                    binary,
                    hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "llama",
                    model_sha256,
                    verifier,
                )
                self.assertEqual(
                    (root / "verify.log").read_text().splitlines(), ["trusted"] * 12
                )
                record = json.loads(path.read_text())
                del record["workflow"]["records"][0]["expected_session"]
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "session expectation is missing"):
                    CHECK.check_client_v04(
                        path,
                        Path("receipts/source-inputs-v04.json"),
                        "linux-x86_64",
                        "x86_64-unknown-linux-gnu",
                        "cuda",
                        binary,
                        hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "llama",
                        model_sha256,
                        verifier,
                    )
                record["workflow"]["records"][0]["expected_session"] = "client-parent"
                del record["workflow"]["records"][0]["leone_receipt"]
                path.write_text(json.dumps(record) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "response receipt is missing"):
                    CHECK.check_client_v04(
                        path,
                        Path("receipts/source-inputs-v04.json"),
                        "linux-x86_64",
                        "x86_64-unknown-linux-gnu",
                        "cuda",
                        binary,
                        hashlib.sha256(binary.read_bytes()).hexdigest(),
                        "llama",
                        model_sha256,
                        verifier,
                    )
            finally:
                CHECK.ROOT = previous

    def test_comparison_stage_dispatches_the_trusted_receipt_parser(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = root / "receipts/comparison.json"
            comparison.parent.mkdir(parents=True)
            comparison.write_text("{}\n", encoding="utf-8")
            record = {
                "validator_mode": "comparison",
                "path": "receipts/comparison.json",
                "artifact_root": "receipts/stage",
                "platform": "darwin-arm64",
                "target": "aarch64-apple-darwin",
                "statistics_target": "x86_64-unknown-linux-gnu",
                "backend": "metal",
                "model_family": "qwen3",
                "model_sha256": "a" * 64,
                "metric_family": "kld",
                "corpus_path": "corpus/quality-long-v3.txt",
                "corpus_sha256": "d" * 64,
                "oracle_model_sha256": "e" * 64,
                "sample_manifest_sha256": "e" * 64,
                "sample_contract": "linspace-inclusive-v1:128+task-rows",
            }
            command = DISPATCH._stage_command(
                Path("/trusted/validate-quality-stage.py"),
                root,
                record,
                Path("/trusted/leone-receipt-verify"),
                "b" * 40,
                "c" * 64,
            )
            self.assertEqual(
                command,
                [
                    sys.executable,
                    "/trusted/validate-quality-stage.py",
                    "comparison",
                    str(comparison),
                    "/trusted/leone-receipt-verify",
                    "--source-commit",
                    "b" * 40,
                    "--platform",
                    "darwin-arm64",
                    "--target",
                    "aarch64-apple-darwin",
                    "--statistics-target",
                    "x86_64-unknown-linux-gnu",
                    "--backend",
                    "metal",
                    "--adapter-path",
                    "research/oracle/llama_logits.cpp",
                    "--adapter-sha256",
                    "c" * 64,
                    "--model-family",
                    "qwen3",
                    "--model-sha256",
                    "a" * 64,
                    "--metric-family",
                    "kld",
                    "--corpus-sha256",
                    "d" * 64,
                    "--oracle-model-sha256",
                    "e" * 64,
                    "--sample-contract",
                    "linspace-inclusive-v1:128+task-rows",
                    "--sample-manifest-sha256",
                    "e" * 64,
                ],
            )

    def test_cuda_comparison_dispatches_the_cuda_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            comparison = root / "receipts/comparison.json"
            comparison.parent.mkdir(parents=True)
            comparison.write_text("{}\n", encoding="utf-8")
            record = {
                "validator_mode": "comparison",
                "path": "receipts/comparison.json",
                "artifact_root": "receipts/stage",
                "platform": "linux-x86_64",
                "target": "x86_64-unknown-linux-gnu",
                "statistics_platform": "linux-x86_64",
                "statistics_target": "x86_64-unknown-linux-gnu",
                "backend": "cuda",
                "model_family": "qwen3",
                "model_sha256": "a" * 64,
                "metric_family": "kld",
                "corpus_path": "corpus/quality-long-v3.txt",
                "corpus_sha256": "d" * 64,
                "oracle_model_sha256": "e" * 64,
                "task_manifest_sha256": "f" * 64,
                "trusted_inputs": "receipts/trusted.json",
                "generation_record": "receipts/generation.json",
                "generation_record_sha256": hashlib.sha256(b"{}\n").hexdigest(),
                "sample_contract": "linspace-inclusive-v1:128+task-rows",
            }
            (root / "receipts/trusted.json").write_text("{}\n", encoding="utf-8")
            (root / "receipts/generation.json").write_text("{}\n", encoding="utf-8")
            command = DISPATCH._stage_command(
                Path("/trusted/validate-quality-stage.py"),
                root,
                record,
                Path("/trusted/leone-receipt-verify"),
                "b" * 40,
                "c" * 64,
            )
            self.assertEqual(command[2:5], ["comparison", str(comparison), "/trusted/leone-receipt-verify"])
            for option, expected in (
                ("--platform", "linux-x86_64"),
                ("--target", "x86_64-unknown-linux-gnu"),
                ("--statistics-platform", "linux-x86_64"),
                ("--backend", "cuda"),
                ("--task-manifest-sha256", "f" * 64),
                ("--trusted-inputs", str(root / "receipts/trusted.json")),
                ("--generation-record", str(root / "receipts/generation.json")),
                ("--model-sha256", "a" * 64),
            ):
                self.assertEqual(command[command.index(option) + 1], expected)

    def test_comparison_stage_requires_trusted_model_identity(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        record = {
            "path": "receipts/v04-darwin-metal-quality-qwen3.json",
            "schema_version": "leone.quality-cross-device.v2",
            "role": "quality",
            "validator": "quality-stage-v1",
            "dependencies": [
                "scripts/validate-quality-stage.py",
                "research/oracle/llama_logits.cpp",
            ],
            "validator_mode": "comparison",
            "artifact_root": "receipts/stage",
            "model_family": "qwen3",
            "metric_family": "kld",
            "corpus_path": "corpus/quality-long-v3.txt",
            "corpus_sha256": "b" * 64,
            "oracle_model_sha256": "c" * 64,
            "sample_contract": "linspace-inclusive-v1:128+task-rows",
        }
        manifest["backend_requirements"][1]["records"] = [record]
        with self.assertRaisesRegex(ValueError, "unreviewed model SHA-256"):
            EVIDENCE.validate(manifest)

    def test_comparison_stage_requires_the_trusted_adapter_dependency(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        record = {
            "path": "receipts/v04-darwin-metal-quality-qwen3.json",
            "schema_version": "leone.quality-cross-device.v2",
            "role": "quality",
            "validator": "quality-stage-v1",
            "dependencies": ["scripts/validate-quality-stage.py"],
            "validator_mode": "comparison",
            "artifact_root": "receipts/stage",
            "model_family": "qwen3",
            "model_sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
            "statistics_target": "x86_64-unknown-linux-gnu",
            "metric_family": "kld",
            "corpus_path": "corpus/quality-long-v3.txt",
            "corpus_sha256": "a4f81b97182b0fb2f48c1e58c14e9ffee8f8e4472aa2742ecf39dd84ac8d95b8",
            "oracle_model_sha256": "5e416a2020fe63e76ea13c8979be35fc6070aaf3578f7876400c55c2f5c3eb30",
            "sample_contract": "linspace-inclusive-v1:128+task-rows",
        }
        manifest["backend_requirements"][1]["records"] = [record]
        with self.assertRaisesRegex(ValueError, "trusted adapter"):
            EVIDENCE.validate(manifest)

    def test_comparison_stage_requires_the_linux_statistics_target(self):
        manifest = EVIDENCE.load(ROOT / "packaging/release-evidence.v0.4.json")
        record = manifest["backend_requirements"][1]["records"][1]
        record["statistics_target"] = "aarch64-apple-darwin"
        with self.assertRaisesRegex(ValueError, "unreviewed statistics_target"):
            EVIDENCE.validate(manifest)

    def test_checker_requires_an_explicit_release_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "release evidence manifest is required"):
                    CHECK.main(["--root", str(root), "--offline"])
            finally:
                CHECK.ROOT = previous

    def test_checker_rejects_manifest_outside_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "data"
            outside = directory / "outside.json"
            root.mkdir()
            outside.write_text("{}\n", encoding="utf-8")
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "outside its root"):
                    CHECK.main(
                        ["--root", str(root), "--manifest", str(outside), "--offline"]
                    )
            finally:
                CHECK.ROOT = previous

    def test_checker_rejects_manifest_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "data"
            outside = directory / "outside.json"
            root.mkdir()
            outside.write_text("{}\n", encoding="utf-8")
            (root / "release-evidence.json").symlink_to(outside)
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "contains a symlink"):
                    CHECK.main(["--root", str(root), "--offline"])
            finally:
                CHECK.ROOT = previous

    def test_checker_rejects_unknown_release_line_without_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = json.loads(
                (ROOT / "packaging/release-evidence.v0.3.json").read_text()
            )
            manifest["release_line"] = "v0.5"
            path = root / "release-evidence.json"
            path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "unsupported release evidence line"):
                    CHECK.main(["--root", str(root), "--offline"])
            finally:
                CHECK.ROOT = previous

    def test_v04_checker_rejects_schema_only_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._complete_v04_skeleton(root)
            record_path = root / "receipts/v04-linux-cuda-runtime.json"
            record_path.write_text(json.dumps({"schema_version": 9}) + "\n", encoding="utf-8")
            manifest = json.loads(manifest_path.read_text())
            for record in manifest["backend_requirements"][0]["records"]:
                if record["path"] == "receipts/v04-linux-cuda-runtime.json":
                    record["sha256"] = hashlib.sha256(record_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
            previous = CHECK.ROOT
            try:
                CHECK.ROOT = root
                with self.assertRaisesRegex(ValueError, "evidence schema differs"):
                    CHECK.check_v04_manifest(manifest_path, offline=True)
            finally:
                CHECK.ROOT = previous

    def test_local_receipt_check_keeps_external_input_gate(self):
        receipt = ROOT / "receipts/concurrent-service-study.json"
        with tempfile.TemporaryDirectory() as temporary:
            missing_model = Path(temporary) / "missing-reproduction-model.gguf"
            errors = STUDY.validate_recorded_receipt(
                receipt, ROOT, offline=False, rerun_paths={"model": missing_model}
            )
        self.assertTrue(any("recorded model" in error for error in errors))

    def test_archive_manifest_rejects_payload_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            root.mkdir()
            self._write_package_fixture(root)
            payload = root / "payload.txt"
            payload.write_text("fixture\n", encoding="utf-8")
            archive = self._repack(directory, root)
            VERIFY.verify_archive(archive)
            payload.write_text("changed\n", encoding="utf-8")
            archive = self._archive_existing_manifest(directory, root, "fixture")
            with self.assertRaises(ValueError):
                VERIFY.verify_archive(archive)

    def test_archive_rejects_extra_top_level_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "extra.tar.gz"
            root = directory / "leone-fixture"
            root.mkdir()
            (root / "payload.txt").write_text("payload\n", encoding="utf-8")
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(root, arcname="leone-fixture")
                info = tarfile.TarInfo("outside.txt")
                payload = b"outside\n"
                info.size = len(payload)
                bundle.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                VERIFY.extract(archive, directory / "extracted")

    def test_archive_rejects_normalized_duplicate_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "duplicate.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                root_info = tarfile.TarInfo("leone-fixture/")
                root_info.type = tarfile.DIRTYPE
                bundle.addfile(root_info)
                for name in ("leone-fixture/item", "leone-fixture/./item"):
                    info = tarfile.TarInfo(name)
                    payload = b"same\n"
                    info.size = len(payload)
                    bundle.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                VERIFY.extract(archive, directory / "extracted")

    def test_archive_rejects_symlink_parent_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            archive = directory / "symlink-parent.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                root_info = tarfile.TarInfo("leone-fixture/")
                root_info.type = tarfile.DIRTYPE
                bundle.addfile(root_info)
                link = tarfile.TarInfo("leone-fixture/scripts")
                link.type = tarfile.SYMTYPE
                link.linkname = "/tmp"
                bundle.addfile(link)
            with self.assertRaisesRegex(ValueError, "non-regular member"):
                VERIFY.extract(archive, directory / "extracted")

    def test_archive_rejects_symlinked_extraction_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root)
            outside = directory / "outside"
            outside.mkdir()
            destination = directory / "extracted"
            destination.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "extraction destination"):
                VERIFY.extract(archive, destination)

    def test_archive_rejects_symlinked_extraction_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root)
            outside = directory / "outside"
            outside.mkdir()
            parent = directory / "parent"
            parent.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink parent"):
                VERIFY.extract(archive, parent / "extracted")

    def test_archive_requires_executable_runtime_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            self._write_package_fixture(root)
            (root / "install.sh").chmod(0o644)
            archive = self._repack(directory, root, "non-executable")
            with self.assertRaisesRegex(ValueError, "not executable"):
                VERIFY.verify_archive(archive)

    def test_nested_manifest_is_part_of_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            (root / "sub").mkdir(parents=True)
            self._write_package_fixture(root)
            (root / "payload.txt").write_text("fixture\n", encoding="utf-8")
            (root / "sub/MANIFEST.sha256").write_text("nested\n", encoding="utf-8")
            payload_digest = hashlib.sha256((root / "payload.txt").read_bytes()).hexdigest()
            (root / "MANIFEST.sha256").write_text(
                f"{payload_digest}  ./payload.txt\n", encoding="utf-8"
            )
            archive = directory / "nested-manifest.tar.gz"
            with tarfile.open(archive, "w:gz") as bundle:
                bundle.add(root, arcname=root.name)
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            (directory / "nested-manifest.tar.gz.sha256").write_text(
                f"{archive_digest}  {archive.name}\n", encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                VERIFY.verify_archive(archive)

    def test_manifest_writer_includes_nested_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "fixture"
            root.mkdir()
            (root / "payload.txt").write_text("payload\n", encoding="utf-8")
            (root / "sub").mkdir()
            (root / "sub/MANIFEST.sha256").write_text("nested\n", encoding="utf-8")
            output = root / "MANIFEST.sha256"
            WRITE_MANIFEST.write_manifest(root, output)
            rows = output.read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(row.endswith("./sub/MANIFEST.sha256") for row in rows))
            self.assertFalse(any(row.endswith("./MANIFEST.sha256") for row in rows))

    def test_manifest_writer_rejects_symlink_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "fixture"
            root.mkdir()
            outside = directory / "outside.txt"
            outside.write_text("private\n", encoding="utf-8")
            (root / "payload.txt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                WRITE_MANIFEST.write_manifest(root, root / "MANIFEST.sha256")

    def test_archive_writer_rejects_symlink_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "fixture"
            root.mkdir()
            outside = directory / "outside.txt"
            outside.write_text("private\n", encoding="utf-8")
            (root / "payload.txt").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                MAKE_ARCHIVE.write_archive(
                    directory, "fixture", directory / "fixture.tar.gz", 0
                )

    def test_archive_verifier_does_not_execute_payload_checker(self):
        marker = Path("/tmp/leone-untrusted-archive-checker")
        marker.unlink(missing_ok=True)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                root = directory / "leone-fixture"
                scripts = root / "scripts"
                scripts.mkdir(parents=True)
                (root / "payload.txt").write_text("fixture\n", encoding="utf-8")
                (scripts / "check-public-tree.sh").write_text(
                    "#!/bin/sh\ntouch /tmp/leone-untrusted-archive-checker\n",
                    encoding="utf-8",
                )
                archive = self._archive(directory, root, "payload-checker")
                VERIFY.verify_archive(archive)
            self.assertFalse(marker.exists())
        finally:
            marker.unlink(missing_ok=True)

    def test_archive_requires_trusted_public_checker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root)
            with self.assertRaisesRegex(ValueError, "trusted public checker"):
                VERIFY.verify_archive(archive, directory / "missing-trusted")

    def test_evidence_requires_trusted_checker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root, kind="evidence")
            trusted = directory / "trusted"
            self._trusted_stub(trusted, evidence=False)
            with self.assertRaisesRegex(ValueError, "trusted checker .* is missing"):
                VERIFY.verify_archive(archive, trusted)

    def test_evidence_rejects_stale_trusted_checker(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root, kind="evidence")
            trusted = directory / "trusted"
            self._trusted_stub(trusted)
            (trusted / "scripts/check-release-evidence.py").write_text(
                "stale\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "trusted checker differs from archive"):
                VERIFY.verify_archive(archive, trusted)

    def test_evidence_checker_file_cannot_be_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "leone-fixture"
            archive = self._archive(directory, root, kind="evidence")
            (root / "scripts/check-release-evidence.py").unlink()
            archive = self._repack(directory, root, "removed-checker")
            trusted = directory / "trusted"
            self._trusted_stub(trusted)
            with self.assertRaisesRegex(ValueError, "missing required files"):
                VERIFY.verify_archive(archive, trusted)

    def test_external_symlink_input_uses_hashed_public_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            workspace = directory / "workspace"
            outside = directory / "private-model-store"
            (workspace / "models").mkdir(parents=True)
            outside.mkdir()
            model = outside / "model.gguf"
            model.write_bytes(b"model fixture\n")
            link = workspace / "models/model.gguf"
            link.symlink_to(model)
            model_digest = hashlib.sha256(model.read_bytes()).hexdigest()
            label = STUDY._relative_path(
                workspace, link.resolve(), "model", model_digest
            )
            self.assertEqual(label, f"<external>/model/{model_digest}")
            self.assertNotIn(str(outside), label)
            launch_value = STUDY.public_metadata(
                str(link.resolve()), workspace, workspace / "run-artifacts"
            )
            self.assertEqual(launch_value, "<external-path>")
            metadata = STUDY.public_metadata(
                {
                    "model": {"path": str(link.resolve()), "sha256": model_digest},
                    "logits": {"path": str(link.resolve()), "sha256": model_digest},
                    "oracle": {
                        "manifest": {
                            "executable": {
                                "path": str(link.resolve()),
                                "sha256": model_digest,
                            }
                        }
                    },
                },
                workspace,
                workspace / "run-artifacts",
            )
            self.assertEqual(metadata["model"]["path"], f"<external>/model/{model_digest}")
            self.assertEqual(metadata["logits"]["path"], f"<external>/logits/{model_digest}")
            self.assertEqual(
                metadata["oracle"]["manifest"]["executable"]["path"],
                f"<external>/executable/{model_digest}",
            )

    def test_public_metadata_redacts_path_after_url(self):
        value = "GET http://127.0.0.1/v1 failed opening /" + "Users/alice/model.gguf"
        result = STUDY.public_metadata(value, Path("/workspace"), Path("/workspace/run"))
        self.assertEqual(result, "GET http://127.0.0.1/v1 failed opening <external-path>")


if __name__ == "__main__":
    unittest.main()
