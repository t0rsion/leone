"""Build release-layout branching study packages from the canonical builders.

The quality records come from the real collectors' test builders and, for CUDA,
from the real `write-cuda-quality-comparison.py`. The study comes from the accepted
frozen control of `frozen_control_fixture`. Nothing here types a quality field.
Native and statistics executable hashes differ, as in a collection on two targets.
The builders take their peer linkage and native hash from this module before any
producer hash is computed, so the canonical validator accepts the finished record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
from pathlib import Path
import subprocess
import sys
import textwrap
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import frozen_control_fixture as CONTROL  # noqa: E402
import test_quality_cuda as QC  # noqa: E402
import test_quality_stage as QS  # noqa: E402
import test_study_branching_service as T  # noqa: E402

HARNESS = T.HARNESS
COMMIT = "c" * 40
NATIVE_SHA256 = hashlib.sha256(b"leone").hexdigest()
CUDA_FAMILIES = ("libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-cuda")
METAL_FAMILIES = ("libllama", "libggml", "libggml-base", "libggml-cpu", "libggml-metal")
STATISTICS_SHA256 = "a" * 64


def peer_libraries(families: tuple[str, ...], suffix: str) -> list[dict]:
    """The linkage records the way the study derives them, one hash per family."""

    return [
        {"name": f"{name}.{suffix}", "sha256": hashlib.sha256(name.encode()).hexdigest()}
        for name in families
    ]


def patched_method(cls, name: str, edits: tuple[tuple[str, str], ...], **names):
    """Compile `cls.name` with regular expression edits. Every edit must match once."""

    source = textwrap.dedent(inspect.getsource(getattr(cls, name)))
    for old, new in edits:
        if len(re.findall(old, source)) != 1:
            raise AssertionError(f"builder text changed, cannot patch {name}: {old!r}")
        source = re.sub(old, lambda _: new, source)
    namespace = dict(vars(sys.modules[cls.__module__]))
    namespace.update(names)
    exec(compile(source, f"<patched {cls.__name__}.{name}>", "exec"), namespace)
    return namespace[name]


def patched_function(module, name: str, edits: tuple[tuple[str, str], ...]):
    """Compile `module.name` with regular expression edits. Every edit must match once."""

    source = textwrap.dedent(inspect.getsource(getattr(module, name)))
    for old, new in edits:
        if len(re.findall(old, source)) != 1:
            raise AssertionError(f"control text changed, cannot patch {name}: {old!r}")
        source = re.sub(old, lambda _: new, source)
    namespace = dict(vars(module))
    exec(compile(source, f"<patched {name}>", "exec"), namespace)
    return namespace[name]


def run_writer(fixture: dict, output: Path) -> None:
    fixture = {**fixture, "output": output}
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        QC.CudaQualityTests.writer_command(fixture), check=True, capture_output=True, text=True
    )


def cuda_case(libraries: list[dict]):
    """Return a CUDA case whose writer inputs carry the peer linkage and a distinct native hash."""

    edits = (
        (re.escape('"executable": {"path": "llama-cuda", "sha256": "c" * 64, "linked_libraries": []},'),
         '"executable": {"path": "llama-cuda", "sha256": "c" * 64, "linked_libraries": PEER_LIBRARIES},'),
        (r'"name": executable\.name,\s+"sha256": digest\(executable\),\s+"build_info": build_info,',
         '"name": "leone", "sha256": NATIVE_SHA256, "build_info": build_info,'),
        (re.escape('source = "d" * 40'), f'source = "{COMMIT}"'),
    )
    method = patched_method(
        QC.CudaQualityTests, "write_writer_fixture", edits,
        PEER_LIBRARIES=libraries, NATIVE_SHA256=NATIVE_SHA256,
    )
    case = QC.CudaQualityTests("test_cuda_comparison_accepts_samples_only")
    case.setUp()
    case.write_writer_fixture = method.__get__(case)
    return case


CONFIG = {
    "cuda": {
        "platform": "linux-x86_64", "target": "x86_64-unknown-linux-gnu",
        "schema": "leone.quality-comparison.v2", "sidecars": ("leone-quality", "llama-cpp-quality"),
    },
    "metal": {
        "platform": "darwin-arm64", "target": "aarch64-apple-darwin",
        "schema": "leone.quality-cross-device.v2", "sidecars": ("leone-quality", "llama-quality"),
    },
}
CUDA_RECORD = "receipts/v04-linux-cuda-quality-comparison-qwen3.json"
METAL_RECORD = "metal-quality/comparison.json"
TEMPLATE = "fixtures/qwen3-legacy-chatml.jinja"
PRODUCER = "scripts/produce-history-tokenization.py"
GENERATOR = "scripts/generate-openai-chat-template-fixtures.py"
HARNESS_SCRIPT = "scripts/study-branching-service.py"
STAGE_SCRIPT = "scripts/validate-quality-stage.py"
ADAPTER = "research/oracle/llama_logits.cpp"
PUBLIC_CHECKERS = ("scripts/check-public-tree.sh", "scripts/check-public-tree.py")
FIXED = (
    "external/PINNED", TEMPLATE, ADAPTER, "scripts/check-history-tokenization.py",
    "scripts/freeze-branching-manifest.py", GENERATOR, "scripts/linked_libraries.py",
    PRODUCER, "scripts/source_inputs.py", HARNESS_SCRIPT, "scripts/fetch-llama-cpp.sh",
    "scripts/study-branching-service.sh",
)
STUDY = ("frozen.json", "calibration.json", "calibration-receipt.json")
SOURCE_MANIFEST = "receipts/source-inputs-v04.json"
CPU_LAUNCH = {
    "argv": ["<llama-server>", "--n-gpu-layers", "0", "--port", "<port>"],
    "env": {"CUDA_VISIBLE_DEVICES": "", "LLAMA_ARG_DEVICE": "none"},
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    return sha(path.read_bytes())


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def copy_repository(root: Path, names) -> None:
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        target.chmod((ROOT / name).stat().st_mode & 0o777)


def tree_files(root: Path, *directories: str) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for directory in directories
        for path in (root / directory).rglob("*")
        if path.is_file()
    )


def publish_cuda(root: Path, libraries: list[dict]) -> dict:
    """Write the real CUDA comparison writer's output under `root/receipts`."""

    case = cuda_case(libraries)
    fixture = case.write_writer_fixture()
    run_writer(fixture, root / CUDA_RECORD)
    trusted_path = root / "receipts/v04-linux-cuda-quality-trusted-qwen3.json"
    generation_path = root / "receipts/v04-linux-cuda-quality-generation-qwen3.json"
    trusted_path.write_bytes(fixture["trusted_inputs"].read_bytes())
    generation_path.write_bytes(fixture["generation_record"].read_bytes())
    trusted = json.loads(trusted_path.read_text())["trusted"]
    stem = CUDA_RECORD.removesuffix(".json")
    return {
        "case": case, "path": CUDA_RECORD, "verifier": fixture["verifier"], "trusted": trusted,
        "sidecars": [f"{stem}-{name}.json" for name in CONFIG["cuda"]["sidecars"]],
        "roots": [f"{stem}-cuda-artifacts", f"{stem}-cuda-samples"],
        "extra": {
            "trusted_inputs": trusted_path.relative_to(root).as_posix(),
            "generation_record": generation_path.relative_to(root).as_posix(),
            "generation_record_sha256": file_sha(generation_path),
            "task_manifest_sha256": trusted["task_manifest_sha256"],
            "statistics_platform": trusted["statistics_platform"],
        },
        "files": [trusted_path.relative_to(root).as_posix(), generation_path.relative_to(root).as_posix()],
    }


def metal_case(libraries: list[dict]):
    """A Metal comparison case whose record carries the peer linkage and a distinct native hash."""

    edits = (
        (re.escape('"executable": {"path": "llama-logits-oracle", "sha256": "c" * 64, "linked_libraries": []},'),
         '"executable": {"path": "llama-logits-oracle", "sha256": "c" * 64, "linked_libraries": PEER_LIBRARIES},'),
        (r'"name": "leone",\s+"sha256": "a" \* 64,\s+"build_info"',
         '"name": "leone", "sha256": NATIVE_SHA256, "build_info"'),
        (re.escape('native_source = "b" * 40'), f'native_source = "{COMMIT}"'),
    )
    method = patched_method(
        QS.QualityStageTests, "write_fixture", edits,
        PEER_LIBRARIES=libraries, NATIVE_SHA256=NATIVE_SHA256,
    )
    case = QS.QualityStageTests("test_comparison_receipt_is_linked_by_hash")
    case.write_fixture = method.__get__(case)
    case.setUp()
    return case


def publish_metal(root: Path, libraries: list[dict]) -> dict:
    """Copy the canonical Metal comparison record and its packaged stages under `root`."""

    case = metal_case(libraries)
    manifest, record, verifier, trusted = case.write_comparison_fixture()
    source = manifest.parent
    directory = root / "metal-quality"
    names = ["comparison.json", "long-context-task.json", "leone-quality.json", "llama-quality.json"]
    for name in names:
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes((source / name).read_bytes())
    roots = ["packaged-oracle-stage", "packaged-metal-stage", "packaged-samples"]
    for name in roots:
        for path in (source / name).rglob("*"):
            if path.is_file():
                target = directory / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    prefix = "metal-quality/"
    return {
        "case": case, "path": METAL_RECORD, "verifier": verifier, "trusted": trusted,
        "sidecars": [prefix + name for name in names[1:]],
        "roots": [prefix + name for name in roots],
        "extra": {"sample_manifest_sha256": trusted["sample_manifest_sha256"]},
        "files": [],
        "model": (source / "subject-model.gguf").read_bytes(),
    }


def metal_libraries(suffix: str = "dylib") -> list[dict]:
    return peer_libraries(METAL_FAMILIES, suffix)


def study_patches(backend: str, published: dict):
    """The control's builders, pointed at the backend's calibration manifest and linkage."""

    patches = []
    if backend == "metal":
        def calibration():
            return json.loads((ROOT / "benchmarks/branching-service-calibration-metal.json").read_text())

        def libraries(_suffix, extra=()):
            return [
                {"name": item["name"], "sha256": item["sha256"]}
                for item in peer_libraries(METAL_FAMILIES, "0.dylib")
            ] + [{"name": f"{name}.0.dylib", "sha256": hashlib.sha256(name.encode()).hexdigest()} for name in extra]

        original_put = CONTROL.put

        def put(root, path, data):
            return original_put(root, path, published["model"] if path == "models/model.gguf" else data)

        patches = [
            mock.patch.object(T, "calibration_manifest", calibration),
            mock.patch.object(CONTROL, "libraries", libraries),
            mock.patch.object(CONTROL, "put", put),
        ]
    return patches


def history_patches(published: dict) -> list:
    """Make the control's retained llama.cpp oracle agree with the package pins.

    The control keeps placeholder source and template digests in each retained
    oracle. A package pins the llama.cpp commit in `external/PINNED` and the
    template by its file digest, so the oracle, the frozen declaration, and the
    served identity take those values. Nothing else about the control changes.
    """

    pin = (ROOT / "external/PINNED").read_text().strip()
    template = file_sha(ROOT / TEMPLATE)
    model = sha(published.get("model", b"subject"))
    attach = patched_function(CONTROL, "attach_llama_tokenization", (
        (re.escape('"source_commit": "d" * 40,'), f'"source_commit": "{pin}",'),
        (re.escape('"template_config_sha256": "2" * 64,'), f'"template_config_sha256": "{template}",'),
        (re.escape('"template_bytes_sha256": LLAMA_TEMPLATE_SHA,'), f'"template_bytes_sha256": "{template}",'),
    ))
    frozen = CONTROL._build_frozen_manifest

    def declare(calibration, receipt_path):
        manifest = frozen(calibration, receipt_path)
        for engine in manifest["engines"]:
            if engine["id"] == "llama_cpp":
                engine["history_tokenization"].update(
                    chat_template_sha256=template, template_config_sha256=template,
                    producer_source_commit=pin, producer_model_sha256=model,
                )
        return manifest

    return [
        mock.patch.object(CONTROL, "attach_llama_tokenization", attach),
        mock.patch.object(CONTROL, "LLAMA_TEMPLATE_SHA", template),
        mock.patch.object(CONTROL, "LLAMA_SOURCE", "llama.cpp:git:" + pin),
        mock.patch.object(CONTROL, "_build_frozen_manifest", declare),
    ]


def build_study(root: Path, backend: str, published: dict) -> dict:
    """Build the accepted frozen control in `root`, bound to the published quality record."""

    def publish(_root, leone_sha256):
        assert leone_sha256 == NATIVE_SHA256
        return {"path": published["path"], "sha256": file_sha(root / published["path"])}

    patches = [
        mock.patch.object(CONTROL, "_publish_quality_record", publish),
        *history_patches(published),
        *study_patches(backend, published),
    ]
    for patch in patches:
        patch.start()
    try:
        return CONTROL.build_control(root)
    finally:
        for patch in reversed(patches):
            patch.stop()


def write_receipt(root: Path, control: dict, path: str) -> None:
    receipt = json.loads(control["receipt_path"].read_text())
    receipt["source"] = {"status": "observed", "commit": COMMIT, "tracked_tree_clean": True}
    write_json(root / path, receipt)
    if control["receipt_path"] != root / path:
        control["receipt_path"].unlink()


def retained_oracle(receipt: dict, peer_id: str) -> dict:
    """The oracle of the first retained llama.cpp branch. The pins derive from it."""

    run = next(item for item in receipt["runs"] if item["engine"] == peer_id)
    return run["branches"][0]["history_reuse"]["tokenization"]["oracle"]


def write_history(root: Path, record: dict, control: dict) -> None:
    """Write history files from the real checker's own result builders."""

    check = importlib.util.spec_from_file_location("history_check_fixture", ROOT / "scripts/check-history-tokenization.py")
    check_module = importlib.util.module_from_spec(check)
    check.loader.exec_module(check_module)
    check = check_module
    receipt = (root / record["path"]).read_bytes()
    template = (root / TEMPLATE).read_bytes()
    peer = next(item for item in json.loads(receipt)["engines"] if item["kind"] in ("llama.cpp", "llama_cpp"))
    oracle = retained_oracle(json.loads(receipt), peer["id"])
    policy = oracle["special_tokens_policy"]
    expected = {
        "schema_version": "leone.history-reexecution-expected.v1",
        "source_commit": oracle["source_commit"],
        "template_mode": "legacy", "vocab_size": oracle["vocab_size"], "prompt_prefix": "",
        "special_tokens_policy": policy, "special_tokens_policy_sha256": sha(canonical(policy)),
        "input_sha256": sha(receipt), "producer_sha256": file_sha(root / PRODUCER),
        "template_generator_sha256": file_sha(root / GENERATOR),
        "executable_sha256": peer["provenance"]["executable_sha256"],
        "loaded_library_sha256": oracle["loaded_library_sha256"], "model_sha256": record["model_sha256"],
        "template_config_sha256": sha(template), "template_bytes_sha256": sha(template),
    }
    write_json(root / record["history_expected"], expected)
    expected_bytes = (root / record["history_expected"]).read_bytes()
    check.load_expected(expected_bytes)
    items, skipped = check.collect_inputs(json.loads(receipt), "receipt")
    snapshot = {
        "llama_server": {"sha256": expected["executable_sha256"]},
        "model": {"sha256": expected["model_sha256"]},
        "template": {"sha256": expected["template_config_sha256"]},
        "producer": {"sha256": expected["producer_sha256"]},
        "template_generator": {"sha256": expected["template_generator_sha256"]},
    }
    report = check.new_report()
    report["input"] = {"format": "receipt", "sha256": sha(receipt), "bytes": len(receipt), "runs_not_llama_cpp": skipped}
    report["expected_sha256"] = sha(expected_bytes)
    report["identity"] = check._identity_record(expected, snapshot)
    report["identity"]["oracle"] = {
        name: items[0]["retained"]["oracle"][name] for name in check.ORACLE_IDENTITY_FIELDS
    }
    report["checked_branches"] = [check.branch_summary(item["label"], item["retained"]) for item in items]
    report["fresh_server_launches"] = [CPU_LAUNCH for _ in items]
    write_json(root / record["history_result"], check.finish(report, "verified", None, None))


def quality_record(backend: str, published: dict) -> dict:
    config, trusted = CONFIG[backend], published["trusted"]
    record = {
        "path": published["path"], "schema_version": config["schema"], "role": "quality",
        "validator": "quality-stage-v1", "validator_mode": "comparison",
        "platform": config["platform"], "target": config["target"], "backend": backend,
        "model_family": trusted["model_family"], "model_sha256": trusted["model_sha256"],
        "metric_family": trusted["metric_family"], "statistics_target": trusted["statistics_target"],
        "corpus_sha256": trusted["corpus_sha256"], "oracle_model_sha256": trusted["oracle_model_sha256"],
        "sample_contract": trusted["sample_contract"],
        "artifact_root": published["roots"][0], "artifact_roots": published["roots"],
        **published["extra"],
    }
    return record


def branching_record(backend: str, control: dict, receipt_path: str, quality: dict) -> dict:
    config, base = CONFIG[backend], f"receipts/v04-{backend}-branching"
    leone = next(item for item in control["receipt"]["engines"] if item["kind"] == "leone")
    return {
        "path": receipt_path, "schema_version": "leone.branching-service.v1", "role": "branching",
        "validator": "branching-service-v1", "platform": config["platform"], "target": config["target"],
        "backend": backend, "model_family": quality["model_family"], "model_sha256": quality["model_sha256"],
        "binary_sha256": leone["provenance"]["executable_sha256"],
        "quality_record": quality["path"], "history_expected": f"{base}-history-expected.json",
        "history_result": f"{base}-history-result.json", "study_files": list(STUDY),
    }


def write_sources(root: Path, manifest: dict, records: list[dict]) -> None:
    """Hash every non-record dependency, trusted validator, and public checker."""

    record_paths = {record["path"] for record in records}
    names = set(manifest["trusted_validators"]) | set(PUBLIC_CHECKERS)
    for record in records:
        names.update(name for name in record["dependencies"] if name not in record_paths | {SOURCE_MANIFEST})
    files = {
        name: {"sha256": file_sha(root / name), "executable": bool((root / name).stat().st_mode & 0o111)}
        for name in sorted(names)
    }
    write_json(root / SOURCE_MANIFEST, {
        "schema_version": "leone.source-inputs.v2", "source_commit": COMMIT, "files": files,
    })


def build_package(root: Path, backend: str) -> dict:
    """Assemble one backend's branching package from the canonical builders.

    Returns the release root, the reduced manifest, the dispatch records, and the
    trusted receipt tool. The reduced manifest names only this backend's quality
    comparison and branching study. The checks that run are the real ones.
    """

    libraries = peer_libraries(CUDA_FAMILIES, "so.0.21.0") if backend == "cuda" else metal_libraries("0.21.0.dylib")
    published = publish_cuda(root, libraries) if backend == "cuda" else publish_metal(root, libraries)
    control = build_study(root, backend, published)
    receipt_path = f"receipts/v04-{backend}-branching-service.json"
    write_receipt(root, control, receipt_path)
    copy_repository(root, [*FIXED, *PUBLIC_CHECKERS, STAGE_SCRIPT])
    quality = quality_record(backend, published)
    record = branching_record(backend, control, receipt_path, quality)
    write_history(root, record, control)
    pins = [SOURCE_MANIFEST, "external/PINNED", ADAPTER, STAGE_SCRIPT]
    quality["artifact_files"] = [*published["sidecars"], *tree_files(root, *published["roots"])]
    quality["dependencies"] = [*pins, *published["files"]]
    record["dependencies"] = [
        SOURCE_MANIFEST, *FIXED, quality["path"], record["history_expected"],
        record["history_result"], *STUDY,
        *(name for name in published["sidecars"] if name.endswith("-quality.json")),
    ]
    records = [quality, record]
    manifest = {
        "source_manifest": SOURCE_MANIFEST, "trusted_validators": [HARNESS_SCRIPT, STAGE_SCRIPT],
    }
    names = {
        SOURCE_MANIFEST, *manifest["trusted_validators"], *PUBLIC_CHECKERS, *quality["artifact_files"],
        *(name for item in records for name in (*item["dependencies"], item["path"])),
    }
    manifest["files"] = [{"source": name, "destination": name} for name in sorted(names)]
    write_sources(root, manifest, records)
    return {
        "root": root, "manifest": manifest, "records": records, "verifier": published["verifier"],
        "published": published, "control": control,
    }


def _load_dispatcher():
    spec = importlib.util.spec_from_file_location(
        "release_evidence_validators_fixture", ROOT / "scripts/release_evidence_validators.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DISPATCH = _load_dispatcher()


def dispatch(package: dict, records: list[dict] | None = None) -> None:
    """Run the release dispatcher over the package, with the real canonical validators."""

    DISPATCH.validate_records(
        package["root"], package["manifest"], records or package["records"], package["root"], package["verifier"]
    )
