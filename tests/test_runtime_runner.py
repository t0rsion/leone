"""Check Runtime study publication and native build controls."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research/prefix_attention"))
import run_runtime as RUNNER
import runtime_build as BUILD

REAL_RUN = RUNNER.subprocess.run

RETAINED = "the completed run is kept in"


class Session:
    """One fake `run_runtime.main()` call with the build and the driver replaced."""

    def __init__(self, root, output, payload=b"logits", criteria=None, during=None, sources=None):
        self.root = root
        self.builds = []
        self.runs = []
        self.raw_text = None
        self.args = argparse.Namespace(
            backend="cuda", phase="evaluation" if criteria else "calibration",
            path="per_row", model=root / "model", output=output, criteria=criteria,
            repetitions=1, warmups=0,
        )
        self.payload = payload
        self.during = during
        self.sources = sources or (lambda _backend: {})

    def build(self, *_args):
        self.builds.append(True)
        return ["cargo", "clean"], ["cargo", "build"]

    def run(self, command, **kwargs):
        if "--output" not in command:
            return REAL_RUN(command, **kwargs)
        self.runs.append(command)
        raw = Path(command[command.index("--output") + 1])
        sidecar = raw.parent / "receipt.json.sidecars/case.logits.bin"
        sidecar.parent.mkdir()
        sidecar.write_bytes(self.payload)
        self.raw_text = json.dumps({"cases": [{"logits_sidecars": [{
            "file": str(sidecar.relative_to(raw.parent)),
            "sha256": hashlib.sha256(self.payload).hexdigest(),
        }]}]})
        raw.write_text(self.raw_text)
        if self.during is not None:
            self.during(raw)

    def main(self, **overrides):
        binary = self.root / "driver"
        binary.write_bytes(b"fixture driver")
        patches = {
            "BINARY": binary, "parse_args": lambda: self.args,
            "source_hashes": self.sources, "build": self.build,
            "native_environment": lambda _backend: ({}, {}),
            "build_identity": lambda _backend: {},
            **overrides,
        }
        with patch.multiple(RUNNER, **patches), \
                patch.object(RUNNER.subprocess, "run", side_effect=self.run), \
                contextlib.redirect_stdout(io.StringIO()):
            return RUNNER.main()

    def retained(self):
        return sorted(self.root.glob("runtime-*.artifacts"))


class RuntimePublicationTests(unittest.TestCase):
    def published_sidecar(self, output):
        return json.loads(output.read_text())["cases"][0]["logits_sidecars"][0]["file"]

    def test_distinct_outputs_preserve_published_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.json"
            Session(root, first, b"first logits").main(provenance=lambda *_args: {})
            receipt = first.read_bytes()
            second = root / "second.json"
            Session(root, second, b"second logits").main(provenance=lambda *_args: {})
            names = {self.published_sidecar(first), self.published_sidecar(second)}
            self.assertEqual(len(names), 2)
            self.assertEqual(first.read_bytes(), receipt)
            self.assertEqual((root / self.published_sidecar(first)).read_bytes(), b"first logits")
            self.assertEqual((root / self.published_sidecar(second)).read_bytes(), b"second logits")
            self.assertFalse(list(root.glob("runtime-*.artifacts/receipt.json")))

    def test_existing_output_causes_no_build_or_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_bytes(b"winner")
            (root / "directory.json").mkdir()
            (root / "link.json").symlink_to(target)
            (root / "dangling.json").symlink_to(root / "absent.json")
            for name in ("target.json", "directory.json", "link.json", "dangling.json"):
                session = Session(root, root / name)
                with self.subTest(name=name), self.assertRaisesRegex(SystemExit, "existing output"):
                    session.main()
                self.assertEqual((session.builds, session.runs), ([], []))
            self.assertEqual(target.read_bytes(), b"winner")
            self.assertEqual(sorted(root.glob("runtime-*.artifacts")), [])

    def test_unusable_criteria_cause_no_build_or_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "criteria-directory"
            directory.mkdir()
            for criteria in (root / "missing.json", directory):
                session = Session(root, root / "out.json", criteria=criteria)
                with self.subTest(criteria=criteria.name), \
                        self.assertRaisesRegex(SystemExit, "criteria"):
                    session.main()
                self.assertEqual((session.builds, session.runs), ([], []))
            self.assertFalse((root / "out.json").exists())

    def test_late_output_collision_retains_the_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out.json"

            def winner(_raw):
                output.write_bytes(b"winner receipt")

            session = Session(root, output, b"measured logits", during=winner)
            with self.assertRaisesRegex(SystemExit, RETAINED):
                session.main(provenance=lambda *_args: {})
            self.assertEqual(output.read_bytes(), b"winner receipt")
            (directory,) = session.retained()
            self.assertEqual((directory / "receipt.json").read_text(), session.raw_text)
            sidecar = directory / "receipt.json.sidecars/case.logits.bin"
            self.assertEqual(sidecar.read_bytes(), b"measured logits")
            self.assertFalse(list(root.glob(".out.json.*.tmp")))

    def test_source_change_after_the_run_retains_the_completed_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls = []

            def sources(_backend):
                calls.append(True)
                return {"source": "before" if len(calls) <= 2 else "after"}

            session = Session(root, root / "out.json", sources=sources)
            with self.assertRaisesRegex(SystemExit, "identity changed"):
                session.main()
            self.assertFalse((root / "out.json").exists())
            (directory,) = session.retained()
            self.assertEqual((directory / "receipt.json").read_text(), session.raw_text)

    def test_criteria_hash_is_taken_before_work_and_rechecked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            criteria = root / "criteria.json"
            criteria.write_bytes(b"frozen")
            frozen = hashlib.sha256(b"frozen").hexdigest()
            session = Session(root, root / "ok.json", criteria=criteria)
            session.main(provenance=lambda *_args: {})
            self.assertEqual(json.loads((root / "ok.json").read_text())["criteria_sha256"], frozen)

            def rewrite(_raw):
                criteria.write_bytes(b"edited")

            changed = Session(root, root / "changed.json", criteria=criteria, during=rewrite)
            with self.assertRaisesRegex(SystemExit, "criteria file changed"):
                changed.main(provenance=lambda *_args: {})
            self.assertFalse((root / "changed.json").exists())
            self.assertEqual(len(changed.retained()), 2)

    def test_driver_failure_without_a_receipt_removes_the_partial_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def crash(command, **_kwargs):
                raw = Path(command[command.index("--output") + 1])
                (raw.parent / "partial.bin").write_bytes(b"partial")
                raise RUNNER.subprocess.CalledProcessError(1, command)

            with patch.multiple(
                RUNNER, BINARY=root / "driver", parse_args=lambda: Session(root, root / "o.json").args,
                source_hashes=lambda _backend: {}, build=lambda *_args: (["a"], ["b"]),
                native_environment=lambda _backend: ({}, {}), build_identity=lambda _backend: {},
            ), patch.object(RUNNER.subprocess, "run", side_effect=crash):
                (root / "driver").write_bytes(b"fixture driver")
                with self.assertRaises(RUNNER.subprocess.CalledProcessError):
                    RUNNER.main()
            self.assertEqual(sorted(root.glob("runtime-*.artifacts")), [])

    def test_receipt_records_the_actual_argv_through_public_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out.json"
            session = Session(root, output)
            session.main()
            actual = session.runs[0]
            self.assertIn(str(root / "driver"), actual)
            self.assertIn(str(root / "model"), actual)
            record = json.loads(output.read_text())["provenance"]
            manifest = "research/prefix_attention/runtime_manifest.json"
            self.assertEqual(record["command"], RUNNER.affinity_prefix("timed_run") + [
                "$RUNTIME_BINARY", "--phase", "calibration", "--path", "per_row",
                "--manifest", manifest, "--model", "$MODEL", "--output", "$RAW_OUTPUT",
                "--repetitions", "1", "--warmups", "0",
            ])
            self.assertEqual(record["build"]["clean_command"], ["cargo", "clean"])
            self.assertEqual(record["build"]["command"], ["cargo", "build"])
            self.assertNotIn(str(root), output.read_text())

    def test_unmapped_private_command_paths_are_refused(self):
        with self.assertRaisesRegex(ValueError, "private command path"):
            RUNNER.public_command(["/private/tool"], {})

    def test_sidecar_paths_must_be_relative_and_contained(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            directory.mkdir()
            inside = directory / "inside.bin"
            inside.write_bytes(b"logits")
            outside = Path(temporary) / "outside.bin"
            outside.write_bytes(b"logits")
            digest = hashlib.sha256(b"logits").hexdigest()
            for file in (str(inside), "../outside.bin", str(outside)):
                receipt = {"cases": [{"logits_sidecars": [{"file": file, "sha256": digest}]}]}
                with self.subTest(file=file), self.assertRaises(ValueError):
                    RUNNER.qualify_sidecars(receipt, directory)
            receipt = {"cases": [{"logits_sidecars": [{"file": "inside.bin", "sha256": digest}]}]}
            RUNNER.qualify_sidecars(receipt, directory)
            self.assertEqual(receipt["cases"][0]["logits_sidecars"][0]["file"], "run/inside.bin")

    def test_the_removed_build_flag_is_rejected(self):
        argv = ["run_runtime.py", "--phase", "calibration", "--path", "per_row",
                "--model", "m", "--output", "o", "--build"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            RUNNER.parse_args()


def identity(**changes):
    value = {
        "features": "cuda", "profile": "release", "native_provenance": "complete",
        "provenance_unknown": False, "source_tree_dirty": False,
    }
    value.update({name: "a" * 64 for name in RUNNER.IDENTITY_DIGESTS})
    value.update(changes)
    return value


class RuntimeIdentityTests(unittest.TestCase):
    def build_identity(self, value, backend="cuda"):
        with patch.object(RUNNER.subprocess, "check_output", return_value=json.dumps(value)):
            return RUNNER.build_identity(backend)

    def test_complete_identity_is_accepted_with_its_source_state(self):
        accepted = self.build_identity(identity())
        self.assertIs(accepted["source_tree_dirty"], False)

    def test_incomplete_or_invalid_identity_is_rejected(self):
        rejected = {
            "features": identity(features="metal"),
            "profile": identity(profile="debug"),
            "native": identity(native_provenance="incomplete"),
            "unknown provenance": identity(provenance_unknown=True),
            "missing unknown flag": {
                key: value for key, value in identity().items() if key != "provenance_unknown"
            },
            "dirty flag not boolean": identity(source_tree_dirty="false"),
            "dirty tree": identity(source_tree_dirty=True),
            "wrapper or toolchain": identity(toolchain_sha256="unknown"),
            "config": identity(build_config_sha256="unknown"),
            "source tree": identity(source_tree_sha256="unknown"),
            "native tools": identity(native_tools_sha256="0" * 63),
            "compiler": identity(rustc_sha256="A" * 64),
        }
        for name, value in rejected.items():
            with self.subTest(name=name), self.assertRaisesRegex(SystemExit, "identity rejected"):
                self.build_identity(value)

    def test_selected_feature_must_match_the_backend(self):
        with self.assertRaisesRegex(SystemExit, "built for 'cuda', not 'metal'"):
            self.build_identity(identity(), backend="metal")

    def test_source_map_covers_the_driver_and_the_runtime(self):
        mapped = set(RUNNER.source_paths("cuda"))
        driver = RUNNER.DRIVER
        runtime = {
            *driver.joinpath("src").glob("*.rs"),
            *RUNNER.ROOT.joinpath("crates/leone/src").rglob("*.rs"),
            driver / "Cargo.toml", driver / "build.rs", driver / "Cargo.lock",
        }
        self.assertLessEqual(runtime, mapped)
        for path in mapped:
            self.assertTrue(path.is_file(), path)

    def test_driver_release_profile_matches_the_root_workspace(self):
        def profile(path):
            return tomllib.loads(path.read_text())["profile"]["release"]

        self.assertEqual(
            profile(RUNNER.DRIVER / "Cargo.toml"), profile(RUNNER.ROOT / "Cargo.toml")
        )

    def test_driver_target_directory_is_ignored(self):
        self.assertEqual((RUNNER.DRIVER / ".gitignore").read_text(), "/target/\n")


class RuntimeBuildTests(unittest.TestCase):
    def test_sdk_alias_preserves_the_selected_sdk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sdk = root / "MacOSX.sdk"
            sdk.mkdir()
            alias = root / "MacOSX27.sdk"
            alias.symlink_to(sdk)
            environment = {"SDKROOT": str(sdk)}
            BUILD.set_native_input(environment, "SDKROOT", str(alias))
            self.assertEqual(environment["SDKROOT"], str(alias))
            other = root / "Other.sdk"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "SDKROOT"):
                BUILD.set_native_input(environment, "SDKROOT", str(other))

    def test_cuda_controls_reject_unrecorded_nvcc_options(self):
        for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                BUILD.cuda_controls({name: "override"})

    def test_cuda_host_compiler_and_lineinfo_change_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            compiler = Path(temporary) / "compiler"
            compiler.write_bytes(b"compiler one")
            with patch.object(BUILD.shutil, "which", return_value=str(compiler)):
                first = BUILD.cuda_controls({})
                self.assertNotEqual(first, BUILD.cuda_controls({"LEONE_CUDA_LINEINFO": "1"}))
                compiler.write_bytes(b"compiler two")
                self.assertNotEqual(first, BUILD.cuda_controls({}))

    def test_metal_controls_bind_tools_sdk_and_deployment(self):
        with tempfile.TemporaryDirectory() as temporary:
            tool = Path(temporary) / "tool"
            tool.write_bytes(b"compiler one")
            values = {"clang": str(tool), "ar": str(tool), "--show-sdk-path": temporary,
                      "--show-sdk-version": "27.0", "--show-sdk-build-version": "first",
                      "-productVersion": "27.0"}
            with patch.object(BUILD, "command_text", side_effect=lambda command: values[command[-1]]):
                environment = {}
                first = BUILD.metal_controls(environment)
                self.assertEqual(environment["CC"], str(tool))
                self.assertEqual(environment["SDKROOT"], temporary)
                self.assertEqual(environment["CARGO_PROFILE_RELEASE_STRIP"], "none")
                values["--show-sdk-build-version"] = "second"
                self.assertNotEqual(first, BUILD.metal_controls({}))
                values["--show-sdk-build-version"] = "first"
                tool.write_bytes(b"compiler two")
                self.assertNotEqual(first, BUILD.metal_controls({}))
                values["-productVersion"] = "27.1"
                self.assertNotEqual(first, BUILD.metal_controls({}))

    def test_metal_rejects_uncontrolled_native_tools(self):
        names = (
            "CC_aarch64_apple_darwin", "HOST_CC", "AR_aarch64_apple_darwin",
            "DEVELOPER_DIR", "RANLIB", "CRATE_CC_NO_DEFAULTS", "RANLIBFLAGS",
            "CFLAGS_aarch64-apple-darwin", "CFLAGS_aarch64_apple_darwin",
            "CXXFLAGS_aarch64_apple_darwin", "CPPFLAGS_aarch64_apple_darwin",
            "OBJCFLAGS_aarch64_apple_darwin", "ARFLAGS_aarch64_apple_darwin",
            "RANLIBFLAGS_aarch64_apple_darwin",
        )
        for name in names:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                BUILD.metal_controls({name: "override"})


if __name__ == "__main__":
    unittest.main()
