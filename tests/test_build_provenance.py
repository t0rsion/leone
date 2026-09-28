"""Check that build provenance is bound by the compiler build script."""

import json
from pathlib import Path
import os
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "build-support" / "provenance.rs"
CARGO = ["cargo", "+1.92", "build", "--offline", "--manifest-path"]


class BuildProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self._write_fixture()
        self._run(["cargo", "+1.92", "generate-lockfile", "--offline", "--manifest-path", str(self.manifest)])
        self._run(["git", "add", "."])
        self._run(
            [
                "git",
                "-c",
                "user.email=fixture",
                "-c",
                "user.name=fixture",
                "commit",
                "-qm",
                "fixture",
            ]
        )

    def _write_fixture(self):
        app = self.root / "crates" / "app"
        (app / "src").mkdir(parents=True)
        (self.root / "build-support").mkdir()
        shutil.copy(HELPER, self.root / "build-support" / "provenance.rs")
        (self.root / "Cargo.toml").write_text(
            """[workspace]
members = [\"crates/app\"]
resolver = \"2\"

[workspace.dependencies]
sha2 = \"0.10.9\"
"""
        )
        (app / "Cargo.toml").write_text(
            """[package]
name = \"fixture-app\"
version = \"0.1.0\"
edition = \"2021\"
build = \"build.rs\"

[features]
metal = []

[build-dependencies]
sha2.workspace = true
"""
        )
        (app / "build.rs").write_text(
            """use std::env;
use std::path::Path;

#[path = \"../../build-support/provenance.rs\"]
mod provenance;

fn main() {
    let root = Path::new(env!(\"CARGO_MANIFEST_DIR\"))
        .parent()
        .and_then(Path::parent)
        .expect(\"fixture package is under crates\");
    provenance::emit(root);
}
"""
        )
        (app / "src" / "main.rs").write_text(
            """fn main() {
    println!(\"{{\\\"source_commit\\\":\\\"{}\\\",\\\"source_tree_dirty\\\":\\\"{}\\\",\\\"source_tree_sha256\\\":\\\"{}\\\",\\\"rustc\\\":\\\"{}\\\",\\\"rustc_sha256\\\":\\\"{}\\\",\\\"toolchain_sha256\\\":\\\"{}\\\",\\\"tool_versions\\\":\\\"{}\\\",\\\"features\\\":\\\"{}\\\",\\\"build_flags\\\":\\\"{}\\\",\\\"build_flags_sha256\\\":\\\"{}\\\",\\\"build_flags_raw_sha256\\\":\\\"{}\\\",\\\"build_flags_redacted\\\":\\\"{}\\\",\\\"build_flags_redaction_count\\\":\\\"{}\\\",\\\"profile_inputs\\\":\\\"{}\\\",\\\"profile_inputs_sha256\\\":\\\"{}\\\",\\\"linker_inputs\\\":\\\"{}\\\",\\\"linker_inputs_sha256\\\":\\\"{}\\\",\\\"native_provenance\\\":\\\"{}\\\",\\\"native_tools_sha256\\\":\\\"{}\\\",\\\"native_tool_versions\\\":\\\"{}\\\",\\\"build_config_sha256\\\":\\\"{}\\\",\\\"provenance_unknown\\\":\\\"{}\\\"}}\", env!(\"LEONE_SOURCE_COMMIT\"), env!(\"LEONE_SOURCE_DIRTY\"), env!(\"LEONE_SOURCE_TREE_SHA256\"), env!(\"LEONE_BUILD_RUSTC\"), env!(\"LEONE_BUILD_RUSTC_SHA256\"), env!(\"LEONE_BUILD_TOOLCHAIN_SHA256\"), env!(\"LEONE_BUILD_TOOL_VERSIONS\"), env!(\"LEONE_BUILD_FEATURES\"), env!(\"LEONE_BUILD_FLAGS\"), env!(\"LEONE_BUILD_FLAGS_SHA256\"), env!(\"LEONE_BUILD_FLAGS_RAW_SHA256\"), env!(\"LEONE_BUILD_FLAGS_REDACTED\"), env!(\"LEONE_BUILD_FLAGS_REDACTION_COUNT\"), env!(\"LEONE_BUILD_PROFILE_INPUTS\"), env!(\"LEONE_BUILD_PROFILE_INPUTS_SHA256\"), env!(\"LEONE_BUILD_LINKER_INPUTS\"), env!(\"LEONE_BUILD_LINKER_INPUTS_SHA256\"), env!(\"LEONE_BUILD_NATIVE_PROVENANCE\"), env!(\"LEONE_BUILD_NATIVE_TOOLS_SHA256\"), env!(\"LEONE_BUILD_NATIVE_TOOL_VERSIONS\"), env!(\"LEONE_BUILD_CONFIG_SHA256\"), env!(\"LEONE_BUILD_PROVENANCE_UNKNOWN\"));
}
"""
        )
        self.manifest = app / "Cargo.toml"
        self.target = self.root / "target"
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "fixture"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "fixture"], cwd=self.root, check=True)

    def _run(self, command, env=None):
        completed = subprocess.run(command, cwd=self.root, env=env, check=True, text=True, capture_output=True)
        return completed.stdout

    def _build(self, env=None, target=None, features=None):
        build_env = os.environ.copy()
        if env:
            build_env.update(env)
        build_env["CARGO_TARGET_DIR"] = str(target or self.target)
        command = CARGO + [str(self.manifest)]
        if features:
            command.extend(["--features", ",".join(features)])
        self._run(command, env=build_env)
        binary = Path(build_env["CARGO_TARGET_DIR"]) / "debug" / "fixture-app"
        return json.loads(subprocess.check_output([str(binary)], text=True, env=build_env))

    def test_clean_dirty_and_source_digest_binding(self):
        clean = self._build({"CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=1\x1f--remap-path-prefix=/private=/source"})
        self.assertEqual(clean["source_tree_dirty"], "false")
        self.assertEqual(clean["provenance_unknown"], "false")
        self.assertEqual(len(clean["source_tree_sha256"]), 64)
        self.assertIn("-C opt-level=1", clean["build_flags"])
        self.assertNotIn("/private", clean["build_flags"])
        self.assertNotIn("/source", clean["build_flags"])
        self.assertEqual(clean["build_flags_redacted"], "true")
        self.assertGreater(int(clean["build_flags_redaction_count"]), 0)
        self.assertEqual(len(clean["build_flags_sha256"]), 64)
        self.assertEqual(len(clean["build_flags_raw_sha256"]), 64)
        self.assertEqual(len(clean["rustc_sha256"]), 64)
        self.assertEqual(len(clean["toolchain_sha256"]), 64)
        self.assertIn("RUSTC_LINKER", clean["linker_inputs"])
        self.assertEqual(clean["native_provenance"], "unavailable")
        self.assertEqual(len(clean["build_config_sha256"]), 64)
        self.assertIn("rustc ", clean["rustc"])

        other_flags = self._build(
            {"CARGO_ENCODED_RUSTFLAGS": "-C\x1fopt-level=1\x1f--remap-path-prefix=/other=/source"}
        )
        self.assertNotEqual(clean["build_flags_raw_sha256"], other_flags["build_flags_raw_sha256"])

        native_a = self._build({"CC": "/private/tool-a"}, self.root / "native-a")
        native_b = self._build({"CC": "/private/tool-b"}, self.root / "native-b")
        self.assertNotEqual(native_a["build_config_sha256"], native_b["build_config_sha256"])

        digest = clean["source_tree_sha256"]
        source = self.root / "crates" / "app" / "src" / "main.rs"
        source.write_text(source.read_text() + "\n// source changed\n")
        dirty = self._build()
        self.assertEqual(dirty["source_tree_dirty"], "true")
        self.assertNotEqual(dirty["source_tree_sha256"], digest)

        self._run(["git", "add", "."])
        self._run(
            [
                "git",
                "-c",
                "user.email=fixture",
                "-c",
                "user.name=fixture",
                "commit",
                "-qm",
                "changed",
            ]
        )
        committed = self._build()
        expected_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        self.assertEqual(committed["source_commit"], expected_commit)
        self.assertEqual(committed["source_tree_dirty"], "false")

    def test_profile_and_tool_identity_binding(self):
        profile_a = self._build(
            {"CARGO_PROFILE_DEV_LTO": "false"}, self.root / "profile-a"
        )
        profile_b = self._build(
            {"CARGO_PROFILE_DEV_LTO": "true"}, self.root / "profile-b"
        )
        self.assertIn("CARGO_PROFILE_DEV_LTO", profile_a["profile_inputs"])
        self.assertNotEqual(profile_a["profile_inputs_sha256"], profile_b["profile_inputs_sha256"])
        self.assertNotEqual(profile_a["build_config_sha256"], profile_b["build_config_sha256"])

        real_rustc = shutil.which("rustc")
        self.assertIsNotNone(real_rustc)
        compiler = self.root / "same-rustc"
        compiler.write_text(f"#!/bin/sh\n# first\nexec {real_rustc} \"$@\"\n")
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        compiler_a = self._build(
            {"RUSTC": str(compiler)}, self.root / "compiler-a"
        )
        compiler.write_text(f"#!/bin/sh\n# second\nexec {real_rustc} \"$@\"\n")
        compiler_b = self._build(
            {"RUSTC": str(compiler)}, self.root / "compiler-b"
        )
        self.assertNotEqual(compiler_a["rustc_sha256"], compiler_b["rustc_sha256"])
        self.assertNotEqual(compiler_a["toolchain_sha256"], compiler_b["toolchain_sha256"])

        wrapper = self.root / "rustc-wrapper"
        wrapper.write_text("#!/bin/sh\n# first\nexec \"$@\"\n")
        wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
        wrapper_a = self._build(
            {"RUSTC_WRAPPER": str(wrapper)}, self.root / "wrapper-a"
        )
        wrapper.write_text("#!/bin/sh\n# second\nexec \"$@\"\n")
        wrapper_b = self._build(
            {"RUSTC_WRAPPER": str(wrapper)}, self.root / "wrapper-b"
        )
        self.assertNotEqual(wrapper_a["toolchain_sha256"], wrapper_b["toolchain_sha256"])

        missing = self._build(
            {"CC": str(self.root / "missing-cc")},
            self.root / "native-incomplete",
            features=["metal"],
        )
        self.assertEqual(missing["native_provenance"], "incomplete")
        self.assertEqual(missing["provenance_unknown"], "true")

        real_cc = shutil.which("cc")
        self.assertIsNotNone(real_cc)
        cc = self.root / "cc-wrapper"
        cc.write_text(f"#!/bin/sh\nexec {real_cc} \"$@\"\n")
        cc.chmod(cc.stat().st_mode | stat.S_IXUSR)
        native = self._build(
            {"CC": str(cc)}, self.root / "native-complete", features=["metal"]
        )
        self.assertEqual(native["native_provenance"], "complete")
        self.assertEqual(len(native["native_tools_sha256"]), 64)

    def test_tool_versions_redact_paths_but_bind_raw_values(self):
        compiler = self.root / "version-compiler"
        compiler.write_text(
            '#!/bin/sh\nprintf "compiler 1.0 InstalledDir: %s\\n" "$LEONE_TEST_TOOL_DIR"\n'
        )
        compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
        first = self._build(
            {"CC": str(compiler), "LEONE_TEST_TOOL_DIR": "/private/compiler-a"},
            self.root / "version-a", features=["metal"],
        )
        second = self._build(
            {"CC": str(compiler), "LEONE_TEST_TOOL_DIR": "/private/compiler-b"},
            self.root / "version-b", features=["metal"],
        )
        self.assertEqual(first["native_provenance"], "complete")
        self.assertNotIn("/private", first["native_tool_versions"])
        self.assertIn("<redacted>", first["native_tool_versions"])
        self.assertEqual(first["native_tool_versions"], second["native_tool_versions"])
        self.assertNotEqual(first["native_tools_sha256"], second["native_tools_sha256"])

    def test_ignored_source_is_incomplete(self):
        (self.root / ".git" / "info" / "exclude").write_text("crates/app/src/ignored.rs\n")
        (self.root / "crates" / "app" / "src" / "ignored.rs").write_text("const X: u8 = 1;\n")
        ignored = self._build(target=self.root / "ignored")
        self.assertEqual(ignored["source_tree_dirty"], "true")
        self.assertEqual(ignored["source_tree_sha256"], "unknown")
        self.assertEqual(ignored["provenance_unknown"], "true")

    def test_unknown_git_fails_closed(self):
        fake_bin = self.root / "fake-bin"
        fake_bin.mkdir()
        fake_git = fake_bin / "git"
        fake_git.write_text("#!/bin/sh\nexit 1\n")
        fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR)
        environment = {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
        unknown = self._build(environment, self.root / "unknown-target")
        self.assertEqual(unknown["source_commit"], "unknown")
        self.assertEqual(unknown["source_tree_dirty"], "true")
        self.assertEqual(unknown["source_tree_sha256"], "unknown")
        self.assertEqual(unknown["provenance_unknown"], "true")


if __name__ == "__main__":
    unittest.main()
