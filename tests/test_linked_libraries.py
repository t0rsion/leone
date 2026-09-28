"""Check the resolved library closure on both platforms without a real llama.cpp build."""

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("linked_libraries", ROOT / "scripts/linked_libraries.py")
LIBRARIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIBRARIES)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class FamilyTests(unittest.TestCase):
    def test_versioned_names_map_to_one_family_per_platform(self):
        for name, family in (
            ("libllama.so.0", "libllama"), ("libggml-base.so.0.21.0", "libggml-base"),
            ("libggml-cuda.so.0.21.0", "libggml-cuda"), ("libllama.0.0.1234.dylib", "libllama"),
            ("libggml-metal.dylib", "libggml-metal"), ("libllama-server-impl.so.0", "libllama-server-impl"),
        ):
            self.assertEqual(LIBRARIES.family(name), family, name)
        self.assertIsNone(LIBRARIES.family("llama-server"))

    def test_the_core_set_names_the_active_backend_and_no_server_layer(self):
        self.assertEqual(LIBRARIES.core_families("cuda")[-1], "libggml-cuda")
        self.assertEqual(LIBRARIES.core_families("MTL")[-1], "libggml-metal")
        self.assertEqual(LIBRARIES.core_families("cpu"), LIBRARIES.CORE_FAMILIES)
        for server in ("libllama-server-impl", "libllama-common", "libmtmd"):
            self.assertNotIn(server, LIBRARIES.core_families("cuda"))

    def test_records_use_name_or_the_older_path_key(self):
        records = [
            {"name": "libggml.so.0", "sha256": "a" * 64},
            {"path": "libggml-cpu.so.0", "sha256": "b" * 64},
            {"name": "libmtmd.so.0", "sha256": "c" * 64},
        ]
        self.assertEqual(
            LIBRARIES.core_hashes(records, "cuda"),
            {"libggml": {"a" * 64}, "libggml-cpu": {"b" * 64}},
        )

    def test_invalid_core_digests_are_not_treated_as_hashes(self):
        records = [
            {"name": "libllama.so.0", "sha256": None},
            {"name": "libggml.so.0", "sha256": "garbage"},
        ]
        self.assertEqual(LIBRARIES.core_hashes(records, "cuda"), {})
        self.assertEqual(LIBRARIES.invalid_core_families(records, "cuda"), {"libllama", "libggml"})


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)

    def library(self, name):
        path = self.root / name
        path.write_bytes(name.encode())
        return path

    def test_linux_resolves_ldd_and_adds_the_backend_libraries_ldd_omits(self):
        binary = self.library("llama-server")
        llama, ggml, cuda = self.library("libllama.so.0"), self.library("libggml.so.0"), self.library("libggml-cuda.so.0.21.0")
        output = f"\tlinux-vdso.so.1 (0x1)\n\tlibllama.so.0 => {llama} (0x2)\n\tlibggml.so.0 => {ggml} (0x3)\n\t/lib64/ld-linux.so (0x4)\n"
        # The loader path in the last ldd line does not exist here, so it must fail.
        with self.assertRaises(LIBRARIES.LinkageError):
            LIBRARIES.resolve(binary, self.root, "cuda", "linux", lambda *_a, **_k: output)
        records = LIBRARIES.resolve(binary, self.root, "cuda", "linux", lambda *_a, **_k: output.replace("\t/lib64/ld-linux.so (0x4)\n", ""))
        self.assertEqual(records, [
            {"name": "libggml-cuda.so.0.21.0", "sha256": sha(cuda.name.encode())},
            {"name": "libggml.so.0", "sha256": sha(ggml.name.encode())},
            {"name": "libllama.so.0", "sha256": sha(llama.name.encode())},
        ])

    def test_linux_rejects_an_unresolved_library(self):
        binary = self.library("llama-server")
        with self.assertRaises(LIBRARIES.LinkageError):
            LIBRARIES.resolve(binary, self.root, "cpu", "linux", lambda *_a, **_k: "\tlibx.so => not found\n")

    def test_darwin_follows_the_closure_because_otool_lists_direct_dependencies_only(self):
        launcher = self.library("llama-server")
        for name in ("libllama.dylib", "libggml.dylib", "libggml-base.dylib", "libggml-metal.dylib"):
            self.library(name)
        graph = {
            "llama-server": ["@rpath/libllama.dylib", "/usr/lib/libSystem.B.dylib"],
            "libllama.dylib": ["@rpath/libllama.dylib", "@loader_path/libggml.dylib"],
            "libggml.dylib": ["@loader_path/libggml-base.dylib"],
            "libggml-base.dylib": ["/usr/lib/libc++.1.dylib"],
        }

        def otool(command, **_kwargs):
            references = graph.get(pathlib.Path(command[2]).name, [])
            return "\n".join([f"{command[2]}:", *[f"\t{item} (compatibility version 1.0.0)" for item in references]])

        records = LIBRARIES.resolve(launcher, self.root, "metal", "darwin", otool)
        self.assertEqual([item["name"] for item in records], ["libggml-base.dylib", "libggml-metal.dylib", "libggml.dylib", "libllama.dylib"])
        self.assertEqual({item["name"]: item["sha256"] for item in records}["libggml.dylib"], sha(b"libggml.dylib"))
        core = LIBRARIES.core_hashes(records, "metal")
        self.assertEqual(set(core), set(LIBRARIES.core_families("metal")) - {"libggml-cpu"})

    def test_a_darwin_dependency_that_is_absent_is_an_error_but_a_system_library_is_not(self):
        launcher = self.library("llama-server")
        run = lambda command, **_k: f"{command[2]}:\n\t@rpath/libmissing.dylib (x)\n"
        with self.assertRaises(LIBRARIES.LinkageError):
            LIBRARIES.resolve(launcher, self.root, "cpu", "darwin", run)

    def test_the_command_line_prints_json_records_sorted_by_name(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts/linked_libraries.py"), sys.executable, str(pathlib.Path(sys.executable).parent), "cpu"],
            capture_output=True, text=True, check=True,
        )
        records = json.loads(completed.stdout)
        self.assertEqual(records, sorted(records, key=lambda item: item["name"]))
        self.assertTrue(all(set(item) == {"name", "sha256"} for item in records))
        self.assertEqual(completed.stdout, json.dumps(records) + "\n")


if __name__ == "__main__":
    unittest.main()
