"""Check that prefix runner provenance does not publish local paths."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_cuda
import run_metal
import run_production
from runner_support import SOURCE_LAYOUT, publish_json


def command_output(command: list[str], **_: object) -> str:
    if command[:2] == ["git", "rev-parse"]:
        return "fixture-revision\n"
    if command[:2] == ["git", "status"]:
        return ""
    return "fixture-tool-version\n"


class RunnerProvenanceTests(unittest.TestCase):
    def assert_public(self, value: dict, private_paths: list[Path]) -> None:
        serialized = json.dumps(value)
        for path in private_paths:
            self.assertNotIn(str(path), serialized)

    def test_cuda_provenance_replaces_absolute_paths_and_compiler(self) -> None:
        binary = Path("/private/user/build/prefix_attention_cuda")
        raw = Path("/private/user/receipts/raw.json")
        build = [str(run_cuda.CUDA_BUILDER), str(binary)]
        command = [str(binary), "--phase", "calibration", "--output", str(raw)]
        with patch.dict(run_cuda.os.environ, {"NVCC": "/private/toolchain/bin/nvcc"}), \
             patch.object(run_cuda.subprocess, "check_output", side_effect=command_output):
            provenance = run_cuda.provenance(raw, binary, build, command, {}, "artifact")
        self.assertEqual(provenance["build"]["command"], ["$CUDA_BUILDER", "$CUDA_BINARY"])
        self.assertEqual(provenance["source_layout"], SOURCE_LAYOUT)
        self.assertEqual(provenance["command"], ["$CUDA_BINARY", "--phase", "calibration", "--output", "$RAW_OUTPUT"])
        self.assertEqual(provenance["build"]["nvcc"], "$NVCC")
        self.assert_public(provenance, [binary, raw, Path("/private/toolchain/bin/nvcc")])

    def test_metal_provenance_replaces_absolute_paths(self) -> None:
        metallib = Path("/private/user/build/prefix_attention.metallib")
        binary = Path("/private/user/build/prefix_attention_metal")
        raw = Path("/private/user/receipts/raw.json")
        build = [str(run_metal.BUILDER), str(metallib), str(binary)]
        command = [str(binary), "--metallib", str(metallib), "--output", str(raw)]
        with patch.object(run_metal.subprocess, "check_output", side_effect=command_output), \
             patch.object(run_metal, "tool_version", return_value="fixture-tool-version\n"):
            provenance = run_metal.make_provenance(
                raw, metallib, binary, build, command, {}, {"runner": "runner", "metallib": "library"}
            )
        self.assertEqual(
            provenance["build"]["command"],
            ["$METAL_BUILDER", "$METALLIB", "$METAL_BINARY"],
        )
        self.assertEqual(
            provenance["command"],
            ["$METAL_BINARY", "--metallib", "$METALLIB", "--output", "$RAW_OUTPUT"],
        )
        self.assert_public(provenance, [metallib, binary, raw])

    def test_production_provenance_replaces_absolute_paths(self) -> None:
        raw = Path("/private/user/receipts/raw.json")
        manifest = run_production.DRIVER_ROOT / "Cargo.toml"
        build = ["cargo", "+1.92", "build", "--manifest-path", str(manifest)]
        command = [str(run_production.BINARY), "--phase", "calibration", "--output", str(raw)]
        with patch.object(run_production.subprocess, "check_output", side_effect=command_output):
            provenance = run_production.provenance(raw, build, command, {}, "artifact")
        self.assertEqual(
            provenance["build"]["command"],
            ["cargo", "+1.92", "build", "--manifest-path", "$DRIVER_MANIFEST"],
        )
        self.assertEqual(
            provenance["command"],
            ["$PRODUCTION_BINARY", "--phase", "calibration", "--output", "$RAW_OUTPUT"],
        )
        self.assert_public(provenance, [manifest, run_production.BINARY, raw])

    def test_unmapped_absolute_command_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmapped private command path") as error:
            run_cuda.public_command(["/private/unmapped/tool"], {})
        self.assertNotIn("/private/unmapped/tool", str(error.exception))

    def test_role_serialization_is_reproducible_across_worktrees(self) -> None:
        def render(root: Path) -> list[str]:
            builder = root / "cuda/build_cuda.sh"
            binary = root / "outputs/prefix_attention_cuda"
            raw = root / "receipts/raw.json"
            return run_cuda.public_command(
                [str(builder), str(binary), "--output", str(raw)],
                {str(builder): "$CUDA_BUILDER", str(binary): "$CUDA_BINARY", str(raw): "$RAW_OUTPUT"},
            )

        self.assertEqual(render(Path("/private/first")), render(Path("/private/second")))
        self.assertEqual(
            render(Path("/private/first")),
            ["$CUDA_BUILDER", "$CUDA_BINARY", "--output", "$RAW_OUTPUT"],
        )

    def test_publication_refuses_collision_without_replacing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            original = b'{"existing":true}\n'
            output.write_bytes(original)
            with self.assertRaisesRegex(FileExistsError, "refusing to replace"):
                publish_json(output, {"replacement": True})
            self.assertEqual(output.read_bytes(), original)

    def test_concurrent_publication_has_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "receipt.json"
            barrier = threading.Barrier(2)

            def publish(value: str) -> str:
                barrier.wait()
                try:
                    publish_json(output, {"writer": value})
                except FileExistsError:
                    return "collision"
                return "published"

            with ThreadPoolExecutor(max_workers=2) as workers:
                results = list(workers.map(publish, ("first", "second")))
            self.assertEqual(sorted(results), ["collision", "published"])
            self.assertIn(json.loads(output.read_text())["writer"], {"first", "second"})


if __name__ == "__main__":
    unittest.main()
