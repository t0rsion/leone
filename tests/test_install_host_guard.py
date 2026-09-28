"""Test archive target validation before installation changes the destination."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "packaging/install.sh"


class InstallHostGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="leone install guard ")
        self.root = Path(self.temporary.name) / "archive root with spaces"
        self.root.mkdir()
        (self.root / "bin").mkdir()
        (self.root / "bin/leone").write_bytes(b"archive binary\n")
        (self.root / "bin/leone").chmod(0o755)
        shutil.copy2(INSTALLER, self.root / "install.sh")
        (self.root / "install.sh").chmod(0o755)
        self.shim = Path(self.temporary.name) / "uname-shim"
        self.shim.mkdir()
        self._write_uname_shim()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_uname_shim(self) -> None:
        script = self.shim / "uname"
        script.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "    -s) printf '%s\\n' \"$LEONE_TEST_UNAME_S\" ;;\n"
            "    -m) printf '%s\\n' \"$LEONE_TEST_UNAME_M\" ;;\n"
            "    *) exit 2 ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        script.chmod(0o755)

    def _write_marker(self, target: str) -> None:
        marker = self.root / "archive-target"
        marker.write_text(f"{target}\n", encoding="utf-8")
        marker.chmod(0o644)

    def _run_installer(
        self, prefix: Path, host: tuple[str, str], check: bool = False
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PATH"] = f"{self.shim}{os.pathsep}{environment['PATH']}"
        environment["PREFIX"] = str(prefix)
        environment["LEONE_TEST_UNAME_S"], environment["LEONE_TEST_UNAME_M"] = host
        return subprocess.run(
            [str(self.root / "install.sh")],
            env=environment,
            check=check,
            capture_output=True,
            text=True,
        )

    def test_direct_marker_fixture_has_stable_mode(self) -> None:
        self._write_marker("x86_64-unknown-linux-gnu")
        marker = self.root / "archive-target"
        self.assertEqual(marker.read_bytes(), b"x86_64-unknown-linux-gnu\n")
        self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o644)

    def test_matching_hosts_install_from_path_with_spaces(self) -> None:
        cases = (
            ("x86_64-unknown-linux-gnu", ("Linux", "x86_64")),
            ("aarch64-apple-darwin", ("Darwin", "arm64")),
        )
        for target, host in cases:
            with self.subTest(target=target):
                marker = self.root / "archive-target"
                marker.unlink(missing_ok=True)
                self._write_marker(target)
                prefix = Path(self.temporary.name) / f"install destination {target}"
                self._run_installer(prefix, host, check=True)
                installed = prefix / "bin/leone"
                self.assertEqual(installed.read_bytes(), b"archive binary\n")
                self.assertEqual(stat.S_IMODE(installed.stat().st_mode), 0o755)

    def test_wrong_hosts_leave_existing_destination_unchanged(self) -> None:
        cases = (
            ("x86_64-unknown-linux-gnu", ("Darwin", "arm64")),
            ("aarch64-apple-darwin", ("Linux", "x86_64")),
            ("aarch64-apple-darwin", ("Darwin", "x86_64")),
        )
        for target, host in cases:
            with self.subTest(target=target, host=host):
                marker = self.root / "archive-target"
                marker.unlink(missing_ok=True)
                self._write_marker(target)
                prefix = Path(self.temporary.name) / f"stale destination {target} {host[0]} {host[1]}"
                stale = prefix / "bin/leone"
                stale.parent.mkdir(parents=True)
                stale.write_bytes(b"stale binary\n")
                stale.chmod(0o711)
                result = self._run_installer(prefix, host)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(stale.read_bytes(), b"stale binary\n")
                self.assertEqual(stat.S_IMODE(stale.stat().st_mode), 0o711)
                self.assertFalse((prefix / "share").exists())

    def test_missing_and_malformed_markers_fail_without_execution(self) -> None:
        prefix = Path(self.temporary.name) / "missing marker destination"
        result = self._run_installer(prefix, ("Linux", "x86_64"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(prefix.exists())

        marker = self.root / "archive-target"
        malformed = (
            b"x86_64-unknown-linux-gnu",
            b"x86_64-unknown-linux-gnu\n\n",
            b"x86_64-unknown-linux-gnu\n"
            + f"printf pwned >'{self.temporary.name}/marker side effect'\n".encode(),
        )
        for index, content in enumerate(malformed):
            with self.subTest(marker=index):
                marker.write_bytes(content)
                prefix = Path(self.temporary.name) / f"malformed marker destination {index}"
                result = self._run_installer(prefix, ("Linux", "x86_64"))
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(prefix.exists())
        self.assertFalse((Path(self.temporary.name) / "marker side effect").exists())

    def test_symlink_marker_fails_without_creating_destination(self) -> None:
        marker = self.root / "archive-target"
        outside = Path(self.temporary.name) / "outside-marker"
        outside.write_text("x86_64-unknown-linux-gnu\n", encoding="utf-8")
        marker.symlink_to(outside)
        prefix = Path(self.temporary.name) / "symlink marker destination"
        result = self._run_installer(prefix, ("Linux", "x86_64"))
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(prefix.exists())

    def test_package_fixture_stages_platform_target(self) -> None:
        repository = Path(self.temporary.name) / "package fixture repository"
        subprocess.run(
            ["git", "clone", "--shared", "--no-hardlinks", str(ROOT), str(repository)],
            check=True,
            capture_output=True,
            text=True,
        )
        tracked_delta = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if tracked_delta:
            subprocess.run(
                ["git", "apply", "--binary", "-"],
                cwd=repository,
                check=True,
                input=tracked_delta,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Leone fixture",
                "-c",
                "user.email=fixture@invalid",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "Prepare package fixture",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

        environment = os.environ.copy()
        environment["CARGO"] = str(repository / "tests/fixtures/fake-cargo")
        environment["LEONE_FAKE_BINARY"] = str(repository / "tests/fixtures/fake-leone")
        environment["SOURCE_DATE_EPOCH"] = "1700000000"
        for platform, target in (
            ("linux-x86_64", "x86_64-unknown-linux-gnu"),
            ("darwin-arm64", "aarch64-apple-darwin"),
        ):
            with self.subTest(platform=platform):
                distribution = Path(self.temporary.name) / f"dist {platform}"
                environment["LEONE_PACKAGE_DIST"] = str(distribution)
                environment["LEONE_FAKE_CARGO_LOG"] = str(
                    Path(self.temporary.name) / f"cargo args {platform}"
                )
                result = subprocess.run(
                    [
                        str(repository / "scripts/package-release.sh"),
                        "--platform",
                        platform,
                        "--runtime-only",
                    ],
                    cwd=repository,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                archives = sorted(distribution.glob(f"leone-*-{platform}.tar.gz"))
                self.assertEqual(len(archives), 1)
                with tarfile.open(archives[0], "r:gz") as bundle:
                    markers = [
                        member
                        for member in bundle.getmembers()
                        if member.name.endswith("/archive-target")
                    ]
                    self.assertEqual(len(markers), 1)
                    self.assertEqual(markers[0].mode & 0o777, 0o644)
                    marker = bundle.extractfile(markers[0])
                    self.assertIsNotNone(marker)
                    self.assertEqual(marker.read(), f"{target}\n".encode())


if __name__ == "__main__":
    unittest.main()
