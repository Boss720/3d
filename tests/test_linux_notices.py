import hashlib
import json
import os
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch

from scripts import collect_linux_notices


class FakeDistribution:
    def __init__(self, name, version, root, files):
        self.metadata = {"Name": name}
        self.version = version
        self.root = Path(root)
        self.files = files

    def locate_file(self, relative):
        return self.root / str(relative)


class LinuxNoticeCollectorTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(__file__).resolve().parents[1] / ".runtime"
        self.work = self.runtime / "linux-notice-tests" / uuid.uuid4().hex
        self.work.mkdir(parents=True)

    def tearDown(self):
        if self.work.exists():
            target = self.runtime / "_quarantine" / "linux-notice-tests" / self.work.name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.work, target)

    def test_collector_copies_real_license_files_and_records_hashes(self):
        site = self.work / "site-packages"
        (site / "demo_pkg-1.2.dist-info").mkdir(parents=True)
        license_path = site / "demo_pkg-1.2.dist-info" / "licenses" / "LICENSE.txt"
        license_path.parent.mkdir()
        license_path.write_text("Demo license\n", encoding="utf-8")
        third_party_files = {
            "ThirdPartyNotices.txt": "ONNX bundled notice\n",
            "third_party_notices.txt": "Underscore notice\n",
            "THIRD-PARTY-NOTICES": "Hyphen notice\n",
        }
        for filename, content in third_party_files.items():
            notice_path = license_path.parent / filename
            notice_path.write_text(content, encoding="utf-8")
        distribution = FakeDistribution(
            "demo-pkg", "1.2", site,
            [
                Path("demo_pkg-1.2.dist-info/licenses/LICENSE.txt"),
                *[Path("demo_pkg-1.2.dist-info/licenses") / filename for filename in third_party_files],
            ],
        )
        stdlib_license = self.work / "python" / "LICENSE.txt"
        stdlib_license.parent.mkdir()
        stdlib_license.write_text("Python license\n", encoding="utf-8")

        manifest = collect_linux_notices.collect_notices(
            self.work / "bundle", distributions=[distribution], stdlib_license=stdlib_license
        )

        copied = self.work / "bundle" / manifest["distributions"][0]["files"][0]["copied_to"]
        self.assertEqual(copied.read_text(encoding="utf-8"), "Demo license\n")
        entry = manifest["distributions"][0]["files"][0]
        self.assertEqual(entry["source_relative_path"], "demo_pkg-1.2.dist-info/licenses/LICENSE.txt")
        self.assertEqual(entry["sha256"], hashlib.sha256(copied.read_bytes()).hexdigest())
        copied_notices = {
            Path(file["copied_to"]).name: (self.work / "bundle" / file["copied_to"]).read_text(encoding="utf-8")
            for file in manifest["distributions"][0]["files"]
            if Path(file["copied_to"]).name in third_party_files
        }
        self.assertEqual(copied_notices, third_party_files)
        self.assertEqual(
            (self.work / "bundle" / manifest["stdlib"]["copied_to"]).read_text(encoding="utf-8"),
            "Python license\n",
        )
        saved_manifest = json.loads(
            (self.work / "bundle" / collect_linux_notices.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(saved_manifest["distributions"], manifest["distributions"])
        self.assertIn("legal completeness is not assessed", saved_manifest["scope_note"])

    def test_distribution_without_notice_is_explicitly_reported(self):
        distribution = FakeDistribution("notice-less", "0.3", self.work, [Path("module.py")])

        with patch.object(collect_linux_notices, "_stdlib_license_path", return_value=None):
            manifest = collect_linux_notices.collect_notices(
                self.work / "bundle", distributions=[distribution], stdlib_license=None
            )

        self.assertTrue(manifest["distributions"][0]["missing_notice"])
        self.assertTrue(manifest["stdlib"]["missing_notice"])

    def test_stdlib_lookup_checks_python_installation_prefix(self):
        prefix = self.work / "python-prefix"
        prefix.mkdir()
        expected = prefix / "LICENSE.txt"
        expected.write_text("Python license\n", encoding="utf-8")
        configured_stdlib = self.work / "stdlib"

        with (
            patch.object(collect_linux_notices.sysconfig, "get_path", return_value=str(configured_stdlib)),
            patch.object(collect_linux_notices.sys, "base_prefix", str(prefix)),
            patch.object(collect_linux_notices.sys, "prefix", str(prefix)),
        ):
            self.assertEqual(collect_linux_notices._stdlib_license_path(), expected)

    def test_unsafe_distribution_relative_paths_are_rejected(self):
        unsafe_paths = (
            "../outside/LICENSE",
            "C:/outside/LICENSE",
            "/outside/LICENSE",
            "a/../../outside/NOTICE",
        )
        for unsafe in unsafe_paths:
            with self.subTest(path=unsafe):
                distribution = FakeDistribution("bad-path", "1", self.work, [Path(unsafe)])
                with self.assertRaisesRegex(ValueError, "unsafe distribution path"):
                    collect_linux_notices.collect_notices(
                        self.work / "bundle", distributions=[distribution], stdlib_license=None
                    )

    def test_output_directory_must_not_escape_requested_root(self):
        distribution = FakeDistribution("bad-output", "1", self.work, [Path("LICENSE")])
        (self.work / "LICENSE").write_text("license", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "output path escapes"):
            collect_linux_notices.collect_notices(
                self.work / "bundle", distributions=[distribution], stdlib_license=None,
                output_root=self.work / "other-root",
            )

    def test_explicit_distribution_selection_limits_inventory(self):
        one = FakeDistribution("one", "1", self.work, [Path("one-1.dist-info/LICENSE")])
        two = FakeDistribution("two", "2", self.work, [Path("two-2.dist-info/LICENSE")])
        for name in ("one-1.dist-info", "two-2.dist-info"):
            (self.work / name).mkdir()
            (self.work / name / "LICENSE").write_text(name, encoding="utf-8")

        manifest = collect_linux_notices.collect_notices(
            self.work / "bundle", distributions=[one, two], stdlib_license=None,
            selected_distributions=["two"],
        )

        self.assertEqual([entry["name"] for entry in manifest["distributions"]], ["two"])


if __name__ == "__main__":
    unittest.main()
