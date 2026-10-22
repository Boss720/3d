import hashlib
import os
from pathlib import Path
import subprocess
import unittest
import uuid
from unittest.mock import patch

from scripts import collect_linux_native_notices as collector


class NativeNoticeTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Path(__file__).resolve().parents[1] / ".runtime"
        self.work = self.runtime / "native-notice-tests" / uuid.uuid4().hex
        self.package = self.work / "package"
        self.internal = self.package / "_internal"
        self.internal.mkdir(parents=True)
        self.docs = self.work / "doc"
        self.common = self.work / "common"
        self.output = self.work / "notices"
        self.origins = {"schema_version": 1, "files": []}

    def tearDown(self):
        target = self.runtime / "_quarantine" / "native-notice-tests" / self.work.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.work, target)

    def add_binary(self, name="libdemo.so", *, source_content=b"\x7fELForiginal"):
        binary = self.internal / name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(b"\x7fELFpackager-modified")
        source = self.work / "source" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(source_content)
        self.origins["files"].append({"destination": name, "source": str(source.resolve()), "kind": "BINARY"})
        return source

    def collect(self):
        return collector.collect(self.origins, self.package, self.output,
                                 doc_root=self.docs, common_root=self.common)

    def test_owned_modified_binary_copies_exact_notice_and_provenance(self):
        source = self.add_binary()
        copyright_path = self.docs / "demo" / "copyright"
        copyright_path.parent.mkdir(parents=True)
        copyright_path.write_bytes(b"Actual demo copyright\nLicense: BSD\n")
        self.common.mkdir()
        (self.common / "BSD").write_bytes(b"Shared BSD supporting terms\n")

        def query(arguments):
            if arguments[0] == "-S":
                self.assertEqual(arguments, ["-S", str(source.resolve())])
                return subprocess.CompletedProcess(arguments, 0, "demo:amd64: " + str(source.resolve()) + "\n", "")
            self.assertEqual(arguments[-1], "demo:amd64")
            return subprocess.CompletedProcess(arguments, 0, "demo:amd64\t1.2-3\tdemo-source\t1.2\n", "")

        with patch.object(collector, "_query", side_effect=query):
            result = self.collect()
        row = result["files"][0]
        self.assertFalse(row["source_matches_collected"])
        self.assertEqual(row["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(row["ownership_status"], "dpkg_owned")
        record = result["packages"][0]
        self.assertEqual(record["source_package"], "demo-source")
        self.assertEqual(record["version"], "1.2-3")
        notice = record["copyright"]
        self.assertEqual((self.output / notice["copied_to"]).read_bytes(), copyright_path.read_bytes())
        self.assertEqual(notice["sha256"], hashlib.sha256(copyright_path.read_bytes()).hexdigest())
        self.assertEqual(notice["resolved_source_path"], str(copyright_path.resolve()))
        self.assertEqual(len(result["common_licenses"]), 1)
        self.assertIn("shared supporting", result["common_licenses_scope"])

    def test_unmapped_wheel_stdlib_and_missing_sources_remain_explicit(self):
        self.add_binary("site-packages/pkg/lib.so")
        self.add_binary("python3.12/lib-dynload/_demo.so")
        missing = self.add_binary("missing.so")
        missing.unlink()
        (self.package / "StereoLab").write_bytes(b"\x7fELFbootloader")
        with patch.object(collector, "_query", return_value=subprocess.CompletedProcess([], 1, "", "no owner")):
            result = self.collect()
        rows = {r["path"]: r for r in result["files"]}
        self.assertEqual(rows["StereoLab"]["ownership_status"], "unmapped")
        self.assertEqual(rows["_internal/site-packages/pkg/lib.so"]["source_classification"], "wheel_or_site_packages")
        self.assertEqual(rows["_internal/python3.12/lib-dynload/_demo.so"]["source_classification"], "stdlib")
        self.assertTrue(rows["_internal/missing.so"]["source_missing"])
        self.assertIsNone(rows["_internal/missing.so"]["source_matches_collected"])
        self.assertEqual(result["packages"], [])

    def test_rejects_traversing_origins_before_any_output(self):
        for destination in ("../escape", "/absolute", "a/../../bad", "a\\..\\bad", "C:/escape"):
            with self.subTest(destination=destination):
                self.origins["files"] = [{"destination": destination, "source": str(self.work.resolve()), "kind": "BINARY"}]
                with self.assertRaises(ValueError):
                    self.collect()
                self.assertFalse(self.output.exists())

    def test_dpkg_timeout_is_recorded_without_inventing_owner(self):
        self.add_binary()
        with patch.object(collector, "_query", side_effect=subprocess.TimeoutExpired("dpkg-query", 15)):
            result = self.collect()
        self.assertEqual(result["files"][0]["ownership_status"], "no_dpkg_owner")
        self.assertIn("error", result["files"][0]["ownership_queries"][0])

    def test_query_uses_argument_list_and_bounded_timeout(self):
        with patch.object(collector.subprocess, "run") as run:
            collector._query(["-S", "/lib/a path.so"])
        self.assertEqual(run.call_args.args[0], ["dpkg-query", "-S", "/lib/a path.so"])
        self.assertEqual(run.call_args.kwargs["timeout"], 15)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_source_alias_fallback_and_symlinked_doc_directory(self):
        source = self.add_binary()
        alias = source.parent / "alias.so"
        real_doc = self.docs / "demo-common"
        real_doc.mkdir(parents=True)
        (real_doc / "copyright").write_bytes(b"Shared installed package copyright\n")
        try:
            alias.symlink_to(source)
            (self.docs / "demo").symlink_to(real_doc, target_is_directory=True)
            (self.internal / "collected-alias.so").symlink_to(self.internal / "libdemo.so")
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")
        self.origins["files"][0]["source"] = str(alias.absolute())

        def query(arguments):
            if arguments[0] == "-S":
                if arguments[-1] == str(alias.absolute()):
                    return subprocess.CompletedProcess(arguments, 1, "", "no alias owner")
                return subprocess.CompletedProcess(arguments, 0, "demo: " + str(source.resolve()), "")
            return subprocess.CompletedProcess(arguments, 0, "demo\t1\tdemo\t1\n", "")

        with patch.object(collector, "_query", side_effect=query):
            result = self.collect()
        rows = {row["path"]: row for row in result["files"]}
        self.assertEqual(len(rows["_internal/libdemo.so"]["ownership_queries"]), 2)
        collected_alias = rows["_internal/collected-alias.so"]
        self.assertEqual(collected_alias["ownership_status"], "dpkg_owned")
        self.assertEqual(collected_alias["origin_mapping"], {
            "method": "collected_symlink_target", "destination": "libdemo.so",
            "resolved_collected_path": "_internal/libdemo.so",
        })
        notice = result["packages"][0]["copyright"]
        self.assertEqual(notice["resolved_source_path"], str((real_doc / "copyright").resolve()))
        self.assertEqual((self.output / notice["copied_to"]).read_bytes(), b"Shared installed package copyright\n")


if __name__ == "__main__":
    unittest.main()
