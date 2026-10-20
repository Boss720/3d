"""Pure and Windows integration tests for read-only process-tree observation."""
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from process_tree import (
    ProcessRootNotFound,
    executable_basename,
    find_descendants,
    match_process_identity,
    snapshot_windows_processes,
)


class ProcessTreePureTests(unittest.TestCase):
    def setUp(self):
        self.entries = [
            {"pid": 102, "parent_pid": 101, "executable": "nested.exe"},
            {"pid": 21, "parent_pid": 20, "executable": "unrelated-child.exe"},
            {"pid": 104, "parent_pid": 10, "executable": "second.exe"},
            {"pid": 10, "parent_pid": 1, "executable": "api.exe"},
            {"pid": 20, "parent_pid": 1, "executable": "other-root.exe"},
            {"pid": 99, "parent_pid": 10, "executable": "first.exe"},
            {"pid": 101, "parent_pid": 10, "executable": "worker.exe"},
            {"pid": 13, "parent_pid": 12, "executable": "deeper.exe"},
            {"pid": 12, "parent_pid": 10, "executable": "third.exe"},
        ]

    def test_finds_direct_and_nested_descendants_excluding_root_and_unrelated(self):
        found = find_descendants(self.entries, 10)

        self.assertEqual(
            [(item["pid"], item["parent_pid"], item["depth"]) for item in found],
            [(12, 10, 1), (99, 10, 1), (101, 10, 1), (104, 10, 1),
             (13, 12, 2), (102, 101, 2)],
        )
        self.assertNotIn(10, [item["pid"] for item in found])
        self.assertNotIn(20, [item["pid"] for item in found])
        self.assertNotIn(21, [item["pid"] for item in found])

    def test_result_order_is_stable_when_snapshot_order_changes(self):
        first = find_descendants(self.entries, 10)
        second = find_descendants(list(reversed(self.entries)), 10)

        self.assertEqual(first, second)

    def test_absent_root_does_not_validate_or_return_orphan_candidates(self):
        with self.assertRaises(ProcessRootNotFound):
            find_descendants(self.entries, 777)
        with self.assertRaises(ProcessRootNotFound):
            find_descendants([{"pid": 501, "parent_pid": 500, "executable": "child.exe"}], 500)
        self.assertEqual(find_descendants([{"pid": 500, "parent_pid": 1, "executable": "api.exe"}], 500), [])

    def test_malformed_root_ids_are_rejected_and_malformed_rows_ignored(self):
        for root_pid in (None, True, 0, -1, "10"):
            with self.subTest(root_pid=root_pid), self.assertRaises(ValueError):
                find_descendants(self.entries, root_pid)

        malformed = [
            {"pid": "10", "parent_pid": 1, "executable": "bad-root.exe"},
            {"pid": 11, "parent_pid": 10, "executable": "orphan.exe"},
            {"pid": 10, "parent_pid": "1", "executable": "bad-parent.exe"},
            {"pid": 12, "parent_pid": 10, "executable": None},
        ]
        with self.assertRaises(ProcessRootNotFound):
            find_descendants(malformed, 10)

    def test_process_identity_distinguishes_present_absent_reused_and_unknown(self):
        observed = {"pid": 80, "parent_pid": 10, "executable": "ffmpeg.exe",
                    "creation_time_100ns": 1234, "identity_status": "verified"}
        self.assertEqual(match_process_identity([observed], observed)["status"], "present")
        self.assertEqual(match_process_identity([], observed)["status"], "absent")
        reused = {**observed, "parent_pid": 1, "creation_time_100ns": 5678}
        self.assertEqual(match_process_identity([reused], observed)["status"], "reused")
        unreadable = {**observed, "creation_time_100ns": None, "identity_status": "unavailable"}
        self.assertEqual(match_process_identity([unreadable], observed)["status"], "unverified")
        self.assertEqual(match_process_identity([observed], unreadable)["status"], "unverified")
        self.assertEqual(match_process_identity([], unreadable)["status"], "unverified")

    def test_parent_pid_reuse_does_not_attach_an_older_process(self):
        entries = [
            {"pid": 10, "parent_pid": 1, "executable": "new-api.exe",
             "creation_time_100ns": 200, "identity_status": "verified"},
            {"pid": 11, "parent_pid": 10, "executable": "old-child.exe",
             "creation_time_100ns": 100, "identity_status": "verified"},
            {"pid": 12, "parent_pid": 10, "executable": "current-child.exe",
             "creation_time_100ns": 300, "identity_status": "verified"},
        ]
        found = find_descendants(entries, 10)

        self.assertEqual([item["pid"] for item in found], [12])
        self.assertEqual(found[0]["tree_identity_status"], "verified")

    def test_unreadable_parent_identity_marks_tree_link_uncertain(self):
        found = find_descendants(self.entries, 10)

        self.assertTrue(found)
        self.assertTrue(all(item["tree_identity_status"] == "unverified" for item in found))

    def test_orphaned_child_is_found_by_identity_after_parent_exits(self):
        root = {"pid": 10, "parent_pid": 1, "executable": "api.exe",
                "creation_time_100ns": 100, "identity_status": "verified"}
        observed_child = {"pid": 11, "parent_pid": 10, "executable": "ffmpeg.exe",
                          "creation_time_100ns": 200, "identity_status": "verified", "depth": 1}
        before_parent_exit = [root, observed_child]
        self.assertEqual(find_descendants(before_parent_exit, 10)[0]["pid"], 11)

        after_parent_exit = [{"pid": 1, "parent_pid": 0, "executable": "system.exe",
                              "creation_time_100ns": 1, "identity_status": "verified"},
                             {**observed_child, "parent_pid": 1}]
        with self.assertRaises(ProcessRootNotFound):
            find_descendants(after_parent_exit, 10)
        self.assertEqual(match_process_identity(after_parent_exit, observed_child)["status"], "present")

    def test_executable_basename_handles_windows_paths_and_case(self):
        self.assertEqual(executable_basename(r"C:\Program Files\FFmpeg\FFMPEG.EXE"), "ffmpeg.exe")
        self.assertEqual(executable_basename("folder/ffmpeg.exe"), "ffmpeg.exe")

    def test_creation_time_must_strictly_precede_snapshot_start(self):
        from process_tree import _creation_time_precedes_snapshot, _normalize_entry

        self.assertTrue(_creation_time_precedes_snapshot(99, 100))
        self.assertFalse(_creation_time_precedes_snapshot(100, 100))
        self.assertFalse(_creation_time_precedes_snapshot(101, 100))
        self.assertFalse(_creation_time_precedes_snapshot(None, 100))

        # Snapshot code leaves a post-boundary identity unavailable; downstream
        # normalization must not promote the row back to verified.
        created = 101 if _creation_time_precedes_snapshot(101, 100) else None
        normalized = _normalize_entry({
            "pid": 11, "parent_pid": 10, "executable": "child.exe",
            "creation_time_100ns": created, "identity_status": "unavailable",
            "identity_error": "PID reuse cannot be ruled out",
        })
        self.assertEqual(normalized["identity_status"], "unavailable")
        self.assertIsNone(normalized["creation_time_100ns"])


@unittest.skipUnless(os.name == "nt", "Toolhelp process snapshots are Windows-only")
class WindowsProcessTreeIntegrationTests(unittest.TestCase):
    def test_snapshot_marks_post_start_creation_times_unavailable(self):
        future_creation_time = 2**62
        with patch("process_tree._windows_creation_time", return_value=(future_creation_time, None)):
            entries = snapshot_windows_processes()

        self.assertTrue(entries, "Toolhelp process snapshot unexpectedly contained no rows")
        self.assertTrue(all(entry["identity_status"] == "unavailable" for entry in entries))
        self.assertTrue(all(
            "snapshot began" in entry.get("identity_error", "") for entry in entries
        ))

    def test_observes_owned_child_and_nested_child_then_confirms_reap(self):
        innermost_code = "import sys; sys.stdin.readline()"
        middle_code = (
            "import subprocess,sys; "
            f"child=subprocess.Popen([sys.executable,'-c',{innermost_code!r}], "
            "stdin=subprocess.PIPE,text=True); "
            "print(child.pid,flush=True); sys.stdin.readline(); "
            "child.stdin.write('stop\\n'); child.stdin.flush(); child.wait(timeout=5)"
        )
        root_code = (
            "import subprocess,sys; "
            f"child=subprocess.Popen([sys.executable,'-c',{middle_code!r}], "
            "stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True); "
            "nested=child.stdout.readline().strip(); "
            "print(str(child.pid)+' '+nested,flush=True); sys.stdin.readline(); "
            "child.stdin.write('stop\\n'); child.stdin.flush(); child.wait(timeout=5)"
        )
        root = subprocess.Popen(
            [sys.executable, "-c", root_code], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        owned_pids = set()
        try:
            line = root.stdout.readline().strip()
            self.assertTrue(line, "owned process did not report its nested process IDs")
            middle_pid, nested_pid = (int(value) for value in line.split())
            owned_pids.update((root.pid, middle_pid, nested_pid))

            process_entries = snapshot_windows_processes()
            descendants = find_descendants(process_entries, root.pid)
            by_pid = {item["pid"]: item for item in descendants}
            self.assertIn(middle_pid, by_pid, msg=f"owned child absent from tree: {by_pid}")
            self.assertIn(nested_pid, by_pid, msg=f"nested child absent from tree: {by_pid}")
            middle_subtree = find_descendants(process_entries, middle_pid)
            self.assertIn(nested_pid, {item["pid"] for item in middle_subtree},
                          msg="nested child was not below the owned child")

            owned_entries = {}
            for pid in owned_pids:
                entry = next(item for item in process_entries if item["pid"] == pid)
                self.assertEqual(entry["identity_status"], "verified",
                                 msg=f"owned PID {pid} lacks creation identity: {entry}")
                self.assertGreater(entry["creation_time_100ns"], 0)
                owned_entries[pid] = entry

            root.stdin.write("stop\n")
            root.stdin.flush()
            self.assertEqual(root.wait(timeout=10), 0)
            after = snapshot_windows_processes()
            for pid, entry in owned_entries.items():
                self.assertEqual(match_process_identity(after, entry)["status"], "absent",
                                 msg=f"owned process {pid} remained or could not be verified after reaping")
        finally:
            if root.poll() is None:
                try:
                    root.stdin.write("stop\n")
                    root.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    root.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    root.kill()
                    root.wait(timeout=5)
            for stream in (root.stdin, root.stdout, root.stderr):
                if stream is not None:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
