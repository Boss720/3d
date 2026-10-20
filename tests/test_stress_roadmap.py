import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import stress_roadmap


class StressManifestTests(unittest.TestCase):
    def test_manifest_is_explicitly_synthetic_and_covers_long_video_stress_tags(self):
        manifest = json.loads((ROOT / "benchmarks" / "stress-manifest.json").read_text(encoding="utf-8"))
        self.assertIn("synthetic", manifest["classification"].lower())
        fixture = manifest["fixtures"][0]
        source = fixture["source"]
        self.assertEqual(source["width"], 320)
        self.assertEqual(source["height"], 180)
        self.assertEqual(source["fps"], 12)
        self.assertEqual(source["frames"], 360)
        self.assertAlmostEqual(source["frames"] / source["fps"], 30.0)
        self.assertTrue(source["audio"])
        self.assertTrue({"low_light", "animation", "thin_detail", "camera_pan", "hard_cuts", "audio"}
                        <= set(fixture["tags"]))


class StressGuardTests(unittest.TestCase):
    def setUp(self):
        self.api_process = {
            "pid": 10, "parent_pid": 1, "executable": "python.exe",
            "creation_time_100ns": 100, "identity_status": "verified",
        }

    def test_fixture_frames_are_repeatable_and_encode_cuts_and_low_light(self):
        first = stress_roadmap.make_stress_frame(17)
        repeat = stress_roadmap.make_stress_frame(17)
        before_cut = stress_roadmap.make_stress_frame(119)
        after_cut = stress_roadmap.make_stress_frame(120)
        normal = stress_roadmap.make_stress_frame(239)
        low_light = stress_roadmap.make_stress_frame(240)
        self.assertTrue((first == repeat).all())
        self.assertGreater(abs(before_cut.astype("int16") - after_cut.astype("int16")).mean(), 50.0)
        self.assertLess(float(low_light.mean()), float(normal.mean()))

    def test_active_jobs_include_queued_and_processing_but_ignore_terminal_history(self):
        jobs = [
            {"id": "done", "state": "completed"},
            {"id": "queued", "state": "queued"},
            {"id": "working", "state": "processing"},
            {"id": "canceled", "state": "canceled"},
        ]
        self.assertEqual([job["id"] for job in stress_roadmap.active_jobs(jobs)], ["queued", "working"])

    def test_cancellation_gate_requires_five_processed_frames_while_job_is_live(self):
        ready = {"state": "processing", "phase": "conversion", "frames_processed": 5, "total_frames": 360}
        self.assertFalse(stress_roadmap.cancellation_ready({**ready, "frames_processed": 4}))
        self.assertTrue(stress_roadmap.cancellation_ready(ready))
        self.assertFalse(stress_roadmap.cancellation_ready({**ready, "phase": "finalization"}))
        self.assertFalse(stress_roadmap.cancellation_ready({**ready, "frames_processed": 360}))
        self.assertFalse(stress_roadmap.cancellation_ready({**ready, "frames_processed": True}))

    def test_successful_export_requires_verified_directml_gpu_execution(self):
        stress_roadmap.require_explicit_gpu_execution({
            "requested_mode": "gpu",
            "active_provider": "DmlExecutionProvider",
            "gpu_verified": True,
            "fallback_reason": None,
        })
        with self.assertRaises(ValueError):
            stress_roadmap.require_explicit_gpu_execution({
                "requested_mode": "gpu",
                "active_provider": "CPUExecutionProvider",
                "gpu_verified": False,
                "fallback_reason": "GPU failed",
            })

    def test_stress_runner_requires_explicit_base_url_and_owned_server_pid(self):
        parser = stress_roadmap.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])
        args = parser.parse_args([
            "--base-url", "http://127.0.0.1:8769", "--expected-server-pid", "10",
        ])
        self.assertEqual(args.expected_server_pid, 10)
        with self.assertRaises(SystemExit):
            parser.parse_args([
                "--base-url", "http://127.0.0.1:8769", "--expected-server-pid", "0",
            ])

    def test_owned_server_preflight_matches_diagnostics_pid_and_process_identity(self):
        self.assertEqual(stress_roadmap.require_expected_api_process(
            {"pid": 10}, 10, [self.api_process]), self.api_process)
        with self.assertRaisesRegex(ValueError, "does not match"):
            stress_roadmap.require_expected_api_process({"pid": 11}, 10, [self.api_process])
        with self.assertRaisesRegex(ValueError, "could not be verified"):
            stress_roadmap.require_expected_api_process(
                {"pid": 10}, 10, [{**self.api_process, "identity_status": "unavailable"}])
        with self.assertRaisesRegex(ValueError, "unrecognized server executable"):
            stress_roadmap.require_expected_api_process(
                {"pid": 10}, 10, [{**self.api_process, "executable": "chrome.exe"}])

    def test_ffmpeg_observer_reports_post_terminal_visibility_by_identity(self):
        ffmpeg = {
            "pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        snapshots = [
            [self.api_process, ffmpeg],
            [self.api_process],
        ]
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=lambda: snapshots.pop(0))
        observer.observe_job_snapshot({
            "state": "processing", "phase": "conversion",
            "frames_processed": 5, "total_frames": 360,
        })
        observer.observe_job_snapshot({
            "state": "completed", "phase": "complete",
            "frames_processed": 360, "total_frames": 360,
        })

        result = observer.after_terminal(timeout_seconds=0)

        self.assertEqual(result["observed_ffmpeg_processes"][0]["pid"], 20)
        self.assertEqual(result["last_job_snapshot"]["phase"], "complete")
        self.assertEqual(result["status"], "not_visible_at_first_post_terminal_sample")
        self.assertIn("does not establish exact exit timing", result["scope"])

    def test_ffmpeg_observer_reports_visibility_then_absence_during_window(self):
        ffmpeg = {
            "pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        snapshots = [[self.api_process, ffmpeg] for _ in range(2)] + [[self.api_process]]
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=lambda: snapshots.pop(0))
        observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        observer.observe_job_snapshot({"state": "completed", "phase": "complete"})

        result = observer.after_terminal(timeout_seconds=1, interval_seconds=0.001)

        self.assertEqual(result["status"], "observed_then_absent_during_window")
        self.assertTrue(result["post_terminal_check"]["ever_visible_after_terminal"])
        self.assertEqual(
            [item["identity_statuses"][0]["status"]
             for item in result["post_terminal_check"]["observations"]],
            ["present", "absent"],
        )

    def test_ffmpeg_observer_treats_reused_pid_as_unverified(self):
        ffmpeg = {
            "pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        reused = {**ffmpeg, "creation_time_100ns": 201}
        snapshots = [[self.api_process, ffmpeg], [self.api_process, reused]]
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=lambda: snapshots.pop(0))
        observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
        result = observer.after_terminal(timeout_seconds=0)

        self.assertEqual(result["status"], "unverified_post_terminal_identity")
        self.assertEqual(result["post_terminal_check"]["identity_statuses"][0]["status"], "reused")

    def test_ffmpeg_observer_does_not_label_timeout_as_post_terminal(self):
        ffmpeg = {
            "pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        calls = []

        def snapshot():
            calls.append(True)
            return [self.api_process, ffmpeg]

        observer = stress_roadmap.FfmpegProcessObserver(self.api_process, 0.5, snapshot_fn=snapshot)
        observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        result = observer.after_terminal(timeout_seconds=0)

        self.assertEqual(result["status"], "unverified_job_not_terminal")
        self.assertFalse(result["job_reached_terminal"])
        self.assertEqual(len(calls), 1, "timeout result must not start a post-terminal observation window")

    def test_ffmpeg_observer_checks_orphaned_child_globally_and_fails_closed(self):
        ffmpeg = {
            "pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=lambda: [self.api_process, ffmpeg])
        observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
        result = observer.after_terminal(timeout_seconds=0)
        self.assertEqual(result["status"], "still_visible_after_0_seconds")
        self.assertEqual(result["post_terminal_check"]["identity_statuses"][0]["status"], "present")

        good_snapshots = [[self.api_process, ffmpeg]]

        def broken_snapshot():
            if good_snapshots:
                return good_snapshots.pop(0)
            raise OSError("snapshot denied")

        broken = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=broken_snapshot)
        broken.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        broken.observe_job_snapshot({"state": "completed", "phase": "complete"})
        self.assertEqual(broken.after_terminal(timeout_seconds=0)["status"],
                         "unverified_snapshot_or_root_error")

    def test_ffmpeg_observer_does_not_claim_tree_association_without_verified_lineage(self):
        intermediary = {
            "pid": 15, "parent_pid": 10, "executable": "worker.exe",
            "creation_time_100ns": None, "identity_status": "unavailable",
        }
        uncertain_ffmpeg = {
            "pid": 20, "parent_pid": 15, "executable": "ffmpeg.exe",
            "creation_time_100ns": 200, "identity_status": "verified",
        }
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5,
            snapshot_fn=lambda: [self.api_process, intermediary, uncertain_ffmpeg])
        observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
        observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
        result = observer.after_terminal(timeout_seconds=0)
        self.assertEqual(result["status"], "unverified_ffmpeg_tree_relationship")

    def test_stop_reason_handles_successful_export_with_null_error(self):
        record = {"export": {"success": True, "error": None},
                  "cancellation": {"status": "passed"}}
        self.assertIsNone(stress_roadmap.stop_reason_after_model(record, {"idle": True}))
        self.assertEqual(stress_roadmap.stop_reason_after_model(
            {"export": {"error": {"code": "job_timeout"}}}, {"idle": True}), "job_timeout_reached")

    def test_cancellation_finalizes_status_after_conversion_trigger(self):
        ready = {
            "id": "owned", "state": "processing", "phase": "conversion",
            "frames_processed": 5, "total_frames": 360,
            "execution": {"requested_mode": "gpu", "active_provider": "DmlExecutionProvider",
                          "gpu_verified": True, "fallback_reason": None},
        }
        client = Mock()
        client.post.return_value = httpx.Response(200, json={"id": "owned"})
        for final_state, download_status, expected_status in (
                ("canceled", 404, "passed"), ("completed", 200, "failed_after_cancel_request")):
            with self.subTest(final_state=final_state):
                source = SimpleNamespace(path=MagicMock(spec=Path), source_name="fixture.mp4")
                client.get.return_value = httpx.Response(download_status)
                with patch.object(stress_roadmap, "_activity", return_value={"idle": True}), \
                        patch.object(stress_roadmap, "_server_resource", return_value={}), \
                        patch.object(stress_roadmap, "_find_job", side_effect=[
                            ready, {"id": "owned", "state": final_state}]), \
                        patch.object(stress_roadmap.benchmarking, "_request_json", return_value=[ready]):
                    result = stress_roadmap._cancel_test(client, source, "model", {}, 10, 0.01)
                self.assertEqual(result["success"], final_state == "canceled")
                self.assertEqual(result["status"], expected_status)

    def test_model_result_checks_cleanup_before_download_and_output_inspection(self):
        events = []
        observer = Mock()
        observer.observe_job_snapshot.side_effect = lambda job: events.append(job["state"])
        observer.after_terminal.side_effect = lambda: events.append("process_window") or {"status": "observed"}

        def execute(*args, **kwargs):
            kwargs["on_job_snapshot"]({"state": "processing", "phase": "conversion"})
            kwargs["on_job_snapshot"]({"state": "completed", "phase": "complete"})
            events.append("download_and_inspection")
            return {"success": False}

        source = SimpleNamespace(fixture_id="fixture", fixture_identity_sha256="identity",
                                 source_sha256="source", source_info={})
        with patch.object(Path, "mkdir"), patch.object(Path, "is_file", return_value=False), \
                patch.object(stress_roadmap, "_activity", return_value={"idle": True}), \
                patch.object(stress_roadmap, "_settings", return_value={}), \
                patch.object(stress_roadmap, "_cancel_test", return_value={"success": False}), \
                patch.object(stress_roadmap, "FfmpegProcessObserver", return_value=observer), \
                patch.object(stress_roadmap.benchmarking, "_model_record", return_value={"state": "ready", "hash": "hash"}), \
                patch.object(stress_roadmap.benchmarking, "_environment", return_value={"runtime_fingerprint": "runtime"}), \
                patch.object(stress_roadmap.benchmarking, "_execute_run", side_effect=execute):
            result = stress_roadmap._model_result(
                object(), "model", {}, {}, source, {}, Path("unused"), 10, 0.5, False, self.api_process)
        self.assertEqual(events, ["processing", "completed", "process_window", "download_and_inspection"])
        self.assertEqual(result["export"]["process_cleanup_observation"], {"status": "observed"})
        observer.after_terminal.assert_called_once()

    def test_delayed_terminal_window_is_unverified_and_uses_terminal_observation_time(self):
        ffmpeg = {"pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
                  "creation_time_100ns": 200, "identity_status": "verified"}
        clock = [0.0]
        snapshot = Mock(side_effect=lambda: [self.api_process, ffmpeg] if clock[0] < 2 else [self.api_process])
        observer = stress_roadmap.FfmpegProcessObserver(self.api_process, 0.5, snapshot_fn=snapshot)
        with patch.object(stress_roadmap.time, "monotonic", side_effect=lambda: clock[0]):
            observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
            clock[0] = 1.0
            observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
            calls_at_terminal = snapshot.call_count
            clock[0] = 8.0
            result = observer.after_terminal()
        self.assertEqual(result["status"], "unverified_post_terminal_window_missed")
        self.assertEqual(result["post_terminal_check"]["elapsed_seconds"], 7.0)
        self.assertIsNotNone(result["terminal_observed_at_utc"])
        self.assertEqual(snapshot.call_count, calls_at_terminal)

    def test_terminal_only_snapshot_cannot_claim_ffmpeg_observed_during_export(self):
        ffmpeg = {"pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
                  "creation_time_100ns": 200, "identity_status": "verified"}
        observer = stress_roadmap.FfmpegProcessObserver(
            self.api_process, 0.5, snapshot_fn=lambda: [self.api_process, ffmpeg])
        observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
        result = observer.after_terminal(timeout_seconds=0)
        self.assertEqual(result["status"], "unverified_ffmpeg_not_observed_during_export")
        self.assertEqual(result["observed_ffmpeg_processes"], [])

    def test_absence_sample_finishing_after_terminal_deadline_is_unverified(self):
        ffmpeg = {"pid": 20, "parent_pid": 10, "executable": "ffmpeg.exe",
                  "creation_time_100ns": 200, "identity_status": "verified"}
        clock = [0.0]
        calls = []

        def snapshot():
            calls.append(True)
            if len(calls) == 1:
                return [self.api_process, ffmpeg]
            clock[0] = 1.0 if len(calls) == 2 else 6.0
            return [self.api_process, ffmpeg] if len(calls) == 2 else [self.api_process]

        observer = stress_roadmap.FfmpegProcessObserver(self.api_process, 0.5, snapshot_fn=snapshot)
        with patch.object(stress_roadmap.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(stress_roadmap.time, "sleep"):
            observer.observe_job_snapshot({"state": "processing", "phase": "conversion"})
            observer.observe_job_snapshot({"state": "completed", "phase": "complete"})
            result = observer.after_terminal()
        self.assertEqual(result["post_terminal_check"]["observations"][0]["elapsed_seconds"], 1.0)
        self.assertEqual(result["post_terminal_check"]["observations"][-1]["elapsed_seconds"], 6.0)
        self.assertTrue(result["post_terminal_check"]["observations"][0]["within_deadline"])
        self.assertFalse(result["post_terminal_check"]["observations"][-1]["within_deadline"])
        self.assertEqual(result["status"], "unverified_post_terminal_window_missed")


if __name__ == "__main__":
    unittest.main()
