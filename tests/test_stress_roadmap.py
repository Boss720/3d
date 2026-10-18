import json
import sys
import unittest
from pathlib import Path

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
        self.assertFalse(stress_roadmap.cancellation_ready({"state": "processing", "frames_processed": 4}))
        self.assertTrue(stress_roadmap.cancellation_ready({"state": "processing", "frames_processed": 5}))
        self.assertFalse(stress_roadmap.cancellation_ready({"state": "completed", "frames_processed": 360}))

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

    def test_stop_reason_handles_successful_export_with_null_error(self):
        record = {"export": {"success": True, "error": None},
                  "cancellation": {"status": "passed"}}
        self.assertIsNone(stress_roadmap.stop_reason_after_model(record, {"idle": True}))
        self.assertEqual(stress_roadmap.stop_reason_after_model(
            {"export": {"error": {"code": "job_timeout"}}}, {"idle": True}), "job_timeout_reached")


if __name__ == "__main__":
    unittest.main()
