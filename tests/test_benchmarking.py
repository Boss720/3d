import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from benchmarking import (
    benchmark_exit_code,
    expected_output_dimensions,
    fixture_identity_sha256,
    job_timing_snapshot,
    normalize_execution_snapshot,
    summarize_runs,
    validate_manifest,
    verify_image_output,
    verify_video_observation,
    _model_record,
)


ROOT = Path(__file__).resolve().parents[1]


class ManifestTests(unittest.TestCase):
    def test_built_in_manifest_is_valid_and_names_motion_cut_and_portrait_fixtures(self):
        manifest = json.loads((ROOT / "benchmarks" / "manifest.json").read_text(encoding="utf-8"))

        fixtures = validate_manifest(manifest)

        ids = {fixture["id"] for fixture in fixtures}
        self.assertIn("synthetic-motion-landscape", ids)
        self.assertIn("synthetic-hard-cut", ids)
        self.assertIn("synthetic-portrait", ids)

    def test_accepts_user_supplied_relative_source_path_without_resolving_it(self):
        manifest = {
            "schema_version": 1,
            "fixtures": [{
                "id": "user-clip",
                "source": {"kind": "path", "path": "media/clip.mp4", "media_type": "video"},
            }],
        }

        fixtures = validate_manifest(manifest)

        self.assertEqual(fixtures[0]["source"]["path"], "media/clip.mp4")

    def test_rejects_duplicate_ids_and_invalid_video_dimensions(self):
        duplicate = {"schema_version": 1, "fixtures": [
            {"id": "same", "source": {"kind": "synthetic_image", "width": 16, "height": 16}},
            {"id": "same", "source": {"kind": "synthetic_image", "width": 16, "height": 16}},
        ]}
        invalid_video = {"schema_version": 1, "fixtures": [{
            "id": "bad-video",
            "source": {"kind": "synthetic_video", "width": 0, "height": 16,
                       "fps": 8, "frames": 8, "pattern": "motion"},
        }]}

        with self.assertRaisesRegex(ValueError, "unique"):
            validate_manifest(duplicate)
        with self.assertRaisesRegex(ValueError, "width"):
            validate_manifest(invalid_video)

    def test_generated_recipe_identity_is_stable_across_encoded_byte_hashes(self):
        recipe = {"width": 160, "height": 96, "fps": 8, "frames": 8,
                  "pattern": "motion", "seed": 23, "audio": False}

        first = fixture_identity_sha256("motion", "synthetic_video", recipe, "encoded-a")
        second = fixture_identity_sha256("motion", "synthetic_video", recipe, "encoded-b")
        supplied_first = fixture_identity_sha256("clip", "path", {"path": "clip.mp4"}, "content-a")
        supplied_second = fixture_identity_sha256("clip", "path", {"path": "other.mp4"}, "content-b")

        self.assertEqual(first, second)
        self.assertNotEqual(supplied_first, supplied_second)

    def test_pinned_model_hash_is_recorded_from_local_catalog(self):
        model = _model_record({"models": [{"id": "depth-anything-v2-small", "state": "ready",
                                            "sha256": "pinned-hash", "revision": "pinned-revision"}]},
                              "depth-anything-v2-small")

        self.assertEqual(model["hash"], "pinned-hash")
        self.assertEqual(model["revision"], "pinned-revision")


class OutputVerificationTests(unittest.TestCase):
    def test_layout_dimensions_match_stereo_packing_rules(self):
        self.assertEqual(expected_output_dimensions(320, 180, "SBS"), (640, 180))
        self.assertEqual(expected_output_dimensions(321, 181, "HALFSBS"), (320, 181))
        self.assertEqual(expected_output_dimensions(320, 180, "TB"), (320, 360))

    def test_decodes_image_and_checks_dimensions(self):
        image = np.zeros((24, 64, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        result = verify_image_output(encoded.tobytes(), {"width": 64, "height": 24})

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["actual_dimensions"], {"width": 64, "height": 24})

    def test_image_dimension_mismatch_fails_verification(self):
        ok, encoded = cv2.imencode(".png", np.zeros((12, 20, 3), dtype=np.uint8))
        self.assertTrue(ok)

        result = verify_image_output(encoded.tobytes(), {"width": 40, "height": 12})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checks"]["dimensions"], "failed")

    def test_video_verification_checks_frames_rate_duration_and_known_audio(self):
        observation = {
            "width": 320,
            "height": 180,
            "decoded_frames": 12,
            "fps": 8.0,
            "duration_seconds": 1.5,
            "audio_present": True,
            "consistent_dimensions": True,
        }
        expected = {
            "width": 320,
            "height": 180,
            "frames": 12,
            "fps": 8.0,
            "duration_seconds": 1.5,
            "audio_present": True,
        }

        result = verify_video_observation(observation, expected)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checks"]["audio"], "passed")

    def test_known_audio_is_partial_when_probe_cannot_observe_audio(self):
        observation = {
            "width": 320,
            "height": 180,
            "decoded_frames": 12,
            "fps": 8.0,
            "duration_seconds": 1.5,
            "audio_present": None,
            "consistent_dimensions": True,
        }
        expected = {"width": 320, "height": 180, "frames": 12, "fps": 8.0,
                    "duration_seconds": 1.5, "audio_present": True}

        result = verify_video_observation(observation, expected)

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["checks"]["audio"], "unavailable")


class SummaryTests(unittest.TestCase):
    def test_only_successful_comparable_runs_contribute_to_median_and_range(self):
        common = {
            "fixture_id": "motion",
            "fixture_identity_sha256": "fixture-hash",
            "source_sha256": "source-hash",
            "settings": {"depth_mode": "balanced", "execution_mode": "cpu"},
            "model_hash": None,
            "execution": {"active_provider": "OpenCV CPU", "device": "CPU"},
            "environment": {"runtime_fingerprint": "runtime-a"},
            "verification": {"status": "passed"},
        }
        runs = [
            {**common, "success": True, "runner_timings": {"total_wall_seconds": 2.0}},
            {**common, "success": True, "runner_timings": {"total_wall_seconds": 4.0}},
            {**common, "success": False, "runner_timings": {"total_wall_seconds": 1.0}},
            {**common, "success": True,
             "execution": {"active_provider": "DmlExecutionProvider", "device": "GPU"},
             "runner_timings": {"total_wall_seconds": 7.0}},
        ]

        summary = summarize_runs(runs)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["successful_runs"], 2)
        self.assertEqual(summary[0]["wall_seconds"]["median"], 3.0)
        self.assertEqual(summary[0]["wall_seconds"]["range"], [2.0, 4.0])

    def test_unknown_runtime_does_not_create_comparison_summary(self):
        common = {
            "fixture_id": "motion",
            "fixture_identity_sha256": "fixture-hash",
            "source_sha256": "source-hash",
            "settings": {"depth_mode": "balanced", "execution_mode": "cpu"},
            "model_hash": None,
            "execution": {"active_provider": "OpenCV CPU", "device": "CPU"},
            "environment": {"runtime_fingerprint": None},
            "verification": {"status": "passed"},
            "success": True,
        }

        summary = summarize_runs([
            {**common, "runner_timings": {"total_wall_seconds": 1.0}},
            {**common, "runner_timings": {"total_wall_seconds": 1.2}},
        ])

        self.assertEqual(summary, [])

    def test_missing_job_timing_fields_are_explicit_nulls(self):
        timings = job_timing_snapshot({"state": "completed"})

        self.assertIn("elapsed_seconds", timings)
        self.assertIsNone(timings["elapsed_seconds"])
        self.assertIsNone(timings["stage_timings"])

    def test_missing_execution_telemetry_is_explicit_null(self):
        execution = normalize_execution_snapshot({"active_provider": "OpenCV CPU"})

        self.assertEqual(execution["active_provider"], "OpenCV CPU")
        self.assertIsNone(execution["session_reused"])
        self.assertIsNone(execution["requested_input_width"])
        self.assertIsNone(execution["requested_input_height"])
        self.assertIsNone(execution["session_initialization_seconds"])
        self.assertIsNone(execution["inference_pipeline_seconds"])

    def test_cli_exit_code_is_nonzero_for_failed_or_unrepeated_results(self):
        self.assertEqual(benchmark_exit_code({"runs": [], "summaries": []}), 1)
        self.assertEqual(benchmark_exit_code({"runs": [{"success": False}], "summaries": []}), 1)
        self.assertEqual(benchmark_exit_code({"runs": [{"success": True}], "summaries": []}), 1)
        self.assertEqual(benchmark_exit_code({"runs": [{"success": True}],
                                               "summaries": [{"successful_runs": 3}]}), 0)


if __name__ == "__main__":
    unittest.main()
