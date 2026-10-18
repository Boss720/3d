"""CPU-only contract tests for the live processed-clip verification runner."""
import unittest
from unittest.mock import Mock

import numpy as np

from scripts.verify_clip_preview import (
    checked_submit,
    classify_pixel_gate,
    compare_decoded_frames,
    cancellation_progress_reached,
    require_versioned_clip_capability,
    require_export_execution,
    require_pixel_gate_passed,
    validate_clip_bounds,
)


CAPABILITY = {
    "schema_version": 1,
    "processed_clip": {
        "supported": True,
        "max_duration_seconds": 5,
        "max_frames": 120,
        "max_fps": 24,
        "max_width": 960,
        "max_height": 540,
        "temporal_context": "fresh",
    },
}


class ClipRunnerContractTests(unittest.TestCase):
    def test_capability_failure_happens_before_any_job_post(self):
        api = Mock()
        api.status.return_value = {"preview_capabilities": {"schema_version": 0}}

        with self.assertRaisesRegex(ValueError, "capability"):
            checked_submit(api, "fixture.mp4", {"purpose": "preview_clip"}, None)

        api.submit.assert_not_called()
        api.jobs.assert_not_called()

    def test_versioned_capability_requires_supported_bounded_clip(self):
        capability = require_versioned_clip_capability({"preview_capabilities": CAPABILITY})
        self.assertEqual(capability["max_frames"], 120)
        with self.assertRaisesRegex(ValueError, "capability"):
            require_versioned_clip_capability({"preview_capabilities": {**CAPABILITY, "schema_version": 2}})
        unavailable = {"schema_version": 1, "processed_clip": {**CAPABILITY["processed_clip"], "supported": False}}
        with self.assertRaisesRegex(ValueError, "capability"):
            require_versioned_clip_capability({"preview_capabilities": unavailable})

    def test_clip_bounds_enforce_interval_fps_dimensions_and_frame_cap(self):
        args = dict(start_seconds=9, duration_seconds=3, source_duration=30,
                    source_fps=12, width=320, height=180, capability=CAPABILITY["processed_clip"])
        result = validate_clip_bounds(**args)
        self.assertEqual(result["expected_frames"], 36)
        capped = validate_clip_bounds(**{**args, "source_fps": 30})
        self.assertEqual(capped["fps"], 24)
        too_many = {**CAPABILITY["processed_clip"], "max_frames": 35}
        with self.assertRaisesRegex(ValueError, "frame bound"):
            validate_clip_bounds(**{**args, "capability": too_many})
        invalid = (
            {"start_seconds": -1}, {"duration_seconds": .5}, {"duration_seconds": 6},
            {"start_seconds": 28}, {"width": 1000}, {"height": 541},
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_clip_bounds(**{**args, **change})

    def test_cpu_comparison_is_exact_and_gpu_allows_only_sparse_one_level_rounding(self):
        reference = np.zeros((36, 8, 8, 3), dtype=np.uint8)
        exact = reference.copy()
        self.assertTrue(compare_decoded_frames(reference, exact, "cpu")["passed"])

        one_level = reference.copy()
        one_level[0, 0, 0, 0] = 1
        self.assertFalse(compare_decoded_frames(reference, one_level, "cpu")["passed"])
        gpu_result = compare_decoded_frames(reference, one_level, "gpu")
        self.assertTrue(gpu_result["passed"])
        self.assertEqual(gpu_result["max_abs_difference"], 1)
        self.assertLessEqual(gpu_result["mean_abs_difference"], .001)

        dense = reference.copy()
        dense[:, :, :, :] = 1
        self.assertFalse(compare_decoded_frames(reference, dense, "gpu")["passed"])
        large = reference.copy()
        large[0, 0, 0, 0] = 2
        self.assertFalse(compare_decoded_frames(reference, large, "gpu")["passed"])

    def test_inference_cancel_gate_requires_two_processed_frames(self):
        self.assertFalse(cancellation_progress_reached({"state": "processing", "frames_processed": 1}))
        self.assertFalse(cancellation_progress_reached({"state": "queued", "frames_processed": 3}))
        self.assertTrue(cancellation_progress_reached({"state": "processing", "frames_processed": 2}))

    def test_nontransitive_gpu_tolerance_requires_clip_to_match_both_exports(self):
        reference = np.full((1, 20, 50, 3), 128, dtype=np.uint8)
        repeat, clip = reference.copy(), reference.copy()
        repeat[0, 0, 0, 0], clip[0, 0, 0, 0] = 129, 127
        self_variation = compare_decoded_frames(reference, repeat, "gpu")
        comparison = compare_decoded_frames(clip, reference, "gpu")
        repeat_comparison = compare_decoded_frames(clip, repeat, "gpu")
        self.assertTrue(self_variation["passed"])
        self.assertTrue(comparison["passed"])
        self.assertFalse(repeat_comparison["passed"])
        self.assertEqual(repeat_comparison["max_abs_difference"], 2)
        conclusion = classify_pixel_gate(self_variation, comparison, repeat_comparison)
        self.assertEqual(conclusion["status"], "failed")
        with self.assertRaisesRegex(RuntimeError, "Pixel gate failed"):
            require_pixel_gate_passed(conclusion)

    def test_pixel_gate_preserves_inconclusive_precedence_and_terminal_failure(self):
        conclusion = classify_pixel_gate({"passed": False}, {"passed": False}, {"passed": True})
        self.assertEqual(conclusion["status"], "inconclusive")
        with self.assertRaisesRegex(RuntimeError, "Pixel gate inconclusive"):
            require_pixel_gate_passed(conclusion)
        passed = classify_pixel_gate({"passed": True}, {"passed": True}, {"passed": True})
        self.assertEqual(passed["status"], "passed")
        require_pixel_gate_passed(passed)

    def test_export_requires_public_purpose_device_and_execution_proof(self):
        valid = {"purpose": "export", "execution_mode": "gpu", "execution": {
            "active_provider": "DmlExecutionProvider", "gpu_verified": True, "fallback_reason": None}}
        self.assertEqual(require_export_execution(valid, "gpu"), valid["execution"])
        invalid = ({}, {**valid, "purpose": "preview_clip"}, {**valid, "execution_mode": "cpu"},
                   {**valid, "execution": {}}, {**valid, "execution": {
                       **valid["execution"], "fallback_reason": "GPU failed"}})
        for job in invalid:
            with self.subTest(job=job), self.assertRaises(RuntimeError):
                require_export_execution(job, "gpu")


if __name__ == "__main__":
    unittest.main()
