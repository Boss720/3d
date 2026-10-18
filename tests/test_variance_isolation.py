"""CPU-only contract tests for bounded GPU variance isolation utilities."""
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.isolate_gpu_variance import (
    CapturingSession,
    assert_no_new_server_jobs,
    assert_server_jobs_idle,
    build_parser,
    compare_sequences,
    encoder_experiment,
    validate_frames,
)


class _Session:
    def __init__(self):
        self.outputs = [np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)]

    def run(self, names, feed, *args, **kwargs):
        return self.outputs

    def __getattr__(self, name):
        if name == "get_providers":
            return lambda: ["DmlExecutionProvider"]
        raise AttributeError(name)


class VarianceIsolationTests(unittest.TestCase):
    def test_capture_wrapper_copies_raw_outputs_and_forwards_session(self):
        captured = []
        source = _Session()
        wrapped = CapturingSession(source, captured.append)
        result = wrapped.run(None, {"input": np.zeros((1,), dtype=np.float32)})

        self.assertIs(result, source.outputs)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0].shape, (1, 1, 2, 4))
        source.outputs[0].fill(-1)
        self.assertTrue(np.array_equal(captured[0][0], np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)))
        self.assertEqual(wrapped.get_providers(), ["DmlExecutionProvider"])

    def test_raw_capture_rejects_nonfinite_and_over_bound_tensors(self):
        captured = []
        with self.assertRaisesRegex(ValueError, "byte bound"):
            CapturingSession(_Session(), captured.append, max_bytes=4).run(None, {})
        self.assertEqual(captured, [])
        source = _Session()
        source.outputs[0].fill(np.nan)
        with self.assertRaisesRegex(ValueError, "finite"):
            CapturingSession(source, captured.append).run(None, {})
        self.assertEqual(captured, [])

    def test_sequence_metrics_record_exactness_hashes_and_per_frame_changes(self):
        reference = np.zeros((2, 3, 4), dtype=np.float32)
        candidate = reference.copy()
        candidate[1, 0, 0] = 1
        result = compare_sequences(reference, candidate)

        self.assertFalse(result["exact_equal"])
        self.assertEqual(result["changed_values"], 1)
        self.assertEqual(result["max_abs_difference"], 1.0)
        self.assertEqual(len(result["per_frame"]), 2)
        self.assertNotEqual(result["reference_sha256"], result["candidate_sha256"])
        self.assertTrue(compare_sequences(reference, reference.copy())["exact_equal"])

    def test_metrics_reject_nonfinite_arrays_and_shape_mismatch(self):
        values = np.zeros((2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_sequences(values, np.full_like(values, np.nan))
        with self.assertRaisesRegex(ValueError, "shape"):
            compare_sequences(values, np.zeros((2, 4), dtype=np.float32))

    def test_fixture_frames_are_exactly_bounded_and_bgr(self):
        frames = [np.zeros((180, 320, 3), dtype=np.uint8) for _ in range(36)]
        self.assertEqual(validate_frames(frames, expected=36, width=320, height=180), (36, 320, 180))
        with self.assertRaisesRegex(ValueError, "36"):
            validate_frames(frames + frames[:1], expected=36, width=320, height=180)
        with self.assertRaisesRegex(ValueError, "BGR"):
            validate_frames([np.zeros((180, 320), dtype=np.uint8)], expected=1, width=320, height=180)

    def test_server_guard_rejects_any_nonterminal_job(self):
        assert_server_jobs_idle({8768: [], 8770: [{"id": "old", "state": "completed"}]})
        with self.assertRaisesRegex(RuntimeError, "8772"):
            assert_server_jobs_idle({8768: [], 8770: [], 8772: [{"id": "other", "state": "processing"}]})
        baseline = {"8768": ["kept"]}
        current = {8768: {"jobs": [{"id": "kept", "state": "completed"},
                                   {"id": "new", "state": "completed"}]}}
        with self.assertRaisesRegex(RuntimeError, "job appeared"):
            assert_no_new_server_jobs(baseline, current)

    def test_cli_defaults_to_cpu_and_requires_explicit_gpu_choice(self):
        self.assertEqual(build_parser().parse_args([]).device, "cpu")
        self.assertEqual(build_parser().parse_args(["--device", "gpu"]).device, "gpu")

    def test_cpu_encoder_experiment_never_probes_hardware(self):
        from video_encoder import _QUALITY_PROFILES

        rendered = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        with patch("video_encoder.select_encoder_profile", side_effect=AssertionError("hardware probe called")) as select, \
                patch("scripts.isolate_gpu_variance.activity_checkpoint"), \
                patch("scripts.isolate_gpu_variance.encode_once", side_effect=lambda ffmpeg, frames, args, target: {
                    "decoded": frames.copy(), "path": str(target), "output_sha256": "cpu"}), \
                patch("scripts.isolate_gpu_variance.write_report"):
            report = encoder_experiment(Path("unused"), "ffmpeg.exe", rendered, "cpu", {})

        select.assert_not_called()
        self.assertEqual(report["profile"]["codec"], "libx264")
        self.assertFalse(report["profile"]["hardware_available"])
        self.assertEqual(report["profile"]["video_args"],
                         ("-c:v", "libx264", *_QUALITY_PROFILES["fast"]["x264"]))
        self.assertEqual(set(report["encoders"]), {"libx264"})
        self.assertEqual(report["nvenc_skipped_reason"], "Explicit GPU mode was not requested.")

    def test_gpu_encoder_experiment_delegates_probe_and_uses_supported_nvenc(self):
        from video_encoder import EncoderProfile

        rendered = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        profile = EncoderProfile("fast", "h264_nvenc", ("-c:v", "h264_nvenc"),
                                 ("-c:v", "libx264"), True)
        with patch("video_encoder.select_encoder_profile", return_value=profile) as select, \
                patch("scripts.isolate_gpu_variance.activity_checkpoint"), \
                patch("scripts.isolate_gpu_variance.encode_once", side_effect=lambda ffmpeg, frames, args, target: {
                    "decoded": frames.copy(), "path": str(target), "output_sha256": "gpu"}), \
                patch("scripts.isolate_gpu_variance.write_report"):
            report = encoder_experiment(Path("unused"), "ffmpeg.exe", rendered, "gpu", {})

        select.assert_called_once_with("ffmpeg.exe", "fast")
        self.assertEqual(set(report["encoders"]), {"libx264", "h264_nvenc"})
        self.assertEqual(report["profile"]["codec"], "h264_nvenc")


if __name__ == "__main__":
    unittest.main()
