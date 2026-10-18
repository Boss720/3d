import subprocess
import unittest
from unittest.mock import patch

from video_encoder import (
    clear_encoder_probe_cache,
    select_encoder_profile,
)


class EncoderProfileTests(unittest.TestCase):
    def setUp(self):
        clear_encoder_probe_cache()

    def tearDown(self):
        clear_encoder_probe_cache()

    @patch("video_encoder.subprocess.run")
    def test_selects_nvenc_only_after_successful_smoke_encode(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")

        profile = select_encoder_profile("C:/tools/ffmpeg.exe", "balanced")

        self.assertEqual(profile.codec, "h264_nvenc")
        self.assertTrue(profile.hardware_available)
        self.assertEqual(profile.video_args, ("-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "24", "-b:v", "0"))
        self.assertEqual(profile.software_fallback_args, ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20"))
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(command[0], "C:/tools/ffmpeg.exe")
        self.assertIn("h264_nvenc", command)
        self.assertIn("-f", command)
        self.assertIn("null", command)
        self.assertLessEqual(run.call_args.kwargs["timeout"], 10)

    @patch("video_encoder.subprocess.run")
    def test_absent_encoder_uses_software_profile(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "Unknown encoder 'h264_nvenc'")

        profile = select_encoder_profile("C:/tools/ffmpeg.exe", "fast")

        self.assertEqual(profile.codec, "libx264")
        self.assertFalse(profile.hardware_available)
        self.assertEqual(profile.video_args, ("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"))
        self.assertEqual(profile.software_fallback_args, profile.video_args)

    @patch("video_encoder.subprocess.run", side_effect=FileNotFoundError("ffmpeg missing"))
    def test_probe_failure_uses_software_profile(self, run):
        profile = select_encoder_profile("missing-ffmpeg", "high")

        self.assertEqual(profile.codec, "libx264")
        self.assertFalse(profile.hardware_available)
        self.assertEqual(profile.software_fallback_args, ("-c:v", "libx264", "-preset", "medium", "-crf", "18"))

    @patch("video_encoder.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 6))
    def test_probe_timeout_uses_software_profile(self, run):
        profile = select_encoder_profile("C:/tools/ffmpeg.exe", "balanced")

        self.assertEqual(profile.codec, "libx264")
        self.assertFalse(profile.hardware_available)
        self.assertEqual(profile.software_fallback_args, ("-c:v", "libx264", "-preset", "veryfast", "-crf", "20"))

    @patch("video_encoder.subprocess.run")
    def test_quality_levels_have_distinct_nvenc_and_software_tradeoffs(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        expected = {
            "fast": ("p1", "30", "ultrafast", "22"),
            "balanced": ("p4", "24", "veryfast", "20"),
            "high": ("p6", "20", "medium", "18"),
        }

        for quality, (nvenc_preset, cq, x264_preset, crf) in expected.items():
            with self.subTest(quality=quality):
                clear_encoder_probe_cache()
                profile = select_encoder_profile("C:/tools/ffmpeg.exe", quality)
                self.assertIn(("-preset", nvenc_preset), tuple(zip(profile.video_args, profile.video_args[1:])))
                self.assertIn(("-cq", cq), tuple(zip(profile.video_args, profile.video_args[1:])))
                self.assertEqual(profile.software_fallback_args, ("-c:v", "libx264", "-preset", x264_preset, "-crf", crf))

    @patch("video_encoder.subprocess.run")
    def test_probe_result_is_cached_for_repeated_selection(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "", "")

        first = select_encoder_profile("C:/tools/ffmpeg.exe", "fast")
        second = select_encoder_profile("C:/tools/ffmpeg.exe", "fast")

        self.assertEqual(first.codec, "h264_nvenc")
        self.assertEqual(second.codec, "h264_nvenc")
        self.assertEqual(run.call_count, 1)

    def test_rejects_unknown_quality(self):
        with self.assertRaises(ValueError):
            select_encoder_profile("ffmpeg", "maximum")


if __name__ == "__main__":
    unittest.main()
