import os
import subprocess
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from preflight import inspect_media


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / '.runtime' / 'preflight-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.source = self.root / 'source.bin'
        self.source.write_bytes(b'local source')
        self.ffmpeg = self.root / 'ffmpeg.exe'
        self.ffmpeg.write_bytes(b'fake executable')
        self.run = patch('preflight.subprocess.run', return_value=subprocess.CompletedProcess([], 0, ' V....D libx264 H.264 encoder\n', ''))
        self.run_mock = self.run.start()
        self.disk = patch('preflight.shutil.disk_usage', return_value=SimpleNamespace(free=10**12))
        self.disk.start()

    def tearDown(self):
        self.run.stop()
        self.disk.stop()
        quarantine = self.root.parents[1] / '_quarantine' / 'preflight-tests' / self.root.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, quarantine)

    def image(self, width, height, layout='SBS', resolution='source'):
        with patch('preflight.cv2.imread', return_value=np.zeros((height, width, 3), np.uint8)):
            return inspect_media(self.source, 'image', layout, resolution, self.root, None)

    def capture(self, width=641, height=479, frames=30, fps=25, decoded=True):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FRAME_COUNT: frames, cv2.CAP_PROP_FPS: fps,
        }[prop]
        capture.read.return_value = (decoded, np.zeros((479, 641, 3), np.uint8) if decoded else None)
        return capture

    def video(self, capture, layout='SBS', resolution='source'):
        with patch('preflight.cv2.VideoCapture', return_value=capture):
            return inspect_media(self.source, 'video', layout, resolution, self.root, str(self.ffmpeg))

    def test_image_odd_and_half_layout_dimensions_follow_actual_packing(self):
        self.assertEqual((self.image(641, 479)['output_width'], self.image(641, 479)['output_height']), (1282, 479))
        self.assertEqual(self.image(641, 479, 'HALFSBS')['output_width'], 640)
        self.assertEqual(self.image(641, 479, 'HALFTB')['output_height'], 478)
        self.run_mock.assert_not_called()

    def test_portrait_1080_image_has_no_even_rounding(self):
        report = self.image(1001, 2000, 'SBS', '1080p')
        self.assertEqual((report['source_width'], report['source_height']), (1001, 2000))
        self.assertEqual((report['output_width'], report['output_height']), (1082, 1080))
        self.assertEqual(report['total_frames'], 1)
        self.assertIsNone(report['fps'])
        self.assertFalse(report['estimate_is_guarantee'])
        self.assertGreater(report['estimated_output_bytes'], 0)
        self.assertNotIn(str(self.root), str(report))

    def test_video_uses_even_rounding_and_half_layout_multiple_of_four(self):
        for layout, expected in [('SBS', (1280, 480)), ('HALFSBS', (640, 480)), ('HALFTB', (640, 480))]:
            with self.subTest(layout=layout):
                capture = self.capture()
                report = self.video(capture, layout)
                self.assertEqual((report['output_width'], report['output_height']), expected)
                self.assertEqual(report['fps'], 25)
                self.assertEqual(report['total_frames'], 30)
                capture.read.assert_called_once()
                capture.release.assert_called_once()
        report = self.video(self.capture(width=1001, height=2000), 'SBS', '1080p')
        self.assertEqual((report['output_width'], report['output_height']), (1080, 1080))
        report = self.video(self.capture(width=646, height=482), 'HALFSBS')
        self.assertEqual((report['output_width'], report['output_height']), (644, 482))
        report = self.video(self.capture(width=646, height=482), 'HALFTB')
        self.assertEqual((report['output_width'], report['output_height']), (646, 480))

    def test_corrupt_source_fails_with_recovery_before_encoder_work(self):
        with patch('preflight.cv2.imread', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'JPG|PNG|re-encode'):
                inspect_media(self.source, 'image', 'SBS', 'source', self.root, None)
        capture = self.capture(decoded=False)
        with self.assertRaisesRegex(RuntimeError, 're-encode|Re-encode'):
            self.video(capture)
        capture.release.assert_called_once()
        self.run_mock.assert_not_called()

    def test_video_open_failure_and_invalid_dimensions_are_actionable(self):
        capture = self.capture()
        capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, 're-encode|Re-encode'):
            self.video(capture)
        capture.release.assert_called_once()
        for width in [0, -1, float('nan'), float('inf')]:
            with self.subTest(width=width), self.assertRaisesRegex(RuntimeError, 'dimensions|frame size'):
                self.video(self.capture(width=width))

    def test_unknown_video_count_and_invalid_fps_are_explicit(self):
        for count in [0, float('nan'), -1, float('inf')]:
            with self.subTest(count=count):
                report = self.video(self.capture(frames=count, fps=float('nan')))
                self.assertIsNone(report['total_frames'])
                self.assertIsNone(report['estimated_output_bytes'])
                self.assertEqual(report['fps'], 30)
                self.assertTrue(report['warnings'])
                self.assertTrue(report['fps_provisional'])

    def test_positive_sub_one_fps_is_preserved(self):
        report = self.video(self.capture(frames=3, fps=.5))
        self.assertEqual(report['fps'], .5)
        self.assertFalse(report['fps_provisional'])

    def test_minimum_disk_reserve_fails_but_uncertain_estimate_only_warns(self):
        with patch('preflight.shutil.disk_usage', return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(RuntimeError, 'space|Space'):
                self.image(10, 10)
        with patch('preflight.shutil.disk_usage', return_value=SimpleNamespace(free=20 * 1024**2)):
            report = self.video(self.capture(frames=1_000_000))
            self.assertGreater(report['estimated_output_bytes'], report['disk_free_bytes'])
            self.assertTrue(any('space' in warning.lower() for warning in report['warnings']))
            self.assertFalse(report['estimate_is_guarantee'])

    def test_output_write_probe_failure_is_actionable(self):
        with patch('preflight.tempfile.TemporaryFile', side_effect=PermissionError('denied')):
            with self.assertRaisesRegex(RuntimeError, 'writable|permission|Permission'):
                self.image(10, 10)

    def test_missing_libx264_or_ffmpeg_fails_with_install_action(self):
        self.run_mock.return_value = subprocess.CompletedProcess([], 0, ' V....D h264_nvenc hardware encoder\n', '')
        with self.assertRaisesRegex(RuntimeError, 'libx264'):
            self.video(self.capture())
        with patch('preflight.cv2.VideoCapture', return_value=self.capture()):
            with self.assertRaisesRegex(RuntimeError, 'Install FFmpeg|install FFmpeg'):
                inspect_media(self.source, 'video', 'SBS', 'source', self.root, None)

    def test_encoder_probe_is_cached_and_replaced_binary_invalidates_it(self):
        self.video(self.capture())
        self.video(self.capture())
        self.assertEqual(self.run_mock.call_count, 1)
        self.ffmpeg.write_bytes(b'replaced fake executable, different size')
        self.video(self.capture())
        self.assertEqual(self.run_mock.call_count, 2)
        command = self.run_mock.call_args.args[0]
        self.assertIn('-encoders', command)
        self.assertLessEqual(self.run_mock.call_args.kwargs['timeout'], 10)

    def test_encoder_probe_timeout_is_actionable(self):
        self.run_mock.side_effect = subprocess.TimeoutExpired('ffmpeg', 6)
        with self.assertRaisesRegex(RuntimeError, 'FFmpeg.*install|Install.*FFmpeg'):
            self.video(self.capture())

    def test_encoder_inventory_rejects_failed_command_and_partial_codec_name(self):
        self.run_mock.return_value = subprocess.CompletedProcess([], 1, '', 'failure')
        with self.assertRaisesRegex(RuntimeError, 'Install FFmpeg'):
            self.video(self.capture())
        self.run_mock.return_value = subprocess.CompletedProcess([], 0, ' V....D libx264rgb RGB only\n', '')
        with self.assertRaisesRegex(RuntimeError, 'libx264'):
            self.video(self.capture())

    def test_output_folder_probe_leaves_no_files(self):
        before = set(self.root.iterdir())
        self.image(10, 10)
        self.assertEqual(set(self.root.iterdir()), before)

    def test_missing_source_and_unsupported_settings_fail_early(self):
        with self.assertRaisesRegex(RuntimeError, 'source|Source'):
            inspect_media(self.root / 'missing.png', 'image', 'SBS', 'source', self.root, None)
        for media_type, layout, resolution in [('audio', 'SBS', 'source'), ('image', 'BOGUS', 'source'), ('image', 'SBS', '8k')]:
            with self.subTest(settings=(media_type, layout, resolution)), self.assertRaises(ValueError):
                inspect_media(self.source, media_type, layout, resolution, self.root, None)


if __name__ == '__main__':
    unittest.main()
