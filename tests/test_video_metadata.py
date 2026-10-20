"""Conversion tolerates unknown totals and owns capture lifetime from creation."""
import io
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import main
from video_encoder import EncoderProfile


class VideoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'video-metadata-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.outputs = patch.object(main, 'OUTPUTS', self.root)
        self.outputs.start()
        self.addCleanup(self.outputs.stop)
        self.addCleanup(self.quarantine)
        self.job = main.Job(id=str(uuid.uuid4()), filename='fixture.mp4',
                            input_path=self.root / 'fixture.mp4', media_type='video',
                            layout='SBS', strength=5, convergence=.5, depth_mode='balanced',
                            input_size=256, resolution='source', quality='fast', invert_depth=False)

    def quarantine(self):
        target = main.RUNTIME / '_quarantine' / 'video-metadata-tests' / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, target)

    def capture(self, count=float('nan'), width=16, height=12, fps=24, frames=1, position_ms=0):
        capture = Mock()
        capture.isOpened.return_value = True
        properties = {
            cv2.CAP_PROP_FRAME_WIDTH: width, cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FRAME_COUNT: count, cv2.CAP_PROP_FPS: fps,
        }
        capture.get.side_effect = lambda prop: (
            position_ms if prop == cv2.CAP_PROP_POS_MSEC else properties[prop])
        frame = np.full((int(height) if np.isfinite(height) and height > 0 else 12,
                         int(width) if np.isfinite(width) and width > 0 else 16, 3), 100, np.uint8)
        capture.read.side_effect = [(True, frame.copy()) for _ in range(frames)] + [(False, None)]
        return capture

    def run_video(self, capture, **changes):
        profile = EncoderProfile(quality='fast', codec='libx264', video_args=('-c:v', 'libx264'),
                                 software_fallback_args=('-c:v', 'libx264'), hardware_available=False)
        process = Mock()
        process.stdin = io.BytesIO()
        process.stderr = None
        process.poll.return_value = 0
        def finalize(*args, **kwargs):
            main.output_path_for(self.job).write_bytes(b'encoded video')
            return 0
        process.wait.side_effect = finalize
        with patch.object(main, 'FFMPEG_PATH', str(Path(main.__file__))), \
             patch.object(main, 'JOBS', {self.job.id: self.job}), \
             patch.object(main, 'open_video_capture', return_value=capture), \
             patch.object(main, 'ffprobe_fps', side_effect=lambda ffmpeg, source, fallback: fallback), \
             patch.object(main, 'select_encoder_profile', side_effect=changes.get('profile_error'),
                          return_value=profile), \
             patch.object(main.subprocess, 'Popen', side_effect=changes.get('spawn_error'),
                          return_value=process):
            return main.run_video(self.job, None)

    def test_unknown_frame_counts_convert_through_eof_without_an_eta(self):
        for count in (float('nan'), float('inf'), -1, 0):
            with self.subTest(count=count):
                capture = self.capture(count=count)
                output = self.run_video(capture)
                self.assertTrue(output.is_file())
                self.assertEqual(self.job.frames_processed, 1)
                self.assertIsNone(self.job.total_frames)
                self.assertIsNone(self.job.public()['eta_seconds'])
                capture.release.assert_called()

    def test_neural_video_filters_native_depth_and_only_infers_alternate_frames(self):
        self.job.depth_mode = 'depth-anything-v2-small'
        self.job.depth_interval = 2
        capture = self.capture(count=4, width=96, height=64, frames=4)
        native = np.full((16, 24), .5, np.float32)
        info = {'active_provider': 'DmlExecutionProvider', 'gpu_verified': True}
        flow = Mock()
        flow.calc.return_value = np.zeros((16, 24, 2), np.float32)
        with patch.object(main.DEPTH_ENGINE, 'predict_native_with_info', return_value=(native, info)) as infer, \
             patch.object(main.DEPTH_ENGINE, 'predict_with_info') as full_infer, \
             patch('video_depth.cv2.DISOpticalFlow_create', return_value=flow):
            self.run_video(capture)
        self.assertEqual(infer.call_count, 2)
        full_infer.assert_not_called()
        self.assertEqual(self.job.frames_processed, 4)
        self.assertEqual(self.job.execution['temporal_width'], 24)
        self.assertEqual(self.job.execution['temporal_height'], 16)
        self.assertEqual(self.job.execution['video_depth_stats']['flow_frames'], 2)
        self.assertEqual(self.job.first_execution['video_depth_stats']['frames'], 1)

    def test_decoder_cancellation_during_startup_keeps_cancellation_semantics(self):
        with patch.object(main, 'FFMPEG_PATH', str(Path(main.__file__))), \
             patch.object(main, 'open_video_capture', side_effect=main.DecoderCancelled('cancelled')):
            with self.assertRaises(main.JobCancelled):
                main.run_video(self.job, None)

    def test_approximate_frame_count_can_exceed_decoded_count_without_failing(self):
        capture = self.capture(count=31, fps=25, frames=30, position_ms=1160)

        output = self.run_video(capture)

        self.assertTrue(output.is_file())
        self.assertEqual(self.job.frames_processed, 30)
        self.assertEqual(self.job.total_frames, 31, 'The original metadata estimate must remain visible.')

    def test_decoder_failure_well_before_estimated_duration_fails(self):
        capture = self.capture(count=100, fps=25, frames=30, position_ms=1160)

        with self.assertRaisesRegex(RuntimeError, 'stopped early after 30 of approximately 100 expected frames'):
            self.run_video(capture)

        capture.release.assert_called()
        self.assertEqual(self.job.total_frames, 100)

    def test_invalid_dimensions_release_capture_and_have_an_actionable_error(self):
        for width in (float('nan'), float('inf'), -1, 0):
            with self.subTest(width=width):
                capture = self.capture(width=width)
                with self.assertRaisesRegex(RuntimeError, 'dimensions'):
                    self.run_video(capture)
                capture.release.assert_called()

    def test_encoder_setup_errors_release_capture(self):
        for change in ({'profile_error': RuntimeError('encoder unavailable')},
                       {'spawn_error': OSError('encoder could not start')}):
            with self.subTest(change=next(iter(change))):
                capture = self.capture(count=1)
                with self.assertRaises((RuntimeError, OSError)):
                    self.run_video(capture, **change)
                capture.release.assert_called()

    def test_unopened_capture_is_released(self):
        capture = self.capture()
        capture.isOpened.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'open this video'):
            self.run_video(capture)
        capture.release.assert_called()

    def test_stderr_reader_start_failure_stops_the_encoder(self):
        capture = self.capture(count=1)
        with patch.object(main.threading.Thread, 'start', side_effect=RuntimeError('reader unavailable')), \
             patch.object(main, 'stop_process') as stop:
            with self.assertRaisesRegex(RuntimeError, 'reader unavailable'):
                self.run_video(capture)
        stop.assert_called_once_with(self.job.process)
        capture.release.assert_called()


if __name__ == '__main__':
    unittest.main()
