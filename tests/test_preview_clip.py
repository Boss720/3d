"""CPU-only extraction and ownership checks; model inference is never used."""
import json
import os
import shutil
import subprocess
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

from preview_clip import ClipCancelled, prepare_preview_clip


class PreviewClipTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parents[1] / '.runtime' / 'preview-clip-tests' / uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.addCleanup(self.cleanup_files)
        self.source = self.folder / 'original.mp4'
        self.destination = self.folder / 'proxy.mp4'
        self.source.write_bytes(b'original-source')
        self.cancel = threading.Event()
        self.process = Mock()
        self.process.poll.return_value = 0
        self.process.wait.return_value = 0

    def cleanup_files(self):
        for path in self.folder.iterdir():
            path.unlink()
        self.folder.rmdir()

    def launch(self, command, **kwargs):
        self.destination.write_bytes(b'encoded-proxy')
        return self.process

    def prepare(self, **changes):
        args = dict(ffmpeg='ffmpeg', source=self.source, destination=self.destination,
                    start_seconds=2.5, duration_seconds=5, source_fps=60,
                    cancel_event=self.cancel)
        args.update(changes)
        return prepare_preview_clip(**args)

    @patch('preview_clip.subprocess.Popen', create=True)
    def test_cpu_proxy_command_enforces_interval_dimensions_frame_and_codec_bounds(self, popen):
        popen.side_effect = self.launch
        callback = Mock()
        result = self.prepare(on_process=callback)
        self.assertEqual(result, self.destination)
        self.assertEqual(self.source.read_bytes(), b'original-source')
        command = popen.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertEqual(command[command.index('-ss') + 1], '2.5')
        self.assertEqual(command[command.index('-t') + 1], '5')
        self.assertEqual(command[command.index('-frames:v') + 1], '120')
        self.assertEqual(command[command.index('-c:v') + 1], 'libx264')
        self.assertEqual(command[command.index('-preset') + 1], 'ultrafast')
        self.assertEqual(command[command.index('-pix_fmt') + 1], 'yuv420p')
        self.assertIn('0:a:0?', command)
        self.assertEqual(command[command.index('-c:a') + 1], 'aac')
        filters = command[command.index('-vf') + 1]
        self.assertIn('960', filters)
        self.assertIn('540', filters)
        self.assertIn('fps=24', filters)
        self.assertIn('trunc', filters)
        self.assertFalse(popen.call_args.kwargs.get('shell', False))
        self.assertIsNot(popen.call_args.kwargs['stderr'], subprocess.PIPE)
        self.assertEqual(callback.call_args_list, [unittest.mock.call(self.process), unittest.mock.call(None)])

    def test_invalid_parameters_and_paths_never_launch_or_mutate_sources(self):
        cases = ({'start_seconds': -1}, {'start_seconds': float('nan')}, {'start_seconds': float('inf')},
                 {'duration_seconds': .99}, {'duration_seconds': 5.01}, {'duration_seconds': float('nan')},
                 {'timeout_seconds': 0}, {'timeout_seconds': float('inf')}, {'timeout_seconds': float('nan')},
                 {'source': self.source.with_name('absent.mp4')}, {'destination': self.source})
        with patch('preview_clip.subprocess.Popen', create=True) as popen:
            for settings in cases:
                with self.subTest(settings=settings), self.assertRaises(ValueError):
                    self.prepare(**settings)
                self.assertFalse(self.destination.exists())
            self.destination.write_bytes(b'existing-output')
            with self.assertRaises(ValueError):
                self.prepare()
            popen.assert_not_called()
        self.assertEqual(self.source.read_bytes(), b'original-source')
        self.assertEqual(self.destination.read_bytes(), b'existing-output')

    def test_already_cancelled_never_launches_or_creates_output(self):
        self.cancel.set()
        with patch('preview_clip.subprocess.Popen', create=True) as popen, self.assertRaises(ClipCancelled):
            self.prepare()
        popen.assert_not_called()
        self.assertFalse(self.destination.exists())

    def test_destination_created_by_another_owner_during_reservation_is_preserved(self):
        original_open = Path.open
        def competing_open(path, mode='r', *args, **kwargs):
            if path == self.destination and mode == 'xb':
                with original_open(path, 'wb') as output:
                    output.write(b'another-owner-output')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', competing_open), patch('preview_clip.subprocess.Popen') as popen, \
                self.assertRaises(ValueError):
            self.prepare()
        popen.assert_not_called()
        self.assertEqual(self.destination.read_bytes(), b'another-owner-output')

    def test_cancellation_during_poll_wait_stops_the_process(self):
        self.process.poll.return_value = None
        with patch('preview_clip.subprocess.Popen', side_effect=self.launch), \
                patch.object(self.cancel, 'wait', side_effect=lambda interval: self.cancel.set()) as wait, \
                self.assertRaises(ClipCancelled):
            self.prepare()
        self.assertEqual(wait.call_args.args, (.05,))
        self.process.terminate.assert_called_once()
        self.assertFalse(self.destination.exists())

    def test_cancellation_terminates_only_owned_process_and_removes_partial(self):
        self.process.poll.return_value = None
        callback = Mock(side_effect=lambda process: self.cancel.set() if process else None)
        with patch('preview_clip.subprocess.Popen', create=True, side_effect=self.launch), self.assertRaises(ClipCancelled):
            self.prepare(on_process=callback)
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once()
        self.assertLessEqual(self.process.wait.call_args.kwargs['timeout'], 2)
        self.assertEqual(callback.call_args_list[-1], unittest.mock.call(None))
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.source.is_file())

    def test_timeout_kills_process_if_terminate_does_not_finish(self):
        self.process.poll.return_value = None
        self.process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 1), 0]
        with patch('preview_clip.subprocess.Popen', create=True, side_effect=self.launch), \
                patch('preview_clip.time.monotonic', side_effect=[100, 101]), self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.prepare(timeout_seconds=.5)
        self.process.terminate.assert_called_once()
        self.process.kill.assert_called_once()
        for call in self.process.wait.call_args_list:
            self.assertLessEqual(call.kwargs['timeout'], 2)
        self.assertFalse(self.destination.exists())

    def test_failure_uses_bounded_stderr_tail_and_cleans_own_partial(self):
        self.process.poll.return_value = 1
        def failed(command, **kwargs):
            self.launch(command, **kwargs)
            kwargs['stderr'].write(b'x' * 20000 + b' actual encoder error')
            return self.process
        callback = Mock()
        with patch('preview_clip.subprocess.Popen', create=True, side_effect=failed), self.assertRaises(RuntimeError) as caught:
            self.prepare(on_process=callback)
        self.assertIn('actual encoder error', str(caught.exception))
        self.assertLess(len(str(caught.exception)), 9000)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.source.read_bytes(), b'original-source')
        self.assertEqual(callback.call_args_list[-1], unittest.mock.call(None))

    def test_launch_failure_cleans_reserved_destination_and_clears_callback(self):
        callback = Mock()
        with patch('preview_clip.subprocess.Popen', create=True, side_effect=OSError('missing binary')), \
                self.assertRaisesRegex(RuntimeError, 'missing binary'):
            self.prepare(on_process=callback)
        self.assertFalse(self.destination.exists())
        self.assertEqual(callback.call_args_list, [unittest.mock.call(None)])

    def test_registration_failure_stops_owned_process_cleans_output_and_clears_registration(self):
        self.process.poll.return_value = None
        callback = Mock(side_effect=[RuntimeError('observer failed'), None])
        with patch('preview_clip.subprocess.Popen', create=True, side_effect=self.launch), \
                self.assertRaisesRegex(RuntimeError, 'observer failed'):
            self.prepare(on_process=callback)
        self.process.terminate.assert_called_once()
        self.assertFalse(self.destination.exists())
        self.assertEqual(callback.call_args_list[-1], unittest.mock.call(None))

    def test_success_without_nonempty_output_is_failure(self):
        with patch('preview_clip.subprocess.Popen', create=True, return_value=self.process), self.assertRaises(RuntimeError):
            self.prepare()
        self.assertFalse(self.destination.exists())

    def test_source_fps_caps_or_falls_back_to_24(self):
        for value, expected, frame_limit in ((.5, 'fps=1', '5'), (12, 'fps=12', '60'), (60, 'fps=24', '120'),
                                             (None, 'fps=24', '120'), (0, 'fps=24', '120'),
                                             (float('nan'), 'fps=24', '120'), (float('inf'), 'fps=24', '120')):
            with self.subTest(source_fps=value), patch('preview_clip.subprocess.Popen', create=True, side_effect=self.launch) as popen:
                self.prepare(source_fps=value)
                command = popen.call_args.args[0]
                self.assertIn(expected, command[command.index('-vf') + 1])
                self.assertEqual(command[command.index('-frames:v') + 1], frame_limit)
            self.destination.unlink()

    def test_real_cpu_extraction_seeks_bounds_video_and_preserves_optional_audio(self):
        import cv2

        ffmpeg = os.environ.get('VIDEO3D_FFMPEG') or shutil.which('ffmpeg')
        if not ffmpeg:
            app = Path(os.environ.get('LOCALAPPDATA', '')) / 'owl3d-desktop-app'
            choices = sorted(app.glob('app-*/resources/app/.webpack/renderer/main_window/bin/win/x64/ffmpeg.exe'))
            ffmpeg = str(choices[-1]) if choices else None
        if not ffmpeg:
            self.skipTest('FFmpeg is unavailable for generated CPU extraction.')
        ffprobe = Path(ffmpeg).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')
        if not ffprobe.is_file():
            self.skipTest('FFprobe is unavailable for generated CPU extraction.')
        for width, height, source_fps, audio, duration, expected in ((1200, 600, 30, True, 5, (960, 480, 24)),
                                                                   (160, 94, 10, False, 1, (160, 94, 10)),
                                                                   (160, 94, .5, False, 3, (160, 94, 1))):
            with self.subTest(width=width, source_fps=source_fps, audio=audio):
                command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                           '-f', 'lavfi', '-i', f'testsrc2=size={width}x{height}:rate={source_fps}:duration={duration + 1}']
                if audio:
                    command += ['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration + 1}']
                command += ['-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p']
                command += ['-c:a', 'aac'] if audio else ['-an']
                command += [str(self.source)]
                fixture = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, timeout=15)
                self.assertEqual(fixture.returncode, 0, fixture.stderr[-2000:].decode(errors='replace'))
                self.prepare(ffmpeg=str(ffmpeg), start_seconds=.5, duration_seconds=duration, source_fps=source_fps)
                probe = subprocess.run([str(ffprobe), '-v', 'error', '-count_frames', '-show_streams', '-show_format',
                                        '-of', 'json', str(self.destination)], capture_output=True, timeout=5, check=True)
                info = json.loads(probe.stdout)
                video = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')
                self.assertEqual((video['width'], video['height']), expected[:2])
                numerator, denominator = map(int, video['r_frame_rate'].split('/'))
                self.assertEqual(numerator / denominator, expected[2])
                self.assertLessEqual(int(video['nb_read_frames']), 120)
                if source_fps >= 1:
                    self.assertEqual(int(video['nb_read_frames']), expected[2] * duration)
                else:
                    # Sparse-source seek can yield fewer frames; never manufacture
                    # a long held-frame tail or claim the requested duration was met.
                    self.assertGreater(int(video['nb_read_frames']), 0)
                    self.assertLessEqual(int(video['nb_read_frames']), expected[2] * duration)
                self.assertEqual(video['pix_fmt'], 'yuv420p')
                self.assertEqual(video['codec_name'], 'h264')
                self.assertEqual(any(stream['codec_type'] == 'audio' for stream in info['streams']), audio)
                actual_duration = int(video['nb_read_frames']) / expected[2]
                self.assertAlmostEqual(float(info['format']['duration']), actual_duration, delta=.1)
                self.assertLessEqual(actual_duration, duration + .05)
                capture = cv2.VideoCapture(str(self.destination))
                try:
                    ok, frame = capture.read()
                    self.assertTrue(ok)
                    self.assertEqual(frame.shape[:2], (expected[1], expected[0]))
                finally:
                    capture.release()
                self.destination.unlink()
