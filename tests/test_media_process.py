"""Probe parsing and actual owned-child cancellation/deadline regressions."""
import json
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import media_process as media


class MediaCommandTests(unittest.TestCase):
    def run_tracked(self, script, **kwargs):
        children = []
        spawn = subprocess.Popen

        def tracked(*args, **options):
            child = spawn(*args, **options)
            children.append(child)
            return child

        with patch.object(media.subprocess, 'Popen', side_effect=tracked):
            try:
                return media.run_media_command([sys.executable, '-c', script], **kwargs)
            finally:
                self.assertEqual(len(children), 1)
                self.assertIsNotNone(children[0].poll())
                self.assertTrue(children[0].stdout.closed)
                self.assertTrue(children[0].stderr.closed)

    def test_nonzero_returns_bounded_byte_outputs(self):
        result = self.run_tracked("import sys;sys.stdout.buffer.write(b'out');sys.stderr.buffer.write(b'err');sys.exit(7)")
        self.assertEqual((result.returncode, result.stdout, result.stderr), (7, b'out', b'err'))

    def test_real_stalled_child_timeout_terminates_and_closes_pipes(self):
        started = time.monotonic()
        with self.assertRaisesRegex(media.MediaCommandError, 'timed out'):
            self.run_tracked('import time;time.sleep(60)', timeout=.15)
        self.assertLess(time.monotonic() - started, 5)

    def test_real_stalled_child_cancellation_terminates_and_next_command_works(self):
        event = threading.Event()
        timer = threading.Timer(.15, event.set)
        timer.start()
        try:
            with self.assertRaises(media.MediaCancelled):
                self.run_tracked('import time;time.sleep(60)', cancel_event=event)
        finally:
            timer.cancel()
        self.assertEqual(self.run_tracked("print('next')").stdout.strip(), b'next')

    def test_no_newline_stdout_or_stderr_overflow_stops_real_child(self):
        for channel in ('stdout', 'stderr'):
            with self.subTest(channel=channel), self.assertRaisesRegex(media.MediaCommandError, 'output limit'):
                self.run_tracked(f"import sys,time;sys.{channel}.buffer.write(b'x'*200000);sys.{channel}.flush();time.sleep(60)", max_output_bytes=4096)

    def test_precancelled_does_not_spawn(self):
        event = threading.Event()
        event.set()
        with patch.object(media.subprocess, 'Popen') as spawn, self.assertRaises(media.MediaCancelled):
            media.run_media_command(['unused'], cancel_event=event)
        spawn.assert_not_called()


class ProbeTests(unittest.TestCase):
    def probe(self, stream=None, fmt=None):
        stream = stream if stream is not None else {'width': 160, 'height': 90,
            'avg_frame_rate': '30000/1001', 'r_frame_rate': '60/1', 'duration': '8',
            'start_time': '2', 'nb_frames': '200'}
        result = subprocess.CompletedProcess([], 0, json.dumps({'streams': [stream],
            'format': fmt or {'start_time': '1', 'duration': '10'}}).encode(), b'')
        with patch.object(media.Path, 'is_file', return_value=True), \
             patch.object(media, 'run_media_command', return_value=result) as run:
            metadata = media.probe_video('source.mp4', 'ffmpeg', timeout=3)
        self.assertEqual(run.call_args.kwargs['timeout'], 3)
        command = run.call_args.args[0]
        self.probe_command = command
        self.assertIn('-select_streams', command)
        self.assertNotIn('-show_streams', command)
        self.assertNotIn('-show_frames', command)
        return metadata

    def test_average_fraction_and_estimated_cfr_count(self):
        metadata = self.probe()
        self.assertEqual(metadata['fps_ratio'], '30000/1001')
        self.assertEqual(metadata['total_frames'], 240)
        self.assertEqual(metadata['duration_seconds'], 8)
        self.assertEqual(metadata['video_start_seconds'], 2)
        self.assertEqual(metadata['format_start_seconds'], 1)
        self.assertFalse(metadata['fps_provisional'])

    def test_probe_reports_pixel_aspect_and_color_characteristics(self):
        metadata = self.probe({'width': 160, 'height': 90, 'avg_frame_rate': '24/1',
                               'sample_aspect_ratio': '4:3', 'color_range': 'pc',
                               'color_space': 'bt709', 'color_transfer': 'bt709',
                               'color_primaries': 'bt709'})
        self.assertEqual(metadata['sample_aspect_ratio'], '4:3')
        self.assertEqual(metadata['color_range'], 'pc')
        self.assertEqual(metadata['color_space'], 'bt709')
        self.assertEqual(metadata['color_transfer'], 'bt709')
        self.assertEqual(metadata['color_primaries'], 'bt709')
        entries = next(arg for arg in self.probe_command if arg.startswith('stream='))
        for field in ('sample_aspect_ratio', 'color_range', 'color_space', 'color_transfer', 'color_primaries'):
            self.assertIn(field, entries)

    def test_missing_timestamps_use_provisional_rate_and_unknown_frame_count(self):
        metadata = self.probe(
            {'width': 160, 'height': 90, 'avg_frame_rate': '0/0', 'r_frame_rate': 'N/A',
             'duration': 'N/A', 'start_time': 'N/A', 'nb_frames': 'N/A'},
            {'start_time': '3.25', 'duration': 'N/A'})
        self.assertEqual(metadata['fps_ratio'], '30')
        self.assertEqual(metadata['fps'], 30)
        self.assertTrue(metadata['fps_provisional'])
        self.assertIsNone(metadata['duration_seconds'])
        self.assertIsNone(metadata['total_frames'])
        self.assertEqual(metadata['video_start_seconds'], 3.25)
        self.assertEqual(metadata['format_start_seconds'], 3.25)

    def test_display_matrix_rotation_swaps_output_geometry(self):
        metadata = self.probe({'width': 160, 'height': 90, 'avg_frame_rate': '15/1',
                               'side_data_list': [{'rotation': -90}]})
        self.assertEqual((metadata['width'], metadata['height']), (90, 160))
        self.assertEqual(metadata['rotation_degrees'], 270)

    def test_unknown_duration_count_and_bad_rate_are_explicit(self):
        metadata = self.probe({'width': 2, 'height': 1, 'avg_frame_rate': '0/0'},
                              {'duration': 'N/A', 'start_time': 'N/A'})
        self.assertEqual(metadata['fps'], 30)
        self.assertTrue(metadata['fps_provisional'])
        self.assertIsNone(metadata['total_frames'])
        self.assertIsNone(metadata['duration_seconds'])

    def test_container_duration_cannot_estimate_unknown_video_length(self):
        metadata = self.probe({'width': 2, 'height': 1, 'avg_frame_rate': '10/1', 'start_time': '2'},
                              {'duration': '10', 'start_time': '1'})
        self.assertIsNone(metadata['duration_seconds'])
        self.assertIsNone(metadata['total_frames'])
        self.assertEqual(metadata['format_duration_seconds'], 10)

    def test_invalid_and_excessive_geometry_fail_actionably(self):
        for width, rotation in ((0, 0), (9000, 0), (160, 45)):
            with self.subTest(width=width, rotation=rotation), self.assertRaisesRegex(media.MediaCommandError, 'Re-encode'):
                self.probe({'width': width, 'height': 90, 'tags': {'rotate': rotation}})

    def test_missing_ffprobe_is_actionable_and_never_opens_native_decoder(self):
        with patch.object(media.Path, 'is_file', return_value=False), \
             patch.object(media.shutil, 'which', return_value=None), \
             self.assertRaisesRegex(media.MediaCommandError, 'FFprobe is required'):
            media.probe_video('source.mp4', 'ffmpeg')


if __name__ == '__main__':
    unittest.main()
