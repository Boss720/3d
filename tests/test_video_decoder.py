"""Decoder framing, startup fallback, and owned-process lifecycle checks."""
import io
import subprocess
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import video_decoder as decoder


ADAPTER = {'id': 2, 'name': 'AMD Radeon', 'vendor_id': 0x1002, 'software': False}
METADATA = (2, 1, 2, 24., False)


class ChunkedPipe(io.BytesIO):
    def read(self, size=-1):
        return super().read(min(size, 2) if size >= 0 else size)


class DecoderTests(unittest.TestCase):
    def process(self, data, result=0):
        process = Mock()
        process.stdout = ChunkedPipe(data)
        process.stderr = io.BytesIO(b'codec detail')
        process.wait.return_value = result
        process.poll.return_value = result
        return process

    def capture(self, process):
        return decoder.D3D11Capture('input.mp4', 'ffmpeg', METADATA, ADAPTER,
                                   popen=Mock(return_value=process), read_timeout=1)

    def software(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {cv2.CAP_PROP_FRAME_WIDTH: 2,
            cv2.CAP_PROP_FRAME_HEIGHT: 1, cv2.CAP_PROP_FRAME_COUNT: 2,
            cv2.CAP_PROP_FPS: 24}[prop]
        return capture

    def test_adapter_is_physical_amd_and_ordinal_is_preserved(self):
        hardware = {'adapters': [dict(ADAPTER, id=0, vendor_id=0x10de),
                                dict(ADAPTER, id=1, software=True), ADAPTER]}
        self.assertEqual(decoder.select_amd_adapter(hardware), ADAPTER)
        self.assertIsNone(decoder.select_amd_adapter({'adapters': [dict(ADAPTER, vendor_id=None)]}))
        self.assertIsNone(decoder.select_amd_adapter({'adapters': [dict(ADAPTER, id=None)]}))
        command = decoder.decoder_command('ffmpeg', 'source.mp4', 2)
        self.assertEqual(command[command.index('-hwaccel_device') + 1], '2')
        self.assertLess(command.index('-hwaccel_device'), command.index('-i'))
        self.assertIn('hwdownload,format=nv12,format=bgr24', command)

    def test_short_pipe_reads_assemble_exact_frames_and_first_probe_is_not_dropped(self):
        process = self.process(bytes(range(12)))
        capture = self.capture(process)
        self.addCleanup(capture.release)
        frames = [capture.read(), capture.read()]
        self.assertTrue(all(ok for ok, _ in frames))
        np.testing.assert_array_equal(frames[0][1], np.arange(6, dtype=np.uint8).reshape(1, 2, 3))
        np.testing.assert_array_equal(frames[1][1], np.arange(6, 12, dtype=np.uint8).reshape(1, 2, 3))
        self.assertEqual(capture.get(cv2.CAP_PROP_POS_FRAMES), 2)
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(capture.read(), (False, None))

    def test_incomplete_first_frame_falls_back_and_releases_process(self):
        software = self.software()
        process = self.process(b'123')
        with patch.object(decoder, 'probe_hardware', return_value={'adapters': [ADAPTER]}), \
             patch.object(decoder.subprocess, 'Popen', return_value=process):
            capture = decoder.open_video_capture('in.mp4', 'ffmpeg',
                                                  software_capture_factory=Mock(return_value=software))
        self.assertFalse(capture.decoder_info['hardware'])
        self.assertIn('incomplete', capture.decoder_info['fallback_reason'])
        software.read.assert_not_called()
        software.release.assert_not_called()
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stderr.closed)
        capture.release()
        software.release.assert_called_once()

    def test_unknown_adapter_never_starts_ffmpeg(self):
        software = self.software()
        with patch.object(decoder, 'probe_hardware', return_value={'adapters': [], 'error': 'DXGI unavailable'}), \
             patch.object(decoder.subprocess, 'Popen') as popen:
            capture = decoder.open_video_capture('in.mp4', 'ffmpeg',
                                                  software_capture_factory=Mock(return_value=software))
        popen.assert_not_called()
        self.assertEqual(capture.decoder_info['fallback_reason'], 'DXGI unavailable')

    def test_explicit_software_preference_skips_hardware_probe_and_process(self):
        software = self.software()
        with patch.object(decoder, 'probe_hardware') as probe, \
             patch.object(decoder.subprocess, 'Popen') as popen:
            capture = decoder.open_video_capture('in.mp4', 'ffmpeg', prefer_hardware=False,
                                                  software_capture_factory=Mock(return_value=software))
        probe.assert_not_called()
        popen.assert_not_called()
        self.assertFalse(capture.decoder_info['hardware'])
        self.assertIsNone(capture.decoder_info['fallback_reason'])
        capture.release()
        software.release.assert_called_once()

    def test_hardware_success_releases_metadata_capture_and_exposes_actual_adapter(self):
        software = self.software()
        process = self.process(bytes(range(12)))
        with patch.object(decoder, 'probe_hardware', return_value={'adapters': [ADAPTER]}), \
             patch.object(decoder.subprocess, 'Popen', return_value=process):
            capture = decoder.open_video_capture('in.mp4', 'ffmpeg',
                                                  software_capture_factory=Mock(return_value=software))
        self.addCleanup(capture.release)
        software.release.assert_called_once()
        self.assertTrue(capture.decoder_info['hardware'])
        self.assertEqual(capture.decoder_info['adapter_id'], 2)
        self.assertTrue(capture.read()[0])

    def test_midstream_partial_frame_fails_instead_of_eof_or_software_fallback(self):
        capture = self.capture(self.process(bytes(range(9))))
        self.addCleanup(capture.release)
        self.assertTrue(capture.read()[0])
        with self.assertRaisesRegex(RuntimeError, 'incomplete raw frame'):
            capture.read()

    def test_nonzero_exit_after_complete_frame_is_a_failure(self):
        capture = self.capture(self.process(bytes(range(6)), result=7))
        self.addCleanup(capture.release)
        self.assertTrue(capture.read()[0])
        with self.assertRaisesRegex(RuntimeError, 'exit 7'):
            capture.read()

    def test_release_stops_a_process_and_joins_threads_even_with_full_queue(self):
        process = self.process(bytes(range(120)))
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 2), 0]
        capture = self.capture(process)
        capture.release()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertTrue(all(not thread.is_alive() for thread in capture._threads))
        self.assertTrue(process.stdout.closed)
        self.assertFalse(capture.isOpened())

    def test_cancel_interrupts_frame_wait_and_cleans_up(self):
        cancel = threading.Event()
        capture = self.capture(self.process(bytes(range(6))))
        capture.cancel_event = cancel
        capture.read()
        cancel.set()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            capture.read()
        self.assertTrue(all(not thread.is_alive() for thread in capture._threads))
        self.assertTrue(capture.process.stdout.closed)

    def test_stderr_is_bounded_even_without_newlines(self):
        process = self.process(bytes(range(6)))
        process.stderr = io.BytesIO(b'x' * 100000)
        capture = self.capture(process)
        capture.release()
        self.assertLessEqual(sum(map(len, capture._errors)), 32768)

    def test_cleanup_os_errors_do_not_mask_cancellation_or_skip_join(self):
        process = self.process(bytes(range(6)))
        capture = self.capture(process)
        for thread in capture._threads:
            thread.join(timeout=1)
        process.poll.return_value = None
        process.terminate.side_effect = OSError('already exited')
        process.wait.side_effect = [subprocess.TimeoutExpired('ffmpeg', 2), OSError('already exited')]
        process.kill.side_effect = OSError('already exited')
        real_stdout, real_stderr = process.stdout, process.stderr
        process.stdout = Mock()
        process.stdout.close.side_effect = OSError('already closed')
        process.stderr = Mock()
        process.stderr.close.side_effect = ValueError('already closed')
        cancel = threading.Event()
        capture.cancel_event = cancel
        cancel.set()
        with self.assertRaises(decoder.DecoderCancelled):
            capture.read()
        self.assertTrue(all(not thread.is_alive() for thread in capture._threads))
        process.kill.assert_called_once()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()
        real_stdout.close()
        real_stderr.close()

    def test_unkillable_child_never_closes_pipes_held_by_live_readers(self):
        capture = decoder.D3D11Capture.__new__(decoder.D3D11Capture)
        capture._closed = False
        capture._stopped = threading.Event()
        capture.decoder_info = {}
        capture.process = Mock()
        capture.process.poll.return_value = None
        capture.process.terminate.side_effect = OSError('OS refused termination')
        capture.process.kill.side_effect = OSError('OS refused kill')
        capture.process.wait.side_effect = subprocess.TimeoutExpired('ffmpeg', 2)
        capture.process.stdout.close.side_effect = AssertionError('Must not acquire an active pipe read lock')
        capture.process.stderr.close.side_effect = AssertionError('Must not acquire an active pipe read lock')
        readers = [Mock(), Mock()]
        for thread in readers:
            thread.ident = 1
            thread.is_alive.return_value = True
        capture._threads = readers
        capture.release()
        for thread in readers:
            thread.join.assert_called_once_with(timeout=2)
        capture.process.stdout.close.assert_not_called()
        capture.process.stderr.close.assert_not_called()
        self.assertFalse(capture.decoder_info['cleanup_complete'])
        self.assertIn('shutdown timeout', capture.decoder_info['cleanup_error'])


class CancellationRegressionTests(unittest.TestCase):
    def test_precancelled_open_does_not_create_any_decoder(self):
        cancel = threading.Event()
        cancel.set()
        factory = Mock()
        with self.assertRaises(decoder.DecoderCancelled):
            decoder.open_video_capture('in.mp4', 'ffmpeg', cancel_event=cancel,
                                       software_capture_factory=factory)
        factory.assert_not_called()

    def test_cancellation_is_honoured_before_returning_buffered_probe_frame(self):
        process = Mock()
        process.stdout = io.BytesIO(bytes(range(6)))
        process.stderr = io.BytesIO()
        process.poll.return_value = 0
        process.wait.return_value = 0
        cancel = threading.Event()
        capture = decoder.D3D11Capture('input.mp4', 'ffmpeg', METADATA, ADAPTER,
                                      cancel_event=cancel, popen=Mock(return_value=process))
        self.addCleanup(capture.release)
        cancel.set()
        with self.assertRaisesRegex(decoder.DecoderCancelled, 'cancelled'):
            capture.read()

    def test_stalled_real_child_is_terminated_on_startup_timeout(self):
        import sys
        import time
        children = []

        def spawn(_command, **options):
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], **options)
            children.append(child)
            return child

        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'timed out'):
            decoder.D3D11Capture('input.mp4', 'ffmpeg', METADATA, ADAPTER,
                                 popen=spawn, read_timeout=.15)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stdout.closed)
        self.assertTrue(children[0].stderr.closed)


if __name__ == '__main__':
    unittest.main()
