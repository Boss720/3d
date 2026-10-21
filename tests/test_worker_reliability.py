"""Per-job errors must not strand later work or hide the conversion outcome."""
import os
import subprocess
import sys
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import main
from job_queue import EditableJobQueue
from video_decoder import D3D11Capture


class WorkerReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'worker-reliability-tests' / uuid.uuid4().hex
        self.uploads, self.outputs = self.root / 'uploads', self.root / 'outputs'
        self.uploads.mkdir(parents=True)
        self.outputs.mkdir()
        for name, value in (('UPLOADS', self.uploads), ('OUTPUTS', self.outputs),
                            ('JOBS', {}), ('JOB_QUEUE', EditableJobQueue()),
                            ('WORKER_STOP', threading.Event()), ('HISTORY', None)):
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.addCleanup(self.quarantine)

    def quarantine(self):
        target = main.RUNTIME / '_quarantine' / 'worker-reliability-tests' / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, target)

    def job(self):
        job_id = str(uuid.uuid4())
        source = self.uploads / f'{job_id}.png'
        cv2.imwrite(str(source), np.full((12, 16, 3), 100, np.uint8))
        job = main.Job(id=job_id, filename='fixture.png', input_path=source,
                       media_type='image', layout='SBS', strength=5, convergence=.5,
                       depth_mode='balanced', input_size=256, resolution='source',
                       quality='fast', invert_depth=False)
        main.JOBS[job.id] = job
        return job

    def run_jobs(self, jobs, processor=None):
        actual = processor or main.process_job
        for job in jobs:
            main.JOB_QUEUE.put(job.id)
        def process(job_id):
            try:
                return actual(job_id)
            finally:
                if job_id == jobs[-1].id:
                    main.WORKER_STOP.set()
        with patch.object(main, 'process_job', side_effect=process):
            main.worker_loop()
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 0)
        self.assertEqual(main.JOB_QUEUE.qsize(), 0)
        self.assertEqual(jobs[-1].state, 'completed')

    def deny_cleanup(self, *paths):
        original = Path.unlink
        def unlink(path, *args, **kwargs):
            if path in paths:
                raise PermissionError('simulated Windows file lock')
            return original(path, *args, **kwargs)
        return patch.object(Path, 'unlink', unlink)

    def test_canceled_upload_cleanup_error_does_not_stop_the_next_job(self):
        first, second = self.job(), self.job()
        first.cancel_event.set()
        with self.deny_cleanup(first.input_path):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'canceled')
        self.assertTrue(first.input_path.exists())
        self.assertTrue(first.public()['cleanup_warning'])

    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
    def test_real_windows_open_upload_does_not_stop_the_next_job(self):
        first, second = self.job(), self.job()
        first.cancel_event.set()
        with first.input_path.open('rb'):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'canceled')
        self.assertTrue(first.input_path.exists())
        self.assertTrue(first.public()['cleanup_warning'])
        self.assertIn(first.input_path, first.cleanup_pending)
        main.cleanup_job_file(first, first.input_path, 'upload')
        self.assertFalse(first.input_path.exists())
        self.assertIsNone(first.public()['cleanup_warning'])
        self.assertEqual(first.cleanup_pending, {})

    def test_completed_upload_cleanup_error_preserves_success_and_download(self):
        first, second = self.job(), self.job()
        with self.deny_cleanup(first.input_path):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'completed')
        self.assertTrue(first.output_path.is_file())
        self.assertTrue(first.public()['cleanup_warning'])
        self.assertEqual(main.download_job(first.id).path, first.output_path)

    def test_failed_partial_output_cleanup_preserves_the_original_error(self):
        first, second = self.job(), self.job()
        target = main.output_path_for(first)
        real_run = main.run_image
        def convert(job, net):
            if job is first:
                target.write_bytes(b'partial conversion')
                raise RuntimeError('original conversion error')
            return real_run(job, net)
        with self.deny_cleanup(target), patch.object(main, 'run_image', side_effect=convert):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'failed')
        self.assertEqual(first.error, 'original conversion error')
        self.assertTrue(target.exists())
        self.assertTrue(first.public()['cleanup_warning'])
        self.assertIsNone(first.output_path)

    def test_canceled_partial_output_cleanup_does_not_become_failed(self):
        first, second = self.job(), self.job()
        target = main.output_path_for(first)
        real_run = main.run_image
        def convert(job, net):
            if job is first:
                target.write_bytes(b'canceled conversion')
                job.cancel_event.set()
                return target
            return real_run(job, net)
        with self.deny_cleanup(target), patch.object(main, 'run_image', side_effect=convert):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'canceled')
        self.assertIsNone(first.error)
        self.assertTrue(first.public()['cleanup_warning'])

    def test_preview_proxy_cleanup_error_does_not_skip_upload_cleanup(self):
        first, second = self.job(), self.job()
        proxy = self.root / 'retained-preview-proxy.mp4'
        proxy.write_bytes(b'preview')
        first.preview_source_path = proxy
        with self.deny_cleanup(proxy):
            self.run_jobs([first, second])
        self.assertEqual(first.state, 'completed')
        self.assertFalse(first.input_path.exists())
        self.assertTrue(proxy.exists())
        self.assertTrue(first.public()['cleanup_warning'])

    def test_unexpected_job_exception_is_terminal_and_worker_keeps_running(self):
        first, second = self.job(), self.job()
        original = main.process_job
        def unexpected(job_id):
            if job_id == first.id:
                raise RuntimeError('unexpected per-job error')
            return original(job_id)
        self.run_jobs([first, second], unexpected)
        self.assertEqual(first.state, 'failed')
        self.assertIn('unexpected per-job error', first.error)

    def test_midstream_stalled_decoder_fails_one_job_then_worker_completes_next_job(self):
        ffmpeg = main.FFMPEG_PATH
        if not ffmpeg or not Path(ffmpeg).is_file():
            self.skipTest('The worker-level decoder recovery regression requires FFmpeg configuration.')

        first, second = self.job(), self.job()
        first.media_type = 'video'
        first.filename = 'stalled-source.mp4'
        source = self.uploads / f'{first.id}.mp4'
        first.input_path.unlink()
        first.input_path = source
        generated = subprocess.run(
            [ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-f', 'lavfi', '-i',
             'testsrc2=size=16x16:rate=15:duration=0.2', '-frames:v', '3', '-c:v', 'libx264',
             '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(source)],
            capture_output=True, text=True, timeout=15, check=False)
        self.assertEqual(generated.returncode, 0, generated.stderr)
        children = []
        metadata = {'width': 16, 'height': 16, 'fps': 15., 'fps_ratio': '15',
                    'total_frames': 3, 'video_start_seconds': 0., 'format_start_seconds': 0.}

        def spawn_stalled(_command, **options):
            child = subprocess.Popen(
                [sys.executable, '-c',
                 'import sys,time;sys.stdout.buffer.write(bytes([80])*768);sys.stdout.flush();time.sleep(60)'],
                **options)
            children.append(child)
            return child

        encoder_children = []
        real_popen = subprocess.Popen

        def track_encoder(command, *args, **options):
            child = real_popen(command, *args, **options)
            if str(command[0]).casefold() == str(ffmpeg).casefold():
                encoder_children.append(child)
            return child

        original_inspect = main.inspect_media
        def inspect(source, media_type, *args, **kwargs):
            if media_type == 'video':
                return {'source_width': 16, 'source_height': 16, 'width': 16, 'height': 16,
                        'total_frames': 3, 'fps': 15., 'fps_provisional': False,
                        'media_type': 'video'}
            return original_inspect(source, media_type, *args, **kwargs)

        def open_stalled(source, binary, **kwargs):
            return D3D11Capture(source, binary, metadata, None,
                                cancel_event=kwargs.get('cancel_event'),
                                popen=spawn_stalled, read_timeout=.15)

        encoder = SimpleNamespace(codec='libx264',
                                  video_args=('-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22'),
                                  software_fallback_args=('-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22'))
        with patch.object(main, 'inspect_media', side_effect=inspect), \
             patch.object(main, 'open_video_capture', side_effect=open_stalled), \
             patch.object(main, 'select_encoder_profile', return_value=encoder), \
             patch.object(main, 'FFMPEG_PATH', ffmpeg), \
             patch.object(subprocess, 'Popen', side_effect=track_encoder):
            self.run_jobs([first, second])

        self.assertEqual(first.state, 'failed')
        self.assertIn('timed out', first.error.lower())
        self.assertEqual(first.frames_processed, 1)
        self.assertEqual(second.state, 'completed')
        self.assertTrue(second.output_path.is_file())
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].poll())
        self.assertTrue(children[0].stdout.closed)
        self.assertTrue(children[0].stderr.closed)
        self.assertEqual(len(encoder_children), 1)
        self.assertIsNotNone(encoder_children[0].poll())
        self.assertTrue(encoder_children[0].stdin.closed)
        self.assertTrue(encoder_children[0].stderr.closed)
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 0)


if __name__ == '__main__':
    unittest.main()
