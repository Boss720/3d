"""Endpoint and pipeline integration checks; model downloads are never automatic."""
import base64
import io
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main
from video_encoder import EncoderProfile


class AppTests(unittest.TestCase):
    def setUp(self):
        self.folder = main.RUNTIME / 'app-tests' / uuid.uuid4().hex
        self.folder.mkdir(parents=True)
        self.outputs = patch.object(main, 'OUTPUTS', self.folder)
        self.outputs.start()
        self.client = TestClient(main.app)
        self.frame = np.zeros((46, 66, 3), np.uint8)
        self.frame[:, :, 0] = np.linspace(0, 255, 66, dtype=np.uint8)
        self.frame[12:35, 20:45] = (80, 150, 230)
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()

    def tearDown(self):
        self.outputs.stop()

    def submit(self, **settings):
        data = {'depth_mode': 'balanced', 'resolution': 'source', **settings}
        return self.client.post('/api/jobs', data=data, files={'file': ('fixture.png', self.png, 'image/png')})

    def wait_job(self, job_id):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = next(j for j in self.client.get('/api/jobs').json() if j['id'] == job_id)
            if job['state'] in {'completed', 'failed', 'canceled'}:
                return job
            time.sleep(.02)
        self.fail('Local job did not finish')

    def test_preview_matches_export_pixels_for_all_layouts(self):
        for layout in sorted(main.LAYOUTS):
            with self.subTest(layout=layout):
                data = {'depth_mode': 'balanced', 'layout': layout, 'strength': '6', 'invert_depth': 'true'}
                preview = self.client.post('/api/preview', data=data, files={'file': ('frame.png', self.png, 'image/png')})
                self.assertEqual(preview.status_code, 200, preview.text[:300])
                response = self.submit(layout=layout, strength='6', invert_depth='true')
                self.assertEqual(response.status_code, 200, response.text)
                job = self.wait_job(response.json()['id'])
                self.assertEqual(job['state'], 'completed', job)
                download = self.client.get(f"/api/jobs/{job['id']}/download")
                exported = cv2.imdecode(np.frombuffer(download.content, np.uint8), cv2.IMREAD_COLOR)
                preview_bytes = base64.b64decode(preview.json()['image'].split(',', 1)[1])
                shown = cv2.imdecode(np.frombuffer(preview_bytes, np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(exported, shown)

    def test_unavailable_named_model_rejected_before_queueing(self):
        with patch.object(main.DEPTH_ENGINE, 'validate', side_effect=RuntimeError('Model not installed')):
            response = self.submit(depth_mode='depth-anything-v2-base')
        self.assertEqual(response.status_code, 409)
        self.assertIn('not installed', response.json()['detail'])

    def test_invalid_numeric_controls_rejected(self):
        for settings in ({'strength': '-1'}, {'strength': 'nan'}, {'convergence': '1.1'}, {'temporal_smoothing': '.95'}):
            with self.subTest(settings=settings):
                self.assertEqual(self.submit(**settings).status_code, 400)

    def test_execution_mode_rejected_if_unknown_or_gpu_with_non_neural_engine(self):
        for mode in ('invalid', 'gpu'):
            response = self.submit(execution_mode=mode)
            self.assertEqual(response.status_code, 400, response.text)

    def video_job(self, **changes):
        job = main.Job(id=str(uuid.uuid4()), filename='fixture.mp4', input_path=self.folder/'fixture.mp4',
                       media_type='video', layout='SBS', strength=5, convergence=.5, depth_mode='balanced',
                       input_size=518, resolution='source', quality='fast', invert_depth=False)
        for name, value in changes.items():
            setattr(job, name, value)
        return job

    def test_job_response_includes_null_video_metrics_for_still_images(self):
        response = self.submit()
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()
        self.assertEqual(job['eta_status'], 'unavailable')
        self.assertIn('elapsed_seconds', job)
        self.assertIn('processing_fps', job)
        self.assertIn('eta_seconds', job)
        self.assertIn('frames_processed', job)
        self.assertIn('total_frames', job)
        self.assertIsNone(job['elapsed_seconds'])
        self.assertIsNone(job['processing_fps'])
        self.assertIsNone(job['eta_seconds'])
        self.assertIsNone(job['frames_processed'])
        self.assertIsNone(job['total_frames'])

    def test_video_eta_is_available_after_speed_estimation_window(self):
        job = self.video_job(state='processing', processing_started_monotonic=98.0,
                             processing_finished_monotonic=None, video_started_monotonic=100.0,
                             frames_processed=20, total_frames=100)
        with patch.object(main.time, 'monotonic', return_value=110.0):
            public = job.public()
        self.assertEqual(public['frames_processed'], 20)
        self.assertEqual(public['total_frames'], 100)
        self.assertEqual(public['eta_status'], 'available')
        self.assertAlmostEqual(public['elapsed_seconds'], 12.0)
        self.assertAlmostEqual(public['processing_fps'], 2.0)
        self.assertAlmostEqual(public['eta_seconds'], 40.0)

    def test_video_eta_is_unavailable_for_unknown_total_and_initial_window(self):
        cases = (
            self.video_job(state='processing', processing_started_monotonic=98.0,
                           processing_finished_monotonic=None, video_started_monotonic=100.0,
                           frames_processed=20, total_frames=None),
            self.video_job(state='processing', processing_started_monotonic=98.0,
                           processing_finished_monotonic=None, video_started_monotonic=100.0,
                           frames_processed=4, total_frames=100),
        )
        with patch.object(main.time, 'monotonic', return_value=110.0):
            for job in cases:
                with self.subTest(total_frames=job.total_frames, frames_processed=job.frames_processed):
                    public = job.public()
                    self.assertIsNone(public['eta_seconds'])
                    expected_status = 'unavailable' if job.total_frames is None else 'estimating'
                    self.assertEqual(public['eta_status'], expected_status)

    def test_completed_video_eta_is_zero_and_elapsed_is_frozen(self):
        job = self.video_job(state='completed', processing_started_monotonic=98.0,
                             processing_finished_monotonic=108.0, video_started_monotonic=100.0,
                             frames_processed=100, total_frames=100)
        with patch.object(main.time, 'monotonic', return_value=120.0):
            public = job.public()
        self.assertEqual(public['eta_status'], 'complete')
        self.assertEqual(public['eta_seconds'], 0.0)
        self.assertAlmostEqual(public['elapsed_seconds'], 10.0)
        self.assertAlmostEqual(public['processing_fps'], 12.5)

        unknown_total = self.video_job(state='completed', processing_started_monotonic=98.0,
                                       processing_finished_monotonic=108.0, video_started_monotonic=100.0,
                                       frames_processed=100, total_frames=None)
        with patch.object(main.time, 'monotonic', return_value=120.0):
            public = unknown_total.public()
        self.assertEqual(public['eta_status'], 'complete')
        self.assertIsNone(public['eta_seconds'])

    def test_cpu_approximation_preview_reports_actual_device(self):
        response = self.client.post('/api/preview', data={'depth_mode':'balanced', 'execution_mode':'hybrid'},
                                    files={'file':('frame.png',self.png,'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        execution = response.json()['execution']
        self.assertEqual(execution['requested_mode'], 'hybrid')
        self.assertEqual(execution['device'], 'CPU')
        self.assertFalse(execution['gpu_verified'])
        self.assertIn('CPU', execution['fallback_reason'])

    def test_corrupt_preview_has_useful_error(self):
        response = self.client.post('/api/preview', data={'depth_mode': 'balanced'}, files={'file': ('bad.png', b'bad', 'image/png')})
        self.assertEqual(response.status_code, 400)
        self.assertIn('decoded image', response.json()['detail'])

    def test_video_cancellation_during_inference_does_not_become_failed(self):
        job = main.Job(id=str(uuid.uuid4()), filename='cancel.mp4', input_path=self.folder/'cancel.mp4',
                       media_type='video', layout='SBS', strength=5, convergence=.5, depth_mode='balanced',
                       input_size=518, resolution='source', quality='fast', invert_depth=False)
        class Capture:
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FRAME_WIDTH:66, cv2.CAP_PROP_FRAME_HEIGHT:46,
                        cv2.CAP_PROP_FPS:24, cv2.CAP_PROP_FRAME_COUNT:1}.get(prop, 0)
            def read(inner): return True, self.frame.copy()
            def release(self): pass
        class Pipe:
            def write(self, data): raise AssertionError('Canceled frame reached the encoder')
        class Process:
            stdin = Pipe()
            stderr = None
            def poll(self): return 0
        def infer(*args):
            job.cancel_event.set()
            return np.full(self.frame.shape[:2], .5, np.float32)
        cpu_profile = EncoderProfile(
            quality='fast', codec='libx264',
            video_args=('libx264',), software_fallback_args=('libx264',),
            hardware_available=False,
        )
        with patch.object(main, 'FFMPEG_PATH', str(Path(main.__file__))), patch.object(main.cv2, 'VideoCapture', return_value=Capture()), patch.object(main, 'ffprobe_fps', return_value=24), patch.object(main.subprocess, 'Popen', return_value=Process()), patch.object(main, 'select_encoder_profile', return_value=cpu_profile), patch.object(main, 'estimate_depth', side_effect=infer):
            with self.assertRaises(main.JobCancelled):
                main.run_video(job, None)

    def test_video_export_reports_nvenc_and_retries_with_cpu_when_needed(self):
        profile = EncoderProfile(
            quality='fast', codec='h264_nvenc',
            video_args=('-c:v', 'h264_nvenc', '-preset', 'p1'),
            software_fallback_args=('-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22'),
            hardware_available=True,
        )

        class Capture:
            def __init__(inner): inner.sent = False
            def isOpened(inner): return True
            def get(inner, prop):
                return {cv2.CAP_PROP_FRAME_WIDTH:66, cv2.CAP_PROP_FRAME_HEIGHT:46,
                        cv2.CAP_PROP_FPS:24, cv2.CAP_PROP_FRAME_COUNT:1}.get(prop, 0)
            def read(inner):
                if inner.sent: return False, None
                inner.sent = True
                return True, self.frame.copy()
            def release(inner): pass

        class Pipe:
            def write(inner, data): return len(data)
            def close(inner): pass

        class Process:
            def __init__(inner, command, return_code):
                inner.command = command
                inner.return_code = return_code
                inner.finished = False
                inner.stdin = Pipe()
                inner.stderr = io.BytesIO(b"NVENC init failed\n" if return_code else b"")
            def poll(inner): return inner.return_code if inner.finished else None
            def wait(inner):
                inner.finished = True
                if inner.return_code == 0:
                    Path(inner.command[-1]).write_bytes(b"test output")
                return inner.return_code
            def terminate(inner): inner.finished = True
            def kill(inner): inner.finished = True

        commands = []
        return_codes = iter((0, 1, 0))
        def popen(command, **_kwargs):
            commands.append(command)
            return Process(command, next(return_codes))

        job = self.video_job()
        fallback_job = self.video_job()
        with patch.object(main, 'FFMPEG_PATH', str(Path(main.__file__))), \
             patch.dict(main.JOBS, {job.id: job, fallback_job.id: fallback_job}), \
             patch.object(main.cv2, 'VideoCapture', side_effect=lambda _path: Capture()), \
             patch.object(main, 'ffprobe_fps', return_value=24), \
             patch.object(main.subprocess, 'Popen', side_effect=popen), \
             patch.object(main, 'estimate_depth', return_value=np.full((46, 66), .5, np.float32)), \
             patch.object(main, 'select_encoder_profile', return_value=profile):
            target = main.run_video(job, None)
            fallback_target = main.run_video(fallback_job, None)

        self.assertTrue(target.is_file())
        self.assertTrue(fallback_target.is_file())
        self.assertEqual(len(commands), 3)
        self.assertIn('h264_nvenc', commands[0])
        self.assertIn('h264_nvenc', commands[1])
        self.assertIn('libx264', commands[2])
        self.assertEqual(job.encoder, 'NVIDIA NVENC')
        self.assertEqual(fallback_job.encoder, 'CPU libx264 (NVENC fallback)')
        self.assertEqual(fallback_job.frames_processed, 1)
        self.assertEqual(fallback_job.total_frames, 1)


if __name__ == '__main__':
    unittest.main()
