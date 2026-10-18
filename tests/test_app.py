"""Endpoint and pipeline integration checks; model downloads are never automatic."""
import base64
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main


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
        with patch.object(main, 'FFMPEG_PATH', str(Path(main.__file__))), patch.object(main.cv2, 'VideoCapture', return_value=Capture()), patch.object(main, 'ffprobe_fps', return_value=24), patch.object(main.subprocess, 'Popen', return_value=Process()), patch.object(main, 'estimate_depth', side_effect=infer):
            with self.assertRaises(main.JobCancelled):
                main.run_video(job, None)


if __name__ == '__main__':
    unittest.main()
