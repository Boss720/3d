import unittest
import uuid
import threading
import base64
from collections import deque
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main
from telemetry import StageTimings
from video_pipeline import run_pipeline


class RoadmapAppTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.frame = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()

    def preview(self, **settings):
        return self.client.post('/api/preview', data={'depth_mode': 'balanced', **settings},
                                files={'file': ('frame.png', self.png, 'image/png')})

    def test_presentation_edits_reuse_raw_depth_and_equal_uncached_pixels(self):
        main.PREVIEW_CACHE.clear()
        original = main.estimate_depth
        with patch.object(main, 'estimate_depth', wraps=original) as infer:
            first = self.preview(strength='2')
            self.assertEqual(first.status_code, 200, first.text[:200])
            changed = self.preview(strength='8', convergence='.3', layout='TB',
                                   invert_depth='true', edge_refinement='false')
            self.assertEqual(changed.status_code, 200)
            self.assertEqual(infer.call_count, 1)
            self.assertTrue(changed.json()['cache']['hit'])
            main.PREVIEW_CACHE.clear()
            uncached = self.preview(strength='8', convergence='.3', layout='TB',
                                    invert_depth='true', edge_refinement='false')
            self.assertEqual(changed.json()['image'], uncached.json()['image'])
            self.assertEqual(changed.json()['depth'], uncached.json()['depth'])
            self.preview(input_size='256')
            self.assertEqual(infer.call_count, 3)

    def test_changed_frame_and_device_invalidate_cache(self):
        main.PREVIEW_CACHE.clear()
        with patch.object(main, 'estimate_depth', wraps=main.estimate_depth) as infer:
            self.preview()
            self.preview(execution_mode='cpu')
            self.png = cv2.imencode('.png', self.frame ^ 127)[1].tobytes()
            self.preview(execution_mode='cpu')
            self.assertEqual(infer.call_count, 3)

    def test_comparison_source_is_the_exact_decoded_inference_frame(self):
        response = self.preview()
        self.assertEqual(response.status_code, 200)
        result = response.json()
        source = cv2.imdecode(np.frombuffer(base64.b64decode(result['source'].split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(source, self.frame)
        self.assertEqual([result['source_width'], result['source_height']], [64, 48])

    def test_gpu_reset_clears_depth_cache(self):
        main.PREVIEW_CACHE.clear()
        self.preview()
        with patch.object(main.DEPTH_ENGINE, 'reset_gpu_failures', create=True) as reset:
            response = self.client.post('/api/runtime/gpu/reset')
        self.assertEqual(response.status_code, 200)
        reset.assert_called_once()
        self.assertEqual(main.PREVIEW_CACHE.info()['entries'], 0)

    def test_export_allows_cache_hit_but_rejects_new_inference(self):
        main.PREVIEW_CACHE.clear()
        model = 'depth-anything-v2-small'
        metadata = {'active_provider': 'CPUExecutionProvider', 'fallback_reason': None}
        with patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                return_value=(np.zeros(self.frame.shape[:2], np.float32), metadata)):
            self.assertEqual(self.preview(depth_mode=model).status_code, 200)
        job = main.Job(id='busy', filename='test.mp4', input_path=Path('unused'),
                       media_type='video', layout='SBS', strength=5, convergence=.5,
                       depth_mode=model, input_size=256, resolution='source',
                       quality='fast', invert_depth=False, state='processing')
        with patch.dict(main.JOBS, {'busy': job}), patch.object(main, 'estimate_depth') as infer, patch.object(main.DEPTH_ENGINE, 'validate'):
            self.assertEqual(self.preview(depth_mode=model, strength='2').status_code, 200)
            blocked = self.preview(depth_mode=model, input_size='256')
            self.assertEqual(blocked.status_code, 409)
            infer.assert_not_called()

    def test_stage_timings_record_errors_and_do_not_sum_overlap(self):
        timing = StageTimings()
        with patch('telemetry.time.perf_counter', side_effect=[0, 1, 3, 5]):
            with timing.measure('outer'):
                with self.assertRaises(ValueError), timing.measure('inner'):
                    raise ValueError('failure')
        result = timing.snapshot()
        self.assertEqual(result['stages']['outer'], {'seconds': 5, 'calls': 1})
        self.assertEqual(result['stages']['inner'], {'seconds': 2, 'calls': 1})
        self.assertIsNone(result['queue_stalls'])

    def test_pipeline_queue_waits_are_separate_from_service_timings(self):
        timing = StageTimings()
        outputs = []
        run_pipeline(range(6), lambda value: value * 2,
                     lambda frame, result: outputs.append(result), threading.Event(),
                     queue_size=1, timings=timing)
        self.assertEqual(outputs, [0, 2, 4, 6, 8, 10])
        result = timing.snapshot()
        self.assertEqual(result['stages'], {})
        for name in ('queue_decode_put', 'queue_decode_get', 'queue_render_put', 'queue_render_get'):
            self.assertEqual(result['queue_stalls'][name]['calls'], 7)
            self.assertGreaterEqual(result['queue_stalls'][name]['seconds'], 0)

    def test_recent_eta_accounts_for_stall_and_finalization(self):
        job = main.Job(id='eta', filename='test.mp4', input_path=Path('unused'),
                       media_type='video', layout='SBS', strength=5, convergence=.5,
                       depth_mode='balanced', input_size=256, resolution='source',
                       quality='fast', invert_depth=False, state='processing',
                       processing_started_monotonic=0, video_started_monotonic=0,
                       frames_processed=50, total_frames=100, phase='conversion',
                       rate_samples=deque([(90, 40), (92, 42), (94, 44),
                                           (96, 46), (98, 48), (100, 50)], maxlen=32))
        with patch.object(main.time, 'monotonic', return_value=100):
            current = job.public()
        self.assertEqual(current['processing_fps'], 1)
        self.assertEqual(current['eta_seconds'], 50)
        with patch.object(main.time, 'monotonic', return_value=110):
            stalled = job.public()
        self.assertEqual(stalled['eta_seconds'], 100)
        self.assertEqual(stalled['eta_range_seconds'], [50, 100])
        job.phase = 'finalization'
        with patch.object(main.time, 'monotonic', return_value=110):
            finalizing = job.public()
        self.assertEqual(finalizing['eta_status'], 'finalizing')
        self.assertIsNone(finalizing['eta_seconds'])

    def test_preflight_rejects_corrupt_media_before_loading_model(self):
        folder = main.RUNTIME / 'roadmap-tests' / uuid.uuid4().hex
        folder.mkdir(parents=True)
        try:
            source = Path(folder) / 'bad.png'
            source.write_bytes(b'not an image')
            job = main.Job(id='bad-preflight', filename='bad.png', input_path=source,
                           media_type='image', layout='SBS', strength=5, convergence=.5,
                           depth_mode='precision', input_size=256, resolution='source',
                           quality='fast', invert_depth=False)
            with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'load_depth_model') as load:
                main.process_job(job.id)
            load.assert_not_called()
            self.assertEqual(job.state, 'failed')
            self.assertIn('decode', job.error)
            self.assertFalse(source.exists())
        finally:
            folder.rmdir()
