"""Full-detail source policy and preview integration; no real model sessions."""
import asyncio
import base64
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main
from preview_cache import PreviewDepthCache
from preview_policy import (PREVIEW_UPLOAD_BYTES, PreviewPolicyError, preview_source_policy,
                            validate_preview_upload_size)


MODEL = 'depth-anything-v2-large'


def decode(value):
    return cv2.imdecode(np.frombuffer(base64.b64decode(value.split(',', 1)[1]), np.uint8), cv2.IMREAD_COLOR)


class FullPreviewTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)
        self.cache = PreviewDepthCache()
        self.cache_patch = patch.object(main, 'PREVIEW_CACHE', self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.frame = np.arange(640 * 1200 * 3, dtype=np.uint8).reshape(640, 1200, 3)
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()

    def preview(self, **settings):
        return self.client.post('/api/preview', data={'depth_mode': 'balanced', 'edge_refinement': 'false', **settings},
                                files={'file': ('frame.png', self.png, 'image/png')})

    def test_default_proxy_preserves_resize_pixels_and_reports_both_dimensions(self):
        response = self.preview(strength='0')
        self.assertEqual(response.status_code, 200, response.text[:200])
        result = response.json()
        self.assertEqual(result['preview_detail'], 'proxy')
        self.assertTrue(result['is_proxy'])
        self.assertEqual([result['original_width'], result['original_height']], [1200, 640])
        self.assertEqual([result['source_width'], result['source_height']], [960, 512])
        expected = cv2.resize(self.frame, (960, 512), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(decode(result['source']), expected)
        np.testing.assert_array_equal(decode(result['image']), np.concatenate((expected, expected), axis=1))

    def test_full_retains_exact_source_and_output_pixels(self):
        response = self.preview(preview_detail='full', strength='0')
        self.assertEqual(response.status_code, 200, response.text[:200])
        result = response.json()
        self.assertEqual(result['preview_detail'], 'full')
        self.assertFalse(result['is_proxy'])
        self.assertEqual([result['original_width'], result['original_height']], [1200, 640])
        self.assertEqual([result['source_width'], result['source_height']], [1200, 640])
        self.assertEqual([result['width'], result['height']], [2400, 640])
        np.testing.assert_array_equal(decode(result['source']), self.frame)
        np.testing.assert_array_equal(decode(result['image']), np.concatenate((self.frame, self.frame), axis=1))
        self.assertEqual(decode(result['depth']).shape, self.frame.shape)

    def test_full_source_dimensions_do_not_replace_capped_neural_execution_metadata(self):
        metadata = {'requested_input_width': 966, 'requested_input_height': 518,
                    'input_width': 518, 'input_height': 280, 'input_size_limit': 518,
                    'active_provider': 'DmlExecutionProvider', 'fallback_reason': None,
                    'inference_succeeded': True}
        with patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                return_value=(np.full(self.frame.shape[:2], .5, np.float32), metadata)) as predict:
            response = self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH')
            self.assertEqual(response.status_code, 200, response.text[:200])
            result = response.json()
        self.assertEqual(result['execution'], metadata)
        self.assertEqual([result['source_width'], result['source_height']], [1200, 640])
        self.assertEqual(predict.call_args.args[0].shape, self.frame.shape)
        self.assertEqual(predict.call_args.args[2], 518)

    def test_invalid_detail_rejected_before_decode_or_inference(self):
        with patch.object(main.cv2, 'imdecode', wraps=cv2.imdecode) as read, \
                patch.object(main, 'estimate_depth', wraps=main.estimate_depth) as infer:
            response = self.preview(preview_detail='native')
        self.assertEqual(response.status_code, 400, response.text[:200])
        self.assertIn('proxy', response.json()['detail'].lower())
        read.assert_not_called()
        infer.assert_not_called()

    def test_full_dimension_and_pixel_limits_reject_before_prediction(self):
        for shape in ((1, 8193, 3), (3000, 4001, 3)):
            with self.subTest(shape=shape), patch.object(main.cv2, 'imdecode', return_value=np.zeros(shape, np.uint8)), \
                    patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                        side_effect=lambda frame, *args, **kwargs: (np.zeros(frame.shape[:2], np.float32), {})) as predict:
                response = self.preview(preview_detail='full', depth_mode=MODEL)
                self.assertEqual(response.status_code, 413, response.text[:200])
                self.assertIn('proxy', response.json()['detail'].lower())
                predict.assert_not_called()

    def test_small_proxy_and_full_have_separate_cache_policy_identity(self):
        frame = self.frame[:48, :64]
        self.png = cv2.imencode('.png', frame)[1].tobytes()
        with patch.object(main, 'estimate_depth', wraps=main.estimate_depth) as infer:
            proxy = self.preview().json()
            full = self.preview(preview_detail='full').json()
            self.assertFalse(proxy['cache']['hit'])
            self.assertFalse(full['cache']['hit'])
            self.assertTrue(self.preview(preview_detail='full', strength='1').json()['cache']['hit'])
            self.assertTrue(self.preview().json()['cache']['hit'])
        self.assertEqual(infer.call_count, 2)

    def test_full_cache_owns_depth_and_execution_and_evicts_by_buffer_budget(self):
        self.frame = self.frame[:8, :10].copy()
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()
        self.cache.max_bytes = self.frame.shape[0] * self.frame.shape[1] * 4 * 2
        raw = np.full(self.frame.shape[:2], .25, np.float32)
        metadata = {'active_provider': 'CPUExecutionProvider', 'fallback_reason': None, 'providers': ['CPU']}
        with patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                return_value=(raw, metadata)) as predict:
            first = self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH').json()
            raw[:] = .8
            metadata['providers'].append('mutated')
            cached = self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH').json()
            self.assertTrue(cached['cache']['hit'])
            self.assertEqual(first['depth'], cached['depth'])
            self.assertEqual(cached['execution']['providers'], ['CPU'])
            for xor in (1, 2):
                self.png = cv2.imencode('.png', self.frame ^ xor)[1].tobytes()
                self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH')
            self.png = cv2.imencode('.png', self.frame)[1].tobytes()
            evicted = self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH').json()
        self.assertFalse(evicted['cache']['hit'])
        self.assertEqual(predict.call_count, 4)
        self.assertEqual(self.cache.info()['depth_bytes'], self.cache.max_bytes)

    def test_full_depth_larger_than_32_mib_is_not_retained(self):
        frame = np.zeros((3000, 3000, 3), np.uint8)
        raw = np.zeros(frame.shape[:2], np.float32)
        with patch.object(main.cv2, 'imdecode', return_value=frame), \
                patch.object(main.DEPTH_ENGINE, 'validate'), \
                patch.object(main.DEPTH_ENGINE, 'predict_with_info', return_value=(raw, {'fallback_reason': None})) as predict, \
                patch.object(main, 'refine_depth', side_effect=lambda frame, depth, enabled: depth), \
                patch.object(main, 'render_output', return_value=frame[:1, :1]), \
                patch.object(main, 'png_data_url', return_value='mock-encoded'):
            for _ in range(2):
                response = self.preview(preview_detail='full', depth_mode=MODEL, layout='DEPTH')
                self.assertEqual(response.status_code, 200, response.text[:200])
                self.assertFalse(response.json()['cache']['hit'])
                self.assertEqual([response.json()['source_width'], response.json()['source_height']], [3000, 3000])
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(self.cache.info()['entries'], 0)

    def test_active_neural_export_allows_full_cache_hit_and_rejects_misses(self):
        self.frame = self.frame[:48, :64]
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()
        with patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                return_value=(np.zeros(self.frame.shape[:2], np.float32), {'fallback_reason': None})):
            self.assertEqual(self.preview(preview_detail='full', depth_mode=MODEL).status_code, 200)
        job = main.Job(id='full-busy', filename='test.mp4', input_path=Path('unused'),
                       media_type='video', layout='SBS', strength=5, convergence=.5,
                       depth_mode=MODEL, input_size=256, resolution='source',
                       quality='fast', invert_depth=False, state='processing')
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main.DEPTH_ENGINE, 'validate'), \
                patch.object(main.DEPTH_ENGINE, 'predict_with_info') as predict:
            hit = self.preview(preview_detail='full', depth_mode=MODEL, strength='2')
            self.assertEqual(hit.status_code, 200)
            self.assertTrue(hit.json()['cache']['hit'])
            self.assertEqual(self.preview(preview_detail='full', depth_mode=MODEL, input_size='256').status_code, 409)
            self.assertEqual(self.preview(depth_mode=MODEL).status_code, 409)
            predict.assert_not_called()

    def test_full_transient_neural_fallback_is_not_cached(self):
        self.frame = self.frame[:48, :64]
        self.png = cv2.imencode('.png', self.frame)[1].tobytes()
        metadata = {'active_provider': 'CPUExecutionProvider', 'fallback_reason': 'GPU inference failed'}
        with patch.object(main.DEPTH_ENGINE, 'validate'), patch.object(main.DEPTH_ENGINE, 'predict_with_info',
                return_value=(np.zeros(self.frame.shape[:2], np.float32), metadata)) as predict:
            for _ in range(2):
                response = self.preview(preview_detail='full', depth_mode=MODEL)
                self.assertEqual(response.status_code, 200, response.text[:200])
                self.assertEqual(response.json()['execution'], metadata)
                self.assertFalse(response.json()['cache']['hit'])
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(self.cache.info()['entries'], 0)

    def test_legacy_process_preview_arguments_retain_proxy_default(self):
        result = main.process_preview(self.png, 'balanced', 518, 'DEPTH', 5, .5, False, False)
        self.assertEqual(result['preview_detail'], 'proxy')
        self.assertEqual([result['source_width'], result['source_height']], [960, 512])

    def test_upload_cap_is_enforced_and_file_closed_before_decode_or_prediction(self):
        file = unittest.mock.Mock()
        file.read = AsyncMock(return_value=b'x' * (PREVIEW_UPLOAD_BYTES + 1))
        file.close = AsyncMock()
        with patch.object(main.cv2, 'imdecode') as read, patch.object(main.DEPTH_ENGINE, 'predict_with_info') as predict:
            with self.assertRaises(main.HTTPException) as caught:
                asyncio.run(main.preview(file=file, depth_mode='balanced', input_size=518,
                                         layout='DEPTH', strength=5, convergence=.5,
                                         invert_depth=False, edge_refinement=False,
                                         execution_mode='cpu', preview_detail='full'))
        self.assertEqual(caught.exception.status_code, 413)
        self.assertIn('proxy', caught.exception.detail.lower())
        file.read.assert_awaited_once_with(PREVIEW_UPLOAD_BYTES + 1)
        file.close.assert_awaited_once()
        read.assert_not_called()
        predict.assert_not_called()


class PreviewSourcePolicyTests(unittest.TestCase):
    def test_exact_full_boundaries_are_allowed(self):
        for width, height in ((4000, 3000), (8192, 1), (1, 8192)):
            with self.subTest(width=width, height=height):
                policy = preview_source_policy(width, height, 'full')
                self.assertEqual((policy.width, policy.height), (width, height))
                self.assertFalse(policy.is_proxy)

    def test_proxy_accepts_large_source_dimensions_and_retains_legacy_resize(self):
        for width, height, expected in ((8193, 1, (960, 1)), (3000, 4001, (405, 540)),
                                        (64, 48, (64, 48)), (1, 8193, (1, 540))):
            with self.subTest(width=width, height=height):
                policy = preview_source_policy(width, height)
                self.assertEqual((policy.width, policy.height), expected)
                self.assertTrue(policy.is_proxy)

    def test_upload_limit_uses_mib_and_accepts_exact_boundary(self):
        validate_preview_upload_size(PREVIEW_UPLOAD_BYTES)
        with self.assertRaises(PreviewPolicyError) as caught:
            validate_preview_upload_size(PREVIEW_UPLOAD_BYTES + 1)
        self.assertEqual(caught.exception.status_code, 413)

    def test_empty_dimensions_are_actionable_errors(self):
        for width, height in ((0, 1), (1, 0)):
            with self.subTest(width=width, height=height), self.assertRaises(PreviewPolicyError) as caught:
                preview_source_policy(width, height, 'full')
            self.assertEqual(caught.exception.status_code, 400)
            self.assertIn('proxy', str(caught.exception))
