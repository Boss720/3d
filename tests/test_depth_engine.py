import hashlib
import io
import json
import os
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from depth_engine import MODEL_SPECS, ModelManager

SMALL = 'depth-anything-v2-small'
LARGE = 'depth-anything-v2-large'
MIDAS = 'dpt-hybrid-midas'


class FakeSession:
    def __init__(self, shape=None, prediction=None, provider='CPUExecutionProvider'):
        self.input = SimpleNamespace(shape=shape or [1, 3, 'height', 'width'], type='tensor(float)', name='pixel_values')
        self.prediction = np.array([[[0., 1.], [2., 3.]]], dtype=np.float32) if prediction is None else prediction
        self.blob = None
        self.provider = provider
        self.fallback_disabled = False

    def get_providers(self):
        return [self.provider, 'CPUExecutionProvider'] if self.provider != 'CPUExecutionProvider' else [self.provider]

    def disable_fallback(self):
        self.fallback_disabled = True

    def get_inputs(self):
        return [self.input]

    def run(self, outputs, inputs):
        self.blob = inputs['pixel_values']
        return [self.prediction]


class FakeResponse(io.BytesIO):
    def __init__(self, data, length=None):
        super().__init__(data)
        self.headers = {} if length is None else {'Content-Length': str(length)}


class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parents[1] / '.runtime' / 'model-tests' / uuid.uuid4().hex
        self.manager = ModelManager(self.tmp)
        self.runtime = patch.object(self.manager, 'runtime_info', return_value={'available': True, 'error': None})
        self.runtime.start()

    def tearDown(self):
        self.runtime.stop()
        # Preserve generated test data; never recursively delete directories.
        quarantine = self.tmp.parents[1] / '_quarantine' / 'model-tests' / self.tmp.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.tmp, quarantine)

    def accepted(self, model_id, data=b'known model'):
        spec = dict(MODEL_SPECS[model_id], size=len(data), sha256=hashlib.sha256(data).hexdigest())
        patcher = patch.dict(MODEL_SPECS, {model_id: spec})
        patcher.start()
        self.addCleanup(patcher.stop)
        path = self.manager._path(model_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.manager._receipt(model_id).write_text(json.dumps({'sha256': spec['sha256'], 'revision': spec['revision']}))
        return path

    def test_catalog_covers_models_and_restricts_noncommercial_weights(self):
        catalog = self.manager.catalog()
        self.assertEqual(len(catalog['models']), 4)
        self.assertEqual(catalog['models'][0]['state'], 'not_installed')
        for entry in catalog['models']:
            self.assertEqual(len(entry['sha256']), 64)
            self.assertEqual(len(entry['revision']), 40)
            if entry['id'].endswith(('base', 'large')):
                self.assertIn('non-commercial', entry['license'])

    def test_unknown_and_uninstalled_models_do_not_fallback(self):
        with self.assertRaises(ValueError):
            self.manager.validate('balanced')
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            self.manager.predict(np.zeros((16, 16, 3), np.uint8), SMALL)

    def test_onnx_without_acceptance_receipt_is_not_installed(self):
        path = self.manager._path(SMALL)
        path.parent.mkdir()
        path.write_bytes(b'unvalidated')
        self.assertFalse(self.manager.available(SMALL))

    def test_missing_runtime_reports_actionable_failure(self):
        with patch.object(self.manager, 'runtime_info', return_value={'available': False, 'error': 'Install ONNX Runtime'}):
            with self.assertRaisesRegex(RuntimeError, 'Install ONNX Runtime'):
                self.manager.install(SMALL)

    def test_depth_anything_rgb_normalization_and_aspect_multiple14(self):
        frame = np.full((300, 600, 3), [0, 127, 255], dtype=np.uint8)
        session = FakeSession()
        depth = self.manager._predict_with_session(session, frame, SMALL, 518)
        self.assertEqual(session.blob.shape, (1, 3, 252, 518))
        np.testing.assert_allclose(session.blob[0, :, 0, 0], (np.array([1, 127/255, 0]) - [.485, .456, .406]) / [.229, .224, .225], rtol=1e-6)
        self.assertEqual(depth.shape, (300, 600))
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(float(depth.min()), 0.)
        self.assertEqual(float(depth.max()), 1.)
        self.assertGreater(depth[-1, -1], depth[0, 0])

    def test_midas_uses_384_and_half_normalization(self):
        session = FakeSession()
        self.manager._predict_with_session(session, np.full((50, 100, 3), 255, np.uint8), MIDAS, 518)
        self.assertEqual(session.blob.shape, (1, 3, 384, 384))
        np.testing.assert_array_equal(session.blob, 1.)

    def test_static_dimensions_override_user_size(self):
        session = FakeSession([1, 3, 392, 518])
        self.manager._predict_with_session(session, np.zeros((100, 200, 3), np.uint8), SMALL, 256)
        self.assertEqual(session.blob.shape, (1, 3, 392, 518))

    def test_nonfinite_and_wrong_output_are_errors(self):
        for prediction in [np.array([[[np.nan]]]), np.zeros((2, 3, 4))]:
            with self.subTest(shape=prediction.shape), self.assertRaisesRegex(RuntimeError, 'finite 2D'):
                self.manager._predict_with_session(FakeSession(prediction=prediction), np.zeros((10, 10, 3), np.uint8), SMALL, 518)

    def test_constant_map_is_finite_and_flat(self):
        depth = self.manager._predict_with_session(FakeSession(prediction=np.ones((1, 1, 2, 2))), np.zeros((8, 8, 3), np.uint8), SMALL, 518)
        np.testing.assert_array_equal(depth, 0)

    def test_install_deduplicates_and_accepts_only_after_smoke_inference(self):
        data = b'validated model bytes'
        spec = dict(MODEL_SPECS[SMALL], size=len(data), sha256=hashlib.sha256(data).hexdigest())
        session = FakeSession()
        with patch.dict(MODEL_SPECS, {SMALL: spec}), patch('depth_engine.urllib.request.urlopen', return_value=FakeResponse(data, len(data))) as opened, patch.object(self.manager, '_new_session', return_value=session):
            result = self.manager.install(SMALL)
            self.assertIn(result['state'], ('downloading', 'ready'))
            self.manager._threads[SMALL].join(5)
            self.assertEqual(self.manager._public(SMALL)['state'], 'ready')
            self.assertTrue(self.manager.available(SMALL))
            self.assertIsNotNone(session.blob)
            self.manager.install(SMALL)
            self.assertEqual(opened.call_count, 1)
            self.assertIn(spec['revision'], opened.call_args.args[0].full_url)
            self.assertEqual(self.manager._path(SMALL).read_bytes(), data)
            self.assertFalse(list(self.manager._path(SMALL).parent.glob('*.part')))

    def test_bad_download_hash_preserves_existing_accepted_cache(self):
        path = self.accepted(SMALL, b'good model')
        with patch('depth_engine.urllib.request.urlopen', return_value=FakeResponse(b'bad! model')), patch.object(self.manager, '_new_session') as created:
            self.manager._download(SMALL)
        self.assertEqual(path.read_bytes(), b'good model')
        self.assertTrue(self.manager._accepted(SMALL))
        self.assertEqual(self.manager._public(SMALL)['state'], 'failed')
        self.assertIn('SHA256', self.manager._public(SMALL)['error'])
        created.assert_not_called()
        self.assertFalse(list(path.parent.glob('*.part')))
        self.assertTrue(list((self.manager.models_dir / '_quarantine').rglob('*.part')))

    def test_valid_bytes_with_unusable_session_never_accepted(self):
        data = b'not really an ONNX'
        spec = dict(MODEL_SPECS[SMALL], size=len(data), sha256=hashlib.sha256(data).hexdigest())
        with patch.dict(MODEL_SPECS, {SMALL: spec}), patch('depth_engine.urllib.request.urlopen', return_value=FakeResponse(data)), patch.object(self.manager, '_new_session', side_effect=RuntimeError('Invalid ONNX graph')):
            self.manager._download(SMALL)
            self.assertFalse(self.manager._accepted(SMALL))
            self.assertFalse(self.manager._path(SMALL).exists())
            self.assertIn('Invalid ONNX graph', self.manager._public(SMALL)['error'])

    def test_oversized_response_rejected_before_session_load(self):
        spec = dict(MODEL_SPECS[SMALL], size=2)
        with patch.dict(MODEL_SPECS, {SMALL: spec}), patch('depth_engine.urllib.request.urlopen', return_value=FakeResponse(b'123')), patch.object(self.manager, '_new_session') as created:
            self.manager._download(SMALL)
        self.assertIn('exceeded', self.manager._public(SMALL)['error'])
        created.assert_not_called()

    def test_one_resident_session_and_reuse(self):
        self.accepted(SMALL)
        self.accepted(MIDAS)
        with patch.object(self.manager, '_new_session', side_effect=lambda path: FakeSession()) as created:
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL)
            self.manager.predict(frame, SMALL)
            self.assertEqual(created.call_count, 1)
            self.manager.predict(frame, MIDAS)
            self.assertEqual(created.call_count, 2)
            self.assertEqual(self.manager._session_id, MIDAS)

    def test_corrupted_accepted_model_fails_and_can_be_reinstalled(self):
        path = self.accepted(SMALL, b'good model')
        path.write_bytes(b'evil model')
        with patch.object(self.manager, '_new_session') as created:
            with self.assertRaisesRegex(RuntimeError, 'SHA256'):
                self.manager.predict(np.zeros((8, 8, 3), np.uint8), SMALL)
        created.assert_not_called()
        self.assertFalse(self.manager.available(SMALL))
        self.assertEqual(self.manager._public(SMALL)['state'], 'failed')

    def test_malformed_acceptance_receipt_is_not_accepted(self):
        self.accepted(SMALL)
        self.manager._receipt(SMALL).write_text('[]')
        self.assertFalse(self.manager.available(SMALL))

    def test_interrupted_download_is_failed_and_partial_is_quarantined(self):
        class Interrupted(FakeResponse):
            def read(self, size=-1):
                if self.tell():
                    raise OSError('network interrupted')
                return super().read(size)

        with patch('depth_engine.urllib.request.urlopen', return_value=Interrupted(b'partial bytes')):
            self.manager._download(SMALL)
        self.assertFalse(self.manager.available(SMALL))
        self.assertEqual(self.manager._public(SMALL)['state'], 'failed')
        self.assertIn('network interrupted', self.manager._public(SMALL)['error'])
        self.assertFalse(list(self.manager._path(SMALL).parent.glob('*.part')))

    def gpu_runtime(self):
        return {'available': True, 'error': None, 'gpu_available': True,
                'gpu_name': 'NVIDIA GeForce GTX 1650 Ti',
                'gpu_device': {'id': 1, 'name': 'NVIDIA GeForce GTX 1650 Ti', 'vendor_id': 0x10DE, 'memory_mb': 4096}}

    def test_gpu_mode_selects_dxgi_index_and_verifies_only_after_run(self):
        self.accepted(SMALL)
        session = FakeSession(provider='DmlExecutionProvider')
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', return_value=session) as created:
            self.assertFalse(self.manager.execution_info().get('gpu_verified', False))
            self.manager.predict(np.zeros((8, 8, 3), np.uint8), SMALL, execution_mode='gpu')
            self.assertEqual(created.call_args.kwargs['device_id'], 1)
            self.assertEqual(created.call_args.kwargs['provider'], 'DmlExecutionProvider')
            self.assertEqual(self.manager.execution_info()['active_provider'], 'DmlExecutionProvider')
            self.assertTrue(self.manager.execution_info()['gpu_verified'])
            self.assertTrue(session.fallback_disabled)
            snapshot = self.manager.execution_info()
            snapshot['active_provider'] = 'made up'
            self.assertEqual(self.manager.execution_info()['active_provider'], 'DmlExecutionProvider')

    def test_capability_schema_exists_even_when_runtime_cannot_load(self):
        expected = {'schema_version': 1, 'named_models_directml': True,
                    'gpu_max_input_dimension': 924, 'large_limited_gpu_max_input_dimension': 518,
                    'limited_gpu_memory_mb': 4096, 'unknown_memory_policy': 'limited',
                    'gpu_failure_cooldown_seconds': 60, 'gpu_retry_supported': True}
        hardware = {'selected_adapter': None, 'adapters': [], 'error': None}
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'], __version__='fake')
        with patch('depth_engine.probe_hardware', return_value=hardware):
            with patch.object(self.manager, '_ort', return_value=ort):
                self.assertEqual(ModelManager.runtime_info(self.manager)['capabilities'], expected)
            with patch.object(self.manager, '_ort', side_effect=RuntimeError('not installed')):
                self.assertEqual(ModelManager.runtime_info(self.manager)['capabilities'], expected)

    def test_unknown_or_zero_memory_conservatively_caps_large(self):
        self.accepted(LARGE)
        for memory in [None, 0, -1, 'unknown', float('nan')]:
            with self.subTest(memory=memory):
                runtime = self.gpu_runtime()
                runtime['gpu_device']['memory_mb'] = memory
                session = FakeSession(provider='DmlExecutionProvider')
                self.manager._session_key = None
                with patch.object(self.manager, 'runtime_info', return_value=runtime), patch.object(self.manager, '_new_session', return_value=session):
                    self.manager.predict(np.zeros((500, 1000, 3), np.uint8), LARGE, execution_mode='gpu')
                info = self.manager.execution_info()
                self.assertLessEqual(max(session.blob.shape[2:]), 518)
                self.assertEqual(info['input_size_limit'], 518)
                self.assertTrue(info['inference_succeeded'])
                self.assertEqual(info['requested_input_size'], 518)
                self.assertEqual((info['requested_input_height'], info['requested_input_width']), (518, 1036))
                self.assertEqual((info['input_height'], info['input_width']), session.blob.shape[2:])
        self.assertEqual(self.manager._gpu_input_limit(LARGE, {'memory_mb': 8192}), 924)
        self.assertEqual(self.manager._gpu_input_limit(SMALL, {'memory_mb': None}), 924)

    def test_gpu_suppression_expires_after_monotonic_cooldown_and_success_clears_it(self):
        self.accepted(SMALL)
        attempts = []
        def create(path, **options):
            provider = options.get('provider', 'CPUExecutionProvider')
            attempts.append(provider)
            if provider == 'DmlExecutionProvider' and attempts.count(provider) == 1:
                raise RuntimeError('transient allocation failure')
            return FakeSession(provider=provider)
        with patch('time.monotonic', return_value=100) as clock, patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=create):
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL)
            self.assertTrue(self.manager.execution_info()['inference_succeeded'])
            self.assertIn('transient allocation failure', self.manager.execution_info()['fallback_reason'])
            clock.return_value = 159.999
            self.manager.predict(frame, SMALL)
            self.assertEqual(attempts.count('DmlExecutionProvider'), 1)
            clock.return_value = 160
            self.manager.predict(frame, SMALL)
            self.assertEqual(attempts.count('DmlExecutionProvider'), 2)
            self.assertTrue(self.manager.execution_info()['gpu_verified'])
            self.assertFalse(self.manager._gpu_failures)

    def test_gpu_suppression_isolates_adapter_model_and_input_settings(self):
        self.accepted(SMALL)
        self.accepted(LARGE)
        runtime = self.gpu_runtime()
        attempts = []
        def create(path, **options):
            provider = options.get('provider', 'CPUExecutionProvider')
            attempts.append(provider)
            if provider == 'DmlExecutionProvider':
                raise RuntimeError('device removed')
            return FakeSession()
        with patch('time.monotonic', return_value=100), patch.object(self.manager, 'runtime_info', return_value=runtime), patch.object(self.manager, '_new_session', side_effect=create):
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL)
            self.manager.predict(frame, SMALL)
            self.assertEqual(attempts.count('DmlExecutionProvider'), 1)
            self.manager.predict(frame, SMALL, input_size=256)
            self.manager.predict(np.zeros((8, 16, 3), np.uint8), SMALL)
            self.manager.predict(frame, LARGE)
            runtime['gpu_device']['luid'] = 'new-physical-adapter'
            self.manager.predict(frame, SMALL)
            runtime['gpu_device']['memory_mb'] = 8192
            self.manager.predict(frame, LARGE)
            self.assertEqual(attempts.count('DmlExecutionProvider'), 6)

    def test_explicit_gpu_retries_during_suppression_and_failure_reports_unsuccessful(self):
        self.accepted(SMALL)
        def create(path, **options):
            if options.get('provider') == 'DmlExecutionProvider':
                raise RuntimeError('device removed')
            return FakeSession()
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=create) as created:
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL)
            with self.assertRaisesRegex(RuntimeError, 'device removed'):
                self.manager.predict(frame, SMALL, execution_mode='gpu')
            self.assertEqual(created.call_count, 3)
        info = self.manager.execution_info()
        self.assertFalse(info['inference_succeeded'])
        self.assertFalse(info['gpu_verified'])
        self.assertEqual(info['input_size_limit'], 924)
        self.assertIsNone(info['input_width'])

    def test_failed_gpu_run_reports_attempted_dimensions_without_success(self):
        self.accepted(LARGE)
        session = FakeSession(provider='DmlExecutionProvider')
        session.run = lambda *args: (_ for _ in ()).throw(RuntimeError('device removed'))
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', return_value=session):
            with self.assertRaisesRegex(RuntimeError, 'device removed'):
                self.manager.predict(np.zeros((500, 1000, 3), np.uint8), LARGE, execution_mode='gpu')
        info = self.manager.execution_info()
        self.assertFalse(info['inference_succeeded'])
        self.assertFalse(info['gpu_verified'])
        self.assertEqual(info['active_provider'], 'DmlExecutionProvider')
        self.assertEqual(info['adapter']['id'], 1)
        self.assertEqual(info['input_size_limit'], 518)
        self.assertEqual((info['requested_input_height'], info['requested_input_width']), (518, 1036))
        self.assertEqual((info['input_height'], info['input_width']), (252, 518))

    def test_cpu_success_reports_requested_and_actual_dimensions_without_gpu_cap(self):
        self.accepted(SMALL)
        with patch.object(self.manager, '_new_session', return_value=FakeSession()):
            self.manager.predict(np.zeros((8, 16, 3), np.uint8), SMALL, input_size=256, execution_mode='cpu')
        info = self.manager.execution_info()
        self.assertTrue(info['inference_succeeded'])
        self.assertFalse(info['gpu_verified'])
        self.assertEqual(info['active_provider'], 'CPUExecutionProvider')
        self.assertEqual(info['requested_device'], 'cpu')
        self.assertEqual((info['requested_input_height'], info['requested_input_width']), (126, 252))
        self.assertEqual((info['input_height'], info['input_width']), (126, 252))
        self.assertIsNone(info['input_size_limit'])

    def test_reset_gpu_failures_waits_for_inference_and_preserves_session(self):
        from concurrent.futures import ThreadPoolExecutor
        self.accepted(SMALL)
        entered, release, reset_attempted = threading.Event(), threading.Event(), threading.Event()
        session = FakeSession(provider='DmlExecutionProvider')
        original_run = session.run
        def run(*args):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test inference release timed out')
            return original_run(*args)
        session.run = run
        def reset():
            reset_attempted.set()
            self.manager.reset_gpu_failures()
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', return_value=session), ThreadPoolExecutor(max_workers=2) as pool:
            inference = pool.submit(self.manager.predict, np.zeros((8, 8, 3), np.uint8), SMALL)
            try:
                self.assertTrue(entered.wait(2))
                reset_future = pool.submit(reset)
                self.assertTrue(reset_attempted.wait(2))
                self.assertFalse(reset_future.done())
                self.assertIs(self.manager._session, session)
            finally:
                release.set()
            inference.result(timeout=3)
            reset_future.result(timeout=3)
        self.assertIs(self.manager._session, session)
        self.assertTrue(self.manager.execution_info()['inference_succeeded'])

    def test_gpu_reset_allows_immediate_retry_and_suppression_storage_is_bounded(self):
        self.accepted(SMALL)
        def create(path, **options):
            if options.get('provider') == 'DmlExecutionProvider':
                raise RuntimeError('out of memory')
            return FakeSession()
        with patch('time.monotonic', return_value=100), patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=create) as created:
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL)
            self.manager.reset_gpu_failures()
            self.assertFalse(self.manager._gpu_failures)
            self.manager.predict(frame, SMALL)
            self.assertEqual(created.call_count, 4)
            for detail in range(128, 220):
                self.manager.predict(frame, SMALL, input_size=detail)
            self.assertLessEqual(len(self.manager._gpu_failures), 64)

    def test_explicit_gpu_rejects_missing_backend(self):
        self.accepted(SMALL)
        unavailable = {'available': True, 'gpu_available': False, 'gpu_error': 'DirectML is unavailable'}
        with patch.object(self.manager, 'runtime_info', return_value=unavailable):
            with self.assertRaisesRegex(RuntimeError, 'DirectML is unavailable'):
                self.manager.validate(SMALL, execution_mode='gpu')

    def test_large_uses_gpu_on_4gb_with_reduced_input_limit(self):
        self.accepted(LARGE)
        session = FakeSession(provider='DmlExecutionProvider')
        frame = np.zeros((2000, 3000, 3), np.uint8)
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), \
             patch.object(self.manager, '_new_session', return_value=session) as created:
            self.manager.validate(LARGE, execution_mode='gpu')
            depth = self.manager.predict(frame, LARGE, input_size=518, execution_mode='gpu')

        self.assertEqual(depth.shape, frame.shape[:2])
        self.assertEqual(created.call_args.kwargs['provider'], 'DmlExecutionProvider')
        self.assertLessEqual(max(session.blob.shape[2:]), 518)
        self.assertEqual(self.manager.execution_info()['active_provider'], 'DmlExecutionProvider')
        self.assertEqual(self.manager.execution_info()['input_size_limit'], 518)
        self.assertIsNone(self.manager.execution_info()['fallback_reason'])

    def test_large_auto_falls_back_to_cpu_after_gpu_memory_failure(self):
        self.accepted(LARGE)
        def create(path, **options):
            if options.get('provider') == 'DmlExecutionProvider':
                raise RuntimeError('DML out of memory')
            return FakeSession()

        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), \
             patch.object(self.manager, '_new_session', side_effect=create) as created:
            self.manager.predict(np.zeros((8, 8, 3), np.uint8), LARGE, execution_mode='auto')

        self.assertEqual(created.call_count, 2)
        self.assertEqual(self.manager.execution_info()['active_provider'], 'CPUExecutionProvider')
        self.assertIn('out of memory', self.manager.execution_info()['fallback_reason'])

    def test_lower_input_detail_retries_large_on_gpu_after_memory_failure(self):
        self.accepted(LARGE)

        class SizeLimitedSession(FakeSession):
            def __init__(self):
                super().__init__(provider='DmlExecutionProvider')

            def run(self, outputs, inputs):
                self.blob = inputs['pixel_values']
                if max(self.blob.shape[2:]) > 256:
                    raise RuntimeError('DML out of memory')
                return [self.prediction]

        providers = []
        def create(path, **options):
            provider = options.get('provider', 'CPUExecutionProvider')
            providers.append(provider)
            return SizeLimitedSession() if provider == 'DmlExecutionProvider' else FakeSession()

        frame = np.zeros((8, 8, 3), np.uint8)
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), \
             patch.object(self.manager, '_new_session', side_effect=create):
            self.manager.predict(frame, LARGE, input_size=518, execution_mode='auto')
            self.assertEqual(self.manager.execution_info()['active_provider'], 'CPUExecutionProvider')
            self.manager.predict(frame, LARGE, input_size=256, execution_mode='auto')

        self.assertEqual(providers, ['DmlExecutionProvider', 'CPUExecutionProvider', 'DmlExecutionProvider'])
        self.assertEqual(self.manager.execution_info()['active_provider'], 'DmlExecutionProvider')

    def test_gpu_failure_auto_falls_back_but_strict_modes_report_it(self):
        self.accepted(SMALL)
        def create(path, **options):
            if options.get('provider') == 'DmlExecutionProvider':
                raise RuntimeError('DML out of memory')
            return FakeSession()
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=create) as created:
            frame = np.zeros((8, 8, 3), np.uint8)
            self.manager.predict(frame, SMALL, execution_mode='auto')
            self.assertEqual(self.manager.execution_info()['active_provider'], 'CPUExecutionProvider')
            self.assertIn('out of memory', self.manager.execution_info()['fallback_reason'])
            self.assertEqual(created.call_count, 2)
            for mode in ('hybrid', 'gpu'):
                with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    self.manager.predict(frame, SMALL, execution_mode=mode)
                self.assertEqual(self.manager.execution_info()['active_provider'], None)
                self.assertIn('out of memory', self.manager.execution_info()['error'])
            self.assertEqual(created.call_count, 4)

    def test_silently_fallen_back_session_is_not_gpu_success(self):
        self.accepted(SMALL)
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', return_value=FakeSession()):
            with self.assertRaisesRegex(RuntimeError, 'DmlExecutionProvider'):
                self.manager.predict(np.zeros((8, 8, 3), np.uint8), SMALL, execution_mode='gpu')
            self.assertFalse(self.manager.execution_info()['gpu_verified'])

    def test_session_cache_includes_provider_and_gpu_dimensions_are_bounded(self):
        self.accepted(SMALL)
        sessions = []
        def create(path, **options):
            session = FakeSession(provider=options.get('provider', 'CPUExecutionProvider'))
            sessions.append(session)
            return session
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=create):
            frame = np.zeros((2000, 20, 3), np.uint8)
            self.manager.predict(frame, SMALL, execution_mode='gpu')
            self.assertLessEqual(max(sessions[-1].blob.shape[2:]), 924)
            self.assertTrue(all(size % 14 == 0 for size in sessions[-1].blob.shape[2:]))
            self.manager.predict(frame, SMALL, execution_mode='gpu')
            self.assertEqual(len(sessions), 1)
            self.manager.predict(np.zeros((8, 8, 3), np.uint8), SMALL, execution_mode='cpu')
            self.assertEqual(len(sessions), 2)

    def test_unknown_execution_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'execution mode'):
            self.manager.validate(SMALL, execution_mode='magic')

    def test_failed_gpu_run_falls_back_and_does_not_mark_gpu_verified(self):
        self.accepted(SMALL)
        session = FakeSession(provider='DmlExecutionProvider')
        session.run = lambda *args: (_ for _ in ()).throw(RuntimeError('device removed'))
        with patch.object(self.manager, 'runtime_info', return_value=self.gpu_runtime()), patch.object(self.manager, '_new_session', side_effect=[session, FakeSession()]):
            self.manager.predict(np.zeros((8, 8, 3), np.uint8), SMALL)
            self.assertFalse(self.manager._gpu_verified)
            self.assertIn('device removed', self.manager.execution_info()['fallback_reason'])
            self.assertEqual(self.manager.execution_info()['active_provider'], 'CPUExecutionProvider')

    def test_runtime_registration_is_available_but_unverified_and_inactive(self):
        adapter = self.gpu_runtime()['gpu_device']
        ort = SimpleNamespace(get_available_providers=lambda: ['DmlExecutionProvider', 'CPUExecutionProvider'], __version__='fake')
        with patch.object(self.manager, '_ort', return_value=ort), patch('depth_engine.probe_hardware', return_value={'selected_adapter': adapter, 'adapters': [adapter], 'error': None}):
            info = ModelManager.runtime_info(self.manager)
            self.assertTrue(info['gpu_available'])
            self.assertFalse(info['gpu_verified'])
            self.assertIsNone(info['active_provider'])

    def test_directml_session_options_are_sequential_and_memory_pattern_disabled(self):
        session = FakeSession(provider='DmlExecutionProvider')
        options = SimpleNamespace()
        ort = SimpleNamespace(SessionOptions=lambda: options,
                              ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL='sequential'),
                              GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL='all'))
        from unittest.mock import Mock
        ort.InferenceSession = Mock(return_value=session)
        with patch.object(self.manager, '_ort', return_value=ort):
            self.manager._new_session(Path('unused.onnx'), provider='DmlExecutionProvider', device_id=1)
        self.assertEqual(options.execution_mode, 'sequential')
        self.assertFalse(options.enable_mem_pattern)
        self.assertFalse(options.enable_cpu_mem_arena)
        self.assertEqual(ort.InferenceSession.call_args.kwargs['providers'][0], ('DmlExecutionProvider', {'device_id': 1}))
        self.assertTrue(session.fallback_disabled)

    def test_predict_with_info_keeps_concurrent_request_metadata_separate(self):
        from concurrent.futures import ThreadPoolExecutor
        first_entered = threading.Event()
        second_attempted = threading.Event()
        second_entered = threading.Event()
        frame = np.zeros((8, 8, 3), np.uint8)

        def fake_predict(frame, model_id, input_size, mode):
            self.manager._last_execution = {'requested_mode': mode}
            if mode == 'cpu':
                first_entered.set()
                self.assertTrue(second_attempted.wait(2))
                # With the enclosing inference lock held, the GPU request cannot
                # replace CPU metadata before predict_with_info takes its snapshot.
                second_entered.wait(0.1)
            else:
                second_entered.set()
            return frame

        def second_request():
            self.assertTrue(first_entered.wait(2))
            second_attempted.set()
            return self.manager.predict_with_info(frame, SMALL, execution_mode='gpu')

        with patch.object(self.manager, 'predict', side_effect=fake_predict), ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.manager.predict_with_info, frame, SMALL, 518, 'cpu')
            second = pool.submit(second_request)
            self.assertEqual(first.result(timeout=3)[1]['requested_mode'], 'cpu')
            self.assertEqual(second.result(timeout=3)[1]['requested_mode'], 'gpu')


if __name__ == '__main__':
    unittest.main()
