"""Persistent reuse stores estimation results and leaves presentation independent."""
import hashlib
import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import main


class DepthCacheIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'depth-cache-integration' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.quarantine)
        self.source = self.root / 'source.mp4'
        self.source.write_bytes(b'exact source bytes')
        self.mode = 'depth-anything-v2-small'
        model = self.root / 'models' / self.mode / 'model.onnx'
        model.parent.mkdir(parents=True)
        model.write_bytes(b'model bytes')
        self.spec = {**main.MODEL_SPECS[self.mode],
                     'sha256': hashlib.sha256(model.read_bytes()).hexdigest()}
        self.frame = np.full((24, 32, 3), 100, np.uint8)
        self.depth = np.linspace(0, 1, 12 * 16, dtype=np.float32).reshape(12, 16)
        self.execution = {'active_provider': 'CPUExecutionProvider', 'device': 'cpu',
                          'fallback_reason': None, 'requested_mode': 'cpu'}
        for context in (patch.object(main, 'MODELS', self.root / 'models'),
                        patch.dict(main.MODEL_SPECS, {self.mode: self.spec}),
                        patch.object(main.DEPTH_ENGINE, 'runtime_info', return_value={
                            'version': 'test', 'providers': ['CPUExecutionProvider'],
                            'gpu_device': None, 'capabilities': {'schema_version': 1}})):
            context.start()
            self.addCleanup(context.stop)

    def quarantine(self):
        target = main.RUNTIME / '_quarantine' / 'depth-cache-integration' / self.root.name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.root, target)

    def job(self, **changes):
        values = dict(id=uuid.uuid4().hex, filename='source.mp4', input_path=self.source,
                      media_type='video', layout='SBS', strength=5, convergence=.5,
                      depth_mode=self.mode, input_size=256, resolution='source', quality='fast',
                      invert_depth=False, execution_mode='cpu', use_depth_cache=True)
        return main.Job(**{**values, **changes})

    def session(self, job, quota=1024 * 1024):
        store = main.DepthCache(self.root / 'cache', quota_bytes=quota)
        session = main.VideoDepthCacheSession(job, store=store)
        session.start(self.source, 32, 24, 24.0, {'backend': 'OpenCV', 'hardware': False})
        self.addCleanup(session.close)
        return session

    def store(self, **changes):
        job = self.job(**changes)
        session = self.session(job)
        self.assertIsNone(session.read(self.frame))
        session.append(self.frame, self.depth, self.execution)
        session.finish()
        session.close()
        self.assertEqual(job.depth_cache['state'], 'stored')
        return job

    def test_cache_opt_in_defaults_off_and_legacy_job_stays_disabled(self):
        job = self.job(use_depth_cache=False)
        self.session(job)
        self.assertEqual(job.public()['depth_cache']['state'], 'disabled')

    def test_completed_entry_reopens_for_presentation_tuning_with_exact_depth(self):
        self.store()
        job = self.job(strength=8, convergence=.2, layout='TB', edge_refinement=False,
                       quality='high')
        session = self.session(job)
        np.testing.assert_array_equal(session.read(self.frame), self.depth)
        session.finish()
        self.assertEqual(job.depth_cache['state'], 'hit')
        self.assertTrue(job.execution['depth_cache_hit'])
        self.assertEqual(job.execution['active_provider'], None)

    def test_temporal_and_source_changes_miss(self):
        self.store()
        for changes in ({'invert_depth': True}, {'depth_interval': 2},
                        {'temporal_smoothing': .6}, {'input_size': 384}):
            with self.subTest(changes=changes):
                session = self.session(self.job(**changes))
                self.assertIsNone(session.read(self.frame))
                session.close()
        self.source.write_bytes(b'changed source bytes')
        session = self.session(self.job())
        self.assertIsNone(session.read(self.frame))

    def test_canceled_or_failed_session_never_publishes_partial_depth(self):
        session = self.session(self.job())
        session.append(self.frame, self.depth, self.execution)
        session.close()
        fresh = self.session(self.job())
        self.assertIsNone(fresh.read(self.frame))

    def test_fallback_and_quota_failure_do_not_cache_or_interrupt_export(self):
        for quota, info in ((128, self.execution), (1024 * 1024,
                           {**self.execution, 'fallback_reason': 'temporary GPU failure'})):
            with self.subTest(quota=quota):
                job = self.job()
                session = self.session(job, quota)
                session.append(self.frame, self.depth, info)
                session.finish()
                self.assertEqual(job.depth_cache['state'], 'unavailable')
                self.assertTrue(job.depth_cache['warning'])
                session.close()

    def test_frame_digest_mismatch_requires_full_uncached_replay(self):
        self.store()
        session = self.session(self.job())
        with self.assertRaises(main.CacheReplayError):
            session.read(np.zeros_like(self.frame))

    def test_known_long_sequence_skips_an_over_quota_cache_before_first_frame_write(self):
        job = self.job()
        job.total_frames = 100000
        session = self.session(job)
        session.append(self.frame, self.depth, self.execution)
        self.assertEqual(job.depth_cache['state'], 'unavailable')
        self.assertGreater(job.depth_cache['estimated_bytes'], session.store.quota_bytes)
        self.assertEqual(job.depth_cache['frames'], 0)

    def test_nonfinite_cached_depth_restarts_instead_of_rendering_nan(self):
        job = self.store()
        session = self.session(self.job())
        key = session.key
        session.close()
        store = main.DepthCache(self.root / 'cache')
        writer = store.begin(key)
        writer.append(np.full_like(self.depth, np.nan), main.frame_digest(self.frame))
        writer.finish({'execution': self.execution, 'stored_stage': 'native-temporal-v1'})
        replay = self.session(self.job())
        with self.assertRaises(main.CacheReplayError):
            replay.read(self.frame)

    def test_cancel_before_finalization_keeps_completed_cache_absent(self):
        job = self.job()
        session = self.session(job)
        session.append(self.frame, self.depth, self.execution)
        job.cancel_event.set()
        with self.assertRaises(main.JobCancelled):
            session.finish()
        session.close()
        fresh = self.session(self.job())
        self.assertIsNone(fresh.read(self.frame))

    def test_unavailable_cache_directory_is_optional_and_disabled_does_not_open_it(self):
        for enabled in (False, True):
            job = self.job(use_depth_cache=enabled)
            session = main.VideoDepthCacheSession(job)
            with patch.object(main, 'get_video_depth_store',
                              side_effect=main.CacheError('cache directory is locked')) as get_store:
                session.start(self.source, 32, 24, 24.0, {})
            self.assertEqual(get_store.call_count, int(enabled))
            self.assertEqual(job.depth_cache['state'], 'unavailable' if enabled else 'disabled')

    def test_create_job_freezes_cache_opt_in_and_queued_edits_keep_it(self):
        from job_queue import EditableJobQueue
        client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(client.close)
        with patch.object(main, 'JOBS', {}), \
                patch.object(main, 'JOB_QUEUE', EditableJobQueue()), \
                patch.object(main, 'HISTORY', None), \
                patch.object(main, 'UPLOADS', self.root), \
                patch.object(main, 'ensure_worker'), \
                patch.object(main, 'validate_depth_mode'):
            response = client.post('/api/jobs', files={'file': ('source.mp4', b'video')},
                                   data={'use_depth_cache': 'true', 'depth_mode': self.mode})
            self.assertEqual(response.status_code, 200, response.text)
            job = response.json()
            self.assertTrue(job['settings']['use_depth_cache'])
            self.assertTrue(main.JOBS[job['id']].use_depth_cache)
            settings = {**job['settings'], 'strength': 8}
            edited = client.post(f"/api/jobs/{job['id']}/settings", json={'settings': settings})
            self.assertEqual(edited.status_code, 200, edited.text)
            self.assertTrue(edited.json()['settings']['use_depth_cache'])

    def test_storage_api_requires_explicit_keys_and_protects_active_entries(self):
        self.store()
        store = main.DepthCache(self.root / 'cache')
        client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(client.close)
        with patch.object(main, 'VIDEO_DEPTH_CACHE', store):
            report = client.get('/api/depth-cache').json()
            key = report['entries'][0]['key']
            reader = store.open_reader(key)
            try:
                response = client.post('/api/depth-cache/cleanup', json={'keys': [key]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['deleted'], [])
                self.assertTrue(response.json()['errors'])
            finally:
                reader.close()
            self.assertEqual(client.post('/api/depth-cache/cleanup',
                                         json={'keys': ['../escape']}).status_code, 400)
            self.assertEqual(client.post('/api/depth-cache/cleanup',
                                         json={'keys': [key]}).json()['deleted'], [key])


if __name__ == '__main__':
    unittest.main()
