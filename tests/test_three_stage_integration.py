"""Boundary checks for new opt-in workflows and ordinary pipeline reuse."""
import hashlib
import json
import unittest
import uuid
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
import main
from job_settings import settings_snapshot, validate_settings
from recovery_store import RecoveryStore


class ThreeStageIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'three-stage-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.client = TestClient(main.app, base_url='http://127.0.0.1')

    def job(self, **changes):
        fields = dict(id=str(uuid.uuid4()), filename='video.mp4', input_path=self.root / 'video.mp4',
                      media_type='video', layout='SBS', strength=5, convergence=.5,
                      depth_mode='depth-anything-v2-small', input_size=256,
                      resolution='source', quality='fast', invert_depth=False,
                      execution_mode='cpu', purpose='calibration')
        return main.Job(**{**fields, **changes})

    def test_new_settings_legacy_defaults_and_strict_values(self):
        settings = settings_snapshot(self.job())
        legacy = {k: v for k, v in settings.items() if k not in {'renderer', 'temporal_depth_scale'}}
        self.assertEqual(validate_settings(legacy), settings)
        for key, value in [('renderer', 'unknown'), ('temporal_depth_scale', 1), ('depth_interval', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings({**settings, key: value})
        self.assertEqual(validate_settings({**settings, 'depth_interval': 0})['depth_interval'], 0)

    def test_invalid_workflows_reject_before_upload_or_model_work(self):
        with patch.object(main, 'reserve_job_upload') as reserve:
            for data in [dict(renderer='invalid'), dict(purpose='calibration', depth_mode='balanced'),
                         dict(purpose='calibration', calibration_budget_seconds='nan')]:
                response = self.client.post('/api/jobs', data=data,
                    files={'file': ('source.mp4', b'unused', 'video/mp4')})
                self.assertEqual(response.status_code, 400, response.text)
            reserve.assert_not_called()

    def test_missing_calibration_outputs_are_not_served(self):
        response = self.client.get('/api/calibration-exports/not-a-job/baseline/0')
        self.assertEqual(response.status_code, 404)

    def test_calibration_uses_regular_pipeline_and_hides_children(self):
        job = self.job()
        job.input_path.write_bytes(b'source')
        proxy = self.root / 'proxy.mp4'
        proxy.write_bytes(b'proxy')
        job.preview_source_path = proxy
        seen = []
        def run(child, net):
            self.assertIs(child.cancel_event, job.cancel_event)
            self.assertFalse(child.use_depth_cache)
            self.assertEqual(child.preview_source_path, proxy)
            self.assertNotIn(child.id, [row['id'] for row in main.get_jobs()])
            seen.append(settings_snapshot(child))
            child.execution = {'active_provider': 'CPUExecutionProvider', 'device': 'CPU',
                               'inference_succeeded': True}
            output = main.output_path_for(child)
            output.write_bytes(b'complete export')
            return output
        with patch.object(main, 'OUTPUTS', self.root), patch.object(main, 'run_video', side_effect=run), \
             patch.object(main, 'probe_video', return_value={'width': 64, 'height': 24}), \
             patch.object(main, 'run_media_command', return_value=type('R', (), {'returncode': 0})()), \
             patch.object(main.DEPTH_ENGINE, 'runtime_info', return_value={'providers': ['CPUExecutionProvider']}):
            main.JOBS[job.id] = job
            try:
                output = main.run_export_calibration(job, None, {'width': 32, 'height': 24})
                report = json.loads(output.read_text())
                self.assertTrue(report['sufficient_evidence'])
                self.assertEqual(len(seen), 4)
                self.assertEqual(len(job.calibration_outputs), 4)
                self.assertTrue(all(row['depth_mode'] == job.depth_mode for row in seen))
                response = self.client.get(f'/api/calibration-exports/{job.id}/baseline/0')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.content, b'complete export')
            finally:
                main.JOBS.pop(job.id, None)

    def test_calibration_failure_cleans_partial_and_child_registration(self):
        job = self.job()
        job.input_path.write_bytes(b'source')
        job.preview_source_path = job.input_path
        def fail(child, net):
            main.output_path_for(child).write_bytes(b'partial')
            raise RuntimeError('simulated decoder failure')
        with patch.object(main, 'OUTPUTS', self.root), patch.object(main, 'run_video', side_effect=fail), \
             patch.object(main.DEPTH_ENGINE, 'runtime_info', return_value={}):
            before = set(main.JOBS)
            output = main.run_export_calibration(job, None, {})
            self.assertEqual(set(main.JOBS), before)
            self.assertFalse(job.calibration_report['sufficient_evidence'])
            self.assertEqual(job.calibration_outputs, {})
            self.assertEqual(list(self.root.glob('*.mp4')), [job.input_path])
            self.assertTrue(output.is_file())

    def test_recovery_restart_claims_source_and_preserves_full_settings(self):
        upload_root = self.root / 'uploads'
        upload_root.mkdir()
        store = RecoveryStore(self.root / 'recovery', upload_root).open()
        job_id = str(uuid.uuid4())
        source = upload_root / f'{job_id}.mp4'
        source.write_bytes(b'unchanged source')
        settings = settings_snapshot(self.job(layout='XREAL_SBS', renderer='visibility',
                                              temporal_depth_scale=True, depth_interval=0))
        store.admit(job_id, 'video.mp4', hashlib.sha256(source.read_bytes()).hexdigest(),
                    source.stat().st_size, settings, '{name}_{layout}')
        store.checkpoint(job_id, 'interrupted', 12)
        before = set(main.JOBS)
        try:
            with patch.object(main, 'RECOVERY', store), patch.object(main, 'UPLOADS', upload_root), \
                 patch.object(main.JOB_QUEUE, 'qsize', return_value=0), \
                 patch.object(main.JOB_QUEUE, 'put_nowait'), patch.object(main, 'ensure_worker'):
                response = self.client.post(f'/api/recovery/{job_id}/restart')
                self.assertEqual(response.status_code, 200, response.text)
                job = main.JOBS[response.json()['id']]
                self.assertEqual(job.input_path, source)
                self.assertEqual(job.layout, 'XREAL_SBS')
                self.assertEqual(job.renderer, 'visibility')
                self.assertEqual(job.depth_interval, 0)
                self.assertTrue(job.temporal_depth_scale)
                self.assertTrue(job.retain_source)
                self.assertEqual(store.list()[0]['claimed_by'], job.id)
                self.assertFalse(store.list()[0]['resume_mid_video'])
        finally:
            for job_key in set(main.JOBS) - before:
                main.JOBS.pop(job_key, None)
            store.close()
