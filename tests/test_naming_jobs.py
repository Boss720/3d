import concurrent.futures
import threading
import unittest
import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

import main


class NamingJobTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'naming-tests' / uuid.uuid4().hex
        self.uploads, self.outputs = self.root / 'uploads', self.root / 'outputs'
        self.uploads.mkdir(parents=True)
        self.outputs.mkdir()
        self.addCleanup(self.cleanup)
        for name, value in (('JOBS', {}), ('JOB_QUEUE', type(main.JOB_QUEUE)()),
                            ('UPLOADS', self.uploads), ('OUTPUTS', self.outputs),
                            ('ACCEPTING_MUTATIONS', True), ('WORKER_STOP', threading.Event()),
                            ('WORKER', None), ('HISTORY', None)):
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)
        self.png = cv2.imencode('.png', np.zeros((12, 16, 3), np.uint8))[1].tobytes()

    def cleanup(self):
        for folder in (self.uploads, self.outputs, self.root):
            for path in folder.iterdir():
                if path.is_file():
                    path.unlink()
            folder.rmdir()

    def submit(self, **settings):
        filename = settings.pop('filename', 'frame.png')
        with patch.object(main, 'ensure_worker'):
            return self.client.post('/api/jobs', data={'depth_mode': 'balanced', **settings},
                                    files={'file': (filename, self.png, 'image/png')})

    def edit(self, job_id, template):
        return self.client.post(f'/api/jobs/{job_id}/naming', json={'output_name_template': template})

    def test_status_and_legacy_default_have_separate_naming_schema(self):
        caps = self.client.get('/api/status').json()['naming_capabilities']
        self.assertEqual(caps, {'schema_version': 1, 'supported': True, 'max_template_length': 120,
                               'fields': ['name', 'layout', 'model', 'date'], 'unique_job_id_suffix': True})
        response = self.submit()
        self.assertEqual(response.status_code, 200, response.text)
        public = response.json()
        expected = f"frame_sbs_{uuid.UUID(public['id']).hex}.png"
        self.assertEqual(public['naming_schema_version'], 1)
        self.assertEqual(public['output_name_template'], '{name}_{layout}')
        self.assertEqual(public['planned_output_name'], expected)
        self.assertIsNone(public['output_name'])
        self.assertEqual(len(public['settings']), 13)
        self.assertEqual(public['settings']['depth_interval'], 1)
        self.assertEqual(public['settings']['video_decode'], 'software')
        self.assertNotIn('output_name_template', public['settings'])

    def test_video_depth_options_are_validated_and_frozen_in_settings(self):
        response = self.submit(depth_interval='2', video_decode='amd')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['settings']['depth_interval'], 2)
        self.assertEqual(response.json()['settings']['video_decode'], 'amd')
        for invalid in ({'depth_interval': '3'}, {'video_decode': 'unsupported'}):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.submit(**invalid).status_code, 400)

    def test_invalid_template_rejects_before_model_validation_or_application_upload(self):
        for template in ('../{name}', '{name!r}', '{model.path}', 'x' * 121, ''):
            with self.subTest(template=template), patch.object(main, 'validate_depth_mode') as validate, \
                    patch.object(main, 'ensure_worker') as worker:
                response = self.client.post('/api/jobs', data={'depth_mode': 'depth-anything-v2-small',
                    'output_name_template': template}, files={'file': ('frame.png', self.png, 'image/png')})
                self.assertEqual(response.status_code, 400, response.text)
                validate.assert_not_called()
                worker.assert_not_called()
                self.assertEqual(list(self.uploads.iterdir()), [])
                self.assertEqual(main.JOBS, {})

    def test_custom_image_output_uses_planned_name_and_preserves_exclusive_ownership(self):
        response = self.submit(output_name_template='Export-{name}-{layout}')
        self.assertEqual(response.status_code, 200, response.text)
        public = response.json()
        job = main.JOBS[public['id']]
        expected = f'Export-frame-sbs_{uuid.UUID(job.id).hex}.png'
        self.assertEqual(public['planned_output_name'], expected)
        self.assertEqual(main.JOB_QUEUE.get_nowait(), job.id)
        main.process_job(job.id)
        main.JOB_QUEUE.task_done()
        self.assertEqual(job.state, 'completed', job.error)
        self.assertEqual(job.output_path.name, expected)
        self.assertTrue(job.output_path.is_file())
        self.assertFalse(job.input_path.exists())
        actual = self.client.get('/api/jobs').json()[0]
        self.assertEqual(actual['output_name'], expected)
        self.assertEqual(actual['planned_output_name'], expected)

    def test_file_object_in_template_field_is_rejected_as_invalid_string_before_model_work(self):
        with patch.object(main, 'validate_depth_mode') as validate, patch.object(main, 'ensure_worker'):
            response = self.client.post('/api/jobs', data={'depth_mode': 'balanced'}, files={
                'file': ('frame.png', self.png, 'image/png'),
                'output_name_template': ('template.txt', b'{name}', 'text/plain')})
        self.assertEqual(response.status_code, 400, response.text)
        validate.assert_not_called()
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertEqual(main.JOBS, {})

    def test_clip_names_ignore_export_template_and_naming_edit_is_rejected(self):
        with patch.object(main, 'FFMPEG_PATH', 'ffmpeg'):
            response = self.submit(filename='frame.mp4', purpose='preview_clip', output_name_template='../ignored')
        self.assertEqual(response.status_code, 200, response.text)
        public = response.json()
        self.assertEqual(public['planned_output_name'], f"frame_preview_sbs_{uuid.UUID(public['id']).hex}.mp4")
        self.assertEqual(self.edit(public['id'], '{date}').status_code, 409)

    def test_pending_template_edit_updates_only_naming_and_date_identity_remain_stable(self):
        public = self.submit().json()
        job = main.JOBS[public['id']]
        job.created_at = datetime(2026, 10, 18, 0, 30, tzinfo=timezone.utc).timestamp()
        original = dict(public['settings'])
        response = self.edit(job.id, '{name}_{layout}_{model}_{date}')
        self.assertEqual(response.status_code, 200, response.text)
        name = response.json()['planned_output_name']
        self.assertEqual(name, f'frame_sbs_balanced_20261018_{uuid.UUID(job.id).hex}.png')
        self.assertEqual(response.json()['settings'], original)
        with patch.object(main, 'validate_depth_mode'):
            changed = self.client.post(f'/api/jobs/{job.id}/settings',
                json={'settings': {**original, 'layout': 'TB', 'depth_mode': 'depth-anything-v2-small'}})
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertEqual(changed.json()['planned_output_name'],
                         f'frame_tb_depth-anything-v2-small_20261018_{uuid.UUID(job.id).hex}.png')
        self.assertEqual(changed.json()['output_name_template'], '{name}_{layout}_{model}_{date}')

    def test_invalid_naming_body_is_atomic_and_unknown_job_is_404(self):
        public = self.submit().json()
        job = main.JOBS[public['id']]
        for body in ({}, [], {'output_name_template': '{date}', 'extra': True},
                     {'output_name_template': 1}, {'output_name_template': None}, {'output_name_template': '{name:>20}'}):
            with self.subTest(body=body):
                response = self.client.post(f'/api/jobs/{job.id}/naming', json=body)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(job.output_name_template, '{name}_{layout}')
                self.assertEqual(main.output_path_for(job).name, public['planned_output_name'])
        self.assertEqual(self.edit('missing', '{date}').status_code, 404)

    def test_dequeued_running_terminal_and_canceled_jobs_cannot_change_naming(self):
        for state in ('queued', 'processing', 'completed', 'failed', 'canceled'):
            with self.subTest(state=state):
                public = self.submit().json()
                job = main.JOBS[public['id']]
                self.assertEqual(main.JOB_QUEUE.get_nowait(), job.id)
                job.state = state
                original = main.output_path_for(job)
                response = self.edit(job.id, '{date}')
                self.assertEqual(response.status_code, 409, response.text)
                self.assertEqual(main.output_path_for(job), original)
                self.assertEqual(job.output_name_template, '{name}_{layout}')
                main.JOB_QUEUE.task_done()

    def test_worker_dequeue_during_validation_rejects_naming_without_mutation(self):
        public = self.submit().json()
        job = main.JOBS[public['id']]
        original = main.output_path_for(job)
        entered, release = threading.Event(), threading.Event()
        validate = main.validate_output_template
        def delayed(template):
            entered.set()
            self.assertTrue(release.wait(2))
            return validate(template)
        with patch.object(main, 'validate_output_template', side_effect=delayed), concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.edit, job.id, 'renamed-{date}')
            try:
                self.assertTrue(entered.wait(1))
                self.assertEqual(pool.submit(main.JOB_QUEUE.get_nowait).result(timeout=1), job.id)
            finally:
                release.set()
            response = future.result(timeout=2)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(job.output_name_template, '{name}_{layout}')
        self.assertEqual(main.output_path_for(job), original)
        main.JOB_QUEUE.task_done()

    def test_custom_existing_output_is_preserved_on_processing_collision(self):
        public = self.submit(output_name_template='same-name').json()
        job = main.JOBS[public['id']]
        target = self.outputs / public['planned_output_name']
        target.write_bytes(b'preexisting-export')
        main.JOB_QUEUE.get_nowait()
        with patch.object(main, 'estimate_depth') as infer:
            main.process_job(job.id)
        main.JOB_QUEUE.task_done()
        self.assertEqual(job.state, 'failed')
        self.assertEqual(target.read_bytes(), b'preexisting-export')
        infer.assert_not_called()

    def test_naming_is_not_an_extra_setting_and_foreign_origin_cannot_change_it(self):
        public = self.submit().json()
        job = main.JOBS[public['id']]
        response = self.client.post(f'/api/jobs/{job.id}/settings',
            json={'settings': {**public['settings'], 'output_name_template': '{date}'}})
        self.assertEqual(response.status_code, 400, response.text)
        rejected = self.client.post(f'/api/jobs/{job.id}/naming', json={'output_name_template': '{date}'},
            headers={'Origin': 'https://foreign.example', 'Sec-Fetch-Site': 'cross-site'})
        self.assertEqual(rejected.status_code, 403, rejected.text)
        self.assertEqual(job.output_name_template, '{name}_{layout}')

    def test_surrogate_naming_json_rejects_before_mutation_and_valid_astral_pair_is_accepted(self):
        public = self.submit().json()
        job = main.JOBS[public['id']]
        # This validation test deliberately keeps its export pending; lifecycle
        # startup must not introduce an inference worker into the fixture.
        with patch.object(main, 'ensure_worker'), TestClient(main.app, base_url='http://127.0.0.1', raise_server_exceptions=False) as client:
            response = client.post(f'/api/jobs/{job.id}/naming',
                content=r'{"output_name_template":"\ud800_{name}"}', headers={'Content-Type': 'application/json'})
            self.assertEqual(response.status_code, 400, response.text)
            self.assertEqual(job.output_name_template, '{name}_{layout}')
            response = client.post(f'/api/jobs/{job.id}/naming',
                content=r'{"output_name_template":"\ud83d\ude80_{name}"}', headers={'Content-Type': 'application/json'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['output_name_template'], '\U0001f680_{name}')
