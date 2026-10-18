import uuid
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import main


class ClipJobTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.root = Path(main.RUNTIME) / 'clip-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.clean_owned)

    def clean_owned(self):
        for job_id, job in list(main.JOBS.items()):
            if job.input_path.parent == self.root:
                main.JOBS.pop(job_id, None)
        for path in sorted(self.root.rglob('*'), key=lambda x: len(x.parts), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.root.rmdir()

    def submit(self, **data):
        name = data.pop('filename', 'source.mp4')
        with patch.object(main, 'UPLOADS', self.root), patch.object(main, 'ensure_worker'), patch.object(main.JOB_QUEUE, 'put'):
            return self.client.post('/api/jobs', data={'depth_mode': 'balanced', **data},
                                    files={'file': (name, b'fake media', 'video/mp4')})

    def test_status_explicitly_advertises_bounded_clip_support(self):
        caps = self.client.get('/api/status').json()['preview_capabilities']
        self.assertEqual(caps['schema_version'], 1)
        self.assertEqual(caps['processed_clip']['max_frames'], 120)
        self.assertEqual(caps['processed_clip']['temporal_context'], 'fresh')

    def test_invalid_clip_requests_create_no_upload_or_job(self):
        for data in ({'purpose': 'unknown'}, {'purpose': 'preview_clip', 'filename': 'x.png'},
                     {'purpose': 'preview_clip', 'clip_start_seconds': '-1'},
                     {'purpose': 'preview_clip', 'clip_start_seconds': 'nan'},
                     {'purpose': 'preview_clip', 'clip_duration_seconds': '0.9'},
                     {'purpose': 'preview_clip', 'clip_duration_seconds': '5.1'},
                     {'purpose': 'preview_clip', 'clip_duration_seconds': 'inf'}):
            with self.subTest(data=data):
                before = set(main.JOBS)
                response = self.submit(**data)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(set(main.JOBS), before)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_export_default_and_clip_settings_are_frozen_publicly(self):
        response = self.submit()
        self.assertEqual(response.json()['purpose'], 'export')
        response = self.submit(purpose='preview_clip', clip_start_seconds='2.5', clip_duration_seconds='3',
                               temporal_smoothing='.7', edge_refinement='false', input_size='384')
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['purpose'], 'preview_clip')
        self.assertEqual(result['clip_start_seconds'], 2.5)
        self.assertEqual(result['clip_duration_seconds'], 3)
        self.assertEqual(result['input_size'], 384)
        self.assertEqual(result['temporal_smoothing'], .7)
        self.assertFalse(result['edge_refinement'])
        for job_id in (response.json()['id'],):
            main.JOBS.pop(job_id, None)

    def job(self, **kwargs):
        source = self.root / 'source.mp4'
        source.write_bytes(b'original')
        return main.Job(id='clip-unit', filename='source.mp4', input_path=source, media_type='video',
                        layout='SBS', strength=5, convergence=.5, depth_mode='balanced', input_size=256,
                        resolution='source', quality='fast', invert_depth=False, purpose='preview_clip',
                        clip_start_seconds=2, clip_duration_seconds=3, **kwargs)

    def test_processing_uses_separate_proxy_same_job_and_cleans_owned_inputs(self):
        job = self.job()
        target = self.root / 'result.mp4'
        proxy_paths = []
        report = {'fps': 12, 'total_frames': 100, 'source_width': 320, 'source_height': 180}
        def prepare(ffmpeg, source, destination, *args, **kwargs):
            self.assertEqual(source, job.input_path)
            self.assertNotEqual(source, destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b'proxy')
            proxy_paths.append(destination)
            return destination
        def convert(actual_job, net):
            self.assertIs(actual_job, job)
            self.assertEqual(actual_job.preview_source_path, proxy_paths[0])
            self.assertEqual(actual_job.temporal_smoothing, .35)
            self.assertTrue(actual_job.edge_refinement)
            target.write_bytes(b'output')
            return target
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'CLIP_PROXIES', self.root / 'proxies'), \
             patch.object(main, 'OUTPUTS', self.root), \
             patch.object(main, 'output_path_for', return_value=target), \
             patch.object(main, 'inspect_media', side_effect=[report, {**report, 'total_frames': 36}]), \
             patch.object(main, 'prepare_preview_clip', side_effect=prepare), \
             patch.object(main, 'run_video', side_effect=convert):
            main.process_job(job.id)
        self.assertEqual(job.state, 'completed', job.error)
        self.assertFalse(job.input_path.exists())
        self.assertFalse(proxy_paths[0].exists())
        self.assertTrue(target.exists())
        self.assertEqual(job.public()['clip']['proxy']['width'], 320)
        self.assertEqual(job.public()['clip']['temporal_context'], 'fresh')

    def test_out_of_range_start_rejects_before_extraction_or_inference(self):
        job = self.job()
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'inspect_media', return_value={'fps': 12, 'total_frames': 12}), \
             patch.object(main, 'prepare_preview_clip') as prepare, patch.object(main, 'estimate_depth') as infer:
            main.process_job(job.id)
        self.assertEqual(job.state, 'failed')
        self.assertIn('start', job.error.lower())
        prepare.assert_not_called()
        infer.assert_not_called()

    def test_unknown_source_timing_does_not_hard_reject_valid_nonzero_start(self):
        job = self.job()
        source_report = {'fps': 30, 'fps_provisional': True, 'total_frames': 3}
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'inspect_media', return_value=source_report), \
             patch.object(main, 'prepare_preview_clip', side_effect=main.ClipCancelled) as prepare:
            main.process_job(job.id)
        self.assertEqual(job.state, 'canceled')
        prepare.assert_called_once()
        self.assertIsNone(prepare.call_args.args[5])

    def test_proxy_exceeding_five_seconds_is_rejected_before_inference(self):
        job = self.job()
        with patch.dict(main.JOBS, {job.id: job}), \
             patch.object(main, 'inspect_media', return_value={'fps': 12, 'total_frames': 100, 'source_width': 320, 'source_height': 180}), \
             patch.object(main, 'prepare_preview_clip'), patch.object(main, 'run_video') as run:
            main.process_job(job.id)
        self.assertEqual(job.state, 'failed')
        self.assertIn('bounds', job.error)
        run.assert_not_called()

    def test_extraction_cancel_maps_to_canceled_and_download_is_unavailable(self):
        job = self.job()
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'inspect_media', return_value={'fps': 12, 'total_frames': 100}), \
             patch.object(main, 'prepare_preview_clip', side_effect=main.ClipCancelled), patch.object(main, 'run_video') as run:
            main.process_job(job.id)
            self.assertEqual(self.client.get(f'/api/preview-clips/{job.id}/video').status_code, 404)
        self.assertEqual(job.state, 'canceled')
        self.assertFalse(job.input_path.exists())
        run.assert_not_called()

    def test_inline_endpoint_requires_completed_clip_purpose(self):
        job = self.job()
        output = self.root / 'done.mp4'; output.write_bytes(b'mp4')
        job.output_path, job.state = output, 'completed'
        with patch.object(main, 'OUTPUTS', self.root), patch.dict(main.JOBS, {job.id: job}):
            response = self.client.get(f'/api/preview-clips/{job.id}/video')
            self.assertEqual(response.status_code, 200)
            self.assertIn('inline', response.headers.get('content-disposition', ''))
            job.purpose = 'export'
            self.assertEqual(self.client.get(f'/api/preview-clips/{job.id}/video').status_code, 404)


if __name__ == '__main__':
    unittest.main()
