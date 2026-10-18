"""ASGI history integration without inference servers or device access."""
import hashlib
import sqlite3
import threading
import unittest
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from job_history import HistoryStore, HistoryUnavailable, HistoryLeaseError


class HistoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'history-backend-verification' / uuid.uuid4().hex
        self.uploads, self.outputs = self.root / 'uploads', self.root / 'outputs'
        self.uploads.mkdir(parents=True)
        self.outputs.mkdir()
        for name, value in (('JOBS', {}), ('JOB_QUEUE', type(main.JOB_QUEUE)()),
                            ('UPLOADS', self.uploads), ('OUTPUTS', self.outputs), ('DATA_ROOT', self.root),
                            ('HISTORY', None), ('HISTORY_WARNING', None), ('WORKER', None),
                            ('WORKER_STOP', threading.Event()), ('ACCEPTING_MUTATIONS', True)):
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)

    def test_disabled_lifespan_owns_stoppable_worker_without_history_files(self):
        with patch.dict('os.environ', {'STEREOLAB_HISTORY_INSTANCE': ''}):
            with TestClient(main.app, base_url='http://127.0.0.1') as client:
                self.assertEqual(client.get('/api/status').json()['history_capabilities']['state'], 'disabled')
                worker = main.WORKER
                self.assertTrue(worker.is_alive())
            self.assertFalse(worker.is_alive())
            self.assertFalse((self.root / '.runtime' / 'history').exists())

    def test_preflight_capacity_precedes_application_upload_staging(self):
        store = HistoryStore(self.root, 'capacity2', max_records=1).open()
        self.addCleanup(store.close)
        store.admit(self.metadata())
        with patch.object(main, 'HISTORY', store), patch.object(main, 'ensure_worker'):
            response = TestClient(main.app, base_url='http://127.0.0.1').post('/api/jobs',
                files={'file': ('frame.png', b'upload', 'image/png')})
        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(list(self.uploads.iterdir()), [])
        self.assertEqual(main.JOBS, {})

    def open_store(self, instance='integration'):
        store = HistoryStore(self.root, instance).open()
        self.addCleanup(store.close)
        main.HISTORY = store
        return store

    def submit(self, payload=b'upload', **values):
        with patch.object(main, 'ensure_worker'):
            return TestClient(main.app, base_url='http://127.0.0.1').post('/api/jobs', data=values,
                files={'file': ('frame.png', payload, 'image/png')})

    def test_admission_hash_and_edits_precede_live_publication(self):
        store = self.open_store()
        original = store.admit
        def admit(metadata):
            self.assertNotIn(metadata['job_id'], main.JOBS)
            self.assertNotIn(metadata['job_id'], main.JOB_QUEUE.pending_ids())
            self.assertEqual(metadata['source_sha256'], hashlib.sha256(b'upload').hexdigest())
            self.assertEqual(metadata['source_size_bytes'], 6)
            return original(metadata)
        with patch.object(store, 'admit', side_effect=admit):
            response = self.submit()
        self.assertEqual(response.status_code, 200, response.text)
        job = main.JOBS[response.json()['id']]
        previous = main.settings_snapshot(job)
        original_edit = store.edit_pending
        def edit(job_id, **changes):
            self.assertEqual(main.settings_snapshot(job), previous)
            return original_edit(job_id, **changes)
        client = TestClient(main.app, base_url='http://127.0.0.1')
        with patch.object(store, 'edit_pending', side_effect=edit):
            response = client.post(f'/api/jobs/{job.id}/settings', json={'settings': {**previous, 'strength': 9}})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(job.strength, 9)
        self.assertEqual(store.list_prior()['items'], [])

    def test_write_failure_rolls_back_admission_and_edit_but_allows_remove(self):
        store = self.open_store()
        first = self.submit().json()
        job = main.JOBS[first['id']]
        previous = main.settings_snapshot(job)
        with patch.object(store, '_save', side_effect=sqlite3.OperationalError('disk failed')):
            response = TestClient(main.app, base_url='http://127.0.0.1').post(f'/api/jobs/{job.id}/settings',
                json={'settings': {**previous, 'strength': 9}})
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(main.settings_snapshot(job), previous)
        self.assertEqual(main.history_capabilities()['state'], 'degraded')
        self.assertEqual(self.submit().status_code, 503)
        response = TestClient(main.app, base_url='http://127.0.0.1').post(f'/api/jobs/{job.id}/remove-queued')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(job.state, 'canceled')
        self.assertTrue(response.json()['history_warning'])
        self.assertFalse(job.input_path.exists())

    def test_owned_source_unlink_failure_keeps_durable_and_live_queue_pending(self):
        store = self.open_store()
        response = self.submit()
        job = main.JOBS[response.json()['id']]
        with patch.object(type(job.input_path), 'unlink', side_effect=PermissionError('owned source busy')):
            response = TestClient(main.app, base_url='http://127.0.0.1').post(f'/api/jobs/{job.id}/remove-queued')
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(job.state, 'queued')
        self.assertIn(job.id, main.JOB_QUEUE.pending_ids())
        self.assertTrue(job.input_path.exists())
        # A successful queued edit proves the durable row was not terminalized.
        self.assertEqual(store.edit_pending(job.id, settings=main.settings_snapshot(job))['state'], 'queued')

    def test_commit_rechecks_capacity_after_concurrent_upload(self):
        store = HistoryStore(self.root, 'commit-capacity', max_records=1).open()
        self.addCleanup(store.close)
        main.HISTORY = store
        admit = store.admit
        def race(metadata):
            admit(self.metadata())
            return admit(metadata)
        with patch.object(store, 'admit', side_effect=race):
            response = self.submit()
        self.assertEqual(response.status_code, 429, response.text)
        self.assertEqual(main.JOBS, {})
        self.assertEqual(list(self.uploads.iterdir()), [])

    def test_invalid_instance_and_lease_contention_start_no_worker(self):
        with patch.dict('os.environ', {'STEREOLAB_HISTORY_INSTANCE': '../bad'}), patch.object(main, 'ensure_worker') as worker:
            with self.assertRaises(ValueError), TestClient(main.app, base_url='http://127.0.0.1'):
                pass
            worker.assert_not_called()
        self.assertFalse((self.root / '.runtime' / 'history').exists())
        store = self.open_store('owned')
        with patch.dict('os.environ', {'STEREOLAB_HISTORY_INSTANCE': 'owned'}), patch.object(main, 'ensure_worker') as worker:
            with self.assertRaises(HistoryLeaseError), TestClient(main.app, base_url='http://127.0.0.1'):
                pass
            worker.assert_not_called()
        self.assertEqual(store.status()['state'], 'ready')

    def test_admission_commit_failure_cleans_only_request_upload(self):
        store = self.open_store()
        unrelated = self.uploads / 'unrelated.png'
        unrelated.write_bytes(b'preserve')
        with patch.object(store, 'admit', side_effect=HistoryUnavailable('disk failed')):
            response = self.submit()
        self.assertEqual(response.status_code, 503, response.text)
        self.assertEqual(main.JOBS, {})
        self.assertEqual(main.JOB_QUEUE.pending_ids(), ())
        self.assertEqual(list(self.uploads.iterdir()), [unrelated])

    def test_prior_download_and_interruption_never_rehydrate_live_jobs(self):
        store = self.open_store()
        completed = store.admit(self.metadata())
        store.mark_processing(completed['job_id'])
        output = self.outputs / completed['planned_output_name']
        output.write_bytes(b'completed output')
        store.finish(completed['job_id'], 'completed', output={'name': output.name, 'size_bytes': output.stat().st_size})
        pending = store.admit(self.metadata())
        store.close()
        store.open()
        client = TestClient(main.app, base_url='http://127.0.0.1')
        response = client.get('/api/history')
        self.assertEqual(response.status_code, 200, response.text)
        rows = {row['job_id']: row for row in response.json()['items']}
        self.assertEqual(rows[pending['job_id']]['state'], 'interrupted')
        self.assertEqual(rows[pending['job_id']]['last_known_state'], 'queued')
        self.assertTrue(rows[completed['job_id']]['completed_at'])
        self.assertEqual(client.get(f"/api/history/{completed['job_id']}/download").content, b'completed output')
        self.assertEqual(main.JOBS, {})
        self.assertEqual(main.JOB_QUEUE.pending_ids(), ())
        output.write_bytes(b'changed')
        self.assertEqual(client.get(f"/api/history/{completed['job_id']}/download").status_code, 404)
        self.assertFalse(client.get('/api/history').json()['items'][0]['source_available'])

    def test_history_http_validation_and_disabled_response(self):
        client = TestClient(main.app, base_url='http://127.0.0.1')
        self.assertEqual(client.get('/api/history').status_code, 409)
        self.open_store()
        for query in ('limit=0', 'limit=101', 'limit=1.0', 'limit=no', 'before=bad', 'before='):
            self.assertEqual(client.get('/api/history?' + query).status_code, 400, query)
        self.assertEqual(client.get('/api/history/not-a-uuid/download').status_code, 404)

    def test_completion_failure_keeps_live_download_and_interrupted_history(self):
        import cv2
        import numpy as np
        store = self.open_store()
        payload = cv2.imencode('.png', np.zeros((12, 16, 3), np.uint8))[1].tobytes()
        response = self.submit(payload)
        self.assertEqual(response.status_code, 200, response.text)
        job = main.JOBS[response.json()['id']]
        main.JOB_QUEUE.get_nowait()
        with patch.object(store, 'finish', side_effect=HistoryUnavailable('completion failed')):
            main.process_job(job.id)
        main.JOB_QUEUE.task_done()
        self.assertEqual(job.state, 'completed', job.error)
        self.assertTrue(job.history_warning)
        client = TestClient(main.app, base_url='http://127.0.0.1')
        self.assertEqual(client.get(f'/api/jobs/{job.id}/download').status_code, 200)
        store.close()
        store.open()
        row = store.list_prior()['items'][0]
        self.assertEqual(row['state'], 'interrupted')
        self.assertFalse(row['output_available'])

    def test_shutdown_holds_lease_until_active_writer_exits_and_keeps_pending(self):
        entered, cancelled, release = threading.Event(), threading.Event(), threading.Event()
        def run(job, net):
            entered.set()
            self.assertTrue(job.cancel_event.wait(5))
            cancelled.set()
            self.assertTrue(release.wait(5))
            raise main.JobCancelled()
        with patch.dict('os.environ', {'STEREOLAB_HISTORY_INSTANCE': 'shutdown'}), patch.object(main, 'run_image', side_effect=run):
            client = TestClient(main.app, base_url='http://127.0.0.1')
            client.__enter__()
            payload = __import__('cv2').imencode('.png', __import__('numpy').zeros((12, 16, 3), __import__('numpy').uint8))[1].tobytes()
            response = client.post('/api/jobs', files={'file': ('active.png', payload, 'image/png')})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(entered.wait(5))
            pending = client.post('/api/jobs', files={'file': ('pending.png', payload, 'image/png')}).json()
            finished = threading.Event()
            shutdown = threading.Thread(target=lambda: (client.__exit__(None, None, None), finished.set()))
            shutdown.start()
            try:
                self.assertTrue(cancelled.wait(5))
                self.assertFalse(finished.is_set())
                self.assertIsNotNone(main.HISTORY._connection)
                with self.assertRaises(HistoryLeaseError):
                    HistoryStore(self.root, 'shutdown').open()
                self.assertIn(pending['id'], main.JOB_QUEUE.pending_ids())
            finally:
                release.set()
                shutdown.join(8)
            self.assertFalse(shutdown.is_alive())
            with HistoryStore(self.root, 'shutdown') as reopened:
                rows = {row['job_id']: row for row in reopened.list_prior()['items']}
                self.assertEqual(rows[pending['id']]['state'], 'interrupted')

    def metadata(self):
        return dict(job_id=str(uuid.uuid4()), created_at=1, filename='frame.png', media_type='image',
                    source_size_bytes=6, source_sha256=hashlib.sha256(b'upload').hexdigest(),
                    settings=dict(layout='SBS', strength=7.5, convergence=.5, depth_mode='balanced', input_size=518,
                                  resolution='source', quality='balanced', invert_depth=False,
                                  temporal_smoothing=.35, edge_refinement=True, execution_mode='auto'),
                    output_name_template='{name}_{layout}', build_fingerprint=None)
