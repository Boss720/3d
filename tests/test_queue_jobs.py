"""Queue mutation semantics and dequeue races, without inference or servers."""
import concurrent.futures
import json
import queue
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from job_queue import EditableJobQueue


SETTINGS = ('layout', 'strength', 'convergence', 'depth_mode', 'input_size', 'resolution',
            'quality', 'invert_depth', 'temporal_smoothing', 'edge_refinement', 'execution_mode', 'depth_interval', 'video_decode', 'use_depth_cache', 'temporal_depth_scale', 'renderer')


def snapshot(job):
    return {name: getattr(job, name) for name in SETTINGS}


class QueueJobTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'queue-tests' / uuid.uuid4().hex
        self.uploads = self.root / 'uploads'
        self.uploads.mkdir(parents=True)
        self.addCleanup(self.cleanup)
        for name, value in (('JOBS', {}), ('JOB_QUEUE', type(main.JOB_QUEUE)()), ('UPLOADS', self.uploads)):
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)

    def cleanup(self):
        for folder in (self.uploads, self.root):
            for path in folder.iterdir():
                if path.is_file():
                    path.unlink()
            folder.rmdir()

    def job(self, purpose='export', **changes):
        job_id = str(uuid.uuid4())
        source = self.uploads / f'{job_id}.png'
        source.write_bytes(b'owned-upload')
        values = dict(id=job_id, filename='source.png', input_path=source, media_type='image',
                      layout='SBS', strength=5, convergence=.5, depth_mode='balanced', input_size=256,
                      resolution='source', quality='fast', invert_depth=False, purpose=purpose)
        values.update(changes)
        job = main.Job(**values)
        main.JOBS[job.id] = job
        main.JOB_QUEUE.put(job.id)
        return job

    def post_settings(self, job, settings):
        return self.client.post(f'/api/jobs/{job.id}/settings',
                                content=json.dumps({'settings': settings}), headers={'Content-Type': 'application/json'})

    def listed(self):
        return {item['id']: item for item in self.client.get('/api/jobs').json()}

    def test_capability_and_public_snapshot_positions_follow_actual_pending_queue(self):
        caps = self.client.get('/api/status').json()['queue_capabilities']
        self.assertEqual(caps, {'schema_version': 1, 'reorder_queued_exports': True,
                               'edit_queued_exports': True, 'remove_queued_exports': True, 'settings_snapshot': True})
        a, clip, b = self.job(), self.job('preview_clip'), self.job()
        listed = self.listed()
        for position, job in enumerate((a, clip, b), 1):
            self.assertEqual(listed[job.id]['queue_position'], position)
            self.assertEqual(listed[job.id]['settings_schema_version'], 1)
            self.assertEqual(listed[job.id]['settings'], snapshot(job))
            self.assertTrue(listed[job.id]['source_available'])
        self.assertEqual(main.JOB_QUEUE.get_nowait(), a.id)
        listed = self.listed()
        self.assertIsNone(listed[a.id]['queue_position'])
        self.assertEqual(listed[clip.id]['queue_position'], 1)
        self.assertEqual(listed[b.id]['queue_position'], 2)
        main.JOB_QUEUE.task_done()

    def test_reorder_is_atomic_and_preserves_clip_slots_and_unfinished_count(self):
        a, clip, b, c = self.job(), self.job('preview_clip'), self.job(), self.job()
        before = main.JOB_QUEUE.unfinished_tasks
        response = self.client.post('/api/queue/reorder', json={'job_ids': [c.id, b.id, a.id]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {'job_ids': [c.id, b.id, a.id]})
        self.assertEqual([main.JOB_QUEUE.get_nowait() for _ in range(4)], [c.id, clip.id, b.id, a.id])
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, before)
        for _ in range(4):
            main.JOB_QUEUE.task_done()
        main.JOB_QUEUE.join()

    def test_stale_reorders_and_malformed_ids_leave_order_unchanged(self):
        a, b = self.job(), self.job()
        for body, status in (({'job_ids': [a.id]}, 409), ({'job_ids': [a.id, a.id]}, 400),
                             ({'job_ids': 'bad'}, 400), ({'job_ids': [a.id, 1]}, 400),
                             ({'job_ids': [a.id, b.id], 'extra': True}, 400)):
            with self.subTest(body=body):
                response = self.client.post('/api/queue/reorder', json=body)
                self.assertEqual(response.status_code, status, response.text)
                self.assertEqual(main.JOB_QUEUE.qsize(), 2)
        self.assertEqual([main.JOB_QUEUE.get_nowait() for _ in range(2)], [a.id, b.id])

    def test_dequeued_job_cannot_be_edited_removed_or_included_in_new_order(self):
        a, b = self.job(), self.job()
        self.assertEqual(main.JOB_QUEUE.get_nowait(), a.id)
        original = snapshot(a)
        self.assertEqual(a.state, 'queued')  # Worker has not published processing yet.
        self.assertEqual(self.post_settings(a, {**original, 'strength': 9}).status_code, 409)
        self.assertEqual(self.client.post(f'/api/jobs/{a.id}/remove-queued').status_code, 409)
        self.assertEqual(self.client.post('/api/queue/reorder', json={'job_ids': [b.id, a.id]}).status_code, 409)
        self.assertEqual(snapshot(a), original)
        self.assertTrue(a.input_path.exists())

    def test_complete_settings_edit_changes_only_selected_pending_export(self):
        a, b = self.job(), self.job()
        original_b = snapshot(b)
        changed = {**snapshot(a), 'layout': 'RGBD', 'input_size': 384, 'resolution': '1080p', 'quality': 'high',
                   'strength': 9, 'convergence': .3, 'invert_depth': True,
                   'temporal_smoothing': .7, 'edge_refinement': False, 'execution_mode': 'cpu'}
        response = self.post_settings(a, changed)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['settings'], changed)
        self.assertEqual(snapshot(a), changed)
        self.assertEqual(snapshot(b), original_b)
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 2)

    def test_strict_settings_reject_invalid_values_and_leave_snapshot_intact(self):
        job = self.job()
        original = snapshot(job)
        invalid = [{**original, 'extra': 1}, {key: value for key, value in original.items() if key != 'quality'}]
        for key, value in (('layout', 'wrong'), ('input_size', True), ('input_size', 518.0), ('input_size', 1024),
                           ('strength', True), ('strength', 10 ** 400), ('strength', float('nan')), ('convergence', float('inf')),
                           ('temporal_smoothing', -.1), ('invert_depth', 'true'), ('edge_refinement', 1),
                           ('depth_mode', 'unknown'), ('resolution', '4k'), ('quality', 'maximum'),
                           ('execution_mode', 'cuda')):
            invalid.append({**original, key: value})
        for settings in invalid:
            with self.subTest(settings=settings):
                response = self.post_settings(job, settings)
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(snapshot(job), original)

    def test_readiness_failure_rolls_back_every_setting(self):
        job = self.job()
        original = snapshot(job)
        with patch.object(main, 'validate_depth_mode', side_effect=main.HTTPException(status_code=409, detail='model unavailable')):
            response = self.post_settings(job, {**original, 'depth_mode': 'depth-anything-v2-large', 'strength': 9})
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(snapshot(job), original)
        self.assertEqual(main.JOB_QUEUE.qsize(), 1)

    def test_dequeue_during_readiness_recheck_rejects_edit_without_holding_queue_lock(self):
        job = self.job()
        original = snapshot(job)
        entered, release = threading.Event(), threading.Event()
        def readiness(*args):
            entered.set()
            self.assertTrue(release.wait(3), 'test readiness was never released')
        with patch.object(main, 'validate_depth_mode', side_effect=readiness), concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(self.post_settings, job, {**original, 'strength': 9})
            try:
                self.assertTrue(entered.wait(2), 'request did not reach readiness validation')
                claimed = pool.submit(main.JOB_QUEUE.get_nowait).result(timeout=1)
                self.assertEqual(claimed, job.id)
            finally:
                release.set()
            response = future.result(timeout=2)
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(snapshot(job), original)

    def test_remove_pending_cleans_owned_upload_and_does_not_break_join(self):
        a, b = self.job(), self.job()
        response = self.client.post(f'/api/jobs/{a.id}/remove-queued')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['state'], 'canceled')
        self.assertFalse(response.json()['source_available'])
        self.assertIsNone(response.json()['queue_position'])
        self.assertFalse(a.input_path.exists())
        self.assertTrue(b.input_path.exists())
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 1)
        joined = threading.Event()
        thread = threading.Thread(target=lambda: (main.JOB_QUEUE.join(), joined.set()), daemon=True)
        thread.start()
        self.assertFalse(joined.wait(.02))
        self.assertEqual(main.JOB_QUEUE.get_nowait(), b.id)
        main.JOB_QUEUE.task_done()
        self.assertTrue(joined.wait(1))
        thread.join(timeout=1)

    def test_remove_preserves_unowned_file_and_reports_source_unavailable(self):
        foreign = self.root / 'not-an-upload.png'
        foreign.write_bytes(b'other-owner')
        job = self.job(input_path=foreign)
        self.assertFalse(self.listed()[job.id]['source_available'])
        response = self.client.post(f'/api/jobs/{job.id}/remove-queued')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(foreign.read_bytes(), b'other-owner')

    def test_clips_running_and_terminal_jobs_reject_edits_and_removal(self):
        for settings in ({'purpose': 'preview_clip'}, {'state': 'processing'}, {'state': 'completed'}, {'state': 'failed'}):
            job = self.job(**settings)
            with self.subTest(settings=settings):
                self.assertEqual(self.post_settings(job, snapshot(job)).status_code, 409)
                self.assertEqual(self.client.post(f'/api/jobs/{job.id}/remove-queued').status_code, 409)
        self.assertEqual(self.client.post('/api/jobs/missing/remove-queued').status_code, 404)

    def test_existing_cancel_keeps_worker_task_accounting_and_hides_canceled_position(self):
        a, b = self.job(), self.job()
        response = self.client.post(f'/api/jobs/{a.id}/cancel')
        self.assertEqual(response.status_code, 200)
        listed = self.listed()
        self.assertIsNone(listed[a.id]['queue_position'])
        self.assertEqual(listed[b.id]['queue_position'], 1)
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 2)
        reorder = self.client.post('/api/queue/reorder', json={'job_ids': [b.id]})
        self.assertEqual(reorder.status_code, 200, reorder.text)

    def test_remove_cleanup_failure_rolls_back_queue_and_job(self):
        job = self.job()
        original_unlink = Path.unlink
        def protected(path, *args, **kwargs):
            if path == job.input_path:
                raise PermissionError('locked upload')
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, 'unlink', protected):
            response = self.client.post(f'/api/jobs/{job.id}/remove-queued')
        self.assertEqual(response.status_code, 409, response.text)
        self.assertEqual(job.state, 'queued')
        self.assertFalse(job.cancel_event.is_set())
        self.assertEqual(main.JOB_QUEUE.pending_ids(), (job.id,))
        self.assertEqual(main.JOB_QUEUE.unfinished_tasks, 1)
        self.assertTrue(job.input_path.is_file())

    def test_gpu_approximation_edit_obeys_existing_device_policy_without_mutation(self):
        job = self.job()
        original = snapshot(job)
        response = self.post_settings(job, {**original, 'execution_mode': 'gpu'})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(snapshot(job), original)

    def test_terminal_source_absence_does_not_erase_settings(self):
        job = self.job(state='failed')
        original = snapshot(job)
        job.input_path.unlink()
        public = self.listed()[job.id]
        self.assertFalse(public['source_available'])
        self.assertIsNone(public['queue_position'])
        self.assertEqual(public['settings'], original)

    def test_unreadable_source_stat_is_best_effort_and_does_not_break_job_list(self):
        job = self.job()
        original_is_file = Path.is_file
        def inaccessible(path):
            if path == job.input_path:
                raise PermissionError('unreadable upload')
            return original_is_file(path)
        with patch.object(Path, 'is_file', inaccessible):
            response = self.client.get('/api/jobs')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()[0]['source_available'])

    def test_settings_envelope_is_exact(self):
        job = self.job()
        for body in ([], {}, {'settings': snapshot(job), 'extra': True}, {'settings': []}):
            with self.subTest(body=body):
                response = self.client.post(f'/api/jobs/{job.id}/settings', json=body)
                self.assertEqual(response.status_code, 400, response.text)

    def test_legacy_upload_still_creates_an_export_with_original_defaults(self):
        with patch.object(main, 'ensure_worker'):
            response = self.client.post('/api/jobs', data={'depth_mode': 'balanced'},
                                        files={'file': ('sample.png', b'uploaded-original', 'image/png')})
        self.assertEqual(response.status_code, 200, response.text)
        public = response.json()
        self.assertEqual(public['purpose'], 'export')
        self.assertEqual(public['layout'], 'SBS')
        self.assertEqual(public['input_size'], 518)
        self.assertEqual(public['strength'], 7.5)
        self.assertEqual(public['quality'], 'balanced')
        self.assertTrue(public['source_available'])
        self.assertEqual(public['queue_position'], 1)
        self.assertEqual(main.JOB_QUEUE.pending_ids(), (public['id'],))


class EditableQueueTests(unittest.TestCase):
    def test_get_uses_condition_wait_and_keeps_standard_timeout_semantics(self):
        work = EditableJobQueue()
        entered, received = threading.Event(), threading.Event()
        def consume():
            entered.set()
            item = work.get(timeout=1)
            received.set()
            return item
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(consume)
            self.assertTrue(entered.wait(1))
            self.assertFalse(received.wait(.02))
            work.put('a')
            self.assertEqual(future.result(timeout=1), 'a')
        work.task_done()
        work.join()
        with self.assertRaises(queue.Empty):
            work.get_nowait()
        with self.assertRaises(ValueError):
            work.get(timeout=-1)
        with self.assertRaises(ValueError):
            work.task_done()

    def test_remove_last_pending_task_notifies_join_waiters(self):
        work = EditableJobQueue()
        work.put('a')
        joined = threading.Event()
        thread = threading.Thread(target=lambda: (work.join(), joined.set()), daemon=True)
        thread.start()
        self.assertFalse(joined.wait(.02))
        work.remove_pending('a')
        self.assertTrue(joined.wait(1))
        thread.join(timeout=1)
        self.assertEqual(work.unfinished_tasks, 0)

    def test_remove_notifies_a_bounded_queue_producer(self):
        work = EditableJobQueue(maxsize=1)
        work.put('a')
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(work.put, 'b', True, 1)
            work.remove_pending('a')
            future.result(timeout=1)
        self.assertEqual(work.get_nowait(), 'b')
        work.task_done()
        work.join()

    def test_dequeue_and_reorder_share_one_lock(self):
        work = EditableJobQueue()
        for item in ('a', 'clip', 'b'):
            work.put(item)
        waiting = threading.Event()
        def consume():
            waiting.set()
            return work.get()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            with work.lock:
                future = pool.submit(consume)
                self.assertTrue(waiting.wait(1))
                self.assertFalse(future.done())
                work.reorder_subset(('b', 'a'), ('a', 'b'))
            self.assertEqual(future.result(timeout=1), 'b')
        self.assertEqual(work.pending_ids(), ('clip', 'a'))
        for _ in range(2):
            work.get_nowait()
        for _ in range(3):
            work.task_done()
        work.join()


class WorkerStartTests(unittest.TestCase):
    def test_concurrent_requests_start_exactly_one_worker(self):
        entered, release, second_done = threading.Event(), threading.Event(), threading.Event()
        starts = []
        class FakeWorker:
            def __init__(self, *args, **kwargs):
                self.alive = False
            def is_alive(self):
                return self.alive
            def start(self):
                starts.append(self)
                if len(starts) == 1:
                    entered.set()
                    if not release.wait(2):
                        raise RuntimeError('test did not release worker startup')
                self.alive = True
        # Construct the actual callers before replacing only the worker factory.
        first = threading.Thread(target=main.ensure_worker)
        second = threading.Thread(target=lambda: (main.ensure_worker(), second_done.set()))
        with patch.object(main, 'WORKER', None), patch.object(main.threading, 'Thread', FakeWorker):
            try:
                first.start()
                self.assertTrue(entered.wait(1))
                second.start()
                second_done.wait(.05)
            finally:
                release.set()
                first.join(timeout=1)
                second.join(timeout=1)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
        self.assertEqual(len(starts), 1)
