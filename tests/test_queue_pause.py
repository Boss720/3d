import threading
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main
from job_queue import EditableJobQueue


class QueuePauseTests(unittest.TestCase):
    def setUp(self):
        self.queue = EditableJobQueue()
        self.stop = threading.Event()
        for name, value in [('JOB_QUEUE', self.queue), ('JOBS', {}), ('WORKER_STOP', self.stop)]:
            context = patch.object(main, name, value)
            context.start()
            self.addCleanup(context.stop)
        self.client = TestClient(main.app, base_url='http://127.0.0.1')
        self.addCleanup(self.client.close)

    def worker(self, callback):
        context = patch.object(main, 'process_job', side_effect=callback)
        context.start()
        thread = threading.Thread(target=main.worker_loop)
        thread.start()
        def finish():
            self.stop.set()
            with self.queue.not_empty:
                self.queue.not_empty.notify_all()
            thread.join(2)
            context.stop()
            self.assertFalse(thread.is_alive(), 'Paused worker must stop without dequeueing')
        self.addCleanup(finish)
        return thread

    def control(self, paused):
        response = self.client.post('/api/queue/control', json={'paused': paused})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_empty_pause_new_admission_waits_and_resume_preserves_order(self):
        self.control(True)
        called = threading.Event()
        seen = []
        def process(identity):
            seen.append(identity)
            if identity == 'b':
                called.set()
        self.worker(process)
        self.queue.put('a')
        self.queue.put('b')
        self.assertFalse(called.wait(.08))
        self.assertEqual(self.queue.pending_ids(), ('a', 'b'))
        self.control(True)
        self.control(False)
        self.assertTrue(called.wait(2))
        self.queue.join()
        self.assertEqual(seen, ['a', 'b'])
        self.assertEqual(self.queue.unfinished_tasks, 0)

    def test_active_job_finishes_but_next_waits(self):
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def process(identity):
            if identity == 'a':
                entered.set()
                release.wait(2)
            else:
                second.set()
        self.queue.put('a')
        self.queue.put('b')
        self.worker(process)
        self.assertTrue(entered.wait(2))
        self.control(True)
        release.set()
        self.assertFalse(second.wait(.08))
        self.assertEqual(self.queue.pending_ids(), ('b',))
        self.control(False)
        self.assertTrue(second.wait(2))

    def test_shutdown_while_paused_preserves_pending_accounting(self):
        self.control(True)
        self.queue.put('a')
        thread = self.worker(lambda _: self.fail('Must not consume paused work'))
        self.stop.set()
        with self.queue.not_empty:
            self.queue.not_empty.notify_all()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.queue.pending_ids(), ('a',))
        self.assertEqual(self.queue.unfinished_tasks, 1)

    def test_invalid_payloads_and_readback(self):
        for payload in [{}, {'paused': 1}, {'paused': 'false'}, {'paused': True, 'extra': 1}]:
            self.assertEqual(self.client.post('/api/queue/control', json=payload).status_code, 400)
        self.assertEqual(self.client.get('/api/queue/control').json(),
                         {'paused': False, 'pending_count': 0, 'restart_behavior': 'running'})
