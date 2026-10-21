"""Admission reservations bound queued uploads and retained session metadata."""
import unittest
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import cv2
import numpy as np

import main
from job_queue import EditableJobQueue


class AdmissionLimitTests(unittest.TestCase):
    def setUp(self):
        for name, value in (('JOBS', {}), ('STAGED_UPLOADS', set()),
                            ('JOB_QUEUE', EditableJobQueue(maxsize=2)),
                            ('MAX_PENDING_JOBS', 2), ('MAX_STAGED_UPLOADS', 2),
                            ('MAX_SESSION_JOBS', 3)):
            context = patch.object(main, name, value, create=True)
            context.start()
            self.addCleanup(context.stop)

    def test_http_session_limit_rejects_before_creating_an_upload(self):
        main.JOBS.update({str(index): SimpleNamespace(state='completed') for index in range(3)})
        _, png = cv2.imencode('.png', np.full((8, 8, 3), 128, np.uint8))
        before = set(main.UPLOADS.iterdir())
        client = TestClient(main.app, base_url="http://127.0.0.1:8765")
        try:
            with patch.object(main, 'ensure_worker'), patch.object(main, 'HISTORY', None), \
                    patch.object(main, 'ACCEPTING_MUTATIONS', True):
                response = client.post('/api/jobs', files={'file': ('tiny.png', png.tobytes(), 'image/png')})
            self.assertEqual(response.status_code, 429, response.text)
            self.assertEqual(set(main.UPLOADS.iterdir()), before)
        finally:
            client.close()
            for path in set(main.UPLOADS.iterdir()) - before:
                path.unlink()

    def assert_full(self):
        before = set(main.STAGED_UPLOADS)
        with self.assertRaises(HTTPException) as error:
            main.reserve_job_upload(Path('rejected'))
        self.assertEqual(error.exception.status_code, 429)
        self.assertEqual(main.STAGED_UPLOADS, before)

    def test_real_app_admission_wraps_body_parser_and_byte_limits(self):
        from upload_admission import UploadAdmissionMiddleware
        middleware = main.app.build_middleware_stack()
        layers = []
        gate = None
        while hasattr(middleware, 'app'):
            layers.append(type(middleware))
            if isinstance(middleware, UploadAdmissionMiddleware):
                gate = middleware
            middleware = middleware.app
        self.assertIsNotNone(gate)
        self.assertEqual(gate.max_inflight, 4)
        self.assertEqual(gate.paths, frozenset(main.REQUEST_BODY_LIMITS))
        self.assertLess(layers.index(UploadAdmissionMiddleware), layers.index(main.RequestBodyLimitMiddleware))

    def test_staged_uploads_reserve_pending_slots(self):
        main.reserve_job_upload(Path('first'))
        main.reserve_job_upload(Path('second'))
        self.assert_full()
        main.STAGED_UPLOADS.remove(Path('first'))
        main.reserve_job_upload(Path('third'))

    def test_pending_jobs_count_toward_admission_before_upload(self):
        main.JOB_QUEUE.put_nowait('queued')
        main.reserve_job_upload(Path('first'))
        self.assert_full()

    def test_terminal_jobs_still_count_toward_session_metadata_limit(self):
        main.JOBS.update({str(index): SimpleNamespace(state='completed') for index in range(3)})
        self.assert_full()

    def test_staged_limit_independent_of_pending_capacity(self):
        with patch.object(main, 'MAX_PENDING_JOBS', 100), \
                patch.object(main, 'MAX_STAGED_UPLOADS', 1):
            main.reserve_job_upload(Path('first'))
            self.assert_full()
