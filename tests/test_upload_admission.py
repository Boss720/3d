"""ASGI-level concurrency and lifecycle coverage for upload admission."""

import asyncio
import importlib.metadata
import io
import json
from pathlib import Path
import re
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from upload_admission import UploadAdmissionMiddleware


def request_scope(path="/api/jobs", method="POST", scope_type="http"):
    return {"type": scope_type, "method": method, "path": path, "headers": []}


async def invoke(middleware, path="/api/jobs", method="POST", scope_type="http", receive=None):
    sent = []
    async def capture(message):
        sent.append(message)
    async def default_receive():
        return {"type": "http.request", "body": b"small", "more_body": False}
    await middleware(request_scope(path, method, scope_type), receive or default_receive, capture)
    return sent


class UploadAdmissionTests(unittest.IsolatedAsyncioTestCase):
    def test_supported_starlette_floor_is_declared_for_both_install_modes(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('requirements.txt', 'requirements-gpu.txt'):
            with self.subTest(requirements=name):
                text = (root / name).read_text(encoding='utf-8')
                match = re.search(r'^starlette>=([0-9]+)\.([0-9]+)', text, re.MULTILINE)
                self.assertIsNotNone(match, f'{name} must constrain the multipart parser version')
                self.assertGreaterEqual(tuple(map(int, match.groups())), (1, 8))

    async def test_real_multipart_parser_closes_partial_files_on_disconnect_and_cancel(self):
        version = importlib.metadata.version('starlette')
        version_match = re.match(r'([0-9]+)\.([0-9]+)', version)
        self.assertIsNotNone(version_match, f'unrecognized Starlette version: {version}')
        if tuple(map(int, version_match.groups())) < (1, 8):
            self.skipTest(f'Starlette {version} predates the enforced parser-cleanup minimum 1.8.0')

        from starlette.datastructures import Headers
        from starlette.formparsers import MultiPartParser
        from starlette.requests import ClientDisconnect
        import starlette.formparsers as formparsers

        chunk = (b'--abc\r\nContent-Disposition: form-data; name="file"; filename="x.bin"\r\n'
                 b'Content-Type: application/octet-stream\r\n\r\npartial')
        for failure in (ClientDisconnect(), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                spools = []

                def tracked_spool(*args, **kwargs):
                    del args, kwargs
                    spool = io.BytesIO()
                    spools.append(spool)
                    return spool

                async def incomplete_stream():
                    yield chunk
                    raise failure

                headers = Headers({'content-type': 'multipart/form-data; boundary=abc'})
                with patch.object(formparsers, 'SpooledTemporaryFile', side_effect=tracked_spool):
                    parser = MultiPartParser(headers, incomplete_stream())
                    with self.assertRaises(type(failure)):
                        await parser.parse()
                self.assertEqual(len(spools), 1)
                self.assertTrue(spools[0].closed, 'failed multipart parsing must close its partial upload spool')

    async def test_existing_body_limit_middleware_baseline_admits_five_downstream(self):
        # Regression baseline: the old byte limiter leaves concurrency unbounded.
        import main

        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()

        async def parser(scope, receive, send):
            nonlocal entered
            entered += 1
            if entered == 5:
                all_entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"invalid multipart"})

        old_limiter = main.RequestBodyLimitMiddleware(parser)
        tasks = [asyncio.create_task(invoke(old_limiter)) for _ in range(5)]
        await asyncio.wait_for(all_entered.wait(), 2)
        self.assertEqual(entered, 5)
        print("BASELINE RequestBodyLimitMiddleware: 5 of 5 requests entered the downstream multipart parser concurrently")
        release.set()
        await asyncio.gather(*tasks)

    async def test_four_streaming_uploads_admitted_fifth_rejected_before_receive(self):
        entered = 0
        all_entered = asyncio.Event()
        release = asyncio.Event()
        admission = UploadAdmissionMiddleware(None, max_inflight=4)
        receive_calls = 0

        async def streaming_parser(scope, receive, send):
            nonlocal entered
            entered += 1
            if entered == 4:
                all_entered.set()
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            await release.wait()

        admission.app = streaming_parser

        async def body_receive():
            nonlocal receive_calls
            receive_calls += 1
            return {"type": "http.request", "body": b"chunk", "more_body": True}

        tasks = [asyncio.create_task(invoke(admission, receive=body_receive)) for _ in range(4)]
        await asyncio.wait_for(all_entered.wait(), 2)
        admitted_receive_calls = receive_calls
        rejected = await invoke(admission, receive=body_receive)
        self.assertEqual(rejected[0]["status"], 429)
        self.assertIn(b"retry shortly", rejected[1]["body"])
        self.assertEqual(receive_calls, admitted_receive_calls, "rejected uploads must not read the body")
        self.assertEqual(entered, 4)
        print("ADMISSION UploadAdmissionMiddleware: 4 requests held; request 5 returned 429 without downstream entry or body receive")
        release.set()
        await asyncio.gather(*tasks)

    async def test_bypass_routes_and_non_http_without_changing_admission(self):
        calls = 0
        async def app(scope, receive, send):
            nonlocal calls
            calls += 1

        admission = UploadAdmissionMiddleware(app, max_inflight=1)
        for path, method, scope_type in (
            ("/api/jobs/abc/cancel", "POST", "http"),
            ("/api/jobs", "GET", "http"),
            ("/api/jobs", "POST", "websocket"),
        ):
            await invoke(admission, path, method, scope_type)
        self.assertEqual(calls, 3)

    async def test_custom_paths_and_constructor_validation(self):
        admission = UploadAdmissionMiddleware(lambda *args: None, paths={"/custom"}, max_inflight=2)
        self.assertEqual(admission.paths, frozenset({"/custom"}))
        for value in (0, -1, True, 1.5, "4"):
            with self.subTest(max_inflight=value), self.assertRaises(ValueError):
                UploadAdmissionMiddleware(None, max_inflight=value)
        for value, exception in (("/custom", TypeError), ({"bad"}, ValueError), ({"/a?b"}, ValueError)):
            with self.subTest(paths=value), self.assertRaises(exception):
                UploadAdmissionMiddleware(None, paths=value)

    async def test_releases_after_downstream_exception_413_and_disconnect(self):
        admission = UploadAdmissionMiddleware(None, max_inflight=1)

        async def raises(scope, receive, send):
            raise RuntimeError("parser failed")
        admission.app = raises
        with self.assertRaisesRegex(RuntimeError, "parser failed"):
            await invoke(admission)

        async def responds_413(scope, receive, send):
            await send({"type": "http.response.start", "status": 413, "headers": []})
            await send({"type": "http.response.body", "body": b"too large"})
        admission.app = responds_413
        self.assertEqual((await invoke(admission))[0]["status"], 413)

        async def disconnects(scope, receive, send):
            self.assertEqual((await receive())["type"], "http.disconnect")
        admission.app = disconnects
        async def disconnect_receive():
            return {"type": "http.disconnect"}
        await invoke(admission, receive=disconnect_receive)

        async def succeeds(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
        admission.app = succeeds
        self.assertEqual((await invoke(admission))[0]["status"], 200)

    async def test_releases_after_cancellation(self):
        admission = UploadAdmissionMiddleware(None, max_inflight=1)
        entered = asyncio.Event()
        async def waiting(scope, receive, send):
            entered.set()
            await asyncio.Event().wait()
        admission.app = waiting
        task = asyncio.create_task(invoke(admission))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        async def succeeds(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
        admission.app = succeeds
        self.assertEqual((await invoke(admission))[0]["status"], 200)

    async def test_shared_admission_across_threads_and_event_loops(self):
        admission = UploadAdmissionMiddleware(None, max_inflight=1)
        entered = threading.Event()
        release = threading.Event()
        async def waiting(scope, receive, send):
            entered.set()
            await asyncio.to_thread(release.wait)
        admission.app = waiting

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(asyncio.run, invoke(admission))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            second = await asyncio.to_thread(asyncio.run, invoke(admission))
            self.assertEqual(second[0]["status"], 429)
            release.set()
            await asyncio.to_thread(first.result, 2)


if __name__ == "__main__":
    unittest.main()
