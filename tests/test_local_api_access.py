"""Host and browser-origin boundaries for the loopback-only API."""
from pathlib import Path
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


SERVICE_PORT = 8765


class LocalApiAccessTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app, base_url=f"http://127.0.0.1:{SERVICE_PORT}")

    def tearDown(self):
        self.client.close()

    def reset_post(self, headers=None, client=None):
        with patch.object(main.DEPTH_ENGINE, "reset_gpu_failures") as reset, \
             patch.object(main.PREVIEW_CACHE, "clear") as clear:
            response = (client or self.client).post("/api/runtime/gpu/reset", headers=headers or {})
        return response, reset, clear

    def test_loopback_aliases_are_allowed_only_on_the_same_port(self):
        for host, origin in (
            (f"127.0.0.1:{SERVICE_PORT}", f"http://localhost:{SERVICE_PORT}"),
            (f"[::1]:{SERVICE_PORT}", f"http://127.0.0.1:{SERVICE_PORT}"),
        ):
            with self.subTest(host=host, origin=origin):
                response, reset, clear = self.reset_post({"Host": host, "Origin": origin})
                self.assertEqual(response.status_code, 200, response.text)
                reset.assert_called_once()
                clear.assert_called_once()

    def test_default_http_port_is_equivalent_when_it_is_the_request_port(self):
        with TestClient(main.app, base_url="http://127.0.0.1") as client:
            with patch.object(main.DEPTH_ENGINE, "reset_gpu_failures"), patch.object(main.PREVIEW_CACHE, "clear"):
                response = client.post("/api/runtime/gpu/reset", headers={"Origin": "http://localhost"})
        self.assertEqual(response.status_code, 200, response.text)

    def test_default_or_wrong_origin_port_does_not_match_service_port(self):
        for origin in ("http://localhost", f"http://localhost:{SERVICE_PORT + 1}"):
            with self.subTest(origin=origin):
                response, reset, clear = self.reset_post({"Origin": origin})
                self.assertEqual(response.status_code, 403, response.text)
                reset.assert_not_called()
                clear.assert_not_called()

    def test_null_and_foreign_origins_cannot_mutate(self):
        for origin in ("null", "https://attacker.example"):
            with self.subTest(origin=origin):
                response, reset, clear = self.reset_post({"Origin": origin})
                self.assertEqual(response.status_code, 403, response.text)
                reset.assert_not_called()
                clear.assert_not_called()

    def test_cross_site_fetch_without_origin_cannot_mutate(self):
        response, reset, clear = self.reset_post({"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(response.status_code, 403, response.text)
        reset.assert_not_called()
        clear.assert_not_called()

    def test_cli_mutation_without_origin_or_fetch_metadata_remains_allowed(self):
        response, reset, clear = self.reset_post()
        self.assertEqual(response.status_code, 200, response.text)
        reset.assert_called_once()
        clear.assert_called_once()

    def test_foreign_origin_is_allowed_for_safe_status_get(self):
        with patch.object(main.DEPTH_ENGINE, "runtime_info", return_value={"available": True}):
            response = self.client.get("/api/status", headers={"Origin": "https://attacker.example"})
        self.assertEqual(response.status_code, 200, response.text)

    def test_foreign_hosts_and_localhost_lookalikes_are_rejected(self):
        for host in (
            f"attacker.example:{SERVICE_PORT}",
            f"evil.localhost.attacker:{SERVICE_PORT}",
        ):
            with self.subTest(host=host):
                response = self.client.get("/api/status", headers={"Host": host})
                self.assertEqual(response.status_code, 403, response.text)

    def test_credentials_and_malformed_authorities_are_rejected_without_server_error(self):
        cases = (
            ({"Origin": f"http://user:secret@localhost:{SERVICE_PORT}"}, None),
            ({"Origin": f"http://[::1:{SERVICE_PORT}"}, None),
            ({"Host": f"user:secret@localhost:{SERVICE_PORT}"}, None),
            ({"Host": f"::1:{SERVICE_PORT}"}, None),
            ({"Host": "127.0.0.1:invalid"}, None),
            ({"Host": "127.0.0.1:", "Origin": "http://localhost"}, None),
            ({"Origin": "http://localhost:0"}, "http://127.0.0.1"),
        )
        for headers, base_url in cases:
            with self.subTest(headers=headers, base_url=base_url):
                client = TestClient(main.app, base_url=base_url or f"http://127.0.0.1:{SERVICE_PORT}")
                try:
                    response, reset, clear = self.reset_post(headers, client=client)
                    self.assertEqual(response.status_code, 403, response.text)
                    reset.assert_not_called()
                    clear.assert_not_called()
                finally:
                    client.close()

    def test_rejected_cancel_does_not_interrupt_an_active_job(self):
        job = main.Job(
            id="busy", filename="test.mp4", input_path=Path("unused"), media_type="video",
            layout="SBS", strength=5, convergence=0.5, depth_mode="balanced",
            input_size=256, resolution="source", quality="fast", invert_depth=False,
            state="processing",
        )
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, "stop_process") as stop:
            response = self.client.post(
                f"/api/jobs/{job.id}/cancel",
                headers={"Origin": "https://attacker.example", "Sec-Fetch-Site": "cross-site"},
            )
        self.assertEqual(response.status_code, 403, response.text)
        self.assertFalse(job.cancel_event.is_set())
        stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
