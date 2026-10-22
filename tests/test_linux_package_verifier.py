"""Portable checks for the Linux package acceptance request encoder."""
from email import policy
from email.parser import BytesParser
from pathlib import Path
import json
import sys
import signal
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from verify_linux_package import multipart, assert_bundled_ui, assert_clean_shutdown, wait_for_job


class LinuxPackageVerifierTests(unittest.TestCase):
    def test_waiting_for_active_work_rejects_an_already_completed_job(self):
        process = SimpleNamespace(poll=lambda: None)
        payload = json.dumps([{'id': 'job', 'state': 'completed', 'frames_processed': 8}]).encode()
        with patch('verify_linux_package.fetch', return_value=(200, payload)):
            with self.assertRaisesRegex(RuntimeError, 'before the required running state'):
                wait_for_job('http://localhost', 'job', process, 1,
                             ready=lambda job: job.get('state') == 'processing')

    def test_waiting_for_job_observes_transition_and_rejects_process_death(self):
        process = SimpleNamespace(poll=lambda: None)
        responses = [(200, json.dumps([{'id': 'job', 'state': state}]).encode())
                     for state in ('processing', 'canceled')]
        with patch('verify_linux_package.fetch', side_effect=responses), patch('verify_linux_package.time.sleep'):
            self.assertEqual(wait_for_job('http://localhost', 'job', process, 1)['state'], 'canceled')
        with self.assertRaisesRegex(RuntimeError, 'process exited'):
            wait_for_job('http://localhost', 'job', SimpleNamespace(poll=lambda: 1), 1)

    def test_shutdown_requires_cleanup_evidence_even_for_expected_signal_exit(self):
        marker = 'StereoLab shutdown complete.'
        for code in (0, -signal.SIGTERM):
            assert_clean_shutdown(code, marker)
            with self.assertRaisesRegex(RuntimeError, 'cleanup'):
                assert_clean_shutdown(code, 'server started')
        with self.assertRaisesRegex(RuntimeError, 'exit'):
            assert_clean_shutdown(-9, marker)

    def test_bundled_ui_uses_pyinstaller_internal_directory(self):
        package = Path(__file__).resolve().parents[1] / '.runtime' / 'linux-verifier-tests' / uuid.uuid4().hex
        package.mkdir(parents=True)
        try:
            static = package / '_internal' / 'static'
            static.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'index.html'):
                assert_bundled_ui(package)
            (static / 'index.html').write_text('StereoLab', encoding='utf-8')
            assert_bundled_ui(package)
        finally:
            quarantine = package.parents[1] / '_quarantine' / 'linux-verifier-tests' / package.name
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            package.rename(quarantine)

    def test_multipart_upload_preserves_form_fields_and_binary_media(self):
        media = b'\x00\xff\r\n\x80generated-video-bytes'
        body, boundary = multipart({'depth_mode': 'depth-anything-v2-small', 'execution_mode': 'cpu'},
                                   'sample.mp4', 'video/mp4', media)
        message = BytesParser(policy=policy.default).parsebytes(
            f'Content-Type: multipart/form-data; boundary={boundary}\r\n'
            'MIME-Version: 1.0\r\n\r\n'.encode('ascii') + body)

        self.assertTrue(message.is_multipart())
        fields = {}
        uploaded = None
        for part in message.iter_parts():
            name = part.get_param('name', header='content-disposition')
            if name == 'file':
                uploaded = part
            else:
                fields[name] = part.get_payload(decode=True).decode('utf-8')

        self.assertEqual(fields, {'depth_mode': 'depth-anything-v2-small', 'execution_mode': 'cpu'})
        self.assertIsNotNone(uploaded)
        self.assertEqual(uploaded.get_filename(), 'sample.mp4')
        self.assertEqual(uploaded.get_content_type(), 'video/mp4')
        self.assertEqual(uploaded.get_payload(decode=True), media)


if __name__ == '__main__':
    unittest.main()
