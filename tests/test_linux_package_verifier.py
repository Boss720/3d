"""Portable checks for the Linux package acceptance request encoder."""
from email import policy
from email.parser import BytesParser
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))

from verify_linux_package import multipart


class LinuxPackageVerifierTests(unittest.TestCase):
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
