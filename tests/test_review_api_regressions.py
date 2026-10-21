"""Review regressions for custom model upload staging ownership."""
import asyncio
import io
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile
import main


class ModelUploadReviewTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'review-api-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(self.quarantine)

    def quarantine(self):
        destination = main.RUNTIME / '_quarantine' / 'review-api-tests' / self.root.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.root.replace(destination)

    def test_cancelled_model_copy_removes_private_candidate_and_closes_upload(self):
        class CancelledUpload(UploadFile):
            reads = 0
            async def read(self, size=-1):
                self.reads += 1
                if self.reads == 1:
                    return b'partial-model'
                raise asyncio.CancelledError()
        upload = CancelledUpload(filename='model.onnx', file=io.BytesIO())
        with patch.object(main, 'MODELS', self.root):
            with self.assertRaises(asyncio.CancelledError):
                asyncio.run(main.upload_model(upload))
        self.assertTrue(upload.file.closed)
        self.assertEqual(list(self.root.iterdir()), [], 'Cancelled model must not retain a permanent candidate')

    def test_publish_failure_preserves_installed_model_and_removes_candidate(self):
        installed = self.root / 'depth.onnx'
        installed.write_bytes(b'previous-model')
        upload = UploadFile(filename='model.onnx', file=io.BytesIO(b'new-model'))
        with patch.object(main, 'MODELS', self.root), patch.object(main, 'MODEL_PATH', installed), \
                patch.object(main.cv2.dnn, 'readNetFromONNX', return_value=object()), \
                patch.object(main.os, 'replace', side_effect=PermissionError('model is locked')):
            with self.assertRaises(PermissionError):
                asyncio.run(main.upload_model(upload))
        self.assertTrue(upload.file.closed)
        self.assertEqual(installed.read_bytes(), b'previous-model')
        self.assertEqual(list(self.root.iterdir()), [installed], 'Failed model must not retain a permanent candidate')

    def test_valid_model_atomically_replaces_installed_model_without_candidate(self):
        installed = self.root / 'depth.onnx'
        installed.write_bytes(b'previous-model')
        upload = UploadFile(filename='model.onnx', file=io.BytesIO(b'new-model'))
        with patch.object(main, 'MODELS', self.root), patch.object(main, 'MODEL_PATH', installed), \
                patch.object(main.cv2.dnn, 'readNetFromONNX', return_value=object()):
            result = asyncio.run(main.upload_model(upload))
        self.assertTrue(result['model_available'])
        self.assertTrue(upload.file.closed)
        self.assertEqual(installed.read_bytes(), b'new-model')
        self.assertEqual(list(self.root.iterdir()), [installed])

    def test_existing_candidate_is_never_overwritten_or_removed(self):
        candidate = self.root / 'collision.onnx'
        candidate.write_bytes(b'unrelated-model')
        upload = UploadFile(filename='model.onnx', file=io.BytesIO(b'new-model'))
        with patch.object(main, 'MODELS', self.root), patch.object(main.uuid, 'uuid4', return_value='collision'):
            with self.assertRaises(FileExistsError):
                asyncio.run(main.upload_model(upload))
        self.assertTrue(upload.file.closed)
        self.assertEqual(candidate.read_bytes(), b'unrelated-model')
