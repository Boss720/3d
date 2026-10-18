import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import main


class OutputOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.root = main.RUNTIME / 'output-ownership-tests' / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.outputs = patch.object(main, 'OUTPUTS', self.root)
        self.outputs.start()
        self.addCleanup(self.outputs.stop)
        self.addCleanup(self.cleanup)

    def cleanup(self):
        for path in self.root.iterdir():
            if path.is_file():
                path.unlink()
        self.root.rmdir()

    def job(self, job_id):
        path = self.root / f'{job_id}-input.png'
        cv2.imwrite(str(path), np.zeros((12, 16, 3), np.uint8))
        return main.Job(id=job_id, filename='same.png', input_path=path, media_type='image',
                        layout='SBS', strength=5, convergence=.5, depth_mode='balanced',
                        input_size=256, resolution='source', quality='fast', invert_depth=False)

    def test_distinct_full_ids_with_same_prefix_have_distinct_names(self):
        a = self.job('12345678-1111-4000-8000-000000000001')
        b = self.job('12345678-1111-4000-8000-000000000002')
        self.assertNotEqual(main.output_path_for(a), main.output_path_for(b))

    def test_existing_output_is_preserved_without_inference_or_cleanup(self):
        job = self.job(str(uuid.uuid4()))
        target = main.output_path_for(job)
        target.write_bytes(b'preexisting-export')
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'estimate_depth') as infer:
            main.process_job(job.id)
        self.assertEqual(job.state, 'failed')
        self.assertIn('exist', job.error.lower())
        self.assertEqual(target.read_bytes(), b'preexisting-export')
        infer.assert_not_called()
        self.assertFalse(job.input_path.exists())

    def test_owned_reservation_is_removed_on_preflight_failure(self):
        job = self.job(str(uuid.uuid4()))
        with patch.dict(main.JOBS, {job.id: job}), patch.object(main, 'inspect_media', side_effect=RuntimeError('decode failure')):
            main.process_job(job.id)
        self.assertEqual(job.state, 'failed')
        self.assertFalse(main.output_path_for(job).exists())
        self.assertFalse(job.input_path.exists())


if __name__ == '__main__':
    unittest.main()
