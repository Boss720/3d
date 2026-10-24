import unittest
from pathlib import Path

from job_settings import settings_snapshot, validate_settings
from main import Job


class DepthCadenceSettingsTests(unittest.TestCase):
    def settings(self):
        job = Job(id='example', filename='example.mp4', input_path=Path('example.mp4'),
                  media_type='video', layout='SBS', strength=5, convergence=.5,
                  depth_mode='depth-anything-v2-small', input_size=256,
                  resolution='source', quality='fast', invert_depth=False)
        return settings_snapshot(job)

    def test_legacy_complete_settings_default_to_every_frame(self):
        settings = self.settings()
        settings.pop('depth_interval')
        settings.pop('video_decode')
        result = validate_settings(settings)
        self.assertEqual(result['depth_interval'], 1)
        self.assertEqual(result['video_decode'], 'software')
        self.assertNotIn('depth_interval', settings)

    def test_alternate_depth_and_fp16_roundtrip(self):
        settings = {**self.settings(), 'depth_interval': 2,
                    'depth_mode': 'depth-anything-v2-small-fp16', 'video_decode': 'amd'}
        self.assertEqual(validate_settings(settings), settings)

    def test_invalid_interval_is_not_silently_coerced(self):
        for interval in (-1, 3, True, 1.5, '2', None):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                validate_settings({**self.settings(), 'depth_interval': interval})


if __name__ == '__main__':
    unittest.main()
