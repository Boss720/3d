import unittest
from types import SimpleNamespace

from job_settings import settings_snapshot, validate_settings


class DepthCacheSettingsTests(unittest.TestCase):
    def settings(self):
        job = SimpleNamespace(layout='SBS', strength=5, convergence=.5,
                              depth_mode='depth-anything-v2-small', input_size=256,
                              resolution='source', quality='fast', invert_depth=False,
                              temporal_smoothing=.35, edge_refinement=True,
                              execution_mode='auto', depth_interval=1,
                              video_decode='software')
        return settings_snapshot(job)

    def test_snapshot_defaults_cache_off(self):
        self.assertFalse(self.settings()['use_depth_cache'])

    def test_old_v1_settings_default_cache_off(self):
        settings = self.settings()
        settings.pop('use_depth_cache')
        self.assertFalse(validate_settings(settings)['use_depth_cache'])

    def test_cache_setting_roundtrips(self):
        enabled = {**self.settings(), 'use_depth_cache': True}
        self.assertEqual(validate_settings(enabled), enabled)

    def test_cache_setting_rejects_non_boolean_values(self):
        for value in (1, 0, 'true', 'false', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_settings({**self.settings(), 'use_depth_cache': value})


if __name__ == '__main__':
    unittest.main()
