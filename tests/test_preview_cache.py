import unittest

import numpy as np

from preview_cache import PreviewDepthCache


class PreviewCacheTests(unittest.TestCase):
    def test_lru_budget_and_copied_ownership(self):
        cache = PreviewDepthCache(max_bytes=32, max_entries=2)
        depth = np.ones((2, 2), np.float32)
        cache.put('a', depth, {'providers': ['cpu']})
        depth[:] = 0
        returned, execution = cache.get('a')
        self.assertTrue(returned.all())
        returned[:] = 0
        execution['providers'].append('changed')
        self.assertTrue(cache.get('a')[0].all())
        self.assertEqual(cache.get('a')[1]['providers'], ['cpu'])
        cache.put('b', depth, {})
        cache.get('a')
        cache.put('c', depth, {})
        self.assertIsNone(cache.get('b'))
        self.assertEqual(cache.info()['depth_bytes'], 32)
        cache.put('large', np.zeros((10, 10), np.float32), {})
        self.assertIsNone(cache.get('large'))
        cache.clear()
        self.assertEqual(cache.info()['depth_bytes'], 0)
