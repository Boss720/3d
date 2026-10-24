import unittest
from types import SimpleNamespace
import numpy as np
from depth_engine import ModelManager


class RawDepthTests(unittest.TestCase):
    def test_raw_native_retains_scale_while_default_remains_normalized(self):
        values = np.arange(16, dtype=np.float32).reshape(1, 4, 4) + 100
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name='image', type='tensor(float)', shape=[1, 3, 4, 4])],
            run=lambda *args: [values])
        manager = ModelManager.__new__(ModelManager)
        manager._input_size = lambda *args: (4, 4)
        frame = np.zeros((4, 4, 3), np.uint8)
        raw = manager._predict_with_session(session, frame, 'depth-anything-v2-small', 256, native=True, raw=True)
        normalized = manager._predict_with_session(session, frame, 'depth-anything-v2-small', 256, native=True)
        np.testing.assert_array_equal(raw, values[0])
        self.assertEqual(normalized.min(), 0)
        self.assertEqual(normalized.max(), 1)
        raw[:] = 0
        self.assertEqual(values.min(), 100)
