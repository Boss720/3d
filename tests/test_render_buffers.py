import unittest
import cv2
import numpy as np
from stereo import LAYOUTS, StereoRenderer, make_eye_pair, pack_output, render_output


def legacy_xreal(left, right):
    def fit(eye):
        h, w = eye.shape[:2]
        scale = min(1920 / w, 1080 / h)
        fw, fh = max(1, min(1920, round(w * scale))), max(1, min(1080, round(h * scale)))
        fitted = cv2.resize(eye, (fw, fh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((1080, 1920, 3), np.uint8)
        x, y = (1920-fw)//2, (1080-fh)//2
        canvas[y:y+fh, x:x+fw] = fitted
        return canvas
    return np.concatenate((fit(left), fit(right)), axis=1)


class RenderBuffersTests(unittest.TestCase):
    def test_context_matches_stateless_and_preserves_previous_frames(self):
        rng = np.random.default_rng(42)
        renderer = StereoRenderer()
        for shape in ((17, 31), (21, 13), (17, 31)):
            frame = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
            depth = rng.random(shape, dtype=np.float32)
            depth.flat[:3] = [np.nan, np.inf, -np.inf]
            original_frame, original_depth = frame.copy(), depth.copy()
            for layout in sorted(LAYOUTS):
                for strength in (0, 4.5, 10):
                    actual = renderer.render(frame, depth, strength, .45, layout)
                    expected = render_output(frame, depth, strength, .45, layout)
                    np.testing.assert_array_equal(actual, expected)
                    retained = actual.copy()
                    renderer.render(255-frame, 1-depth, strength, .7, layout)
                    np.testing.assert_array_equal(actual, retained)
                    self.assertFalse(np.shares_memory(actual, frame))
            np.testing.assert_array_equal(frame, original_frame)
            np.testing.assert_array_equal(depth, original_depth)

    def test_xreal_matches_original_packing(self):
        rng = np.random.default_rng(9)
        for h, w in ((1, 1), (7, 5), (48, 64), (80, 20), (20, 100), (1100, 2000)):
            left = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
            right = 255-left
            np.testing.assert_array_equal(pack_output(left, np.zeros((h,w)), left, right, 'XREAL_SBS'), legacy_xreal(left, right))

    def test_context_validates_depth_and_scalars(self):
        renderer = StereoRenderer()
        frame = np.zeros((3, 4, 3), np.uint8)
        for depth, strength, convergence in ((np.zeros((2,4)), 1, .5), (np.zeros((3,4)), float('nan'), .5), (np.zeros((3,4)), 1, float('inf'))):
            with self.assertRaises(ValueError):
                renderer.render(frame, depth, strength, convergence, 'SBS')
