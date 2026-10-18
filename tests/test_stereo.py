import unittest

import numpy as np

from stereo import TemporalDepthFilter, layout_size, make_eye_pair, pack_output, refine_depth


class StereoTests(unittest.TestCase):
    def test_strength_is_neutral_then_monotonic_without_eye_reversal(self):
        frame = np.zeros((20, 400, 3), dtype=np.uint8)
        frame[:, 190:211] = 255
        depth = np.ones(frame.shape[:2], dtype=np.float32)
        disparities = []
        x = np.arange(frame.shape[1])
        for strength in (0, 1, 3, 5, 7, 10):
            left, right = make_eye_pair(frame, depth, strength, 0)
            if strength == 0:
                np.testing.assert_array_equal(left, frame)
                np.testing.assert_array_equal(right, frame)
            left_center = np.average(x, weights=left[10, :, 0])
            right_center = np.average(x, weights=right[10, :, 0])
            disparities.append(left_center - right_center)
        self.assertTrue(all(a < b for a, b in zip(disparities, disparities[1:])), disparities)
        self.assertAlmostEqual(disparities[-1], 400 * .03, places=2)

    def test_far_depth_reverses_disparity_relative_to_convergence(self):
        frame = np.tile(np.arange(200, dtype=np.uint8)[None, :, None], (10, 1, 3))
        for value, direction in ((0, 1), (1, -1)):
            depth = np.full((10, 200), value, dtype=np.float32)
            left, right = make_eye_pair(frame, depth, 10, .5)
            self.assertEqual(np.sign(int(left[5, 100, 0]) - int(right[5, 100, 0])), direction)
        neutral = np.full((10, 200), .5, dtype=np.float32)
        left, right = make_eye_pair(frame, neutral, 10, .5)
        np.testing.assert_array_equal(left, frame)
        np.testing.assert_array_equal(right, frame)

    def test_all_layout_dimensions_match_packed_pixels_even_and_odd(self):
        for width, height in ((8, 6), (7, 5), (1, 1)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.ones((height, width), dtype=np.float32)
            expected = {
                'SBS': (2 * width, height), 'CROSS': (2 * width, height),
                'RGBD': (2 * width, height), 'TB': (width, 2 * height),
                'HALFSBS': (2 * max(1, width // 2), height),
                'HALFTB': (width, 2 * max(1, height // 2)),
                'ANAGLYPH': (width, height), 'DEPTH': (width, height),
            }
            for layout, size in expected.items():
                with self.subTest(width=width, height=height, layout=layout):
                    packed = pack_output(frame, depth, frame, frame, layout)
                    self.assertEqual(layout_size(width, height, layout), size)
                    self.assertEqual(packed.shape, (size[1], size[0], 3))
                    self.assertEqual(packed.dtype, np.uint8)

    def test_depth_output_preserves_orientation_and_rgbd_source(self):
        frame = np.full((3, 3, 3), (10, 20, 30), dtype=np.uint8)
        depth = np.tile(np.array([0, .5, 1], dtype=np.float32), (3, 1))
        output = pack_output(frame, depth, frame, frame, 'DEPTH')
        np.testing.assert_array_equal(output[0, :, 0], [0, 128, 255])
        np.testing.assert_array_equal(output[:, :, 0], output[:, :, 2])
        rgbd = pack_output(frame, depth, frame, frame, 'RGBD')
        np.testing.assert_array_equal(rgbd[:, :3], frame)
        np.testing.assert_array_equal(rgbd[:, 3:], output)

    def test_eye_order_and_anaglyph_channels(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        depth = np.zeros((4, 6), dtype=np.float32)
        left = np.full_like(frame, (11, 22, 33))
        right = np.full_like(frame, (44, 55, 66))
        for layout, expected in (('SBS', (left, right)), ('CROSS', (right, left))):
            packed = pack_output(frame, depth, left, right, layout)
            np.testing.assert_array_equal(packed[:, :6], expected[0])
            np.testing.assert_array_equal(packed[:, 6:], expected[1])
        anaglyph = pack_output(frame, depth, left, right, 'ANAGLYPH')
        np.testing.assert_array_equal(anaglyph[0, 0], (44, 55, 33))

    def test_refinement_bounds_invalid_values_and_preserves_input(self):
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        depth = np.array([[np.nan, np.inf, -np.inf], [-1, .5, 2], [0, 1, .4]])
        original = depth.copy()
        for enabled in (False, True):
            result = refine_depth(frame, depth, enabled)
            self.assertEqual(result.dtype, np.float32)
            self.assertEqual(result.shape, (3, 3))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(((0 <= result) & (result <= 1)).all())
        np.testing.assert_array_equal(depth, original)

    def test_guided_refinement_smooths_noise_and_preserves_aligned_edges(self):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, 48:] = 255
        target = np.full((64, 96), .2, dtype=np.float32)
        target[:, 48:] = .8
        noisy = target + np.random.default_rng(7).normal(0, .06, target.shape).astype(np.float32)
        refined = refine_depth(frame, noisy, True)
        self.assertLess(np.std(refined[:, :40] - target[:, :40]), np.std(noisy[:, :40] - target[:, :40]) * .5)
        self.assertGreater(refined[:, 48].mean() - refined[:, 47].mean(), .4)

    def test_invalid_layout_and_mismatched_depth_are_rejected(self):
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            layout_size(6, 4, 'INVALID')
        with self.assertRaises(ValueError):
            make_eye_pair(frame, np.zeros((6, 4)), 5, .5)


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((40, 60, 3), 60, dtype=np.uint8)
        self.low = np.full((40, 60), .2, dtype=np.float32)
        self.high = np.full((40, 60), .8, dtype=np.float32)

    def test_static_smoothing_reset_and_instance_isolation(self):
        temporal = TemporalDepthFilter(.75)
        np.testing.assert_array_equal(temporal.apply(self.frame, self.low), self.low)
        np.testing.assert_allclose(temporal.apply(self.frame, self.high), .35, atol=1e-6)
        independent = TemporalDepthFilter(.75)
        np.testing.assert_array_equal(independent.apply(self.frame, self.high), self.high)
        temporal.reset()
        np.testing.assert_array_equal(temporal.apply(self.frame, self.high), self.high)

    def test_scene_cut_resets_and_following_frame_uses_new_history(self):
        temporal = TemporalDepthFilter(.9)
        temporal.apply(self.frame, self.low)
        cut = np.full_like(self.frame, 240)
        np.testing.assert_array_equal(temporal.apply(cut, self.high), self.high)
        np.testing.assert_allclose(temporal.apply(cut, self.low), .74, atol=1e-6)

    def test_motion_suppresses_history_at_old_and_new_object_positions(self):
        temporal = TemporalDepthFilter(.9)
        before = self.frame.copy()
        before[10:20, 10:20] = 220
        after = self.frame.copy()
        after[10:20, 25:35] = 220
        old_depth = self.low.copy()
        old_depth[10:20, 10:20] = .8
        new_depth = self.low.copy()
        new_depth[10:20, 25:35] = .8
        temporal.apply(before, old_depth)
        result = temporal.apply(after, new_depth)
        np.testing.assert_allclose(result[12:18, 12:18], .2, atol=.01)
        np.testing.assert_allclose(result[12:18, 27:33], .8, atol=.01)

    def test_zero_smoothing_shape_change_and_output_copies(self):
        temporal = TemporalDepthFilter(0)
        temporal.apply(self.frame, self.low)
        result = temporal.apply(self.frame, self.high)
        np.testing.assert_array_equal(result, self.high)
        result[:] = 0
        np.testing.assert_array_equal(self.high, np.full((40, 60), .8, dtype=np.float32))
        small = np.ones((4, 6), dtype=np.float32)
        np.testing.assert_array_equal(temporal.apply(self.frame[:4, :6], small), small)


if __name__ == '__main__':
    unittest.main()
