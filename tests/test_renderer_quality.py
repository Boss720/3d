import unittest

import numpy as np

from renderer_quality import _eye, render_visibility_pair


class VisibilityRendererTests(unittest.TestCase):
    def test_zero_strength_is_exact_and_copies_odd_shaped_source(self):
        frame = np.arange(7 * 101 * 3, dtype=np.uint8).reshape(7, 101, 3)
        left, right, stats = render_visibility_pair(frame, np.ones((7, 101), np.float32), 0, .5)
        np.testing.assert_array_equal(left, frame)
        np.testing.assert_array_equal(right, frame)
        self.assertFalse(np.shares_memory(left, frame))
        self.assertFalse(np.shares_memory(left, right))
        for eye in ('left', 'right'):
            self.assertEqual(stats[eye]['hole_fraction'], 0)
            self.assertFalse(stats[eye]['hole_mask'].any())
            self.assertEqual(stats[eye]['max_fill_distance'], 0)

    def test_near_inverse_depth_wins_collisions_and_eye_order_is_correct(self):
        frame = np.full((3, 200, 3), [200, 40, 10], np.uint8)
        frame[:, 100] = [5, 10, 240]
        depth = np.zeros((3, 200), np.float32)
        depth[:, 100] = 1
        left, right, stats = render_visibility_pair(frame, depth, 10, 0)
        np.testing.assert_array_equal(left[:, 103], frame[:, 100])
        np.testing.assert_array_equal(right[:, 97], frame[:, 100])
        for eye in ('left', 'right'):
            self.assertEqual(stats[eye]['collision_count'], 3)
            self.assertTrue(stats[eye]['hole_mask'][:, 100].all())
            self.assertGreater(stats[eye]['hole_fraction'], 0)

    def test_disocclusion_fill_chooses_farther_background_in_both_eyes(self):
        frame = np.full((5, 400, 3), [210, 60, 20], np.uint8)
        frame[:, 200:210] = [20, 40, 245]
        depth = np.zeros((5, 400), np.float32)
        depth[:, 200:210] = 1
        left, right, stats = render_visibility_pair(frame, depth, 10, 0)
        expected = np.broadcast_to([210, 60, 20], (5, 6, 3))
        np.testing.assert_array_equal(left[:, 200:206], expected)
        np.testing.assert_array_equal(right[:, 204:210], expected)
        self.assertTrue(stats['left']['hole_mask'][:, 200:206].all())
        self.assertTrue(stats['right']['hole_mask'][:, 204:210].all())
        for eye in ('left', 'right'):
            self.assertEqual(stats[eye]['hole_fraction'], 6 / 400)
            self.assertEqual(stats[eye]['unfilled_fraction'], 0)
            self.assertEqual(stats[eye]['max_fill_distance'], 6)
            self.assertLessEqual(stats[eye]['max_fill_distance'], stats['fill_radius'])

    def test_occlusion_is_independent_of_source_iteration_order(self):
        frame = np.zeros((1, 200, 3), np.uint8)
        frame[:, 100] = [10, 20, 255]
        frame[:, 102] = [255, 20, 10]
        depth = np.zeros((1, 200), np.float32)
        depth[:, 100] = 1
        depth[:, 102] = 1 / 3
        left, _, _ = render_visibility_pair(frame, depth, 10, 0)
        # Both nonzero-depth samples land at 103; the nearer source at 100 wins.
        np.testing.assert_array_equal(left[:, 103], frame[:, 100])

    def test_equal_depth_row_splat_tie_selects_rightmost_source(self):
        # Exercise the deterministic row primitive directly: normal public
        # equal-depth pixels share a displacement and ordinarily cannot tie.
        frame = np.zeros((1, 200, 3), np.uint8)
        frame[:, 100] = [1, 2, 200]
        frame[:, 102] = [200, 2, 1]
        depth = np.zeros((1, 200), np.float32)
        depth[:, 100] = depth[:, 102] = .5
        disparity = np.zeros_like(depth)
        disparity[:, 100], disparity[:, 102] = 3, 1
        result, stats = _eye(frame, depth, disparity, 1, 4, 3)
        np.testing.assert_array_equal(result[:, 103], frame[:, 102])
        self.assertEqual(stats['collision_count'], 2)

    def test_border_treatment_is_measured_and_fill_distance_is_bounded(self):
        frame = np.full((2, 200, 3), [12, 34, 56], np.uint8)
        depth = np.ones((2, 200), np.float32)
        left, right, stats = render_visibility_pair(frame, depth, 10, 0)
        np.testing.assert_array_equal(left, frame)
        np.testing.assert_array_equal(right, frame)
        self.assertTrue(stats['left']['hole_mask'][:, :3].all())
        self.assertTrue(stats['right']['hole_mask'][:, -3:].all())
        self.assertEqual(stats['left']['outside_source_count'], 6)
        self.assertEqual(stats['right']['outside_source_count'], 6)
        self.assertEqual(stats['max_fill_distance'], 3)

    def test_extreme_borders_do_not_fill_beyond_fixed_radius(self):
        frame = np.full((1, 4001, 3), [11, 22, 33], np.uint8)
        left, right, stats = render_visibility_pair(frame, np.ones((1, 4001), np.float32), 10, 0)
        self.assertEqual(stats['fill_radius'], 32)
        for eye in ('left', 'right'):
            self.assertLessEqual(stats[eye]['max_fill_distance'], 32)
            self.assertGreater(stats[eye]['unfilled_fraction'], 0)
            self.assertTrue(stats[eye]['unfilled_mask'].any())
        self.assertEqual(left.dtype, np.uint8)
        self.assertEqual(right.shape, frame.shape)

    def test_thin_foreground_colors_remain_source_colors(self):
        frame = np.zeros((5, 301, 3), np.uint8)
        frame[:] = [190, 80, 20]
        frame[:, 150] = [5, 10, 250]
        depth = np.zeros((5, 301), np.float32)
        depth[:, 150] = 1
        left, right, _ = render_visibility_pair(frame, depth, 10, .5)
        for eye in (left, right):
            self.assertEqual(eye.shape, frame.shape)
            colors = set(map(tuple, eye.reshape(-1, 3)))
            self.assertEqual(colors, {(190, 80, 20), (5, 10, 250)})

    def test_uniform_half_pixel_disparity_does_not_create_checkerboard_holes(self):
        frame = np.full((2, 200, 3), [10, 20, 30], np.uint8)
        _, _, stats = render_visibility_pair(frame, np.full((2, 200), 1 / 6, np.float32), 10, 0)
        for eye in ('left', 'right'):
            self.assertEqual(stats[eye]['collision_count'], 0)
            self.assertFalse(stats[eye]['hole_mask'][:, 1:-1].any())

    def test_rasterized_shift_never_exceeds_disparity_budget(self):
        frame = np.zeros((1, 641, 3), np.uint8)
        frame[:, 300] = [1, 2, 255]
        left, right, stats = render_visibility_pair(frame, np.ones((1, 641), np.float32), 10, 0)
        for eye in (left, right):
            red_columns = np.flatnonzero(eye[0, :, 2] == 255)
            self.assertEqual(len(red_columns), 1)
            self.assertLessEqual(abs(int(red_columns[0]) - 300), stats['max_displacement_pixels'])

    def test_depth_is_clamped_sanitized_and_source_is_not_modified(self):
        frame = np.zeros((1, 5, 3), np.uint8)
        depth = np.array([[np.nan, np.inf, -np.inf, -5, 5]], np.float32)
        original = depth.copy()
        left, right, stats = render_visibility_pair(frame, depth, 10, .5)
        self.assertTrue(np.isfinite(left).all() and np.isfinite(right).all())
        np.testing.assert_array_equal(depth, original)
        self.assertGreaterEqual(stats['left']['hole_fraction'], 0)
        self.assertLessEqual(stats['right']['hole_fraction'], 1)

    def test_invalid_inputs_raise_clear_errors(self):
        frame = np.zeros((3, 5, 3), np.uint8)
        for bad_frame in (frame.astype(np.float32), frame[:, :, 0], np.zeros((0, 5, 3), np.uint8)):
            with self.subTest(shape=bad_frame.shape), self.assertRaises(ValueError):
                render_visibility_pair(bad_frame, np.zeros(bad_frame.shape[:2]), 5, .5)
        with self.assertRaisesRegex(ValueError, 'Depth|depth'):
            render_visibility_pair(frame, np.zeros((5, 3)), 5, .5)
        for strength, convergence in [(float('nan'), .5), (5, float('inf'))]:
            with self.assertRaisesRegex(ValueError, 'finite'):
                render_visibility_pair(frame, np.zeros((3, 5)), strength, convergence)


if __name__ == '__main__':
    unittest.main()
