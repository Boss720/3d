import unittest
from unittest.mock import patch

import numpy as np

from temporal_quality import MotionAlignedDepthFilter


class MotionAlignedTemporalTests(unittest.TestCase):
    def test_static_scene_blends_aligned_history_and_first_frame_is_unchanged(self):
        smoother = MotionAlignedDepthFilter(.6)
        frame = np.full((32, 48, 3), 100, np.uint8)
        first = smoother.apply(frame, np.full((32, 48), .8, np.float32))
        np.testing.assert_array_equal(first, np.float32(.8))
        self.assertEqual(smoother.stats['reset_reason'], 'first_frame')
        second = smoother.apply(frame, np.full((32, 48), .7, np.float32))
        np.testing.assert_allclose(second, .76, atol=1e-6)
        self.assertGreater(smoother.stats['accepted_fraction'], .99)

    def test_known_backward_flow_aligns_previous_depth_instead_of_old_location(self):
        height, width = 40, 80
        previous = np.full((height, width, 3), 70, np.uint8)
        previous[:, 20:30] = [20, 40, 230]
        current = np.full_like(previous, 70)
        current[:, 28:38] = [20, 40, 230]
        previous_depth = np.full((height, width), .2, np.float32)
        previous_depth[:, 20:30] = .8
        current_depth = np.full((height, width), .2, np.float32)
        current_depth[:, 28:38] = .7
        backward = np.zeros((height, width, 2), np.float32)
        backward[:, :, 0] = -8
        forward = -backward
        smoother = MotionAlignedDepthFilter(.6)
        smoother.apply(previous, previous_depth)
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback', side_effect=[backward, forward]) as flow:
            output = smoother.apply(current, current_depth)
        np.testing.assert_allclose(output[:, 28:38], .76, atol=1e-6)
        np.testing.assert_allclose(output[:, 20:28], .2, atol=1e-6)
        self.assertFalse(smoother.stats['accepted_mask'][:, :8].any())
        self.assertEqual(flow.call_count, 2)

    def test_forward_backward_inconsistency_rejects_history(self):
        frame = np.full((24, 40, 3), 90, np.uint8)
        smoother = MotionAlignedDepthFilter(.8)
        smoother.apply(frame, np.full((24, 40), .5, np.float32))
        backward = np.zeros((24, 40, 2), np.float32)
        forward = np.full_like(backward, 4)
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback', side_effect=[backward, forward]):
            output = smoother.apply(frame, np.full((24, 40), .4, np.float32))
        np.testing.assert_array_equal(output, np.float32(.4))
        self.assertEqual(smoother.stats['accepted_fraction'], 0)

    def test_new_disocclusion_rejects_old_foreground_even_with_false_zero_flow(self):
        frame = np.full((32, 80, 3), 80, np.uint8)
        first = np.full((32, 80), .2, np.float32)
        first[:, 20:30] = .8
        smoother = MotionAlignedDepthFilter(.8)
        smoother.apply(frame, first)
        zero = np.zeros((32, 80, 2), np.float32)
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback', return_value=zero):
            output = smoother.apply(frame, np.full((32, 80), .2, np.float32))
        np.testing.assert_allclose(output[:, 20:30], .2, atol=1e-6)
        self.assertFalse(smoother.stats['accepted_mask'][:, 20:30].any())

    def test_local_appearance_change_rejects_history_without_global_cut(self):
        previous = np.full((32, 80, 3), 80, np.uint8)
        current = previous.copy()
        current[10:15, 20:25] = 240
        smoother = MotionAlignedDepthFilter(.7)
        smoother.apply(previous, np.full((32, 80), .6, np.float32))
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback', return_value=np.zeros((32, 80, 2), np.float32)):
            output = smoother.apply(current, np.full((32, 80), .5, np.float32))
        self.assertFalse(smoother.stats['scene_cut'])
        np.testing.assert_array_equal(output[10:15, 20:25], .5)
        self.assertFalse(smoother.stats['accepted_mask'][10:15, 20:25].any())

    def test_scene_cut_shape_change_and_explicit_reset_drop_history(self):
        smoother = MotionAlignedDepthFilter(.8)
        smoother.apply(np.zeros((20, 30, 3), np.uint8), np.ones((20, 30), np.float32))
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback') as flow:
            output = smoother.apply(np.full((20, 30, 3), 255, np.uint8), np.zeros((20, 30), np.float32))
        np.testing.assert_array_equal(output, 0)
        self.assertTrue(smoother.stats['scene_cut'])
        self.assertEqual(smoother.stats['reset_reason'], 'scene_cut')
        flow.assert_not_called()
        smoother.apply(np.zeros((21, 31, 3), np.uint8), np.full((21, 31), .3, np.float32))
        self.assertEqual(smoother.stats['reset_reason'], 'shape_change')
        smoother.reset()
        output = smoother.apply(np.zeros((21, 31, 3), np.uint8), np.full((21, 31), .7, np.float32))
        np.testing.assert_array_equal(output, np.float32(.7))
        self.assertEqual(smoother.stats['reset_reason'], 'explicit_reset')

    def test_actual_flow_moving_foreground_leaves_no_persistent_old_location_trail(self):
        smoother = MotionAlignedDepthFilter(.8)
        for start in (15, 30, 45):
            frame = np.full((48, 96, 3), 70, np.uint8)
            frame[12:36, start:start + 10] = [20, 40, 230]
            depth = np.full((48, 96), .2, np.float32)
            depth[12:36, start:start + 10] = .8
            output = smoother.apply(frame, depth)
            np.testing.assert_allclose(output[12:36, 15:start], .2, atol=1e-6)

    def test_same_color_small_depth_change_is_a_documented_ambiguity(self):
        # Images alone cannot identify this subtle depth-only scene change.
        frame = np.full((24, 40, 3), 100, np.uint8)
        smoother = MotionAlignedDepthFilter(.7)
        smoother.apply(frame, np.full((24, 40), .6, np.float32))
        output = smoother.apply(frame, np.full((24, 40), .5, np.float32))
        self.assertGreater(float(output.mean()), .5)
        self.assertFalse(smoother.stats['depth_scale_recovered'])

    def test_finite_bounded_output_and_input_ownership(self):
        smoother = MotionAlignedDepthFilter(.7)
        frame = np.zeros((3, 5, 3), np.uint8)
        depth = np.array([[np.nan, np.inf, -np.inf, -3, 4]] * 3, np.float32)
        original = depth.copy()
        output = smoother.apply(frame, depth)
        self.assertTrue(np.isfinite(output).all())
        self.assertGreaterEqual(float(output.min()), 0)
        self.assertLessEqual(float(output.max()), 1)
        self.assertEqual(output.dtype, np.float32)
        np.testing.assert_array_equal(depth, original)
        self.assertFalse(np.shares_memory(output, depth))
        with self.assertRaises(ValueError):
            smoother.apply(frame, np.zeros((5, 3), np.float32))
        with self.assertRaises(ValueError):
            MotionAlignedDepthFilter(float('nan'))

    def test_disabled_filter_skips_flow_and_returns_current_depth(self):
        smoother = MotionAlignedDepthFilter(0)
        frame = np.full((24, 40, 3), 100, np.uint8)
        smoother.apply(frame, np.full((24, 40), .6, np.float32))
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback') as flow:
            result = smoother.apply(frame, np.full((24, 40), .5, np.float32))
        np.testing.assert_array_equal(result, np.float32(.5))
        self.assertEqual(smoother.stats['reset_reason'], 'disabled')
        flow.assert_not_called()

    def test_nonfinite_flow_is_rejected_without_nonfinite_output(self):
        smoother = MotionAlignedDepthFilter(.7)
        frame = np.full((24, 40, 3), 100, np.uint8)
        smoother.apply(frame, np.full((24, 40), .6, np.float32))
        bad_flow = np.full((24, 40, 2), np.nan, np.float32)
        with patch('temporal_quality.cv2.calcOpticalFlowFarneback', return_value=bad_flow):
            result = smoother.apply(frame, np.full((24, 40), .5, np.float32))
        np.testing.assert_array_equal(result, np.float32(.5))
        self.assertEqual(smoother.stats['accepted_fraction'], 0)


if __name__ == '__main__':
    unittest.main()
