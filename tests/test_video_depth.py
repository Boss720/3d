import unittest
from unittest.mock import Mock, patch

import numpy as np

from video_depth import VideoDepthEstimator


class VideoDepthTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((108, 192, 3), np.uint8)
        self.depth = np.linspace(0, 1, 32 * 48, dtype=np.float32).reshape(32, 48)

    def test_every_frame_uses_native_depth_and_filters_before_upscale(self):
        estimator = VideoDepthEstimator(smoothing=.35)
        predict = Mock(return_value=self.depth)
        with patch.object(estimator._filter, 'apply', wraps=estimator._filter.apply) as apply:
            first = estimator.process(self.frame, predict)
            second = estimator.process(self.frame, predict)
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(first.shape, (32, 48))
        self.assertEqual(second.shape, (32, 48))
        self.assertEqual(apply.call_args.args[0].shape, (32, 48, 3))
        self.assertFalse(np.shares_memory(first, self.depth))
        self.assertEqual(estimator.stats['neural_frames'], 2)

    def test_alternate_frames_use_backward_flow_and_preserve_every_frame(self):
        flow = Mock()
        vectors = np.zeros((32, 48, 2), np.float32)
        vectors[:, :, 0] = -2
        flow.calc.return_value = vectors
        estimator = VideoDepthEstimator(interval=2, smoothing=0, flow_factory=lambda: flow)
        predict = Mock(return_value=self.depth)
        estimator.process(self.frame, predict)
        second = estimator.process(self.frame, predict)
        np.testing.assert_array_equal(second[:, 2:], self.depth[:, :-2])
        estimator.process(self.frame, predict)
        estimator.process(self.frame, predict)
        self.assertEqual(predict.call_count, 2)
        self.assertEqual(flow.calc.call_count, 2)
        self.assertEqual(estimator.stats['flow_frames'], 2)
        self.assertEqual(estimator.stats['frames'], 4)

    def test_scene_cut_forces_fresh_depth_on_even_frame_and_resets_smoothing(self):
        flow = Mock()
        estimator = VideoDepthEstimator(interval=2, smoothing=.9, flow_factory=lambda: flow)
        predict = Mock(side_effect=[self.depth, np.ones_like(self.depth)])
        estimator.process(self.frame, predict)
        cut = np.full_like(self.frame, 255)
        result = estimator.process(cut, predict)
        np.testing.assert_array_equal(result, 1)
        self.assertEqual(predict.call_count, 2)
        flow.calc.assert_not_called()
        self.assertEqual(estimator.stats['scene_cuts'], 1)

    def test_nonfinite_or_broken_flow_falls_back_to_inference(self):
        for result in [np.full((32, 48, 2), np.nan, np.float32), RuntimeError('unavailable')]:
            with self.subTest(result=type(result).__name__):
                flow = Mock()
                if isinstance(result, Exception):
                    flow.calc.side_effect = result
                else:
                    flow.calc.return_value = result
                estimator = VideoDepthEstimator(interval=2, flow_factory=lambda: flow)
                predict = Mock(return_value=self.depth)
                estimator.process(self.frame, predict)
                self.assertTrue(np.isfinite(estimator.process(self.frame, predict)).all())
                self.assertEqual(predict.call_count, 2)
                self.assertEqual(estimator.stats['flow_fallbacks'], 1)

    def test_source_shape_change_forces_fresh_inference(self):
        estimator = VideoDepthEstimator(interval=2)
        predict = Mock(return_value=self.depth)
        estimator.process(self.frame, predict)
        estimator.process(np.zeros((120, 160, 3), np.uint8), predict)
        self.assertEqual(predict.call_count, 2)

    def test_invalid_depth_is_reported(self):
        for depth in [np.zeros((0, 5)), np.zeros((3, 3, 1)), np.full((3, 3), np.nan)]:
            with self.subTest(shape=depth.shape):
                with self.assertRaisesRegex(ValueError, 'finite 2D'):
                    VideoDepthEstimator().process(self.frame, lambda _: depth)

    def test_interval_is_strictly_one_or_two(self):
        for invalid in [0, 3, True, 1.5, '2']:
            with self.subTest(interval=invalid), self.assertRaises(ValueError):
                VideoDepthEstimator(interval=invalid)


if __name__ == '__main__':
    unittest.main()
