import unittest
from unittest.mock import Mock

import numpy as np

from depth_normalization import SceneDepthNormalizer
from video_depth import VideoDepthEstimator


class TemporalStabilityTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((32, 48, 3), np.uint8)
        self.depth = np.linspace(10, 20, 32 * 48, dtype=np.float32).reshape(32, 48)

    def test_every_frame_cut_is_recorded_and_resets_state(self):
        estimator = VideoDepthEstimator(interval=1, smoothing=.9)
        predict = Mock(side_effect=[np.zeros((32, 48), np.float32), np.ones((32, 48), np.float32)])
        estimator.process(self.frame, predict)
        result = estimator.process(np.full_like(self.frame, 255), predict)
        np.testing.assert_array_equal(result, 1)
        self.assertEqual(estimator.stats['scene_cuts'], 1)

    def test_raw_normalization_resets_before_prediction_for_each_interval(self):
        for interval in (0, 1, 2):
            with self.subTest(interval=interval):
                normalizer = SceneDepthNormalizer()
                estimator = VideoDepthEstimator(interval=interval, smoothing=.9, normalizer=normalizer)
                first = estimator.process(self.frame, lambda _: self.depth)
                result = estimator.process(np.full_like(self.frame, 255), lambda _: self.depth * 4 + 100)
                np.testing.assert_allclose(result, first, atol=2e-6)
                self.assertEqual(estimator.stats['scene_cuts'], 1)
                self.assertEqual(estimator.stats['neural_frames'], 2)

    def test_lower_contrast_histogram_cut_is_detected(self):
        estimator = VideoDepthEstimator(interval=1, smoothing=.9)
        estimator.process(np.full_like(self.frame, 64), lambda _: self.depth / 20)
        estimator.process(np.full_like(self.frame, 96), lambda _: self.depth / 20)
        self.assertEqual(estimator.stats['scene_cuts'], 1)

    def test_robust_raw_range_prevents_single_outlier_pumping(self):
        normalizer = SceneDepthNormalizer()
        stable = normalizer.normalize(self.depth)
        outlier = self.depth.copy()
        outlier[0, 0] = 10000
        result = normalizer.normalize(outlier)
        baseline = (outlier - outlier.min()) / (outlier.max() - outlier.min())
        robust_error = float(np.abs(result.ravel()[1:] - stable.ravel()[1:]).mean())
        baseline_error = float(np.abs(baseline.ravel()[1:] - stable.ravel()[1:]).mean())
        self.assertLess(robust_error, .001)
        self.assertLess(robust_error, baseline_error * .01)

    def test_range_updates_are_bounded_and_reset_reanchors(self):
        normalizer = SceneDepthNormalizer(update_rate=1, max_step=.1)
        normalizer.normalize(self.depth)
        previous = normalizer._bounds
        normalizer.normalize(self.depth * 100)
        span = previous[1] - previous[0]
        for old, new in zip(previous, normalizer._bounds):
            self.assertLessEqual(abs(new - old), span * .1 + 1e-6)
        normalizer.reset()
        result = normalizer.normalize(self.depth * 100)
        np.testing.assert_allclose(result, SceneDepthNormalizer().normalize(self.depth), atol=1e-6)

    def test_normalizer_shape_change_reanchors_and_never_aliases(self):
        normalizer = SceneDepthNormalizer()
        raw = self.depth.copy()
        first = normalizer.normalize(raw)
        saved = first.copy()
        raw[:] = 0
        np.testing.assert_array_equal(first, saved)
        second_raw = np.arange(48, dtype=np.float32).reshape(6, 8) + 200
        second = normalizer.normalize(second_raw)
        np.testing.assert_array_equal(second, SceneDepthNormalizer().normalize(second_raw))
        second[:] = 0
        self.assertFalse(np.shares_memory(first, second_raw))
        np.testing.assert_array_equal(first, saved)

    def test_normalizer_rejects_invalid_maps_and_stays_finite_for_constant(self):
        for raw in (np.zeros((0, 2)), np.zeros((2, 2, 2)), np.full((2, 2), np.nan), np.full((2, 2), np.inf)):
            with self.subTest(shape=raw.shape), self.assertRaises(ValueError):
                SceneDepthNormalizer().normalize(raw)
        constant = SceneDepthNormalizer().normalize(np.ones((32, 48), np.float32))
        self.assertTrue(np.isfinite(constant).all())
        np.testing.assert_array_equal(constant, 0)

    def test_initial_constant_prediction_does_not_lock_range_tracking(self):
        normalizer = SceneDepthNormalizer()
        normalizer.normalize(np.ones_like(self.depth))
        result = normalizer.normalize(self.depth)
        np.testing.assert_allclose(result, SceneDepthNormalizer().normalize(self.depth))

    def test_adaptive_maximum_age_forces_periodic_neural_refresh(self):
        flow = Mock()
        flow.calc.return_value = np.zeros((32, 48, 2), np.float32)
        estimator = VideoDepthEstimator(interval=0, smoothing=0, max_propagation_age=2, flow_factory=lambda: flow)
        predict = Mock(return_value=self.depth / 20)
        methods = []
        for _ in range(7):
            estimator.process(self.frame, predict)
            methods.append(estimator.last_method)
        self.assertEqual(methods, ['neural', 'optical_flow', 'optical_flow', 'neural', 'optical_flow', 'optical_flow', 'neural'])
        self.assertEqual(predict.call_count, 3)
        self.assertEqual(estimator.stats['flow_frames'], 4)

    def test_small_new_object_rejects_reuse_in_both_modes(self):
        for interval in (0, 2):
            with self.subTest(interval=interval):
                flow = Mock()
                flow.calc.return_value = np.zeros((32, 48, 2), np.float32)
                estimator = VideoDepthEstimator(interval=interval, smoothing=0, flow_factory=lambda: flow)
                predict = Mock(side_effect=[np.zeros((32, 48), np.float32), np.ones((32, 48), np.float32)])
                estimator.process(self.frame, predict)
                changed = self.frame.copy()
                changed[10:14, 10:14] = (255, 0, 0)
                result = estimator.process(changed, predict)
                np.testing.assert_array_equal(result, 1)
                self.assertEqual(estimator.stats['flow_fallbacks'], 1)
                self.assertEqual(estimator.stats['scene_cuts'], 0)

    def test_small_background_disocclusion_and_inconsistent_correspondence_refresh(self):
        for reason in ('border', 'inconsistent'):
            with self.subTest(reason=reason):
                backward = np.zeros((32, 48, 2), np.float32)
                forward = backward.copy()
                if reason == 'border':
                    backward[:, 0, 0] = -2
                else:
                    forward[10:14, 10:14, 0] = 8
                flow = Mock()
                flow.calc.side_effect = [backward, forward]
                estimator = VideoDepthEstimator(interval=0, smoothing=0, flow_factory=lambda: flow)
                predict = Mock(return_value=self.depth / 20)
                estimator.process(self.frame, predict)
                estimator.process(self.frame, predict)
                self.assertEqual(predict.call_count, 2)
                self.assertEqual(estimator.stats['flow_fallbacks'], 1)

    def test_malformed_flow_falls_back_and_result_buffers_are_independent(self):
        for vectors in (None, [[0]], np.zeros((32, 48)), np.full((32, 48, 2), np.nan), np.full((32, 48, 2), 'x'), np.full((32, 48, 2), 1e300), np.full((32, 48, 2), 1j)):
            with self.subTest(kind=type(vectors).__name__):
                flow = Mock()
                flow.calc.return_value = vectors
                estimator = VideoDepthEstimator(interval=0, smoothing=0, flow_factory=lambda: flow)
                raw = self.depth / 20
                predict = Mock(return_value=raw)
                first = estimator.process(self.frame, predict)
                first[:] = 0
                result = estimator.process(self.frame, predict)
                np.testing.assert_array_equal(result, raw)
                self.assertFalse(np.shares_memory(result, raw))
                self.assertFalse(np.shares_memory(result, estimator._previous_depth))
                self.assertEqual(estimator.stats['flow_fallbacks'], 1)

    def test_adaptive_shape_change_forces_fresh_depth(self):
        flow = Mock()
        estimator = VideoDepthEstimator(interval=0, flow_factory=lambda: flow, normalizer=SceneDepthNormalizer())
        estimator.process(self.frame, lambda _: self.depth)
        result = estimator.process(np.zeros((48, 64, 3), np.uint8), lambda _: self.depth * 3 + 20)
        np.testing.assert_allclose(result, SceneDepthNormalizer().normalize(self.depth), atol=1e-6)
        flow.calc.assert_not_called()

    def test_normalization_remains_opt_in(self):
        estimator = VideoDepthEstimator(smoothing=0)
        np.testing.assert_array_equal(estimator.process(self.frame, lambda _: self.depth), 1)
        self.assertIsNone(estimator.normalizer)

    def test_age_and_normalizer_options_are_validated(self):
        for invalid in (0, 31, True, 1.5, '2'):
            with self.subTest(age=invalid), self.assertRaises(ValueError):
                VideoDepthEstimator(max_propagation_age=invalid)
        for invalid in (0, 2, True, float('nan')):
            with self.subTest(rate=invalid), self.assertRaises(ValueError):
                SceneDepthNormalizer(update_rate=invalid)


if __name__ == '__main__':
    unittest.main()
