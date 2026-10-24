import json
import unittest
from unittest.mock import patch

import numpy as np

from stereo import LAYOUTS, StereoRenderer, layout_size, make_eye_pair, pack_output, render_output
from renderer_quality import render_visibility_pair, visibility_summary


class RendererIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(7).integers(0, 256, (9, 201, 3), dtype=np.uint8)
        self.depth = np.linspace(0, 1, 9 * 201, dtype=np.float32).reshape(9, 201)

    def test_visibility_is_selectable_and_uses_foreground_visibility(self):
        frame = np.full((3, 200, 3), [200, 40, 10], np.uint8)
        frame[:, 100] = [5, 10, 240]
        depth = np.zeros((3, 200), np.float32)
        depth[:, 100] = 1
        renderer = StereoRenderer()
        result = renderer.render(frame, depth, 10, 0, 'SBS', method='visibility')
        left, right, _ = render_visibility_pair(frame, depth, 10, 0)
        np.testing.assert_array_equal(result, np.concatenate((left, right), axis=1))
        np.testing.assert_array_equal(result[:, 103], frame[:, 100])
        np.testing.assert_array_equal(result[:, 200 + 97], frame[:, 100])

    def test_default_and_explicit_fast_preserve_exact_legacy_pixels_all_layouts(self):
        renderer = StereoRenderer()
        left, right = make_eye_pair(self.frame, self.depth, 7, .4)
        for layout in sorted(LAYOUTS):
            with self.subTest(layout=layout):
                expected = pack_output(self.frame, self.depth, left, right, layout)
                np.testing.assert_array_equal(renderer.render(self.frame, self.depth, 7, .4, layout), expected)
                np.testing.assert_array_equal(renderer.render(self.frame, self.depth, 7, .4, layout, method='fast'), expected)
                np.testing.assert_array_equal(render_output(self.frame, self.depth, 7, .4, layout, method='fast'), expected)

    def test_visibility_all_layouts_match_existing_pair_and_packing(self):
        renderer = StereoRenderer()
        left, right, _ = render_visibility_pair(self.frame, self.depth, 8, .3)
        for layout in sorted(LAYOUTS):
            with self.subTest(layout=layout):
                expected = pack_output(self.frame, self.depth, left, right, layout)
                result = renderer.render(self.frame, self.depth, 8, .3, layout, method='visibility')
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(render_output(self.frame, self.depth, 8, .3, layout, method='visibility'), expected)
                width, height = layout_size(201, 9, layout)
                self.assertEqual(result.shape, (height, width, 3))

    def test_zero_strength_preserves_exact_pair_and_independent_packed_pixels(self):
        renderer = StereoRenderer()
        for method in ('fast', 'visibility'):
            with self.subTest(method=method):
                result = renderer.render(self.frame, self.depth, 0, .5, 'SBS', method=method)
                np.testing.assert_array_equal(result[:, :201], self.frame)
                np.testing.assert_array_equal(result[:, 201:], self.frame)
                self.assertFalse(np.shares_memory(result, self.frame))
                original = result.copy()
                renderer.render(np.zeros_like(self.frame), self.depth, 10, 0, 'SBS', method=method)
                np.testing.assert_array_equal(result, original)

    def test_visibility_diagnostics_are_compact_json_and_reset_for_fast(self):
        renderer = StereoRenderer()
        renderer.render(self.frame, self.depth, 10, .5, 'SBS', method='visibility')
        summary = renderer.last_stats
        self.assertEqual(summary['method'], 'visibility')
        self.assertFalse(summary['surface_recovery_claimed'])
        self.assertLess(len(json.dumps(summary)), 1200)
        for eye in ('left', 'right'):
            self.assertNotIn('hole_mask', summary[eye])
            self.assertNotIn('unfilled_mask', summary[eye])
            self.assertLessEqual(summary[eye]['max_fill_distance'], summary['fill_radius'])
            self.assertLessEqual(summary[eye]['unfilled_fraction'], summary[eye]['hole_fraction'])
        saved = json.dumps(summary, sort_keys=True)
        renderer.render(self.frame, self.depth, 10, .5, 'SBS')
        self.assertEqual(renderer.last_stats, {'method': 'fast', 'stereo_views_rendered': True})
        self.assertEqual(json.dumps(summary, sort_keys=True), saved)

    def test_depth_only_layouts_skip_visibility_and_report_no_rendered_views(self):
        renderer = StereoRenderer()
        with patch('stereo.render_visibility_pair', side_effect=AssertionError('not consumed')):
            for layout in ('DEPTH', 'RGBD'):
                with self.subTest(layout=layout):
                    expected = pack_output(self.frame, self.depth, self.frame, self.frame, layout)
                    np.testing.assert_array_equal(renderer.render(self.frame, self.depth, 8, .5, layout, method='visibility'), expected)
                    np.testing.assert_array_equal(render_output(self.frame, self.depth, 8, .5, layout, method='visibility'), expected)
                    self.assertEqual(renderer.last_stats, {'method': 'visibility', 'stereo_views_rendered': False})

    def test_switching_methods_and_shapes_preserves_source_and_prior_output(self):
        renderer = StereoRenderer()
        source = self.frame.copy()
        depth = self.depth.copy()
        first = renderer.render(self.frame, self.depth, 10, .5, 'SBS')
        saved = first.copy()
        renderer.render(self.frame, self.depth, 10, .5, 'SBS', method='visibility')
        small_frame = self.frame[:5, :101].copy()
        small_depth = self.depth[:5, :101].copy()
        expected = render_output(small_frame, small_depth, 10, .5, 'SBS')
        np.testing.assert_array_equal(renderer.render(small_frame, small_depth, 10, .5, 'SBS'), expected)
        np.testing.assert_array_equal(first, saved)
        np.testing.assert_array_equal(self.frame, source)
        np.testing.assert_array_equal(self.depth, depth)

    def test_invalid_method_is_rejected_in_both_apis_including_depth_exports(self):
        for method in ('quality', '', None, True):
            for layout in ('SBS', 'DEPTH'):
                for render in (render_output, StereoRenderer().render):
                    with self.subTest(method=method, layout=layout), self.assertRaisesRegex(ValueError, 'fast or visibility'):
                        render(self.frame, self.depth, 10, .5, layout, method=method)

    def test_summary_does_not_alias_prototype_eye_stats(self):
        _, _, stats = render_visibility_pair(self.frame, self.depth, 10, .5)
        summary = visibility_summary(stats)
        saved = summary['left']['hole_fraction']
        stats['left']['hole_fraction'] = 1
        self.assertEqual(summary['left']['hole_fraction'], saved)


if __name__ == '__main__':
    unittest.main()
