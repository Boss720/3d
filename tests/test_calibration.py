import unittest

from calibration import fingerprint, measure_inputs


class CalibrationTests(unittest.TestCase):
    def test_measured_fastest_requires_repeated_comparable_candidates(self):
        now = [0.0]
        first = [True]
        def predict(size):
            now[0] += {256: .3, 384: .2, 518: .4}[size]
            reused = not first[0]
            first[0] = False
            return {'inference_succeeded': True, 'session_reused': reused, 'active_provider': 'CPU',
                    'input_width': size, 'input_height': size}
        result = measure_inputs(predict, {'model': 'fixture'}, clock=lambda: now[0])
        self.assertTrue(result['sufficient_evidence'])
        self.assertEqual(result['fastest_measured_input_size'], 384)
        self.assertEqual(result['candidates'][0]['warmup_seconds'], [.3])
        self.assertEqual(len(result['candidates'][0]['samples_seconds']), 2)

    def test_single_slow_inference_overrun_is_reported_without_more_work(self):
        now = [0]
        calls = []
        def predict(size):
            calls.append(size)
            now[0] += 20
            return {'session_reused': False, 'inference_succeeded': True, 'active_provider': 'CPU',
                    'input_width': size, 'input_height': size}
        report = measure_inputs(predict, {}, budget_seconds=5, clock=lambda: now[0])
        self.assertEqual(calls, [256])
        self.assertTrue(report['budget_exceeded'])
        self.assertFalse(report['sufficient_evidence'])
        self.assertIsNone(report['fastest_measured_input_size'])

    def test_fallback_failure_cannot_be_a_recommendation(self):
        result = measure_inputs(lambda size: {'fallback_reason': 'GPU unavailable'}, {})
        self.assertFalse(result['sufficient_evidence'])
        self.assertIsNone(result['fastest_measured_input_size'])
        self.assertTrue(all(candidate['error'] for candidate in result['candidates']))

    def test_fingerprint_invalidates_model_runtime_hardware_and_source(self):
        identity = {'model_hash': 'a', 'runtime': '1', 'adapter': 'a', 'source_sha256': 'a'}
        original = fingerprint(identity)
        for key in identity:
            self.assertNotEqual(original, fingerprint({**identity, key: 'changed'}))

    def test_invalid_budget_rejected(self):
        for budget in (True, 0, 31, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                measure_inputs(lambda _: {}, {}, budget)

    def test_provider_changes_reject_comparison(self):
        calls = [0]
        def predict(_):
            calls[0] += 1
            return {'active_provider': 'GPU' if calls[0] == 1 else 'CPU', 'inference_succeeded': True, 'session_reused': True}
        result = measure_inputs(predict, {})
        self.assertFalse(result['sufficient_evidence'])
        self.assertIsNone(result['fastest_measured_input_size'])

    def test_preparation_overhead_can_exhaust_budget_before_inference(self):
        calls = []
        result = measure_inputs(lambda size: calls.append(size), {}, budget_seconds=5,
                                clock=lambda: 6, started_at=0)
        self.assertEqual(calls, [])
        self.assertTrue(result['budget_exceeded'])
