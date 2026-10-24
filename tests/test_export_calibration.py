import threading
import unittest

from export_calibration import measure_exports


class ExportCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.calls = []
        self.settings = {'depth_mode': 'depth-anything-small', 'input_size': 384,
                         'quality': 'balanced', 'layout': 'sbs', 'strength': 3,
                         'resolution': 'source', 'use_depth_cache': True}
        self.identity = {'source_sha256': 'abc', 'runtime': 'fixture', 'hardware': {'gpu': 'test'},
                         'sample': {'start_seconds': 2.5, 'duration_seconds': 1.5,
                                    'scope': 'bounded_proxy', 'width': 960}}

    def run_sample(self, settings, sample, cancel_event):
        self.calls.append((settings.copy(), sample.copy()))
        self.now += {256: 2, 384: 4, 518: 6}[settings['input_size']]
        return {'completed': True, 'validated': True,
                'output': {'url': '/output/' + str(len(self.calls))},
                'execution': {'active_provider': 'CPUExecutionProvider', 'device': 'CPU',
                              'inference_succeeded': True}}

    def measure(self, callback=None, **kwargs):
        return measure_exports(callback or self.run_sample, self.identity, self.settings,
                               clock=lambda: self.now, **kwargs)

    def test_exact_candidates_same_interval_model_and_unmodified_settings(self):
        report = self.measure()
        self.assertTrue(report['sufficient_evidence'])
        self.assertEqual([(s['input_size'], s['quality']) for s, _ in self.calls],
                         [(384, 'balanced')] * 2 + [(256, 'balanced')] * 2
                         + [(518, 'balanced')] * 2 + [(384, 'fast')] * 2)
        for settings, sample in self.calls:
            self.assertEqual(settings['depth_mode'], self.settings['depth_mode'])
            self.assertEqual(settings['layout'], 'sbs')
            self.assertFalse(settings['use_depth_cache'])
            self.assertEqual(sample['start_seconds'], 2.5)
            self.assertEqual(sample['duration_seconds'], 1.5)
        self.assertTrue(self.settings['use_depth_cache'])
        self.assertEqual(report['identity'], self.identity)
        self.assertEqual(report['recommendation']['settings']['input_size'], 256)
        self.assertFalse(report['settings_applied'])
        self.assertIn('proxy', report['scope'].lower())

    def test_budget_checked_between_full_exports_and_overrun_reported(self):
        report = self.measure(budget_seconds=5)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report['elapsed_seconds'], 8)
        self.assertTrue(report['budget_exceeded'])
        self.assertFalse(report['sufficient_evidence'])
        self.assertIsNone(report['recommendation'])
        self.assertEqual(report['candidates'][0]['samples_seconds'], [4, 4])

    def test_cancel_during_sample_stops_next_export_and_suppresses_recommendation(self):
        event = threading.Event()
        def cancel(settings, sample, cancel_event):
            result = self.run_sample(settings, sample, cancel_event)
            cancel_event.set()
            return result
        report = self.measure(cancel, cancel_event=event)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report['state'], 'canceled')
        self.assertIsNone(report['recommendation'])

    def test_canceled_before_start_has_no_callback(self):
        event = threading.Event()
        event.set()
        report = self.measure(cancel_event=event)
        self.assertEqual(self.calls, [])
        self.assertEqual(report['state'], 'canceled')

    def test_successful_encoding_without_validation_is_not_evidence(self):
        def unvalidated(settings, sample, event):
            result = self.run_sample(settings, sample, event)
            result['validated'] = False
            return result
        report = self.measure(unvalidated)
        self.assertFalse(report['sufficient_evidence'])
        self.assertIsNone(report['recommendation'])
        self.assertTrue(report['candidates'][0]['error'])
        self.assertEqual(report['candidates'][0]['samples_seconds'], [])

    def test_wall_time_includes_finalization_and_validation_not_stage_sum(self):
        def complete(settings, sample, event):
            result = self.run_sample(settings, sample, event)
            self.now += 3.0  # Simulate encoder finalization and output verification.
            result['timings'] = {'depth': .01}
            return result
        report = self.measure(complete)
        self.assertEqual(report['candidates'][0]['samples_seconds'], [7, 7])
        self.assertEqual(len(report['candidates'][0]['outputs']), 2)
        self.assertEqual(report['candidates'][0]['runs'][0]['timing_context'], 'first_export')

    def test_provider_change_or_fallback_invalidates_recommendation(self):
        for fallback in (True, False):
            with self.subTest(fallback=fallback):
                self.now, self.calls = 0, []
                def changed(settings, sample, event):
                    result = self.run_sample(settings, sample, event)
                    if len(self.calls) > 1:
                        result['execution']['fallback_reason' if fallback else 'device'] = 'changed'
                    return result
                report = self.measure(changed)
                self.assertFalse(report['sufficient_evidence'])
                self.assertIsNone(report['recommendation'])

    def test_callback_mutation_cannot_change_remaining_candidates_or_interval(self):
        def mutate(settings, sample, event):
            result = self.run_sample(settings, sample, event)
            settings['depth_mode'] = 'wrong'
            sample['start_seconds'] = 99
            return result
        self.measure(mutate)
        self.assertTrue(all(s['depth_mode'] == self.settings['depth_mode'] for s, _ in self.calls))
        self.assertTrue(all(s['start_seconds'] == 2.5 for _, s in self.calls))

    def test_invalid_budget_and_interval_rejected_before_work(self):
        for value in (True, 0, 121, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.measure(budget_seconds=value)
        for value in (-1, True, float('nan')):
            self.identity['sample']['start_seconds'] = value
            with self.assertRaises(ValueError):
                self.measure()
        self.assertEqual(self.calls, [])

    def test_baseline_deduplicated_and_preparation_budget_included(self):
        self.settings.update(input_size=256, quality='fast')
        report = self.measure()
        self.assertEqual(len(report['candidates']), 2)
        self.calls = []
        report = self.measure(budget_seconds=1, started_at=0)
        self.assertEqual(self.calls, [])
        self.assertFalse(report['sufficient_evidence'])

    def test_failed_callback_records_failed_work_and_starts_no_more_exports(self):
        def fail(settings, sample, event):
            self.calls.append((settings, sample))
            self.now += 9
            raise RuntimeError('Encoder finalization failed; adapter removed partial output')
        report = self.measure(fail)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(report['attempts'], 1)
        self.assertEqual(report['elapsed_seconds'], 9)
        self.assertEqual(report['candidates'][0]['runs'][0]['wall_seconds'], 9)
        self.assertEqual(report['candidates'][0]['outputs'], [])
        self.assertIsNone(report['recommendation'])

    def test_missing_output_or_execution_cannot_be_complete_export_evidence(self):
        for field in ('output', 'execution'):
            with self.subTest(field=field):
                def missing(settings, sample, event):
                    result = self.run_sample(settings, sample, event)
                    result.pop(field)
                    return result
                report = self.measure(missing)
                self.assertFalse(report['sufficient_evidence'])
                self.assertIsNone(report['recommendation'])

    def test_identity_and_settings_are_fingerprinted_and_retained(self):
        report = self.measure()
        self.identity['hardware']['gpu'] = 'different-device'
        changed = self.measure()
        self.assertNotEqual(report['fingerprint'], changed['fingerprint'])
        self.assertEqual(report['identity']['hardware']['gpu'], 'test')
        self.assertTrue(report['selected_settings']['use_depth_cache'])
        self.assertFalse(report['effective_baseline_settings']['use_depth_cache'])

    def test_high_quality_baseline_is_preserved_except_explicit_fast_candidate(self):
        self.settings['quality'] = 'high'
        report = self.measure()
        self.assertEqual([c['settings']['quality'] for c in report['candidates']],
                         ['high', 'high', 'high', 'fast'])
        self.assertEqual(self.settings['quality'], 'high')

    def test_progress_work_can_exhaust_budget_without_starting_an_export(self):
        updates = []
        def progress(snapshot):
            updates.append(snapshot)
            self.now += 2
        report = self.measure(budget_seconds=1, on_progress=progress)
        self.assertEqual(self.calls, [])
        self.assertEqual(report['attempts'], 0)
        self.assertTrue(report['budget_exceeded'])
        self.assertEqual(updates[0]['total_samples'], 8)


if __name__ == '__main__':
    unittest.main()
