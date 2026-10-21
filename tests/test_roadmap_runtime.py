"""Independent roadmap runtime checks; no downloads, devices or encoder runs."""
import hashlib
import json
import os
import queue
import threading
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import main
from depth_engine import MODEL_SPECS, ModelManager
from telemetry import StageTimings
from video_pipeline import PipelineCancelled, run_pipeline


SMALL = 'depth-anything-v2-small'


class RoadmapRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.folder = main.RUNTIME / 'roadmap-runtime-tests' / uuid.uuid4().hex
        self.folder.mkdir(parents=True)

    def tearDown(self):
        quarantine = self.folder.parents[1] / '_quarantine' / 'roadmap-runtime-tests' / self.folder.name
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.folder, quarantine)

    def test_prediction_reports_cold_and_reused_session_timings_per_call(self):
        manager = ModelManager(self.folder / 'models')
        data = b'accepted local fake model'
        spec = dict(MODEL_SPECS[SMALL], size=len(data), sha256=hashlib.sha256(data).hexdigest())
        model = manager._path(SMALL)
        model.parent.mkdir(parents=True)
        model.write_bytes(data)
        manager._receipt(SMALL).write_text(json.dumps({'sha256': spec['sha256'], 'revision': spec['revision']}))
        session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(shape=[1, 3, 'height', 'width'], type='tensor(float)', name='pixels')],
            run=lambda *_: [np.array([[[0., 1.], [2., 3.]]], dtype=np.float32)],
        )
        clock = [10., 10.25, 20., 20.5, 30., 30.125]
        with patch.dict(MODEL_SPECS, {SMALL: spec}), patch.object(manager, 'runtime_info', return_value={'available': True}), patch.object(manager, '_new_session', return_value=session) as created, patch('depth_engine.time.perf_counter', side_effect=clock):
            first_depth, first = manager.predict_with_info(np.zeros((8, 16, 3), np.uint8), SMALL, 256, 'cpu')
            second_depth, second = manager.predict_with_info(np.zeros((8, 16, 3), np.uint8), SMALL, 256, 'cpu')
        created.assert_called_once()
        np.testing.assert_array_equal(first_depth, second_depth)
        self.assertFalse(first['session_reused'])
        self.assertEqual(first['session_initialization_seconds'], .25)
        self.assertEqual(first['inference_pipeline_seconds'], .5)
        self.assertTrue(second['session_reused'])
        self.assertEqual(second['session_initialization_seconds'], 0)
        self.assertEqual(second['inference_pipeline_seconds'], .125)
        self.assertTrue(first['inference_succeeded'])
        self.assertTrue(second['inference_succeeded'])

    def test_failed_session_initialization_reports_elapsed_attempt_time(self):
        manager = ModelManager(self.folder / 'failed-model')
        with patch.object(manager, '_hash', return_value=MODEL_SPECS[SMALL]['sha256']), patch.object(manager, '_new_session', side_effect=RuntimeError('local fake initialization failure')), patch('depth_engine.time.perf_counter', side_effect=[10., 12.5]):
            with self.assertRaisesRegex(RuntimeError, 'local fake initialization failure'):
                manager._run(np.zeros((8, 16, 3), np.uint8), SMALL, 256, 'CPUExecutionProvider', None)
        info = manager.execution_info()
        self.assertFalse(info['session_reused'])
        self.assertEqual(info['session_initialization_seconds'], 2.5)
        self.assertIsNone(info['inference_pipeline_seconds'])

    def test_export_processing_transition_waits_for_preview_admission_gate(self):
        source = self.folder / 'source.png'
        source.write_bytes(b'local source; preflight mocked')
        job = main.Job(id=uuid.uuid4().hex, filename='source.png', input_path=source,
                       media_type='image', layout='SBS', strength=5, convergence=.5,
                       depth_mode=SMALL, input_size=256, resolution='source', quality='fast', invert_depth=False)
        attempted = threading.Event()
        gate = threading.RLock()

        class ObservedGate:
            def __enter__(self):
                attempted.set()
                gate.acquire()

            def __exit__(self, *_):
                gate.release()

        states_at_preflight = []
        def preflight(*_, **_kwargs):
            states_at_preflight.append(job.state)
            return {'estimate_is_guarantee': False}

        def convert(actual_job, _net):
            target = main.output_path_for(actual_job)
            target.write_bytes(b'owned mock output')
            return target

        with patch.object(main, 'JOBS', {job.id: job}), patch.object(main.PREVIEW_CACHE, 'lock', ObservedGate()), patch.object(main, 'OUTPUTS', self.folder), patch.object(main, 'inspect_media', side_effect=preflight) as inspected, patch.object(main.DEPTH_ENGINE, 'validate') as validated, patch.object(main, 'run_image', side_effect=convert):
            runner = threading.Thread(target=main.process_job, args=(job.id,), name='roadmap-test-export')
            try:
                with gate:
                    runner.start()
                    self.assertTrue(attempted.wait(3), 'Export never attempted the preview admission gate')
                    self.assertEqual(job.state, 'queued')
                    inspected.assert_not_called()
                    validated.assert_not_called()
            finally:
                runner.join(3)
            self.assertFalse(runner.is_alive(), 'Export did not finish after preview released the gate')
        self.assertEqual(states_at_preflight, ['processing'])
        self.assertEqual(job.state, 'completed', job.error)

    def test_canceled_full_queues_keep_timing_samples_and_join_workers(self):
        cancel = threading.Event()
        render_entered = threading.Event()
        decode_full = threading.Event()
        render_full = threading.Event()
        source_closed = threading.Event()
        queues = []
        consumed = []
        failures = []
        original_queue = queue.Queue
        before = {thread for thread in threading.enumerate() if thread.name.startswith('video-pipeline-')}

        class ObservedQueue(original_queue):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.index = len(queues)
                queues.append(self)

            def put(self, item, *args, **kwargs):
                if self.full():
                    (decode_full if self.index == 0 else render_full).set()
                return super().put(item, *args, **kwargs)

        def source():
            try:
                yield from range(1000)
            finally:
                source_closed.set()

        def consume(frame, value):
            consumed.append((frame, value))
            render_entered.set()
            if not cancel.wait(3):
                raise AssertionError('Test did not release blocked render callback')

        timings = StageTimings()
        def run():
            try:
                run_pipeline(source(), lambda frame: frame * 10, consume, cancel, queue_size=1, timings=timings)
            except BaseException as error:
                failures.append(error)

        with patch('video_pipeline.queue.Queue', ObservedQueue):
            runner = threading.Thread(target=run, name='roadmap-test-pipeline')
            runner.start()
            try:
                self.assertTrue(render_entered.wait(3))
                self.assertTrue(render_full.wait(3), 'Inference never encountered render backpressure')
                self.assertTrue(decode_full.wait(3), 'Decode never encountered bounded-queue backpressure')
            finally:
                cancel.set()
                runner.join(3)
        self.assertFalse(runner.is_alive())
        self.assertTrue(source_closed.is_set())
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], PipelineCancelled)
        self.assertEqual(consumed, [(0, 0)])
        self.assertTrue(all(target.maxsize == 1 and target.qsize() <= 1 for target in queues))
        self.assertEqual({thread for thread in threading.enumerate() if thread.name.startswith('video-pipeline-')}, before)
        report = timings.snapshot()
        self.assertIsNotNone(report['queue_stalls'])
        for name in ('queue_decode_put', 'queue_decode_get', 'queue_render_put', 'queue_render_get'):
            self.assertGreater(report['queue_stalls'][name]['calls'], 0)
            self.assertGreaterEqual(report['queue_stalls'][name]['seconds'], 0)
        self.assertEqual(report['stages'], {})
        self.assertIn('overlap', report['queue_wait_note'])


if __name__ == '__main__':
    unittest.main()
