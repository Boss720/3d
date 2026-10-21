import threading
import unittest
from unittest.mock import patch

from video_pipeline import PipelineCancelled, run_pipeline


class PipelineShutdownRegressions(unittest.TestCase):
    def test_cancel_during_final_render_join_stops_owned_io(self):
        cancel = threading.Event()
        rendering = threading.Event()
        joining_render = threading.Event()
        release_io = threading.Event()
        stopped = threading.Event()
        stop_calls = []
        errors = []
        original_join = threading.Thread.join

        def consume(*_):
            rendering.set()
            release_io.wait(5)

        def stop_io():
            stop_calls.append(threading.get_ident())
            stopped.set()
            release_io.set()

        def observe_join(worker, *args, **kwargs):
            if worker.name == 'video-pipeline-render':
                joining_render.set()
            return original_join(worker, *args, **kwargs)

        def run():
            try:
                run_pipeline([1], lambda frame: frame, consume, cancel, on_stop=stop_io)
            except BaseException as error:
                errors.append(error)

        runner = threading.Thread(target=run)
        with patch.object(threading.Thread, 'join', observe_join):
            runner.start()
            try:
                self.assertTrue(rendering.wait(2))
                self.assertTrue(joining_render.wait(2))
                cancel.set()
                self.assertTrue(stopped.wait(1), 'Late cancellation must interrupt owned I/O while joining')
            finally:
                release_io.set()
                original_join(runner, 3)
        self.assertFalse(runner.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], PipelineCancelled)
        self.assertEqual(len(stop_calls), 1)

    def test_final_render_failure_calls_stop_hook_and_preserves_error(self):
        joining_render = threading.Event()
        stopped = []
        original_error = BrokenPipeError('last encoder write failed')
        original_join = threading.Thread.join

        def consume(*_):
            if not joining_render.wait(2):
                raise AssertionError('Pipeline did not reach its final join')
            raise original_error

        def observe_join(worker, *args, **kwargs):
            if worker.name == 'video-pipeline-render':
                joining_render.set()
            return original_join(worker, *args, **kwargs)

        with patch.object(threading.Thread, 'join', observe_join):
            with self.assertRaises(BrokenPipeError) as raised:
                run_pipeline([1], lambda frame: frame, consume, threading.Event(),
                             on_stop=lambda: stopped.append(True))
        self.assertIs(raised.exception, original_error)
        self.assertEqual(stopped, [True])


if __name__ == '__main__':
    unittest.main()
