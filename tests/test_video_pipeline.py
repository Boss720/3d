import threading
import unittest

from video_pipeline import PipelineCancelled, run_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.before = set(threading.enumerate())

    def tearDown(self):
        self.assertEqual(set(threading.enumerate()) - self.before, set())

    def test_empty_and_ordered_output_use_one_inference_thread(self):
        self.assertEqual(run_pipeline([], lambda frame: frame, lambda *_: None, threading.Event()), 0)
        caller = threading.get_ident()
        producer_threads, inference_threads, consumer_threads = [], [], []
        received = []

        def source():
            for frame in range(40):
                producer_threads.append(threading.get_ident())
                yield frame

        def infer(frame):
            inference_threads.append(threading.get_ident())
            return frame * 10

        def consume(frame, depth):
            consumer_threads.append(threading.get_ident())
            received.append((frame, depth))

        self.assertEqual(run_pipeline(source(), infer, consume, threading.Event()), 40)
        self.assertEqual(received, [(frame, frame * 10) for frame in range(40)])
        self.assertEqual(set(inference_threads), {caller})
        self.assertEqual(len(set(producer_threads)), 1)
        self.assertEqual(len(set(consumer_threads)), 1)
        self.assertNotIn(caller, producer_threads + consumer_threads)
        self.assertNotEqual(producer_threads[0], consumer_threads[0])

    def test_three_stages_overlap_using_events(self):
        consuming = threading.Event()
        inferring = threading.Event()
        decoded = threading.Event()

        def source():
            yield 0
            yield 1
            self.assertTrue(consuming.wait(3), 'Consumer never started')
            self.assertTrue(inferring.wait(3), 'Inference did not overlap consumption')
            decoded.set()
            yield 2

        def infer(frame):
            if frame == 1:
                self.assertTrue(consuming.wait(3), 'Consumer did not overlap inference')
                inferring.set()
                self.assertTrue(decoded.wait(3), 'Decode did not overlap both stages')
            return frame

        def consume(frame, depth):
            if frame == 0:
                consuming.set()
                self.assertTrue(decoded.wait(3), 'Decode did not run during consumption')

        self.assertEqual(run_pipeline(source(), infer, consume, threading.Event()), 3)

    def test_queues_bound_producer_and_stop_on_full_queue_cancellation(self):
        cancel = threading.Event()
        decoded = []
        # Frame 0 is being consumed, frame 1 is in infer, two raw frames
        # can be queued and one more can be held by the decoder's put.
        producer_blocked = threading.Event()
        consuming = threading.Event()

        def source():
            for frame in range(10000):
                decoded.append(frame)
                if frame == 4:
                    producer_blocked.set()
                yield frame

        def infer(frame):
            if frame == 1:
                self.assertTrue(consuming.wait(3))
                self.assertTrue(producer_blocked.wait(3))
                self.assertLessEqual(len(decoded), 5)
                cancel.set()
            return frame

        def consume(frame, depth):
            consuming.set()
            self.assertTrue(cancel.wait(3))

        with self.assertRaises(PipelineCancelled):
            run_pipeline(source(), infer, consume, cancel, queue_size=2)
        self.assertLessEqual(len(decoded), 5)

    def test_cancellation_with_both_queues_full(self):
        cancel = threading.Event()
        consume_started = threading.Event()
        sixth_decoded = threading.Event()
        fourth_inferred = threading.Event()
        consumed = []

        def source():
            for frame in range(10000):
                if frame == 6:
                    sixth_decoded.set()
                yield frame

        def infer(frame):
            if frame == 3:
                fourth_inferred.set()
            return frame

        def consume(frame, depth):
            consume_started.set()
            self.assertTrue(fourth_inferred.wait(3))
            self.assertTrue(sixth_decoded.wait(3))
            cancel.set()
            consumed.append(frame)

        with self.assertRaises(PipelineCancelled):
            run_pipeline(source(), infer, consume, cancel, queue_size=2)
        self.assertTrue(consume_started.is_set())
        self.assertEqual(consumed, [0])

    def test_pre_cancel_does_not_touch_any_callback_or_source(self):
        cancel = threading.Event()
        cancel.set()

        class Untouched:
            def __iter__(self):
                self.fail_if_called()

            def fail_if_called(self, *_):
                raise AssertionError('Pre-canceled pipeline touched its source')

        source = Untouched()
        with self.assertRaises(PipelineCancelled):
            run_pipeline(source, source.fail_if_called, source.fail_if_called, cancel)

    def test_each_stage_propagates_original_exception_and_closes_source(self):
        for stage in ('iter', 'source', 'infer', 'consume', 'close'):
            with self.subTest(stage=stage):
                error = RuntimeError(stage)
                closed_on = []
                producer_on = []

                class Source:
                    def __iter__(self):
                        producer_on.append(threading.get_ident())
                        if stage == 'iter':
                            raise error
                        return self

                    def __next__(self):
                        if stage == 'source':
                            raise error
                        return 1

                    def close(self):
                        closed_on.append(threading.get_ident())
                        if stage == 'close':
                            raise error

                def infer(frame):
                    if stage == 'infer':
                        raise error
                    return frame

                def consume(frame, depth):
                    if stage == 'consume':
                        raise error
                    if stage == 'close':
                        # Trigger source closure without replacing its error.
                        cancel.set()

                cancel = threading.Event()
                with self.assertRaises(RuntimeError) as raised:
                    run_pipeline(Source(), infer, consume, cancel)
                self.assertIs(raised.exception, error)
                self.assertEqual(closed_on, [] if stage == 'iter' else producer_on)

    def test_source_cleanup_and_eof_finish_before_return(self):
        calls = []

        def source():
            try:
                yield 1
                yield 2
            finally:
                calls.append(('close', threading.get_ident()))

        def consume(frame, depth):
            calls.append((frame, threading.get_ident()))

        self.assertEqual(run_pipeline(source(), lambda frame: frame, consume, threading.Event()), 2)
        self.assertEqual([frame for frame, _ in calls if frame != 'close'], [1, 2])
        self.assertEqual(sum(frame == 'close' for frame, _ in calls), 1)
        self.assertNotEqual(next(thread for frame, thread in calls if frame == 'close'), threading.get_ident())

    def test_consumer_error_unblocks_full_queues_and_stops_future_writes(self):
        error = BrokenPipeError('Encoder closed')
        inferred_four = threading.Event()
        decoded_seven = threading.Event()
        consumed = []

        def source():
            for frame in range(10000):
                if frame == 6:
                    decoded_seven.set()
                yield frame

        def infer(frame):
            if frame == 3:
                inferred_four.set()
            return frame

        def consume(frame, depth):
            consumed.append(frame)
            self.assertTrue(inferred_four.wait(3))
            self.assertTrue(decoded_seven.wait(3))
            raise error

        with self.assertRaises(BrokenPipeError) as raised:
            run_pipeline(source(), infer, consume, threading.Event())
        self.assertIs(raised.exception, error)
        self.assertEqual(consumed, [0])

    def test_cancel_joins_every_in_progress_callback_before_return(self):
        for stage in ('source', 'infer', 'consume'):
            with self.subTest(stage=stage):
                cancel = threading.Event()
                entered = threading.Event()
                noticed_cancel = threading.Event()
                release = threading.Event()
                finished = threading.Event()
                returned = threading.Event()
                failures = []

                def block():
                    entered.set()
                    self.assertTrue(cancel.wait(3))
                    noticed_cancel.set()
                    self.assertTrue(release.wait(3))
                    finished.set()

                def source():
                    if stage == 'source':
                        block()
                    yield 1

                def infer(frame):
                    if stage == 'infer':
                        block()
                    return frame

                def consume(frame, depth):
                    if stage == 'consume':
                        block()

                def run():
                    try:
                        run_pipeline(source(), infer, consume, cancel)
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        returned.set()

                runner = threading.Thread(target=run)
                runner.start()
                try:
                    self.assertTrue(entered.wait(3))
                    cancel.set()
                    self.assertTrue(noticed_cancel.wait(3))
                    self.assertFalse(returned.is_set())
                    self.assertTrue(runner.is_alive())
                finally:
                    cancel.set()
                    release.set()
                    runner.join(3)
                self.assertFalse(runner.is_alive())
                self.assertTrue(finished.is_set())
                self.assertTrue(returned.is_set())
                self.assertEqual(len(failures), 1)
                self.assertIsInstance(failures[0], PipelineCancelled)

    def test_cleanup_failure_does_not_mask_the_original_stage_error(self):
        error = ValueError('inference failed')
        cleanup = RuntimeError('cleanup failed')

        def source():
            try:
                while True:
                    yield 1
            finally:
                raise cleanup

        def infer(frame):
            raise error

        with self.assertRaises(ValueError) as raised:
            run_pipeline(source(), infer, lambda *_: None, threading.Event())
        self.assertIs(raised.exception, error)

    def test_invalid_queue_capacity_never_creates_threads(self):
        for size in (0, -1, 1.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                run_pipeline([], lambda frame: frame, lambda *_: None, threading.Event(), queue_size=size)


if __name__ == '__main__':
    unittest.main()
