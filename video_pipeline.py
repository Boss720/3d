"""Bounded, ordered CPU decode / inference / CPU render cooperation."""

from __future__ import annotations

import queue
import threading
from contextlib import nullcontext
from collections.abc import Callable, Iterable
from typing import TypeVar


Frame = TypeVar('Frame')
Value = TypeVar('Value')
_POLL_SECONDS = 0.05


class PipelineCancelled(Exception):
    """The caller canceled a pipeline, after its workers stopped."""


class _Stopped(Exception):
    """Internal queue wakeup when another stage stops."""


def run_pipeline(
    source: Iterable[Frame],
    infer: Callable[[Frame], Value],
    consume: Callable[[Frame, Value], None],
    cancel_event: threading.Event,
    queue_size: int = 2,
    timings=None,
    on_stop=None,
) -> int:
    """Process frames in order with exactly one inference call at a time.

    Decode/resize (``source``) and render/write (``consume``) each own one
    worker. Inference stays on the calling thread so one GPU session can
    overlap both CPU stages. Each queue holds at most ``queue_size`` items;
    workers may additionally hold their current frame. No video is buffered
    in full. A stateful temporal filter belongs in ``consume``.

    Exceptions retain their original identity. On failure or cancellation,
    queued work is discarded and both non-daemon workers are joined before
    returning or raising. The source iterator is closed by its producer.
    Callbacks must be finite or cooperate with cancellation: Python cannot
    interrupt a blocked decoder, GPU call, or encoder write. The owner should
    terminate its encoder on cancellation to release a blocked pipe write.
    This function waits for any in-progress callback rather than allowing a
    worker to continue writing after return.

    on_stop, when provided, stops caller-owned I/O before workers are joined
    after failure/cancellation. It must be finite and safe to call once.
    """
    if isinstance(queue_size, bool) or not isinstance(queue_size, int) or queue_size < 1:
        raise ValueError('queue_size must be a positive integer')
    if cancel_event.is_set():
        raise PipelineCancelled()

    decoded: queue.Queue = queue.Queue(maxsize=queue_size)
    inferred: queue.Queue = queue.Queue(maxsize=queue_size)
    end = object()
    stop = threading.Event()
    failure_lock = threading.Lock()
    failures: list[BaseException] = []
    processed = 0

    def check_running() -> None:
        if stop.is_set() or cancel_event.is_set():
            raise _Stopped()

    def fail(error: BaseException) -> None:
        with failure_lock:
            if not failures:
                failures.append(error)
        stop.set()

    def put(target: queue.Queue, item: object) -> None:
        name = 'queue_decode_put' if target is decoded else 'queue_render_put'
        with timings.measure(name) if timings is not None else nullcontext():
            while True:
                check_running()
                try:
                    target.put(item, timeout=_POLL_SECONDS)
                    return
                except queue.Full:
                    pass

    def get(target: queue.Queue):
        name = 'queue_decode_get' if target is decoded else 'queue_render_get'
        with timings.measure(name) if timings is not None else nullcontext():
            while True:
                check_running()
                try:
                    item = target.get(timeout=_POLL_SECONDS)
                    check_running()
                    return item
                except queue.Empty:
                    pass

    def produce() -> None:
        iterator = None
        try:
            check_running()
            iterator = iter(source)
            while True:
                check_running()
                try:
                    frame = next(iterator)
                except StopIteration:
                    break
                put(decoded, frame)
        except _Stopped:
            pass
        except BaseException as error:
            fail(error)
        finally:
            # Only this thread can be inside next(), so close never races
            # with an executing generator when inference or rendering fails.
            if iterator is not None:
                try:
                    close = getattr(iterator, 'close', None)
                    if close is not None:
                        close()
                except BaseException as error:
                    fail(error)
        try:
            put(decoded, end)
        except _Stopped:
            pass

    def render() -> None:
        nonlocal processed
        try:
            while True:
                item = get(inferred)
                if item is end:
                    return
                frame, value = item
                check_running()
                consume(frame, value)
                processed += 1
        except _Stopped:
            pass
        except BaseException as error:
            fail(error)

    producer = threading.Thread(target=produce, name='video-pipeline-decode', daemon=False)
    consumer = threading.Thread(target=render, name='video-pipeline-render', daemon=False)
    started: list[threading.Thread] = []
    completed_inference = False
    try:
        for worker in (consumer, producer):
            worker.start()
            started.append(worker)
        while True:
            frame = get(decoded)
            if frame is end:
                put(inferred, end)
                completed_inference = True
                break
            check_running()
            value = infer(frame)
            put(inferred, (frame, value))
    except _Stopped:
        pass
    except BaseException as error:
        fail(error)
    finally:
        if not completed_inference:
            stop.set()
        if (stop.is_set() or cancel_event.is_set()) and on_stop is not None:
            try:
                on_stop()
            except BaseException as error:
                fail(error)
        for worker in started:
            worker.join()

    if failures:
        raise failures[0]
    if cancel_event.is_set():
        raise PipelineCancelled()
    return processed
