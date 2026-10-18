"""Thread-safe stage samples; overlapping service times are not wall time."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager


class StageTimings:
    def __init__(self):
        self._lock = threading.Lock()
        self._stages = {}
        self._queues = {}

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = max(0.0, time.perf_counter() - started)
            with self._lock:
                collection = self._queues if name.startswith('queue_') else self._stages
                stage = collection.setdefault(name, {"seconds": 0.0, "calls": 0})
                stage["seconds"] += elapsed
                stage["calls"] += 1

    def snapshot(self):
        with self._lock:
            return {"schema_version": 1, "stages": {name: dict(value) for name, value in self._stages.items()},
                    "overlap_warning": "Stage service times may overlap; do not sum them as wall time.",
                    "queue_stalls": {name: dict(value) for name, value in self._queues.items()} or None,
                    "queue_wait_note": "Queue wait durations include queue-operation overhead and overlap stage work."}
