"""Bounded process-local cache of raw preview depth, before presentation edits."""
from __future__ import annotations

import copy
import threading
from collections import OrderedDict

import numpy as np


class PreviewDepthCache:
    def __init__(self, max_bytes: int = 32 * 1024 * 1024, max_entries: int = 8):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.lock = threading.RLock()
        self._entries = OrderedDict()
        self._bytes = 0

    def clear(self):
        with self.lock:
            self._entries.clear()
            self._bytes = 0

    def get(self, key):
        with self.lock:
            value = self._entries.get(key)
            if value is None:
                return None
            self._entries.move_to_end(key)
            depth, execution = value
            return depth.copy(), copy.deepcopy(execution)

    def put(self, key, depth, execution):
        # Only numerical depth buffers count toward this explicit buffer budget;
        # metadata and temporary rendering/encoding buffers are additional RAM.
        with self.lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[0].nbytes
            if depth.nbytes > self.max_bytes or self.max_entries < 1:
                return
            self._entries[key] = (np.array(depth, copy=True), copy.deepcopy(execution))
            self._bytes += depth.nbytes
            while self._bytes > self.max_bytes or len(self._entries) > self.max_entries:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed[0].nbytes

    def info(self):
        with self.lock:
            return {"entries": len(self._entries), "depth_bytes": self._bytes,
                    "max_depth_bytes": self.max_bytes, "max_entries": self.max_entries}
