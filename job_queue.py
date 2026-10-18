"""FIFO task queue with atomic pending-only edits and normal join accounting."""
from __future__ import annotations

import queue
import threading


class EditableJobQueue(queue.Queue):
    def __init__(self, maxsize: int = 0):
        super().__init__(maxsize)
        # Queue.get/put, condition waits and application edits share this lock.
        # Reentrancy lets the API hold queue-before-jobs locks around a mutation.
        self.mutex = threading.RLock()
        self.lock = self.mutex
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.all_tasks_done = threading.Condition(self.mutex)

    def pending_ids(self) -> tuple:
        with self.mutex:
            return tuple(self.queue)

    def reorder_subset(self, order, eligible) -> None:
        """Replace only eligible slots, leaving clips and other slots fixed."""
        with self.mutex:
            eligible = tuple(eligible)
            order = tuple(order)
            eligible_set = set(eligible)
            if len(order) != len(eligible) or len(set(order)) != len(order) or set(order) != eligible_set:
                raise ValueError('The pending queue changed.')
            slots = [index for index, item in enumerate(self.queue) if item in eligible_set]
            if len(slots) != len(eligible) or any(item not in self.queue for item in eligible):
                raise ValueError('The pending queue changed.')
            replacement = list(self.queue)
            for index, item in zip(slots, order):
                replacement[index] = item
            self.queue.clear()
            self.queue.extend(replacement)

    def remove_pending(self, item) -> None:
        """Remove an unclaimed task and acknowledge precisely that task."""
        with self.mutex:
            if item not in self.queue:
                raise ValueError('The pending queue changed.')
            if self.unfinished_tasks <= 0:
                raise ValueError('Queue task accounting is inconsistent.')
            self.queue.remove(item)
            self.unfinished_tasks -= 1
            if self.unfinished_tasks == 0:
                self.all_tasks_done.notify_all()
            self.not_full.notify()
