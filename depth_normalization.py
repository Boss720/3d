"""Opt-in, job-owned robust range tracking for raw relative depth.

This limits range pumping from a few extreme predictions. It cannot establish
metric scale or disambiguate a real scene change from a model scale change.
The caller must reset it on cuts and source-size changes.
"""
from __future__ import annotations

import numpy as np


class SceneDepthNormalizer:
    """Track 2nd/98th percentile bounds with bounded exponential updates.

    Raw model output must reach this class before per-frame min/max scaling.
    ``update_rate`` controls the new-range weight; ``max_step`` limits each
    bound's change as a fraction of the previous robust span. Defaults are an
    experimental policy, deliberately enabled only by an explicit job option.
    """

    def __init__(self, update_rate: float = .2, max_step: float = .1):
        for name, value in [('update_rate', update_rate), ('max_step', max_step)]:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{name} must be finite and in (0, 1].')
        self.update_rate = float(update_rate)
        self.max_step = float(max_step)
        self.reset()

    def reset(self):
        self._bounds = None
        self._shape = None

    def normalize(self, raw_depth):
        values = np.asarray(raw_depth, dtype=np.float32)
        if values.ndim != 2 or not values.size or not np.isfinite(values).all():
            raise ValueError('Raw depth must be a nonempty finite 2D map.')
        # float64 arithmetic avoids overflow in differences of large finite maps.
        values64 = values.astype(np.float64)
        low, high = np.percentile(values64, (2, 98))
        if self._shape != values.shape:
            self.reset()
        self._shape = values.shape
        if high - low <= 1e-6:
            # A flat prediction carries no range information. Do not let its
            # zero span trap subsequent updates at microscopic slew steps.
            return np.zeros(values.shape, dtype=np.float32)
        if self._bounds is None:
            self._bounds = (float(low), float(high))
        else:
            old_low, old_high = self._bounds
            span = max(old_high - old_low, 1e-6)
            step = self.max_step * span
            low = old_low + float(np.clip(self.update_rate * (low - old_low), -step, step))
            high = old_high + float(np.clip(self.update_rate * (high - old_high), -step, step))
            self._bounds = (low, max(high, low))
        low, high = self._bounds
        result = np.clip((values64 - low) / max(high - low, 1e-6), 0, 1)
        return np.array(result, dtype=np.float32, order='C', copy=True)
