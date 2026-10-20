"""Job-owned native-resolution video depth and optional alternate-frame flow.

Flow transports relative depth rather than predicting newly revealed geometry.
Cuts, changed frame sizes and unusable flow force a fresh neural estimate.
"""
from __future__ import annotations

import cv2
import numpy as np

from stereo import TemporalDepthFilter


class VideoDepthEstimator:
    def __init__(self, interval: int = 1, smoothing: float = .35, flow_factory=None):
        if type(interval) is not int or interval not in {1, 2}:
            raise ValueError('Depth interval must be 1 or 2.')
        self.interval = interval
        self._filter = TemporalDepthFilter(smoothing)
        self._flow_factory = flow_factory or (
            lambda: cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST))
        self._flow = None
        self._previous_guide = None
        self._previous_depth = None
        self._source_shape = None
        self.stats = {'frames': 0, 'neural_frames': 0, 'flow_frames': 0,
                      'scene_cuts': 0, 'flow_fallbacks': 0}
        self.last_method = 'neural'

    @staticmethod
    def _guide(frame, shape):
        height, width = shape
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    def _warped(self, guide):
        # remap needs current -> previous coordinates, so calculate backwards.
        if min(guide.shape[:2]) < 12:
            raise ValueError('Frame is too small for DIS flow.')
        if self._flow is None:
            self._flow = self._flow_factory()
        current = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(self._previous_guide, cv2.COLOR_BGR2GRAY)
        flow = self._flow.calc(current, previous, None)
        height, width = current.shape
        if flow is None or flow.shape != (height, width, 2) or not np.isfinite(flow).all():
            raise ValueError('Optical flow is not finite or has invalid dimensions.')
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_x, map_y = x + flow[:, :, 0], y + flow[:, :, 1]
        valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        if np.mean(valid) < .9:
            raise ValueError('Too much newly exposed content for depth reuse.')
        aligned = cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        residual = cv2.absdiff(current, aligned).astype(np.float32) / 255.0
        if float(residual.mean()) > .12 or float(np.mean(residual > .18)) > .35:
            raise ValueError('Optical flow cannot align the source appearance reliably.')
        return cv2.remap(self._previous_depth, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def process(self, frame, predict):
        depth = None
        guide = None
        shape_changed = self._source_shape != frame.shape
        if shape_changed:
            self._filter.reset()
        elif self.interval == 2 and self._previous_depth is not None:
            guide = self._guide(frame, self._previous_depth.shape)
            difference = np.max(cv2.absdiff(guide, self._previous_guide), axis=2) / 255.0
            cut = float(difference.mean()) > .24 or float(np.mean(difference > .18)) > .65
            if cut:
                self.stats['scene_cuts'] += 1
                self._filter.reset()
            elif self.stats['frames'] % 2 == 1:
                try:
                    depth = self._warped(guide)
                    self.stats['flow_frames'] += 1
                except (cv2.error, RuntimeError, ValueError, AttributeError):
                    self.stats['flow_fallbacks'] += 1
                    self._flow = None
        self.last_method = 'optical_flow' if depth is not None else 'neural'
        if depth is None:
            depth = np.asarray(predict(frame), dtype=np.float32)
            if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
                raise ValueError('Depth inference must return a nonempty finite 2D map.')
            self.stats['neural_frames'] += 1
            guide = self._guide(frame, depth.shape)
        depth = np.ascontiguousarray(np.clip(depth, 0, 1), dtype=np.float32)
        result = self._filter.apply(guide, depth)
        if self.interval == 2:
            self._previous_guide = guide.copy()
            self._previous_depth = depth.copy()
        self._source_shape = frame.shape
        self.stats['frames'] += 1
        return result
