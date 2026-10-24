"""Job-owned native-resolution video depth and optional optical-flow reuse.

Interval 0 is opt-in adaptive reuse, bounded by a maximum propagation age.
Flow transports relative depth; appearance and consistency checks conservatively
request fresh inference for visible changes. Similar-looking new geometry can
remain undetectable, so reuse is never an unlimited substitute for inference.
"""
from __future__ import annotations

import cv2
import numpy as np

from stereo import TemporalDepthFilter


class VideoDepthEstimator:
    def __init__(self, interval: int = 1, smoothing: float = .35, flow_factory=None,
                 normalizer=None, max_propagation_age: int = 3):
        if type(interval) is not int or interval not in {0, 1, 2}:
            raise ValueError('Depth interval must be 0 (adaptive), 1 or 2.')
        if type(max_propagation_age) is not int or not 1 <= max_propagation_age <= 30:
            raise ValueError('Maximum propagation age must be an integer from 1 to 30.')
        self.interval = interval
        self.normalizer = normalizer
        self.max_propagation_age = max_propagation_age
        self._propagation_age = 0
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

    def _is_cut(self, guide):
        difference = np.max(cv2.absdiff(guide, self._previous_guide), axis=2).astype(np.float32) / 255.0
        if float(difference.mean()) > .24 or float(np.mean(difference > .18)) > .65:
            return True
        # A large histogram change catches lower-contrast edits too. Require
        # widespread pixel change so a small entering object is not a cut.
        if float(difference.mean()) <= .08 or float(np.mean(difference > .06)) <= .65:
            return False
        histograms = [cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
                      for image in (guide, self._previous_guide)]
        for histogram in histograms:
            cv2.normalize(histogram, histogram, norm_type=cv2.NORM_L1)
        return cv2.compareHist(*histograms, cv2.HISTCMP_BHATTACHARYYA) > .75

    @staticmethod
    def _validate_flow(flow, height, width):
        if not isinstance(flow, np.ndarray) or flow.shape != (height, width, 2) or not np.issubdtype(flow.dtype, np.number) or np.iscomplexobj(flow) or not np.isfinite(flow).all():
            raise ValueError('Optical flow is not finite or has invalid dimensions.')
        with np.errstate(over='ignore', invalid='ignore'):
            converted = np.asarray(flow, dtype=np.float32)
        if not np.isfinite(converted).all():
            raise ValueError('Optical flow exceeds the supported finite range.')
        return converted

    def _warped(self, guide):
        # remap needs current -> previous coordinates, so calculate backwards.
        if min(guide.shape[:2]) < 12:
            raise ValueError('Frame is too small for DIS flow.')
        if self._flow is None:
            self._flow = self._flow_factory()
        current = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        previous = cv2.cvtColor(self._previous_guide, cv2.COLOR_BGR2GRAY)
        height, width = current.shape
        flow = self._validate_flow(self._flow.calc(current, previous, None), height, width)
        x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        map_x, map_y = x + flow[:, :, 0], y + flow[:, :, 1]
        valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
        invalid_limit = .01 if self.interval == 0 else .1
        if float(np.mean(~valid)) > invalid_limit:
            raise ValueError('Too much newly exposed content for depth reuse.')
        aligned = cv2.remap(self._previous_guide, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # Color residuals detect small moving objects even when global mean
        # error is tiny or two colors happen to have equal luminance.
        residual = np.max(cv2.absdiff(guide, aligned), axis=2).astype(np.float32) / 255.0
        changed = (residual > .18) & valid
        local_limit = max(4, int(height * width * .002))
        if float(residual.mean()) > .12 or int(np.count_nonzero(changed)) >= local_limit:
            raise ValueError('Optical flow cannot align local source appearance reliably.')
        if self.interval == 0:
            # Forward/backward disagreement exposes occlusion and invalid
            # correspondences even when a patch has plausible appearance.
            forward = self._validate_flow(self._flow.calc(previous, current, None), height, width)
            aligned_forward = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            disagreement = np.linalg.norm(flow + aligned_forward, axis=2)
            if np.count_nonzero((disagreement > 1.5) & valid) >= local_limit:
                raise ValueError('Optical flow correspondence is inconsistent.')
        return cv2.remap(self._previous_depth, map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def process(self, frame, predict):
        depth = None
        guide = None
        shape_changed = self._source_shape != frame.shape
        cut = False
        if not shape_changed and self._previous_depth is not None:
            guide = self._guide(frame, self._previous_depth.shape)
            cut = self._is_cut(guide)
        if shape_changed or cut:
            self._filter.reset()
            self._flow = None
            self._propagation_age = 0
            if self.normalizer is not None:
                self.normalizer.reset()
            if cut:
                self.stats['scene_cuts'] += 1
        elif self._previous_depth is not None:
            reuse = (self.interval == 2 and self.stats['frames'] % 2 == 1) or (
                self.interval == 0 and self._propagation_age < self.max_propagation_age)
            if reuse:
                try:
                    depth = self._warped(guide)
                    self.stats['flow_frames'] += 1
                    self._propagation_age += 1
                except (cv2.error, RuntimeError, ValueError, AttributeError, TypeError):
                    self.stats['flow_fallbacks'] += 1
                    self._flow = None
        self.last_method = 'optical_flow' if depth is not None else 'neural'
        if depth is None:
            depth = np.asarray(predict(frame), dtype=np.float32)
            if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
                raise ValueError('Depth inference must return a nonempty finite 2D map.')
            if self.normalizer is not None:
                depth = np.asarray(self.normalizer.normalize(depth), dtype=np.float32)
                if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
                    raise ValueError('Normalized depth must be a nonempty finite 2D map.')
            self.stats['neural_frames'] += 1
            self._propagation_age = 0
            guide = self._guide(frame, depth.shape)
        depth = np.ascontiguousarray(np.clip(depth, 0, 1), dtype=np.float32)
        result = self._filter.apply(guide, depth)
        self._previous_guide = guide.copy()
        self._previous_depth = depth.copy()
        self._source_shape = frame.shape
        self.stats['frames'] += 1
        return result
