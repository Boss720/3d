"""Optional CPU motion-aligned history; no depth-scale recovery is attempted.

Farneback flow can fail on flat/repeated textures, similar-color motion and subtle
cuts. Forward/backward consistency is not proof of a correct correspondence.
Appearance and large depth-disagreement rejection reduce obvious disocclusion
trails, but small same-color/depth-only changes can still lag. Per-frame engine
normalization has already discarded raw scale; scene-aware RAW normalization is
still pending. This experiment does not replace the default appearance filter.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


class MotionAlignedDepthFilter:
    """One instance per ordered stream. apply returns depth; .stats describes it."""

    def __init__(self, amount: float = .6):
        amount = float(amount)
        if not math.isfinite(amount):
            raise ValueError('Temporal smoothing amount must be finite.')
        self.amount = min(.9, max(0., amount))
        self._previous_frame = self._previous_depth = self._previous_gray = None
        self._pending_reason = 'first_frame'
        self.stats = {}

    def reset(self) -> None:
        self._previous_frame = self._previous_depth = self._previous_gray = None
        self._pending_reason = 'explicit_reset'
        self.stats = {}

    @staticmethod
    def _flow(source, target):
        return cv2.calcOpticalFlowFarneback(source, target, None, .5, 3, 15, 3, 5, 1.2, 0)

    def apply(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
            raise ValueError('Frame must be a nonempty uint8 BGR image.')
        if not isinstance(depth, np.ndarray) or depth.shape != frame.shape[:2]:
            raise ValueError('Depth dimensions must match the frame.')
        current = np.nan_to_num(np.array(depth, np.float32, copy=True), nan=.5, posinf=1., neginf=0.)
        np.clip(current, 0., 1., out=current)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        accepted = np.zeros(depth.shape, bool)
        reason, scene_cut, flow_seconds = None, False, 0.
        mean_residual = mean_weight = 0.
        if self._previous_frame is None:
            reason = self._pending_reason
        elif self._previous_frame.shape != frame.shape:
            reason = 'shape_change'
        elif self.amount == 0:
            reason = 'disabled'
        else:
            difference = np.max(cv2.absdiff(frame, self._previous_frame), axis=2).astype(np.float32) / 255
            scene_cut = float(difference.mean()) > .24 or float((difference > .18).mean()) > .65
            if scene_cut:
                reason = 'scene_cut'
            else:
                flow_started = time.perf_counter()
                backward = self._flow(gray, self._previous_gray)  # Current -> previous.
                forward = self._flow(self._previous_gray, gray)
                flow_seconds = max(0., time.perf_counter() - flow_started)
                finite_backward = np.isfinite(backward).all(axis=2)
                finite_forward = np.isfinite(forward).all(axis=2).astype(np.float32)
                backward = np.nan_to_num(backward, nan=0., posinf=0., neginf=0.)
                forward = np.nan_to_num(forward, nan=0., posinf=0., neginf=0.)
                height, width = depth.shape
                grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
                map_x, map_y = grid_x + backward[:, :, 0], grid_y + backward[:, :, 1]
                in_bounds = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
                def warp(value):
                    return cv2.remap(value, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                aligned_forward = warp(forward)
                residual = np.linalg.norm(backward.astype(np.float64) + aligned_forward, axis=2)
                aligned_frame = warp(self._previous_frame)
                aligned_depth = warp(self._previous_depth)
                # Reject bilinear footprints mixing distinct depth layers.
                # Exact integer correspondences retain zero footprint variance.
                footprint_variance = np.maximum(0., warp(self._previous_depth ** 2) - aligned_depth ** 2)
                appearance = np.max(cv2.absdiff(frame, aligned_frame), axis=2).astype(np.float32) / 255
                depth_difference = np.abs(current - aligned_depth)
                accepted = (in_bounds & finite_backward & (warp(finite_forward) > .999) &
                            (residual <= 1.) & (appearance <= .1) & (depth_difference <= .2) &
                            (footprint_variance <= .0025))
                confidence = np.exp(-np.square(np.minimum(residual, 100.))) * np.exp(-np.square(appearance / .06))
                weight = np.where(accepted, confidence * self.amount, 0.).astype(np.float32)
                current = current * (1. - weight) + aligned_depth * weight
                mean_residual, mean_weight = float(residual.mean()), float(weight.mean())
        current = np.ascontiguousarray(np.clip(current, 0., 1.), dtype=np.float32)
        self._previous_frame = frame.copy()
        self._previous_gray = gray
        self._previous_depth = current.copy()
        self._pending_reason = None
        self.stats = {'schema_version': 1, 'accepted_mask': accepted,
                      'accepted_fraction': float(accepted.mean()), 'rejected_fraction': float((~accepted).mean()),
                      'reset_reason': reason, 'scene_cut': scene_cut, 'history_weight_mean': mean_weight,
                      'forward_backward_residual_mean': mean_residual, 'flow_seconds': flow_seconds,
                      'total_seconds': max(0., time.perf_counter() - started), 'depth_scale_recovered': False}
        return current
