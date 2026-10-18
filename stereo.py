"""CPU stereo synthesis and job-local depth stabilization.

Depth is relative inverse depth: 0 is far and 1 is near. Eye synthesis uses
inverse image warps, so it cannot reveal hidden surfaces or resolve visibility
at overlapping depth edges. Replicated borders and stretched edge pixels are
an approximation, not reconstructed geometry or an occlusion inpainting model.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


LAYOUTS = frozenset({'SBS', 'HALFSBS', 'TB', 'HALFTB', 'CROSS', 'ANAGLYPH', 'DEPTH', 'RGBD'})


def _frame_size(frame: np.ndarray) -> tuple[int, int]:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Frame must be a uint8 BGR image with three channels.')
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError('Frame dimensions must be positive.')
    return height, width


def _bounded_depth(depth: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if depth.shape != shape:
        raise ValueError('Depth dimensions must match the source frame.')
    result = np.array(depth, dtype=np.float32, copy=True, order='C')
    np.nan_to_num(result, copy=False, nan=.5, posinf=1.0, neginf=0.0)
    return np.clip(result, 0.0, 1.0, out=result)


def _bounded_number(value: float, low: float, high: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    return min(high, max(low, value))


def make_eye_pair(
    frame: np.ndarray,
    depth: np.ndarray,
    strength: float,
    convergence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return BGR eyes with at most 3% image-width total disparity.

    Strength is clamped to 0..10 and scales disparity linearly. Convergence is
    the depth plane with zero disparity. Near pixels shift right in the left
    view and left in the right view; strength never swaps these directions.
    OpenCV's interpolation quantizes subpixel shifts, so very small strength
    changes can produce identical pixels despite increasing the warp distance.
    """
    height, width = _frame_size(frame)
    normalized = _bounded_depth(depth, (height, width))
    scale = _bounded_number(strength, 0.0, 10.0, 'Strength') / 10.0
    plane = _bounded_number(convergence, 0.0, 1.0, 'Convergence')
    if scale == 0:
        return frame.copy(), frame.copy()
    disparity = (normalized - plane) * np.float32(width * .015 * scale)
    x_grid = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    y_grid = np.ascontiguousarray(np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)))
    left = cv2.remap(frame, x_grid - disparity, y_grid, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    right = cv2.remap(frame, x_grid + disparity, y_grid, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return left, right


def layout_size(width: int, height: int, layout: str) -> tuple[int, int]:
    """Return exact packed dimensions, including rounded-down half views.

    Video encoders may need even dimensions; the encoding caller owns that
    padding/cropping policy. Stills retain their actual packed dimensions.
    """
    if width < 1 or height < 1:
        raise ValueError('Frame dimensions must be positive.')
    if layout not in LAYOUTS:
        raise ValueError(f'Unsupported output layout: {layout}')
    if layout in {'SBS', 'CROSS', 'RGBD'}:
        return width * 2, height
    if layout == 'TB':
        return width, height * 2
    if layout == 'HALFSBS':
        return max(1, width // 2) * 2, height
    if layout == 'HALFTB':
        return width, max(1, height // 2) * 2
    return width, height


def pack_output(
    frame: np.ndarray,
    depth: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    layout: str,
) -> np.ndarray:
    """Pack stereo eyes, or export near-white depth / source-plus-depth."""
    height, width = _frame_size(frame)
    layout_size(width, height, layout)
    if layout in {'DEPTH', 'RGBD'}:
        values = np.rint(_bounded_depth(depth, (height, width)) * 255.0).astype(np.uint8)
        gray = cv2.cvtColor(values, cv2.COLOR_GRAY2BGR)
        return gray if layout == 'DEPTH' else np.concatenate((frame, gray), axis=1)
    if _frame_size(left) != (height, width) or _frame_size(right) != (height, width):
        raise ValueError('Stereo eye dimensions must match the source frame.')
    if layout == 'SBS':
        return np.concatenate((left, right), axis=1)
    if layout == 'CROSS':
        return np.concatenate((right, left), axis=1)
    if layout == 'TB':
        return np.concatenate((left, right), axis=0)
    if layout == 'HALFSBS':
        size = (max(1, width // 2), height)
        return np.concatenate((cv2.resize(left, size, interpolation=cv2.INTER_AREA), cv2.resize(right, size, interpolation=cv2.INTER_AREA)), axis=1)
    if layout == 'HALFTB':
        size = (width, max(1, height // 2))
        return np.concatenate((cv2.resize(left, size, interpolation=cv2.INTER_AREA), cv2.resize(right, size, interpolation=cv2.INTER_AREA)), axis=0)
    result = right.copy()  # Red from left eye, cyan from right (BGR storage).
    result[:, :, 2] = left[:, :, 2]
    return result


def refine_depth(frame: np.ndarray, depth: np.ndarray, edge_refinement: bool) -> np.ndarray:
    """Optionally apply a grayscale-guided filter using ordinary OpenCV.

    Local linear coefficients follow luminance edges in the source image and
    smooth depth noise in flat regions. This can transfer image texture into
    depth, and edges of equal luminance are not protected by the guide.
    """
    shape = _frame_size(frame)
    values = _bounded_depth(depth, shape)
    if not edge_refinement:
        return values
    guide = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    size = (9, 9)

    def mean(array: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(array, -1, size, normalize=True, borderType=cv2.BORDER_REFLECT)

    mean_i = mean(guide)
    mean_p = mean(values)
    variance = np.maximum(mean(guide * guide) - mean_i * mean_i, 0)
    covariance = mean(guide * values) - mean_i * mean_p
    a = covariance / (variance + .0025)
    b = mean_p - a * mean_i
    refined = mean(a) * guide + mean(b)
    return _bounded_depth(refined, shape)


class TemporalDepthFilter:
    """Per-job exponential smoothing with cut reset and local motion gating.

    ``amount`` is the previous-depth weight, clamped to 0..0.9. Appearance
    changes suppress history in a small neighborhood to limit moving-object
    trails. This is not optical-flow alignment: similar-color motion, subtle
    cuts, and depth changes without image changes can still leave lag.
    """

    def __init__(self, amount: float):
        self.amount = _bounded_number(amount, 0.0, .9, 'Temporal smoothing')
        self.reset()

    def reset(self) -> None:
        self._previous_frame: np.ndarray | None = None
        self._previous_depth: np.ndarray | None = None

    def apply(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        shape = _frame_size(frame)
        current = _bounded_depth(depth, shape)
        if self.amount > 0 and self._previous_frame is not None and self._previous_frame.shape == frame.shape:
            difference = np.max(cv2.absdiff(frame, self._previous_frame), axis=2).astype(np.float32) / 255.0
            scene_cut = float(difference.mean()) > .24 or float(np.mean(difference > .18)) > .65
            if not scene_cut:
                motion = cv2.dilate(difference, np.ones((5, 5), dtype=np.uint8))
                history_weight = self.amount * np.exp(-np.square(motion / .06))
                current = current * (1.0 - history_weight) + self._previous_depth * history_weight
        self._previous_frame = frame.copy()
        self._previous_depth = np.ascontiguousarray(current, dtype=np.float32).copy()
        return np.ascontiguousarray(current, dtype=np.float32)
