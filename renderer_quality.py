"""Optional CPU forward splat with explicit holes, not reconstructed surfaces.

Nearest-pixel splats can alias thin objects and move in pixel-sized steps. Higher
inverse depth wins collisions; equal-depth ties select the rightmost source.
Displacements round before coordinates are added, avoiding parity-dependent
half-pixel ties, and are clamped to the integer part of the disparity budget.
Horizontal hole treatment selects the lower-depth of the nearest visible samples
on either side, within a radius capped at 32 pixels. At borders a sole nearby
sample is replicated. Remaining holes retain the same-coordinate source color
and remain marked unfilled. There is no temporal stabilization or ground truth.
"""
from __future__ import annotations

import math

import numpy as np


def _eye(frame: np.ndarray, depth: np.ndarray, disparity: np.ndarray,
         direction: int, radius: int, raster_limit: int) -> tuple[np.ndarray, dict]:
    height, width = depth.shape
    image = frame.copy()
    holes = np.zeros((height, width), dtype=bool)
    unfilled = np.zeros_like(holes)
    x = np.arange(width, dtype=np.int32)
    collision_count = outside_count = max_distance = 0
    # Row-local work arrays bound scratch memory to O(width), not O(pixels).
    for row in range(height):
        shift = np.rint(disparity[row]).astype(np.int32)
        np.clip(shift, -raster_limit, raster_limit, out=shift)
        target = x + direction * shift
        valid = (target >= 0) & (target < width)
        source = x[valid]
        targets = target[valid]
        zbuffer = np.full(width, -np.inf, np.float32)
        np.maximum.at(zbuffer, targets, depth[row, source])
        nearest = depth[row, source] == zbuffer[targets]
        winner = np.full(width, -1, np.int32)
        np.maximum.at(winner, targets[nearest], source[nearest])
        visible = winner >= 0
        image[row, visible] = frame[row, winner[visible]]
        holes[row] = ~visible
        collision_count += int(valid.sum() - visible.sum())
        outside_count += int(width - valid.sum())
        if visible.all():
            continue
        left = np.maximum.accumulate(np.where(visible, x, -width))
        right = np.minimum.accumulate(np.where(visible, x, width * 2)[::-1])[::-1]
        left_distance, right_distance = x - left, right - x
        left_ok = (left >= 0) & (left_distance <= radius)
        right_ok = (right < width) & (right_distance <= radius)
        left = np.clip(left, 0, width - 1)
        right = np.clip(right, 0, width - 1)
        left_depth, right_depth = zbuffer[left], zbuffer[right]
        choose_left = left_ok & (~right_ok | (left_depth < right_depth) |
                                 ((left_depth == right_depth) & (left_distance <= right_distance)))
        fill = holes[row] & (left_ok | right_ok)
        selected = np.where(choose_left, left, right)
        image[row, fill] = image[row, selected[fill]]
        unfilled[row] = holes[row] & ~fill
        if fill.any():
            distances = np.where(choose_left, left_distance, right_distance)
            max_distance = max(max_distance, int(distances[fill].max()))
    return image, {'hole_mask': holes, 'unfilled_mask': unfilled,
                   'hole_fraction': float(holes.mean()), 'unfilled_fraction': float(unfilled.mean()),
                   'max_fill_distance': max_distance, 'collision_count': collision_count,
                   'outside_source_count': outside_count}


def render_visibility_pair(frame: np.ndarray, depth: np.ndarray, strength: float,
                           convergence: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return left/right BGR eyes and both pre-fill disocclusion masks.

    Near pixels move right in the left eye and left in the right eye. Per-eye
    displacement is bounded by width * .015 * strength / 10, as in make_eye_pair.
    Controls are finite and clamped to their supported range; depth is copied,
    sanitized (NaN=.5, +inf=1, -inf=0), and clamped to [0, 1]. The masks measure
    forward-splat coverage, not correctness or recovered hidden surfaces.
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or min(frame.shape[:2]) < 1:
        raise ValueError('Frame must be a nonempty uint8 BGR image with three channels.')
    if not isinstance(depth, np.ndarray) or depth.shape != frame.shape[:2]:
        raise ValueError('Depth dimensions must match the source frame.')
    strength, convergence = float(strength), float(convergence)
    if not math.isfinite(strength) or not math.isfinite(convergence):
        raise ValueError('Strength and convergence must be finite.')
    strength, convergence = min(10., max(0., strength)), min(1., max(0., convergence))
    normalized = np.nan_to_num(np.array(depth, dtype=np.float32, copy=True), nan=.5, posinf=1., neginf=0.)
    np.clip(normalized, 0., 1., out=normalized)
    max_displacement = frame.shape[1] * .015 * strength / 10
    raster_limit = math.floor(max_displacement)
    radius = min(32, max(1, math.ceil(max_displacement * 2) + 1))
    if strength == 0:
        def empty_stats():
            return {'hole_mask': np.zeros(depth.shape, bool), 'unfilled_mask': np.zeros(depth.shape, bool),
                    'hole_fraction': 0., 'unfilled_fraction': 0., 'max_fill_distance': 0,
                    'collision_count': 0, 'outside_source_count': 0}
        left, right = frame.copy(), frame.copy()
        left_stats, right_stats = empty_stats(), empty_stats()
    else:
        disparity = (normalized - convergence) * np.float32(max_displacement)
        left, left_stats = _eye(frame, normalized, disparity, 1, radius, raster_limit)
        right, right_stats = _eye(frame, normalized, disparity, -1, radius, raster_limit)
    return left, right, {'schema_version': 1, 'method': 'nearest-forward-visibility-experimental',
                         'fill_radius': radius, 'max_displacement_pixels': max_displacement,
                         'raster_displacement_limit_pixels': raster_limit,
                         'max_fill_distance': max(left_stats['max_fill_distance'], right_stats['max_fill_distance']),
                         'left': left_stats, 'right': right_stats,
                         'surface_recovery_claimed': False}


def visibility_summary(stats: dict) -> dict:
    """Return bounded JSON diagnostics without retaining per-pixel masks."""
    eye_fields = ('hole_fraction', 'unfilled_fraction', 'max_fill_distance',
                  'collision_count', 'outside_source_count')
    return {
        'schema_version': stats['schema_version'],
        'method': 'visibility',
        'algorithm': stats['method'],
        'stereo_views_rendered': True,
        'fill_radius': stats['fill_radius'],
        'max_displacement_pixels': stats['max_displacement_pixels'],
        'raster_displacement_limit_pixels': stats['raster_displacement_limit_pixels'],
        'max_fill_distance': stats['max_fill_distance'],
        'left': {key: stats['left'][key] for key in eye_fields},
        'right': {key: stats['right'][key] for key in eye_fields},
        'surface_recovery_claimed': False,
    }
