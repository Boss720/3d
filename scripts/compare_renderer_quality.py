"""Matched synthetic renderer comparisons; no novel-view ground truth claims.

Run with .venv/Scripts/python.exe scripts/compare_renderer_quality.py.
Artifacts preserve source BGR colors without per-image normalization. CPU timings
cover renderer calls only, not depth inference, encoding, API or whole exports.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from renderer_quality import render_visibility_pair
from stereo import make_eye_pair


def fixtures():
    for name, width, height in [('foreground', 481, 321), ('thin-object', 641, 359), ('portrait', 321, 481)]:
        x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        frame = np.empty((height, width, 3), np.uint8)
        frame[:, :, 0] = np.rint(80 + 140 * x)
        frame[:, :, 1] = np.rint(40 + 120 * y)
        frame[:, :, 2] = 35
        frame[::16, :, :] = [95, 105, 115]
        frame[:, ::16, :] = [95, 105, 115]
        depth = np.full((height, width), .15, np.float32)
        if name == 'thin-object':
            for offset, thickness in [(-60, 1), (-20, 2), (20, 3), (60, 5)]:
                start = width // 2 + offset
                frame[height // 6:height * 5 // 6, start:start + thickness] = [30, 40, 245]
                depth[height // 6:height * 5 // 6, start:start + thickness] = .95
        else:
            rows, cols = slice(height // 5, height * 4 // 5), slice(width // 3, width * 2 // 3)
            frame[rows, cols] = [25, 45, 240]
            depth[rows, cols] = .9
            # A narrow near object in front of the broad foreground layer.
            frame[height // 8:height * 7 // 8, width // 2:width // 2 + 2] = [35, 235, 75]
            depth[height // 8:height * 7 // 8, width // 2:width // 2 + 2] = 1
        yield name, frame, depth


def _write(path, image):
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f'Could not write comparison artifact: {path.name}')


def _tile(image, text):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 26), (12, 12, 12), -1)
    cv2.putText(result, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def _distribution(samples):
    return {'samples_seconds': samples, 'median_seconds': statistics.median(samples),
            'min_seconds': min(samples), 'max_seconds': max(samples)}


def compare(output_root: Path, repeats: int = 7) -> Path:
    if repeats < 3:
        raise ValueError('Use at least three timing repeats.')
    output = Path(output_root) / f'visibility-{uuid.uuid4().hex[:12]}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'experimental': True, 'default_renderer_changed': False,
              'novel_view_ground_truth_available': False, 'quality_superiority_claimed': False,
              'settings': {'strength': 10., 'convergence': .5},
              'notes': ['Synthetic source colors and depth are controlled; novel-view hidden surfaces are unknown.',
                        'Hole masks measure forward coverage before fill, not correctness.',
                        'Nearest-pixel splats may alias thin objects and step during motion.',
                        'Fill prefers bounded horizontal background neighbors; border replication cannot reconstruct surfaces.',
                        'Unfilled holes retain same-coordinate source samples and remain explicitly marked.',
                        'Timings are CPU renderer calls only; no inference, encoding or full-export speed claim.'],
              'fixtures': []}
    for name, frame, depth in fixtures():
        target = output / name
        target.mkdir()
        renderers = {'inverse': lambda: make_eye_pair(frame, depth, 10., .5),
                     'visibility': lambda: render_visibility_pair(frame, depth, 10., .5)}
        for renderer in renderers.values():
            renderer()  # One unmeasured warmup each.
        samples = {key: [] for key in renderers}
        for index in range(repeats):
            order = ['inverse', 'visibility'] if index % 2 == 0 else ['visibility', 'inverse']
            for key in order:
                started = time.perf_counter()
                renderers[key]()
                samples[key].append(time.perf_counter() - started)
        inverse_left, inverse_right = renderers['inverse']()
        left, right, stats = renderers['visibility']()
        depth_image = cv2.cvtColor(np.rint(depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        images = {'source': frame, 'source-depth': depth_image, 'inverse-left': inverse_left,
                  'inverse-right': inverse_right, 'visibility-left': left, 'visibility-right': right}
        for eye in ('left', 'right'):
            images[f'{eye}-holes-before-fill'] = stats[eye]['hole_mask'].astype(np.uint8) * 255
            images[f'{eye}-unfilled'] = stats[eye]['unfilled_mask'].astype(np.uint8) * 255
        for key, image in images.items():
            _write(target / f'{key}.png', image)
        rows = [np.concatenate([_tile(images[a], a), _tile(images[b], b)], axis=1) for a, b in
                [('source', 'source-depth'), ('inverse-left', 'inverse-right'), ('visibility-left', 'visibility-right')]]
        masks = [cv2.cvtColor(images[f'{eye}-holes-before-fill'], cv2.COLOR_GRAY2BGR) for eye in ('left', 'right')]
        rows.append(np.concatenate([_tile(masks[0], 'left holes before fill'), _tile(masks[1], 'right holes before fill')], axis=1))
        _write(target / 'comparison.png', np.concatenate(rows, axis=0))
        eye_metrics = {eye: {key: value for key, value in stats[eye].items() if not isinstance(value, np.ndarray)}
                       for eye in ('left', 'right')}
        report['fixtures'].append({'id': name, 'source_width': frame.shape[1], 'source_height': frame.shape[0],
                                   'cpu_renderer_timings': {key: _distribution(value) for key, value in samples.items()},
                                   'visibility': {'fill_radius': stats['fill_radius'],
                                                  'max_displacement_pixels': stats['max_displacement_pixels'],
                                                  'raster_displacement_limit_pixels': stats['raster_displacement_limit_pixels'],
                                                  **eye_metrics},
                                   'inverse_hole_metrics': None,
                                   'inverse_hole_note': 'Inverse sampling has no explicit coverage mask; no zero-hole correctness claim.',
                                   'artifacts': [f'{name}/{key}.png' for key in images] + [f'{name}/comparison.png']})
    report_path = output / 'metrics.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / '.runtime' / 'quality-verification')
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    print(compare(args.output_root, args.repeats))
