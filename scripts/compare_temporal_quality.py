"""Compare history filters against known generated depth, never learned-model truth."""
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

from stereo import TemporalDepthFilter
from temporal_quality import MotionAlignedDepthFilter


def sequence(kind, count=24, width=320, height=180):
    rng = np.random.default_rng(1729)
    texture = rng.integers(40, 150, (height, width, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    for index in range(count):
        frame = texture.copy()
        truth = np.full((height, width), .25, np.float32)
        if kind == 'static':
            frame[45:135, 120:200] = [30, 60, 230]
            truth[45:135, 120:200] = .8
        elif kind in {'motion', 'hard-cut'}:
            start = 30 + index * 7
            patch = (slice(55, 125), slice(start, start + 36))
            frame[patch] = [30, 60, 230]
            frame[60:120:5, start:start + 36:4] = [40, 210, 250]
            truth[patch] = .8
            if kind == 'hard-cut' and index >= count // 2:
                frame = 255 - frame
                truth = 1 - truth
        elif kind == 'same-color-ambiguity':
            frame[:] = 100
            start = 30 + index * 7
            truth[:] = .35
            truth[55:125, start:start + 36] = .5
        else:
            raise ValueError('Unknown synthetic clip kind.')
        noisy = np.clip(truth + rng.normal(0, .035, truth.shape), 0, 1).astype(np.float32)
        yield frame, truth, noisy


def gray(depth):
    return cv2.cvtColor(np.rint(depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def tile(image, label):
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 23), (10, 10, 10), -1)
    cv2.putText(result, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def retained_array_bytes(processor):
    values = list(vars(processor).values())
    values.extend(getattr(processor, 'stats', {}).values())
    return sum(value.nbytes for value in values if isinstance(value, np.ndarray))


def compare(output_root: Path, count=24) -> Path:
    if not 12 <= count <= 36:
        raise ValueError('Use 12 to 36 frames per synthetic clip.')
    output = Path(output_root) / f'temporal-{uuid.uuid4().hex[:12]}'
    output.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'experimental': True, 'default_filter_changed': False,
              'ground_truth_scope': 'Generated synthetic depth only; no real learned-model or hidden-surface ground truth.',
              'raw_scale_recovery_claimed': False,
              'settings': {'amount': .6, 'frames': count, 'width': 320, 'height': 180, 'fps': 12, 'noise_sigma': .035, 'seed': 1729,
                           'flow': {'method': 'Farneback', 'pyr_scale': .5, 'levels': 3, 'winsize': 15, 'iterations': 3, 'poly_n': 5, 'poly_sigma': 1.2},
                           'confidence': {'forward_backward_max_pixels': 1., 'appearance_max': .1, 'depth_delta_max': .2, 'bilinear_depth_variance_max': .0025}},
              'notes': ['Only history alignment is tested; scene-aware RAW normalization is pending.',
                        'Per-frame engine normalization has already discarded scale; this filter cannot recover it.',
                        'Temporal delta error compares output changes against known generated truth changes at matching pixels.',
                        'Same-color/depth-only motion is intentionally ambiguous and may retain trails.',
                        'Synthetic error differences do not establish quality on real learned depth or video.',
                        'CPU timing covers filter calls only, with no model inference or full-export speed claim.',
                        'Retained array bytes are measured from live filter state and diagnostics; native scratch allocations and process/GPU peaks are not measured.'],
              'clips': []}
    selected = {0, count // 4, count // 2 - 1, count // 2, count * 3 // 4, count - 1}
    for kind in ('static', 'motion', 'hard-cut', 'same-color-ambiguity'):
        target = output / kind
        target.mkdir()
        baseline, aligned = TemporalDepthFilter(.6), MotionAlignedDepthFilter(.6)
        times = {'baseline': [], 'aligned': []}
        errors = {'noisy': [], 'baseline': [], 'aligned': []}
        delta_errors = {key: [] for key in errors}
        previous, previous_truth = {}, None
        diagnostics, contact_rows = [], []
        writer = cv2.VideoWriter(str(target / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 12, (640, 360))
        video_available = writer.isOpened()
        try:
            for index, (frame, truth, noisy) in enumerate(sequence(kind, count)):
                values = {'noisy': noisy}
                order = [('baseline', baseline), ('aligned', aligned)]
                if index % 2:
                    order.reverse()
                for name, processor in order:
                    started = time.perf_counter()
                    values[name] = processor.apply(frame, noisy)
                    times[name].append(time.perf_counter() - started)
                for name, value in values.items():
                    errors[name].append(float(np.mean(np.abs(value - truth))))
                    if previous_truth is not None:
                        delta_errors[name].append(float(np.mean(np.abs((value - previous[name]) - (truth - previous_truth)))))
                    previous[name] = value
                previous_truth = truth
                diagnostics.append({key: value for key, value in aligned.stats.items() if not isinstance(value, np.ndarray)})
                if index in selected:
                    images = [tile(frame, f'frame {index}: source'), tile(gray(truth), 'generated truth'),
                              tile(gray(noisy), 'noisy input'), tile(gray(values['baseline']), 'appearance baseline'),
                              tile(gray(values['aligned']), 'motion-aligned experiment'),
                              tile(cv2.cvtColor(aligned.stats['accepted_mask'].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR), 'accepted history mask')]
                    contact_rows.append(np.concatenate(images, axis=1))
                if video_available:
                    top = np.concatenate([tile(frame, 'source'), tile(gray(truth), 'generated truth')], axis=1)
                    bottom = np.concatenate([tile(gray(values['baseline']), 'appearance baseline'), tile(gray(values['aligned']), 'motion-aligned experiment')], axis=1)
                    writer.write(np.concatenate([top, bottom], axis=0))
        finally:
            writer.release()
        if not cv2.imwrite(str(target / 'comparison.png'), np.concatenate(contact_rows, axis=0)):
            raise RuntimeError('Could not write temporal comparison contact sheet.')
        report['clips'].append({'id': kind,
                                'depth_mae_synthetic': {key: statistics.mean(value) for key, value in errors.items()},
                                'temporal_delta_error_mae_synthetic': {key: statistics.mean(value) for key, value in delta_errors.items()},
                                'cpu_filter_timings': {key: {'samples_seconds': value, 'median_seconds': statistics.median(value), 'max_seconds': max(value)} for key, value in times.items()},
                                'retained_state_array_bytes': {'baseline': retained_array_bytes(baseline), 'aligned': retained_array_bytes(aligned)},
                                'aligned_frame_diagnostics': diagnostics,
                                'contact_sheet': f'{kind}/comparison.png',
                                'video': f'{kind}/comparison.mp4' if video_available else None,
                                'video_note': 'OpenCV mp4v writer was available.' if video_available else 'Local OpenCV MP4 writer unavailable; no dependency was installed.'})
    destination = output / 'metrics.json'
    destination.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / '.runtime' / 'quality-verification')
    parser.add_argument('--frames', type=int, default=24)
    args = parser.parse_args()
    print(compare(args.output_root, args.frames))
