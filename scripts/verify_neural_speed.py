"""Compare original and optimized complete DirectML exports on synthetic media.

Uses installed accepted weights, downloads nothing, and saves all reports/media.
Timings include decode through finalization, with model initialization warmed
outside the timer. Software decode/encode isolates the neural/video changes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import main
from depth_engine import ModelManager
from stereo import TemporalDepthFilter
from video_encoder import EncoderProfile


def reference_module(filename, revision, destination):
    source = subprocess.run(['git', 'show', f'{revision}:{filename}'], cwd=ROOT,
                            check=True, capture_output=True).stdout
    (destination / filename).write_bytes(source)
    module = types.ModuleType('_speed_reference_' + Path(filename).stem)
    module.__file__ = str(ROOT / filename)
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def decoded_output(path, ffmpeg, expected):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) != expected or any(frame.shape != (360, 1280, 3) for frame in frames):
        raise AssertionError('Export lost frames or changed expected stereo geometry.')
    if abs(fps - 12) > .01:
        raise AssertionError(f'Unexpected output FPS {fps}.')
    audio = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-map', '0:a:0',
                            '-f', 's16le', '-acodec', 'pcm_s16le', '-'],
                           check=True, capture_output=True).stdout
    if not audio:
        raise AssertionError('Audio missing.')
    return frames, {'frames': len(frames), 'shape': list(frames[0].shape), 'fps': fps,
                    'audio_bytes': len(audio), 'audio_sha256': hashlib.sha256(audio).hexdigest(),
                    'pixels_sha256': hashlib.sha256(b''.join(frame.tobytes() for frame in frames)).hexdigest()}


def temporal_measurement():
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    native = rng.random((252, 448), dtype=np.float32)
    small = cv2.resize(frame, (448, 252), interpolation=cv2.INTER_AREA)
    full = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    measurements = {'full': [], 'native': []}
    for repeat in range(3):
        for label in (('full', 'native') if repeat % 2 == 0 else ('native', 'full')):
            filter_ = TemporalDepthFilter(.35)
            image, depth = (frame, full) if label == 'full' else (small, native)
            filter_.apply(image, depth)
            started = time.perf_counter()
            for _ in range(30):
                filter_.apply(image, depth)
            measurements[label].append((time.perf_counter() - started) / 30)
    return {'seconds_per_filter_call': measurements,
            'median_seconds': {key: statistics.median(values) for key, values in measurements.items()},
            'shapes': {'full': list(full.shape), 'native': list(native.shape)},
            'scope': 'Smoothing and cut checks only; excludes resize, inference, refinement and rendering.'}


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-revision', default='70e5dc7')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--frames', type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5 or not 4 <= args.frames <= 120:
        parser.error('Use 1-5 repeats and 4-120 frames.')
    if not main.FFMPEG_PATH:
        raise RuntimeError('FFmpeg unavailable.')
    cv2.setNumThreads(1)
    root = ROOT / '.runtime' / ('neural-speed-' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    reference = root / 'reference'
    reference.mkdir()
    old_engine = reference_module('depth_engine.py', args.baseline_revision, reference)
    old_main = reference_module('main.py', args.baseline_revision, reference)
    source = root / 'source.mp4'
    subprocess.run([main.FFMPEG_PATH, '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    f'testsrc2=size=640x360:rate=12:duration={args.frames / 12}',
                    '-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={args.frames / 12}',
                    '-frames:v', str(args.frames), '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', str(source)],
                   check=True, timeout=30)
    capture = cv2.VideoCapture(str(source))
    try:
        ok, warm_frame = capture.read()
        if not ok:
            raise AssertionError('Source generation failed.')
    finally:
        capture.release()
    encoder = ('-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20', '-threads', '1')
    profile = EncoderProfile('fast', 'libx264', encoder, encoder, False)
    variants = [('original_fp32', old_main, old_engine.ModelManager, 'depth-anything-v2-small', 1),
                ('optimized_fp32', main, ModelManager, 'depth-anything-v2-small', 1),
                ('optimized_flow_fp32', main, ModelManager, 'depth-anything-v2-small', 2)]
    if ModelManager(ROOT / '.runtime/models')._accepted('depth-anything-v2-small-fp16'):
        variants.append(('optimized_fp16', main, ModelManager, 'depth-anything-v2-small-fp16', 1))
    rows = []
    reference_frames = None
    for repeat in range(args.repeats):
        order = variants if repeat % 2 == 0 else list(reversed(variants))
        for label, module, manager_type, model_id, interval in order:
            manager = manager_type(ROOT / '.runtime/models')
            manager.predict(warm_frame, model_id, 256, 'gpu')
            warm_info = manager.execution_info()
            values = dict(id=uuid.uuid4().hex, filename=source.name, input_path=source,
                          media_type='video', layout='SBS', strength=5, convergence=.5,
                          depth_mode=model_id, input_size=256, resolution='source', quality='fast',
                          invert_depth=False, temporal_smoothing=.35, edge_refinement=True,
                          execution_mode='gpu')
            if module is main:
                values['depth_interval'] = interval
            job = module.Job(**values)
            target = root / f'{repeat}-{label}.mp4'
            with patch.object(module, 'DEPTH_ENGINE', manager), \
                    patch.object(module, 'JOBS', {job.id: job}), \
                    patch.object(module, 'output_path_for', return_value=target), \
                    patch.object(module, 'select_encoder_profile', return_value=profile):
                with patch.object(main, 'open_video_capture', side_effect=lambda source, *_args, **_kw: cv2.VideoCapture(str(source))):
                    started = time.perf_counter()
                    module.run_video(job, None, True)
                    elapsed = time.perf_counter() - started
            frames, verification = decoded_output(target, main.FFMPEG_PATH, args.frames)
            if reference_frames is None:
                reference_frames = frames
            errors = [float(np.mean(np.abs(frame.astype(np.float32) - base)))
                      for frame, base in zip(frames, reference_frames)]
            rows.append({'repeat': repeat, 'variant': label, 'seconds': elapsed,
                         'warm_execution': warm_info, 'execution': job.execution,
                         'timings': job.timings.snapshot(), 'verification': verification,
                         'decoded_bgr_mae_vs_first_baseline': statistics.mean(errors)})
            print(json.dumps({'variant': label, 'repeat': repeat, 'seconds': elapsed,
                              'video_depth_stats': job.execution.get('video_depth_stats')}), flush=True)
            manager._session = None
            del manager, frames
            gc.collect()
    audio_hashes = {row['verification']['audio_sha256'] for row in rows}
    if len(audio_hashes) != 1:
        raise AssertionError('Decoded audio differs between variants.')
    report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'baseline_revision': args.baseline_revision, 'frames': args.frames, 'repeats': args.repeats,
              'measurements': rows, 'median_seconds': {
                  label: statistics.median(row['seconds'] for row in rows if row['variant'] == label)
                  for label, *_ in variants}, 'temporal_filter': temporal_measurement(),
              'scope': 'Synthetic complete stereo exports on DirectML, input256, model warmed before each export; OpenCV software decode and libx264 encode. Quality-changing variants are not byte-identical. No whole-film or universal speed claim.'}
    report_path = root / 'report.json'
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(report_path), 'median_seconds': report['median_seconds']}, indent=2))


if __name__ == '__main__':
    main_cli()
