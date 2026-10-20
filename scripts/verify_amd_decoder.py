"""Matched full-lifecycle OpenCV versus AMD D3D11VA decode benchmark (no inference)."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from gpu_hardware import probe_hardware
from video_decoder import open_video_capture


def software_capture(source):
    # Make the baseline explicitly software, rather than rely on OpenCV defaults.
    return cv2.VideoCapture(str(source), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE])


def decode_run(source, ffmpeg, backend, expected_frames):
    digest = hashlib.sha256()
    count = 0
    hash_seconds = 0.
    capture = None
    started = time.perf_counter()
    try:
        capture = (open_video_capture(source, ffmpeg, software_capture_factory=software_capture)
                   if backend == 'amd_d3d11va' else software_capture(source))
        startup_seconds = time.perf_counter() - started
        if not capture.isOpened():
            raise RuntimeError(f'{backend} could not open fixture.')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            if frame.shape != (720, 1280, 3) or frame.dtype != np.uint8:
                raise RuntimeError(f'Unexpected frame: {frame.shape} {frame.dtype}')
            hash_started = time.perf_counter()
            digest.update(frame.tobytes())
            hash_seconds += time.perf_counter() - hash_started
    finally:
        cleanup_started = time.perf_counter()
        if capture is not None:
            capture.release()
        cleanup_seconds = time.perf_counter() - cleanup_started
    wall_seconds = time.perf_counter() - started
    info = getattr(capture, 'decoder_info', {'backend': 'OpenCV FFmpeg', 'hardware': False,
                                          'fallback_reason': None})
    result = {'requested_backend': backend, 'decoder': info, 'frames': count,
              'expected_frames': expected_frames, 'wall_seconds_including_hash': wall_seconds,
              'hash_seconds': hash_seconds,
              'decode_lifecycle_seconds': wall_seconds - hash_seconds,
              'startup_seconds': startup_seconds, 'cleanup_seconds': cleanup_seconds,
              'bgr_sha256': digest.hexdigest(),
              'threads_joined': all(not thread.is_alive() for thread in getattr(capture, '_threads', [])),
              'process_exit': capture.process.poll() if hasattr(capture, 'process') else None}
    if count != expected_frames or not result['threads_joined']:
        raise RuntimeError(f'Incomplete decode: {result}')
    return result


def compare_pixels(source, ffmpeg, expected_frames):
    """Independent untimed comparison; error metrics do not imply visual approval."""
    software = software_capture(source)
    hardware = open_video_capture(source, ffmpeg, software_capture_factory=software_capture)
    count = elements = maximum = nonzero = 0
    absolute_sum = squared_sum = 0.
    try:
        while True:
            software_ok, a = software.read()
            hardware_ok, b = hardware.read()
            if software_ok != hardware_ok:
                raise RuntimeError('Hardware/software comparison frame counts differ.')
            if not software_ok:
                break
            if a.shape != b.shape:
                raise RuntimeError('Hardware/software comparison dimensions differ.')
            error = a.astype(np.float32) - b.astype(np.float32)
            absolute = np.abs(error)
            absolute_sum += float(absolute.sum(dtype=np.float64))
            squared_sum += float(np.square(error).sum(dtype=np.float64))
            maximum = max(maximum, int(absolute.max()))
            nonzero += int(np.count_nonzero(error))
            elements += error.size
            count += 1
    finally:
        software.release()
        hardware.release()
    if count != expected_frames:
        raise RuntimeError('Pixel comparison ended before expected frame count.')
    mse = squared_sum / elements
    return {'frames': count, 'exact_pixels_equal': nonzero == 0,
            'different_channel_values': nonzero, 'channel_values': elements,
            'mean_absolute_error_8bit': absolute_sum / elements,
            'maximum_absolute_error_8bit': maximum, 'rmse_8bit': math.sqrt(mse),
            'psnr_db': 10 * math.log10(255 ** 2 / mse) if mse else None,
            'note': 'Pixel error metrics only; real-footage visual quality remains unverified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.frames < 120 or args.repeats < 3:
        parser.error('Use at least 120 frames and 3 repeats for a matched benchmark.')
    root = ROOT / '.runtime' / ('amd-decoder-' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    source = root / 'source.mp4'
    subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'testsrc2=size=1280x720:rate=24', '-frames:v', str(args.frames),
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-y', str(source)],
                    check=True, timeout=60, capture_output=True)
    report = {'source': str(source), 'hardware': probe_hardware(), 'runs': [],
              'measurement': 'Same 1280x720 H.264 file; alternating order; wall clock begins before capture construction and ends after cleanup; hashes measured equally in both lanes. No neural inference.'}
    for repeat in range(args.repeats):
        order = ['opencv_software', 'amd_d3d11va'] if repeat % 2 == 0 else ['amd_d3d11va', 'opencv_software']
        for backend in order:
            report['runs'].append({'repeat': repeat, **decode_run(source, args.ffmpeg, backend, args.frames)})
    medians = {backend: statistics.median(run['decode_lifecycle_seconds'] for run in report['runs']
                                        if run['requested_backend'] == backend)
               for backend in ('opencv_software', 'amd_d3d11va')}
    wall_medians = {backend: statistics.median(run['wall_seconds_including_hash'] for run in report['runs']
                                             if run['requested_backend'] == backend)
                    for backend in medians}
    report.update(median_decode_lifecycle_seconds=medians, median_wall_seconds_including_hash=wall_medians,
                  amd_lifecycle_reduction_percent=100 * (1 - medians['amd_d3d11va'] / medians['opencv_software']),
                  amd_wall_reduction_percent=100 * (1 - wall_medians['amd_d3d11va'] / wall_medians['opencv_software']),
                  actual_hardware_all_runs=all(run['decoder']['hardware'] for run in report['runs']
                                              if run['requested_backend'] == 'amd_d3d11va'),
                  quality_comparison=compare_pixels(source, args.ffmpeg, args.frames))
    report['recommendation'] = ('AMD decode was faster on this synthetic fixture; broader workload evidence is required.'
                                if report['actual_hardware_all_runs'] and report['amd_lifecycle_reduction_percent'] > 0
                                else 'Retain software decoding by default; AMD decode did not improve this synthetic fixture.')
    path = root / 'report.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(path), **report}, indent=2))


if __name__ == '__main__':
    main()
