"""Measure the disabled temporal-history copy fast path and verify export parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import main
from stereo import TemporalDepthFilter, _bounded_depth
from video_encoder import EncoderProfile


class CopyingZeroSmoothingFilter:
    """Reproduce the former zero-smoothing path for a matched baseline."""

    def __init__(self, amount: float):
        if amount != 0:
            raise ValueError("This baseline only models disabled smoothing.")
        self._previous_frame: np.ndarray | None = None
        self._previous_depth: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_frame = None
        self._previous_depth = None

    def apply(self, frame: np.ndarray, depth: np.ndarray) -> np.ndarray:
        current = _bounded_depth(depth, frame.shape[:2])
        self._previous_frame = frame.copy()
        self._previous_depth = np.ascontiguousarray(current, dtype=np.float32).copy()
        return np.ascontiguousarray(current, dtype=np.float32)


def filter_benchmark(iterations: int, repeats: int) -> dict[str, object]:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    depth = np.full((1080, 1920), 0.5, dtype=np.float32)
    baseline = CopyingZeroSmoothingFilter(0).apply(frame, depth)
    optimized = TemporalDepthFilter(0).apply(frame, depth)
    if not np.array_equal(baseline, optimized):
        raise AssertionError("Zero-smoothing filter outputs differ.")

    measurements = {"copying_baseline": [], "optimized": []}
    warmup = 12
    for filter_type in (CopyingZeroSmoothingFilter, TemporalDepthFilter):
        instance = filter_type(0)
        for _ in range(warmup):
            instance.apply(frame, depth)

    order = (("copying_baseline", CopyingZeroSmoothingFilter), ("optimized", TemporalDepthFilter))
    for repeat in range(repeats):
        current_order = order if repeat % 2 == 0 else tuple(reversed(order))
        for name, filter_type in current_order:
            instance = filter_type(0)
            start = time.perf_counter()
            for _ in range(iterations):
                instance.apply(frame, depth)
            elapsed = time.perf_counter() - start
            measurements[name].append(elapsed)

    old_median = statistics.median(measurements["copying_baseline"])
    new_median = statistics.median(measurements["optimized"])
    return {
        "shape": [1080, 1920],
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "seconds": measurements,
        "median_seconds": {"copying_baseline": old_median, "optimized": new_median},
        "reduction_percent": (old_median - new_median) / old_median * 100 if old_median else 0,
        "outputs_equal": True,
    }


def decode_export(path: Path, expected_frames: int) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    digest = hashlib.sha256()
    count = 0
    shape: list[int] | None = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            count += 1
            shape = list(frame.shape)
            digest.update(frame.tobytes())
    finally:
        capture.release()
    audio = subprocess.run(
        [main.FFMPEG_PATH, "-v", "error", "-i", str(path), "-map", "0:a:0",
         "-f", "s16le", "-acodec", "pcm_s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    if count != expected_frames or shape != [1080, 3840, 3] or not audio:
        raise AssertionError(f"Unexpected decoded output: frames={count}, shape={shape}, audio={len(audio)} bytes")
    return {
        "frames": count,
        "shape": shape,
        "decoded_pixel_sha256": digest.hexdigest(),
        "audio_bytes": len(audio),
        "decoded_audio_sha256": hashlib.sha256(audio).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def export_benchmark(repeats: int) -> dict[str, object]:
    if not main.FFMPEG_PATH or not Path(main.FFMPEG_PATH).is_file():
        raise RuntimeError("FFmpeg is unavailable; set VIDEO3D_FFMPEG or install FFmpeg.")
    root = ROOT / ".runtime" / f"temporal-history-benchmark-{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    source = root / "source.mp4"
    subprocess.run(
        [main.FFMPEG_PATH, "-v", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=1280x720:rate=12:duration=3", "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "18", "-c:a", "aac", "-shortest", str(source)],
        check=True,
    )
    cv2.setNumThreads(1)
    encoder_args = ("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1")
    profile = EncoderProfile("fast", "libx264", encoder_args, encoder_args, False)
    filter_types = {
        "copying_baseline": CopyingZeroSmoothingFilter,
        "optimized": TemporalDepthFilter,
    }
    measurements: list[dict[str, object]] = []
    for repeat in range(repeats):
        labels = list(filter_types)
        if repeat % 2:
            labels.reverse()
        for label in labels:
            job = main.Job(
                id=uuid.uuid4().hex,
                filename=source.name,
                input_path=source,
                media_type="video",
                layout="XREAL_SBS",
                strength=5,
                convergence=0.5,
                depth_mode="balanced",
                input_size=384,
                resolution="1080p",
                quality="fast",
                invert_depth=False,
                temporal_smoothing=0,
                edge_refinement=True,
                execution_mode="cpu",
            )
            target = root / f"{repeat}-{label}.mp4"
            main.JOBS[job.id] = job
            try:
                with patch.object(main, "TemporalDepthFilter", filter_types[label]), \
                        patch.object(main, "select_encoder_profile", return_value=profile), \
                        patch.object(main, "output_path_for", return_value=target):
                    start = time.perf_counter()
                    result = main.run_video(job, None, True)
                    elapsed = time.perf_counter() - start
                measurements.append({
                    "repeat": repeat,
                    "implementation": label,
                    "seconds": elapsed,
                    "render_seconds": job.timings.snapshot()["stages"]["render"]["seconds"],
                    "verification": decode_export(result, 36),
                })
            finally:
                main.JOBS.pop(job.id, None)

    reference = measurements[0]["verification"]
    if any(item["verification"] != reference for item in measurements):
        raise AssertionError("Encoded files, decoded frames, or decoded audio differ.")
    old_median = statistics.median(item["seconds"] for item in measurements if item["implementation"] == "copying_baseline")
    new_median = statistics.median(item["seconds"] for item in measurements if item["implementation"] == "optimized")
    report = {
        "root": str(root),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "clip": "36 generated frames, 1280x720, 12 fps, sine audio",
        "output": "XREAL_SBS, 3840x1080",
        "depth": "balanced approximation, CPU",
        "temporal_smoothing": 0,
        "encoder": list(encoder_args),
        "opencv_threads": cv2.getNumThreads(),
        "repeats": repeats,
        "measurements": measurements,
        "median_seconds": {"copying_baseline": old_median, "optimized": new_median},
        "reduction_percent": (old_median - new_median) / old_median * 100 if old_median else 0,
        "all_encoded_files_and_decoded_frames_and_audio_equal": True,
        "scope": "run_video wall time includes capture setup, ffprobe, decode, approximation depth, refinement, temporal filtering, rendering, encoding, audio, and process completion; excludes model loading, HTTP upload, history, and cleanup",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="interleaved complete-export pairs (default: 3)")
    parser.add_argument("--filter-iterations", type=int, default=120, help="1080p filter calls per repeat (default: 120)")
    args = parser.parse_args()
    if not 2 <= args.repeats <= 10 or not 1 <= args.filter_iterations <= 10_000:
        parser.error("--repeats must be 2-10 and --filter-iterations must be 1-10000")
    report = {
        "filter_microbenchmark": filter_benchmark(args.filter_iterations, args.repeats),
        "complete_export": export_benchmark(args.repeats),
        "note": "Synthetic CPU timings are local evidence, not a universal speed claim or neural/GPU measurement.",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
