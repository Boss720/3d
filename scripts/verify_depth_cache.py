"""Verify complete native-depth-cache video exports using local generated media.

The check requires an already accepted Small model and local FFmpeg. It does
not download weights, start the service, or promise a speed improvement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import main
from depth_engine import ModelManager
from video_encoder import EncoderProfile


MODEL_ID = "depth-anything-v2-small"
INPUT_SIZE = 256
SOURCE_FPS = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ffmpeg_version(ffmpeg: str) -> str:
    result = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True, timeout=15
    )
    return (result.stdout or "").splitlines()[0]


def generate_source(ffmpeg: str, destination: Path, frames: int) -> None:
    duration = frames / SOURCE_FPS
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=320x180:rate={SOURCE_FPS}:duration={duration:.8f}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration:.8f}",
         "-frames:v", str(frames), "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-shortest", str(destination)],
        check=True, capture_output=True, timeout=45,
    )


def inspect_export(path: Path, ffmpeg: str, expected_frames: int, layout: str) -> dict:
    expected_shape = (180, 640, 3) if layout == "SBS" else (360, 320, 3)
    capture = cv2.VideoCapture(str(path))
    frames = 0
    pixel_digest = hashlib.sha256()
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != expected_shape:
                raise AssertionError(f"{path.name}: frame {frames} has shape {frame.shape}, expected {expected_shape}.")
            pixel_digest.update(memoryview(np.ascontiguousarray(frame)).cast("B"))
            frames += 1
    finally:
        capture.release()
    if frames != expected_frames:
        raise AssertionError(f"{path.name}: decoded {frames} video frames, expected {expected_frames}.")
    if not np.isfinite(fps) or abs(fps - SOURCE_FPS) > 0.01:
        raise AssertionError(f"{path.name}: output FPS {fps} does not match {SOURCE_FPS}.")

    video_stream = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
         "-an", "-c:v", "copy", "-bsf:v", "h264_mp4toannexb", "-f", "h264", "pipe:1"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    audio_pcm = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    if not video_stream:
        raise AssertionError(f"{path.name}: FFmpeg extracted no encoded video stream.")
    if not audio_pcm:
        raise AssertionError(f"{path.name}: audio stream is absent or cannot be fully decoded.")
    return {
        "path": str(path),
        "container_sha256": sha256_file(path),
        "encoded_video_sha256": hashlib.sha256(video_stream).hexdigest(),
        "pixels_sha256": pixel_digest.hexdigest(),
        "audio_pcm_sha256": hashlib.sha256(audio_pcm).hexdigest(),
        "audio_pcm_bytes": len(audio_pcm),
        "frames": frames,
        "shape": list(expected_shape),
        "fps": fps,
    }


def make_job(source: Path, execution_mode: str, *, cache: bool, layout: str = "SBS",
             strength: float = 5.0, convergence: float = 0.5) -> main.Job:
    return main.Job(
        id=uuid.uuid4().hex,
        filename=source.name,
        input_path=source,
        media_type="video",
        layout=layout,
        strength=strength,
        convergence=convergence,
        depth_mode=MODEL_ID,
        input_size=INPUT_SIZE,
        resolution="source",
        quality="fast",
        invert_depth=False,
        temporal_smoothing=0.35,
        edge_refinement=True,
        execution_mode=execution_mode,
        depth_interval=1,
        video_decode="software",
        use_depth_cache=cache,
    )


def run_export(job: main.Job, manager: ModelManager, store, output_root: Path,
               encoder: EncoderProfile) -> tuple[dict, int, float]:
    target = output_root / f"{job.id}.mp4"
    started = time.perf_counter()
    with patch.object(manager, "predict_native_with_info", wraps=manager.predict_native_with_info) as counted, \
            patch.object(main, "DEPTH_ENGINE", manager), \
            patch.object(main, "VIDEO_DEPTH_CACHE", store), \
            patch.object(main, "OUTPUTS", output_root), \
            patch.object(main, "JOBS", {job.id: job}), \
            patch.object(main, "output_path_for", return_value=target), \
            patch.object(main, "select_encoder_profile", return_value=encoder):
        result = main.run_video(job, None)
    elapsed = time.perf_counter() - started
    call_delta = counted.call_count
    if result != target or not target.is_file() or target.stat().st_size <= 0:
        raise AssertionError("run_video did not produce its isolated output path.")
    measurement = {
        "job_id": job.id,
        "layout": job.layout,
        "elapsed_seconds": elapsed,
        "inference_calls": call_delta,
        "cache": dict(job.depth_cache),
        "execution": dict(job.execution),
        "first_execution": dict(job.first_execution),
        "decoder": dict(job.decoder),
        "timings": job.timings.snapshot(),
        "verification": inspect_export(target, main.FFMPEG_PATH, EXPECTED_FRAMES, job.layout),
    }
    return measurement, call_delta, elapsed


def cache_disk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main_cli() -> None:
    global EXPECTED_FRAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-mode", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 4 <= args.frames <= 120:
        parser.error("--frames must be between 4 and 120")
    if not 1 <= args.repeats <= 3:
        parser.error("--repeats must be between 1 and 3")
    EXPECTED_FRAMES = args.frames
    if not main.FFMPEG_PATH or not Path(main.FFMPEG_PATH).is_file():
        raise RuntimeError("A local FFmpeg executable is required.")

    manager = ModelManager(main.MODELS)
    if not manager._accepted(MODEL_ID):
        raise RuntimeError(
            f"Accepted model {MODEL_ID!r} is unavailable under {main.MODELS}; "
            "install and accept it before running this offline verification."
        )
    runtime = manager.runtime_info()
    if args.execution_mode == "gpu" and not runtime.get("gpu_available"):
        raise RuntimeError(f"GPU mode was requested but DirectML is unavailable: {runtime.get('gpu_error')}")

    root = ROOT / ".runtime" / f"depth-cache-verification-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    source = root / "generated-source.mp4"
    generate_source(main.FFMPEG_PATH, source, args.frames)
    capture = cv2.VideoCapture(str(source))
    try:
        ok, warm_frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError("FFmpeg generated a source that OpenCV could not decode.")

    # Warm the real accepted model once before any measured export.
    warm_depth, warm_info = manager.predict_native_with_info(warm_frame, MODEL_ID, INPUT_SIZE,
                                                              args.execution_mode)
    if warm_depth.dtype != np.float32 or warm_depth.ndim != 2 or not np.isfinite(warm_depth).all():
        raise AssertionError("Warm native inference did not produce finite float32 depth.")
    if warm_info.get("active_provider") not in {"CPUExecutionProvider", "DmlExecutionProvider"}:
        raise AssertionError(f"Unexpected active inference provider: {warm_info!r}")
    if args.execution_mode == "gpu" and not warm_info.get("gpu_verified"):
        raise RuntimeError(f"GPU inference was requested but not verified: {warm_info!r}")

    encoder_args = ("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1")
    encoder = EncoderProfile("fast", "libx264", encoder_args, encoder_args, False)
    rows = []
    all_calls = 0
    for repeat in range(args.repeats):
        repeat_root = root / f"repeat-{repeat + 1:02d}"
        outputs = repeat_root / "outputs"
        cache_root = repeat_root / "depth-cache"
        outputs.mkdir(parents=True)
        cache_root.mkdir(parents=True)
        store = main.DepthCache(cache_root)

        baseline_job = make_job(source, args.execution_mode, cache=False)
        baseline, _, _ = run_export(baseline_job, manager, store, outputs, encoder)
        all_calls += baseline["inference_calls"]

        write_job = make_job(source, args.execution_mode, cache=True)
        write, write_calls, _ = run_export(write_job, manager, store, outputs, encoder)
        all_calls += write_calls
        if write["cache"].get("state") != "stored":
            raise AssertionError(f"Cache write did not complete: {write['cache']!r}")
        if write_calls <= 0:
            raise AssertionError("Cache population made no real native model calls.")
        float_bytes = write["cache"].get("bytes")
        stored_frames = write["cache"].get("frames")
        disk_bytes = cache_disk_bytes(cache_root)
        if not isinstance(float_bytes, int) or float_bytes <= 0 or stored_frames != args.frames:
            raise AssertionError(f"Cache write did not retain every native float frame: {write['cache']!r}")
        if disk_bytes <= 0:
            raise AssertionError("DepthCache produced no on-disk entry.")

        hit_job = make_job(source, args.execution_mode, cache=True)
        hit, hit_calls, _ = run_export(hit_job, manager, store, outputs, encoder)
        all_calls += hit_calls
        if hit["cache"].get("state") != "hit" or hit_calls != 0:
            raise AssertionError(f"Same-settings cache replay inferred or missed: {hit['cache']!r}, calls={hit_calls}.")
        for hash_name in ("encoded_video_sha256", "pixels_sha256", "audio_pcm_sha256"):
            if hit["verification"][hash_name] != write["verification"][hash_name]:
                raise AssertionError(f"Same-settings cache hit changed {hash_name}.")

        presentation_job = make_job(source, args.execution_mode, cache=True, layout="TB",
                                    strength=8.0, convergence=0.2)
        presentation, presentation_calls, _ = run_export(presentation_job, manager, store, outputs, encoder)
        all_calls += presentation_calls
        if presentation["cache"].get("state") != "hit" or presentation_calls != 0:
            raise AssertionError(
                f"Changed-presentation cache replay inferred or missed: {presentation['cache']!r}, "
                f"calls={presentation_calls}."
            )
        if presentation["verification"]["pixels_sha256"] == hit["verification"]["pixels_sha256"]:
            raise AssertionError("Changed TB/strength/convergence settings did not change rendered pixels.")
        if presentation["verification"]["audio_pcm_sha256"] != hit["verification"]["audio_pcm_sha256"]:
            raise AssertionError("Changing presentation settings changed the copied source audio.")

        rows.append({
            "repeat": repeat + 1,
            "uncached_baseline": baseline,
            "cache_write": write,
            "same_settings_hit": hit,
            "changed_presentation_hit": presentation,
            "cache_storage": {
                "root": str(cache_root),
                "native_float32_bytes": float_bytes,
                "frames": stored_frames,
                "disk_bytes": disk_bytes,
            },
            "baseline_matches_cache_write_pixels": (
                baseline["verification"]["pixels_sha256"] == write["verification"]["pixels_sha256"]
            ),
        })
        print(json.dumps({"repeat": repeat + 1, "cache_state": hit["cache"].get("state"),
                          "inference_calls": {"baseline": baseline["inference_calls"],
                                              "write": write_calls, "hit": hit_calls,
                                              "presentation_hit": presentation_calls},
                          "cache_native_float32_bytes": float_bytes,
                          "cache_disk_bytes": disk_bytes}), flush=True)

    report = {
        "schema": "stereolab-depth-cache-verification-v1",
        "created_at_unix": time.time(),
        "source": {"path": str(source), "sha256": sha256_file(source),
                   "frames": args.frames, "size": [320, 180], "fps": SOURCE_FPS,
                   "audio": "generated 440 Hz sine at 48 kHz and AAC encoded"},
        "model": {"id": MODEL_ID, "sha256": main.MODEL_SPECS[MODEL_ID]["sha256"],
                  "accepted": manager._accepted(MODEL_ID), "input_size": INPUT_SIZE},
        "requested_execution_mode": args.execution_mode,
        "warm_execution": warm_info,
        "runtime": runtime,
        "ffmpeg": {"path": main.FFMPEG_PATH, "version": ffmpeg_version(main.FFMPEG_PATH),
                   "output_encoder": "libx264 ultrafast crf22 threads=1"},
        "repeats": args.repeats,
        "total_real_native_inference_calls_during_exports": all_calls,
        "measurements": rows,
        "scope": "Complete generated-media exports with real accepted Small-model native inference; cache hits are verified by zero inference calls and exact encoded-video, decoded-pixel, and decoded-audio hashes. Changed presentation uses stored depth; no fresh-model equality or speed claim is made.",
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "report": str(report_path),
                      "root": str(root), "repeats": args.repeats,
                      "total_real_native_inference_calls_during_exports": all_calls}, indent=2))


if __name__ == "__main__":
    EXPECTED_FRAMES = 24
    main_cli()
