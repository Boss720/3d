from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Body
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from depth_engine import ModelManager
from depth_engine import MODEL_SPECS
from job_queue import EditableJobQueue
from job_settings import SETTINGS_SCHEMA_VERSION, settings_snapshot, validate_settings
from output_naming import (DEFAULT_OUTPUT_NAME_TEMPLATE, MAX_TEMPLATE_LENGTH, NAMING_FIELDS,
                           NAMING_SCHEMA_VERSION, render_output_name, validate_output_template)
from preview_cache import PreviewDepthCache
from preview_clip import ClipCancelled, prepare_preview_clip
from preview_policy import (PREVIEW_UPLOAD_BYTES, PreviewPolicyError, preview_source_policy,
                            validate_preview_detail, validate_preview_upload_size)
from preflight import inspect_media
from telemetry import StageTimings
from calibration import CalibrationStore, measure_inputs
from resource_sampling import process_snapshot
from stereo import TemporalDepthFilter, layout_size, refine_depth, render_output
from video_encoder import select_encoder_profile
from video_pipeline import PipelineCancelled, run_pipeline


FROZEN = bool(getattr(sys, "frozen", False))
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
DATA_ROOT = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "StereoLab"
    if FROZEN else ROOT
)
STATIC = ROOT / "static"
RUNTIME = DATA_ROOT / ".runtime"
UPLOADS = RUNTIME / "uploads"
MODELS = RUNTIME / "models"
OUTPUTS = DATA_ROOT / "outputs"
CLIP_PROXIES = RUNTIME / "clip-proxies"
MODEL_PATH = MODELS / "depth.onnx"

VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".wmv", ".flv",
    ".ts", ".mts", ".m2ts", ".m4v",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LAYOUTS = {"SBS", "HALFSBS", "TB", "HALFTB", "CROSS", "ANAGLYPH", "DEPTH", "RGBD"}
NEURAL_MODES = {"depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large", "dpt-hybrid-midas"}
DEPTH_MODES = {"balanced", "precision"} | NEURAL_MODES
EXECUTION_MODES = {"auto", "cpu", "gpu", "hybrid"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
ETA_MIN_FRAMES = 5
ETA_MIN_SECONDS = 2.0

for directory in (UPLOADS, MODELS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

DEPTH_ENGINE = ModelManager(MODELS)
EXECUTION_LOCAL = threading.local()
PREVIEW_CACHE = PreviewDepthCache()
CALIBRATIONS = CalibrationStore(RUNTIME / 'calibrations.json')


def find_ffmpeg() -> str | None:
    configured = os.environ.get("VIDEO3D_FFMPEG")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())

    found = shutil.which("ffmpeg")
    if found:
        return found

    # Reuse the local FFmpeg utility already present on this machine, when
    # available. It is not copied into or redistributed with this project.
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        install_root = Path(local_app_data) / "owl3d-desktop-app"
        if install_root.is_dir():
            candidates = sorted(
                install_root.glob("app-*/resources/bin/win/x64/ffmpeg.exe"),
                key=lambda p: p.parents[4].name,
                reverse=True,
            )
            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate.resolve())
    return None


FFMPEG_PATH = find_ffmpeg()
_SOURCE_COMPONENTS = ('main.py', 'depth_engine.py', 'stereo.py', 'video_encoder.py',
                      'video_pipeline.py', 'preview_cache.py', 'telemetry.py', 'preflight.py',
                      'calibration.py', 'resource_sampling.py', 'preview_policy.py', 'preview_clip.py',
                      'job_queue.py', 'job_settings.py', 'output_naming.py')
APP_BUILD_FINGERPRINT = (hashlib.sha256(b''.join((ROOT / name).read_bytes() for name in _SOURCE_COMPONENTS)).hexdigest()
                         if all((ROOT / name).is_file() for name in _SOURCE_COMPONENTS) else None)
JOBS: dict[str, "Job"] = {}
JOBS_LOCK = threading.RLock()
JOB_QUEUE = EditableJobQueue()
WORKER: threading.Thread | None = None
WORKER_START_LOCK = threading.Lock()


@dataclass
class Job:
    id: str
    filename: str
    input_path: Path
    media_type: str
    layout: str
    strength: float
    convergence: float
    depth_mode: str
    input_size: int
    resolution: str
    quality: str
    invert_depth: bool
    temporal_smoothing: float = 0.35
    edge_refinement: bool = True
    execution_mode: str = "auto"
    execution: dict[str, Any] = field(default_factory=dict)
    first_execution: dict[str, Any] = field(default_factory=dict)
    pipeline: str = "sequential"
    encoder: str | None = None
    created_at: float = field(default_factory=time.time)
    processing_started_monotonic: float | None = None
    processing_finished_monotonic: float | None = None
    video_started_monotonic: float | None = None
    frames_processed: int = 0
    total_frames: int | None = None
    state: str = "queued"
    progress: int = 0
    message: str = "Waiting in the local queue"
    output_path: Path | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen[bytes] | None = None
    timings: StageTimings = field(default_factory=StageTimings)
    phase: str = "queued"
    retry_count: int = 0
    retry_reason: str | None = None
    rate_samples: deque = field(default_factory=lambda: deque(maxlen=32))
    preflight: dict[str, Any] | None = None
    purpose: str = "export"
    clip_start_seconds: float = 0.0
    clip_duration_seconds: float = 3.0
    preview_source_path: Path | None = None
    clip: dict[str, Any] | None = None
    output_name_template: str = DEFAULT_OUTPUT_NAME_TEMPLATE

    def public(self, queue_position: int | None = None) -> dict[str, Any]:
        elapsed_seconds = None
        processing_fps = None
        eta_seconds = None
        eta_status = "unavailable"
        frames_processed = None
        total_frames = None
        eta_range = None
        if self.media_type == "video":
            frames_processed = self.frames_processed
            total_frames = self.total_frames
            now = time.monotonic()
            if self.processing_started_monotonic is not None:
                processing_end = self.processing_finished_monotonic
                if processing_end is None:
                    processing_end = now
                elapsed_seconds = max(0.0, processing_end - self.processing_started_monotonic)
            video_elapsed = None
            if self.video_started_monotonic is not None:
                video_end = self.processing_finished_monotonic
                if video_end is None:
                    video_end = now
                video_elapsed = max(0.0, video_end - self.video_started_monotonic)
                if video_elapsed > 0 and self.frames_processed > 0:
                    processing_fps = self.frames_processed / video_elapsed
            if len(self.rate_samples) >= 2:
                sample_time, sample_frames = self.rate_samples[0]
                sample_end = self.processing_finished_monotonic if self.processing_finished_monotonic is not None else now
                span = max(0.0, sample_end - sample_time)
                gained = self.frames_processed - sample_frames
                processing_fps = gained / span if span > 0 and gained > 0 else None

            if self.state == "completed":
                eta_status = "complete"
                if self.total_frames is not None and self.total_frames > 0:
                    eta_seconds = 0.0
            elif self.state == "processing" and self.total_frames is not None and self.total_frames > 0:
                eta_status = "estimating"
                if (
                    self.frames_processed >= ETA_MIN_FRAMES
                    and self.frames_processed < self.total_frames
                    and video_elapsed is not None
                    and video_elapsed >= ETA_MIN_SECONDS
                    and processing_fps is not None
                    and processing_fps > 0
                ):
                    eta_seconds = (self.total_frames - self.frames_processed) / processing_fps
                    eta_status = "available"
                    intervals = [(b[0] - a[0]) / (b[1] - a[1])
                                 for a, b in zip(self.rate_samples, list(self.rate_samples)[1:])
                                 if b[1] > a[1] and b[0] > a[0]]
                    if len(intervals) >= 4:
                        remaining = self.total_frames - self.frames_processed
                        # Observed variation, not a confidence interval or completion guarantee.
                        eta_range = [min(eta_seconds, remaining * min(intervals)),
                                     max(eta_seconds, remaining * max(intervals))]
            if self.state == "processing" and self.phase == "finalization":
                eta_status, eta_seconds, eta_range = "finalizing", None, None

        return {
            "id": self.id,
            "filename": self.filename,
            "media_type": self.media_type,
            "layout": self.layout,
            "purpose": self.purpose,
            "settings_schema_version": SETTINGS_SCHEMA_VERSION,
            "settings": settings_snapshot(self),
            "source_available": source_available(self),
            "queue_position": queue_position,
            "naming_schema_version": NAMING_SCHEMA_VERSION,
            "output_name_template": self.output_name_template,
            "planned_output_name": output_path_for(self).name,
            "input_size": self.input_size,
            "resolution": self.resolution,
            "quality": self.quality,
            "clip_start_seconds": self.clip_start_seconds if self.purpose == 'preview_clip' else None,
            "clip_duration_seconds": self.clip_duration_seconds if self.purpose == 'preview_clip' else None,
            "clip": self.clip,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "output_name": self.output_path.name if self.output_path else None,
            "created_at": self.created_at,
            "depth_mode": self.depth_mode,
            "strength": self.strength,
            "convergence": self.convergence,
            "temporal_smoothing": self.temporal_smoothing,
            "edge_refinement": self.edge_refinement,
            "invert_depth": self.invert_depth,
            "execution_mode": self.execution_mode,
            "execution": dict(self.execution),
            "first_execution": dict(self.first_execution),
            "pipeline": self.pipeline,
            "encoder": self.encoder,
            "eta_status": eta_status,
            "elapsed_seconds": elapsed_seconds,
            "processing_fps": processing_fps,
            "eta_seconds": eta_seconds,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "timings": self.timings.snapshot(),
            "phase": self.phase,
            "retry_count": self.retry_count,
            "retry_reason": self.retry_reason,
            "eta_range_seconds": eta_range,
            "preflight": self.preflight,
        }


class JobCancelled(Exception):
    pass


def owned_upload_path(job: Job) -> Path | None:
    """Only the unique upload assigned by create_job belongs to this job."""
    path = job.input_path
    try:
        if (path.parent.resolve() == UPLOADS.resolve() and path.stem == job.id
                and path.suffix in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS and not path.is_symlink()):
            return path
    except OSError:
        pass
    return None


def source_available(job: Job) -> bool:
    path = owned_upload_path(job)
    if path is None:
        return False
    try:
        if not path.is_file():
            return False
        with path.open('rb'):
            return True
    except OSError:
        return False


def pending_jobs_locked() -> list[Job]:
    # Caller holds JOB_QUEUE.lock before JOBS_LOCK, as does every queue API.
    return [JOBS[job_id] for job_id in JOB_QUEUE.pending_ids() if job_id in JOBS
            and JOBS[job_id].state == 'queued' and not JOBS[job_id].cancel_event.is_set()]


def public_job_locked(job: Job) -> dict[str, Any]:
    positions = {item.id: position for position, item in enumerate(pending_jobs_locked(), 1)}
    return job.public(queue_position=positions.get(job.id))


def pending_export_locked(job_id: str) -> Job:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Job not found.')
    if (job.purpose != 'export' or job.state != 'queued' or job.cancel_event.is_set()
            or job_id not in JOB_QUEUE.pending_ids()):
        raise HTTPException(status_code=409, detail='This export is no longer pending. Refresh the queue before editing it.')
    return job


def safe_stem(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    stem = Path(name).stem or "stereo-output"
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._")
    return (cleaned or "stereo-output")[:100]


def output_path_for(job: Job) -> Path:
    if job.purpose == 'export':
        return OUTPUTS / render_output_name(job.output_name_template, job.filename, job.layout,
                                            job.depth_mode, job.created_at, job.id, job.media_type)
    suffix = "png" if job.media_type == "image" else "mp4"
    layout_name = job.layout.lower()
    purpose = "_preview" if job.purpose == 'preview_clip' else ""
    try:
        identity = uuid.UUID(job.id).hex
    except (ValueError, TypeError, AttributeError):
        identity = hashlib.sha256(str(job.id).encode('utf-8')).hexdigest()[:32]
    return OUTPUTS / f"{safe_stem(job.filename)}{purpose}_{layout_name}_{identity}.{suffix}"


def set_job(job_id: str, **changes: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job:
            for name, value in changes.items():
                setattr(job, name, value)


def load_depth_model() -> cv2.dnn_Net:
    if not MODEL_PATH.is_file():
        raise RuntimeError("Precision mode needs a compatible ONNX depth model. Add one in the Depth mode panel.")
    try:
        net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV could not load this ONNX model: {exc}") from exc


def estimate_depth(frame: np.ndarray, mode: str, net: cv2.dnn_Net | None, input_size: int, invert: bool, execution_mode: str = "auto") -> np.ndarray:
    height, width = frame.shape[:2]
    if mode in NEURAL_MODES:
        depth, execution = DEPTH_ENGINE.predict_with_info(frame, mode, input_size, execution_mode=execution_mode)
        EXECUTION_LOCAL.snapshot = execution
    elif mode == "precision":
        if net is None:
            raise RuntimeError("The ONNX depth model is not loaded.")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
        values = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        blob = np.transpose((values - mean) / std, (2, 0, 1))[None, ...]
        try:
            net.setInput(blob)
            prediction = np.asarray(net.forward(), dtype=np.float32).squeeze()
        except cv2.error as exc:
            raise RuntimeError(f"Depth inference failed. This ONNX model may use an unsupported input/output shape: {exc}") from exc
        if prediction.ndim != 2 or not np.isfinite(prediction).all():
            raise RuntimeError("The ONNX model must return one 2D depth map for each frame.")
        depth = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_CUBIC)
        depth = cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)
    else:
        # A deterministic offline proxy: combine luminance, image height and a
        # soft center prior. This creates adjustable disparity but is not AI.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        center = np.exp(-((x / 0.9) ** 2 + ((y - 0.55) / 0.75) ** 2)).astype(np.float32)
        depth = 0.42 * (1.0 - gray) + 0.28 * y + 0.30 * center
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=max(1.0, min(width, height) / 160.0))
        low, high = np.percentile(depth, (2, 98))
        depth = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)

    if invert:
        depth = 1.0 - depth
    return np.ascontiguousarray(depth, dtype=np.float32)


def execution_snapshot(mode: str, requested: str) -> dict[str, Any]:
    if mode in NEURAL_MODES:
        return dict(EXECUTION_LOCAL.snapshot)
    return {"requested_mode": requested, "active_provider": "OpenCV CPU", "device": "CPU",
            "gpu_verified": False, "fallback_reason":
            "The selected approximation/custom ONNX engine uses CPU processing." if requested in {"auto", "hybrid"} else None}


def ffprobe_fps(ffmpeg_path: str, source: Path, fallback: float) -> float:
    ffprobe = str(Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe"))
    if not Path(ffprobe).is_file():
        return fallback
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(source)],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        streams = json.loads(result.stdout or "{}").get("streams", [])
        if streams:
            for key in ("avg_frame_rate", "r_frame_rate"):
                raw = streams[0].get(key, "")
                if "/" in raw:
                    numerator, denominator = raw.split("/", 1)
                    fps = float(numerator) / max(float(denominator), 1e-6)
                    if 0.1 <= fps <= 240:
                        return fps
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return fallback


def stop_process(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except OSError:
            pass


def run_image(job: Job, net: cv2.dnn_Net | None) -> Path:
    with job.timings.measure("decode"):
        image = cv2.imread(str(job.input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV could not read this image. Try a standard JPG, PNG, or WebP file.")
    if job.resolution == "1080p":
        scale = min(1.0, 1920 / image.shape[1], 1080 / image.shape[0])
        if scale < 1:
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    set_job(job.id, progress=15, phase="conversion", message="Estimating relative depth")
    with job.timings.measure("depth"):
        depth = estimate_depth(image, job.depth_mode, net, job.input_size, job.invert_depth, job.execution_mode)
    execution = execution_snapshot(job.depth_mode, job.execution_mode)
    set_job(job.id, execution=execution, first_execution=dict(execution))
    with job.timings.measure("refinement"):
        depth = refine_depth(image, depth, job.edge_refinement)
    if job.cancel_event.is_set():
        raise JobCancelled()
    with job.timings.measure("render"):
        result = render_output(image, depth, job.strength, job.convergence, job.layout)
    target = output_path_for(job)
    with job.timings.measure("encode_write"):
        if not cv2.imwrite(str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError("Could not write the PNG output. Check that the output drive has free space.")
    return target


def run_video(job: Job, net: cv2.dnn_Net | None, _force_software: bool = False) -> Path:
    ffmpeg = FFMPEG_PATH
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe.")
    source = job.preview_source_path or job.input_path
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("Could not open this video. The file may use an unsupported codec or be damaged.")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    set_job(job.id, total_frames=total_frames or None, frames_processed=0)
    if source_width < 1 or source_height < 1:
        capture.release()
        raise RuntimeError("The video does not report a valid frame size.")
    fps = ffprobe_fps(ffmpeg, source, source_fps if math.isfinite(source_fps) and source_fps > 0 else 30.0)

    scale = min(1.0, 1920.0 / source_width, 1080.0 / source_height) if job.resolution == "1080p" else 1.0
    width = max(2, int(round(source_width * scale / 2.0) * 2))
    height = max(2, int(round(source_height * scale / 2.0) * 2))
    if job.layout == "HALFSBS":
        width = max(4, width // 4 * 4)
    if job.layout == "HALFTB":
        height = max(4, height // 4 * 4)
    output_width, output_height = layout_size(width, height, job.layout)
    target = output_path_for(job)
    encoder_profile = select_encoder_profile(ffmpeg, job.quality)
    encoder_args = encoder_profile.software_fallback_args if _force_software else encoder_profile.video_args
    encoder_name = (
        "CPU libx264 (NVENC fallback)" if _force_software and encoder_profile.codec == "h264_nvenc"
        else "NVIDIA NVENC" if not _force_software and encoder_profile.codec == "h264_nvenc"
        else "CPU libx264"
    )
    set_job(job.id, encoder=encoder_name, progress=1, message="Preparing video encoder")

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{output_width}x{output_height}",
        "-framerate", f"{fps:.8f}", "-i", "pipe:0",
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?",
        *encoder_args,
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    conversion_start = time.monotonic()
    set_job(job.id, process=proc, encoder=encoder_name, phase="conversion", message="Converting frames locally", video_started_monotonic=conversion_start)
    with JOBS_LOCK:
        job.rate_samples.clear()
        job.rate_samples.append((conversion_start, 0))
    stderr_lines: list[bytes] = []

    def drain_stderr() -> None:
        if proc.stderr:
            for line in proc.stderr:
                stderr_lines.append(line)
                if len(stderr_lines) > 80:
                    del stderr_lines[:20]

    reader = threading.Thread(target=drain_stderr, daemon=True)
    reader.start()
    frame_index = 0
    depth_filter = TemporalDepthFilter(job.temporal_smoothing)

    def source_frames():
        try:
            while True:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                with job.timings.measure("decode"):
                    ok, frame = capture.read()
                if not ok:
                    return
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                yield frame
        finally:
            capture.release()

    def infer_frame(frame):
        if job.cancel_event.is_set():
            raise JobCancelled()
        with job.timings.measure("depth"):
            depth = estimate_depth(frame, job.depth_mode, net, job.input_size, job.invert_depth, job.execution_mode)
        execution = execution_snapshot(job.depth_mode, job.execution_mode)
        set_job(job.id, execution=execution)
        if not job.first_execution:
            set_job(job.id, first_execution=dict(execution))
        return depth

    def consume_frame(frame, depth):
        nonlocal frame_index
        if job.cancel_event.is_set():
            raise JobCancelled()
        with job.timings.measure("refinement_temporal"):
            depth = refine_depth(frame, depth, job.edge_refinement)
            depth = depth_filter.apply(frame, depth)
        with job.timings.measure("render"):
            output_frame = render_output(frame, depth, job.strength, job.convergence, job.layout)
        if job.cancel_event.is_set():
            raise JobCancelled()
        if proc.stdin is None:
            raise RuntimeError("FFmpeg did not open its frame input.")
        with job.timings.measure("encode_write"):
            proc.stdin.write(output_frame.tobytes())
        frame_index += 1
        with JOBS_LOCK:
            job.frames_processed = frame_index
            job.rate_samples.append((time.monotonic(), frame_index))
        if frame_index == 1 or frame_index % max(1, round(fps)) == 0:
            progress = min(98, int(frame_index * 100 / total_frames)) if total_frames else min(98, frame_index // 10)
            set_job(job.id, frames_processed=frame_index, progress=progress,
                    message=f"Frame {frame_index:,}" + (f" of {total_frames:,}" if total_frames else ""))

    try:
        frames = source_frames()
        if job.execution_mode in {"auto", "hybrid"} and job.depth_mode in NEURAL_MODES:
            set_job(job.id, pipeline="overlapped")
            run_pipeline(frames, infer_frame, consume_frame, job.cancel_event, queue_size=2, timings=job.timings)
        else:
            try:
                for frame in frames:
                    consume_frame(frame, infer_frame(frame))
            finally:
                frames.close()

        capture.release()
        if frame_index == 0:
            stop_process(proc)
            raise RuntimeError("No video frames could be decoded.")
        set_job(job.id, phase="finalization", message="Finalizing video and audio")
        with job.timings.measure("finalization"):
            if proc.stdin:
                proc.stdin.close()
            return_code = proc.wait()
        reader.join(timeout=2)
        if job.cancel_event.is_set():
            raise JobCancelled()
        if return_code != 0:
            detail = b"".join(stderr_lines).decode("utf-8", errors="replace").strip()
            detail = detail[-1200:] or f"FFmpeg exited with code {return_code}."
            if encoder_profile.codec == "h264_nvenc" and not _force_software:
                capture.release()
                target.unlink(missing_ok=True)
                set_job(job.id, process=None, encoder="CPU libx264 (NVENC fallback)",
                        progress=1, frames_processed=0, video_started_monotonic=None,
                        phase="retrying", retry_count=job.retry_count + 1, retry_reason="NVENC export failed; restarting with libx264",
                        message="NVIDIA encoding failed; retrying this export with the CPU encoder")
                return run_video(job, net, _force_software=True)
            raise RuntimeError(detail)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("FFmpeg finished without producing an output file.")
        set_job(job.id, frames_processed=frame_index,
                total_frames=frame_index if total_frames > 0 else None)
        return target
    except PipelineCancelled as exc:
        raise JobCancelled() from exc
    except (BrokenPipeError, OSError) as exc:
        stop_process(proc)
        reader.join(timeout=2)
        if job.cancel_event.is_set():
            raise JobCancelled() from exc
        detail = b"".join(stderr_lines).decode("utf-8", errors="replace").strip()
        if encoder_profile.codec == "h264_nvenc" and not _force_software:
            capture.release()
            target.unlink(missing_ok=True)
            set_job(job.id, process=None, encoder="CPU libx264 (NVENC fallback)",
                    progress=1, frames_processed=0, video_started_monotonic=None,
                    phase="retrying", retry_count=job.retry_count + 1, retry_reason="NVENC pipe stopped; restarting with libx264",
                    message="NVIDIA encoding stopped; retrying this export with the CPU encoder")
            return run_video(job, net, _force_software=True)
        raise RuntimeError((detail[-1200:] or str(exc)) + "\nCheck that this FFmpeg build includes libx264.") from exc
    finally:
        capture.release()
        if proc.poll() is None:
            stop_process(proc)
        reader.join(timeout=2)


def process_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    if job.cancel_event.is_set():
        set_job(job_id, state="canceled", message="Canceled before processing")
        job.input_path.unlink(missing_ok=True)
        return

    target = output_path_for(job)
    target_owned = False
    # Finish an already admitted preview before reserving inference for export.
    # New preview misses see processing under the same gate and cannot jump ahead.
    with PREVIEW_CACHE.lock:
        set_job(job_id, state="processing", phase="initialization", progress=1, message="Preparing local depth engine",
                processing_started_monotonic=time.monotonic(), processing_finished_monotonic=None)
    try:
        # Own only an exclusively reserved path. A failed job must never remove
        # or overwrite an export already present at the selected output name.
        try:
            with target.open('xb'):
                pass
        except FileExistsError as exc:
            raise RuntimeError('The output already exists. Start a new job to choose a unique output name.') from exc
        target_owned = True
        with job.timings.measure("preflight"):
            report = inspect_media(job.input_path, job.media_type, job.layout,
                                   job.resolution, OUTPUTS, FFMPEG_PATH)
            set_job(job.id, preflight=report)
        if job.purpose == 'preview_clip':
            source_fps = report.get('fps') if not report.get('fps_provisional') else None
            source_frames = report.get('total_frames')
            if source_fps and source_frames and job.clip_start_seconds >= source_frames / source_fps:
                raise RuntimeError('Clip start is beyond the source duration. Choose an earlier timestamp.')
            if job.cancel_event.is_set():
                raise JobCancelled()
            CLIP_PROXIES.mkdir(parents=True, exist_ok=True)
            # Unique to this job; never substitute the upload path or clone mutable Job state.
            proxy = CLIP_PROXIES / f'{job.id}.mp4'
            set_job(job.id, phase='preparing', message='Preparing bounded preview segment')
            with job.timings.measure('clip_prepare'):
                prepare_preview_clip(FFMPEG_PATH, job.input_path, proxy, job.clip_start_seconds,
                                     job.clip_duration_seconds, source_fps, job.cancel_event,
                                     on_process=lambda proc: set_job(job.id, process=proc))
            job.preview_source_path = proxy
            with job.timings.measure('preflight'):
                report = inspect_media(proxy, 'video', job.layout, 'source', OUTPUTS, FFMPEG_PATH)
            # Validate the helper output before any inference, rather than trusting metadata alone.
            if (report['source_width'] > 960 or report['source_height'] > 540
                    or not report.get('total_frames') or report['total_frames'] > 120
                    or not report.get('fps') or report['fps'] > 24.001 or report.get('fps_provisional')
                    or report['total_frames'] / report['fps'] > 5.05):
                raise RuntimeError('Preview segment exceeded its processing bounds. Re-encode the source and retry.')
            set_job(job.id, preflight=report, clip={
                'source_start_seconds': job.clip_start_seconds,
                'requested_duration_seconds': job.clip_duration_seconds,
                'temporal_context': 'fresh',
                'proxy': {'width': report['source_width'], 'height': report['source_height'],
                          'fps': report['fps'], 'total_frames': report['total_frames']},
            })
        if job.cancel_event.is_set():
            raise JobCancelled()
        with job.timings.measure("initialization"):
            if job.depth_mode in NEURAL_MODES:
                DEPTH_ENGINE.validate(job.depth_mode, execution_mode=job.execution_mode)
            net = load_depth_model() if job.depth_mode == "precision" else None
        target = run_image(job, net) if job.media_type == "image" else run_video(job, net)
        if job.cancel_event.is_set():
            target.unlink(missing_ok=True)
            raise JobCancelled()
        set_job(job_id, state="completed", phase="complete", progress=100, output_path=target, process=None,
                processing_finished_monotonic=time.monotonic(),
                message="Processed clip ready" if job.purpose == 'preview_clip' else "Stereo export ready")
    except (JobCancelled, ClipCancelled):
        if target_owned:
            target.unlink(missing_ok=True)
        set_job(job_id, state="canceled", phase="canceled", process=None, processing_finished_monotonic=time.monotonic(), message="Conversion canceled")
    except Exception as exc:  # Surface local decoder/encoder/model errors in the queue.
        if target_owned:
            target.unlink(missing_ok=True)
        set_job(job_id, state="failed", phase="failed", process=None, processing_finished_monotonic=time.monotonic(), error=str(exc), message="Conversion failed")
    finally:
        if job.preview_source_path:
            job.preview_source_path.unlink(missing_ok=True)
        job.input_path.unlink(missing_ok=True)
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current:
                current.process = None


def worker_loop() -> None:
    while True:
        job_id = JOB_QUEUE.get()
        try:
            process_job(job_id)
        finally:
            JOB_QUEUE.task_done()


def ensure_worker() -> None:
    global WORKER
    # Independent from queue/job locks: concurrent uploads must not create two
    # consumers while the first thread has been created but has not started yet.
    with WORKER_START_LOCK:
        if WORKER is None or not WORKER.is_alive():
            WORKER = threading.Thread(target=worker_loop, daemon=True, name="stereolab-worker")
            WORKER.start()


app = FastAPI(title="StereoLab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def local_api_access(host: str, scheme: str, origin: str | None,
                     fetch_site: str | None, mutating: bool) -> bool:
    """Reject foreign hosts and browser cross-origin writes to the local service."""
    allowed = {'localhost', '127.0.0.1', '::1'}
    try:
        authority = urlsplit('//' + host)
        port = authority.port if authority.port is not None else (443 if scheme == 'https' else 80)
        if port < 1 or authority.netloc.endswith(':'):
            return False
        if authority.hostname not in allowed or authority.username is not None or authority.password is not None or authority.path or authority.query or authority.fragment:
            return False
        if not mutating:
            return True
        if origin is None:
            return fetch_site in {None, 'same-origin', 'none'}
        source = urlsplit(origin)
        source_port = source.port if source.port is not None else (443 if source.scheme == 'https' else 80)
        if source_port < 1 or source.netloc.endswith(':'):
            return False
        return (source.scheme == scheme and source.hostname in allowed and source_port == port
                and source.username is None and source.password is None
                and not source.path and not source.query and not source.fragment)
    except (ValueError, TypeError):
        return False


@app.middleware('http')
async def local_api_boundary(request: Request, call_next):
    if not local_api_access(request.headers.get('host', ''), request.scope.get('scheme', 'http'),
                            request.headers.get('origin'), request.headers.get('sec-fetch-site'),
                            request.method not in {'GET', 'HEAD', 'OPTIONS'}):
        return JSONResponse(status_code=403, content={'detail': 'Use the local StereoLab page on this server to make changes.'})
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "ffmpeg_found": bool(FFMPEG_PATH),
        "ffmpeg_path": FFMPEG_PATH,
        "model_available": MODEL_PATH.is_file(),
        "model_name": MODEL_PATH.name if MODEL_PATH.is_file() else None,
        "output_dir": str(OUTPUTS),
        "runtime": DEPTH_ENGINE.runtime_info(),
        "build_fingerprint": APP_BUILD_FINGERPRINT,
        "preview_capabilities": {
            "schema_version": 1,
            "processed_clip": {"supported": True, "max_duration_seconds": 5,
                               "max_frames": 120, "max_fps": 24, "max_width": 960,
                               "max_height": 540, "temporal_context": "fresh"},
        },
        "queue_capabilities": {"schema_version": 1, "reorder_queued_exports": True,
                               "edit_queued_exports": True, "remove_queued_exports": True,
                               "settings_snapshot": True},
        "naming_capabilities": {"schema_version": NAMING_SCHEMA_VERSION, "supported": True,
                                "max_template_length": MAX_TEMPLATE_LENGTH, "fields": list(NAMING_FIELDS),
                                "unique_job_id_suffix": True},
    }


@app.get('/api/diagnostics/resources')
def resources() -> dict[str, Any]:
    return process_snapshot()


@app.get("/api/models")
def get_models() -> dict[str, Any]:
    return DEPTH_ENGINE.catalog()


@app.post("/api/runtime/gpu/reset")
def reset_gpu_runtime() -> dict[str, Any]:
    # Lock order matches previews: cache first, then the engine's inference lock.
    with PREVIEW_CACHE.lock:
        DEPTH_ENGINE.reset_gpu_failures()
        PREVIEW_CACHE.clear()
    return {"reset": True}


@app.post("/api/models/{model_id}/install")
def install_model(model_id: str) -> dict[str, Any]:
    try:
        return DEPTH_ENGINE.install(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def validate_depth_mode(mode: str, execution_mode: str = "auto") -> None:
    if execution_mode not in EXECUTION_MODES:
        raise HTTPException(status_code=400, detail="Choose Auto, CPU, GPU, or GPU + CPU processing.")
    if mode not in DEPTH_MODES:
        raise HTTPException(status_code=400, detail="Choose a supported depth model or approximation.")
    if mode == "precision" and not MODEL_PATH.is_file():
        raise HTTPException(status_code=400, detail="Custom ONNX mode needs a compatible model. Import one in the model panel.")
    if execution_mode == "gpu" and mode not in NEURAL_MODES:
        raise HTTPException(status_code=400, detail="GPU inference requires a named neural model. Choose Auto or CPU for approximation or custom ONNX.")
    if mode in NEURAL_MODES:
        try:
            DEPTH_ENGINE.validate(mode, execution_mode=execution_mode)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


def png_data_url(image: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode the preview.")
    return "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")


@lru_cache(maxsize=8)
def custom_model_digest(path: str, size: int, modified: int, created: int) -> str:
    # Stat identity invalidates the bounded digest cache after model replacement.
    with open(path, 'rb') as source:
        digest = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def process_preview(payload: bytes, mode: str, size: int, layout: str, strength: float, convergence: float, invert: bool, refinement: bool, execution_mode: str = "auto", preview_detail: str = "proxy") -> dict[str, Any]:
    started = time.perf_counter()
    validate_preview_detail(preview_detail)
    validate_preview_upload_size(len(payload))
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Preview needs a decoded image frame. Try a JPG, PNG or a browser-playable video.")
    height, width = frame.shape[:2]
    policy = preview_source_policy(width, height, preview_detail)
    if (policy.width, policy.height) != (width, height):
        frame = cv2.resize(frame, (policy.width, policy.height), interpolation=cv2.INTER_AREA)
    if mode in NEURAL_MODES:
        model_path = MODELS / mode / "model.onnx"
        model_hash = MODEL_SPECS[mode]['sha256']
        runtime = DEPTH_ENGINE.runtime_info()
        runtime_key = json.dumps({key: runtime.get(key) for key in
                                 ('version', 'providers', 'gpu_device', 'capabilities')}, sort_keys=True)
    else:
        model_path = MODEL_PATH if mode == "precision" else None
        model_hash, runtime_key = None, "OpenCV CPU"
    identity = None
    if model_path is not None and model_path.is_file():
        stat = model_path.stat()
        identity = (str(model_path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if mode == "precision":
            model_hash = custom_model_digest(*identity)
    key = (hashlib.sha256(frame.tobytes()).digest(), frame.shape, mode, size,
           execution_mode, model_hash, identity, runtime_key, "preprocessing-v1", policy.cache_identity)
    # Serialize cache misses so simultaneous presentation edits cannot infer twice.
    with PREVIEW_CACHE.lock:
        cached = PREVIEW_CACHE.get(key)
        cache_hit = cached is not None
        if cached is None:
            with JOBS_LOCK:
                export_active = mode in NEURAL_MODES and any(
                    job.state == "processing" and job.depth_mode in NEURAL_MODES for job in JOBS.values())
            if export_active:
                raise HTTPException(status_code=409, detail="An export is processing. Reuse a cached preview or wait for it to finish before generating new depth.")
            net = load_depth_model() if mode == "precision" else None
            depth = estimate_depth(frame, mode, net, size, False, execution_mode)
            execution = execution_snapshot(mode, execution_mode)
            # A fallback is transient; caching it would conceal the next GPU retry.
            if mode not in NEURAL_MODES or not execution.get('fallback_reason'):
                PREVIEW_CACHE.put(key, depth, execution)
        else:
            depth, execution = cached
    if invert:
        depth = 1.0 - depth
    depth = refine_depth(frame, depth, refinement)
    output = render_output(frame, depth, strength, convergence, layout)
    depth_image = render_output(frame, depth, strength, convergence, "DEPTH")
    return {"image": png_data_url(output), "depth": png_data_url(depth_image), "model": mode,
            "source": png_data_url(frame), "source_width": frame.shape[1], "source_height": frame.shape[0],
            "original_width": policy.original_width, "original_height": policy.original_height,
            "preview_detail": policy.detail, "is_proxy": policy.is_proxy,
            "width": output.shape[1], "height": output.shape[0], "execution": execution,
            "cache": {"hit": cache_hit, **PREVIEW_CACHE.info()},
            "elapsed_seconds": max(0.0, time.perf_counter() - started)}


@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...), depth_mode: str = Form("balanced"), input_size: int = Form(518),
    layout: str = Form("SBS"), strength: float = Form(7.5), convergence: float = Form(0.5),
    invert_depth: bool = Form(False), edge_refinement: bool = Form(True),
    execution_mode: str = Form("auto"), preview_detail: str = Form("proxy"),
) -> dict[str, Any]:
    try:
        validate_preview_detail(preview_detail)
        validate_depth_mode(depth_mode, execution_mode)
        if layout not in LAYOUTS or not 0 <= strength <= 10 or not 0 <= convergence <= 1:
            raise HTTPException(status_code=400, detail="Invalid preview format, strength or convergence.")
        if input_size not in {256, 384, 518}:
            raise HTTPException(status_code=400, detail="Choose a supported model input size.")
        payload = await file.read(PREVIEW_UPLOAD_BYTES + 1)
        validate_preview_upload_size(len(payload))
        return await run_in_threadpool(process_preview, payload, depth_mode, input_size, layout, strength, convergence, invert_depth, edge_refinement, execution_mode, preview_detail)
    except HTTPException:
        raise
    except PreviewPolicyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preview failed: {exc}") from exc
    finally:
        await file.close()


def process_calibration(payload: bytes, mode: str, execution_mode: str, budget_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError('Calibration needs a decoded image frame. Select a source and try again.')
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 540 / height)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    with PREVIEW_CACHE.lock:
        with JOBS_LOCK:
            if any(job.state == 'processing' for job in JOBS.values()):
                raise HTTPException(status_code=409, detail='Wait for the active export to finish before calibrating.')
        runtime = DEPTH_ENGINE.runtime_info()
        identity = {'schema_version': 1, 'model_id': mode, 'model_hash': MODEL_SPECS[mode]['sha256'],
                    'server_build_fingerprint': APP_BUILD_FINGERPRINT,
                    'execution_mode': execution_mode, 'runtime_version': runtime.get('version'),
                    'providers': runtime.get('providers'), 'adapter': runtime.get('gpu_device'),
                    'capabilities': runtime.get('capabilities'), 'opencv_version': cv2.__version__,
                    'numpy_version': np.__version__, 'cpu_count': os.cpu_count(),
                    'cpu_identity': os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
                    'architecture': platform.machine(), 'os_version': platform.version(),
                    'source_dimensions': [frame.shape[1], frame.shape[0]],
                    'source_sha256': hashlib.sha256(frame.tobytes()).hexdigest()}
        def predict(size):
            _, execution = DEPTH_ENGINE.predict_with_info(frame, mode, size, execution_mode)
            return execution
        report = measure_inputs(predict, identity, budget_seconds, started_at=started)
        report['total_elapsed_seconds'] = max(0.0, time.monotonic() - started)
        report['elapsed_scope'] = 'Decode, identity and measurements. Response total also includes local history persistence.'
        try:
            CALIBRATIONS.save(report)
            report['saved'] = True
        except (OSError, ValueError):
            report['saved'] = False
            report['storage_warning'] = 'Measurements completed but could not be saved. Check local data-folder permissions and disk space.'
        report['total_elapsed_seconds'] = max(0.0, time.monotonic() - started)
        report['budget_exceeded'] = report['total_elapsed_seconds'] > budget_seconds
        return report


@app.post("/api/calibration")
async def calibration(file: UploadFile = File(...), depth_mode: str = Form('depth-anything-v2-small'),
                      execution_mode: str = Form('auto'), budget_seconds: float = Form(15)) -> dict[str, Any]:
    try:
        if depth_mode not in NEURAL_MODES or depth_mode == 'dpt-hybrid-midas':
            raise HTTPException(status_code=400, detail='Input-detail calibration requires a dynamic Depth Anything model. MiDaS uses fixed 384 px input.')
        if not 1 <= budget_seconds <= 30:
            raise HTTPException(status_code=400, detail='Choose a calibration time budget from 1 to 30 seconds.')
        validate_depth_mode(depth_mode, execution_mode)
        payload = await file.read(24 * 1024 * 1024 + 1)
        if len(payload) > 24 * 1024 * 1024:
            raise HTTPException(status_code=413, detail='The calibration frame exceeds 24 MB.')
        return await run_in_threadpool(process_calibration, payload, depth_mode, execution_mode, budget_seconds)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()


@app.post("/api/model")
async def upload_model(model: UploadFile = File(...)) -> dict[str, Any]:
    if Path(model.filename or "").suffix.lower() != ".onnx":
        await model.close()
        raise HTTPException(status_code=400, detail="Choose an ONNX model file (.onnx).")
    candidate = MODELS / f"{uuid.uuid4()}.onnx"
    total = 0
    try:
        with candidate.open("wb") as target:
            while chunk := await model.read(1024 * 1024):
                total += len(chunk)
                if total > 2 * 1024 * 1024 * 1024:
                    target.close()
                    candidate.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="The model file is larger than 2 GB.")
                target.write(chunk)
    except Exception:
        candidate.unlink(missing_ok=True)
        raise
    finally:
        await model.close()
    try:
        loaded = cv2.dnn.readNetFromONNX(str(candidate))
        del loaded
    except cv2.error as exc:
        candidate.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"OpenCV could not load this ONNX model: {exc}") from exc
    os.replace(candidate, MODEL_PATH)
    return {"ok": True, "model_available": True, "model_name": Path(model.filename or "depth.onnx").name, "bytes": total}


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    layout: str = Form("SBS"),
    strength: float = Form(7.5),
    convergence: float = Form(0.5),
    depth_mode: str = Form("balanced"),
    input_size: int = Form(518),
    resolution: str = Form("source"),
    quality: str = Form("balanced"),
    invert_depth: bool = Form(False),
    temporal_smoothing: float = Form(0.35),
    edge_refinement: bool = Form(True),
    execution_mode: str = Form("auto"),
    purpose: str = Form("export"),
    clip_start_seconds: float = Form(0.0),
    clip_duration_seconds: float = Form(3.0),
    output_name_template: Any = Form(DEFAULT_OUTPUT_NAME_TEMPLATE), request: Request = None,
) -> dict[str, Any]:
    filename = Path((file.filename or "input").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if purpose not in {'export', 'preview_clip'}:
        await file.close()
        raise HTTPException(status_code=400, detail='Choose export or preview_clip as job purpose.')
    if purpose == 'export':
        # FastAPI applies a Form default to an explicitly empty field. Inspect
        # the already parsed form so an empty supplied template is rejected.
        if request is not None:
            form = await request.form()
            if 'output_name_template' in form:
                output_name_template = form['output_name_template']
        try:
            output_name_template = validate_output_template(output_name_template)
        except ValueError as exc:
            await file.close()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        output_name_template = DEFAULT_OUTPUT_NAME_TEMPLATE
    if purpose == 'preview_clip' and (extension not in VIDEO_EXTENSIONS
            or not math.isfinite(clip_start_seconds) or clip_start_seconds < 0
            or not math.isfinite(clip_duration_seconds) or not 1 <= clip_duration_seconds <= 5):
        await file.close()
        raise HTTPException(status_code=400, detail='Processed clips need a video, a finite start at or above zero, and a duration from 1 to 5 seconds.')
    if extension not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
        await file.close()
        raise HTTPException(status_code=415, detail="Choose a supported video or still image.")
    if layout not in LAYOUTS:
        await file.close()
        raise HTTPException(status_code=400, detail="Choose a supported stereo layout.")
    try:
        validate_depth_mode(depth_mode, execution_mode)
    except HTTPException:
        await file.close()
        raise
    if not 0.0 <= strength <= 10.0:
        await file.close()
        raise HTTPException(status_code=400, detail="3D strength must be between 0 and 10.")
    if not 0.0 <= convergence <= 1.0:
        await file.close()
        raise HTTPException(status_code=400, detail="Convergence must be between 0 and 1.")
    if not 0 <= temporal_smoothing <= 0.9:
        await file.close()
        raise HTTPException(status_code=400, detail="Temporal smoothing must be between 0 and 0.9.")
    if extension in VIDEO_EXTENSIONS and not FFMPEG_PATH:
        await file.close()
        raise HTTPException(status_code=409, detail="FFmpeg is needed for video export. Install it or set VIDEO3D_FFMPEG.")

    job_id = str(uuid.uuid4())
    input_path = UPLOADS / f"{job_id}{extension}"
    total = 0
    try:
        with input_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    target.close()
                    input_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="The media file is larger than 8 GB.")
                target.write(chunk)
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    media_type = "image" if extension in IMAGE_EXTENSIONS else "video"
    job = Job(
        id=job_id,
        filename=filename,
        input_path=input_path,
        media_type=media_type,
        layout=layout,
        strength=strength,
        convergence=convergence,
        depth_mode=depth_mode,
        input_size=input_size if input_size in {256, 384, 518} else 256,
        resolution=resolution if resolution in {"source", "1080p"} else "source",
        quality=quality if quality in {"fast", "balanced", "high"} else "balanced",
        invert_depth=invert_depth,
        temporal_smoothing=temporal_smoothing,
        edge_refinement=edge_refinement,
        execution_mode=execution_mode,
        purpose=purpose,
        clip_start_seconds=clip_start_seconds,
        clip_duration_seconds=clip_duration_seconds,
        output_name_template=output_name_template,
    )
    with JOB_QUEUE.lock, JOBS_LOCK:
        JOBS[job_id] = job
        JOB_QUEUE.put(job_id)
    ensure_worker()
    with JOB_QUEUE.lock, JOBS_LOCK:
        return public_job_locked(job)


@app.get("/api/jobs")
def get_jobs() -> list[dict[str, Any]]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        positions = {item.id: position for position, item in enumerate(pending_jobs_locked(), 1)}
        return [job.public(queue_position=positions.get(job.id))
                for job in sorted(JOBS.values(), key=lambda item: item.created_at)]


@app.post('/api/queue/reorder')
def reorder_queue(body: Any = Body(...)) -> dict[str, Any]:
    if (type(body) is not dict or set(body) != {'job_ids'} or type(body['job_ids']) is not list
            or any(type(item) is not str for item in body['job_ids'])
            or len(body['job_ids']) != len(set(body['job_ids']))):
        raise HTTPException(status_code=400, detail='Provide each pending export ID once in a job_ids array, without extra fields.')
    order = body['job_ids']
    with JOB_QUEUE.lock, JOBS_LOCK:
        eligible = [job.id for job in pending_jobs_locked() if job.purpose == 'export']
        try:
            JOB_QUEUE.reorder_subset(order, eligible)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail='The queue changed or work started. Refresh the queue and try again.') from exc
    return {'job_ids': order}


@app.post('/api/jobs/{job_id}/settings')
def edit_queued_settings(job_id: str, body: Any = Body(...)) -> dict[str, Any]:
    try:
        if type(body) is not dict or set(body) != {'settings'}:
            raise ValueError('Provide one complete settings object, without extra fields.')
        settings = validate_settings(body['settings'])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with JOB_QUEUE.lock, JOBS_LOCK:
        pending_export_locked(job_id)
    # Model/device readiness can inspect local runtime state. Do not prevent a
    # worker from starting while this check runs; membership is rechecked below.
    validate_depth_mode(settings['depth_mode'], settings['execution_mode'])
    with JOB_QUEUE.lock, JOBS_LOCK:
        job = pending_export_locked(job_id)
        for name, value in settings.items():
            setattr(job, name, value)
        return public_job_locked(job)


@app.post('/api/jobs/{job_id}/naming')
def edit_queued_naming(job_id: str, body: Any = Body(...)) -> dict[str, Any]:
    try:
        if type(body) is not dict or set(body) != {'output_name_template'}:
            raise ValueError('Provide only an output_name_template string.')
        template = validate_output_template(body['output_name_template'])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with JOB_QUEUE.lock, JOBS_LOCK:
        job = pending_export_locked(job_id)
        job.output_name_template = template
        return public_job_locked(job)


@app.post('/api/jobs/{job_id}/remove-queued')
def remove_queued_job(job_id: str) -> dict[str, Any]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        job = pending_export_locked(job_id)
        path = owned_upload_path(job)
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise HTTPException(status_code=409, detail='The queued upload could not be removed. Check file permissions and retry.') from exc
        JOB_QUEUE.remove_pending(job.id)
        job.cancel_event.set()
        job.state, job.phase, job.message = 'canceled', 'canceled', 'Removed from the local queue'
        job.processing_finished_monotonic = time.monotonic()
        return public_job_locked(job)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.state in {"completed", "failed", "canceled"}:
            return public_job_locked(job)
        job.cancel_event.set()
        proc = job.process
    stop_process(proc)
    set_job(job_id, message="Canceling local conversion")
    with JOB_QUEUE.lock, JOBS_LOCK:
        current = JOBS[job_id]
        return public_job_locked(current)


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        output = job.output_path if job else None
    if output is None or not output.is_file() or output.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail="Completed output was not found.")
    return FileResponse(output, filename=output.name)


@app.get('/api/preview-clips/{job_id}/video')
def preview_clip_video(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        output = job.output_path if job and job.purpose == 'preview_clip' and job.state == 'completed' else None
    if output is None or not output.is_file() or output.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail='Completed preview clip was not found.')
    return FileResponse(output, media_type='video/mp4', filename=output.name, content_disposition_type='inline')


if __name__ == "__main__":
    print("StereoLab is running at http://127.0.0.1:8765")
    print("All processing stays on this computer. Press Ctrl+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
