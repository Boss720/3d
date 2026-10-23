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
from contextlib import asynccontextmanager
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Annotated
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
from build_identity import SOURCE_COMPONENTS, build_fingerprint
from depth_cache import DepthCache, CacheError
from depth_cache_identity import build_cache_key, hash_source, frame_digest
from job_queue import EditableJobQueue
from local_workspace import (storage_inventory, cleanup_temporary_files,
                             validate_output_directory, open_folder, reveal_file)
from job_history import HistoryStore, HistoryUnavailable, HistoryCapacityError, HistoryConflict
from job_settings import SETTINGS_SCHEMA_VERSION, settings_snapshot, validate_settings
from output_naming import (DEFAULT_OUTPUT_NAME_TEMPLATE, MAX_TEMPLATE_LENGTH, NAMING_FIELDS,
                           NAMING_SCHEMA_VERSION, render_output_name, validate_output_template)
from preview_cache import PreviewDepthCache
from preview_clip import ClipCancelled, prepare_preview_clip
from preview_policy import (PREVIEW_UPLOAD_BYTES, PreviewPolicyError, preview_source_policy,
                            validate_preview_detail, validate_preview_upload_size)
from preflight import FFMPEG_INSTALL_GUIDANCE, inspect_media, video_metadata
from telemetry import StageTimings
from calibration import CalibrationStore, measure_inputs
from resource_sampling import process_snapshot
from stereo import StereoRenderer, TemporalDepthFilter, layout_size, refine_depth, render_output
from video_encoder import select_encoder_profile
from video_pipeline import PipelineCancelled, run_pipeline
from video_depth import VideoDepthEstimator
from video_decoder import DecoderCancelled, open_video_capture
from upload_admission import UploadAdmissionMiddleware
from platform_paths import resolve_paths
from media_process import external_process_environment, fps_passthrough_args


FROZEN = bool(getattr(sys, "frozen", False))
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
PATHS = resolve_paths(frozen=FROZEN, root=ROOT)
DATA_ROOT = PATHS.data_root
STATIC = ROOT / "static"
RUNTIME = PATHS.runtime
UPLOADS = PATHS.uploads
MODELS = PATHS.models
OUTPUTS = PATHS.outputs
CLIP_PROXIES = PATHS.clip_proxies
MODEL_PATH = MODELS / "depth.onnx"

VIDEO_EXTENSIONS = {
    ".mp4", ".webm", ".mkv", ".mov", ".avi", ".wmv", ".flv",
    ".ts", ".mts", ".m2ts", ".m4v",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LAYOUTS = {"SBS", "XREAL_SBS", "HALFSBS", "TB", "HALFTB", "CROSS", "ANAGLYPH", "DEPTH", "RGBD"}
NEURAL_MODES = set(MODEL_SPECS)
DEPTH_MODES = {"balanced", "precision"} | NEURAL_MODES
EXECUTION_MODES = {"auto", "cpu", "gpu", "cuda", "hybrid"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
MAX_MODEL_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
CALIBRATION_UPLOAD_BYTES = 24 * 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 256 * 1024
ETA_MIN_FRAMES = 5
ETA_MIN_SECONDS = 2.0
MAX_PENDING_JOBS = 32
MAX_STAGED_UPLOADS = 4
MAX_SESSION_JOBS = 512

for directory in (UPLOADS, MODELS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

DEPTH_ENGINE = ModelManager(MODELS)
EXECUTION_LOCAL = threading.local()
PREVIEW_CACHE = PreviewDepthCache()
VIDEO_DEPTH_CACHE: DepthCache | None = None
DEPTH_CACHE_START_LOCK = threading.Lock()
CALIBRATIONS = CalibrationStore(PATHS.calibration_file)


def find_ffmpeg() -> str | None:
    configured = os.environ.get("VIDEO3D_FFMPEG")
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())

    found = shutil.which("ffmpeg")
    if found:
        return found

    # Reuse the Windows FFmpeg utility already present on this machine, when
    # available. It is not copied into or redistributed with this project.
    if sys.platform != 'win32':
        return None
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
_SOURCE_COMPONENTS = SOURCE_COMPONENTS
APP_BUILD_FINGERPRINT = build_fingerprint(ROOT, frozen=FROZEN)
JOBS: dict[str, "Job"] = {}
JOBS_LOCK = threading.RLock()
STAGED_UPLOADS: set[Path] = set()
JOB_QUEUE = EditableJobQueue(maxsize=MAX_PENDING_JOBS)
WORKER: threading.Thread | None = None
WORKER_START_LOCK = threading.Lock()
WORKER_STOP = threading.Event()
ACCEPTING_MUTATIONS = True
HISTORY: HistoryStore | None = None
HISTORY_WARNING: str | None = None


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
    depth_interval: int = 1
    video_decode: str = 'software'
    use_depth_cache: bool = False
    depth_cache: dict[str, Any] = field(default_factory=lambda: {
        'state': 'disabled', 'frames': 0, 'bytes': 0, 'warning': None})
    decoder: dict[str, Any] = field(default_factory=dict)
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
    history_admitted: bool = False
    history_warning: str | None = None
    cleanup_warning: str | None = None
    cleanup_pending: dict[Path, str] = field(default_factory=dict, repr=False)

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
            "history_warning": self.history_warning,
            "cleanup_warning": self.cleanup_warning,
            "output_name": self.output_path.name if self.output_path else None,
            "created_at": self.created_at,
            "depth_mode": self.depth_mode,
            "strength": self.strength,
            "convergence": self.convergence,
            "temporal_smoothing": self.temporal_smoothing,
            "edge_refinement": self.edge_refinement,
            "invert_depth": self.invert_depth,
            "execution_mode": self.execution_mode,
            "depth_interval": self.depth_interval,
            "video_decode": self.video_decode,
            "depth_cache": dict(self.depth_cache),
            "execution": dict(self.execution),
            "first_execution": dict(self.first_execution),
            "pipeline": self.pipeline,
            "encoder": self.encoder,
            "decoder": dict(self.decoder),
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
            env=external_process_environment(),
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


def stop_video_encoder(proc: subprocess.Popen[bytes], reader: threading.Thread) -> None:
    """Stop one owned export encoder and close its pipes after the stderr reader exits."""
    if proc.poll() is None:
        stop_process(proc)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            pass

    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except (OSError, ValueError):
            pass
    if reader.ident is not None:
        reader.join(timeout=2)
    reader_stopped = not reader.is_alive()
    if reader_stopped and proc.stderr is not None:
        try:
            proc.stderr.close()
        except (OSError, ValueError):
            pass
    if proc.poll() is None or not reader_stopped:
        raise RuntimeError('Video encoder cleanup failed: owned process or stderr reader survived the shutdown timeout.')


def cleanup_job_file(job: Job, path: Path, label: str) -> None:
    """Keep cleanup errors separate from the actual conversion outcome."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        with JOBS_LOCK:
            job.cleanup_pending[path] = label
            labels = ', '.join(sorted(set(job.cleanup_pending.values())))
            job.cleanup_warning = (f'Temporary files could not be removed ({labels}) and remain on disk. '
                                   'Close apps using them and check permissions in the local data folder.')
    else:
        with JOBS_LOCK:
            job.cleanup_pending.pop(path, None)
            if not job.cleanup_pending:
                job.cleanup_warning = None


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


def decoder_stopped_early(expected_frames: int | None, decoded_frames: int,
                          source_fps: float, position_ms: float | None) -> bool:
    """Detect a substantial short decode while tolerating approximate metadata."""
    if expected_frames is None or expected_frames <= decoded_frames or source_fps <= 0:
        return False

    expected_seconds = expected_frames / source_fps
    tolerance_seconds = max(2.0 / source_fps, min(0.5, expected_seconds * 0.1))
    tolerance_frames = max(2, math.ceil(source_fps * tolerance_seconds))
    shortfall = expected_frames - decoded_frames
    if shortfall <= tolerance_frames:
        return False

    if position_ms is not None and math.isfinite(position_ms) and position_ms > 0:
        # Backends commonly report the timestamp of the last decoded frame,
        # rather than the end boundary, so include one frame interval.
        decoded_seconds = position_ms / 1000.0 + 1.0 / source_fps
        return expected_seconds - decoded_seconds > tolerance_seconds

    # If the backend exposes no playback position, require a clear count gap.
    return shortfall > max(tolerance_frames, math.ceil(expected_frames * 0.001))


class CacheReplayError(Exception):
    """A cache cannot describe this decode; restart once from original inference."""


def get_video_depth_store() -> DepthCache:
    global VIDEO_DEPTH_CACHE
    with DEPTH_CACHE_START_LOCK:
        if VIDEO_DEPTH_CACHE is None:
            VIDEO_DEPTH_CACHE = DepthCache(PATHS.depth_cache)
        return VIDEO_DEPTH_CACHE


class VideoDepthCacheSession:
    """Own a cache lease for one complete attempt, including encoder finalization."""

    def __init__(self, job: Job, store=None, bypass: bool = False):
        self.job = job
        self.store = store
        self.bypass = bypass
        self.reader = self.writer = None
        self.key = None
        self.frames = 0
        self.bytes = 0
        self.execution = {}
        self.estimated_bytes = None

    def status(self, state, warning=None, **extra):
        self.job.depth_cache = {'state': state, 'frames': self.frames, 'bytes': self.bytes,
                                'warning': warning, 'estimated_bytes': self.estimated_bytes, **extra}

    def unavailable(self, error):
        warning = f'Depth reuse unavailable: {error}'[:400]
        self.status('unavailable', warning)
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.abort()
            except (CacheError, OSError, ValueError):
                self.status('unavailable', warning + ' Partial cache cleanup can be retried in Local storage.')

    def start(self, source, width, height, fps, decoder):
        job = self.job
        if not job.use_depth_cache:
            self.status('disabled')
            return
        if self.bypass:
            self.status('unavailable', 'Cached frames did not match this decode; rendering again with fresh depth.')
            return
        if job.purpose != 'export' or job.media_type != 'video' or job.depth_mode not in NEURAL_MODES:
            self.status('ineligible', 'Depth reuse is available for ordinary named-model video exports.')
            return
        if APP_BUILD_FINGERPRINT is None:
            self.status('unavailable', 'Depth reuse requires a verified app build identity. Rendered output is unaffected.')
            return
        try:
            if self.store is None:
                self.store = get_video_depth_store()
            # Verify exact bytes for both source and model. A receipt or filename
            # alone cannot authorize reuse of a replaced model or changed source.
            source_hash = hash_source(source, job.cancel_event)
            model_hash = hash_source(MODELS / job.depth_mode / 'model.onnx', job.cancel_event)
            if model_hash != MODEL_SPECS[job.depth_mode]['sha256']:
                raise CacheError('Installed model bytes do not match the pinned model.')
            runtime = DEPTH_ENGINE.runtime_info()
            runtime_identity = {name: runtime.get(name) for name in
                                ('version', 'providers', 'gpu_device', 'capabilities')}
            runtime_identity['opencv'] = cv2.__version__
            self.key = build_cache_key(
                source_hash, model_id=job.depth_mode, model_sha256=model_hash,
                input_size=job.input_size, execution_mode=job.execution_mode,
                working_size=(width, height), fps=fps, invert_depth=job.invert_depth,
                temporal_smoothing=job.temporal_smoothing, depth_interval=job.depth_interval,
                decoder_identity={name: decoder.get(name) for name in
                                  ('backend', 'hardware', 'adapter_id', 'adapter_name')},
                runtime_identity=runtime_identity,
                pipeline_version=f'native-temporal-v1:{APP_BUILD_FINGERPRINT}')
            warning = None
            try:
                self.reader = self.store.open_reader(self.key)
            except (CacheError, OSError, ValueError) as exc:
                warning = f'Previous depth entry was rejected: {exc}'[:400]
            if self.reader is not None:
                if self.reader.metadata.get('stored_stage') != 'native-temporal-v1':
                    raise CacheError('The saved depth stage is incompatible.')
                self.execution = self.reader.metadata.get('execution', {})
                if type(self.execution) is not dict:
                    raise CacheError('The saved execution record is invalid.')
                self.status('hit')
            else:
                self.writer = self.store.begin(self.key)
                self.status('miss', warning)
        except InterruptedError as exc:
            raise JobCancelled() from exc
        except (CacheError, OSError, ValueError) as exc:
            self.close()
            self.unavailable(exc)

    def read(self, frame):
        if self.reader is None:
            return None
        try:
            depth = self.reader.read(frame_digest(frame))
            if depth.ndim != 2 or not depth.size or not np.isfinite(depth).all():
                raise CacheError('Saved native depth must contain a nonempty finite 2D map.')
        except (CacheError, OSError, ValueError) as exc:
            raise CacheReplayError(str(exc)) from exc
        self.frames += 1
        self.bytes += depth.nbytes
        self.status('hit')
        # The original provider describes the saved computation. This attempt
        # does no neural inference; publish that distinction explicitly.
        execution = {'requested_mode': self.job.execution_mode, 'active_provider': None,
                     'device': 'cpu', 'gpu_verified': False, 'fallback_reason': None,
                     'depth_cache_hit': True, 'cached_execution': self.execution,
                     'temporal_width': depth.shape[1], 'temporal_height': depth.shape[0],
                     'video_depth_stats': {'frames': self.frames, 'neural_frames': 0,
                                          'flow_frames': 0, 'cached_frames': self.frames}}
        set_job(self.job.id, execution=execution)
        self.job.execution = execution
        if not self.job.first_execution:
            self.job.first_execution = dict(execution)
        return depth

    def append(self, frame, depth, execution):
        if self.writer is None:
            return
        if execution.get('fallback_reason'):
            self.unavailable('Temporary CPU fallback results are not retained.')
            return
        try:
            if self.frames == 0 and self.job.total_frames:
                # Conservative first-map projection: frame geometry is known
                # only after inference, and container frame counts can vary.
                self.estimated_bytes = self.job.total_frames * (depth.nbytes + 512) + 256 * 1024
                if self.estimated_bytes > self.store.quota_bytes:
                    self.unavailable('Estimated depth storage exceeds the 2 GiB cache quota; exporting without reuse.')
                    return
            self.writer.append(depth, frame_digest(frame))
            self.frames += 1
            self.bytes += depth.nbytes
            self.execution = dict(execution)
            self.status('writing')
        except (CacheError, OSError, ValueError) as exc:
            self.unavailable(exc)

    def check_complete(self):
        if self.reader is not None and self.frames != self.reader.frame_count:
            raise CacheReplayError('The decoded frame count differs from the saved depth.')

    def finish(self):
        self.check_complete()
        if self.job.cancel_event.is_set():
            raise JobCancelled()
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                with JOBS_LOCK:
                    if self.job.cancel_event.is_set():
                        self.writer = writer
                        raise JobCancelled()
                    writer.finish({'execution': self.execution, 'stored_stage': 'native-temporal-v1'})
                self.status('stored')
            except (CacheError, OSError, ValueError) as exc:
                try:
                    writer.abort()
                except (CacheError, OSError, ValueError):
                    pass
                self.unavailable(exc)

    def close(self):
        if self.reader is not None:
            reader, self.reader = self.reader, None
            try:
                reader.close()
            except (CacheError, OSError, ValueError) as exc:
                self.unavailable(exc)
        if self.writer is not None:
            writer, self.writer = self.writer, None
            try:
                writer.abort()
                self.status('unavailable', 'Incomplete depth was not saved.')
            except (CacheError, OSError, ValueError) as exc:
                self.unavailable(exc)


def run_video(job: Job, net: cv2.dnn_Net | None, _force_software: bool = False,
              _cache_bypass: bool = False) -> Path:
    ffmpeg = FFMPEG_PATH
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg was not found. " + FFMPEG_INSTALL_GUIDANCE)
    source = job.preview_source_path or job.input_path
    cache = VideoDepthCacheSession(job, bypass=_cache_bypass)
    try:
        capture = open_video_capture(source, ffmpeg, cancel_event=job.cancel_event,
                                     prefer_hardware=job.video_decode == 'amd')
    except DecoderCancelled as exc:
        raise JobCancelled() from exc
    try:
        info = getattr(capture, 'decoder_info', None)
        if isinstance(info, dict):
            set_job(job.id, decoder=dict(info))
        try:
            return run_video_capture(job, net, ffmpeg, source, capture, _force_software, cache)
        except CacheReplayError:
            if _cache_bypass:
                raise
            # Entire temporal sequence must restart; mixing cached history and
            # fresh estimates midway through a stream would change the result.
            cache.close()
            capture.release()
            job.first_execution = {}
            set_job(job.id, execution={}, frames_processed=0, video_started_monotonic=None,
                    retry_count=job.retry_count + 1, phase='retrying',
                    retry_reason='Depth cache did not match decoded frames; restarting with fresh depth')
            return run_video(job, net, _force_software, _cache_bypass=True)
    finally:
        cache.close()
        # Own the whole capture lifetime, including metadata and encoder setup.
        capture.release()
        info = getattr(capture, 'decoder_info', None)
        if isinstance(info, dict):
            set_job(job.id, decoder=dict(info))


def run_video_capture(job: Job, net: cv2.dnn_Net | None, ffmpeg: str, source: Path,
                      capture, _force_software: bool, cache=None) -> Path:
    if not capture.isOpened():
        raise RuntimeError("Could not open this video. The file may use an unsupported codec or be damaged.")

    source_width, source_height, total_frames, source_fps, _ = video_metadata(capture)
    metadata_total_frames = total_frames
    set_job(job.id, total_frames=total_frames, frames_processed=0)
    probe = getattr(capture, 'probe', None)
    # The decoder's timestamp resampling rate must also drive rawvideo encoding
    # and cache identity. Legacy injected captures have no owned probe record.
    fps = source_fps if isinstance(probe, dict) else ffprobe_fps(ffmpeg, source, source_fps)
    fps_spec = probe.get('fps_ratio', f'{fps:.8f}') if isinstance(probe, dict) else f'{fps:.8f}'

    scale = min(1.0, 1920.0 / source_width, 1080.0 / source_height) if job.resolution == "1080p" else 1.0
    width = max(2, int(round(source_width * scale / 2.0) * 2))
    height = max(2, int(round(source_height * scale / 2.0) * 2))
    if job.layout == "HALFSBS":
        width = max(4, width // 4 * 4)
    if job.layout == "HALFTB":
        height = max(4, height // 4 * 4)
    output_width, output_height = layout_size(width, height, job.layout)
    if cache is not None:
        info = getattr(capture, 'decoder_info', {})
        cache.start(source, width, height, fps, info if isinstance(info, dict) else {})
    target = output_path_for(job)
    encoder_profile = select_encoder_profile(ffmpeg, job.quality)
    encoder_args = encoder_profile.software_fallback_args if _force_software else encoder_profile.video_args
    encoder_name = (
        "CPU libx264 (NVENC fallback)" if _force_software and encoder_profile.codec == "h264_nvenc"
        else "NVIDIA NVENC" if not _force_software and encoder_profile.codec == "h264_nvenc"
        else "CPU libx264"
    )
    set_job(job.id, encoder=encoder_name, progress=1, message="Preparing video encoder")

    video_offset = (max(0.0, probe['video_start_seconds'] - probe['format_start_seconds'])
                    if isinstance(probe, dict) else 0.0)
    # An input -itsoffset rounds to the raw frame time base (seconds at low
    # FPS). Shift in a microsecond filter/encoder time base instead.
    timeline_args = (['-vf', f'settb=AVTB,setpts=PTS+{video_offset:.9f}/TB',
                      '-enc_time_base:v', '1:1000000'] if video_offset else [])

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{output_width}x{output_height}",
        "-framerate", fps_spec, "-i", "pipe:0",
        "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?",
        *encoder_args,
        *timeline_args,
        *fps_passthrough_args(ffmpeg, stream='v'),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, env=external_process_environment())
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
    frame_index = 0
    decoded_frame_count = 0
    decoder_end_position_ms: float | None = None
    try:
        reader.start()
        depth_filter = TemporalDepthFilter(job.temporal_smoothing)
        native_estimator = (VideoDepthEstimator(job.depth_interval, job.temporal_smoothing)
                            if job.depth_mode in NEURAL_MODES else None)
        renderer = StereoRenderer()
    except Exception as exc:
        try:
            stop_video_encoder(proc, reader)
        except Exception as cleanup_error:
            exc.video_cleanup_error = str(cleanup_error)
        raise

    def source_frames():
        nonlocal decoded_frame_count, decoder_end_position_ms
        try:
            while True:
                if job.cancel_event.is_set():
                    raise JobCancelled()
                with job.timings.measure("decode"):
                    ok, frame = capture.read()
                if not ok:
                    try:
                        decoder_end_position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                    except (cv2.error, TypeError, ValueError, OverflowError, KeyError):
                        decoder_end_position_ms = None
                    return
                decoded_frame_count += 1
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
                yield frame
        finally:
            capture.release()

    def infer_frame(frame):
        if job.cancel_event.is_set():
            raise JobCancelled()
        if cache is not None:
            with job.timings.measure('depth_cache_read'):
                cached_depth = cache.read(frame)
            if cached_depth is not None:
                return cached_depth
        with job.timings.measure("depth"):
            if native_estimator is not None:
                def predict_native(source_frame):
                    result, info = DEPTH_ENGINE.predict_native_with_info(
                        source_frame, job.depth_mode, job.input_size, job.execution_mode)
                    EXECUTION_LOCAL.snapshot = info
                    return 1.0 - result if job.invert_depth else result
                depth = native_estimator.process(frame, predict_native)
            else:
                depth = estimate_depth(frame, job.depth_mode, net, job.input_size, job.invert_depth, job.execution_mode)
        execution = execution_snapshot(job.depth_mode, job.execution_mode)
        if native_estimator is not None:
            execution.update({'video_depth_method': native_estimator.last_method,
                              'video_depth_stats': dict(native_estimator.stats),
                              'temporal_width': depth.shape[1], 'temporal_height': depth.shape[0]})
        set_job(job.id, execution=execution)
        if not job.first_execution:
            set_job(job.id, first_execution=dict(execution))
        if cache is not None:
            with job.timings.measure('depth_cache_write'):
                cache.append(frame, depth, execution)
        return depth

    def consume_frame(frame, depth):
        nonlocal frame_index
        if job.cancel_event.is_set():
            raise JobCancelled()
        with job.timings.measure("refinement_temporal"):
            if native_estimator is not None:
                depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
            depth = refine_depth(frame, depth, job.edge_refinement)
            if native_estimator is None:
                depth = depth_filter.apply(frame, depth)
        with job.timings.measure("render"):
            output_frame = renderer.render(frame, depth, job.strength, job.convergence, job.layout)
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
        def stop_video_io():
            # Stop both owned children before pipeline joins. A failed inference
            # must unblock a decoder read as well as an encoder pipe write.
            stop = getattr(capture, 'stop', None)
            if callable(stop):
                stop()
            stop_process(proc)
        if job.execution_mode in {"auto", "hybrid"} and job.depth_mode in NEURAL_MODES:
            set_job(job.id, pipeline="overlapped")
            run_pipeline(frames, infer_frame, consume_frame, job.cancel_event, queue_size=2,
                         timings=job.timings, on_stop=stop_video_io)
        else:
            try:
                for frame in frames:
                    consume_frame(frame, infer_frame(frame))
            finally:
                frames.close()

        capture.release()
        if cache is not None:
            cache.check_complete()
        if frame_index == 0:
            stop_process(proc)
            raise RuntimeError("No video frames could be decoded.")
        if decoder_stopped_early(metadata_total_frames, decoded_frame_count,
                                  source_fps, decoder_end_position_ms):
            stop_process(proc)
            raise RuntimeError(
                f"Video decoding stopped early after {decoded_frame_count:,} of approximately "
                f"{metadata_total_frames:,} expected frames. Re-encode the source and retry."
            )
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
                if cache is not None:
                    cache.close()
                capture.release()
                target.unlink(missing_ok=True)
                set_job(job.id, process=None, encoder="CPU libx264 (NVENC fallback)",
                        progress=1, frames_processed=0, video_started_monotonic=None,
                        phase="retrying", retry_count=job.retry_count + 1, retry_reason="NVENC export failed; restarting with libx264",
                        message="NVIDIA encoding failed; retrying this export with the CPU encoder")
                return run_video(job, net, _force_software=True,
                                 _cache_bypass=bool(cache and cache.bypass))
            raise RuntimeError(detail)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("FFmpeg finished without producing an output file.")
        set_job(job.id, frames_processed=frame_index, total_frames=metadata_total_frames)
        if cache is not None:
            with job.timings.measure('depth_cache_finalize'):
                cache.finish()
        return target
    except (PipelineCancelled, DecoderCancelled) as exc:
        raise JobCancelled() from exc
    except (BrokenPipeError, OSError) as exc:
        stop_process(proc)
        reader.join(timeout=2)
        if job.cancel_event.is_set():
            raise JobCancelled() from exc
        detail = b"".join(stderr_lines).decode("utf-8", errors="replace").strip()
        if encoder_profile.codec == "h264_nvenc" and not _force_software:
            if cache is not None:
                cache.close()
            capture.release()
            target.unlink(missing_ok=True)
            set_job(job.id, process=None, encoder="CPU libx264 (NVENC fallback)",
                    progress=1, frames_processed=0, video_started_monotonic=None,
                    phase="retrying", retry_count=job.retry_count + 1, retry_reason="NVENC pipe stopped; restarting with libx264",
                    message="NVIDIA encoding stopped; retrying this export with the CPU encoder")
            return run_video(job, net, _force_software=True,
                             _cache_bypass=bool(cache and cache.bypass))
        raise RuntimeError((detail[-1200:] or str(exc)) + "\nCheck that this FFmpeg build includes libx264.") from exc
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error = None
        try:
            capture.release()
        except BaseException as exc:
            cleanup_error = exc
        try:
            stop_video_encoder(proc, reader)
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            active_error.video_cleanup_error = str(cleanup_error)


def process_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return
    if job.cancel_event.is_set():
        save_history_boundary(job, 'finish', 'canceled')
        set_job(job_id, state="canceled", phase="canceled", message="Canceled before processing",
                processing_finished_monotonic=time.monotonic())
        cleanup_job_file(job, job.input_path, 'upload')
        return

    target = output_path_for(job)
    target_owned = False
    # Finish an already admitted preview before reserving inference for export.
    # New preview misses see processing under the same gate and cannot jump ahead.
    with PREVIEW_CACHE.lock:
        save_history_boundary(job, 'mark_processing')
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
                                   job.resolution, OUTPUTS, FFMPEG_PATH, cancel_event=job.cancel_event)
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
                report = inspect_media(proxy, 'video', job.layout, 'source', OUTPUTS, FFMPEG_PATH,
                                       cancel_event=job.cancel_event)
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
            raise JobCancelled()
        if (target != output_path_for(job) or target.is_symlink()
                or target.parent.resolve() != OUTPUTS.resolve() or not target.is_file()
                or target.stat().st_size <= 0):
            raise RuntimeError('The converter did not produce a nonempty owned output.')
        save_history_boundary(job, 'finish', 'completed',
                              output={'name': target.name, 'size_bytes': target.stat().st_size},
                              execution=job.execution, encoder=job.encoder, timings=job.timings.snapshot())
        set_job(job_id, state="completed", phase="complete", progress=100, output_path=target, process=None,
                processing_finished_monotonic=time.monotonic(),
                message="Processed clip ready" if job.purpose == 'preview_clip' else "Stereo export ready")
    except (JobCancelled, ClipCancelled, DecoderCancelled):
        if target_owned:
            cleanup_job_file(job, target, 'partial output')
        save_history_boundary(job, 'finish', 'canceled')
        set_job(job_id, state="canceled", phase="canceled", process=None, processing_finished_monotonic=time.monotonic(), message="Conversion canceled")
    except Exception as exc:  # Surface local decoder/encoder/model errors in the queue.
        if target_owned:
            cleanup_job_file(job, target, 'partial output')
        # Bound persisted error text independently from the live diagnostic.
        detail = str(exc)
        cleanup_detail = getattr(exc, 'video_cleanup_error', None)
        if cleanup_detail:
            detail = f'{detail} (video cleanup also failed: {cleanup_detail})'
        error = re.sub(r'[\x00-\x1f\x7f]', ' ', detail)[:4096] or 'Conversion failed'
        save_history_boundary(job, 'finish', 'failed', error=error)
        set_job(job_id, state="failed", phase="failed", process=None, processing_finished_monotonic=time.monotonic(), error=str(exc), message="Conversion failed")
    finally:
        if job.preview_source_path:
            cleanup_job_file(job, job.preview_source_path, 'preview segment')
        cleanup_job_file(job, job.input_path, 'upload')
        with JOBS_LOCK:
            current = JOBS.get(job_id)
            if current:
                current.process = None


def worker_loop() -> None:
    while not WORKER_STOP.is_set():
        # Share Queue's condition so shutdown can wake an idle worker without
        # consuming or moving any queued export.
        with JOB_QUEUE.not_empty:
            while (not JOB_QUEUE.qsize() or JOB_QUEUE.paused) and not WORKER_STOP.is_set():
                JOB_QUEUE.not_empty.wait()
            if WORKER_STOP.is_set():
                return
            job_id = JOB_QUEUE.get_nowait()
        try:
            process_job(job_id)
        except Exception as exc:
            # A single unexpected job error must not strand the remaining queue.
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is not None:
                stop_process(job.process)
                if job.state not in {'completed', 'failed', 'canceled'}:
                    error = re.sub(r'[\x00-\x1f\x7f]', ' ', str(exc))[:4096] or 'Conversion failed'
                    save_history_boundary(job, 'finish', 'failed', error=error)
                    set_job(job_id, state='failed', phase='failed', error=str(exc),
                            processing_finished_monotonic=time.monotonic(), message='Conversion failed')
                if job.preview_source_path:
                    cleanup_job_file(job, job.preview_source_path, 'preview segment')
                cleanup_job_file(job, job.input_path, 'upload')
                set_job(job_id, process=None)
        finally:
            JOB_QUEUE.task_done()


def ensure_worker() -> None:
    global WORKER
    # Independent from queue/job locks: concurrent uploads must not create two
    # consumers while the first thread has been created but has not started yet.
    with WORKER_START_LOCK:
        if WORKER_STOP.is_set() or not ACCEPTING_MUTATIONS:
            return
        if WORKER is None or not WORKER.is_alive():
            WORKER = threading.Thread(target=worker_loop, daemon=True, name="stereolab-worker")
            WORKER.start()


def history_capabilities() -> dict[str, Any]:
    history_root = PATHS.history_root if FROZEN and sys.platform.startswith("linux") else DATA_ROOT
    result = (HISTORY or HistoryStore(history_root, outputs_root=OUTPUTS)).status()
    if HISTORY_WARNING and result['enabled']:
        result.update(state='degraded', warning=HISTORY_WARNING)
    return result


def history_failure(job: Job | None = None) -> None:
    global HISTORY_WARNING
    HISTORY_WARNING = 'History was not saved. Account for current outputs, check local storage, then restart to recover history.'
    if HISTORY is not None:
        HISTORY.mark_degraded()
    if job is not None:
        job.history_warning = HISTORY_WARNING


def save_history_boundary(job: Job, method: str, *args, required=False, **kwargs) -> None:
    if not job.history_admitted or HISTORY is None:
        return
    try:
        getattr(HISTORY, method)(job.id, *args, **kwargs)
    except (HistoryUnavailable, HistoryConflict, HistoryCapacityError, ValueError, OSError):
        history_failure(job)
        if required:
            raise HTTPException(status_code=503, detail=HISTORY_WARNING)


def require_history_admission() -> None:
    if HISTORY is None or not HISTORY.enabled:
        return
    try:
        HISTORY.check_capacity()
    except HistoryCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except HistoryUnavailable as exc:
        history_failure()
        raise HTTPException(status_code=503, detail=HISTORY_WARNING) from exc


@asynccontextmanager
async def application_lifespan(app):
    global HISTORY, HISTORY_WARNING, ACCEPTING_MUTATIONS, WORKER
    # Configuration and lease acquisition precede worker creation or serving.
    history_root = PATHS.history_root if FROZEN and sys.platform.startswith("linux") else DATA_ROOT
    store = HistoryStore.from_environment(history_root, outputs_root=OUTPUTS)
    store.open()
    HISTORY, HISTORY_WARNING = store, None
    ACCEPTING_MUTATIONS = True
    WORKER_STOP.clear()
    JOB_QUEUE.set_paused(False)  # Scheduling pause is local to this server launch.
    try:
        ensure_worker()
        yield
    finally:
        with WORKER_START_LOCK:
            ACCEPTING_MUTATIONS = False
            WORKER_STOP.set()
        with JOB_QUEUE.not_empty:
            JOB_QUEUE.not_empty.notify_all()
        with JOB_QUEUE.lock, JOBS_LOCK:
            active = [job for job in JOBS.values() if job.state == 'processing'
                      or (WORKER is not None and job.state == 'queued' and job.id not in JOB_QUEUE.pending_ids())]
            for job in active:
                job.cancel_event.set()
            processes = [job.process for job in active]
        for process in processes:
            await run_in_threadpool(stop_process, process)
        worker = WORKER
        if worker is not None:
            # No timeout may release the lease beneath a live writer.
            await run_in_threadpool(worker.join)
        store.close()
        HISTORY = None
        WORKER = None
        print('StereoLab shutdown complete.', flush=True)


app = FastAPI(title="StereoLab", docs_url=None, redoc_url=None, lifespan=application_lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

REQUEST_BODY_LIMITS = {
    "/api/preview": PREVIEW_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
    "/api/calibration": CALIBRATION_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
    "/api/model": MAX_MODEL_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
    "/api/jobs": MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES,
}


class RequestBodyTooLarge(OSError):
    """Raised while parsing a multipart upload so Starlette closes partial files."""


class RequestBodyLimitMiddleware:
    """Reject oversized upload bodies before FastAPI creates multipart files."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        limit = REQUEST_BODY_LIMITS.get(scope.get("path")) if scope.get("type") == "http" and scope.get("method") == "POST" else None
        if limit is None:
            await self.app(scope, receive, send)
            return

        async def reject() -> None:
            body = b'{"detail":"The complete upload request exceeds the allowed size."}'
            await send({"type": "http.response.start", "status": 413, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]})
            await send({"type": "http.response.body", "body": body, "more_body": False})

        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    if int(value) > limit:
                        await reject()
                        return
                except ValueError:
                    pass
                break

        received = 0
        exceeded = False
        response_replaced = False

        async def limited_receive():
            nonlocal received, exceeded
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    # MultiPartParser catches OSError, closes all staged UploadFiles,
                    # then re-raises it. FastAPI may translate it to a 400, which
                    # the send wrapper below replaces with the intended 413.
                    raise RequestBodyTooLarge()
            return message

        async def limited_send(message):
            nonlocal response_replaced
            if exceeded:
                if message.get("type") == "http.response.start" and not response_replaced:
                    response_replaced = True
                    await reject()
                return
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except RequestBodyTooLarge:
            if not response_replaced:
                await reject()


app.add_middleware(RequestBodyLimitMiddleware)
app.add_middleware(UploadAdmissionMiddleware, paths=frozenset(REQUEST_BODY_LIMITS), max_inflight=4)


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
    if request.method not in {'GET', 'HEAD', 'OPTIONS'} and not ACCEPTING_MUTATIONS:
        return JSONResponse(status_code=503, content={'detail': 'StereoLab is shutting down; new changes are no longer accepted.'})
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
        "history_capabilities": history_capabilities(),
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


def workspace_roots() -> dict[str, Path]:
    return {'uploads': UPLOADS, 'previews': CLIP_PROXIES, 'outputs': OUTPUTS,
            'models': MODELS}


def protected_workspace_paths() -> set[Path]:
    # Caller holds queue-before-jobs locks, including during deletion. Staged
    # uploads are registered before creation and stay protected until admission.
    protected = set(STAGED_UPLOADS)
    pending = set(JOB_QUEUE.pending_ids())
    for job in JOBS.values():
        if job.state in {'queued', 'processing'} or job.id in pending:
            protected.add(job.input_path)
            protected.add(CLIP_PROXIES)  # Proxy creation can precede publication.
            if job.preview_source_path:
                protected.add(job.preview_source_path)
    return protected


@app.get('/api/storage')
def get_storage() -> dict[str, Any]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        roots, protected = workspace_roots(), protected_workspace_paths()
    report = storage_inventory(roots, protected)
    report['scope'] = 'Managed uploads, preview segments, outputs and models. RAM preview depth is excluded. Persistent video depth is measured separately below.'
    return report


@app.get('/api/depth-cache')
def get_depth_cache() -> dict[str, Any]:
    try:
        return get_video_depth_store().inventory()
    except (CacheError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f'Depth cache could not be measured: {exc}') from exc


@app.post('/api/depth-cache/cleanup')
def cleanup_depth_cache(body: Any = Body(...)) -> dict[str, Any]:
    if (type(body) is not dict or set(body) != {'keys'} or type(body['keys']) is not list
            or len(body['keys']) > 1000 or any(type(key) is not str or
            re.fullmatch(r'[0-9a-f]{64}', key) is None for key in body['keys'])
            or len(set(body['keys'])) != len(body['keys'])):
        raise HTTPException(status_code=400, detail='Select up to 1000 distinct depth cache entries.')
    try:
        return get_video_depth_store().delete(body['keys'])
    except (CacheError, OSError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=f'Depth cache cleanup failed: {exc}') from exc


@app.post('/api/storage/cleanup')
def cleanup_storage(body: Any = Body(...)) -> dict[str, Any]:
    if (type(body) is not dict or set(body) != {'paths'} or type(body['paths']) is not list
            or len(body['paths']) > 1000 or any(type(path) is not str for path in body['paths'])):
        raise HTTPException(status_code=400, detail='Select up to 1000 temporary file paths, without extra fields.')
    with JOB_QUEUE.lock, JOBS_LOCK:
        result = cleanup_temporary_files(body['paths'], workspace_roots(), protected_workspace_paths())
        removed = set(result['deleted'])
        for job in JOBS.values():
            for path in tuple(job.cleanup_pending):
                if str(path.absolute()) in removed:
                    job.cleanup_pending.pop(path)
            if not job.cleanup_pending:
                job.cleanup_warning = None
        return result


@app.post('/api/jobs/{job_id}/cleanup')
def retry_job_cleanup(job_id: str) -> dict[str, Any]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail='Job not found.')
        if job.state not in {'completed', 'failed', 'canceled'} or job_id in JOB_QUEUE.pending_ids():
            raise HTTPException(status_code=409, detail='Wait for this job to finish before retrying cleanup.')
        for path, label in tuple(job.cleanup_pending.items()):
            # Only files already registered by this job's cleanup are retried.
            cleanup_job_file(job, path, label)
        return public_job_locked(job)


@app.post('/api/output-folder/open')
def open_output_folder() -> dict[str, bool]:
    try:
        open_folder(OUTPUTS)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {'ok': True}


def completed_output(job_id: str) -> Path:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        output = job.output_path if job and job.state == 'completed' else None
    if output is None and HISTORY is not None and HISTORY.enabled:
        try:
            output = HISTORY.lookup_output(job_id)
        except (ValueError, HistoryUnavailable):
            output = None
    if (output is None or output.is_symlink() or not output.is_file()
            or output.resolve().parent != OUTPUTS.resolve()):
        raise HTTPException(status_code=404, detail='Completed output was not found.')
    return output


@app.post('/api/jobs/{job_id}/reveal')
def reveal_output(job_id: str) -> dict[str, bool]:
    output = completed_output(job_id)
    try:
        reveal_file(output)
    except (OSError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {'ok': True}


@app.post('/api/jobs/{job_id}/copy')
def copy_output(job_id: str, body: Any = Body(...)) -> dict[str, str]:
    if (type(body) is not dict or set(body) != {'directory'}
            or type(body['directory']) is not str or not body['directory'].strip()):
        raise HTTPException(status_code=400, detail='Provide an existing destination directory.')
    source = completed_output(job_id)
    created = False
    destination = None
    try:
        directory = validate_output_directory(body['directory'])
        # Managed working folders cannot be copy destinations: cleanup and model
        # ownership must remain unambiguous.
        for root in (UPLOADS, CLIP_PROXIES, MODELS):
            if directory == root.resolve() or root.resolve() in directory.parents:
                raise ValueError('Choose a destination outside temporary and model folders.')
        destination = directory / source.name
        with source.open('rb') as reader, destination.open('xb') as writer:
            created = True
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
        return {'path': str(destination)}
    except (OSError, ValueError) as exc:
        if created and destination is not None:
            try:
                destination.unlink(missing_ok=True)
            except OSError:
                raise HTTPException(status_code=409, detail=f'Copy failed; a partial copy remains at {destination}.') from exc
        raise HTTPException(status_code=409, detail=f'Could not save copy: {exc}') from exc


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
    if execution_mode in {"gpu", "cuda", "hybrid"} and mode not in NEURAL_MODES:
        raise HTTPException(status_code=400, detail="NVIDIA GPU and GPU + CPU modes require a named neural model. Choose Auto or CPU for approximation or custom ONNX.")
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
        payload = await file.read(CALIBRATION_UPLOAD_BYTES + 1)
        if len(payload) > CALIBRATION_UPLOAD_BYTES:
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
    candidate_owned = False
    try:
        try:
            with candidate.open("xb") as target:
                candidate_owned = True
                while chunk := await model.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_MODEL_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="The model file is larger than 2 GB.")
                    target.write(chunk)
        finally:
            await model.close()
        try:
            loaded = await run_in_threadpool(cv2.dnn.readNetFromONNX, str(candidate))
            del loaded
        except cv2.error as exc:
            raise HTTPException(status_code=400, detail=f"OpenCV could not load this ONNX model: {exc}") from exc
        os.replace(candidate, MODEL_PATH)
        return {"ok": True, "model_available": True, "model_name": Path(model.filename or "depth.onnx").name, "bytes": total}
    finally:
        # A canceled copy, rejected model, or failed atomic publication must not
        # strand a large private candidate in the protected models directory.
        if candidate_owned:
            candidate.unlink(missing_ok=True)


def reserve_job_upload(input_path: Path) -> None:
    """Reserve capacity before copying media, including simultaneous admissions."""
    with JOB_QUEUE.lock, JOBS_LOCK:
        if not ACCEPTING_MUTATIONS:
            raise HTTPException(status_code=503, detail='StereoLab is shutting down.')
        if len(JOBS) + len(STAGED_UPLOADS) >= MAX_SESSION_JOBS:
            raise HTTPException(status_code=429, detail=(
                'This session has reached its job limit. Finish active work, then restart StereoLab. '
                'Saved outputs and durable history remain on disk.'))
        if (len(STAGED_UPLOADS) >= MAX_STAGED_UPLOADS
                or JOB_QUEUE.qsize() + len(STAGED_UPLOADS) >= MAX_PENDING_JOBS):
            raise HTTPException(status_code=429, detail=(
                'The local queue or upload slots are full. Let queued work finish, '
                'or remove pending jobs, then retry.'))
        STAGED_UPLOADS.add(input_path)


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
    output_name_template: Any = Form(DEFAULT_OUTPUT_NAME_TEMPLATE),
    expected_source_sha256: str | None = Form(None), request: Request = None,
    depth_interval: Annotated[int, Form()] = 1,
    video_decode: Annotated[str, Form()] = 'software',
    use_depth_cache: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    filename = Path((file.filename or "input").replace("\\", "/")).name
    if type(depth_interval) is not int or depth_interval not in {1, 2}:
        await file.close()
        raise HTTPException(status_code=400, detail="Choose depth on every frame or alternate frames (1 or 2).")
    if video_decode not in {'software', 'amd'}:
        await file.close()
        raise HTTPException(status_code=400, detail="Choose software or AMD video decoding.")
    if sys.platform.startswith('linux') and video_decode == 'amd':
        # D3D11VA is Windows-only. Normalize stale saved/API settings so the
        # job metadata matches the software decoder that Linux will actually use.
        video_decode = 'software'
    extension = Path(filename).suffix.lower()
    if purpose not in {'export', 'preview_clip'}:
        await file.close()
        raise HTTPException(status_code=400, detail='Choose export or preview_clip as job purpose.')
    if expected_source_sha256 is not None and (
            purpose != 'export' or re.fullmatch(r'[0-9a-f]{64}', expected_source_sha256) is None):
        await file.close()
        raise HTTPException(status_code=400, detail='Saved-project source identity must be a lowercase SHA-256 for an ordinary export.')
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
        raise HTTPException(status_code=409, detail="FFmpeg is needed for video export. " + FFMPEG_INSTALL_GUIDANCE)

    if purpose == 'export':
        try:
            require_history_admission()
        except HTTPException:
            await file.close()
            raise

    job_id = str(uuid.uuid4())
    input_path = UPLOADS / f"{job_id}{extension}"
    try:
        reserve_job_upload(input_path)
    except HTTPException:
        await file.close()
        raise
    try:
        total = 0
        digest = hashlib.sha256()
        staged_owned = False
        try:
            with input_path.open("xb") as target:
                staged_owned = True
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    digest.update(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        target.close()
                        input_path.unlink(missing_ok=True)
                        raise HTTPException(status_code=413, detail="The media file is larger than 8 GB.")
                    target.write(chunk)
            if expected_source_sha256 is not None and digest.hexdigest() != expected_source_sha256:
                raise HTTPException(status_code=409, detail='This source no longer matches the saved project. Locate the verified source again.')
        except BaseException:
            if staged_owned:
                try:
                    input_path.unlink(missing_ok=True)
                except OSError:
                    pass  # Retain an inactive cleanup candidate without masking cancellation.
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
            depth_interval=depth_interval,
            video_decode=video_decode,
            use_depth_cache=use_depth_cache,
            purpose=purpose,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            output_name_template=output_name_template,
        )
        try:
            with JOB_QUEUE.lock, JOBS_LOCK:
                if not ACCEPTING_MUTATIONS:
                    raise HTTPException(status_code=503, detail='StereoLab is shutting down.')
                if purpose == 'export' and HISTORY is not None and HISTORY.enabled:
                    HISTORY.admit(dict(job_id=job.id, created_at=job.created_at, filename=job.filename,
                                       media_type=job.media_type, source_size_bytes=total,
                                       source_sha256=digest.hexdigest(), settings=settings_snapshot(job),
                                       output_name_template=job.output_name_template,
                                       build_fingerprint=APP_BUILD_FINGERPRINT))
                    job.history_admitted = True
                JOBS[job_id] = job
                JOB_QUEUE.put_nowait(job_id)
        except HistoryCapacityError as exc:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except HistoryUnavailable as exc:
            input_path.unlink(missing_ok=True)
            history_failure()
            raise HTTPException(status_code=503, detail=HISTORY_WARNING) from exc
        except ValueError as exc:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            input_path.unlink(missing_ok=True)
            raise
        ensure_worker()
        with JOB_QUEUE.lock, JOBS_LOCK:
            return public_job_locked(job)

    finally:
        with JOBS_LOCK:
            STAGED_UPLOADS.discard(input_path)


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


@app.get('/api/queue/control')
def queue_control() -> dict[str, Any]:
    with JOB_QUEUE.lock, JOBS_LOCK:
        return {'paused': JOB_QUEUE.paused, 'pending_count': len(pending_jobs_locked()),
                'restart_behavior': 'running'}


@app.post('/api/queue/control')
def change_queue_control(body: Any = Body(...)) -> dict[str, Any]:
    if type(body) is not dict or set(body) != {'paused'} or type(body['paused']) is not bool:
        raise HTTPException(status_code=400, detail='Provide paused as a boolean without extra fields.')
    with JOB_QUEUE.lock:
        JOB_QUEUE.set_paused(body['paused'])
        return queue_control()


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
        save_history_boundary(job, 'edit_pending', settings=settings, required=True)
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
        save_history_boundary(job, 'edit_pending', output_name_template=template, required=True)
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
        # Cancellation remains available when durable storage has failed.
        save_history_boundary(job, 'finish', 'canceled')
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
        if JOB_QUEUE.paused and job_id in JOB_QUEUE.pending_ids():
            # Paused scheduling must not defer cancellation/cleanup until resume.
            JOB_QUEUE.remove_pending(job_id)
            save_history_boundary(job, 'finish', 'canceled')
            set_job(job_id, state='canceled', phase='canceled', message='Canceled while queue paused',
                    processing_finished_monotonic=time.monotonic())
            cleanup_job_file(job, job.input_path, 'upload')
            return public_job_locked(job)
        proc = job.process
    stop_process(proc)
    with JOB_QUEUE.lock, JOBS_LOCK:
        current = JOBS[job_id]
        if current.state not in {"completed", "failed", "canceled"}:
            current.message = "Canceling local conversion"
        return public_job_locked(current)


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        output = job.output_path if job else None
    if output is None or not output.is_file() or output.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail="Completed output was not found.")
    return FileResponse(output, filename=output.name)


@app.get('/api/history')
def get_history(limit: str = '50', before: str | None = None) -> dict[str, Any]:
    if HISTORY is None or not HISTORY.enabled:
        raise HTTPException(status_code=409, detail='History is disabled for this server instance.')
    try:
        if re.fullmatch(r'[0-9]{1,3}', limit) is None:
            raise ValueError('History page limit must be an integer from 1 to 100.')
        return HISTORY.list_prior(limit=int(limit), before=before)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HistoryUnavailable as exc:
        history_failure()
        raise HTTPException(status_code=503, detail=HISTORY_WARNING) from exc


@app.get('/api/history/{job_id}/download')
def download_history(job_id: str) -> FileResponse:
    if HISTORY is None or not HISTORY.enabled:
        raise HTTPException(status_code=409, detail='History is disabled for this server instance.')
    try:
        output = HISTORY.lookup_output(job_id)
    except ValueError:
        output = None
    except HistoryUnavailable as exc:
        history_failure()
        raise HTTPException(status_code=503, detail=HISTORY_WARNING) from exc
    if output is None or output.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail='Saved output is unavailable.')
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
    from app_launcher import run_app
    run_app(app)
