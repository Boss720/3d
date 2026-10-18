from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from depth_engine import ModelManager
from stereo import TemporalDepthFilter, layout_size, make_eye_pair, pack_output, refine_depth
from video_pipeline import PipelineCancelled, run_pipeline


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNTIME = ROOT / ".runtime"
UPLOADS = RUNTIME / "uploads"
MODELS = RUNTIME / "models"
OUTPUTS = ROOT / "outputs"
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

for directory in (UPLOADS, MODELS, OUTPUTS):
    directory.mkdir(parents=True, exist_ok=True)

DEPTH_ENGINE = ModelManager(MODELS)
EXECUTION_LOCAL = threading.local()


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
JOBS: dict[str, "Job"] = {}
JOBS_LOCK = threading.RLock()
JOB_QUEUE: queue.Queue[str] = queue.Queue()
WORKER: threading.Thread | None = None


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
    pipeline: str = "sequential"
    created_at: float = field(default_factory=time.time)
    state: str = "queued"
    progress: int = 0
    message: str = "Waiting in the local queue"
    output_path: Path | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen[bytes] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filename": self.filename,
            "media_type": self.media_type,
            "layout": self.layout,
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
            "pipeline": self.pipeline,
        }


class JobCancelled(Exception):
    pass


def safe_stem(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name
    stem = Path(name).stem or "stereo-output"
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._")
    return (cleaned or "stereo-output")[:100]


def output_path_for(job: Job) -> Path:
    suffix = "png" if job.media_type == "image" else "mp4"
    layout_name = job.layout.lower()
    return OUTPUTS / f"{safe_stem(job.filename)}_{layout_name}_{job.id[:8]}.{suffix}"


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
    image = cv2.imread(str(job.input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("OpenCV could not read this image. Try a standard JPG, PNG, or WebP file.")
    if job.resolution == "1080p":
        scale = min(1.0, 1920 / image.shape[1], 1080 / image.shape[0])
        if scale < 1:
            image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    set_job(job.id, progress=15, message="Estimating relative depth")
    depth = estimate_depth(image, job.depth_mode, net, job.input_size, job.invert_depth, job.execution_mode)
    set_job(job.id, execution=execution_snapshot(job.depth_mode, job.execution_mode))
    depth = refine_depth(image, depth, job.edge_refinement)
    if job.cancel_event.is_set():
        raise JobCancelled()
    left, right = make_eye_pair(image, depth, job.strength, job.convergence)
    result = pack_output(image, depth, left, right, job.layout)
    target = output_path_for(job)
    if not cv2.imwrite(str(target), result, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise RuntimeError("Could not write the PNG output. Check that the output drive has free space.")
    return target


def run_video(job: Job, net: cv2.dnn_Net | None) -> Path:
    ffmpeg = FFMPEG_PATH
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("FFmpeg was not found. Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe.")
    capture = cv2.VideoCapture(str(job.input_path))
    if not capture.isOpened():
        raise RuntimeError("Could not open this video. The file may use an unsupported codec or be damaged.")

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if source_width < 1 or source_height < 1:
        capture.release()
        raise RuntimeError("The video does not report a valid frame size.")
    fps = ffprobe_fps(ffmpeg, job.input_path, source_fps if source_fps >= 1 else 30.0)

    scale = min(1.0, 1920.0 / source_width, 1080.0 / source_height) if job.resolution == "1080p" else 1.0
    width = max(2, int(round(source_width * scale / 2.0) * 2))
    height = max(2, int(round(source_height * scale / 2.0) * 2))
    if job.layout == "HALFSBS":
        width = max(4, width // 4 * 4)
    if job.layout == "HALFTB":
        height = max(4, height // 4 * 4)
    output_width, output_height = layout_size(width, height, job.layout)
    target = output_path_for(job)
    quality = {
        "fast": ("ultrafast", "24"),
        "balanced": ("medium", "20"),
        "high": ("slow", "17"),
    }.get(job.quality, ("medium", "20"))

    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", f"{output_width}x{output_height}",
        "-framerate", f"{fps:.8f}", "-i", "pipe:0",
        "-i", str(job.input_path),
        "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "libx264", "-preset", quality[0], "-crf", quality[1],
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", str(target),
    ]

    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    set_job(job.id, process=proc, message="Converting frames locally")
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
        depth = estimate_depth(frame, job.depth_mode, net, job.input_size, job.invert_depth, job.execution_mode)
        set_job(job.id, execution=execution_snapshot(job.depth_mode, job.execution_mode))
        return depth

    def consume_frame(frame, depth):
        nonlocal frame_index
        if job.cancel_event.is_set():
            raise JobCancelled()
        depth = refine_depth(frame, depth, job.edge_refinement)
        depth = depth_filter.apply(frame, depth)
        left, right = make_eye_pair(frame, depth, job.strength, job.convergence)
        output_frame = pack_output(frame, depth, left, right, job.layout)
        if job.cancel_event.is_set():
            raise JobCancelled()
        if proc.stdin is None:
            raise RuntimeError("FFmpeg did not open its frame input.")
        proc.stdin.write(output_frame.tobytes())
        frame_index += 1
        if frame_index == 1 or frame_index % max(1, round(fps)) == 0:
            progress = min(98, int(frame_index * 100 / total_frames)) if total_frames else min(98, frame_index // 10)
            set_job(job.id, progress=progress, message=f"Frame {frame_index:,}" + (f" of {total_frames:,}" if total_frames else ""))

    try:
        frames = source_frames()
        if job.execution_mode in {"auto", "hybrid"} and job.depth_mode in NEURAL_MODES:
            set_job(job.id, pipeline="overlapped")
            run_pipeline(frames, infer_frame, consume_frame, job.cancel_event, queue_size=2)
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
        if proc.stdin:
            proc.stdin.close()
        return_code = proc.wait()
        reader.join(timeout=2)
        if job.cancel_event.is_set():
            raise JobCancelled()
        if return_code != 0:
            detail = b"".join(stderr_lines).decode("utf-8", errors="replace").strip()
            detail = detail[-1200:] or f"FFmpeg exited with code {return_code}."
            raise RuntimeError(detail)
        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError("FFmpeg finished without producing an output file.")
        return target
    except PipelineCancelled as exc:
        raise JobCancelled() from exc
    except (BrokenPipeError, OSError) as exc:
        stop_process(proc)
        reader.join(timeout=2)
        if job.cancel_event.is_set():
            raise JobCancelled() from exc
        detail = b"".join(stderr_lines).decode("utf-8", errors="replace").strip()
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
    set_job(job_id, state="processing", progress=1, message="Preparing local depth engine")
    try:
        if job.depth_mode in NEURAL_MODES:
            DEPTH_ENGINE.validate(job.depth_mode, execution_mode=job.execution_mode)
        net = load_depth_model() if job.depth_mode == "precision" else None
        target = run_image(job, net) if job.media_type == "image" else run_video(job, net)
        if job.cancel_event.is_set():
            target.unlink(missing_ok=True)
            raise JobCancelled()
        set_job(job_id, state="completed", progress=100, output_path=target, process=None, message="Stereo export ready")
    except JobCancelled:
        target.unlink(missing_ok=True)
        set_job(job_id, state="canceled", process=None, message="Conversion canceled")
    except Exception as exc:  # Surface local decoder/encoder/model errors in the queue.
        target.unlink(missing_ok=True)
        set_job(job_id, state="failed", process=None, error=str(exc), message="Conversion failed")
    finally:
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
    if WORKER is None or not WORKER.is_alive():
        WORKER = threading.Thread(target=worker_loop, daemon=True, name="stereolab-worker")
        WORKER.start()


app = FastAPI(title="StereoLab", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


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
    }


@app.get("/api/models")
def get_models() -> dict[str, Any]:
    return DEPTH_ENGINE.catalog()


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


def process_preview(payload: bytes, mode: str, size: int, layout: str, strength: float, convergence: float, invert: bool, refinement: bool, execution_mode: str = "auto") -> dict[str, Any]:
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Preview needs a decoded image frame. Try a JPG, PNG or a browser-playable video.")
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 540 / height)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    net = load_depth_model() if mode == "precision" else None
    depth = estimate_depth(frame, mode, net, size, invert, execution_mode)
    execution = execution_snapshot(mode, execution_mode)
    depth = refine_depth(frame, depth, refinement)
    left, right = make_eye_pair(frame, depth, strength, convergence)
    output = pack_output(frame, depth, left, right, layout)
    depth_image = pack_output(frame, depth, left, right, "DEPTH")
    return {"image": png_data_url(output), "depth": png_data_url(depth_image), "model": mode,
            "width": output.shape[1], "height": output.shape[0], "execution": execution}


@app.post("/api/preview")
async def preview(
    file: UploadFile = File(...), depth_mode: str = Form("balanced"), input_size: int = Form(518),
    layout: str = Form("SBS"), strength: float = Form(7.5), convergence: float = Form(0.5),
    invert_depth: bool = Form(False), edge_refinement: bool = Form(True),
    execution_mode: str = Form("auto"),
) -> dict[str, Any]:
    try:
        validate_depth_mode(depth_mode, execution_mode)
        if layout not in LAYOUTS or not 0 <= strength <= 10 or not 0 <= convergence <= 1:
            raise HTTPException(status_code=400, detail="Invalid preview format, strength or convergence.")
        if input_size not in {256, 384, 518}:
            raise HTTPException(status_code=400, detail="Choose a supported model input size.")
        payload = await file.read(24 * 1024 * 1024 + 1)
        if len(payload) > 24 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="The preview frame exceeds 24 MB.")
        return await run_in_threadpool(process_preview, payload, depth_mode, input_size, layout, strength, convergence, invert_depth, edge_refinement, execution_mode)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preview failed: {exc}") from exc
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
) -> dict[str, Any]:
    filename = Path((file.filename or "input").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
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
    )
    with JOBS_LOCK:
        JOBS[job_id] = job
    ensure_worker()
    JOB_QUEUE.put(job_id)
    return job.public()


@app.get("/api/jobs")
def get_jobs() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        return [job.public() for job in sorted(JOBS.values(), key=lambda item: item.created_at)]


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job.state in {"completed", "failed", "canceled"}:
            return job.public()
        job.cancel_event.set()
        proc = job.process
    stop_process(proc)
    set_job(job_id, message="Canceling local conversion")
    with JOBS_LOCK:
        current = JOBS[job_id]
        return current.public()


@app.get("/api/jobs/{job_id}/download")
def download_job(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        output = job.output_path if job else None
    if output is None or not output.is_file() or output.parent.resolve() != OUTPUTS.resolve():
        raise HTTPException(status_code=404, detail="Completed output was not found.")
    return FileResponse(output, filename=output.name)


if __name__ == "__main__":
    print("StereoLab is running at http://127.0.0.1:8765")
    print("All processing stays on this computer. Press Ctrl+C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
