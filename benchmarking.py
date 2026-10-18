"""Reproducible local benchmark support for StereoLab roadmap 0.1.

Reports contain source basenames and SHA-256 identities by default. Synthetic
fixtures are useful for repeatable pipeline checks, not visual-quality truth.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import ipaddress
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np
from resource_sampling import DeviceMemorySampler


SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
SYNTHETIC_GENERATOR_VERSION = 1
DEFAULT_API_URL = "http://127.0.0.1:8766"
DEFAULT_MODEL = "balanced"
DEFAULT_DEVICE = "cpu"
DEFAULT_REPEATS = 3
VALID_LAYOUTS = {"SBS", "HALFSBS", "TB", "HALFTB", "CROSS", "ANAGLYPH", "DEPTH", "RGBD"}
VALID_DEVICES = {"auto", "cpu", "gpu", "hybrid"}
VALID_QUALITIES = {"fast", "balanced", "high"}
SYNTHETIC_IMAGE_PATTERNS = {"fine_detail", "thin_objects", "portrait"}
SYNTHETIC_VIDEO_PATTERNS = {"motion", "hard_cut", "portrait_motion"}

JOB_TIMING_FIELDS = (
    "elapsed_seconds",
    "processing_fps",
    "eta_seconds",
    "eta_range_seconds",
    "frames_processed",
    "total_frames",
    "initialization_seconds",
    "conversion_seconds",
    "finalization_seconds",
    "queue_wait_seconds",
    "decode_seconds",
    "inference_seconds",
    "render_seconds",
    "encode_seconds",
    "total_seconds",
    "stage_timings",
)
EXECUTION_FIELDS = (
    "requested_mode",
    "active_provider",
    "device",
    "device_id",
    "fallback_reason",
    "gpu_verified",
    "requested_input_width",
    "requested_input_height",
    "input_width",
    "input_height",
    "input_size_limit",
    "session_reused",
    "session_initialization_seconds",
    "inference_pipeline_seconds",
    "error",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using stable key ordering."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def fixture_identity_sha256(fixture_id: str, source_kind: str, source: dict[str, Any], source_sha256: str) -> str:
    """Fingerprint a generated recipe independently from its container encoder.

    The exact bytes sent to the service are still recorded separately as
    ``source_sha256``. User-provided files use that content hash as identity.
    """
    if source_kind in {"synthetic_image", "synthetic_video"}:
        recipe = {key: value for key, value in source.items() if key != "path"}
        identity = {
            "fixture_id": fixture_id,
            "source_kind": source_kind,
            "generator": f"stereolab-synthetic-v{SYNTHETIC_GENERATOR_VERSION}",
            "recipe": recipe,
        }
    else:
        identity = {"fixture_id": fixture_id, "source_kind": source_kind, "source_sha256": source_sha256}
    return stable_sha256(identity)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(value: Any) -> list[dict[str, Any]]:
    """Validate a v1 manifest and return its fixture entries.

    Path sources are intentionally not resolved here; callers resolve them
    relative to the manifest file and omit their full path from reports.
    """
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"Manifest schema_version must be {MANIFEST_SCHEMA_VERSION}.")
    fixtures = value.get("fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("Manifest fixtures must be a non-empty list.")

    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, fixture in enumerate(fixtures):
        prefix = f"fixtures[{index}]"
        if not isinstance(fixture, dict):
            raise ValueError(f"{prefix} must be an object.")
        fixture_id = fixture.get("id")
        if not isinstance(fixture_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", fixture_id):
            raise ValueError(f"{prefix}.id must use 1-80 letters, numbers, dots, underscores or hyphens.")
        if fixture_id in seen:
            raise ValueError(f"Fixture ids must be unique; duplicate id: {fixture_id}.")
        seen.add(fixture_id)

        source = fixture.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{prefix}.source must be an object.")
        kind = source.get("kind")
        if kind == "path":
            path = source.get("path")
            media_type = source.get("media_type")
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{prefix}.source.path must be a non-empty path string.")
            if media_type not in {"image", "video"}:
                raise ValueError(f"{prefix}.source.media_type must be image or video.")
            audio = source.get("audio")
            if media_type == "image" and audio not in (None, False):
                raise ValueError(f"{prefix}.source.audio only applies to video sources.")
            if media_type == "video" and audio is not None and not isinstance(audio, bool):
                raise ValueError(f"{prefix}.source.audio must be true, false or null.")
        elif kind in {"synthetic_image", "synthetic_video"}:
            media_type = "image" if kind == "synthetic_image" else "video"
            width = source.get("width")
            height = source.get("height")
            for name, dimension in (("width", width), ("height", height)):
                if isinstance(dimension, bool) or not isinstance(dimension, int) or not 2 <= dimension <= 4096:
                    raise ValueError(f"{prefix}.source.{name} must be an integer from 2 to 4096.")
            if kind == "synthetic_image":
                pattern = source.get("pattern", "fine_detail")
                if pattern not in SYNTHETIC_IMAGE_PATTERNS:
                    raise ValueError(f"{prefix}.source.pattern is not a supported synthetic image pattern.")
            else:
                pattern = source.get("pattern", "motion")
                if pattern not in SYNTHETIC_VIDEO_PATTERNS:
                    raise ValueError(f"{prefix}.source.pattern is not a supported synthetic video pattern.")
                frames = source.get("frames")
                fps = source.get("fps")
                if isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= 10_000:
                    raise ValueError(f"{prefix}.source.frames must be an integer from 2 to 10000.")
                if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or not 0.1 <= float(fps) <= 120:
                    raise ValueError(f"{prefix}.source.fps must be between 0.1 and 120.")
                audio = source.get("audio", False)
                if not isinstance(audio, bool):
                    raise ValueError(f"{prefix}.source.audio must be a boolean.")
                frequency = source.get("audio_frequency_hz", 440)
                if isinstance(frequency, bool) or not isinstance(frequency, (int, float)) or not 20 <= float(frequency) <= 20_000:
                    raise ValueError(f"{prefix}.source.audio_frequency_hz must be between 20 and 20000.")
            seed = source.get("seed", 0)
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"{prefix}.source.seed must be an integer.")
        else:
            raise ValueError(f"{prefix}.source.kind must be path, synthetic_image or synthetic_video.")

        normalized.append({"id": fixture_id, "source": dict(source)})
    return normalized


def expected_output_dimensions(width: int, height: int, layout: str) -> tuple[int, int]:
    """Compute packed dimensions using the app's still/video layout contract."""
    if layout not in VALID_LAYOUTS:
        raise ValueError(f"Unsupported output layout: {layout}.")
    if width < 1 or height < 1:
        raise ValueError("Input dimensions must be positive.")
    if layout in {"SBS", "CROSS", "RGBD"}:
        return width * 2, height
    if layout == "TB":
        return width, height * 2
    if layout == "HALFSBS":
        return max(1, width // 2) * 2, height
    if layout == "HALFTB":
        return width, max(1, height // 2) * 2
    return width, height


def expected_video_dimensions(width: int, height: int, layout: str) -> tuple[int, int]:
    """Apply the service's even-size and half-layout encoder constraints."""
    source_width = max(2, int(round(width / 2.0) * 2))
    source_height = max(2, int(round(height / 2.0) * 2))
    if layout == "HALFSBS":
        source_width = max(4, source_width // 4 * 4)
    if layout == "HALFTB":
        source_height = max(4, source_height // 4 * 4)
    return expected_output_dimensions(source_width, source_height, layout)


def verify_image_output(payload: bytes, expected: dict[str, Any]) -> dict[str, Any]:
    decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim < 2:
        return {
            "status": "failed",
            "actual_dimensions": None,
            "checks": {"decode": "failed", "dimensions": "unavailable"},
        }
    height, width = decoded.shape[:2]
    expected_width = expected.get("width")
    expected_height = expected.get("height")
    dimensions_ok = (
        (expected_width is None or width == expected_width)
        and (expected_height is None or height == expected_height)
    )
    return {
        "status": "passed" if dimensions_ok else "failed",
        "actual_dimensions": {"width": width, "height": height},
        "checks": {"decode": "passed", "dimensions": "passed" if dimensions_ok else "failed"},
    }


def verify_video_observation(observation: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    """Compare decoded output observations with known source/layout expectations."""
    checks: dict[str, str] = {}
    required = ("width", "height", "decoded_frames")
    for field in required:
        checks[field] = "unavailable" if observation.get(field) is None else "passed"
    if observation.get("decoded_frames") == 0:
        checks["decode"] = "failed"
    else:
        checks["decode"] = "passed" if observation.get("decoded_frames") is not None else "unavailable"
    if observation.get("consistent_dimensions") is False:
        checks["consistent_dimensions"] = "failed"
    elif observation.get("consistent_dimensions") is None:
        checks["consistent_dimensions"] = "unavailable"
    else:
        checks["consistent_dimensions"] = "passed"

    for field, observed_field in (("width", "width"), ("height", "height"), ("frames", "decoded_frames")):
        target = expected.get(field)
        actual = observation.get(observed_field)
        if target is not None:
            checks[field] = "unavailable" if actual is None else ("passed" if actual == target else "failed")

    for field in ("fps", "duration_seconds"):
        target = expected.get(field)
        actual = observation.get(field)
        if target is None:
            checks[field] = "unavailable"
        elif actual is None:
            checks[field] = "unavailable"
        else:
            tolerance = max(0.01, abs(float(target)) * 0.01) if field == "fps" else max(0.15, 1.0 / max(float(expected.get("fps") or 1), 1))
            checks[field] = "passed" if abs(float(actual) - float(target)) <= tolerance else "failed"

    expected_audio = expected.get("audio_present")
    actual_audio = observation.get("audio_present")
    if expected_audio is None:
        checks["audio"] = "unavailable"
    elif actual_audio is None:
        checks["audio"] = "unavailable"
    else:
        checks["audio"] = "passed" if bool(actual_audio) is bool(expected_audio) else "failed"

    values = list(checks.values())
    status = "failed" if "failed" in values else "partial" if "unavailable" in values else "passed"
    return {
        "status": status,
        "actual": {
            "width": observation.get("width"),
            "height": observation.get("height"),
            "decoded_frames": observation.get("decoded_frames"),
            "fps": observation.get("fps"),
            "duration_seconds": observation.get("duration_seconds"),
            "audio_present": observation.get("audio_present"),
        },
        "expected": {
            "width": expected.get("width"),
            "height": expected.get("height"),
            "frames": expected.get("frames"),
            "fps": expected.get("fps"),
            "duration_seconds": expected.get("duration_seconds"),
            "audio_present": expected_audio,
        },
        "checks": checks,
    }


def job_timing_snapshot(job: dict[str, Any]) -> dict[str, Any]:
    """Keep a stable timing shape and explicit nulls when API fields are absent."""
    snapshot = {field: job.get(field) for field in JOB_TIMING_FIELDS}
    nested = job.get("timings")
    if snapshot["stage_timings"] is None:
        snapshot["stage_timings"] = nested if isinstance(nested, dict) else None
    return snapshot


def normalize_execution_snapshot(value: Any) -> dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    return {
        key: (_safe_error_text(source.get(key)) if isinstance(source.get(key), str) else source.get(key))
        for key in EXECUTION_FIELDS
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "range": [min(values), max(values)],
    }


def _comparison_identity(run: dict[str, Any]) -> str | None:
    if not run.get("success") or (run.get("verification") or {}).get("status") != "passed":
        return None
    wall = (run.get("runner_timings") or {}).get("total_wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(float(wall)) or float(wall) < 0:
        return None
    execution = run.get("execution") or {}
    runtime_fingerprint = (run.get("environment") or {}).get("runtime_fingerprint")
    provider = execution.get("active_provider")
    device = execution.get("device")
    if not provider or not device or not runtime_fingerprint:
        return None
    identity = {
        "fixture_identity_sha256": run.get("fixture_identity_sha256"),
        "source_sha256": run.get("source_sha256"),
        "settings": run.get("settings"),
        "model_hash": run.get("model_hash"),
        "model_id": run.get('model_id'),
        "actual_model_dimensions": [execution.get('input_width'), execution.get('input_height')],
        "input_size_limit": execution.get('input_size_limit'),
        "encoder": run.get('encoder'),
        "pipeline": run.get('pipeline'),
        "provider": provider,
        "device": device,
        "device_id": execution.get("device_id"),
        "runtime_fingerprint": runtime_fingerprint,
    }
    if not identity["fixture_identity_sha256"] or not identity["source_sha256"] or not identity["settings"]:
        return None
    return _canonical_json(identity)


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize only repeated, successful runs with an identical comparison identity."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        identity = _comparison_identity(run)
        if identity is not None:
            groups.setdefault(identity, []).append(run)

    summaries = []
    for identity, group in groups.items():
        if len(group) < 2:
            continue
        wall_values = [float(run["runner_timings"]["total_wall_seconds"]) for run in group]
        representative = group[0]
        summary: dict[str, Any] = {
            "comparison_id": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
            "fixture_id": representative.get("fixture_id"),
            "successful_runs": len(group),
            "settings": representative.get("settings"),
            "model_hash": representative.get("model_hash"),
            "execution": representative.get("execution"),
            "wall_seconds": _distribution(wall_values),
        }
        job_elapsed = [
            float(run["job_timings"]["elapsed_seconds"])
            for run in group
            if isinstance((run.get("job_timings") or {}).get("elapsed_seconds"), (int, float))
            and not isinstance((run.get("job_timings") or {}).get("elapsed_seconds"), bool)
        ]
        if len(job_elapsed) >= 2:
            summary["job_elapsed_seconds"] = _distribution(job_elapsed)
        summaries.append(summary)
    return sorted(summaries, key=lambda item: (str(item.get("fixture_id")), str(item.get("comparison_id"))))


@dataclass
class PreparedFixture:
    fixture_id: str
    media_type: str
    source_name: str
    path: Path | None
    payload: bytes | None
    source_sha256: str
    fixture_identity_sha256: str
    source_info: dict[str, Any]
    audio_expected: bool | None
    source_kind: str


def _source_name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1] or "source"


def _gradient_frame(width: int, height: int, seed: int) -> np.ndarray:
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    noise = np.random.default_rng(seed).integers(0, 12, (height, width), dtype=np.uint8).astype(np.uint16)
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = ((x * 2 + y + noise) % 256).astype(np.uint8)
    frame[:, :, 1] = ((y * 3 + (x // 2) + noise * 2) % 256).astype(np.uint8)
    frame[:, :, 2] = ((x + y * 2 + 70 + noise) % 256).astype(np.uint8)
    return frame


def _fixture_frame(width: int, height: int, pattern: str, index: int, total: int, seed: int) -> np.ndarray:
    frame = _gradient_frame(width, height, seed)
    if pattern == "hard_cut" and index >= total // 2:
        frame = np.ascontiguousarray(255 - frame)
        cv2.rectangle(frame, (8, 8), (width - 9, height - 9), (20, 230, 35), 2)
    elif pattern in {"motion", "portrait_motion"}:
        shift = index * max(1, width // max(total, 2))
        frame = np.roll(frame, shift, axis=1)
        radius = max(3, min(width, height) // 10)
        center = (int((index * max(1, width - 2 * radius)) / max(total - 1, 1)) + radius,
                  int(height * 0.45 + math.sin(index * math.pi / max(total, 1)) * height * 0.12))
        cv2.circle(frame, center, radius, (245, 235, 45), -1, cv2.LINE_AA)
        cv2.circle(frame, center, max(1, radius // 3), (15, 20, 210), -1, cv2.LINE_AA)
    elif pattern == "portrait":
        center = (width // 2, int(height * 0.4))
        cv2.ellipse(frame, center, (max(5, width // 4), max(8, height // 7)), 0, 0, 360, (210, 185, 140), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (width // 3, int(height * 0.55)), (2 * width // 3, height - 1), (45, 80, 170), -1)
        eye_y = max(1, center[1] - height // 35)
        cv2.circle(frame, (width * 2 // 5, eye_y), max(1, width // 40), (15, 15, 15), -1)
        cv2.circle(frame, (width * 3 // 5, eye_y), max(1, width // 40), (15, 15, 15), -1)
    elif pattern == "thin_objects":
        for offset in range(0, width, max(4, width // 18)):
            cv2.line(frame, (offset, 0), (width - offset // 2, height - 1), (245, 245, 245), 1, cv2.LINE_AA)
    else:  # fine_detail
        for row in range(0, height, max(4, height // 20)):
            cv2.line(frame, (0, row), (width - 1, row), (245, 245, 245), 1, cv2.LINE_AA)
        for column in range(0, width, max(4, width // 24)):
            cv2.line(frame, (column, 0), (column, height - 1), (15, 15, 15), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (width // 3, height // 4), (2 * width // 3, 3 * height // 4), (240, 180, 35), 1)
    return np.ascontiguousarray(frame)


def _find_ffprobe(ffmpeg_path: str | None) -> str | None:
    if ffmpeg_path:
        candidate = Path(ffmpeg_path).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
        if candidate.is_file():
            return str(candidate)
    return shutil.which("ffprobe")


def _probe_video(path: Path, ffprobe_path: str | None) -> dict[str, Any] | None:
    if not ffprobe_path:
        return None
    try:
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-count_frames", "-show_entries",
             "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames:format=duration",
             "-of", "json", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        payload = json.loads(result.stdout or "{}")
        streams = payload.get("streams", [])
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        if video is None:
            return None

        def parse_rate(value: Any) -> float | None:
            if not isinstance(value, str):
                return None
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                denominator_value = float(denominator)
                return float(numerator) / denominator_value if denominator_value else None
            try:
                return float(value)
            except ValueError:
                return None

        format_duration = (payload.get("format") or {}).get("duration")
        return {
            "fps": parse_rate(video.get("avg_frame_rate")) or parse_rate(video.get("r_frame_rate")),
            "duration_seconds": float(format_duration) if format_duration not in (None, "N/A") else None,
            "audio_present": any(stream.get("codec_type") == "audio" for stream in streams),
            "reported_frames": _coerce_int(video.get("nb_read_frames")) or _coerce_int(video.get("nb_frames")),
        }
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


def _inspect_image(payload: bytes) -> dict[str, Any]:
    image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image source could not be decoded.")
    height, width = image.shape[:2]
    return {"media_type": "image", "width": width, "height": height,
            "frames": None, "fps": None, "duration_seconds": None, "audio_present": False}


def _inspect_image_file(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Image source could not be decoded.")
    height, width = image.shape[:2]
    return {"media_type": "image", "width": width, "height": height,
            "frames": None, "fps": None, "duration_seconds": None, "audio_present": False}


def _inspect_video(path: Path, ffprobe_path: str | None, declared_audio: bool | None = None) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Video source could not be opened by the local decoder.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()
    if width < 1 or height < 1:
        raise ValueError("Video source has invalid dimensions.")
    probe = _probe_video(path, ffprobe_path)
    if probe and probe.get("fps"):
        fps = float(probe["fps"])
    if probe and probe.get("reported_frames"):
        frames = int(probe["reported_frames"])
    audio_present = probe.get("audio_present") if probe else None
    if declared_audio is not None:
        if audio_present is not None and bool(audio_present) != declared_audio:
            raise ValueError("Video source audio does not match the manifest expectation.")
        audio_present = declared_audio
    duration = probe.get("duration_seconds") if probe else None
    if duration is None and frames > 0 and fps > 0:
        duration = frames / fps
    return {
        "media_type": "video",
        "width": width,
        "height": height,
        "frames": frames or None,
        "fps": fps if fps > 0 else None,
        "duration_seconds": duration,
        "duration_source": "ffprobe" if probe and probe.get("duration_seconds") is not None else "frames_divided_by_fps" if duration is not None else None,
        "audio_present": audio_present,
    }


def _write_synthetic_video(source: dict[str, Any], target: Path) -> None:
    width, height = int(source["width"]), int(source["height"])
    fps, frames = float(source["fps"]), int(source["frames"])
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV could not initialize the synthetic video encoder.")
    try:
        for index in range(frames):
            writer.write(_fixture_frame(width, height, source.get("pattern", "motion"), index, frames,
                                        int(source.get("seed", 0))))
    finally:
        writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("Synthetic video generation produced no file.")


def prepare_fixture(
    fixture: dict[str, Any],
    manifest_directory: Path,
    work_directory: Path,
    ffmpeg_path: str | None,
) -> PreparedFixture:
    source = fixture["source"]
    fixture_id = fixture["id"]
    kind = source["kind"]
    audio_expected: bool | None = source.get("audio") if kind in {"synthetic_video", "path"} else None
    path: Path | None = None
    payload: bytes | None = None
    media_type = "image" if kind == "synthetic_image" else source.get("media_type", "video")

    if kind == "path":
        raw_path = Path(source["path"])
        path = raw_path if raw_path.is_absolute() else manifest_directory / raw_path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"Source file is missing: {_source_name(source['path'])}.")
        source_name = path.name
    elif kind == "synthetic_image":
        frame = _fixture_frame(int(source["width"]), int(source["height"]), source.get("pattern", "fine_detail"),
                               0, 1, int(source.get("seed", 0)))
        success, encoded = cv2.imencode(".png", frame)
        if not success:
            raise RuntimeError("OpenCV could not encode the synthetic image.")
        payload = encoded.tobytes()
        source_name = f"{fixture_id}.png"
    else:
        base_path = work_directory / f"{fixture_id}-source.mp4"
        _write_synthetic_video(source, base_path)
        path = base_path
        if audio_expected:
            if not ffmpeg_path:
                raise RuntimeError("Audio fixture skipped because the local FFmpeg executable is unavailable.")
            muxed_path = work_directory / f"{fixture_id}-with-audio.mp4"
            duration = int(source["frames"]) / float(source["fps"])
            command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", str(base_path),
                       "-f", "lavfi", "-i",
                       f"sine=frequency={float(source.get('audio_frequency_hz', 440))}:sample_rate=44100:duration={duration:.6f}",
                       "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
                       "-shortest", str(muxed_path)]
            result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
            if result.returncode != 0 or not muxed_path.is_file() or muxed_path.stat().st_size == 0:
                raise RuntimeError("FFmpeg could not add the synthetic audio timing track.")
            path = muxed_path
        source_name = f"{fixture_id}.mp4"

    if media_type == "image":
        info = _inspect_image(payload) if payload is not None else _inspect_image_file(path)  # type: ignore[arg-type]
    else:
        assert path is not None
        info = _inspect_video(path, _find_ffprobe(ffmpeg_path), audio_expected)

    digest = hashlib.sha256(payload).hexdigest() if payload is not None else sha256_file(path)  # type: ignore[arg-type]
    identity = fixture_identity_sha256(fixture_id, kind, source, digest)
    return PreparedFixture(
        fixture_id=fixture_id,
        media_type=media_type,
        source_name=_source_name(source_name),
        path=path,
        payload=payload,
        source_sha256=digest,
        fixture_identity_sha256=identity,
        source_info=info,
        audio_expected=audio_expected if audio_expected is not None else info.get("audio_present"),
        source_kind=kind,
    )


class BenchmarkError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def validate_local_api_url(value: str) -> str:
    """Require loopback so fixture media cannot be sent to a remote API by mistake."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("API URL must be an HTTP(S) loopback URL without credentials.")
    hostname = parsed.hostname.lower()
    local = hostname == "localhost"
    try:
        local = local or ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        pass
    if not local:
        raise ValueError("Only a local loopback API URL is allowed; benchmark media stays on this computer.")
    return value.rstrip("/")


def _safe_error_text(value: Any, limit: int = 320) -> str:
    text = str(value or "")
    text = re.sub(r"(?i)\b[A-Z]:\\[^\r\n\"']+", "[local path]", text)
    text = re.sub(r"(?<![:\w])/(?:[^/\s\"']+/)*[^/\s\"']+", "[local path]", text)
    return text.strip()[:limit]


def _loopback_origin(base_url: str) -> str:
    parsed = urlsplit(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _binary_version(binary: str | None, flag: str) -> str | None:
    if not binary:
        return None
    try:
        result = subprocess.run([binary, flag], capture_output=True, text=True, check=False, timeout=5)
        text = result.stdout or result.stderr
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
        return first_line[:180] if first_line else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _nvidia_driver_version() -> str | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if result.returncode != 0:
            return None
        versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
        return versions[0] if len(versions) == 1 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _environment(status: dict[str, Any], ffmpeg_path: str | None, sample_resources: bool = False) -> dict[str, Any]:
    runtime = status.get("runtime") if isinstance(status.get("runtime"), dict) else {}
    adapter_value = runtime.get("gpu_device")
    adapter = None
    if isinstance(adapter_value, dict):
        adapter = {
            "id": adapter_value.get("id"),
            "name": adapter_value.get("name"),
            "vendor_id": adapter_value.get("vendor_id"),
            "device_id": adapter_value.get("device_id"),
            "dedicated_memory_mb": adapter_value.get("memory_mb"),
        }
    driver_version = _nvidia_driver_version()
    ffmpeg_version = _binary_version(ffmpeg_path, "-version")
    providers = runtime.get("providers") if isinstance(runtime.get("providers"), list) else []
    public_runtime = {
        "onnxruntime_version": runtime.get("version"),
        "available_providers": sorted(str(item) for item in providers),
        "active_provider_at_start": runtime.get("active_provider"),
        "gpu_available": runtime.get("gpu_available"),
        "adapter": adapter,
        "driver_version": driver_version,
        "driver_source": "nvidia-smi" if driver_version else None,
        "capabilities": runtime.get('capabilities'),
        "ffmpeg_version": ffmpeg_version,
    }
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_identity": os.environ.get('PROCESSOR_IDENTIFIER') or platform.processor(),
        "cpu_count": os.cpu_count(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "httpx_version": httpx.__version__,
        "server_build_fingerprint": status.get('build_fingerprint'),
        "resource_sampling_enabled": sample_resources,
        "runtime": public_runtime,
        "device_memory_sampling": {
            "available": False,
            "sampled_memory_mib": None,
            "exact_process_peak_claimed": False,
        },
    }
    fingerprint_environment = dict(environment)
    fingerprint_environment['runtime'] = {key: value for key, value in public_runtime.items()
                                          if key != 'active_provider_at_start'}
    environment["runtime_fingerprint"] = stable_sha256(fingerprint_environment)
    return environment


def _model_record(catalog: Any, model_id: str) -> dict[str, Any]:
    models = catalog.get("models", []) if isinstance(catalog, dict) else []
    if isinstance(models, list):
        match = next((model for model in models if isinstance(model, dict) and model.get("id") == model_id), None)
    else:
        match = None
    if match:
        return {
            "id": model_id,
            "state": match.get("state"),
            "hash": match.get("sha256"),
            "revision": match.get("revision"),
            "hash_source": "local model catalog" if match.get("sha256") else None,
        }
    if model_id in {"balanced", "approximation"}:
        return {"id": model_id, "state": "built_in_approximation", "hash": None, "revision": None,
                "hash_source": "not applicable; deterministic OpenCV approximation"}
    if model_id == "precision":
        return {"id": model_id, "state": "custom_onnx", "hash": None, "revision": None,
                "hash_source": "custom model hash is not exposed by the local API"}
    return {"id": model_id, "state": None, "hash": None, "revision": None,
            "hash_source": "selected model was not present in the local catalog"}


def _request_json(client: httpx.Client, method: str, path: str) -> Any:
    try:
        response = client.request(method, path)
    except httpx.HTTPError as error:
        raise BenchmarkError("api_unreachable", f"Local API request failed ({type(error).__name__}).") from None
    if not 200 <= response.status_code < 300:
        raise BenchmarkError("api_http_error", f"Local API returned HTTP {response.status_code} for {path}.")
    try:
        return response.json()
    except (ValueError, json.JSONDecodeError):
        raise BenchmarkError("invalid_api_json", f"Local API returned invalid JSON for {path}.") from None


def _probe_output(path: Path, ffprobe_path: str | None) -> dict[str, Any] | None:
    return _probe_video(path, ffprobe_path)


def inspect_video_output(path: Path, ffprobe_path: str | None) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    width = height = None
    decoded_frames = 0
    consistent_dimensions = True
    fps_from_decoder: float | None = None
    try:
        if not capture.isOpened():
            return {"width": None, "height": None, "decoded_frames": 0, "fps": None,
                    "duration_seconds": None, "audio_present": None, "consistent_dimensions": False}
        fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_from_decoder = fps_value if fps_value > 0 else None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_height, current_width = frame.shape[:2]
            if width is None:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                consistent_dimensions = False
            decoded_frames += 1
    finally:
        capture.release()
    probe = _probe_output(path, ffprobe_path)
    return {
        "width": width,
        "height": height,
        "decoded_frames": decoded_frames,
        "fps": probe.get("fps") if probe and probe.get("fps") else fps_from_decoder,
        "duration_seconds": probe.get("duration_seconds") if probe else None,
        "audio_present": probe.get("audio_present") if probe else None,
        "consistent_dimensions": consistent_dimensions,
        "reported_frames": probe.get("reported_frames") if probe else None,
    }


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "depth_mode": args.model,
        "execution_mode": args.device,
        "input_size": args.input_size,
        "layout": args.layout,
        "strength": args.strength,
        "convergence": args.convergence,
        "resolution": "source",
        "quality": args.quality,
        "invert_depth": False,
        "temporal_smoothing": 0.35,
        "edge_refinement": True,
    }


def _expected_output(source_info: dict[str, Any], media_type: str, settings: dict[str, Any], audio_expected: bool | None) -> dict[str, Any]:
    if media_type == "image":
        width, height = expected_output_dimensions(source_info["width"], source_info["height"], settings["layout"])
        return {"width": width, "height": height}
    width, height = expected_video_dimensions(source_info["width"], source_info["height"], settings["layout"])
    return {
        "width": width,
        "height": height,
        "frames": source_info.get("frames"),
        "fps": source_info.get("fps"),
        "duration_seconds": source_info.get("duration_seconds"),
        "audio_present": audio_expected,
    }


def _empty_timing() -> dict[str, Any]:
    return {
        "submission_seconds": None,
        "job_wait_seconds": None,
        "download_seconds": None,
        "validation_seconds": None,
        "total_wall_seconds": None,
    }


def _failed_run(
    fixture_id: str,
    fixture_identity: str | None,
    source_hash: str | None,
    settings: dict[str, Any],
    model: dict[str, Any],
    repeat_index: int,
    error_code: str,
    error_message: str,
    elapsed: float | None = None,
    state: str = "failed",
) -> dict[str, Any]:
    timings = _empty_timing()
    timings["total_wall_seconds"] = elapsed
    return {
        "run_id": str(uuid.uuid4()),
        "fixture_id": fixture_id,
        "fixture_identity_sha256": fixture_identity,
        "source_sha256": source_hash,
        "repeat_index": repeat_index,
        "state": state,
        "success": False,
        "error": {"code": error_code, "message": _safe_error_text(error_message)},
        "settings": settings,
        "model_id": model.get("id"),
        "model_hash": model.get("hash"),
        "encoder": None,
        "pipeline": None,
        "retry_count": None,
        "execution": {},
        "job_timings": job_timing_snapshot({}),
        "runner_timings": timings,
        "verification": None,
    }


def _poll_job(client: httpx.Client, job_id: str, timeout_seconds: float, poll_seconds: float) -> dict[str, Any]:
    started = time.perf_counter()
    while True:
        jobs = _request_json(client, "GET", "/api/jobs")
        if not isinstance(jobs, list):
            raise BenchmarkError("invalid_jobs_response", "Local API returned an invalid jobs list.")
        job = next((candidate for candidate in jobs if isinstance(candidate, dict) and candidate.get("id") == job_id), None)
        if job is None:
            raise BenchmarkError("job_disappeared", "Submitted job was not present in the local queue response.")
        if job.get("state") in {"completed", "failed", "canceled"}:
            return job
        if time.perf_counter() - started >= timeout_seconds:
            try:
                client.post(f"/api/jobs/{job_id}/cancel")
            except httpx.HTTPError:
                pass
            raise BenchmarkError("job_timeout", f"Job did not finish within {timeout_seconds:g} seconds.")
        time.sleep(poll_seconds)


def _execute_run(
    client: httpx.Client,
    source: PreparedFixture,
    settings: dict[str, Any],
    model: dict[str, Any],
    environment: dict[str, Any],
    ffmpeg_path: str | None,
    repeat_index: int,
    job_timeout: float,
    poll_seconds: float,
    work_directory: Path,
    sample_resources: bool = False,
) -> dict[str, Any]:
    wall_started = time.perf_counter()
    def resource_record():
        if not sample_resources:
            return None
        try:
            response = client.get('/api/diagnostics/resources')
            return response.json() if response.status_code == 200 else {'unavailable': f'HTTP {response.status_code}'}
        except (httpx.HTTPError, ValueError):
            return {'unavailable': 'Resource diagnostics could not be read.'}
    resource_before = resource_record()
    runner_timings = _empty_timing()
    job: dict[str, Any] = {}
    verification: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    state = "failed"
    job_id = None
    execution: dict[str, Any] = {}
    output_payload: bytes | None = None
    output_path = work_directory / f"{source.fixture_id}-run-{repeat_index}-output.mp4"

    submit_started = time.perf_counter()
    try:
        mime = mimetypes.guess_type(source.source_name)[0] or "application/octet-stream"
        if source.path is not None:
            with source.path.open("rb") as file_handle:
                response = client.post("/api/jobs", data=settings,
                                       files={"file": (source.source_name, file_handle, mime)})
        else:
            response = client.post("/api/jobs", data=settings,
                                   files={"file": (source.source_name, source.payload or b"", mime)})
        runner_timings["submission_seconds"] = time.perf_counter() - submit_started
        if not 200 <= response.status_code < 300:
            raise BenchmarkError("submit_http_error", f"Job submission returned HTTP {response.status_code}.")
        try:
            created = response.json()
        except (ValueError, json.JSONDecodeError):
            raise BenchmarkError("submit_invalid_json", "Job submission returned invalid JSON.") from None
        job_id = created.get("id") if isinstance(created, dict) else None
        if not job_id:
            raise BenchmarkError("submit_missing_id", "Job submission did not return a job id.")

        poll_started = time.perf_counter()
        job = _poll_job(client, str(job_id), job_timeout, poll_seconds)
        runner_timings["job_wait_seconds"] = time.perf_counter() - poll_started
        state = str(job.get("state", "unknown"))
        execution = job.get("execution") if isinstance(job.get("execution"), dict) else {}
        if state != "completed":
            error = {"code": "job_" + state, "message": _safe_error_text(job.get("error") or job.get("message") or "Job did not complete.")}
        else:
            download_started = time.perf_counter()
            download_url = f"/api/jobs/{job_id}/download"
            try:
                if source.media_type == "image":
                    download = client.get(download_url)
                    if not 200 <= download.status_code < 300:
                        raise BenchmarkError("download_http_error", f"Output download returned HTTP {download.status_code}.")
                    output_payload = download.content
                else:
                    with client.stream("GET", download_url) as download:
                        if not 200 <= download.status_code < 300:
                            raise BenchmarkError("download_http_error", f"Output download returned HTTP {download.status_code}.")
                        with output_path.open("wb") as output:
                            for chunk in download.iter_bytes():
                                output.write(chunk)
            except httpx.HTTPError as exc:
                raise BenchmarkError("download_failed", f"Output download failed ({type(exc).__name__}).") from None
            runner_timings["download_seconds"] = time.perf_counter() - download_started

            validation_started = time.perf_counter()
            expected = _expected_output(source.source_info, source.media_type, settings, source.audio_expected)
            if source.media_type == "image":
                verification = verify_image_output(output_payload or b"", expected)
            else:
                observation = inspect_video_output(output_path, _find_ffprobe(ffmpeg_path))
                verification = verify_video_observation(observation, expected)
            runner_timings["validation_seconds"] = time.perf_counter() - validation_started
            if verification.get("status") != "passed":
                error = {"code": "output_verification_" + str(verification.get("status")),
                         "message": "Decoded output did not pass every available manifest and media check."}
    except BenchmarkError as exc:
        error = {"code": exc.code, "message": _safe_error_text(exc.message)}
    except (OSError, ValueError, TypeError, httpx.HTTPError) as exc:
        error = {"code": "runner_error", "message": _safe_error_text(f"{type(exc).__name__}: {exc}")}
    finally:
        resource_after = resource_record()
        runner_timings["total_wall_seconds"] = time.perf_counter() - wall_started

    timings = job_timing_snapshot(job)
    first_execution = job.get("first_execution") if isinstance(job.get("first_execution"), dict) else None
    execution_report = normalize_execution_snapshot(execution)
    first_execution_report = normalize_execution_snapshot(first_execution)
    return {
        "run_id": str(uuid.uuid4()),
        "job_id": job_id,
        "fixture_id": source.fixture_id,
        "fixture_identity_sha256": source.fixture_identity_sha256,
        "source_sha256": source.source_sha256,
        "source_name": source.source_name,
        "source_kind": source.source_kind,
        "resources": {'process_before': resource_before, 'process_after': resource_after,
                      'scope': 'Server process snapshots; peak working set is process-lifetime, not a per-job allocation peak. Optional requests add measurement overhead.'},
        "repeat_index": repeat_index,
        "state": state,
        "success": error is None and verification is not None and verification.get("status") == "passed",
        "error": error,
        "settings": settings,
        "model_id": model.get("id"),
        "model_hash": model.get("hash"),
        "encoder": job.get('encoder'),
        "pipeline": job.get('pipeline'),
        "retry_count": job.get('retry_count'),
        "source_input_dimensions": {"width": source.source_info.get("width"), "height": source.source_info.get("height")},
        "requested_model_input": {"detail": settings.get("input_size")},
        "requested_model_input_dimensions": {
            "width": execution.get("requested_input_width"),
            "height": execution.get("requested_input_height"),
        },
        "actual_model_input_dimensions": {
            "width": execution.get("input_width"),
            "height": execution.get("input_height"),
        },
        "execution": execution_report,
        "first_execution": first_execution_report,
        "environment": {"runtime_fingerprint": environment.get("runtime_fingerprint")},
        "job_timings": timings,
        "runner_timings": runner_timings,
        "verification": verification,
    }


def _positive_timeout(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive number") from None
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return number


def _minimum_repeats(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an integer") from None
    if number < 3:
        raise argparse.ArgumentTypeError("must be at least 3 for repeatability")
    return number


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run reproducible local StereoLab image/video benchmarks.")
    parser.add_argument("--base-url", default=DEFAULT_API_URL, help=f"Local StereoLab API (default: {DEFAULT_API_URL})")
    parser.add_argument("--manifest", type=Path, default=root / "benchmarks" / "manifest.json")
    parser.add_argument("--output", type=Path, default=None, help="JSON report path (default: ignored .runtime/benchmarks directory)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Approximation, custom ONNX, or named model id")
    parser.add_argument("--device", choices=sorted(VALID_DEVICES), default=DEFAULT_DEVICE)
    parser.add_argument("--repeats", type=_minimum_repeats, default=DEFAULT_REPEATS)
    parser.add_argument("--fixture", action="append", dest="fixtures", help="Run only this manifest fixture id; may be repeated")
    parser.add_argument("--input-size", type=int, choices=(256, 384, 518), default=256)
    parser.add_argument("--layout", choices=sorted(VALID_LAYOUTS), default="SBS")
    parser.add_argument("--strength", type=float, default=6.0)
    parser.add_argument("--convergence", type=float, default=0.5)
    parser.add_argument("--quality", choices=sorted(VALID_QUALITIES), default="fast")
    parser.add_argument("--timeout-seconds", type=_positive_timeout, default=300.0)
    parser.add_argument("--poll-seconds", type=_positive_timeout, default=0.25)
    parser.add_argument("--include-source-paths", action="store_true", help="Include full source paths in the report")
    parser.add_argument('--sample-resources', action='store_true', help='Opt in to process snapshots and sampled whole-device NVIDIA memory; adds observation overhead')
    return parser


def _run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    base_url = validate_local_api_url(args.base_url)
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixtures = validate_manifest(manifest)
    if args.fixtures:
        fixture_ids = set(args.fixtures)
        unknown = fixture_ids - {fixture["id"] for fixture in fixtures}
        if unknown:
            raise ValueError("Unknown fixture id(s): " + ", ".join(sorted(unknown)))
        fixtures = [fixture for fixture in fixtures if fixture["id"] in fixture_ids]

    settings = _settings(args)
    started_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": {
            "name": "StereoLab roadmap 0.1 local benchmark",
            "started_at_utc": started_at,
            "requested_repeats": args.repeats,
            "api_origin": _loopback_origin(base_url),
            "settings": settings,
        },
        "measurement_notes": [
            "Synthetic fixtures are deterministic pipeline checks and do not provide visual-quality ground truth.",
            "Session initialization and first-frame timings are recorded when the API exposes them; they do not guarantee a fresh-process cold start because the service may already have a model session.",
            "Stage service times may overlap and must not be summed as wall time.",
            "Device memory sampling was not performed; no exact process VRAM allocation or peak is claimed.",
            "Medians and ranges include only repeated successful runs with identical source, settings, model, actual provider/device and runtime fingerprint.",
            "Generated fixture and output-check files are kept in the ignored .runtime/benchmark-cache directory.",
        ],
        "environment": {},
        "model": {},
        "fixtures": [],
        "runs": [],
        "summaries": [],
    }

    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        status = _request_json(client, "GET", "/api/status")
        if not isinstance(status, dict):
            raise BenchmarkError("invalid_status_response", "Local API returned an invalid status object.")
        catalog = _request_json(client, "GET", "/api/models")
        model = _model_record(catalog, args.model)
        ffmpeg_path = status.get("ffmpeg_path") if isinstance(status.get("ffmpeg_path"), str) else shutil.which("ffmpeg")
        environment = _environment(status, ffmpeg_path, args.sample_resources)
        report["environment"] = environment
        report["model"] = model

        temporary_parent = Path(__file__).resolve().parent / ".runtime" / "benchmark-cache"
        temporary_parent.mkdir(parents=True, exist_ok=True)
        working_directory = temporary_parent / uuid.uuid4().hex
        working_directory.mkdir()
        with nullcontext(str(working_directory)) as temporary:
            work_directory = Path(temporary)
            for fixture in fixtures:
                fixture_path_value = fixture["source"].get("path")
                try:
                    source = prepare_fixture(fixture, manifest_path.parent, work_directory, ffmpeg_path)
                    fixture_record: dict[str, Any] = {
                        "id": source.fixture_id,
                        "kind": source.source_kind,
                        "media_type": source.media_type,
                        "source_name": source.source_name,
                        "source_sha256": source.source_sha256,
                        "fixture_identity_sha256": source.fixture_identity_sha256,
                        "source_info": source.source_info,
                    }
                    if args.include_source_paths and fixture_path_value:
                        source_path = Path(fixture_path_value)
                        if not source_path.is_absolute():
                            source_path = manifest_path.parent / source_path
                        fixture_record["source_path"] = str(source_path.resolve())
                    report["fixtures"].append(fixture_record)
                    for repeat_index in range(1, args.repeats + 1):
                        run = _execute_run(client, source, settings, model, environment,
                                           ffmpeg_path, repeat_index, args.timeout_seconds, args.poll_seconds,
                                           work_directory, args.sample_resources)
                        run["environment"] = {"runtime_fingerprint": environment["runtime_fingerprint"]}
                        report["runs"].append(run)
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    source_name = _source_name(str(fixture_path_value)) if fixture_path_value else f"{fixture['id']}.generated"
                    fixture_record = {"id": fixture["id"], "kind": fixture["source"]["kind"],
                                      "source_name": source_name, "preparation_status": "failed"}
                    if args.include_source_paths and fixture_path_value:
                        source_path = Path(fixture_path_value)
                        if not source_path.is_absolute():
                            source_path = manifest_path.parent / source_path
                        fixture_record["source_path"] = str(source_path.resolve())
                    report["fixtures"].append(fixture_record)
                    for repeat_index in range(1, args.repeats + 1):
                        report["runs"].append(_failed_run(
                            fixture["id"], None, None, settings, model, repeat_index,
                            "fixture_preparation_failed", _safe_error_text(f"{type(exc).__name__}: {exc}"),
                        ))

    report["summaries"] = summarize_runs(report["runs"])
    report["benchmark"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["benchmark"]["run_records"] = len(report["runs"])
    report["benchmark"]["successful_runs"] = sum(bool(run.get("success")) for run in report["runs"])
    return report


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    with DeviceMemorySampler() if args.sample_resources else nullcontext(None) as sampler:
        report = _run_benchmark(args)
    if sampler is not None:
        report['environment']['device_memory_sampling'] = sampler.snapshot()
        report['measurement_notes'] = [note for note in report['measurement_notes'] if not note.startswith('Device memory sampling was not performed')]
        report['measurement_notes'].append('Opt-in whole-device memory samples cover fixture preparation and all runs, include other processes, and add observation overhead. Not a process VRAM peak.')
    return report


def benchmark_exit_code(report: dict[str, Any]) -> int:
    runs = report.get("runs")
    summaries = report.get("summaries")
    if not isinstance(runs, list) or not runs or not isinstance(summaries, list) or not summaries:
        return 1
    if any(not isinstance(run, dict) or not run.get("success") for run in runs):
        return 1
    if not any(isinstance(item, dict) and item.get("successful_runs", 0) >= 2 for item in summaries):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args)
    except (OSError, ValueError, BenchmarkError, json.JSONDecodeError) as exc:
        print(_safe_error_text(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parent
    output_path = args.output
    if output_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = root / ".runtime" / "benchmarks" / f"roadmap-{stamp}.json"
    output_path = Path(output_path).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    exit_code = benchmark_exit_code(report)
    print(json.dumps({"report": str(output_path), "schema_version": report["schema_version"],
                      "runs": len(report["runs"]), "successful_runs": report["benchmark"]["successful_runs"],
                      "summaries": len(report["summaries"]), "status": "complete" if exit_code == 0 else "incomplete"}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
