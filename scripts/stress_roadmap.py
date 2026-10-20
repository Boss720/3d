"""Run a bounded, sequential all-model synthetic video stress check."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import mimetypes
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmarking
from resource_sampling import DeviceMemorySampler
from process_tree import (
    ProcessRootNotFound,
    executable_basename,
    find_descendants,
    match_process_identity,
    snapshot_windows_processes,
)


MODEL_IDS = (
    "depth-anything-v2-small",
    "depth-anything-v2-base",
    "dpt-hybrid-midas",
    "depth-anything-v2-large",
)
TERMINAL_STATES = {"completed", "failed", "canceled"}
LIVE_STATES = {"queued", "processing", "running", "canceling"}
FIXTURE_ID = "stress-pan-thin-lowlight-cuts-audio-30s"
MANIFEST_PATH = ROOT / "benchmarks" / "stress-manifest.json"


def active_jobs(jobs: Any, excluded_ids: set[str] | None = None) -> list[dict[str, Any]]:
    """Return non-terminal API jobs, preserving only safe activity metadata."""
    excluded = excluded_ids or set()
    if not isinstance(jobs, list):
        return []
    return [
        {key: job.get(key) for key in ("id", "state", "media_type", "depth_mode", "execution_mode")}
        for job in jobs
        if isinstance(job, dict)
        and str(job.get("id")) not in excluded
        and str(job.get("state", "")).lower() not in TERMINAL_STATES
    ]


def cancellation_ready(job: dict[str, Any], minimum_frames: int = 5) -> bool:
    """Gate cancellation on a live conversion before its final frame."""
    if not isinstance(job, dict) or isinstance(minimum_frames, bool) or not isinstance(minimum_frames, int):
        return False
    frames, total_frames = job.get("frames_processed"), job.get("total_frames")
    if (isinstance(frames, bool) or not isinstance(frames, int)
            or isinstance(total_frames, bool) or not isinstance(total_frames, int)):
        return False
    return (str(job.get("state", "")).lower() == "processing"
            and str(job.get("phase", "")).lower() == "conversion"
            and frames >= minimum_frames and total_frames > frames)


def require_explicit_gpu_execution(execution: dict[str, Any]) -> None:
    """Reject a nominally successful export that fell back from explicit DirectML."""
    valid = (
        execution.get("requested_mode") == "gpu"
        and execution.get("active_provider") == "DmlExecutionProvider"
        and execution.get("gpu_verified") is True
        and execution.get("fallback_reason") is None
    )
    if not valid:
        raise ValueError("Export did not verify explicit DmlExecutionProvider execution.")


def require_expected_api_process(diagnostics: Any, expected_pid: int,
                                 process_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed unless the loopback API reports the expected live Windows process."""
    if isinstance(expected_pid, bool) or not isinstance(expected_pid, int) or expected_pid < 1:
        raise ValueError("An explicit positive expected API PID is required.")
    if not isinstance(diagnostics, dict):
        raise ValueError("API process diagnostics are unavailable or malformed.")
    reported_pid = diagnostics.get("pid")
    if isinstance(reported_pid, bool) or not isinstance(reported_pid, int) or reported_pid != expected_pid:
        raise ValueError("The loopback API PID does not match --expected-server-pid.")
    if not isinstance(process_entries, list):
        raise ValueError("Windows process snapshot is unavailable or malformed.")
    root = next((entry for entry in process_entries
                 if isinstance(entry, dict) and entry.get("pid") == expected_pid), None)
    if root is None:
        raise ProcessRootNotFound(f"Expected API PID {expected_pid} is absent from the Windows process snapshot.")
    executable = executable_basename(root.get("executable", ""))
    if executable.casefold() not in {"python.exe", "pythonw.exe", "stereolab.exe"}:
        raise ValueError(f"Expected API PID has an unrecognized server executable: {executable or 'unknown'}.")
    if (root.get("identity_status") != "verified"
            or isinstance(root.get("creation_time_100ns"), bool)
            or not isinstance(root.get("creation_time_100ns"), int)
            or root["creation_time_100ns"] < 1):
        raise ValueError("Expected API process creation identity could not be verified.")
    return dict(root)


class FfmpegProcessObserver:
    """Read-only, bounded observation of FFmpeg children for one verified API process."""

    def __init__(self, api_process: dict[str, Any], sample_interval_seconds: float,
                 snapshot_fn=snapshot_windows_processes):
        self.api_process = dict(api_process)
        self.snapshot_fn = snapshot_fn
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.sample_count = 0
        self.sample_error_count = 0
        self.errors: list[str] = []
        self._observed: dict[tuple[int, int | None, str], dict[str, Any]] = {}
        self._last_job_snapshot: dict[str, Any] | None = None
        self._terminal_observed_monotonic: float | None = None
        self._terminal_observed_at_utc: str | None = None

    def _record_error(self, error: Exception) -> None:
        self.sample_error_count += 1
        if len(self.errors) < 16:
            self.errors.append(f"{type(error).__name__}: {str(error)[:200]}")

    def observe_job_snapshot(self, job: dict[str, Any]) -> None:
        if isinstance(job, dict):
            self._last_job_snapshot = {
                key: job.get(key) for key in ("state", "phase", "frames_processed", "total_frames")
            }
            if job.get("state") in TERMINAL_STATES:
                if self._terminal_observed_monotonic is None:
                    self._terminal_observed_monotonic = time.monotonic()
                    self._terminal_observed_at_utc = datetime.now(timezone.utc).isoformat()
                # The immediate after_terminal() check owns this sample. It must
                # count as post-terminal visibility, not a sample during export.
                return
        self.sample()

    def sample(self) -> None:
        try:
            entries = self.snapshot_fn()
            self.sample_count += 1
            api_pid = self.api_process["pid"]
            root_status = match_process_identity(entries, self.api_process)["status"]
            if root_status != "present":
                raise ProcessRootNotFound(f"Verified API process identity changed: {root_status}.")
            descendants = find_descendants(entries, api_pid)
            for entry in descendants:
                if executable_basename(entry.get("executable", "")).casefold() != "ffmpeg.exe":
                    continue
                creation = entry.get("creation_time_100ns")
                identity_ok = (entry.get("identity_status") == "verified"
                               and isinstance(creation, int) and not isinstance(creation, bool) and creation > 0)
                relationship_ok = entry.get("tree_identity_status") == "verified"
                key = (int(entry["pid"]), int(creation) if identity_ok else None,
                       executable_basename(entry.get("executable", "")).casefold())
                record = self._observed.setdefault(key, {
                    "pid": key[0], "parent_pid": entry.get("parent_pid"),
                    "executable": executable_basename(entry.get("executable", "")),
                    "creation_time_100ns": key[1],
                    "identity_status": "verified" if identity_ok else "unverified",
                    "tree_relationship_status": "verified" if relationship_ok else "unverified",
                    "samples_observed": 0,
                })
                record["samples_observed"] += 1
        except Exception as exc:
            self._record_error(exc)

    def after_terminal(self, timeout_seconds: float = 5.0, interval_seconds: float = 0.1) -> dict[str, Any]:
        started = (self._terminal_observed_monotonic
                   if self._terminal_observed_monotonic is not None else time.monotonic())
        terminal_states = {"completed", "failed", "canceled"}
        job_reached_terminal = (self._last_job_snapshot is not None
                                and self._last_job_snapshot.get("state") in terminal_states)
        if not job_reached_terminal:
            return {
                "api_process": {key: self.api_process.get(key) for key in
                                ("pid", "executable", "creation_time_100ns", "identity_status")},
                "sample_interval_seconds": self.sample_interval_seconds,
                "sample_count": self.sample_count,
                "sample_error_count": self.sample_error_count,
                "errors": list(self.errors),
                "last_job_snapshot": self._last_job_snapshot,
                "observed_ffmpeg_processes": list(self._observed.values()),
                "post_terminal_check": {
                    "deadline_seconds": timeout_seconds,
                    "attempts": 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "observations": [],
                    "ever_visible_after_terminal": False,
                    "identity_statuses": [],
                },
                "job_reached_terminal": False,
                "terminal_observed_at_utc": None,
                "status": "unverified_job_not_terminal",
                "scope": "No post-terminal process observation was made because the job had no observed terminal snapshot.",
            }
        deadline = started + timeout_seconds
        statuses: dict[tuple[int, int | None, str], str] = {}
        observations: list[dict[str, Any]] = []
        ever_visible: set[tuple[int, int | None, str]] = set()
        uncertain_identity_seen = False
        attempts = 0
        window_missed = timeout_seconds > 0 and time.monotonic() > deadline
        while not window_missed:
            attempts += 1
            try:
                entries = self.snapshot_fn()
                for key, observed in self._observed.items():
                    statuses[key] = match_process_identity(entries, observed)["status"]
            except Exception as exc:
                self._record_error(exc)
                statuses = {key: "unverified" for key in self._observed}
                uncertain_identity_seen = True
                break
            sample_completed = time.monotonic()
            if timeout_seconds > 0 and sample_completed > deadline:
                window_missed = True
            ever_visible.update(key for key, status in statuses.items() if status == "present")
            uncertain_identity_seen = uncertain_identity_seen or any(
                status in {"unverified", "reused"} for status in statuses.values())
            observations.append({
                "elapsed_seconds": round(sample_completed - started, 3),
                "within_deadline": timeout_seconds == 0 or sample_completed <= deadline,
                "identity_statuses": [
                    {"pid": key[0], "creation_time_100ns": key[1], "status": value}
                    for key, value in statuses.items()
                ],
            })
            if uncertain_identity_seen or not any(status == "present" for status in statuses.values()):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(min(interval_seconds, max(0.0, deadline - time.monotonic())))

        status_values = list(statuses.values())
        if self.sample_error_count:
            status = "unverified_snapshot_or_root_error"
        elif window_missed:
            status = "unverified_post_terminal_window_missed"
        elif not self.sample_count or not self._observed:
            status = "unverified_ffmpeg_not_observed_during_export"
        elif any(item.get("identity_status") != "verified" for item in self._observed.values()):
            status = "unverified_ffmpeg_identity"
        elif any(item.get("tree_relationship_status") != "verified" for item in self._observed.values()):
            status = "unverified_ffmpeg_tree_relationship"
        elif uncertain_identity_seen or any(value in {"unverified", "reused"} for value in status_values):
            status = "unverified_post_terminal_identity"
        elif any(value == "present" for value in status_values):
            status = f"still_visible_after_{timeout_seconds:g}_seconds"
        elif ever_visible:
            status = "observed_then_absent_during_window"
        else:
            status = "not_visible_at_first_post_terminal_sample"

        return {
            "api_process": {key: self.api_process.get(key) for key in
                            ("pid", "executable", "creation_time_100ns", "identity_status")},
            "sample_interval_seconds": self.sample_interval_seconds,
            "sample_count": self.sample_count,
            "sample_error_count": self.sample_error_count,
            "errors": list(self.errors),
            "last_job_snapshot": self._last_job_snapshot,
            "observed_ffmpeg_processes": list(self._observed.values()),
            "post_terminal_check": {
                "deadline_seconds": timeout_seconds,
                "attempts": attempts,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "observations": observations,
                "ever_visible_after_terminal": bool(ever_visible),
                "identity_statuses": [
                    {"pid": key[0], "creation_time_100ns": key[1], "status": value}
                    for key, value in statuses.items()
                ],
            },
            "job_reached_terminal": True,
            "terminal_observed_at_utc": self._terminal_observed_at_utc,
            "status": status,
            "scope": ("The post-terminal observation window was missed; no timely cleanup conclusion can be drawn."
                      if window_missed else
                      "Observed process identities were checked globally after terminal status. Visibility is reported at snapshot times; sampling does not establish exact exit timing."),
        }


def stop_reason_after_model(record: dict[str, Any], activity: dict[str, Any]) -> str | None:
    if not activity.get("idle"):
        return "server_has_active_jobs_after_model"
    error = (record.get("export") or {}).get("error") or {}
    cancellation = record.get("cancellation") or {}
    if error.get("code") == "job_timeout" or cancellation.get("status") == "frame_threshold_timeout":
        return "job_timeout_reached"
    return None


def make_stress_frame(index: int, width: int = 320, height: int = 180, seed: int = 20261018) -> np.ndarray:
    """Create a deterministic pan, thin-detail, animation, cut, and low-light frame."""
    segment = index // 120
    frame = benchmarking._gradient_frame(width, height, seed + segment * 97)
    if segment == 1:
        frame = np.ascontiguousarray(255 - frame)
    elif segment >= 2:
        frame = frame[:, :, [2, 0, 1]].copy()
        frame[:, :, 1] = np.minimum(255, frame[:, :, 1].astype(np.uint16) + 32).astype(np.uint8)
        frame = ((frame.astype(np.uint16) * 0.24).clip(0, 255)).astype(np.uint8)
    frame = np.roll(frame, -(index * 2 % width), axis=1)

    # One-pixel details move independently across the panning background.
    thin_x = index * 3 % width
    cv2.line(frame, (thin_x, 4), ((thin_x + 17) % width, height - 5), (250, 250, 250), 1, cv2.LINE_8)
    cv2.line(frame, ((thin_x + width // 2) % width, 0),
             ((thin_x + width // 2) % width, height - 1), (8, 8, 8), 1, cv2.LINE_8)

    # A small animated shape follows a repeatable curved path.
    cx = int(width * 0.15 + (index % 120) * width * 0.70 / 119)
    cy = int(height * 0.48 + np.sin(index * np.pi / 36.0) * height * 0.12)
    cv2.rectangle(frame, (max(0, cx - 7), max(0, cy - 5)),
                  (min(width - 1, cx + 7), min(height - 1, cy + 5)), (25, 235, 245), -1)
    cv2.circle(frame, ((cx + width // 3) % width, (height - cy) % height), 4, (230, 30, 220), 1)
    return np.ascontiguousarray(frame)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest() -> dict[str, Any]:
    value = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not value.get("fixtures"):
        raise ValueError("Stress manifest schema is invalid.")
    fixture = value["fixtures"][0]
    source = fixture.get("source", {})
    if (source.get("width"), source.get("height"), source.get("fps"), source.get("frames")) != (320, 180, 12, 360):
        raise ValueError("Stress fixture must remain 320x180, 12 fps, and 360 frames.")
    if source.get("audio") is not True or set(value.get("models", [])) != set(MODEL_IDS):
        raise ValueError("Stress manifest must include audio and all four built-in models.")
    return value


def _generate_source(manifest: dict[str, Any], directory: Path, ffmpeg_path: str) -> benchmarking.PreparedFixture:
    fixture = manifest["fixtures"][0]
    recipe = fixture["source"]
    width, height, fps, frames = (int(recipe[k]) for k in ("width", "height", "fps", "frames"))
    raw_path = directory / f"{FIXTURE_ID}-video-only.mp4"
    source_path = directory / f"{FIXTURE_ID}.mp4"
    directory.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError("OpenCV could not initialize the synthetic stress video encoder.")
    try:
        for index in range(frames):
            writer.write(make_stress_frame(index, width, height, int(recipe["seed"])))
    finally:
        writer.release()
    if not raw_path.is_file() or raw_path.stat().st_size == 0:
        raise RuntimeError("Synthetic video generation produced no frames.")
    duration = frames / float(fps)
    command = [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path),
               "-f", "lavfi", "-i",
               f"sine=frequency={float(recipe['audio_frequency_hz']):g}:sample_rate=44100:duration={duration:.6f}",
               "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
               "-shortest", str(source_path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or not source_path.is_file() or source_path.stat().st_size == 0:
        detail = (result.stderr or "ffmpeg returned no diagnostic").strip()[:240]
        raise RuntimeError(f"Could not add deterministic audio to the stress source ({detail}).")

    ffprobe_path = benchmarking._find_ffprobe(ffmpeg_path)
    probe = benchmarking._probe_video(source_path, ffprobe_path)
    info = benchmarking._inspect_video(source_path, ffprobe_path, declared_audio=True)
    if not probe or probe.get("audio_present") is not True:
        raise RuntimeError("ffprobe could not verify the generated source audio stream.")
    if info.get("frames") != frames or abs(float(info.get("fps") or 0) - fps) > 0.01:
        raise RuntimeError("Generated source does not decode to the manifest frame count and frame rate.")
    if info.get("width") != width or info.get("height") != height:
        raise RuntimeError("Generated source dimensions do not match the stress manifest.")
    if abs(float(info.get("duration_seconds") or 0) - duration) > 0.15:
        raise RuntimeError("Generated source duration does not match the stress manifest.")
    source_hash = benchmarking.sha256_file(source_path)
    identity = benchmarking.stable_sha256({
        "fixture_id": fixture["id"], "source_kind": "synthetic_video",
        "generator": recipe["generator"], "recipe": recipe,
    })
    return benchmarking.PreparedFixture(
        fixture_id=fixture["id"], media_type="video", source_name=source_path.name,
        path=source_path, payload=None, source_sha256=source_hash,
        fixture_identity_sha256=identity, source_info=info, audio_expected=True,
        source_kind="synthetic_video",
    )


def _activity(client: httpx.Client) -> dict[str, Any]:
    jobs = benchmarking._request_json(client, "GET", "/api/jobs")
    if not isinstance(jobs, list):
        raise benchmarking.BenchmarkError("invalid_jobs_response", "Local API returned an invalid jobs list.")
    counts: dict[str, int] = {}
    for job in jobs:
        if isinstance(job, dict):
            state = str(job.get("state", "unknown"))
            counts[state] = counts.get(state, 0) + 1
    active = active_jobs(jobs)
    return {
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_records": len(jobs), "state_counts": counts,
        "other_active_jobs": active, "idle": not active,
    }


def _server_resource(client: httpx.Client) -> dict[str, Any] | None:
    try:
        response = client.get("/api/diagnostics/resources")
        return response.json() if response.status_code == 200 else {"unavailable": f"HTTP {response.status_code}"}
    except (httpx.HTTPError, ValueError):
        return {"unavailable": "Server process diagnostics could not be read."}


def _find_job(client: httpx.Client, job_id: str) -> dict[str, Any] | None:
    jobs = benchmarking._request_json(client, "GET", "/api/jobs")
    if not isinstance(jobs, list):
        raise benchmarking.BenchmarkError("invalid_jobs_response", "Local API returned an invalid jobs list.")
    return next((job for job in jobs if isinstance(job, dict) and job.get("id") == job_id), None)


def _cancel_test(client: httpx.Client, source: benchmarking.PreparedFixture, model_id: str,
                 settings: dict[str, Any], timeout_seconds: float, poll_seconds: float) -> dict[str, Any]:
    before_activity = _activity(client)
    result: dict[str, Any] = {
        "success": False, "minimum_frames_required": 5, "trigger_frames_processed": None,
        "job_id": None, "final_state": None, "download_http_status": None,
        "os_encoder_process_status": "unverified",
        "note": "API cancellation and HTTP 404 do not establish that no OS encoder process remains.",
        "before_activity": before_activity, "server_process_before": _server_resource(client),
    }
    if not before_activity["idle"]:
        result["status"] = "skipped_server_busy"
        return result
    mime = mimetypes.guess_type(source.source_name)[0] or "application/octet-stream"
    with source.path.open("rb") as file_handle:
        response = client.post("/api/jobs", data=settings,
                               files={"file": (source.source_name, file_handle, mime)})
    if not 200 <= response.status_code < 300:
        result.update(status="submit_failed", error=f"HTTP {response.status_code}")
        return result
    created = response.json()
    job_id = created.get("id") if isinstance(created, dict) else None
    if not job_id:
        result["status"] = "submit_missing_id"
        return result
    result["job_id"] = str(job_id)
    started = time.perf_counter()
    trigger_job: dict[str, Any] | None = None
    while time.perf_counter() - started < timeout_seconds:
        job = _find_job(client, str(job_id))
        if job is None:
            result["status"] = "job_disappeared"
            return result
        result["last_job_snapshot"] = {key: job.get(key) for key in
                                        ("state", "phase", "progress", "frames_processed", "total_frames")}
        concurrent = active_jobs([job for job in benchmarking._request_json(client, "GET", "/api/jobs")
                                  if isinstance(job, dict)], {str(job_id)})
        if concurrent:
            result["other_active_jobs_during_run"] = concurrent
            result["status"] = "other_job_started_during_cancellation_fixture"
            try:
                client.post(f"/api/jobs/{job_id}/cancel")
            except httpx.HTTPError:
                pass
            break
        if cancellation_ready(job):
            trigger_job = job
            result["trigger_frames_processed"] = int(job.get("frames_processed", 0))
            result["trigger_phase"] = job.get("phase")
            result["trigger_total_frames"] = int(job.get("total_frames", 0))
            result["status"] = "cancel_requested_during_conversion_after_frame_threshold"
            try:
                canceled = client.post(f"/api/jobs/{job_id}/cancel")
                result["cancel_http_status"] = canceled.status_code
            except httpx.HTTPError as exc:
                result["status"] = f"cancel_request_failed_{type(exc).__name__}"
            break
        if str(job.get("state", "")).lower() in TERMINAL_STATES:
            result["final_state"] = job.get("state")
            result["status"] = "job_finished_before_frame_threshold"
            break
        time.sleep(poll_seconds)
    else:
        result["status"] = "frame_threshold_timeout"
        try:
            client.post(f"/api/jobs/{job_id}/cancel")
        except httpx.HTTPError:
            pass

    settle_started = time.perf_counter()
    while time.perf_counter() - settle_started < 60:
        try:
            final_job = _find_job(client, str(job_id))
        except benchmarking.BenchmarkError:
            final_job = None
        if final_job and str(final_job.get("state", "")).lower() in TERMINAL_STATES:
            result["final_state"] = final_job.get("state")
            break
        time.sleep(poll_seconds)
    if result.get("final_state") == "canceled":
        try:
            download = client.get(f"/api/jobs/{job_id}/download")
            result["download_http_status"] = download.status_code
        except httpx.HTTPError as exc:
            result["download_error"] = type(exc).__name__
    result["after_activity"] = _activity(client)
    result["server_process_after"] = _server_resource(client)
    threshold_met = trigger_job is not None and cancellation_ready(trigger_job)
    if trigger_job is not None:
        trigger_execution = trigger_job.get("execution") if isinstance(trigger_job.get("execution"), dict) else {}
        result["trigger_execution"] = benchmarking.normalize_execution_snapshot(trigger_execution)
        try:
            require_explicit_gpu_execution(result["trigger_execution"])
            result["explicit_gpu_verified"] = True
        except ValueError:
            result["explicit_gpu_verified"] = False
    result["success"] = (threshold_met and result.get("explicit_gpu_verified") is True
                         and result.get("final_state") == "canceled"
                         and result.get("download_http_status") == 404)
    result["api_canceled"] = result.get("final_state") == "canceled"
    if result.get("status") == "cancel_requested_during_conversion_after_frame_threshold":
        result["status"] = "passed" if result["success"] else "failed_after_cancel_request"
    return result


def _settings(model_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    configured = dict(manifest["settings"])
    return {"depth_mode": model_id, **configured}


def _model_result(client: httpx.Client, model_id: str, catalog: Any, status: dict[str, Any],
                  source: benchmarking.PreparedFixture, manifest: dict[str, Any], run_root: Path,
                  timeout_seconds: float, poll_seconds: float, sample_resources: bool,
                  api_process: dict[str, Any]) -> dict[str, Any]:
    model = benchmarking._model_record(catalog, model_id)
    settings = _settings(model_id, manifest)
    activity = _activity(client)
    record: dict[str, Any] = {
        "model": model, "settings": settings,
        "source": {"fixture_id": source.fixture_id, "fixture_identity_sha256": source.fixture_identity_sha256,
                   "source_sha256": source.source_sha256, "source_info": source.source_info},
        "runtime": {"build_fingerprint": status.get("build_fingerprint"),
                    "runtime_version": (status.get("runtime") or {}).get("version"),
                    "available_providers": (status.get("runtime") or {}).get("providers"),
                    "gpu": (status.get("runtime") or {}).get("gpu_name")},
        "pre_export_activity": activity,
    }
    if model.get("state") != "ready" or not model.get("hash"):
        record.update(export={"success": False, "status": "model_not_ready_no_download_attempted"},
                      cancellation={"success": False, "status": "skipped_model_not_ready",
                                    "os_encoder_process_status": "unverified"})
        return record
    if not activity["idle"]:
        record.update(export={"success": False, "status": "skipped_server_busy"},
                      cancellation={"success": False, "status": "skipped_server_busy",
                                    "os_encoder_process_status": "unverified"})
        return record

    work_directory = run_root / "outputs" / model_id
    work_directory.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = status.get("ffmpeg_path") if isinstance(status.get("ffmpeg_path"), str) else None
    environment = benchmarking._environment(status, ffmpeg_path, sample_resources)
    record["runtime"]["runtime_fingerprint"] = environment["runtime_fingerprint"]
    with DeviceMemorySampler() if sample_resources else nullcontext(None) as sampler:
        process_observer = FfmpegProcessObserver(api_process, poll_seconds)
        cleanup_observation = None

        def observe_job_snapshot(job: dict[str, Any]) -> None:
            nonlocal cleanup_observation
            process_observer.observe_job_snapshot(job)
            if job.get("state") in TERMINAL_STATES and cleanup_observation is None:
                cleanup_observation = process_observer.after_terminal()

        export = benchmarking._execute_run(
            client, source, settings, model, environment, ffmpeg_path, 1,
            timeout_seconds, poll_seconds, work_directory, sample_resources=True,
            on_job_snapshot=observe_job_snapshot,
        )
        export["process_cleanup_observation"] = (
            cleanup_observation if cleanup_observation is not None else process_observer.after_terminal())
        export["runtime_build_fingerprint"] = status.get("build_fingerprint")
        export["model_hash"] = model.get("hash")
        export["settings"] = settings
        export["source_file_sha256"] = source.source_sha256
        export_path = work_directory / f"{source.fixture_id}-run-1-output.mp4"
        if export_path.is_file():
            export["output_file"] = {"name": export_path.name, "size_bytes": export_path.stat().st_size,
                                     "sha256": benchmarking.sha256_file(export_path)}
        export["stress_gate"] = {"explicit_gpu": "not_checked"}
        if export.get("success"):
            try:
                require_explicit_gpu_execution(export.get("execution") or {})
                export["stress_gate"]["explicit_gpu"] = "passed"
            except ValueError as exc:
                export["success"] = False
                export["stress_gate"]["explicit_gpu"] = "failed"
                export["error"] = {"code": "explicit_gpu_not_verified", "message": str(exc)}
        else:
            export["stress_gate"]["explicit_gpu"] = "unverified_export_failed"
        record["export"] = export
        try:
            after_export = _activity(client)
        except (httpx.HTTPError, ValueError, benchmarking.BenchmarkError) as exc:
            after_export = {"idle": False, "other_active_jobs": [],
                            "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
        record["after_export_activity"] = after_export
        if after_export["idle"]:
            try:
                record["cancellation"] = _cancel_test(client, source, model_id, settings,
                                                       timeout_seconds, poll_seconds)
            except (OSError, ValueError, RuntimeError, httpx.HTTPError, benchmarking.BenchmarkError) as exc:
                record["cancellation"] = {
                    "success": False, "status": "runner_error",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "os_encoder_process_status": "unverified",
                }
        else:
            record["cancellation"] = {"success": False, "status": "skipped_server_busy_after_export",
                                       "other_active_jobs": after_export["other_active_jobs"],
                                       "os_encoder_process_status": "unverified"}
        record["device_memory_sampling"] = sampler.snapshot() if sampler is not None else {
            "enabled": False, "note": "Use --sample-resources to opt in to whole-device NVIDIA memory samples."}
    return record


def _positive_timeout(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a number from 1 to 1200 seconds") from None
    if not 1 <= number <= 1200:
        raise argparse.ArgumentTypeError("must be from 1 to 1200 seconds")
    return number


def _positive_server_pid(value: str) -> int:
    try:
        pid = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive process ID") from None
    if pid < 1:
        raise argparse.ArgumentTypeError("must be a positive process ID")
    return pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sequential 30-second all-model StereoLab stress exports.")
    parser.add_argument("--base-url", required=True, help="Explicit owned local GPU API; loopback only")
    parser.add_argument("--expected-server-pid", type=_positive_server_pid, required=True,
                        help="Require API diagnostics and Windows process snapshot to match this owned PID")
    parser.add_argument("--timeout-seconds", type=_positive_timeout, default=1200.0,
                        help="Maximum wait per export or cancellation job (1-1200; default 1200)")
    parser.add_argument("--poll-seconds", type=float, default=0.5, help="Local API job poll interval")
    parser.add_argument("--sample-resources", action="store_true",
                        help="Opt in to whole-device NVIDIA memory sampling; adds observation overhead")
    parser.add_argument("--model", action="append", choices=MODEL_IDS,
                        help="Run a selected model; repeat the flag. Default runs all four sequentially.")
    return parser


def run_stress(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    base_url = benchmarking.validate_local_api_url(args.base_url)
    expected_pid = getattr(args, "expected_server_pid", None)
    if isinstance(expected_pid, bool) or not isinstance(expected_pid, int) or expected_pid < 1:
        raise benchmarking.BenchmarkError("expected_server_pid_required",
                                          "Specify the positive PID of the private API process owned for this run.")
    manifest = _manifest()
    model_ids = args.model or list(MODEL_IDS)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = ROOT / ".runtime" / "stress-verification" / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_origin": benchmarking._loopback_origin(base_url),
        "manifest": {"path": "benchmarks/stress-manifest.json", "classification": manifest["classification"],
                     "fixture": manifest["fixtures"][0], "requested_models": model_ids},
        "measurement_notes": [
            "This generated 30-second clip is a pipeline stress fixture, not representative real-footage quality evidence.",
            "It does not establish multi-hour stability.",
            "Model jobs run sequentially and are submitted only after the API reports no active jobs.",
            "Server CPU time and working-set data come from /api/diagnostics/resources; device-wide memory is opt-in and includes other processes.",
            "API cancellation plus a 404 download check does not verify OS encoder-process cleanup.",
            "Windows process snapshots are taken on each job poll. A post-terminal process identity check reports visibility for up to five seconds; it does not establish exact exit timing.",
            "Terminal process observation runs before output download and adds up to five seconds to runner job-wait and total-wall timings; API job timings remain service measurements.",
        ],
        "preflight": {}, "models": [], "completed_at_utc": None,
    }
    _write_json(report_path, report)
    with httpx.Client(base_url=base_url, timeout=httpx.Timeout(60.0, connect=5.0)) as client:
        status = benchmarking._request_json(client, "GET", "/api/status")
        catalog = benchmarking._request_json(client, "GET", "/api/models")
        if not isinstance(status, dict) or not isinstance(catalog, dict):
            raise benchmarking.BenchmarkError("invalid_preflight", "Local API status or model catalog was invalid.")
        initial_activity = _activity(client)
        diagnostics = _server_resource(client)
        try:
            api_process = require_expected_api_process(
                diagnostics, expected_pid, snapshot_windows_processes())
        except (OSError, ValueError, RuntimeError, ProcessRootNotFound) as exc:
            report["preflight"] = {
                "activity": initial_activity,
                "server_process_resources": diagnostics,
                "owned_server_validation": {"status": "failed", "expected_pid": expected_pid,
                                            "error": f"{type(exc).__name__}: {str(exc)[:240]}"},
            }
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            report["success"] = False
            _write_json(report_path, report)
            raise benchmarking.BenchmarkError("owned_server_validation_failed", str(exc)) from None
        runtime = status.get("runtime") if isinstance(status.get("runtime"), dict) else {}
        ffmpeg_path = status.get("ffmpeg_path") if isinstance(status.get("ffmpeg_path"), str) else None
        model_records = [benchmarking._model_record(catalog, model_id) for model_id in model_ids]
        report["preflight"] = {
            "build_fingerprint": status.get("build_fingerprint"), "runtime": runtime,
            "server_process_resources": diagnostics, "activity": initial_activity,
            "owned_server_validation": {
                "status": "passed", "expected_pid": expected_pid,
                "api_process": {key: api_process.get(key) for key in
                                ("pid", "executable", "creation_time_100ns", "identity_status")},
            },
            "ffmpeg_path": ffmpeg_path,
            "models": model_records,
            "no_downloads_performed": True,
        }
        _write_json(report_path, report)
        if not ffmpeg_path or not runtime.get("gpu_available") or "DmlExecutionProvider" not in (runtime.get("providers") or []):
            raise benchmarking.BenchmarkError("gpu_preflight_failed", "The explicitly owned API does not currently report usable DirectML and FFmpeg.")
        try:
            source = _generate_source(manifest, run_root / "source", ffmpeg_path)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            report["source_preparation_error"] = f"{type(exc).__name__}: {str(exc)[:320]}"
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            report["success"] = False
            _write_json(report_path, report)
            return report, report_path
        report["source"] = {
            "fixture_id": source.fixture_id, "source_name": source.source_name,
            "source_file_sha256": source.source_sha256,
            "fixture_identity_sha256": source.fixture_identity_sha256,
            "source_info": source.source_info,
        }
        _write_json(report_path, report)

        stop_reason = None
        for index, model_id in enumerate(model_ids):
            if stop_reason:
                record = {"model": benchmarking._model_record(catalog, model_id),
                          "export": {"success": False, "status": "skipped_after_activity_or_timeout"},
                          "cancellation": {"success": False, "status": "skipped", "os_encoder_process_status": "unverified"},
                          "stop_reason": stop_reason}
            else:
                record = _model_result(client, model_id, catalog, status, source, manifest, run_root,
                                       args.timeout_seconds, args.poll_seconds, args.sample_resources,
                                       api_process)
                activity_after = _activity(client)
                stop_reason = stop_reason_after_model(record, activity_after)
                if stop_reason == "server_has_active_jobs_after_model":
                    record["post_model_activity"] = activity_after
            report["models"].append(record)
            model_report_path = run_root / "models" / f"{model_id}.json"
            _write_json(model_report_path, record)
            report["completed_model_count"] = len(report["models"])
            _write_json(report_path, report)
            print(json.dumps({"event": "model_complete", "model": model_id,
                              "export_success": (record.get("export") or {}).get("success"),
                              "cancellation_success": (record.get("cancellation") or {}).get("success"),
                              "report": str(model_report_path.relative_to(ROOT))}), flush=True)

    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["success"] = (len(report["models"]) == len(model_ids)
                         and all((item.get("export") or {}).get("success")
                                 and (item.get("cancellation") or {}).get("success")
                                 for item in report["models"]))
    _write_json(report_path, report)
    return report, report_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.poll_seconds <= 0 or args.poll_seconds > 5:
        print("poll-seconds must be greater than 0 and at most 5.", file=sys.stderr)
        return 2
    try:
        report, path = run_stress(args)
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired,
            benchmarking.BenchmarkError, json.JSONDecodeError) as exc:
        print(f"Stress run failed: {type(exc).__name__}: {str(exc)[:320]}", file=sys.stderr)
        return 2
    print(json.dumps({"event": "stress_complete", "success": report.get("success"), "report": str(path)}), flush=True)
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
