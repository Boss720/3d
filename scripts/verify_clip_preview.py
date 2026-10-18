"""Bounded live verification for the local processed-clip API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import numpy as np

from benchmarking import (
    PreparedFixture, _environment, _execute_run, _find_ffprobe, _model_record,
    expected_video_dimensions, inspect_video_output, job_timing_snapshot,
    sha256_file, validate_local_api_url,
)
from preview_clip import ClipCancelled, prepare_preview_clip

MODEL = "depth-anything-v2-small"
SOURCE = ROOT / ".runtime/stress-verification/20261018T110251Z-f0392917/source/stress-pan-thin-lowlight-cuts-audio-30s.mp4"
TAGS = ["generated_synthetic", "landscape", "low_light", "animation", "thin_detail",
        "moving_thin_objects", "camera_pan", "hard_cuts", "audio", "not_licensed_footage"]
ACTIVE = {"queued", "processing"}
TERMINAL = {"completed", "failed", "canceled"}
TIMEOUT = 1200


def require_versioned_clip_capability(status: dict) -> dict:
    caps = status.get("preview_capabilities") if isinstance(status, dict) else None
    clip = caps.get("processed_clip") if isinstance(caps, dict) else None
    if not isinstance(caps, dict) or caps.get("schema_version") != 1 or not isinstance(clip, dict) or clip.get("supported") is not True:
        raise ValueError("Processed clip capability schema is missing, unsupported, or unavailable.")
    for key in ("max_duration_seconds", "max_frames", "max_fps", "max_width", "max_height"):
        value = clip.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Processed clip capability has invalid {key}.")
    if clip.get("temporal_context") != "fresh":
        raise ValueError("Processed clip capability does not guarantee fresh temporal context.")
    return clip


def assert_idle(jobs: list[dict]) -> None:
    if not isinstance(jobs, list):
        raise ValueError("Local API returned an invalid jobs list.")
    active = [{"id": j.get("id"), "state": j.get("state"), "purpose": j.get("purpose")}
              for j in jobs if isinstance(j, dict) and j.get("state") in ACTIVE]
    if active:
        raise RuntimeError(f"Refusing to submit while API jobs are active: {active}")


def validate_clip_bounds(start_seconds: float, duration_seconds: float, source_duration: float,
                         source_fps: float, width: int, height: int, capability: dict) -> dict:
    values = (start_seconds, duration_seconds, source_duration, source_fps)
    if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in values):
        raise ValueError("Clip interval and source metadata must be finite.")
    start, duration, total, fps = map(float, values)
    if start < 0 or not 1 <= duration <= min(5, float(capability["max_duration_seconds"])):
        raise ValueError("Clip interval is outside the supported 1–5 second range.")
    if total <= 0 or start + duration > total + .01 or fps <= 0:
        raise ValueError("Clip interval exceeds the source bounds.")
    if width < 2 or height < 2 or width > capability["max_width"] or height > capability["max_height"]:
        raise ValueError("Clip source dimensions exceed the advertised bounds.")
    fps = min(fps, float(capability["max_fps"]))
    frames = math.ceil(duration * fps - 1e-9)
    if frames < 1 or frames > capability["max_frames"]:
        raise ValueError("Clip would exceed the advertised frame bound.")
    return {"start_seconds": start, "duration_seconds": duration, "fps": fps,
            "expected_frames": frames, "source_width": width, "source_height": height}


def check_submission_gate(api, fields: dict, bounds: dict | None) -> dict:
    status = api.status()
    capability = require_versioned_clip_capability(status)
    jobs = api.jobs()
    assert_idle(jobs)
    if fields.get("purpose") == "preview_clip":
        if not bounds:
            raise ValueError("Clip submission requires validated source bounds.")
        validate_clip_bounds(**bounds, capability=capability)
    api.last_submission_gate = {"build_fingerprint": status.get("build_fingerprint"),
                                "capability_schema_version": 1, "jobs_before": len(jobs),
                                "active_jobs_before": []}
    return capability


def checked_submit(api, source: Path | str, fields: dict, bounds: dict | None):
    check_submission_gate(api, fields, bounds)
    return api.submit(source, fields)


def compare_decoded_frames(reference: np.ndarray, candidate: np.ndarray, device: str) -> dict:
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or not a.size:
        return {"passed": False, "reason": "decoded frame shape mismatch or empty output",
                "reference_shape": list(a.shape), "candidate_shape": list(b.shape)}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    maximum, mean = int(diff.max(initial=0)), float(diff.mean())
    if device == "cpu":
        passed, policy = maximum == 0, {"max_abs_difference": 0, "mean_abs_difference": 0.0}
    elif device == "gpu":
        passed, policy = maximum <= 1 and mean <= .001, {"max_abs_difference": 1, "mean_abs_difference": .001}
    else:
        raise ValueError("Comparison device must be cpu or gpu.")
    return {"passed": passed, "device": device, "reference_shape": list(a.shape),
            "candidate_shape": list(b.shape), "max_abs_difference": maximum,
            "mean_abs_difference": mean, "changed_values": int(np.count_nonzero(diff)),
            "value_count": int(diff.size), "policy": policy}


def cancellation_progress_reached(job: dict, minimum_frames: int = 2) -> bool:
    try:
        return job.get("state") == "processing" and int(job.get("frames_processed", 0)) >= minimum_frames
    except (TypeError, ValueError):
        return False


class LocalAPI:
    def __init__(self, base_url: str):
        self.client = httpx.Client(base_url=validate_local_api_url(base_url),
                                   timeout=httpx.Timeout(30, connect=10))

    def close(self):
        self.client.close()

    def json(self, method: str, route: str, **kwargs):
        response = self.client.request(method, route, **kwargs)
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Local API HTTP {response.status_code} at {route}: {response.text[:400]}")
        return response.json()

    def status(self): return self.json("GET", "/api/status")
    def jobs(self): return self.json("GET", "/api/jobs")
    def models(self): return self.json("GET", "/api/models")

    def resources(self):
        try:
            return self.json("GET", "/api/diagnostics/resources")
        except Exception as error:
            return {"unavailable": f"{type(error).__name__}: {error}"}

    def submit(self, source: Path | str, fields: dict):
        source = Path(source)
        if source.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Source exceeds the local fixture upload cap of 64 MiB.")
        with source.open("rb") as media:
            response = self.client.post("/api/jobs", data=fields,
                                        files={"file": (source.name, media, "video/mp4")})
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"Job POST HTTP {response.status_code}: {response.text[:400]}")
        return response.json()

    def cancel(self, job_id: str): return self.json("POST", f"/api/jobs/{job_id}/cancel")

    def route_status(self, route: str) -> int:
        with self.client.stream("GET", route) as response:
            return response.status_code

    def download(self, route: str, target: Path) -> dict:
        with self.client.stream("GET", route) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Download route HTTP {response.status_code}: {route}")
            with target.open("wb") as out:
                size = 0
                for chunk in response.iter_bytes(1024 * 1024):
                    size += len(chunk)
                    if size > 256 * 1024 * 1024:
                        raise RuntimeError("Output exceeded the 256 MiB limit.")
                    out.write(chunk)
        return {"size_bytes": size, "sha256": sha256_file(target)}


def write_report(folder: Path, report: dict):
    tmp = folder / "report.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    tmp.replace(folder / "report.json")


def settings_for(device: str, purpose: str) -> dict:
    return {"layout": "SBS", "strength": 7.5, "convergence": .5, "depth_mode": MODEL,
            "input_size": 256, "resolution": "source", "quality": "balanced", "invert_depth": False,
            "temporal_smoothing": .65, "edge_refinement": True, "execution_mode": device, "purpose": purpose}


def check_execution(execution: dict, device: str) -> dict:
    provider = execution.get("active_provider")
    valid = (provider == "DmlExecutionProvider" and execution.get("gpu_verified") is True
             and execution.get("fallback_reason") in (None, "")) if device == "gpu" else (
             provider == "CPUExecutionProvider" and execution.get("gpu_verified") is False
             and execution.get("fallback_reason") in (None, ""))
    if not valid:
        raise RuntimeError(f"Requested {device} execution was not verified without fallback: {execution}")
    return execution


def wait_for_job(api: LocalAPI, job_id: str, report: dict, folder: Path,
                 label: str, cancel: bool = False) -> dict:
    deadline, saved, printed, cancel_sent = time.monotonic() + TIMEOUT, 0, 0, False
    while True:
        job = next((j for j in api.jobs() if isinstance(j, dict) and j.get("id") == job_id), None)
        if job is None:
            raise RuntimeError(f"Job {job_id} disappeared from /api/jobs.")
        report["jobs"][label]["last_public_status"] = {k: job.get(k) for k in
            ("state", "phase", "progress", "frames_processed", "total_frames", "execution",
             "first_execution", "clip", "timings", "error", "encoder", "pipeline")}
        now = time.monotonic()
        if cancel and not cancel_sent and cancellation_progress_reached(job):
            report["jobs"][label]["cancel_requested_after_frames"] = job.get("frames_processed")
            api.cancel(job_id)
            cancel_sent = True
        if job.get("state") in TERMINAL:
            write_report(folder, report)
            if cancel and not cancel_sent:
                raise RuntimeError("Cancellation job became terminal before processing two frames.")
            return job
        if now > deadline:
            if not cancel_sent:
                api.cancel(job_id)
            raise TimeoutError(f"Job exceeded {TIMEOUT}s; cancellation was requested.")
        if now - saved >= 5:
            write_report(folder, report)
            saved = now
        if now - printed >= 20:
            print(f"{label}: {job.get('state')} {job.get('frames_processed')}/{job.get('total_frames')}", flush=True)
            printed = now
        time.sleep(2)


def prepared_fixture(label: str, source: Path, observation: dict, digest: str) -> PreparedFixture:
    info = {"width": observation["width"], "height": observation["height"],
            "frames": observation["decoded_frames"], "fps": observation["fps"],
            "duration_seconds": observation["duration_seconds"]}
    return PreparedFixture(label, "video", source.name, source, None, digest, digest,
                           info, True, "generated_synthetic")


def run_export(api: LocalAPI, report: dict, folder: Path, label: str, source: Path,
               observation: dict, device: str, model: dict, environment: dict, ffmpeg: str,
               output_limit_mb: int = 256) -> dict:
    fields = settings_for(device, "export")
    check_submission_gate(api, fields, None)
    work = folder / "outputs"
    work.mkdir(parents=True, exist_ok=True)
    record = {"requested_settings": fields, "pre_submit_gate": api.last_submission_gate,
              "server_resources_before": api.resources()}
    report["jobs"][label] = record
    write_report(folder, report)
    fixture = prepared_fixture(label, source, observation, sha256_file(source))
    result = _execute_run(api.client, fixture, fields, model, environment, ffmpeg, 1, TIMEOUT, 2,
                          work, sample_resources=False)
    record.update(result)
    record["pre_submit_gate"] = api.last_submission_gate
    record["server_resources_after"] = api.resources()
    job = next((j for j in api.jobs() if isinstance(j, dict) and j.get("id") == result.get("job_id")), {})
    record["public_job"] = job
    if job:
        if job.get("purpose") != "export" or job.get("execution_mode") != device:
            raise RuntimeError(f"Ordinary export purpose/device mismatch: {job}")
        if result.get("success"):
            record["actual_execution"] = check_execution(job.get("execution", {}), device)
    output = work / f"{label}-run-1-output.mp4"
    if output.exists():
        record["output_path"], record["output_sha256"] = str(output), sha256_file(output)
        if output.stat().st_size > output_limit_mb * 1024 * 1024:
            raise RuntimeError(f"Output exceeded {output_limit_mb} MiB bound.")
    write_report(folder, report)
    if not result.get("success"):
        raise RuntimeError(f"Export {label} failed: {result.get('error')}")
    return record


def run_clip(api: LocalAPI, report: dict, folder: Path, label: str, source: Path,
             fields: dict, bounds: dict, expected: dict, ffprobe: str, cancel: bool = False) -> dict:
    record = {"requested_settings": fields, "server_resources_before": api.resources()}
    report["jobs"][label] = record
    write_report(folder, report)
    created = checked_submit(api, source, fields, bounds)
    if created.get("purpose") != "preview_clip" or created.get("execution_mode") != fields["execution_mode"]:
        raise RuntimeError(f"Processed clip purpose/device mismatch: {created}")
    job_id = created.get("id")
    if not job_id:
        raise RuntimeError("Clip POST did not return a job id.")
    record.update({"job_id": job_id, "submitted_job": created, "pre_submit_gate": api.last_submission_gate})
    write_report(folder, report)
    job = wait_for_job(api, job_id, report, folder, label, cancel)
    record.update({"terminal_job": job, "job_timings": job_timing_snapshot(job),
                   "server_resources_after": api.resources()})
    if cancel:
        if job.get("state") != "canceled":
            raise RuntimeError(f"Inference cancellation ended {job.get('state')} instead of canceled.")
        record["actual_execution"] = check_execution(job.get("execution", {}), fields["execution_mode"])
        preview = api.route_status(f"/api/preview-clips/{job_id}/video")
        download = api.route_status(f"/api/jobs/{job_id}/download")
        record["cancel_verification"] = {"cancel_requested_after_frames": record.get("cancel_requested_after_frames"),
            "state": "canceled", "preview_route_http": preview, "download_route_http": download,
            "both_routes_404": preview == download == 404, "api_checks_prove_encoder_exit": False}
        if not record["cancel_verification"]["both_routes_404"]:
            raise RuntimeError(f"Canceled clip routes did not both return 404: {record['cancel_verification']}")
    elif job.get("state") == "completed":
        record["actual_execution"] = check_execution(job.get("execution", {}), fields["execution_mode"])
        route = f"/api/preview-clips/{job_id}/video"
        record["preview_route"] = {"http": api.route_status(route)}
        if record["preview_route"]["http"] != 200:
            raise RuntimeError("Completed processed clip route did not return HTTP 200.")
        target = folder / "outputs" / f"{label}.mp4"
        record["download"] = {"route": route, **api.download(route, target)}
        record["output_path"] = str(target)
        record["output_validation"] = inspect_video_output(target, ffprobe)
        from benchmarking import verify_video_observation
        record["verification"] = verify_video_observation(record["output_validation"], expected)
        if record["verification"]["status"] != "passed":
            raise RuntimeError(f"Processed clip decoded output failed validation: {record['verification']}")
    else:
        raise RuntimeError(f"Processed clip ended {job.get('state')}: {job.get('error') or job.get('message')}")
    write_report(folder, report)
    return record


def cancel_extraction(ffmpeg: str, source: Path, folder: Path) -> dict:
    target = folder / "inputs" / "extract-cancel-partial.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    before, stop, registered, proc, result = sha256_file(source), threading.Event(), threading.Event(), [], []

    def observe(process):
        if process is not None and not proc:
            proc.append(process)
            stop.set()
            registered.set()

    def run():
        try:
            prepare_preview_clip(ffmpeg, source, target, 9, 3, 12, stop, on_process=observe, timeout_seconds=30)
            result.append(RuntimeError("Extraction unexpectedly completed."))
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=run, daemon=True, name="verify-owned-extraction-cancel")
    thread.start()
    if not registered.wait(10):
        raise RuntimeError("Owned FFmpeg process did not register within 10 seconds.")
    thread.join(10)
    if thread.is_alive():
        stop.set()
        thread.join(5)
    process = proc[0]
    code = process.wait(timeout=5)
    data = {"exception": type(result[0]).__name__, "message": str(result[0]), "pid": process.pid,
            "return_code": code, "process_exited": process.poll() is not None,
            "partial_exists": target.exists(), "source_preserved": source.is_file(),
            "source_hash_before": before, "source_hash_after": sha256_file(source)}
    data["passed"] = (isinstance(result[0], ClipCancelled) and not thread.is_alive()
                      and data["process_exited"] and not data["partial_exists"]
                      and data["source_preserved"] and before == data["source_hash_after"])
    if not data["passed"]:
        raise RuntimeError(f"Owned extraction cancellation failed: {data}")
    return data


def run_cancel_inference(api: LocalAPI, report: dict, folder: Path, source: Path,
                         fields: dict, bounds: dict) -> dict:
    return run_clip(api, report, folder, "processed-clip-cancellation", source,
                    fields, bounds, {}, "", cancel=True)


def make_folder(output_root: str | None) -> Path:
    allowed = (ROOT / ".runtime/clip-verification").resolve()
    parent = Path(output_root).resolve() if output_root else allowed
    if not parent.is_relative_to(allowed):
        raise ValueError(f"Output root must remain under {allowed}.")
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    return folder


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--output-root")
    return parser


def run(args) -> int:
    folder = make_folder(args.output_root)
    report = {"schema_version": 1, "run_id": folder.name, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv], "api_origin": args.base_url, "device_requested": args.device,
        "status": "running", "fixture": {"id": "stress-pan-thin-lowlight-cuts-audio-30s", "tags": TAGS,
        "classification": "Generated synthetic fixture; not licensed or representative live footage.",
        "source_path": str(SOURCE)}, "jobs": {}, "measurement_limits": [
        "Synthetic motion, thin detail, low light, pan, cuts, and audio do not establish live-footage quality or multi-hour stability.",
        "Matched export uses the helper-created proxy; original source pixels are not claimed to match proxy pixels.",
        "API cancellation and 404 routes do not prove its encoder process exited; local extraction Popen exit is observed separately."]}
    write_report(folder, report)
    api = LocalAPI(args.base_url)
    try:
        status = api.status()
        capability = require_versioned_clip_capability(status)
        jobs = api.jobs()
        assert_idle(jobs)
        model = _model_record(api.models(), MODEL)
        runtime = status.get("runtime", {})
        if model.get("state") != "ready" or not model.get("hash"):
            raise RuntimeError(f"Installed Small model is not ready/hashed; no weights were downloaded: {model}")
        if args.device == "gpu" and ("DmlExecutionProvider" not in runtime.get("providers", [])
                                      or runtime.get("gpu_available") is not True):
            raise RuntimeError(f"Explicit GPU unavailable; CPU fallback is disabled: {runtime}")
        ffmpeg = status.get("ffmpeg_path")
        ffprobe = _find_ffprobe(ffmpeg)
        if not SOURCE.is_file() or not ffprobe:
            raise RuntimeError("Generated fixture or FFprobe is unavailable.")
        source = inspect_video_output(SOURCE, ffprobe)
        if (source.get("decoded_frames") != 360 or (source.get("width"), source.get("height")) != (320, 180)
                or abs(float(source.get("fps") or 0) - 12) > .01 or source.get("audio_present") is not True):
            raise RuntimeError(f"Generated 30s source preflight failed: {source}")
        raw_bounds = {"start_seconds": 9, "duration_seconds": 3, "source_duration": source["duration_seconds"],
                      "source_fps": source["fps"], "width": source["width"], "height": source["height"]}
        bounds = validate_clip_bounds(**raw_bounds, capability=capability)
        environment = _environment(status, ffmpeg, sample_resources=False)
        report.update({"preflight": {"build_fingerprint": status.get("build_fingerprint"), "runtime": runtime,
            "runtime_fingerprint": environment.get("runtime_fingerprint"), "capability": capability,
            "jobs_before": jobs, "active_jobs_before": [], "source_observation": source,
            "source_sha256": sha256_file(SOURCE), "model": model, "environment": environment,
            "weights_downloaded": False}, "fixture": {**report["fixture"], "source_sha256": sha256_file(SOURCE),
            "source_duration_seconds": source["duration_seconds"], "source_fps": source["fps"],
            "source_frames": source["decoded_frames"], "audio_present": source["audio_present"]},
            "settings": {**settings_for(args.device, "export"), "model_hash": model["hash"]}})
        write_report(folder, report)
        print(f"Preflight passed: {args.device}, build={status.get('build_fingerprint')}", flush=True)

        report["extraction_cancellation"] = cancel_extraction(ffmpeg, SOURCE, folder)
        write_report(folder, report)
        run_export(api, report, folder, "full-source-export", SOURCE, source, args.device, model, environment, ffmpeg)

        proxy = folder / "inputs" / "matched-proxy.mp4"
        proxy.parent.mkdir(parents=True, exist_ok=True)
        prepare_preview_clip(ffmpeg, SOURCE, proxy, 9, 3, 12, threading.Event(), timeout_seconds=30)
        proxy_info = inspect_video_output(proxy, ffprobe)
        if (proxy_info.get("decoded_frames") != 36 or (proxy_info.get("width"), proxy_info.get("height")) != (320, 180)
                or abs(float(proxy_info.get("fps") or 0) - 12) > .01 or proxy_info.get("audio_present") is not True):
            raise RuntimeError(f"Matched proxy failed its 3s/36-frame/audio check: {proxy_info}")
        report["matched_proxy"] = {"source_sha256": sha256_file(SOURCE), "proxy_sha256": sha256_file(proxy),
            "helper": "prepare_preview_clip", "start_seconds": 9, "duration_seconds": 3,
            "observation": proxy_info, "source_pixel_identity_claimed": False}
        write_report(folder, report)

        expected = {"width": 640, "height": 180, "frames": 36, "fps": 12,
                    "duration_seconds": 3, "audio_present": True}
        clip_fields = settings_for(args.device, "preview_clip")
        clip_fields.update({"clip_start_seconds": 9, "clip_duration_seconds": 3})
        clip = run_clip(api, report, folder, "processed-clip", SOURCE, clip_fields, raw_bounds, expected, ffprobe)
        proxy_meta = clip["terminal_job"].get("clip", {}).get("proxy", {})
        matched_proxy = (proxy_meta.get("width") == proxy_info.get("width") and
            proxy_meta.get("height") == proxy_info.get("height") and
            proxy_meta.get("total_frames") == proxy_info.get("decoded_frames") and
            abs(float(proxy_meta.get("fps") or 0) - float(proxy_info.get("fps") or 0)) <= .01)
        report["matched_proxy"].update({"server_metadata": proxy_meta, "metadata_matches": matched_proxy})
        if not matched_proxy:
            raise RuntimeError(f"Server and independently prepared proxy metadata differ: {proxy_meta}")
        write_report(folder, report)

        ordinary = run_export(api, report, folder, "ordinary-export-matched-proxy", proxy,
                              proxy_info, args.device, model, environment, ffmpeg)
        ordinary_repeat = run_export(api, report, folder, "ordinary-export-matched-proxy-repeat", proxy,
                                     proxy_info, args.device, model, environment, ffmpeg)
        import cv2
        def frames(path):
            cap, out = cv2.VideoCapture(str(path)), []
            try:
                while True:
                    ok, image = cap.read()
                    if not ok: break
                    out.append(image)
                    if len(out) > 120: raise RuntimeError("Comparison exceeded 120 frames.")
            finally: cap.release()
            return np.stack(out)
        ordinary_frames = frames(Path(ordinary["output_path"]))
        repeat_frames = frames(Path(ordinary_repeat["output_path"]))
        clip_frames = frames(Path(clip["output_path"]))
        self_variation = compare_decoded_frames(ordinary_frames, repeat_frames, args.device)
        self_variation["compared_frames"] = 36
        comparison = compare_decoded_frames(clip_frames, ordinary_frames, args.device)
        comparison["compared_frames"] = 36
        repeat_comparison = compare_decoded_frames(clip_frames, repeat_frames, args.device)
        repeat_comparison["compared_frames"] = 36
        report["ordinary_export_repeat_comparison"] = self_variation
        report["matched_output_comparison"] = comparison
        report["matched_output_repeat_comparison"] = repeat_comparison
        if not self_variation["passed"]:
            report["pixel_gate_conclusion"] = {
                "status": "inconclusive",
                "reason": "Ordinary GPU export self-repeat exceeds the unchanged strict pixel policy.",
                "clip_matches_repeat_export": repeat_comparison["passed"],
            }
        elif not comparison["passed"]:
            report["pixel_gate_conclusion"] = {
                "status": "failed",
                "reason": "Processed clip differs from a stable ordinary-export reference.",
                "clip_matches_repeat_export": repeat_comparison["passed"],
            }
        else:
            report["pixel_gate_conclusion"] = {"status": "passed", "clip_matches_repeat_export": repeat_comparison["passed"]}
        write_report(folder, report)

        cancel_fields = settings_for(args.device, "preview_clip")
        cancel_fields.update({"clip_start_seconds": 9, "clip_duration_seconds": 3})
        canceled = run_cancel_inference(api, report, folder, SOURCE, cancel_fields, raw_bounds)
        if not self_variation["passed"]:
            raise RuntimeError(f"GPU pixel gate inconclusive because ordinary-export self-repeat exceeds policy: {self_variation}")
        if not comparison["passed"]:
            raise RuntimeError(f"Matched proxy comparison failed after ordinary-export self-repeat: {comparison}")
        report.update({"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {"full_source_export": "passed", "processed_clip": "passed",
            "matched_proxy_export": "passed", "matched_proxy_repeat": "passed",
            "ordinary_export_self_repeat": self_variation, "matched_comparison": comparison,
            "matched_repeat_comparison": repeat_comparison,
            "inference_cancel": canceled["cancel_verification"],
            "extraction_cancel": report["extraction_cancellation"]}})
    except Exception as error:
        report.update({"status": "failed", "failure": {"type": type(error).__name__, "message": str(error)},
                       "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Verification failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        report["report_path"] = str(folder / "report.json")
        write_report(folder, report)
        api.close()
        print(f"Report: {folder / 'report.json'}", flush=True)
    return 0 if report["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
