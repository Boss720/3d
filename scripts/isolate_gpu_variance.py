"""Local, bounded stage and encoder repeatability experiment for StereoLab."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORT_SOURCE = ROOT / ".runtime/clip-verification/20261018T113023Z-3a3457b7/report.json"
DEFAULT_SOURCE = ROOT / ".runtime/clip-verification/20261018T113023Z-3a3457b7/inputs/matched-proxy.mp4"
ALLOWED_OUTPUT = (ROOT / ".runtime/variance-isolation").resolve()
MODEL_ID = "depth-anything-v2-small"
MODEL_HASH = "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c"
SERVER_PORTS = (8768, 8770, 8772, 8774)
TERMINAL_STATES = {"completed", "failed", "canceled"}
MAX_FRAMES = 36
MAX_INPUT_BYTES_PER_FRAME = 16 * 1024 * 1024
MAX_CAPTURE_BYTES = 256 * 1024 * 1024
PROCESS_TIMEOUT_SECONDS = 60
SETTINGS = {
    "layout": "SBS", "strength": 7.5, "convergence": 0.5,
    "depth_mode": MODEL_ID, "input_size": 256, "invert_depth": False,
    "edge_refinement": True, "temporal_smoothing": 0.65,
    "execution_mode": "gpu", "fps": 12, "frames": 36,
}


class CapturingSession:
    """Delegate an ORT session while copying bounded inputs and outputs for diagnostics."""

    def __init__(self, session, capture, max_bytes: int = 16 * 1024 * 1024,
                 capture_inputs=None, max_input_bytes: int = MAX_INPUT_BYTES_PER_FRAME):
        self._session, self._capture, self._max_bytes = session, capture, max_bytes
        self._capture_inputs = capture_inputs or (lambda tensors: None)
        self._max_input_bytes = max_input_bytes

    def __getattr__(self, name):
        return getattr(self._session, name)

    def run(self, output_names, input_feed, *args, **kwargs):
        if not isinstance(input_feed, dict) or not input_feed:
            raise ValueError("ONNX input feed must be a non-empty name-to-tensor mapping.")
        input_copies, input_size = [], 0
        for name, value in input_feed.items():
            if not isinstance(name, str) or not name:
                raise ValueError("ONNX input feed names must be non-empty strings.")
            array = np.asarray(value)
            if array.dtype.kind not in "fiu":
                raise ValueError(f"ONNX input {name!r} must be a numeric tensor.")
            input_size += array.nbytes
            if input_size > self._max_input_bytes:
                raise ValueError("ONNX input feed exceeded the per-frame input byte bound.")
            if not np.isfinite(array).all():
                raise ValueError(f"ONNX input {name!r} must contain only finite values.")
            input_copies.append((name, np.array(array, copy=True, order="C")))
        # Capture the preprocessed values before ORT receives the original feed.
        self._capture_inputs(input_copies)
        outputs = self._session.run(output_names, input_feed, *args, **kwargs)
        copies, size = [], 0
        for output in outputs:
            array = np.asarray(output)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError("Raw ONNX output must be a finite numeric tensor.")
            size += array.nbytes
            if size > self._max_bytes:
                raise ValueError("Raw ONNX capture exceeded the per-frame byte bound.")
            copies.append(np.array(array, copy=True, order="C"))
        self._capture(copies)
        return outputs


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def tensor_bundle_hash(tensors: list[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for index, (name, array) in enumerate(tensors):
        digest.update(str(index).encode("ascii"))
        digest.update(name.encode("utf-8"))
        digest.update(array_hash(array).encode("ascii"))
    return digest.hexdigest()


def compare_sequences(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape:
        raise ValueError(f"Variance comparison shape mismatch: {a.shape} vs {b.shape}.")
    if a.dtype.kind not in "fiu" or b.dtype.kind not in "fiu":
        raise ValueError("Variance comparison requires numeric arrays.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Variance comparison requires finite arrays.")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    count = int(delta.size)
    frame_count = int(a.shape[0]) if a.ndim else 1
    per_frame = []
    if a.ndim:
        for index in range(frame_count):
            left, right, diff = a[index], b[index], delta[index]
            per_frame.append({
                "index": index, "reference_sha256": array_hash(left),
                "candidate_sha256": array_hash(right),
                "exact_equal": bool(np.array_equal(left, right)),
                "changed_values": int(np.count_nonzero(diff)),
                "max_abs_difference": float(diff.max(initial=0)),
                "mean_abs_difference": float(diff.mean()) if diff.size else 0.0,
            })
    return {
        "shape": list(a.shape), "reference_dtype": a.dtype.str, "candidate_dtype": b.dtype.str,
        "reference_sha256": array_hash(a), "candidate_sha256": array_hash(b),
        "exact_equal": bool(np.array_equal(a, b)), "value_count": count,
        "changed_values": int(np.count_nonzero(delta)),
        "max_abs_difference": float(delta.max(initial=0)),
        "mean_abs_difference": float(delta.mean()) if count else 0.0,
        "per_frame": per_frame,
    }


def compare_all_repeat_pairs(repeats: list[dict[str, np.ndarray]]) -> list[dict]:
    if not 2 <= len(repeats) <= 3:
        raise ValueError("Repeat comparisons require two or three captured repeats.")
    reference_stages = set(repeats[0])
    if not reference_stages:
        raise ValueError("Repeat captures must contain at least one stage.")
    comparisons = []
    for left in range(len(repeats)):
        for right in range(left + 1, len(repeats)):
            if set(repeats[left]) != reference_stages or set(repeats[right]) != reference_stages:
                raise ValueError("Repeat captures contain different stage names.")
            comparisons.append({
                "reference_repeat": left + 1,
                "candidate_repeat": right + 1,
                "stages": {name: compare_sequences(repeats[left][name], repeats[right][name])
                           for name in sorted(reference_stages)},
            })
    return comparisons


def run_warmup(manager, raw_sink: dict, input_sink: dict, frame: np.ndarray,
               source_frame_index: int, device: str, created_sessions: list) -> dict:
    """Run one separately measured presequence inference and discard tensor copies."""
    raw_calls: list[list[np.ndarray]] = []
    input_calls: list[list[tuple[str, np.ndarray]]] = []
    raw_sink["target"], input_sink["target"] = raw_calls, input_calls
    warmup_frame = np.array(frame, copy=True, order="C")
    started = time.perf_counter()
    try:
        _, execution = manager.predict_with_info(
            warmup_frame, MODEL_ID, SETTINGS["input_size"], execution_mode=device)
    finally:
        raw_sink["target"], input_sink["target"] = None, None
    elapsed = time.perf_counter() - started
    if len(raw_calls) != 1 or len(input_calls) != 1:
        raise RuntimeError("Warmup must capture exactly one preprocessed input and one raw ONNX output call.")
    expected_provider = "DmlExecutionProvider" if device == "gpu" else "CPUExecutionProvider"
    provider_verified = (execution.get("active_provider") == expected_provider
                         and execution.get("gpu_verified") is (device == "gpu")
                         and not execution.get("fallback_reason"))
    if not provider_verified:
        raise RuntimeError(f"Warmup provider was not verified without fallback: {execution}")
    if len(created_sessions) != 1:
        raise RuntimeError(f"Warmup must use exactly one owned model session, found {len(created_sessions)}.")
    return {
        "status": "completed", "requested_frames": 1, "completed_frames": 1,
        "source_frame_index": source_frame_index, "source_frame_sha256": array_hash(frame),
        "input_sha256": tensor_bundle_hash(input_calls[0]),
        "output_sha256": tensor_bundle_hash([(f"raw_onnx_output_{index}", value)
                                              for index, value in enumerate(raw_calls[0])]),
        "elapsed_seconds": elapsed, "provider_verified": True, "execution": execution,
        "session_count": len(created_sessions), "captured_input_discarded": True,
        "raw_output_discarded": True,
    }


def validate_frames(frames: list[np.ndarray], expected: int, width: int, height: int) -> tuple[int, int, int]:
    if not 1 <= expected <= MAX_FRAMES or len(frames) != expected:
        raise ValueError(f"Fixture must contain exactly the bounded {expected}-frame sequence.")
    for frame in frames:
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.shape != (height, width, 3):
            raise ValueError("Fixture frames must be matching uint8 BGR images.")
    return len(frames), width, height


def assert_server_jobs_idle(jobs_by_port: dict[int, list[dict]]) -> None:
    busy = []
    for port, jobs in jobs_by_port.items():
        for job in jobs:
            if job.get("state") not in TERMINAL_STATES:
                busy.append({"port": port, "id": job.get("id"), "state": job.get("state"),
                             "purpose": job.get("purpose")})
    if busy:
        raise RuntimeError(f"Refusing diagnostic work while local API work is active: {busy}")


def assert_no_new_server_jobs(baseline: dict, current: dict) -> None:
    added = []
    for port in SERVER_PORTS:
        previous = set(baseline.get(str(port), []))
        jobs = current.get(port, {}).get("jobs", [])
        added.extend({"port": port, "id": job.get("id"), "state": job.get("state")}
                     for job in jobs if job.get("id") not in previous)
    if added:
        raise RuntimeError(f"Another local API job appeared during the diagnostic: {added}")


def server_snapshot(timeout: float = 3) -> dict:
    snapshots, by_port = {}, {}
    for port in SERVER_PORTS:
        base = f"http://127.0.0.1:{port}"
        with urlopen(base + "/api/status", timeout=timeout) as response:
            status = json.loads(response.read())
        with urlopen(base + "/api/jobs", timeout=timeout) as response:
            jobs = json.loads(response.read())
        if not isinstance(status, dict) or not isinstance(jobs, list):
            raise RuntimeError(f"Local API {port} returned invalid status or jobs data.")
        snapshots[port] = {"build_fingerprint": status.get("build_fingerprint"),
                           "runtime": status.get("runtime"), "ffmpeg_path": status.get("ffmpeg_path"), "active_jobs": [
                               {"id": j.get("id"), "state": j.get("state"), "purpose": j.get("purpose")}
                               for j in jobs if j.get("state") not in TERMINAL_STATES],
                           "jobs": [{"id": j.get("id"), "state": j.get("state"), "purpose": j.get("purpose")}
                                    for j in jobs]}
        by_port[port] = jobs
    assert_server_jobs_idle(by_port)
    snapshots["ffmpeg_path"] = snapshots[8772].get("ffmpeg_path")
    return snapshots


def activity_checkpoint(report: dict, folder: Path, phase: str) -> dict:
    snapshot = server_snapshot()
    if report.get("server_job_baseline"):
        assert_no_new_server_jobs(report["server_job_baseline"], snapshot)
    report.setdefault("activity_checks", []).append({"phase": phase, **snapshot})
    write_report(folder, report)
    return snapshot


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(folder: Path, report: dict) -> None:
    temp = folder / "report.json.tmp"
    temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temp.replace(folder / "report.json")


def output_folder(output_root: str | None) -> Path:
    root = Path(output_root).resolve() if output_root else ALLOWED_OUTPUT
    if not root.is_relative_to(ALLOWED_OUTPUT):
        raise ValueError(f"Output root must stay under {ALLOWED_OUTPUT}.")
    root.mkdir(parents=True, exist_ok=True)
    folder = root / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    folder.mkdir()
    (folder / "captures").mkdir()
    (folder / "encodes").mkdir()
    return folder


def decode_proxy(source: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not decode the matched proxy: {source}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) > MAX_FRAMES:
                raise ValueError("Proxy exceeded the 36-frame diagnostic bound.")
    finally:
        capture.release()
    validate_frames(frames, 36, 320, 180)
    return frames


def source_fingerprint() -> str:
    names = ("depth_engine.py", "stereo.py", "video_encoder.py", "preview_clip.py",
             "video_pipeline.py", "scripts/isolate_gpu_variance.py")
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def capture_repeat(manager, raw_sink: dict, input_sink: dict, frames: list[np.ndarray],
                   repeat_index: int, device: str) -> tuple[dict[str, np.ndarray], dict]:
    from stereo import TemporalDepthFilter, refine_depth, render_output

    raw_calls: list[list[np.ndarray]] = []
    input_calls: list[list[tuple[str, np.ndarray]]] = []
    raw_sink["target"] = raw_calls
    input_sink["target"] = input_calls
    raw_by_output: list[list[np.ndarray]] = []
    input_by_index: list[list[np.ndarray]] = []
    input_specs: list[tuple[str, tuple[int, ...], str]] = []
    normalized, refined, temporal, rendered, executions = [], [], [], [], []
    depth_filter = TemporalDepthFilter(SETTINGS["temporal_smoothing"])
    durations = {"inference": 0.0, "refinement": 0.0, "temporal": 0.0, "render": 0.0}
    started = time.perf_counter()
    captured_bytes = 0
    try:
        for frame in frames:
            before = time.perf_counter()
            depth, execution = manager.predict_with_info(
                frame, MODEL_ID, SETTINGS["input_size"], execution_mode=device)
            durations["inference"] += time.perf_counter() - before
            expected_provider = "DmlExecutionProvider" if device == "gpu" else "CPUExecutionProvider"
            if (execution.get("active_provider") != expected_provider
                    or execution.get("gpu_verified") is not (device == "gpu")
                    or execution.get("fallback_reason")):
                raise RuntimeError(f"Strict GPU execution was not verified: {execution}")
            if len(raw_calls) != len(normalized) + 1 or len(input_calls) != len(normalized) + 1:
                raise RuntimeError("The diagnostic wrapper did not capture exactly one input/output call per frame.")
            raw_outputs, feed_tensors = raw_calls[-1], input_calls[-1]
            if not raw_by_output:
                raw_by_output = [[] for _ in raw_outputs]
            if len(raw_outputs) != len(raw_by_output):
                raise RuntimeError("ONNX output count changed during the repeat.")
            current_input_specs = [(name, tuple(array.shape), array.dtype.str) for name, array in feed_tensors]
            if not input_specs:
                input_specs = current_input_specs
                input_by_index = [[] for _ in input_specs]
            elif current_input_specs != input_specs:
                raise RuntimeError("ONNX input names, shapes, order, or dtypes changed during the repeat.")

            normalized_value = np.asarray(depth, dtype=np.float32).copy()
            executions.append(execution)
            before = time.perf_counter()
            refined_value = refine_depth(frame, depth, SETTINGS["edge_refinement"]).copy()
            durations["refinement"] += time.perf_counter() - before
            before = time.perf_counter()
            temporal_value = depth_filter.apply(frame, refined_value).copy()
            durations["temporal"] += time.perf_counter() - before
            before = time.perf_counter()
            rendered_value = render_output(
                frame, temporal_value, SETTINGS["strength"], SETTINGS["convergence"], SETTINGS["layout"]).copy()
            durations["render"] += time.perf_counter() - before

            frame_size = (sum(array.nbytes for _, array in feed_tensors)
                          + sum(array.nbytes for array in raw_outputs)
                          + normalized_value.nbytes + refined_value.nbytes
                          + temporal_value.nbytes + rendered_value.nbytes)
            captured_bytes += frame_size
            if captured_bytes > MAX_CAPTURE_BYTES:
                raise ValueError(f"Captured stage arrays exceed {MAX_CAPTURE_BYTES} bytes.")
            for values, output in zip(raw_by_output, raw_outputs):
                values.append(output)
            for values, (_, input_value) in zip(input_by_index, feed_tensors):
                values.append(input_value)
            normalized.append(normalized_value)
            refined.append(refined_value)
            temporal.append(temporal_value)
            rendered.append(rendered_value)
    finally:
        raw_sink["target"], input_sink["target"] = None, None

    stages = {f"preprocessed_input_{index}": np.stack(values)
              for index, values in enumerate(input_by_index)}
    stages.update({f"raw_onnx_output_{i}": np.stack(items) for i, items in enumerate(raw_by_output)})
    stages.update({"normalized_model_depth": np.stack(normalized), "refined_depth": np.stack(refined),
                   "temporal_depth": np.stack(temporal), "rendered_bgr": np.stack(rendered)})
    size = sum(a.nbytes for a in stages.values())
    if size > MAX_CAPTURE_BYTES:
        raise ValueError(f"Captured stage arrays exceed {MAX_CAPTURE_BYTES} bytes.")
    for name, array in stages.items():
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Stage {name} contains a non-finite or nonnumeric value.")
    input_tensors = [{"stage": f"preprocessed_input_{index}", "name": name,
                      "tensor_shape": list(shape)}
                     for index, (name, shape, _) in enumerate(input_specs)]
    return stages, {"wall_seconds": time.perf_counter() - started, "stage_seconds": durations,
                    "execution_first": executions[0], "execution_last": executions[-1],
                    "executions": executions, "fresh_temporal_history": True,
                    "first_temporal_equals_refined": bool(np.array_equal(stages["temporal_depth"][0], stages["refined_depth"][0])),
                    "captured_bytes": size, "input_tensors": input_tensors}


def save_stages(folder: Path, repeat: int, stages: dict[str, np.ndarray],
                input_tensors: list[dict] | None = None) -> dict:
    final = folder / "captures" / f"repeat-{repeat}.npz"
    partial = folder / "captures" / f"repeat-{repeat}.partial.npz"
    try:
        np.savez_compressed(partial, **stages)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if partial.stat().st_size > MAX_CAPTURE_BYTES:
        partial.unlink(missing_ok=True)
        raise ValueError("Compressed stage capture exceeded the artifact byte bound.")
    partial.replace(final)
    stage_metadata = {name: {"shape": list(array.shape), "dtype": array.dtype.str,
                             "nbytes": array.nbytes, "sha256": array_hash(array)}
                      for name, array in stages.items()}
    capture = {"path": str(final), "sha256": sha256_file(final), "size_bytes": final.stat().st_size,
               "stages": stage_metadata}
    if input_tensors:
        capture["input_tensors"] = [
            {**item, **stage_metadata[item["stage"]]}
            for item in input_tensors]
    return capture


def read_video_frames(path: Path) -> np.ndarray:
    cap, frames = cv2.VideoCapture(str(path)), []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Encoder output cannot be decoded: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) > MAX_FRAMES:
                raise ValueError("Encoded output exceeded the frame bound.")
    finally:
        cap.release()
    validate_frames(frames, 36, 640, 180)
    return np.stack(frames)


def encode_once(ffmpeg: str, frames: np.ndarray, args: tuple[str, ...], target: Path) -> dict:
    if target.exists():
        raise FileExistsError(f"Refusing to replace preexisting output: {target}")
    raw = np.ascontiguousarray(frames, dtype=np.uint8).tobytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "640x180",
               "-framerate", "12", "-i", "pipe:0", "-frames:v", "36", "-an",
               *args, "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    owned = False
    try:
        result = subprocess.run(command, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=PROCESS_TIMEOUT_SECONDS, check=False)
        owned = target.exists()
        if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
            detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise RuntimeError(f"FFmpeg encoding failed ({result.returncode}): {detail}")
        decoded = read_video_frames(target)
        return {"path": str(target), "command": command, "input_frames_sha256": raw_hash,
                "output_sha256": sha256_file(target), "size_bytes": target.stat().st_size,
                "decoded_frames": len(decoded), "decoded_sha256": array_hash(decoded),
                "decoded": decoded}
    except BaseException:
        if owned or target.exists():
            target.unlink(missing_ok=True)
        raise


def encoder_experiment(folder: Path, ffmpeg: str, rendered: np.ndarray, device: str,
                       report_state: dict) -> dict:
    from video_encoder import EncoderProfile, _QUALITY_PROFILES, select_encoder_profile

    activity_checkpoint(report_state, folder,
                        "encoder-capability-probe" if device == "gpu" else "encoder-setup-software-only")
    if device == "gpu":
        probe_started = time.perf_counter()
        profile = select_encoder_profile(ffmpeg, "fast")
        probe_seconds = time.perf_counter() - probe_started
        probe = {"status": "completed", "elapsed_seconds": probe_seconds}
    else:
        software_args = ("-c:v", "libx264", *_QUALITY_PROFILES["fast"]["x264"])
        profile = EncoderProfile("fast", "libx264", software_args, software_args, False,
                                 "Hardware encoder was not probed in explicit CPU mode.")
        probe_seconds = None
        probe = {"status": "skipped", "reason": "Explicit CPU mode does not probe hardware encoders."}
    report = {"fixed_input_sha256": array_hash(rendered), "quality_profile": "fast",
              "capability_probe": probe, "capability_probe_seconds": probe_seconds,
              "hardware_probe_performed": device == "gpu",
              "profile": {"codec": profile.codec, "hardware_available": profile.hardware_available,
                          "video_args": profile.video_args, "software_fallback_args": profile.software_fallback_args,
                          "probe_message": profile.probe_message}, "encoders": {}}
    variants = {"libx264": tuple(profile.software_fallback_args)}
    if device == "gpu" and profile.hardware_available and profile.codec == "h264_nvenc":
        variants["h264_nvenc"] = tuple(profile.video_args)
    elif device != "gpu":
        report["nvenc_skipped_reason"] = "Explicit GPU mode was not requested."
    else:
        report["nvenc_skipped_reason"] = "Current FFmpeg/driver fast-profile probe did not support NVENC."

    for encoder, args in variants.items():
        pair, sequences = [], []
        for index in (1, 2):
            activity_checkpoint(report_state, folder, f"encode-{encoder}-{index}")
            target = folder / "encodes" / f"{encoder}-{index}.mp4"
            result = encode_once(ffmpeg, rendered, args, target)
            sequences.append(result.pop("decoded"))
            pair.append(result)
            report["encoders"][encoder] = {"runs": pair}
            write_report(folder, {**_active_report, "encoder_experiment": report})
        report["encoders"][encoder]["decoded_repeat_comparison"] = compare_sequences(sequences[0], sequences[1])
        report["encoders"][encoder]["comparison_scope"] = "same fixed BGR frame sequence, same encoder, no cross-codec equality claim"
    return report


_active_report: dict = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup-frames", choices=(0, 1), type=int, default=0,
                        help="Run zero or one provider-verified inference before the measured repeats.")
    parser.add_argument("--output-root")
    return parser


def run(args) -> int:
    folder = output_folder(args.output_root)
    global _active_report
    report = _active_report = {
        "schema_version": 1, "run_id": folder.name, "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "command": [sys.executable, *sys.argv],
        "device_requested": args.device, "source_path": str(DEFAULT_SOURCE), "model_id": MODEL_ID,
        "model_hash_expected": MODEL_HASH, "settings": {**SETTINGS, "execution_mode": args.device},
        "warmup_frames": args.warmup_frames,
        "warmup_accounting": {"status": "pending" if args.warmup_frames else "disabled",
                              "requested_frames": args.warmup_frames, "completed_frames": 0,
                              "provider_verified": None, "captured_input_discarded": True,
                              "raw_output_discarded": True},
        "repeats_max": 3, "repeats_completed": [], "measurements": {},
        "limits": ["Only this synthetic 3s proxy was exercised; no live-footage or long-export claim.",
                   "DirectML may include CPU graph nodes; verified provider evidence is per inference.",
                   "Encoder comparisons are separate per codec and use one fixed rendered BGR sequence."],
    }
    write_report(folder, report)
    try:
        status = server_snapshot()
        report["activity_checks"] = [{"phase": "preflight", **status}]
        report["server_job_baseline"] = {str(port): [j.get("id") for j in status[port]["jobs"]]
                                          for port in SERVER_PORTS}
        if not DEFAULT_SOURCE.is_file() or not REPORT_SOURCE.is_file():
            raise FileNotFoundError("The prior matched proxy/report is unavailable.")
        prior = json.loads(REPORT_SOURCE.read_text(encoding="utf-8"))
        source_digest = sha256_file(DEFAULT_SOURCE)
        expected_digest = prior.get("matched_proxy", {}).get("proxy_sha256")
        if not expected_digest or source_digest != expected_digest:
            raise ValueError("Matched proxy SHA-256 does not match its prior verification report.")
        if prior.get("fixture", {}).get("source_sha256") != "1ec864cc2172cc935f2f0c6ec8d2ef288f6b8fc66ae1ef5fbd652fb976680245":
            raise ValueError("Prior generated source identity differs from the frozen fixture.")
        ffmpeg = status.get("ffmpeg_path")
        if not ffmpeg or not Path(ffmpeg).is_file():
            raise RuntimeError("The current API FFmpeg binary is unavailable.")
        from benchmarking import _find_ffprobe, inspect_video_output
        media = inspect_video_output(DEFAULT_SOURCE, _find_ffprobe(ffmpeg))
        if (media.get("width"), media.get("height"), media.get("decoded_frames")) != (320, 180, 36):
            raise ValueError(f"Matched proxy decoded geometry is invalid: {media}")
        if abs(float(media.get("fps") or 0) - 12) > .01 or media.get("audio_present") is not True:
            raise ValueError(f"Matched proxy FPS/audio is invalid: {media}")
        frames = decode_proxy(DEFAULT_SOURCE)
        frame_hashes = [array_hash(frame) for frame in frames]
        report.update({"source": {"sha256": source_digest, "original_source_sha256": prior["fixture"]["source_sha256"],
            "frame_sequence_sha256": array_hash(np.stack(frames)), "per_frame_sha256": frame_hashes,
            "observation": media, "source_pixel_identity_claimed": False},
            "prior_report": str(REPORT_SOURCE), "api_servers_initial": status,
            "ffmpeg_path": ffmpeg, "api_activity_checks": report["activity_checks"]})

        from depth_engine import MODEL_SPECS, ModelManager
        import onnxruntime as ort
        model_spec = MODEL_SPECS[MODEL_ID]
        model_path = ROOT / ".runtime/models" / MODEL_ID / "model.onnx"
        if not model_path.is_file() or sha256_file(model_path) != MODEL_HASH:
            raise RuntimeError("Pinned Small model is unavailable or its hash differs; no download is attempted.")
        manager = ModelManager(ROOT / ".runtime/models")
        manager.validate(MODEL_ID, execution_mode=args.device)
        runtime = manager.runtime_info()
        if args.device == "gpu" and ("DmlExecutionProvider" not in runtime.get("providers", []) or not runtime.get("gpu_available")):
            raise RuntimeError(f"DirectML is unavailable; explicit GPU mode will not fall back: {runtime}")
        raw_sink, input_sink = {"target": None}, {"target": None}
        created_sessions = []
        original_new_session = manager._new_session

        def capture_run(outputs):
            target = raw_sink.get("target")
            if target is None:
                raise RuntimeError("Raw ONNX call occurred outside a bounded repeat capture.")
            target.append(outputs)

        def capture_inputs(inputs):
            target = input_sink.get("target")
            if target is None:
                raise RuntimeError("Preprocessed ONNX input capture occurred outside a bounded capture phase.")
            target.append(inputs)

        def new_session(path, provider="CPUExecutionProvider", device_id=None):
            session = original_new_session(path, provider=provider, device_id=device_id)
            wrapper = CapturingSession(session, capture_run, capture_inputs=capture_inputs)
            created_sessions.append(wrapper)
            return wrapper

        manager._new_session = new_session
        env = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
               "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
               "opencv": cv2.__version__, "hardware": runtime.get("gpu_device"),
               "runtime_build_fingerprint": source_fingerprint(), "ffmpeg_path": ffmpeg}
        env["runtime_fingerprint"] = hashlib.sha256(json.dumps(env, sort_keys=True, default=str).encode()).hexdigest()
        report.update({"runtime": env, "model": {"id": MODEL_ID, "sha256": MODEL_HASH,
            "revision": model_spec["revision"], "path": str(model_path), "weights_downloaded": False,
            "input_size": SETTINGS["input_size"]}, "gpu_verified": False})
        write_report(folder, report)

        if args.warmup_frames:
            activity_checkpoint(report, folder, "inference-warmup")
            report["warmup_accounting"] = run_warmup(
                manager, raw_sink, input_sink, frames[0], 0, args.device, created_sessions)
            write_report(folder, report)

        stage_records, stage_arrays = {}, []
        for repeat in range(1, 4):
            activity_checkpoint(report, folder, f"inference-repeat-{repeat}")
            stages, timing = capture_repeat(manager, raw_sink, input_sink, frames, repeat, args.device)
            if not created_sessions:
                raise RuntimeError("ModelManager did not create an owned session.")
            session_id = id(created_sessions[0])
            if any(id(session) != session_id for session in created_sessions):
                raise RuntimeError("ModelManager replaced the session during the repeat sequence.")
            if len(created_sessions) != 1:
                raise RuntimeError("The diagnostic used more than one model session.")
            capture = save_stages(folder, repeat, stages, timing["input_tensors"])
            record = {"repeat": repeat, "source_sha256": source_digest,
                      "frame_sequence_sha256": array_hash(np.stack(frames)), "settings": dict(report["settings"]),
                      "warmup_presequence": {
                          "requested_frames": report["warmup_accounting"]["requested_frames"],
                          "completed_frames": report["warmup_accounting"]["completed_frames"],
                          "source_frame_sha256": report["warmup_accounting"].get("source_frame_sha256")},
                      "timing": {k: v for k, v in timing.items()
                                 if k not in ("executions", "input_tensors")},
                      "execution_first": timing["execution_first"], "execution_last": timing["execution_last"],
                      "provider_verified_all_frames": all(e.get("active_provider") ==
                          ("DmlExecutionProvider" if args.device == "gpu" else "CPUExecutionProvider") and
                          e.get("gpu_verified") is (args.device == "gpu") and not e.get("fallback_reason")
                          for e in timing["executions"]),
                      "session_count": len(created_sessions), "session_reused_frames": sum(bool(e.get("session_reused")) for e in timing["executions"]),
                      "capture": capture}
            if record["provider_verified_all_frames"] is not True:
                raise RuntimeError("At least one repeat did not verify DML execution without fallback.")
            report["gpu_verified"] = args.device == "gpu"
            if stage_arrays:
                record["stage_comparisons_to_repeat_1"] = {
                    name: compare_sequences(stage_arrays[0][name], stages[name]) for name in stages}
            stage_records[repeat] = record
            stage_arrays.append(stages)
            report["repeats_completed"].append(record)
            report["stage_comparison_summary"] = {
                str(index + 1): stage_records[index + 1].get("stage_comparisons_to_repeat_1", {})
                for index in range(1, repeat)}
            write_report(folder, report)

        report["stage_pair_comparisons"] = compare_all_repeat_pairs(stage_arrays)
        report["stage_comparison_summary"] = {
            str(pair["candidate_repeat"]): pair["stages"]
            for pair in report["stage_pair_comparisons"] if pair["reference_repeat"] == 1}
        write_report(folder, report)

        encoders = encoder_experiment(folder, ffmpeg, stage_arrays[0]["rendered_bgr"], args.device, report)
        report["encoder_experiment"] = encoders
        report["status"] = "completed_diagnostic"
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    except Exception as error:
        report.update({"status": "failed", "error": {"type": type(error).__name__, "message": str(error)},
                       "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        print(f"Diagnostic failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        report["report_path"] = str(folder / "report.json")
        write_report(folder, report)
        print(f"Report: {folder / 'report.json'}", flush=True)
    return 0 if report["status"] == "completed_diagnostic" else 1


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
