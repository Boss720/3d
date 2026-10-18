"""Independently validate and compare bounded GPU variance capture reports."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import uuid
import zipfile
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_CAPTURE_BYTES = 256 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_REPEATS = 3
MAX_FRAMES = 36
MAX_NPZ_MEMBERS = 64
MAX_ENCODE_TIMEOUT_SECONDS = 60
SERVER_PORTS = (8768, 8770, 8772, 8774)
TERMINAL_STATES = {"completed", "failed", "canceled"}
CAPTURE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.npz$")
FAST_VIDEO_ARGS = ("-c:v", "h264_nvenc", "-preset", "p1", "-tune", "hq",
                   "-rc", "vbr", "-cq", "30", "-b:v", "0")
FAST_X264_ARGS = ("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22")


def array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _no_symlink_path(path: Path) -> None:
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Capture path contains a symlink: {cursor}")


def _resolve_capture(run_dir: Path, raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("Capture path is missing.")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    _no_symlink_path(candidate.absolute())
    resolved_run = run_dir.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        rel = resolved.relative_to(resolved_run)
    except ValueError as exc:
        raise ValueError("Capture file is not confined to this report run.") from exc
    if len(rel.parts) != 2 or rel.parts[0].casefold() != "captures" or not CAPTURE_NAME.fullmatch(rel.name):
        raise ValueError("Capture file must be a named NPZ directly under this run's captures folder.")
    if not resolved.is_file():
        raise ValueError("Capture path is not a regular file.")
    return resolved


def _hash_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{label} must be a SHA-256 hex digest.")
    return value.lower()


def _inspect_npz(path: Path) -> list[zipfile.ZipInfo]:
    if path.stat().st_size > MAX_CAPTURE_BYTES:
        raise ValueError("Capture compressed file exceeded the 256 MiB bound.")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Capture is not a valid NPZ archive: {path.name}") from exc
    if not infos or len(infos) > MAX_NPZ_MEMBERS:
        raise ValueError("Capture archive has no arrays or too many members.")
    names = set()
    expanded = 0
    for info in infos:
        name = info.filename
        if name in names or not name.endswith(".npy") or Path(name).name != name:
            raise ValueError("Capture archive has duplicate or unsafe member names.")
        names.add(name)
        expanded += info.file_size
        if info.file_size > MAX_CAPTURE_BYTES or expanded > MAX_CAPTURE_BYTES:
            raise ValueError("Capture archive expansion exceeds the 256 MiB bound.")
    return infos


def _preflight_npy_headers(path: Path, infos: list[zipfile.ZipInfo]) -> None:
    """Check declared NPY allocation sizes before np.load can allocate arrays."""
    try:
        with zipfile.ZipFile(path) as archive:
            for info in infos:
                with archive.open(info, "r") as stream:
                    version = np.lib.format.read_magic(stream)
                    header_reader = getattr(np.lib.format, "_read_array_header", None)
                    if header_reader is not None:
                        shape, _fortran_order, dtype = header_reader(stream, version, max_header_size=10_000)
                    elif version == (1, 0):
                        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=10_000)
                    elif version in {(2, 0), (3, 0)}:
                        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=10_000)
                    else:
                        raise ValueError(f"Unsupported NPY format version {version}.")
                    if len(shape) > 16 or any(type(dimension) is not int or dimension < 0 for dimension in shape):
                        raise ValueError("Capture array header has invalid dimensions.")
                    if dtype.hasobject or dtype.kind not in "fiu" or dtype.itemsize <= 0:
                        raise ValueError("Capture array header declares a nonnumeric dtype.")
                    count = math.prod(shape)
                    payload_bytes = count * dtype.itemsize
                    header_bytes = stream.tell()
                    if payload_bytes > MAX_CAPTURE_BYTES or payload_bytes != info.file_size - header_bytes:
                        raise ValueError("Capture NPY header declares an oversized or truncated array.")
    except (OSError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError("Could not safely preflight a capture NPY header.") from exc


def _metadata_map(capture: dict) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    stages = capture.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("Capture stages metadata is invalid.")
    for name, record in stages.items():
        if not isinstance(name, str) or not isinstance(record, dict):
            raise ValueError("Capture stage metadata entry is invalid.")
        metadata[name] = record
    input_tensors = capture.get("input_tensors", [])
    if input_tensors is None:
        input_tensors = []
    if not isinstance(input_tensors, list):
        raise ValueError("Capture input tensor metadata is invalid.")
    for record in input_tensors:
        if not isinstance(record, dict) or not isinstance(record.get("stage"), str):
            raise ValueError("Capture input tensor metadata entry is invalid.")
        name = record["stage"]
        if name in metadata:
            existing = metadata[name]
            fields = ("shape", "dtype", "nbytes", "sha256")
            if any(existing.get(field) != record.get(field) for field in fields):
                raise ValueError(f"Capture has inconsistent duplicate metadata for {name}.")
            metadata[name] = {**existing, "name": record.get("name"),
                              "tensor_shape": record.get("tensor_shape")}
        else:
            metadata[name] = record
    return metadata


def _validate_stage_array(name: str, array: np.ndarray, meta: dict) -> None:
    if array.dtype.hasobject or array.dtype.kind not in "fiu":
        raise ValueError(f"Stage {name} must use a non-object numeric dtype.")
    if array.nbytes <= 0 or array.nbytes > MAX_CAPTURE_BYTES:
        raise ValueError(f"Stage {name} exceeds the byte bound.")
    if not np.isfinite(array).all():
        raise ValueError(f"Stage {name} contains a non-finite value.")
    shape = meta.get("shape", meta.get("tensor_shape"))
    if not isinstance(shape, list) or shape != list(array.shape):
        raise ValueError(f"Stage {name} shape does not match its metadata.")
    tensor_shape = meta.get("tensor_shape")
    if tensor_shape is not None:
        if (not isinstance(tensor_shape, list) or array.ndim < 1 or
                tensor_shape != list(array.shape[1:])):
            raise ValueError(f"Input stage {name} tensor_shape does not match its stacked array.")
        if not isinstance(meta.get("name"), str) or not meta["name"]:
            raise ValueError(f"Input stage {name} has no valid model input name.")
    if meta.get("dtype") != array.dtype.str:
        raise ValueError(f"Stage {name} dtype does not match its metadata.")
    if type(meta.get("nbytes")) is not int or meta["nbytes"] != array.nbytes:
        raise ValueError(f"Stage {name} nbytes does not match its metadata.")
    if _hash_text(meta.get("sha256"), f"Stage {name} hash") != array_hash(array):
        raise ValueError(f"Stage hash mismatch for {name}.")
    if array.ndim and (array.shape[0] > MAX_FRAMES or any(dim > 640 for dim in array.shape[1:])):
        raise ValueError(f"Stage {name} shape exceeds capture dimensions.")


def _load_capture(run_dir: Path, capture: dict) -> tuple[Path, dict[str, np.ndarray]]:
    if not isinstance(capture, dict):
        raise ValueError("Repeat capture metadata is missing.")
    path = _resolve_capture(run_dir, capture.get("path"))
    expected_size = capture.get("size_bytes")
    if type(expected_size) is not int or expected_size != path.stat().st_size:
        raise ValueError("Capture file size does not match the report.")
    if _hash_text(capture.get("sha256"), "Capture hash") != sha256_file(path):
        raise ValueError("Capture file SHA-256 does not match the report.")
    infos = _inspect_npz(path)
    _preflight_npy_headers(path, infos)
    metadata = _metadata_map(capture)
    if not metadata:
        raise ValueError("Capture report has no verifiable stage metadata.")
    arrays = {}
    total = 0
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(metadata):
                raise ValueError("Capture NPZ members do not match report stage metadata.")
            for name in archive.files:
                array = archive[name]
                _validate_stage_array(name, array, metadata[name])
                total += array.nbytes
                if total > MAX_CAPTURE_BYTES:
                    raise ValueError("Capture arrays exceed the 256 MiB aggregate bound.")
                arrays[name] = np.array(array, copy=True, order="C")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Could not safely load capture {path.name}.") from exc
    return path, arrays


def load_and_validate_report(report_path: str | Path) -> tuple[dict, list[dict[str, np.ndarray]]]:
    report_file = Path(report_path).expanduser().absolute()
    _no_symlink_path(report_file)
    if not report_file.is_file() or report_file.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError("Report is missing or larger than the 16 MiB bound.")
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Report is not valid UTF-8 JSON.") from exc
    if not isinstance(report, dict) or report.get("status") != "completed_diagnostic":
        raise ValueError("Report is not a completed diagnostic run.")
    repeats = report.get("repeats_completed")
    if report.get("repeats_max") != 3 or not isinstance(repeats, list) or len(repeats) != 3:
        raise ValueError("Report must contain exactly three completed repeats.")
    run_dir = report_file.parent.resolve(strict=True)
    loaded = []
    for expected_index, repeat in enumerate(repeats, 1):
        if not isinstance(repeat, dict) or repeat.get("repeat") != expected_index:
            raise ValueError("Report repeat sequence must be complete and ordered 1, 2, 3.")
        capture = repeat.get("capture")
        path, arrays = _load_capture(run_dir, capture)
        loaded.append({"repeat": expected_index, "path": str(path), "arrays": arrays,
                       "capture": capture})
    stage_sets = [set(item["arrays"]) for item in loaded]
    if stage_sets[1:] != stage_sets[:-1]:
        raise ValueError("All repeat captures must contain the same stages.")
    return report, loaded


def compare_sequences(reference: np.ndarray, candidate: np.ndarray) -> dict:
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("Variance comparison shape or dtype mismatch.")
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Variance comparison requires finite numeric arrays.")
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    count = int(delta.size)
    per_frame = []
    if a.ndim:
        for index in range(a.shape[0]):
            left, right, frame_delta = a[index], b[index], delta[index]
            per_frame.append({"index": index, "reference_sha256": array_hash(left),
                              "candidate_sha256": array_hash(right),
                              "exact_equal": bool(np.array_equal(left, right)),
                              "changed_values": int(np.count_nonzero(frame_delta)),
                              "max_abs_difference": float(frame_delta.max(initial=0)),
                              "mean_abs_difference": float(frame_delta.mean()) if frame_delta.size else 0.0})
    return {"shape": list(a.shape), "reference_dtype": a.dtype.str, "candidate_dtype": b.dtype.str,
            "reference_sha256": array_hash(a), "candidate_sha256": array_hash(b),
            "exact_equal": bool(np.array_equal(a, b)), "value_count": count,
            "changed_values": int(np.count_nonzero(delta)),
            "max_abs_difference": float(delta.max(initial=0)),
            "mean_abs_difference": float(delta.mean()) if count else 0.0,
            "per_frame": per_frame}


def compare_all_pairs(loaded: list[dict]) -> list[dict]:
    results = []
    for left, right in itertools.combinations(loaded, 2):
        stage_results = {name: compare_sequences(left["arrays"][name], right["arrays"][name])
                         for name in sorted(left["arrays"])}
        results.append({"reference_repeat": left["repeat"], "candidate_repeat": right["repeat"],
                        "stages": stage_results})
    return results


def validate_encoder_profile(report: dict, codecs: set[str]) -> dict[str, tuple[str, ...]]:
    experiment = report.get("encoder_experiment")
    profile = experiment.get("profile") if isinstance(experiment, dict) else None
    if not isinstance(profile, dict) or experiment.get("quality_profile") != "fast":
        raise ValueError("Saved encoder experiment is not the production fast profile.")
    expected_video = FAST_VIDEO_ARGS if "h264_nvenc" in codecs else FAST_X264_ARGS
    if tuple(profile.get("video_args", ())) != expected_video or tuple(profile.get("software_fallback_args", ())) != FAST_X264_ARGS:
        raise ValueError("Saved encoder arguments do not match the production fast profile.")
    args = {}
    if "libx264" in codecs:
        args["libx264"] = FAST_X264_ARGS
    if "h264_nvenc" in codecs:
        args["h264_nvenc"] = FAST_VIDEO_ARGS
    unsupported = codecs - set(args)
    if unsupported:
        raise ValueError(f"Report requested unsupported encoder(s): {sorted(unsupported)}")
    records = experiment.get("encoders", {})
    for codec in codecs:
        runs = records[codec].get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"Saved {codec} encoder runs are missing.")
        expected = list(args[codec])
        for run in runs:
            command = run.get("command") if isinstance(run, dict) else None
            if not isinstance(command, list) or "-an" not in command:
                raise ValueError(f"Saved {codec} encoder run has no verifiable argument list.")
            start = command.index("-an") + 1
            if command[start:start + len(expected)] != expected:
                raise ValueError(f"Saved {codec} run arguments do not match the production fast profile.")
    return args


def server_snapshot(timeout: float = 2.0) -> dict:
    snapshot = {}
    for port in SERVER_PORTS:
        base = f"http://127.0.0.1:{port}"
        try:
            with urlopen(base + "/api/status", timeout=timeout) as response:
                status = json.loads(response.read())
            with urlopen(base + "/api/jobs", timeout=timeout) as response:
                jobs = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError(f"Could not verify local API {port} is idle: {exc}") from exc
        if not isinstance(status, dict) or not isinstance(jobs, list):
            raise RuntimeError(f"Local API {port} returned invalid status or jobs.")
        if any(not isinstance(job, dict) for job in jobs):
            raise RuntimeError(f"Local API {port} returned a malformed jobs list.")
        active_or_unknown = [job for job in jobs if job.get("state") not in TERMINAL_STATES]
        if active_or_unknown:
            raise RuntimeError(f"Refusing encode phase while local API {port} has active or unknown jobs: "
                               f"{[{k: job.get(k) for k in ('id', 'state', 'purpose')} for job in active_or_unknown]}")
        snapshot[str(port)] = {"available": True, "build_fingerprint": status.get("build_fingerprint"),
                               "job_ids": [job.get("id") for job in jobs if isinstance(job, dict)]}
    return snapshot


def assert_no_new_jobs(baseline: dict, current: dict) -> None:
    added = []
    for port, before in baseline.items():
        after = current.get(port, {})
        old_ids = set(before.get("job_ids", [])) if before.get("available") else set()
        new_ids = set(after.get("job_ids", [])) - old_ids if after.get("available") else set()
        added.extend({"port": port, "id": job_id} for job_id in new_ids)
    if added:
        raise RuntimeError(f"A new local API job appeared during offline analysis: {added}")


def _new_output_folder() -> Path:
    base = ROOT / ".runtime" / "variance-capture-analysis"
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        candidate = base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Could not allocate a unique variance analysis folder.")


def _encode_captured_sequences(report: dict, loaded: list[dict], output: Path) -> dict:
    codec_records = report.get("encoder_experiment", {}).get("encoders", {})
    if not isinstance(codec_records, dict):
        raise ValueError("Report encoder records are invalid.")
    codecs = {name for name, value in codec_records.items()
              if name in {"libx264", "h264_nvenc"} and isinstance(value, dict) and value.get("runs")}
    if not codecs:
        raise ValueError("Report does not contain a previously exercised supported codec.")
    encode_args = validate_encoder_profile(report, codecs)
    baseline = server_snapshot()
    from main import find_ffmpeg
    ffmpeg = find_ffmpeg()
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError("Current trusted FFmpeg could not be resolved through main.find_ffmpeg().")
    if "h264_nvenc" in codecs:
        from video_encoder import select_encoder_profile
        current_profile = select_encoder_profile(ffmpeg, "fast")
        assert_no_new_jobs(baseline, server_snapshot())
        if not current_profile.hardware_available or current_profile.codec != "h264_nvenc" or tuple(current_profile.video_args) != FAST_VIDEO_ARGS:
            raise RuntimeError("Current trusted FFmpeg/driver no longer verifies the saved fast NVENC profile.")

    from scripts.isolate_gpu_variance import encode_once, read_video_frames
    from scripts.verify_clip_preview import compare_decoded_frames

    frames_per_sequence = report.get("settings", {}).get("frames", 36)
    fps = report.get("settings", {}).get("fps", 12)
    if frames_per_sequence != 36 or fps != 12:
        raise ValueError("Captured encoding requires the contract's 36-frame, 12 FPS sequence.")
    for item in loaded:
        frames = item["arrays"].get("rendered_bgr")
        if frames is None or frames.shape != (36, 180, 640, 3) or frames.dtype != np.uint8:
            raise ValueError("Each rendered_bgr capture must be 36x180x640x3 uint8.")

    results = {}
    (output / "encodes").mkdir()
    for codec in sorted(codecs):
        current = server_snapshot()
        assert_no_new_jobs(baseline, current)
        decoded, runs = [], []
        for item in loaded:
            checkpoint = server_snapshot()
            assert_no_new_jobs(baseline, checkpoint)
            target = output / "encodes" / f"{codec}-repeat-{item['repeat']}.mp4"
            started = time.monotonic()
            encoded = encode_once(ffmpeg, item["arrays"]["rendered_bgr"], encode_args[codec], target)
            sequence = encoded.pop("decoded")
            sequence = read_video_frames(target) if sequence is None else sequence
            assert_no_new_jobs(baseline, server_snapshot())
            decoded.append(sequence)
            runs.append({"repeat": item["repeat"], **encoded,
                         "elapsed_seconds": time.monotonic() - started})
        pairs = []
        for left_index, right_index in itertools.combinations(range(3), 2):
            pairs.append({"reference_repeat": loaded[left_index]["repeat"],
                          "candidate_repeat": loaded[right_index]["repeat"],
                          "comparison": compare_decoded_frames(decoded[left_index], decoded[right_index], "gpu")})
        results[codec] = {"runs": runs, "cross_inference_comparisons": pairs,
                          "scope": "different captured inference sequences, same encoder; unchanged GPU pixel gate"}
    final = server_snapshot()
    assert_no_new_jobs(baseline, final)
    return {"ffmpeg_path": ffmpeg, "server_baseline": baseline,
            "post_encode_server_snapshot": final, "codecs": results}


def analyze(report_path: str | Path, encode: bool = False) -> Path:
    report, loaded = load_and_validate_report(report_path)
    output = _new_output_folder()
    all_stages = sorted(loaded[0]["arrays"])
    missing_input_capture = not any(name.startswith("preprocessed_input_") for name in all_stages)
    result = {"schema_version": 1, "source_report": str(Path(report_path).resolve()),
              "source_run_id": report.get("run_id"), "status": "completed",
              "analysis_performed_without_inference": True,
              "captures": [{"repeat": item["repeat"], "path": item["path"],
                            "sha256": item["capture"].get("sha256"),
                            "stages": all_stages} for item in loaded],
              "stage_input_capture": "missing_in_legacy_artifact" if missing_input_capture else "present",
              "stage_pair_comparisons": compare_all_pairs(loaded),
              "encoder_analysis": None}
    if encode:
        result["encoder_analysis"] = _encode_captured_sequences(report, loaded, output)
        result["encoder_analysis"]["cross_inference_is_distinct_from_fixed_input_repeats"] = True
    output_report = output / "analysis.json"
    output_report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Completed variance isolation report JSON")
    parser.add_argument("--encode", action="store_true", help="Encode each captured rendered sequence per saved fast codec")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(args.report, args.encode)
    except Exception as exc:
        print(f"Variance capture analysis failed: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
