"""Offline media timing checks for StereoLab's documented CFR export policy.

Generates one mixed-rate source with non-zero timestamps and delayed audio plus
one five-minute low-resolution CFR source. Both run through the real OpenCV
software decoder and ``main.run_video`` using CPU approximation depth. Outputs,
inputs, and reports stay under ``.runtime/production-media``.

This verifies generated media only. It does not establish VFR timestamp-exact,
rotation, HDR, subtitle, spatial metadata, or real-footage fidelity.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import main
import numpy as np
from video_encoder import EncoderProfile


OUTPUT_ROOT = ROOT / ".runtime" / "production-media"
VFR_FRAMES = 120
VFR_EXPECTED_DURATION = 8.0
VFR_EXPECTED_AV_OFFSET = 0.25
LONG_FPS = 10
LONG_FRAMES = 3001
OFFSET_FPS = 2
OFFSET_FRAMES = 20
OFFSET_VIDEO_START_SECONDS = 0.25
OFFSET_VIDEO_DURATION_SECONDS = OFFSET_FRAMES / OFFSET_FPS
OFFSET_AUDIO_DURATION_SECONDS = OFFSET_VIDEO_DURATION_SECONDS + OFFSET_VIDEO_START_SECONDS
ENDPOINT_TOLERANCE_SECONDS = 0.12
ALIGNMENT_TOLERANCE_SECONDS = 0.12
CPU_ENCODER = EncoderProfile(
    quality="fast",
    codec="libx264",
    video_args=("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1"),
    software_fallback_args=("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1"),
    hardware_available=False,
)


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)


def probe(ffmpeg: str, path: Path) -> dict:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe"))
    result = run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    return json.loads(result.stdout)


def stream_info(data: dict, kind: str) -> dict:
    return next((stream for stream in data.get("streams", []) if stream.get("codec_type") == kind), {})


def decoded_frame_summary(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    advertised_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    size = None
    checksum = hashlib.sha256()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count == 0:
                size = [int(frame.shape[1]), int(frame.shape[0])]
            checksum.update(memoryview(frame).cast("B"))
            count += 1
    finally:
        capture.release()
    return {
        "decoded_frames": count,
        "advertised_frames": advertised_count,
        "opencv_fps": fps,
        "frame_size": size,
        "decoded_pixels_sha256": checksum.hexdigest(),
        "decode_seconds": round(time.perf_counter() - started, 3),
    }


def decoded_audio_seconds(ffmpeg: str, path: Path) -> dict:
    pcm = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
         "-vn", "-ac", "1", "-ar", "48000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True, timeout=180,
    ).stdout
    return {"decoded_pcm_bytes": len(pcm), "sample_rate_hz": 48000, "channels": 1,
            "decoded_seconds": len(pcm) / 2 / 48000,
            "decoded_pcm_sha256": hashlib.sha256(pcm).hexdigest()}


def media_timing(ffmpeg: str, path: Path) -> dict:
    data = probe(ffmpeg, path)
    video, audio = stream_info(data, "video"), stream_info(data, "audio")
    fmt = data.get("format", {})
    video_start = float(video.get("start_time", fmt.get("start_time", 0)) or 0)
    audio_start = float(audio.get("start_time", fmt.get("start_time", 0)) or 0) if audio else None
    format_start = float(fmt.get("start_time") or 0)
    return {
        "container_duration_seconds": float(fmt.get("duration") or 0),
        "format_start_seconds": format_start,
        "video_start_seconds": video_start,
        "video_duration_seconds": float(video.get("duration") or 0),
        "video_avg_frame_rate": video.get("avg_frame_rate"),
        "video_r_frame_rate": video.get("r_frame_rate"),
        "video_nb_frames": video.get("nb_frames"),
        "audio_start_seconds": audio_start,
        "audio_duration_seconds": float(audio.get("duration") or 0) if audio else None,
        "audio_video_start_offset_seconds": audio_start - video_start if audio_start is not None else None,
        "audio_present": bool(audio),
    }


def source_vfr_pts(ffmpeg: str, path: Path) -> dict:
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe"))
    result = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
                  "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(path)])
    pts = [float(frame["best_effort_timestamp_time"]) for frame in json.loads(result.stdout).get("frames", [])
           if frame.get("best_effort_timestamp_time") is not None]
    deltas = [b - a for a, b in zip(pts, pts[1:])]
    return {"frame_pts_count": len(pts), "first_frame_pts_seconds": pts[0] if pts else None,
            "last_frame_pts_seconds": pts[-1] if pts else None,
            "frame_pts_seconds": pts,
            "distinct_rounded_frame_deltas_seconds": sorted({round(delta, 4) for delta in deltas}),
            "variable_frame_timing_observed": len({round(delta, 4) for delta in deltas}) > 1}


def decode_ordinal_markers(path: Path, *, side_by_side: bool) -> dict:
    """Read the seven-bit source ordinal burned into the generated frame top band."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open marker media {path}")
    count = 0
    eyes = {"left": [], "right": []} if side_by_side else {"source": []}
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            halves = (frame[:, :160], frame[:, 160:320]) if side_by_side else (frame,)
            names = ("left", "right") if side_by_side else ("source",)
            for name, image in zip(names, halves):
                decoded = 0
                for bit in range(7):
                    x = 8 + bit * 16
                    patch = image[4:12, x + 4:x + 10]
                    if float(np.median(patch)) > 127:
                        decoded |= 1 << bit
                eyes[name].append(decoded)
            count += 1
    finally:
        capture.release()
    matches = {name: sum(actual == ordinal for ordinal, actual in enumerate(values))
               for name, values in eyes.items()}
    return {"decoded_frames": count, "ordinal_matches": matches,
            "ordinal_match_all_frames": {name: matched == count for name, matched in matches.items()},
            "ordinals": eyes,
            "first_12_decoded_ordinals": {name: values[:12] for name, values in eyes.items()}}


def make_vfr_raw_markers(path: Path) -> None:
    """Create 120 distinct ordinal-marked frames without changing source order."""
    with path.open("wb") as output:
        for ordinal in range(VFR_FRAMES):
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            # Moving pattern keeps the synthetic content visibly distinct beyond
            # the marker, while the upper marker band remains stable under warp.
            frame[:, :, 0] = ((np.arange(160, dtype=np.int16)[None, :] + ordinal * 3) % 256).astype(np.uint8)
            frame[:, :, 1] = ((np.arange(90, dtype=np.int16)[:, None] * 2 + ordinal * 5) % 256).astype(np.uint8)
            x = 20 + (ordinal * 7) % 110
            cv2.rectangle(frame, (x, 30), (x + 24, 78), (255, 40, 180), -1)
            for bit in range(7):
                x = 8 + bit * 16
                color = 255 if ordinal & (1 << bit) else 0
                frame[2:14, x:x + 14] = color
            output.write(frame.tobytes())


def make_job(source: Path, fixture_id: str) -> main.Job:
    return main.Job(
        id=uuid.uuid4().hex, filename=f"{fixture_id}.mp4", input_path=source,
        media_type="video", layout="SBS", strength=5.0, convergence=0.5,
        depth_mode="balanced", input_size=256, resolution="source", quality="fast",
        invert_depth=False, temporal_smoothing=0.0, edge_refinement=False,
        execution_mode="cpu", video_decode="software", use_depth_cache=False,
    )


def export_fixture(source: Path, fixture_id: str, run_root: Path, ffmpeg: str) -> dict:
    output = run_root / "outputs" / f"{fixture_id}-stereo.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    job = make_job(source, fixture_id)
    main.JOBS[job.id] = job
    started = time.perf_counter()
    record = {"fixture_id": fixture_id, "source": str(source), "output": str(output)}
    try:
        with patch.object(main, "OUTPUTS", output.parent), \
                patch.object(main, "output_path_for", return_value=output), \
                patch.object(main, "select_encoder_profile", return_value=CPU_ENCODER):
            returned = main.run_video(job, None)
        record["run_video_returned"] = str(returned)
        record["run_video_seconds"] = round(time.perf_counter() - started, 3)
        record["job_frames_processed"] = job.frames_processed
        record["job_total_frames"] = job.total_frames
        record["job_decoder"] = job.decoder
        record["job_encoder"] = job.encoder
        record["job_execution"] = job.execution
        record["output_exists"] = output.is_file()
        if output.is_file():
            record["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            record["output_timing"] = media_timing(ffmpeg, output)
            record["output_decoded_video"] = decoded_frame_summary(output)
            record["output_decoded_audio"] = decoded_audio_seconds(ffmpeg, output)
    except Exception as exc:
        record["run_video_error"] = f"{type(exc).__name__}: {exc}"
        record["run_video_seconds"] = round(time.perf_counter() - started, 3)
        record["output_exists"] = output.is_file()
        record["job_frames_processed"] = job.frames_processed
        record["job_total_frames"] = job.total_frames
        if output.is_file():
            try:
                record["partial_output_timing"] = media_timing(ffmpeg, output)
                record["partial_output_decoded_video"] = decoded_frame_summary(output)
            except Exception as inspect_error:
                record["partial_output_inspection_error"] = f"{type(inspect_error).__name__}: {inspect_error}"
    finally:
        main.JOBS.pop(job.id, None)
    return record


def fixture_assertions(source_timing: dict, source_video: dict, output_record: dict,
                       expected_frames: int, expected_duration: float, *,
                       expected_fps: float | None, expected_av_offset: float | None) -> dict:
    output_timing = output_record.get("output_timing")
    output_video = output_record.get("output_decoded_video")
    output_audio = output_record.get("output_decoded_audio")
    checks = {
        "run_video_succeeded": "run_video_error" not in output_record,
        "output_exists": output_record.get("output_exists") is True,
        "source_expected_frame_count": source_video.get("decoded_frames") == expected_frames,
        "output_all_frames_decoded": bool(output_video and output_video.get("decoded_frames") == expected_frames),
        "output_job_processed_all_frames": output_record.get("job_frames_processed") == expected_frames,
        "output_nominal_fps_preserved": bool(output_video and expected_fps is not None
                                                and math.isfinite(output_video.get("opencv_fps", 0))
                                                and abs(output_video["opencv_fps"] - expected_fps) <= 0.02),
        "video_duration_within_endpoint_tolerance": bool(output_timing and
            abs(output_timing["video_duration_seconds"] - expected_duration) <= ENDPOINT_TOLERANCE_SECONDS),
        "audio_stream_present_and_decodable": bool(output_timing and output_timing["audio_present"]
                                                     and output_audio and output_audio["decoded_seconds"] > 0),
        "audio_decoded_length_within_endpoint_tolerance": bool(output_audio and
            abs(output_audio["decoded_seconds"] -
                (expected_duration - (expected_av_offset or 0.0))) <= ENDPOINT_TOLERANCE_SECONDS),
    }
    if expected_av_offset is not None:
        offset = output_timing.get("audio_video_start_offset_seconds") if output_timing else None
        checks["relative_audio_video_start_alignment_preserved"] = bool(offset is not None and
            abs(offset - expected_av_offset) <= ALIGNMENT_TOLERANCE_SECONDS)
    return {"checks": checks, "passed": all(checks.values()),
            "expected": {"frames": expected_frames, "duration_seconds": expected_duration,
                         "fps": expected_fps, "audio_video_start_offset_seconds": expected_av_offset},
            "tolerances": {"endpoint_seconds": ENDPOINT_TOLERANCE_SECONDS,
                           "relative_av_start_offset_seconds": ALIGNMENT_TOLERANCE_SECONDS}}


def generate_fixtures(ffmpeg: str, root: Path) -> dict:
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    vfr = inputs / "mixed-rate-nonzero-start-delayed-audio.mp4"
    raw_markers = inputs / "mixed-rate-ordinal-marker-source.bgr24"
    make_vfr_raw_markers(raw_markers)
    # 40 frames at 10 fps (4s), followed by 80 at 20 fps (4s). The shared
    # +2s offset and extra +0.25s audio delay make both PTS and AV offset known.
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", "160x90", "-framerate", "20", "-i", str(raw_markers), "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=8.2", "-vf",
         "setpts=if(lt(N\\,40)\\,N/(10*TB)\\,4/TB+(N-40)/(20*TB))+2/TB", "-af",
         "asetpts=PTS+2.25/TB", "-fps_mode:v", "vfr", "-frames:v", str(VFR_FRAMES),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-avoid_negative_ts", "disabled", str(vfr)], timeout=60)

    long = inputs / "cfr-five-minutes-plus.mp4"
    duration = LONG_FRAMES / LONG_FPS
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=160x90:rate={LONG_FPS}:duration={duration:.6f}", "-f", "lavfi", "-i",
         f"sine=frequency=523.25:sample_rate=48000:duration={duration:.6f}", "-frames:v", str(LONG_FRAMES),
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-shortest", str(long)], timeout=180)

    offset = inputs / "audio-first-video-start-plus-250ms.mp4"
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         f"testsrc2=size=160x90:rate={OFFSET_FPS}:duration={OFFSET_VIDEO_DURATION_SECONDS:.6f}",
         "-f", "lavfi", "-i",
         f"sine=frequency=659.25:sample_rate=48000:duration={OFFSET_AUDIO_DURATION_SECONDS:.6f}",
         "-map", "0:v:0", "-map", "1:a:0", "-vf",
         f"settb=AVTB,setpts=PTS+{OFFSET_VIDEO_START_SECONDS:.9f}/TB",
         "-c:v", "libx264", "-preset", "ultrafast",
         "-crf", "22", "-pix_fmt", "yuv420p", "-enc_time_base:v", "1:1000000",
         "-c:a", "aac", "-b:a", "96k",
         "-fps_mode:v", "passthrough", "-avoid_negative_ts", "disabled", str(offset)], timeout=60)
    return {"vfr": vfr, "long": long, "long_duration": duration,
            "vfr_raw_markers": raw_markers, "offset": offset}


def run_verification() -> dict:
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    run_root = OUTPUT_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    report_path = run_root / "report.json"
    report = {
        "status": "running", "run_id": run_id, "run_root": str(run_root),
        "ffmpeg": main.FFMPEG_PATH, "cpu_only": True,
        "depth_mode": "balanced visual approximation", "execution_mode": "cpu",
        "decoder": "Production software capture selected by video_decode=software",
        "encoder_override": {"codec": "libx264", "preset": "ultrafast", "threads": 1,
                             "reason": "Keep this verification off the GPU; actual run_video FPS probing and ffmpeg pipeline remain unchanged."},
        "policy": "Check VFR output frame count, CFR rate, duration, and relative AV start offset separately from interior frame timing.",
        "source_lines": {"README.md": 166, "main.py": "459-480, 747-824, 846-876, 957-966"},
        "fixtures": {},
        "limits": ["Generated test patterns only; not representative real footage.",
                   "CFR resampling may duplicate/drop source frames; interior correspondence must remain within the stated 120ms tolerance.",
                   "No neural model, GPU inference, hardware decoder, or packaged executable was exercised."],
    }
    try:
        if not main.FFMPEG_PATH or not Path(main.FFMPEG_PATH).is_file():
            raise RuntimeError(f"FFmpeg unavailable: {main.FFMPEG_PATH!r}")
        ffmpeg = main.FFMPEG_PATH
        report["ffmpeg_version"] = run([ffmpeg, "-version"]).stdout.splitlines()[0]
        cv2.setNumThreads(1)
        report["opencv_version"] = cv2.__version__
        report["opencv_threads"] = cv2.getNumThreads()
        fixture_paths = generate_fixtures(ffmpeg, run_root)

        vfr_path = fixture_paths["vfr"]
        vfr_timing = media_timing(ffmpeg, vfr_path)
        vfr_video = decoded_frame_summary(vfr_path)
        vfr_pts = source_vfr_pts(ffmpeg, vfr_path)
        vfr_source_markers = decode_ordinal_markers(vfr_path, side_by_side=False)
        vfr_source_audio = decoded_audio_seconds(ffmpeg, vfr_path)
        vfr_export = export_fixture(vfr_path, "mixed-rate", run_root, ffmpeg)
        vfr_expected_fps = VFR_FRAMES / VFR_EXPECTED_DURATION
        vfr_output_markers = (decode_ordinal_markers(vfr_export["output"], side_by_side=True)
                              if vfr_export.get("output_exists") else None)
        vfr_export["assertions"] = fixture_assertions(
            vfr_timing, vfr_video, vfr_export, VFR_FRAMES, VFR_EXPECTED_DURATION,
            expected_fps=vfr_expected_fps, expected_av_offset=VFR_EXPECTED_AV_OFFSET)
        transition_ordinal = 40
        source_pts = vfr_pts["frame_pts_seconds"]
        output_fps = (vfr_export.get("output_decoded_video") or {}).get("opencv_fps", 0)
        source_transition_elapsed = source_pts[transition_ordinal] - source_pts[0]
        output_ordinals = (vfr_output_markers or {}).get('ordinals', {})
        left = output_ordinals.get('left', [])
        right = output_ordinals.get('right', [])
        transition_index = next((index for index, ordinal in enumerate(left)
                                 if ordinal >= transition_ordinal), None)
        output_transition_elapsed = (transition_index / output_fps
                                     if transition_index is not None and output_fps else None)
        correspondence_errors = [abs(index / output_fps - (source_pts[ordinal] - source_pts[0]))
                                 for index, ordinal in enumerate(left)
                                 if output_fps and 0 <= ordinal < len(source_pts)]
        valid_ordinals = bool(left and left == right and all(
            0 <= ordinal < VFR_FRAMES for ordinal in left)
            and all(a <= b for a, b in zip(left, left[1:])))
        transition_drift = (output_transition_elapsed - source_transition_elapsed
                            if output_transition_elapsed is not None else None)
        vfr_export["interior_timing"] = {
            "transition_frame_ordinal": transition_ordinal,
            "source_boundary_pts_seconds": source_pts[transition_ordinal],
            "source_first_frame_pts_seconds": source_pts[0],
            "source_elapsed_at_transition_seconds": source_transition_elapsed,
            "output_transition_frame_index": transition_index,
            "output_frame_ordinal_verified_by_markers": valid_ordinals,
            "maximum_frame_correspondence_error_seconds": max(correspondence_errors, default=None),
            "output_elapsed_at_same_ordinal_seconds": output_transition_elapsed,
            "output_minus_source_transition_drift_seconds": transition_drift,
            "tolerance_seconds": ALIGNMENT_TOLERANCE_SECONDS,
            "ordinal_marker_width_bits": 7,
            "source_ordinal_marker_verification": vfr_source_markers,
            "output_left_right_ordinal_marker_verification": vfr_output_markers,
        }
        report["fixtures"]["mixed_rate_nonzero_timestamps"] = {
            "source_timing": vfr_timing, "source_decoded_video": vfr_video,
            "source_vfr_pts": vfr_pts, "export": vfr_export,
            "source_sha256": hashlib.sha256(vfr_path.read_bytes()).hexdigest(),
        }
        vfr_assertions = vfr_export["assertions"]
        vfr_assertions["checks"]["source_has_nonzero_video_start"] = vfr_timing["video_start_seconds"] >= 1.9
        vfr_assertions["checks"]["source_vfr_timing_observed"] = vfr_pts["variable_frame_timing_observed"]
        vfr_assertions["checks"]["source_ordinal_markers_match_every_frame"] = all(
            vfr_source_markers["ordinal_match_all_frames"].values())
        vfr_assertions["checks"]["output_eyes_have_valid_matching_monotonic_source_ordinals"] = valid_ordinals
        vfr_assertions["checks"]["all_frame_correspondence_within_120ms"] = bool(
            valid_ordinals and len(correspondence_errors) == len(left)
            and max(correspondence_errors) <= ALIGNMENT_TOLERANCE_SECONDS)
        vfr_assertions["checks"]["interior_transition_timing_within_120ms"] = bool(
            transition_drift is not None and abs(transition_drift) <= ALIGNMENT_TOLERANCE_SECONDS)
        vfr_assertions["checks"]["source_audio_decoded_length_matches_generated_timeline"] = (
            vfr_source_audio["decoded_seconds"] > 0 and
            abs(vfr_source_audio["decoded_seconds"] -
                (VFR_EXPECTED_DURATION - VFR_EXPECTED_AV_OFFSET)) <= ENDPOINT_TOLERANCE_SECONDS)
        vfr_assertions["passed"] = all(vfr_assertions["checks"].values())
        report["fixtures"]["mixed_rate_nonzero_timestamps"]["source_decoded_audio"] = vfr_source_audio

        offset_path = fixture_paths["offset"]
        offset_timing = media_timing(ffmpeg, offset_path)
        offset_video = decoded_frame_summary(offset_path)
        offset_source_audio = decoded_audio_seconds(ffmpeg, offset_path)
        offset_export = export_fixture(offset_path, "audio-first-video-offset", run_root, ffmpeg)
        offset_export["assertions"] = fixture_assertions(
            offset_timing, offset_video, offset_export, OFFSET_FRAMES,
            OFFSET_VIDEO_DURATION_SECONDS, expected_fps=OFFSET_FPS,
            expected_av_offset=-OFFSET_VIDEO_START_SECONDS)
        offset_output_timing = offset_export.get("output_timing") or {}
        offset_export["assertions"]["checks"]["source_format_starts_before_video"] = (
            abs(offset_timing["video_start_seconds"] - OFFSET_VIDEO_START_SECONDS) <= .02
            and abs(offset_timing.get("format_start_seconds") or 0.) <= .02)
        offset_export["assertions"]["checks"]["output_video_offset_preserved_at_low_fps"] = bool(
            offset_output_timing and abs(offset_output_timing["video_start_seconds"] -
                                         OFFSET_VIDEO_START_SECONDS) <= ALIGNMENT_TOLERANCE_SECONDS)
        offset_export["assertions"]["checks"]["output_audio_starts_before_video"] = bool(
            offset_output_timing.get("audio_start_seconds") is not None and
            offset_output_timing["audio_start_seconds"] < offset_output_timing["video_start_seconds"])
        offset_export["assertions"]["checks"]["source_audio_decoded_length_matches_generated_timeline"] = (
            abs(offset_source_audio["decoded_seconds"] - OFFSET_AUDIO_DURATION_SECONDS) <= ENDPOINT_TOLERANCE_SECONDS)
        offset_export["assertions"]["passed"] = all(offset_export["assertions"]["checks"].values())
        report["fixtures"]["audio_first_video_offset"] = {
            "source_timing": offset_timing, "source_decoded_video": offset_video,
            "source_decoded_audio": offset_source_audio, "export": offset_export,
            "source_sha256": hashlib.sha256(offset_path.read_bytes()).hexdigest(),
            "expected_video_start_seconds": OFFSET_VIDEO_START_SECONDS,
            "expected_audio_video_start_offset_seconds": -OFFSET_VIDEO_START_SECONDS,
        }

        long_path = fixture_paths["long"]
        long_timing = media_timing(ffmpeg, long_path)
        long_video = decoded_frame_summary(long_path)
        long_source_audio = decoded_audio_seconds(ffmpeg, long_path)
        long_export = export_fixture(long_path, "five-minute-cfr", run_root, ffmpeg)
        long_export["assertions"] = fixture_assertions(
            long_timing, long_video, long_export, LONG_FRAMES, fixture_paths["long_duration"],
            expected_fps=LONG_FPS, expected_av_offset=None)
        long_export["assertions"]["checks"]["source_audio_decoded_length_matches_generated_timeline"] = (
            abs(long_source_audio["decoded_seconds"] - fixture_paths["long_duration"]) <= ENDPOINT_TOLERANCE_SECONDS)
        long_export["assertions"]["passed"] = all(long_export["assertions"]["checks"].values())
        report["fixtures"]["five_minute_plus_cfr"] = {
            "source_timing": long_timing, "source_decoded_video": long_video,
            "source_decoded_audio": long_source_audio,
            "export": long_export, "source_sha256": hashlib.sha256(long_path.read_bytes()).hexdigest(),
            "size": [160, 90],
        }
        report["status"] = "passed" if all(item["export"]["assertions"]["passed"] for item in report["fixtures"].values()) else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"REPORT={report_path}")
    return report


if __name__ == "__main__":
    result = run_verification()
    raise SystemExit(0 if result["status"] == "passed" else 1)
