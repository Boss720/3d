"""Exercise production decoding and export for video packets without PTS/DTS."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import main
import numpy as np
from gpu_hardware import probe_hardware
from media_process import probe_video
from video_encoder import EncoderProfile
from video_decoder import open_video_capture, select_amd_adapter


OUTPUT_ROOT = ROOT / ".runtime" / "vfr-wave" / "missing-pts-release-check"
EXPECTED_FPS = 12
EXPECTED_FRAMES = 12
EXPECTED_DURATION = EXPECTED_FRAMES / EXPECTED_FPS
CPU_ENCODER = EncoderProfile(
    quality="fast", codec="libx264",
    video_args=("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1"),
    software_fallback_args=("-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-threads", "1"),
    hardware_available=False,
)


def run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_video_summary(path: Path, ffprobe: str) -> dict:
    probe = run([ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_read_frames",
                 "-of", "json", str(path)])
    streams = json.loads(probe.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("The exported file has no video stream.")
    stream = streams[0]
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the exported file.")
    frames = 0
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    try:
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
    finally:
        capture.release()
    return {"stream": stream, "opencv_decoded_frames": frames,
            "opencv_fps": fps}


def decode_source(source: Path, ffmpeg: str, *, prefer_hardware: bool) -> dict:
    capture = open_video_capture(source, ffmpeg, prefer_hardware=prefer_hardware)
    frames = 0
    checksum = hashlib.sha256()
    images = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            checksum.update(memoryview(frame).cast("B"))
            images.append(frame.copy())
            frames += 1
    finally:
        capture.release()
    return {"frames": frames, "bgr_sha256": checksum.hexdigest(),
            "decoder": capture.decoder_info, "probe": capture.probe, "_images": images}


def make_job(source: Path, job_id: str) -> main.Job:
    return main.Job(
        id=job_id, filename=source.name, input_path=source, media_type="video",
        layout="SBS", strength=5.0, convergence=0.5, depth_mode="balanced",
        input_size=256, resolution="source", quality="fast", invert_depth=False,
        temporal_smoothing=0.0, edge_refinement=False, execution_mode="cpu",
        video_decode="software", use_depth_cache=False,
    )


def main_check() -> dict:
    ffmpeg = main.FFMPEG_PATH
    if not ffmpeg or not Path(ffmpeg).is_file():
        raise RuntimeError(f"FFmpeg unavailable: {ffmpeg!r}")
    ffprobe = str(Path(ffmpeg).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe"))
    run_root = OUTPUT_ROOT / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8])
    inputs, outputs = run_root / "inputs", run_root / "outputs"
    inputs.mkdir(parents=True, exist_ok=False)
    outputs.mkdir(parents=True, exist_ok=False)
    source = inputs / "video-with-no-packet-timestamps.h264"
    output = outputs / "stereo.mp4"
    report_path = run_root / "report.json"
    report = {"status": "running", "run_root": str(run_root.resolve()),
              "ffmpeg": ffmpeg, "ffmpeg_version": run([ffmpeg, "-version"]).stdout.splitlines()[0],
              "fixture": "12 generated H.264 Annex B frames in an elementary stream; no container timestamps"}
    try:
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
             f"testsrc2=size=160x90:rate={EXPECTED_FPS}:duration=1", "-frames:v", str(EXPECTED_FRAMES),
             "-c:v", "libx264", "-preset", "ultrafast", "-f", "h264", str(source)])
        packet_result = run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
                             "-show_entries", "packet=pts,dts,pts_time,dts_time", "-of", "json", str(source)])
        packets = json.loads(packet_result.stdout).get("packets", [])
        packet_pts_count = sum(packet.get("pts") is not None or packet.get("pts_time") is not None
                               for packet in packets)
        packet_dts_count = sum(packet.get("dts") is not None or packet.get("dts_time") is not None
                               for packet in packets)
        source_metadata = probe_video(source, ffmpeg)
        software_decode = decode_source(source, ffmpeg, prefer_hardware=False)
        hardware_inventory = probe_hardware()
        amd_adapter = select_amd_adapter(hardware_inventory)
        hardware_choice_decode = decode_source(source, ffmpeg, prefer_hardware=True)
        software_images = software_decode.pop("_images")
        hardware_images = hardware_choice_decode.pop("_images")
        pixel_difference = None
        if software_decode["frames"] == hardware_choice_decode["frames"] and software_decode["frames"]:
            differences = [np.abs(software.astype(np.int16) - hardware.astype(np.int16))
                           for software, hardware in zip(software_images, hardware_images)]
            flattened = np.concatenate([difference.reshape(-1) for difference in differences])
            pixel_difference = {"mean_absolute_error": float(np.mean(flattened)),
                                "p95_absolute_error": float(np.percentile(flattened, 95)),
                                "max_absolute_error": int(np.max(flattened))}

        job = make_job(source, uuid.uuid4().hex)
        main.JOBS[job.id] = job
        try:
            with patch.object(main, "OUTPUTS", outputs), \
                    patch.object(main, "output_path_for", return_value=output), \
                    patch.object(main, "select_encoder_profile", return_value=CPU_ENCODER):
                returned = main.run_video(job, None)
        finally:
            main.JOBS.pop(job.id, None)

        output_summary = output_video_summary(output, ffprobe) if output.is_file() else None
        output_stream = output_summary.get("stream", {}) if output_summary else {}
        fps = None
        try:
            numerator, denominator = output_stream.get("avg_frame_rate", "0/1").split("/")
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError, TypeError):
            pass
        duration = float(output_stream["duration"]) if output_stream.get("duration") else None
        report.update({
            "source": str(source.resolve()), "source_bytes": source.stat().st_size,
            "source_sha256": sha256(source), "packet_count": len(packets),
            "packet_pts_count": packet_pts_count, "packet_dts_count": packet_dts_count,
            "source_metadata": source_metadata, "software_decode": software_decode,
            "hardware_inventory": hardware_inventory, "selected_amd_adapter": amd_adapter,
            "hardware_choice_decode": hardware_choice_decode,
            "software_vs_hardware_pixel_difference": pixel_difference,
            "decoder": job.decoder, "job_frames_processed": job.frames_processed,
            "job_total_frames": job.total_frames, "returned_output": str(returned),
            "output": str(output.resolve()), "output_sha256": sha256(output) if output.is_file() else None,
            "output_summary": output_summary,
            "checks": {
                "all_source_packets_lack_pts": bool(packets) and packet_pts_count == 0,
                "all_source_packets_lack_dts": bool(packets) and packet_dts_count == 0,
                "software_decoder_processed_all_frames": software_decode["frames"] == EXPECTED_FRAMES and
                    software_decode["decoder"].get("cleanup_complete") is True,
                "hardware_choice_processed_all_frames": hardware_choice_decode["frames"] == EXPECTED_FRAMES and
                    hardware_choice_decode["decoder"].get("cleanup_complete") is True,
                "hardware_choice_matches_available_adapter": (
                    hardware_choice_decode["decoder"].get("hardware") is True if amd_adapter
                    else hardware_choice_decode["decoder"].get("hardware") is False),
                "production_job_processed_every_frame": job.frames_processed == EXPECTED_FRAMES,
                "decoder_cleanup_complete": bool(job.decoder.get("cleanup_complete")),
                "output_has_every_frame": bool(output_summary and
                    output_summary["opencv_decoded_frames"] == EXPECTED_FRAMES),
                "output_rate_preserved": fps is not None and math.isclose(fps, EXPECTED_FPS, abs_tol=0.02),
                "output_duration_preserved": duration is not None and
                    abs(duration - EXPECTED_DURATION) <= 0.12,
            },
        })
        report["passed"] = all(report["checks"].values())
        report["status"] = "passed" if report["passed"] else "failed"
    except Exception as exc:
        report.update({"status": "failed", "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"})
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report.get("passed"):
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    main_check()
