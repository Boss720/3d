"""Choose an FFmpeg video encoder after a bounded local NVENC smoke encode."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_PROBE_TIMEOUT_SECONDS = 6.0
_QUALITY_PROFILES = {
    "fast": {
        "nvenc": ("-preset", "p1", "-tune", "hq", "-rc", "vbr", "-cq", "30", "-b:v", "0"),
        "x264": ("-preset", "ultrafast", "-crf", "22"),
    },
    "balanced": {
        "nvenc": ("-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "24", "-b:v", "0"),
        "x264": ("-preset", "veryfast", "-crf", "20"),
    },
    "high": {
        "nvenc": ("-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", "20", "-b:v", "0"),
        "x264": ("-preset", "medium", "-crf", "18"),
    },
}


@dataclass(frozen=True)
class EncoderProfile:
    """Selected codec arguments plus a software fallback for the same quality."""

    quality: str
    codec: str
    video_args: tuple[str, ...]
    software_fallback_args: tuple[str, ...]
    hardware_available: bool
    probe_message: str | None = None


def _resolve_binary(ffmpeg_path: str | os.PathLike[str]) -> str:
    configured = os.fspath(ffmpeg_path)
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return os.path.normcase(str(candidate.resolve()))
    on_path = shutil.which(configured)
    return os.path.normcase(str(Path(on_path).resolve())) if on_path else configured


def _binary_fingerprint(binary: str) -> tuple[int, int, int]:
    try:
        stat = Path(binary).stat()
    except OSError:
        return 0, 0, 0
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


@lru_cache(maxsize=32)
def _cached_nvenc_smoke_test(
    binary: str,
    fingerprint: tuple[int, int, int],
    quality: str,
) -> tuple[bool, str | None]:
    # Keep this encode tiny and local. It validates the requested NVENC option
    # set against the configured FFmpeg binary and installed NVIDIA driver.
    del fingerprint  # Included in the cache key so replacing FFmpeg invalidates it.
    arguments = _QUALITY_PROFILES[quality]["nvenc"]
    command = [
        binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=64x64:r=1:d=1",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        *arguments,
        "-f",
        "null",
        "-",
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"NVENC smoke encode could not run: {error}"

    if result.returncode == 0:
        return True, None
    detail = (result.stderr or result.stdout or "").strip()
    if len(detail) > 400:
        detail = detail[-400:]
    return False, detail or f"NVENC smoke encode exited with status {result.returncode}"


def clear_encoder_probe_cache() -> None:
    """Clear cached capability results, useful after changing FFmpeg or drivers."""

    _cached_nvenc_smoke_test.cache_clear()


def select_encoder_profile(
    ffmpeg_path: str | os.PathLike[str],
    quality: str,
) -> EncoderProfile:
    """Return NVENC arguments only when this FFmpeg can encode at that quality.

    A small encode is cached by the resolved FFmpeg binary, its file signature,
    and quality. Any missing binary, timeout, driver error, or failed encode
    selects the matching libx264 profile. Callers can also use
    ``software_fallback_args`` if a later full NVENC encode fails.
    """

    if quality not in _QUALITY_PROFILES:
        raise ValueError(f"Unknown video quality {quality!r}; expected fast, balanced, or high")

    binary = _resolve_binary(ffmpeg_path)
    capable, message = _cached_nvenc_smoke_test(
        binary,
        _binary_fingerprint(binary),
        quality,
    )
    software_args = ("-c:v", "libx264", *_QUALITY_PROFILES[quality]["x264"])
    if capable:
        return EncoderProfile(
            quality=quality,
            codec="h264_nvenc",
            video_args=("-c:v", "h264_nvenc", *_QUALITY_PROFILES[quality]["nvenc"]),
            software_fallback_args=software_args,
            hardware_available=True,
        )
    return EncoderProfile(
        quality=quality,
        codec="libx264",
        video_args=software_args,
        software_fallback_args=software_args,
        hardware_available=False,
        probe_message=message,
    )
