"""Local conversion checks that never initialize a depth model."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import math
import os
from functools import lru_cache
from pathlib import Path

import cv2

from stereo import LAYOUTS, layout_size


MINIMUM_FREE_BYTES = 16 * 1024**2
ENCODER_PROBE_TIMEOUT_SECONDS = 6


@lru_cache(maxsize=32)
def _libx264_available(binary: str, size: int, modified_ns: int) -> bool:
    """Inventory only; no encode, inference, or session creation is performed."""
    del size, modified_ns  # Binary replacement invalidates the cache key.
    try:
        result = subprocess.run([binary, '-hide_banner', '-nostdin', '-encoders'],
                                stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                timeout=ENCODER_PROBE_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError('FFmpeg encoder check failed or timed out. Install FFmpeg with libx264 support and retry.') from None
    if result.returncode:
        raise RuntimeError('FFmpeg could not list its encoders. Install FFmpeg with libx264 support and retry.')
    return any(len(parts := line.split()) >= 2 and parts[1] == 'libx264'
               for line in (result.stdout or '').splitlines())


def _check_encoder(ffmpeg: str | None) -> None:
    if not ffmpeg:
        raise RuntimeError('Install FFmpeg or set VIDEO3D_FFMPEG to ffmpeg.exe before converting video.')
    try:
        binary = Path(ffmpeg).expanduser()
        if not binary.is_file():
            found = shutil.which(ffmpeg)
            if not found:
                raise OSError('missing binary')
            binary = Path(found)
        binary = binary.resolve()
        stat = binary.stat()
    except OSError:
        raise RuntimeError('Install FFmpeg or correct VIDEO3D_FFMPEG so it points to a readable ffmpeg.exe.') from None
    if not _libx264_available(os.path.normcase(str(binary)), stat.st_size, stat.st_mtime_ns):
        raise RuntimeError('This FFmpeg has no libx264 encoder required for software fallback. Install an FFmpeg build with libx264 support.')


def _positive_number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def _decoded_size(frame: object) -> tuple[int, int]:
    shape = getattr(frame, 'shape', ())
    if len(shape) != 3 or shape[2] != 3 or shape[0] < 1 or shape[1] < 1:
        raise RuntimeError('Source frame dimensions are invalid. Re-encode the source as a standard image or H.264 MP4.')
    return int(shape[1]), int(shape[0])


def video_metadata(capture) -> tuple[int, int, int | None, float, bool]:
    """Normalize decoder metadata consistently for preflight and conversion."""
    width = _positive_number(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = _positive_number(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width is None or height is None or int(width) < 1 or int(height) < 1:
        raise RuntimeError('Video frame dimensions are invalid. Re-encode it as H.264 MP4 and retry.')
    frames = _positive_number(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = int(frames) if frames is not None and int(frames) > 0 else None
    fps = _positive_number(capture.get(cv2.CAP_PROP_FPS))
    return int(width), int(height), total_frames, fps if fps is not None else 30.0, fps is None


def open_preflight_capture(source, ffmpeg, cancel_event=None):
    # Local import also keeps metadata normalization usable by legacy adapters.
    from video_decoder import open_video_capture
    return open_video_capture(source, ffmpeg, cancel_event=cancel_event, prefer_hardware=False)


def _sample_aspect_ratio(value: object) -> float | None:
    if not isinstance(value, str) or ':' not in value:
        return None
    try:
        numerator, denominator = (float(part) for part in value.split(':', 1))
        ratio = numerator / denominator
        return ratio if math.isfinite(ratio) and numerator > 0 and denominator > 0 else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _read_video(source: Path, warnings: list[str], ffmpeg: str,
                cancel_event=None) -> tuple[int, int, int | None, float, bool]:
    capture = None
    try:
        capture = open_preflight_capture(source, ffmpeg, cancel_event)
        if not capture.isOpened():
            raise RuntimeError('Could not open this video. Re-encode it as H.264 MP4 and retry.')
        width, height, total_frames, fps, fps_provisional = video_metadata(capture)
        probe = getattr(capture, 'probe', None)
        if isinstance(probe, dict):
            fps_provisional = bool(probe.get('fps_provisional', fps_provisional))
            transfer = str(probe.get('color_transfer') or '').lower()
            if transfer in {'smpte2084', 'arib-std-b67'}:
                raise RuntimeError('HDR video is not supported yet. Convert it to SDR BT.709 H.264 MP4, then retry.')
            sar = _sample_aspect_ratio(probe.get('sample_aspect_ratio'))
            if sar is not None and not math.isclose(sar, 1.0, rel_tol=0, abs_tol=1e-6):
                raise RuntimeError('Anamorphic video with non-square pixels is not supported yet. Convert it to square-pixel SDR video, then retry.')
            range_name = str(probe.get('color_range') or 'unspecified').lower()
            matrix = str(probe.get('color_space') or 'unspecified')
            transfer_name = str(probe.get('color_transfer') or 'unspecified')
            primaries = str(probe.get('color_primaries') or 'unspecified')
            warnings.append(f'SDR color path: range {range_name}, matrix {matrix}, transfer {transfer_name}, primaries {primaries}; FFmpeg decodes to 8-bit BGR and export is limited-range YUV420.')
            if range_name in {'pc', 'jpeg'}:
                warnings.append('Full-range input is normalized during FFmpeg decoding before limited-range YUV420 export.')
            elif range_name not in {'tv', 'mpeg'}:
                warnings.append('Source color range is unspecified; FFmpeg decoder defaults determine the 8-bit conversion.')
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError('The first video frame could not be decoded. Re-encode it as H.264 MP4 and retry.')
        _decoded_size(frame)
        if total_frames is None:
            warnings.append('Frame count is unavailable; total output disk demand cannot be estimated.')
        if fps_provisional:
            warnings.append('Source frame rate is unavailable or invalid; 30 FPS is the provisional fallback. Export probes timing again.')
        return width, height, total_frames, fps, fps_provisional
    except (cv2.error, OSError):
        raise RuntimeError('Video decoding failed. Re-encode it as H.264 MP4 and retry.') from None
    finally:
        if capture is not None:
            capture.release()


def _output_dimensions(width: int, height: int, media_type: str, layout: str, resolution: str) -> tuple[int, int]:
    scale = min(1.0, 1920.0 / width, 1080.0 / height) if resolution == '1080p' else 1.0
    if media_type == 'video':
        # Must match run_video's encoder-safe rounding before layout packing.
        width = max(2, int(round(width * scale / 2.0) * 2))
        height = max(2, int(round(height * scale / 2.0) * 2))
        if layout == 'HALFSBS':
            width = max(4, width // 4 * 4)
        if layout == 'HALFTB':
            height = max(4, height // 4 * 4)
    elif scale < 1:
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return layout_size(width, height, layout)


def _disk_budget(output_dir: Path) -> int:
    try:
        # Unique OS temporary file; automatic close removes only this probe.
        with tempfile.TemporaryFile(prefix='.stereolab-preflight-', dir=output_dir) as probe:
            probe.write(b'\0' * 1024)
            probe.flush()
        free = int(shutil.disk_usage(output_dir).free)
    except (OSError, ValueError, TypeError):
        raise RuntimeError('The output folder is not writable or disk space cannot be checked. Choose a writable folder and check its permissions and free space.') from None
    if free < MINIMUM_FREE_BYTES:
        raise RuntimeError('Insufficient output disk space: free at least 16 MiB before converting, or choose another output drive.')
    return free


def inspect_media(source: Path, media_type: str, layout: str, resolution: str,
                  output_dir: Path, ffmpeg: str | None, *, cancel_event=None) -> dict:
    """Check readable media, dimensions, encoder and destination before inference.

    Only the first video frame is decoded; later corrupt frames can still fail.
    Disk estimates use rough compression assumptions and never guarantee fit.
    The public report includes file stat identity, never filesystem paths.
    """
    if media_type not in {'image', 'video'}:
        raise ValueError('Choose image or video as the source media type.')
    if layout not in LAYOUTS:
        raise ValueError('Choose a supported stereo output layout.')
    if resolution not in {'source', '1080p'}:
        raise ValueError('Choose source or 1080p output resolution.')
    source = Path(source)
    try:
        if not source.is_file():
            raise OSError('not a file')
        source_stat = source.stat()
    except OSError:
        raise RuntimeError('The source file is missing or unreadable. Select the source again and check its permissions.') from None
    warnings: list[str] = []
    if media_type == 'image':
        try:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        except (cv2.error, OSError):
            image = None
        if image is None:
            raise RuntimeError('Could not decode this image. Save it as a standard JPG, PNG or WebP and retry.')
        source_width, source_height = _decoded_size(image)
        total_frames, fps, fps_provisional = 1, None, False
    else:
        if not ffmpeg:
            raise RuntimeError('Install FFmpeg or set VIDEO3D_FFMPEG before converting video.')
        source_width, source_height, total_frames, fps, fps_provisional = _read_video(
            source, warnings, ffmpeg, cancel_event)
        _check_encoder(ffmpeg)
    output_width, output_height = _output_dimensions(source_width, source_height, media_type, layout, resolution)
    free = _disk_budget(Path(output_dir))
    raw_frame_bytes = output_width * output_height * 3
    if media_type == 'image':
        estimated = math.ceil(raw_frame_bytes * 1.1) + 65536
        basis = 'PNG budget: raw RGB bytes plus 10 percent and 64 KiB overhead; content-dependent.'
    elif total_frames is not None:
        estimated = math.ceil(raw_frame_bytes * total_frames * .1 + total_frames / fps * 24000) + 4 * 1024**2
        basis = 'Video budget: 10 percent of raw RGB frames plus 192 kbit/s audio and 4 MiB overhead; codec, quality and content-dependent.'
    else:
        estimated = None
        basis = 'Unknown video frame count; no total disk estimate is available.'
    warnings.append('Output disk demand is an uncertain estimate; compression, content and concurrent disk use can change it.')
    if estimated is not None and estimated + MINIMUM_FREE_BYTES > free:
        warnings.append('Free disk space is below the estimated output budget plus reserve. Free more space or choose another output drive; the export may not fit.')
    return {'source_width': source_width, 'source_height': source_height,
            'output_width': output_width, 'output_height': output_height,
            'total_frames': total_frames, 'fps': fps, 'fps_provisional': fps_provisional, 'estimated_output_bytes': estimated,
            'disk_free_bytes': free, 'warnings': warnings, 'estimate_is_guarantee': False,
            'estimate_basis': basis,
            'source_identity': {'size_bytes': source_stat.st_size, 'modified_ns': source_stat.st_mtime_ns}}
