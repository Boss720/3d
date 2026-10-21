"""Owned, bounded media-tool commands and selected-stream video metadata."""
from __future__ import annotations

from collections import deque
from fractions import Fraction
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time


class MediaCancelled(RuntimeError):
    """The caller cancelled media work; no owned child remains running."""


class MediaCommandError(RuntimeError):
    """A media command exceeded its time/output budget or returned bad metadata."""


def stop_media_process(process):
    """Terminate this exact child, escalating to kill within four seconds."""
    if process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass


def run_media_command(command, cancel_event=None, timeout=30, max_output_bytes=1024 * 1024):
    """Return CompletedProcess with bytes stdout/stderr (including nonzero exits).

    Each output pipe has max_output_bytes capacity. Overflow, timeout and caller
    cancellation terminate the exact owned Popen, join pipe readers and raise
    MediaCommandError or MediaCancelled. Commands must not spawn descendants.
    """
    if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes < 1:
        raise ValueError('Media command requires positive finite time and output limits.')
    if cancel_event is not None and cancel_event.is_set():
        raise MediaCancelled('Media command cancelled.')
    process = subprocess.Popen(list(map(str, command)), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    buffers = [bytearray(), bytearray()]
    overflow = threading.Event()
    failures = deque(maxlen=2)

    def drain(stream, destination):
        try:
            while True:
                part = stream.read(8192)
                if not part:
                    break
                available = max_output_bytes - len(destination)
                destination.extend(part[:available])
                if len(part) > available:
                    overflow.set()
        except (OSError, ValueError) as exc:
            failures.append(exc)
        finally:
            stream.close()

    readers = [threading.Thread(target=drain, args=(stream, buffers[index]), daemon=True)
               for index, stream in enumerate((process.stdout, process.stderr))]
    deadline = time.monotonic() + timeout
    try:
        for reader in readers:
            reader.start()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise MediaCancelled('Media command cancelled.')
            if overflow.is_set():
                raise MediaCommandError('Media command exceeded the bounded output limit.')
            if process.poll() is not None and not any(reader.is_alive() for reader in readers):
                break
            if time.monotonic() >= deadline:
                raise MediaCommandError('Media command timed out.')
            time.sleep(.02)
        if failures:
            raise MediaCommandError('Media command output could not be read.') from failures[0]
        return subprocess.CompletedProcess(command, process.returncode,
                                            bytes(buffers[0]), bytes(buffers[1]))
    finally:
        stop_media_process(process)
        for reader in readers:
            if reader.ident is not None:
                reader.join(timeout=2)
        # Each reader owns its stream; closing a live BufferedReader can block.
        for reader, stream in zip(readers, (process.stdout, process.stderr)):
            if not reader.is_alive():
                stream.close()
        if process.poll() is None or any(reader.is_alive() for reader in readers):
            raise MediaCommandError('Media command cleanup failed: owned process or pipe readers survived the shutdown timeout.')


def _number(value, *, positive=False):
    try:
        value = float(value)
        return value if math.isfinite(value) and (not positive or value > 0) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _rate(value):
    try:
        ratio = Fraction(str(value))
        return ratio if 0 < ratio <= 1000 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe_video(source, ffmpeg, cancel_event=None, timeout=30):
    """Probe v:0 only, without arbitrary tags, packets, frames or metadata blobs.

    total_frames is a duration*output-CFR estimate, never a decoded-frame proof.
    Dimensions describe FFmpeg's default autorotated output. Unknown FPS uses a
    documented provisional 30 FPS; unsupported geometry/rotation fails early.
    """
    binary = Path(ffmpeg)
    if not binary.is_file():
        binary = Path(shutil.which(str(ffmpeg)) or ffmpeg)
    ffprobe = binary.with_name('ffprobe.exe' if binary.suffix.lower() == '.exe' else 'ffprobe')
    if not ffprobe.is_file():
        raise MediaCommandError('FFprobe is required beside FFmpeg for safe video decoding. Install a complete FFmpeg build.')
    result = run_media_command([str(ffprobe), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate,duration,start_time,nb_frames:stream_tags=rotate:stream_side_data=rotation:format=duration,start_time',
        '-of', 'json', str(source)], cancel_event=cancel_event, timeout=timeout, max_output_bytes=65536)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()[-2000:]
        raise MediaCommandError(f'Could not probe this video. Re-encode it as H.264 MP4 and retry. {detail}'.strip())
    try:
        data = json.loads(result.stdout)
        stream = data['streams'][0]
        width, height = int(stream['width']), int(stream['height'])
        fmt = data.get('format', {})
        rotation = _number(next((item['rotation'] for item in stream.get('side_data_list', [])
                                 if 'rotation' in item), stream.get('tags', {}).get('rotate', 0)))
        if rotation is None or abs(rotation / 90 - round(rotation / 90)) > .001:
            raise ValueError('unsupported rotation')
        rotation = int(round(rotation)) % 360
        if rotation in (90, 270):
            width, height = height, width
        if width < 1 or height < 1 or max(width, height) > 8192 or width * height * 3 > 128 * 1024**2:
            raise ValueError('unsupported dimensions')
        avg, nominal = _rate(stream.get('avg_frame_rate')), _rate(stream.get('r_frame_rate'))
        rate = avg or nominal or Fraction(30)
        video_start = _number(stream.get('start_time'))
        format_start = _number(fmt.get('start_time'))
        video_start = video_start if video_start is not None else (format_start or 0.)
        duration = _number(stream.get('duration'), positive=True)
        count = max(1, int(math.floor(duration * float(rate) + .5))) if duration else None
        return {'width': width, 'height': height, 'fps': float(rate), 'fps_ratio': str(rate),
                'fps_provisional': not bool(avg or nominal), 'avg_frame_rate': stream.get('avg_frame_rate'),
                'r_frame_rate': stream.get('r_frame_rate'), 'total_frames': count,
                'source_total_frames': stream.get('nb_frames'), 'duration_seconds': duration,
                'format_duration_seconds': _number(fmt.get('duration'), positive=True),
                'video_start_seconds': video_start, 'format_start_seconds': format_start or 0.,
                'rotation_degrees': rotation}
    except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
        raise MediaCommandError('Video metadata or display geometry is unsupported. Re-encode as a standard H.264 MP4 (maximum 8192 pixels per dimension).') from exc
