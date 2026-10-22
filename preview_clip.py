"""Finite, cancellation-aware extraction of a CPU video preview proxy."""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import time
from pathlib import Path


class ClipCancelled(Exception):
    """The owning preview job requested cancellation."""


def _finite_number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite number.') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be a finite number.')
    return result


def _stop_owned_process(process):
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        pass  # The child may have exited between poll and terminate.
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('The preview extraction process did not stop after termination and kill.') from exc


def _stderr_tail(stream):
    stream.seek(0, os.SEEK_END)
    stream.seek(max(0, stream.tell() - 8192))
    return stream.read(8192).decode('utf-8', errors='replace').strip()


def prepare_preview_clip(ffmpeg, source, destination, start_seconds, duration_seconds,
                         source_fps, cancel_event, on_process=None, timeout_seconds=30.0) -> Path:
    """Extract 1–5 seconds with CPU H.264 and optional AAC audio.

    Invalid/nonpositive source FPS explicitly falls back to 24; valid rates use
    1–24 FPS. Very slow sources duplicate frames so a single frame cannot make
    the encoded clip longer than the selected interval. Stderr is
    spooled to disk, with only its final 8192 bytes read into memory on failure.
    Destination ownership is acquired with exclusive creation before launch.
    """
    start = _finite_number(start_seconds, 'Clip start')
    duration = _finite_number(duration_seconds, 'Clip duration')
    timeout = _finite_number(timeout_seconds, 'Extraction timeout')
    if start < 0:
        raise ValueError('Clip start must be at least zero seconds.')
    if not 1 <= duration <= 5:
        raise ValueError('Choose a clip duration from 1 to 5 seconds.')
    if timeout <= 0:
        raise ValueError('Extraction timeout must be positive.')
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise ValueError('The preview source video does not exist.')
    if source.resolve() == destination.resolve():
        raise ValueError('The preview proxy must have a separate output path.')
    if destination.exists() or destination.is_symlink():
        raise ValueError('The preview proxy output already exists. Choose a new output path.')
    if not destination.parent.is_dir():
        raise ValueError('The preview proxy output directory does not exist.')
    if cancel_event.is_set():
        raise ClipCancelled('Preview extraction canceled.')
    try:
        fps = _finite_number(source_fps, 'Source FPS')
    except ValueError:
        fps = 24.0
    fps = max(1.0, min(24.0, fps)) if fps > 0 else 24.0
    frame_limit = min(120, max(1, math.ceil(duration * fps)))
    # The aspect-preserving scale factor never exceeds 1. Round each result down
    # to an even dimension for yuv420p; unusably tiny sources fail in FFmpeg.
    scale = 'min(1,min(960/iw,540/ih))'
    filters = (f"scale=w='trunc(iw*{scale}/2)*2':h='trunc(ih*{scale}/2)*2',"
               f'fps={fps:.12g},setpts=PTS-STARTPTS')
    command = [os.fspath(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
               '-ss', f'{start:.12g}', '-i', str(source), '-t', f'{duration:.12g}',
               '-map', '0:v:0', '-map', '0:a:0?', '-sn', '-dn',
               '-vf', filters, '-frames:v', str(frame_limit),
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '22', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-movflags', '+faststart', '-f', 'mp4', str(destination)]
    # An exclusive reservation closes the exists-check race without ever owning
    # or deleting a preexisting destination, including another preview's output.
    try:
        with destination.open('xb'):
            pass
    except FileExistsError as exc:
        raise ValueError('The preview proxy output already exists. Choose a new output path.') from exc
    process = None
    succeeded = False
    try:
        deadline = time.monotonic() + timeout
        with tempfile.TemporaryFile(mode='w+b') as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=errors, shell=False,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            if on_process is not None:
                on_process(process)
            while True:
                if cancel_event.is_set():
                    raise ClipCancelled('Preview extraction canceled.')
                return_code = process.poll()
                if return_code is not None:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'Preview extraction timed out after {timeout:g} seconds. Try a shorter or simpler source.')
                cancel_event.wait(.05)
            if cancel_event.is_set():
                raise ClipCancelled('Preview extraction canceled.')
            if return_code != 0:
                detail = _stderr_tail(errors)
                raise RuntimeError(f'Preview extraction failed: {detail or "FFmpeg exited with status " + str(return_code)}')
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError('Preview extraction produced no video. Choose an earlier timestamp or another source.')
        succeeded = True
        return destination
    except OSError as exc:
        raise RuntimeError(f'Preview extraction could not run: {exc}') from exc
    finally:
        try:
            if process is not None and not succeeded:
                _stop_owned_process(process)
        finally:
            try:
                if on_process is not None:
                    on_process(None)
            except BaseException:
                succeeded = False
                raise
            finally:
                if not succeeded:
                    destination.unlink(missing_ok=True)
