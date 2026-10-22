"""Owned FFmpeg software/D3D11VA capture with timestamp-aware CFR output."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from gpu_hardware import probe_hardware
from media_process import MediaCancelled, external_process_environment, fps_passthrough_args, probe_video


DecoderCancelled = MediaCancelled


class DecoderCleanupError(RuntimeError):
    """An owned decoder process or pipe reader survived bounded shutdown."""


class SoftwareCapture:
    def __init__(self, capture, reason):
        self.capture = capture
        self.decoder_info = {'backend': 'OpenCV', 'hardware': False, 'adapter_id': None,
                             'adapter_name': None, 'fallback_reason': reason}

    def isOpened(self):
        return self.capture.isOpened()

    def get(self, prop):
        return self.capture.get(prop)

    def read(self):
        return self.capture.read()

    def stop(self):
        self.capture.release()

    def release(self):
        self.capture.release()


def select_amd_adapter(hardware):
    """DXGI device ordinals, never assumed from display labels or NVIDIA IDs."""
    return next((item for item in hardware.get('adapters', [])
                 if item.get('vendor_id') == 0x1002 and not item.get('software')
                 and isinstance(item.get('id'), int) and item['id'] >= 0), None)


def decoder_command(ffmpeg, source, adapter_id=None, fps=None):
    """Normalize source PTS before CFR sampling; raw frames carry no timestamps."""
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin']
    filters = []
    if adapter_id is not None:
        command += ['-hwaccel', 'd3d11va', '-hwaccel_device', str(adapter_id),
                    '-hwaccel_output_format', 'd3d11']
        filters.append('hwdownload,format=nv12,format=bgr24')
    if fps is not None:
        filters += ['setpts=PTS-STARTPTS', f'fps=fps={fps}:round=near']
    command += ['-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn']
    if filters:
        command += ['-vf', ','.join(filters)]
    return command + fps_passthrough_args(ffmpeg) + ['-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']


class D3D11Capture:
    """Owns a process and two bounded pipe readers; does not seek.

    The historical class name remains for test/API compatibility. adapter=None
    selects software decoding with identical timestamp normalization.
    """
    def __init__(self, source, ffmpeg, metadata, adapter, cancel_event=None,
                 popen=None, read_timeout=30):
        self.probe = dict(metadata) if isinstance(metadata, dict) else {
            'width': metadata[0], 'height': metadata[1], 'total_frames': metadata[2],
            'fps': metadata[3], 'fps_ratio': str(metadata[3]), 'video_start_seconds': 0.,
            'format_start_seconds': 0., 'fps_provisional': metadata[4]}
        self.width, self.height = int(self.probe['width']), int(self.probe['height'])
        self.count, self.fps = self.probe['total_frames'], float(self.probe['fps'])
        if self.width < 1 or self.height < 1 or max(self.width, self.height) > 8192 or self.width * self.height * 3 > 128 * 1024**2:
            raise ValueError('Video display dimensions exceed the safe decoder limit; re-encode the source.')
        if not np.isfinite(self.fps) or not 0 < self.fps <= 1000 or not np.isfinite(read_timeout) or read_timeout <= 0:
            raise ValueError('Invalid decoder frame rate or read timeout.')
        self.cancel_event = cancel_event
        self.read_timeout = read_timeout
        self.position = 0
        self._pending = None
        self._closed = False
        self._stopped = threading.Event()
        self._stop_lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._stop_requested = False
        self._frames = queue.Queue(maxsize=2)
        self._errors = deque(maxlen=32)
        self.decoder_info = {'backend': 'FFmpeg D3D11VA' if adapter else 'FFmpeg software',
                             'hardware': bool(adapter),
                             'adapter_id': adapter['id'] if adapter else None,
                             'adapter_name': adapter.get('name') if adapter else None,
                             'fallback_reason': None,
                             'timing_policy': 'normalize video PTS to zero; resample timestamps to CFR at probed average FPS'}
        self.command = decoder_command(ffmpeg, source, adapter['id'] if adapter else None,
                                       self.probe.get('fps_ratio') or str(self.fps))
        self.process = (popen or subprocess.Popen)(self.command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=external_process_environment(),
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._threads = []
        try:
            for target in (self._drain_stderr, self._decode):
                thread = threading.Thread(target=target, daemon=True)
                self._threads.append(thread)
                thread.start()
            self._pending = self._next_frame()
            if self._pending is None:
                raise RuntimeError('Video decoder produced no frames.')
        except BaseException:
            self.release()
            raise

    def _drain_stderr(self):
        try:
            while not self._stopped.is_set():
                part = self.process.stderr.read(1024)
                if not part:
                    break
                self._errors.append(part)
        except (OSError, ValueError):
            pass
        finally:
            self._close_pipe(self.process.stderr)

    @staticmethod
    def _close_pipe(stream):
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def _deliver(self, value):
        while not self._stopped.is_set():
            try:
                self._frames.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def _decode(self):
        size = self.width * self.height * 3
        try:
            while not self._stopped.is_set():
                data = bytearray()
                while len(data) < size:
                    part = self.process.stdout.read(size - len(data))
                    if not part:
                        break
                    data.extend(part)
                if len(data) != size:
                    if data:
                        raise RuntimeError('Video decoder returned an incomplete raw frame.')
                    try:
                        result = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('Video decoder did not exit after end of stream.') from None
                    if result:
                        raise RuntimeError(f'Video decoder failed (exit {result}).')
                    self._deliver(None)
                    return
                self._deliver(np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3).copy())
        except Exception as exc:
            self._deliver(exc)
        finally:
            self._close_pipe(self.process.stdout)

    def _next_frame(self):
        deadline = time.monotonic() + self.read_timeout
        while not self._closed:
            if self.cancel_event is not None and self.cancel_event.is_set():
                self.release()
                raise DecoderCancelled('Video decoding cancelled.')
            try:
                value = self._frames.get(timeout=.1)
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self.release()
                    raise DecoderCancelled('Video decoding cancelled.')
                if self._closed:
                    return None
                if isinstance(value, Exception):
                    detail = b''.join(self._errors).decode('utf-8', errors='replace').strip()
                    self.release()
                    raise RuntimeError(f'{value} {detail}'.strip()) from value
                return value
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self.release()
                    raise RuntimeError('Video decoder timed out waiting for a frame.')
        return None

    def isOpened(self):
        return not self._closed

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FRAME_COUNT: self.count or 0, cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_POS_FRAMES: self.position,
                cv2.CAP_PROP_POS_MSEC: max(0, self.position - 1) * 1000 / self.fps}.get(prop, 0)

    def read(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.release()
            raise DecoderCancelled('Video decoding cancelled.')
        if self._closed:
            return False, None
        frame = self._pending
        self._pending = None
        if frame is None:
            frame = self._next_frame()
        if frame is None:
            self.release()
            return False, None
        self.position += 1
        return True, frame

    def stop(self):
        """Interrupt owned pipe I/O; safe from a pipeline control thread."""
        with self._stop_lock:
            self._closed = True
            self._stopped.set()
            if not self._stop_requested:
                self._stop_requested = True
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                    except OSError:
                        pass

    def release(self):
        # Tests constructing __new__ retain the historical cleanup seam.
        if not hasattr(self, '_release_lock'):
            self._release_lock = threading.Lock()
            self._stop_lock = threading.Lock()
            self._stop_requested = False
        with self._release_lock:
            if self.decoder_info.get('cleanup_complete'):
                return
            self._release()

    def _release(self):
        self.stop()
        proc = self.process
        if proc.poll() is None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
                try:
                    proc.wait(timeout=2)
                except (subprocess.TimeoutExpired, OSError):
                    pass
            except OSError:
                pass
        for thread in self._threads:
            if thread is not threading.current_thread() and thread.ident is not None:
                thread.join(timeout=2)
        for index, stream in enumerate((proc.stderr, proc.stdout)):
            if index >= len(self._threads) or not self._threads[index].is_alive():
                self._close_pipe(stream)
        complete = not any(thread.is_alive() for thread in self._threads) and proc.poll() is not None
        self.decoder_info['cleanup_complete'] = complete
        self.decoder_info['cleanup_error'] = (
            None if complete else 'Decoder process or pipe reader did not stop within the shutdown timeout.')
        if not complete:
            raise DecoderCleanupError(self.decoder_info['cleanup_error'])


def _legacy_metadata(capture):
    return (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS), False)


def open_video_capture(source: Path, ffmpeg: str, *, software_capture_factory=None,
                       cancel_event=None, prefer_hardware=True):
    """Own FFmpeg capture; fallback is allowed only before a first frame succeeds.

    software_capture_factory is an explicit legacy test seam. Production callers
    omit it and never enter native OpenCV open/read code.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise DecoderCancelled('Video decoding cancelled.')
    software = None
    if software_capture_factory is not None:
        software = software_capture_factory(str(source))
        if cancel_event is not None and cancel_event.is_set():
            software.release()
            raise DecoderCancelled('Video decoding cancelled.')
        if not prefer_hardware:
            return SoftwareCapture(software, None)
        if not software.isOpened():
            return SoftwareCapture(software, 'OpenCV could not open the source.')
        metadata = _legacy_metadata(software)
    else:
        metadata = probe_video(source, ffmpeg, cancel_event=cancel_event)

    reason = None
    if prefer_hardware:
        hardware = probe_hardware()
        adapter = select_amd_adapter(hardware)
        if adapter is None:
            reason = hardware.get('error') or 'No physical AMD DXGI adapter found.'
        elif isinstance(metadata, dict) and metadata.get('rotation_degrees'):
            reason = 'Rotated video uses software decoding to preserve display orientation.'
        else:
            try:
                capture = D3D11Capture(source, ffmpeg, metadata, adapter, cancel_event)
            except DecoderCancelled:
                if software is not None:
                    software.release()
                raise
            except DecoderCleanupError:
                if software is not None:
                    software.release()
                raise
            except (OSError, RuntimeError, ValueError) as exc:
                reason = str(exc)
            else:
                if software is not None:
                    software.release()
                return capture
    if software is not None:
        return SoftwareCapture(software, reason)
    if cancel_event is not None and cancel_event.is_set():
        raise DecoderCancelled('Video decoding cancelled.')
    capture = D3D11Capture(source, ffmpeg, metadata, None, cancel_event)
    capture.decoder_info['fallback_reason'] = reason
    return capture
