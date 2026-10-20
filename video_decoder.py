"""Bounded FFmpeg D3D11VA capture with startup-only OpenCV fallback."""
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
from preflight import video_metadata


class DecoderCancelled(RuntimeError):
    """Caller should translate this into its normal cancellation outcome."""


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

    def release(self):
        self.capture.release()


def select_amd_adapter(hardware):
    """DXGI device ordinals, never assumed from display labels or NVIDIA IDs."""
    return next((item for item in hardware.get('adapters', [])
                 if item.get('vendor_id') == 0x1002 and not item.get('software')
                 and isinstance(item.get('id'), int) and item['id'] >= 0), None)


def decoder_command(ffmpeg, source, adapter_id):
    # Explicit hardware output plus hwdownload fails rather than falsely claiming
    # hardware decoding if FFmpeg silently chooses a software decoder.
    return [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-hwaccel', 'd3d11va', '-hwaccel_device', str(adapter_id),
            '-hwaccel_output_format', 'd3d11', '-i', str(source),
            '-map', '0:v:0', '-an', '-sn', '-dn', '-vf',
            'hwdownload,format=nv12,format=bgr24', '-fps_mode', 'passthrough',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']


class D3D11Capture:
    """Owns a process and two bounded pipe readers; does not seek."""
    def __init__(self, source, ffmpeg, metadata, adapter, cancel_event=None,
                 popen=None, read_timeout=30):
        self.width, self.height, self.count, self.fps, _ = metadata
        self.cancel_event = cancel_event
        self.read_timeout = read_timeout
        self.position = 0
        self._pending = None
        self._closed = False
        self._stopped = threading.Event()
        self._frames = queue.Queue(maxsize=2)
        self._errors = deque(maxlen=32)
        self.decoder_info = {'backend': 'FFmpeg D3D11VA', 'hardware': True,
                             'adapter_id': adapter['id'], 'adapter_name': adapter.get('name'),
                             'fallback_reason': None}
        self.command = decoder_command(ffmpeg, source, adapter['id'])
        self.process = (popen or subprocess.Popen)(self.command, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self._threads = []
        try:
            for target in (self._drain_stderr, self._decode):
                thread = threading.Thread(target=target, daemon=True)
                self._threads.append(thread)
                thread.start()
            self._pending = self._next_frame()
            if self._pending is None:
                raise RuntimeError('Hardware decoder produced no frames.')
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
                        raise RuntimeError('Hardware decoder returned an incomplete raw frame.')
                    try:
                        result = self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('Hardware decoder did not exit after end of stream.') from None
                    if result:
                        raise RuntimeError(f'Hardware decoder failed (exit {result}).')
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
                if isinstance(value, Exception):
                    detail = b''.join(self._errors).decode('utf-8', errors='replace').strip()
                    raise RuntimeError(f'{value} {detail}'.strip()) from value
                return value
            except queue.Empty:
                if time.monotonic() >= deadline:
                    self.release()
                    raise RuntimeError('Hardware decoder timed out waiting for a frame.')
        return None

    def isOpened(self):
        return not self._closed

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.width, cv2.CAP_PROP_FRAME_HEIGHT: self.height,
                cv2.CAP_PROP_FRAME_COUNT: self.count or 0, cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_POS_FRAMES: self.position,
                cv2.CAP_PROP_POS_MSEC: max(0, self.position - 1) * 1000 / self.fps}.get(prop, 0)

    def read(self):
        if self._closed:
            return False, None
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.release()
            raise DecoderCancelled('Video decoding cancelled.')
        frame = self._pending
        self._pending = None
        if frame is None:
            frame = self._next_frame()
        if frame is None:
            self._closed = True
            return False, None
        self.position += 1
        return True, frame

    def release(self):
        self._closed = True
        self._stopped.set()
        proc = self.process
        if proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
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
        # BufferedReader.close() takes its read lock. If the OS could not kill
        # the child, a pipe reader can still hold that lock indefinitely. Each
        # reader closes its own pipe on exit; only close idle pipes here.
        for index, stream in enumerate((proc.stderr, proc.stdout)):
            if index >= len(self._threads) or not self._threads[index].is_alive():
                self._close_pipe(stream)
        live_readers = any(thread.is_alive() for thread in self._threads)
        complete = not live_readers and proc.poll() is not None
        self.decoder_info['cleanup_complete'] = complete
        self.decoder_info['cleanup_error'] = (
            None if complete else 'Decoder process or pipe reader did not stop within the shutdown timeout.')


def open_video_capture(source: Path, ffmpeg: str, *, software_capture_factory=None,
                       cancel_event=None, prefer_hardware=True):
    """Probe one hardware frame, preserving it; only startup can fall back."""
    if cancel_event is not None and cancel_event.is_set():
        raise DecoderCancelled('Video decoding cancelled.')
    software = (software_capture_factory or cv2.VideoCapture)(str(source))
    if cancel_event is not None and cancel_event.is_set():
        software.release()
        raise DecoderCancelled('Video decoding cancelled.')
    if not prefer_hardware:
        return SoftwareCapture(software, None)
    if not software.isOpened():
        return SoftwareCapture(software, 'OpenCV could not open the source.')
    hardware = probe_hardware()
    adapter = select_amd_adapter(hardware)
    if adapter is None:
        return SoftwareCapture(software, hardware.get('error') or 'No physical AMD DXGI adapter found.')
    try:
        capture = D3D11Capture(source, ffmpeg, video_metadata(software), adapter, cancel_event)
    except (OSError, RuntimeError, ValueError) as exc:
        if cancel_event is not None and cancel_event.is_set():
            software.release()
            raise
        return SoftwareCapture(software, str(exc))
    software.release()
    return capture
