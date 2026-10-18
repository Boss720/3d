"""Best-effort current-process and opt-in whole-device resource sampling."""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque


SCHEMA_VERSION = 1
MAX_DEVICE_SAMPLES = 128
DEVICE_TIMEOUT_SECONDS = 2.0
SHUTDOWN_JOIN_SECONDS = 3.0
_MEMORY_SCOPE = ("Per-poll maximum nvidia-smi memory.used across reported NVIDIA devices; each value includes "
                 "other processes. Sampled high, not process allocation or an exact VRAM peak.")


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_working_set() -> tuple[int, int]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    get_memory = psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        error = ctypes.get_last_error()
        raise OSError(error, "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def process_snapshot() -> dict:
    """Return current server-process CPU time and best-effort working-set data."""
    result = {"schema_version": SCHEMA_VERSION, "pid": os.getpid(), "cpu_time_seconds": None,
              "cpu_time_error": None, "working_set_bytes": None, "peak_working_set_bytes": None,
              "memory_method": None, "memory_error": None}
    try:
        result["cpu_time_seconds"] = float(time.process_time())
    except Exception as exc:
        result["cpu_time_error"] = f"{type(exc).__name__}: {exc}"[:240]
    try:
        if sys.platform == "win32":
            result["memory_method"] = "psapi.GetProcessMemoryInfo current and peak working set (bytes)"
            result["working_set_bytes"], result["peak_working_set_bytes"] = _windows_working_set()
        elif sys.platform.startswith("linux") or sys.platform == "darwin":
            import resource
            peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            multiplier = 1 if sys.platform == "darwin" else 1024
            unit = "bytes" if multiplier == 1 else "KiB"
            if not math.isfinite(peak) or peak < 0:
                raise ValueError("ru_maxrss returned an invalid value")
            result["peak_working_set_bytes"] = int(peak * multiplier)
            result["memory_method"] = f"resource.getrusage(RUSAGE_SELF).ru_maxrss ({unit}; peak only)"
        else:
            raise OSError(f"No working-set API configured for {sys.platform}.")
    except Exception as exc:
        result["memory_error"] = f"{type(exc).__name__}: {exc}"[:240]
    return result


class DeviceMemorySampler:
    """Opt-in, bounded sampler of whole-device NVIDIA memory.used readings."""
    def __init__(self, interval_seconds: float = 1.0, timeout_seconds: float = DEVICE_TIMEOUT_SECONDS):
        if isinstance(interval_seconds, bool) or not math.isfinite(interval_seconds) or not 0.1 <= interval_seconds <= 60:
            raise ValueError("Sampling interval must be between 0.1 and 60 seconds.")
        if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= DEVICE_TIMEOUT_SECONDS:
            raise ValueError("nvidia-smi timeout must be greater than 0 and at most 2 seconds.")
        self.interval_seconds = float(interval_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self._samples = deque(maxlen=MAX_DEVICE_SAMPLES)
        self._sample_count = self._error_count = 0
        self._minimum = self._maximum = self._device_count = None
        self._last_error = self._shutdown_error = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._binary = None
        self._started = False

    def start(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop = threading.Event()
            self._started = True
            self._shutdown_error = None
            try:
                self._binary = shutil.which("nvidia-smi")
            except Exception as exc:
                self._record_error_locked(f"Could not locate nvidia-smi: {exc}")
                return self
            if not self._binary:
                self._record_error_locked("nvidia-smi was not found; device sampling is unavailable.")
                return self
            self._thread = threading.Thread(target=self._run, name="device-memory-sampler", daemon=True)
            try:
                self._thread.start()
            except Exception as exc:
                self._thread = None
                self._record_error_locked(f"Could not start device sampler: {exc}")
        return self

    def _record_error_locked(self, message: str) -> None:
        self._error_count += 1
        self._last_error = str(message)[:240]

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._record_error_locked(message)

    def _record_sample(self, device_values: list[int] | tuple[int, ...]) -> None:
        values = [int(value) for value in device_values]
        if not values or any(value < 0 for value in values):
            self._record_error("nvidia-smi returned no valid memory.used values.")
            return
        used = max(values)
        with self._lock:
            self._samples.append(used)
            self._sample_count += 1
            self._device_count = len(values)
            self._minimum = used if self._minimum is None else min(self._minimum, used)
            self._maximum = used if self._maximum is None else max(self._maximum, used)

    def _sample_once(self) -> None:
        try:
            command = [self._binary, "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=min(self.timeout_seconds, DEVICE_TIMEOUT_SECONDS), check=False,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode:
                detail = (result.stderr or "").strip()
                self._record_error(f"nvidia-smi exited {result.returncode}: {detail or 'no diagnostic'}")
                return
            values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            self._record_sample(values)
        except subprocess.TimeoutExpired:
            self._record_error(f"nvidia-smi timed out after at most {self.timeout_seconds:g} seconds.")
        except Exception as exc:
            self._record_error(f"{type(exc).__name__}: {exc}")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            if self._stop.wait(self.interval_seconds):
                break

    def snapshot(self) -> dict:
        with self._lock:
            samples = list(self._samples)
            return {"schema_version": SCHEMA_VERSION, "started": self._started,
                    "available": self._sample_count > 0, "unit": "MiB",
                    "sample_count": self._sample_count, "retained_sample_count": len(samples),
                    "sample_limit": MAX_DEVICE_SAMPLES, "samples_mib": samples,
                    "sample_min_mib": self._minimum, "sample_max_mib": self._maximum,
                    "sampled_high_mib": self._maximum, "device_count": self._device_count,
                    "interval_seconds": self.interval_seconds, "timeout_seconds": self.timeout_seconds,
                    "error_count": self._error_count, "errors": self._error_count,
                    "last_error": self._last_error, "shutdown_error": self._shutdown_error,
                    "memory_scope": _MEMORY_SCOPE}

    def close(self) -> dict:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=SHUTDOWN_JOIN_SECONDS)
            if thread.is_alive():
                with self._lock:
                    self._shutdown_error = f"Sampler thread did not stop within {SHUTDOWN_JOIN_SECONDS:g} seconds."
        return self.snapshot()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
