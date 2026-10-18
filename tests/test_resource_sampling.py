"""Best-effort process and opt-in device memory sampling tests."""
from __future__ import annotations

import ctypes
import subprocess
import sys
import threading
import types
import unittest
from unittest.mock import patch

import resource_sampling


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class ProcessSnapshotTests(unittest.TestCase):
    def test_windows_memory_counters_report_current_and_peak_working_set(self):
        def fill_counters(handle, pointer, size):
            counters = pointer._obj
            counters.WorkingSetSize = 123456
            counters.PeakWorkingSetSize = 234567
            return 1

        kernel32 = types.SimpleNamespace(GetCurrentProcess=FakeFunction(lambda: 1))
        psapi = types.SimpleNamespace(GetProcessMemoryInfo=FakeFunction(fill_counters))

        def load_library(name, **kwargs):
            return {"kernel32": kernel32, "psapi": psapi}[name]

        with patch.object(resource_sampling.sys, "platform", "win32"), \
             patch.object(resource_sampling.ctypes, "WinDLL", side_effect=load_library, create=True), \
             patch.object(resource_sampling.time, "process_time", return_value=3.25):
            result = resource_sampling.process_snapshot()

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["pid"], resource_sampling.os.getpid())
        self.assertEqual(result["cpu_time_seconds"], 3.25)
        self.assertEqual(result["working_set_bytes"], 123456)
        self.assertEqual(result["peak_working_set_bytes"], 234567)
        self.assertIn("GetProcessMemoryInfo", result["memory_method"])
        self.assertIsNone(result["memory_error"])

    def test_windows_api_failure_is_explicit_and_leaves_memory_fields_null(self):
        kernel32 = types.SimpleNamespace(GetCurrentProcess=FakeFunction(lambda: 1))
        psapi = types.SimpleNamespace(GetProcessMemoryInfo=FakeFunction(lambda *args: 0))

        def load_library(name, **kwargs):
            return {"kernel32": kernel32, "psapi": psapi}[name]

        with patch.object(resource_sampling.sys, "platform", "win32"), \
             patch.object(resource_sampling.ctypes, "WinDLL", side_effect=load_library, create=True), \
             patch.object(resource_sampling.ctypes, "get_last_error", return_value=5, create=True):
            result = resource_sampling.process_snapshot()

        self.assertIsNone(result["working_set_bytes"])
        self.assertIsNone(result["peak_working_set_bytes"])
        self.assertIn("5", result["memory_error"])

    def test_linux_resource_fallback_documents_kibibyte_conversion(self):
        resource = types.SimpleNamespace(
            RUSAGE_SELF=0,
            getrusage=lambda who: types.SimpleNamespace(ru_maxrss=123),
        )
        with patch.object(resource_sampling.sys, "platform", "linux"), \
             patch.object(resource_sampling.time, "process_time", return_value=1.5), \
             patch.dict(sys.modules, {"resource": resource}):
            result = resource_sampling.process_snapshot()

        self.assertEqual(result["cpu_time_seconds"], 1.5)
        self.assertIsNone(result["working_set_bytes"])
        self.assertEqual(result["peak_working_set_bytes"], 123 * 1024)
        self.assertIn("KiB", result["memory_method"])


class DeviceMemorySamplerTests(unittest.TestCase):
    def test_constructor_does_not_start_process_or_thread(self):
        with patch.object(resource_sampling.subprocess, "run") as run:
            sampler = resource_sampling.DeviceMemorySampler()
            snapshot = sampler.snapshot()
            self.assertFalse(snapshot["started"])
            self.assertEqual(snapshot["sample_count"], 0)
            run.assert_not_called()
            sampler.close()

    def test_missing_nvidia_smi_is_reported_without_starting_a_thread(self):
        with patch.object(resource_sampling.shutil, "which", return_value=None):
            with resource_sampling.DeviceMemorySampler() as sampler:
                pass
            snapshot = sampler.snapshot()

        self.assertEqual(snapshot["error_count"], 1)
        self.assertIn("nvidia-smi", snapshot["last_error"])
        self.assertEqual(snapshot["sample_count"], 0)

    def test_nvidia_smi_without_a_gpu_is_reported_as_an_error(self):
        invoked = threading.Event()

        def run(command, **kwargs):
            invoked.set()
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="No devices were found")

        with patch.object(resource_sampling.shutil, "which", return_value="nvidia-smi"), \
             patch.object(resource_sampling.subprocess, "run", side_effect=run):
            sampler = resource_sampling.DeviceMemorySampler(interval_seconds=10).start()
            self.assertTrue(invoked.wait(1))
            snapshot = sampler.close()

        self.assertEqual(snapshot["sample_count"], 0)
        self.assertEqual(snapshot["error_count"], 1)
        self.assertIn("No devices were found", snapshot["last_error"])

    def test_successful_query_reports_whole_device_used_memory_and_double_close(self):
        invoked = threading.Event()

        def run(command, **kwargs):
            self.assertLessEqual(kwargs["timeout"], 2)
            self.assertIn("memory.used", command[1])
            invoked.set()
            return subprocess.CompletedProcess(command, 0, stdout="100\n250\n", stderr="")

        with patch.object(resource_sampling.shutil, "which", return_value="nvidia-smi"), \
             patch.object(resource_sampling.subprocess, "run", side_effect=run):
            sampler = resource_sampling.DeviceMemorySampler(interval_seconds=10).start()
            self.assertTrue(invoked.wait(1))
            snapshot = sampler.close()
            second_close = sampler.close()

        self.assertEqual(snapshot["sample_count"], 1)
        self.assertEqual(snapshot["sampled_high_mib"], 250)
        self.assertEqual(snapshot["sample_min_mib"], 250)
        self.assertEqual(snapshot["sample_max_mib"], 250)
        self.assertEqual(snapshot["errors"], 0)
        self.assertIn("other processes", snapshot["memory_scope"])
        self.assertIn("not process allocation", snapshot["memory_scope"])
        self.assertFalse(second_close.get("shutdown_error"))

    def test_timeout_is_bounded_and_reported(self):
        invoked = threading.Event()

        def run(command, **kwargs):
            invoked.set()
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with patch.object(resource_sampling.shutil, "which", return_value="nvidia-smi"), \
             patch.object(resource_sampling.subprocess, "run", side_effect=run):
            sampler = resource_sampling.DeviceMemorySampler(interval_seconds=10).start()
            self.assertTrue(invoked.wait(1))
            snapshot = sampler.close()

        self.assertEqual(snapshot["sample_count"], 0)
        self.assertEqual(snapshot["error_count"], 1)
        self.assertIn("timed out", snapshot["last_error"].lower())
        self.assertLessEqual(snapshot["timeout_seconds"], 2)

    def test_sample_cache_is_capped_and_summary_tracks_all_samples(self):
        sampler = resource_sampling.DeviceMemorySampler()
        for value in range(140):
            sampler._record_sample([value, value + 100])

        snapshot = sampler.snapshot()
        self.assertEqual(snapshot["sample_count"], 140)
        self.assertEqual(len(snapshot["samples_mib"]), 128)
        self.assertEqual(snapshot["sample_min_mib"], 100)
        self.assertEqual(snapshot["sample_max_mib"], 239)
        self.assertEqual(snapshot["sampled_high_mib"], 239)

    def test_error_text_is_bounded(self):
        sampler = resource_sampling.DeviceMemorySampler()
        sampler._record_error("detail" * 500)
        self.assertLessEqual(len(sampler.snapshot()["last_error"]), 240)


if __name__ == "__main__":
    unittest.main()
