"""Trust-boundary and pairwise tests for offline variance capture analysis."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from scripts import analyze_variance_captures as analyzer


def _array_hash(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


class VarianceCaptureAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.root = analyzer.ROOT / ".runtime" / "variance-analysis-verification" / ("test-" + os.urandom(6).hex())
        self.root.mkdir(parents=True)
        self.run = self.root / "run"
        (self.run / "captures").mkdir(parents=True)
        self.arrays = [np.full((2, 3), i, dtype=np.float32) for i in (1, 2, 2)]
        records = []
        for index, array in enumerate(self.arrays, 1):
            path = self.run / "captures" / f"repeat-{index}.npz"
            np.savez_compressed(path, normalized_model_depth=array)
            records.append({"repeat": index, "capture": {
                "path": str(path), "sha256": analyzer.sha256_file(path),
                "size_bytes": path.stat().st_size,
                "stages": {"normalized_model_depth": {
                    "shape": list(array.shape), "dtype": array.dtype.str,
                    "nbytes": array.nbytes, "sha256": _array_hash(array)}}}})
        self.report = self.run / "report.json"
        fixture = {
            "status": "completed_diagnostic", "repeats_max": 3,
            "repeats_completed": records, "settings": {"fps": 12, "frames": 2},
            "encoder_experiment": {
                "quality_profile": "fast",
                "profile": {
                    "video_args": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"],
                    "software_fallback_args": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"],
                },
                "encoders": {
                    "libx264": {"runs": [{"command": ["ffmpeg", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p"]}]}
                },
            },
        }
        self.report.write_text(json.dumps(fixture), encoding="utf-8")

    def tearDown(self):
        for path in sorted(self.root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                path.rmdir()
        self.root.rmdir()

    def test_validates_capture_then_compares_every_stage_pair(self):
        report, arrays = analyzer.load_and_validate_report(self.report)
        result = analyzer.compare_all_pairs(arrays)
        self.assertEqual([(item["reference_repeat"], item["candidate_repeat"])
                          for item in result], [(1, 2), (1, 3), (2, 3)])
        self.assertTrue(result[0]["stages"]["normalized_model_depth"]["exact_equal"] is False)
        self.assertTrue(result[2]["stages"]["normalized_model_depth"]["exact_equal"])

    def test_rejects_capture_outside_run_and_symlinked_capture(self):
        data = json.loads(self.report.read_text(encoding="utf-8"))
        outside = self.root / "outside.npz"
        outside.write_bytes((self.run / "captures" / "repeat-1.npz").read_bytes())
        data["repeats_completed"][0]["capture"]["path"] = str(outside)
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "confined"):
            analyzer.load_and_validate_report(self.report)

        data["repeats_completed"][0]["capture"]["path"] = str(self.run / "captures" / "linked.npz")
        linked = self.run / "captures" / "linked.npz"
        linked.write_bytes((self.run / "captures" / "repeat-1.npz").read_bytes())
        self.report.write_text(json.dumps(data), encoding="utf-8")
        original_is_symlink = Path.is_symlink
        with patch.object(Path, "is_symlink", lambda path: path == linked or original_is_symlink(path)):
            with self.assertRaisesRegex(ValueError, "symlink"):
                analyzer.load_and_validate_report(self.report)

    def test_rejects_tampered_metadata_and_nonfinite_capture(self):
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["repeats_completed"][0]["capture"]["stages"]["normalized_model_depth"]["sha256"] = "0" * 64
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "(?i)stage hash"):
            analyzer.load_and_validate_report(self.report)

        array = np.full((2, 3), np.nan, dtype=np.float32)
        path = self.run / "captures" / "repeat-1.npz"
        np.savez_compressed(path, normalized_model_depth=array)
        data = json.loads(self.report.read_text(encoding="utf-8"))
        record = data["repeats_completed"][0]["capture"]
        record["sha256"], record["size_bytes"] = analyzer.sha256_file(path), path.stat().st_size
        record["stages"]["normalized_model_depth"].update(
            shape=list(array.shape), dtype=array.dtype.str, nbytes=array.nbytes, sha256=_array_hash(array))
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite"):
            analyzer.load_and_validate_report(self.report)

    def test_rejects_incomplete_repeats_and_archive_expansion_over_bound(self):
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["repeats_completed"].pop()
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "three"):
            analyzer.load_and_validate_report(self.report)

        # A small compressed file whose declared expansion exceeds the bound is rejected before np.load.
        data = json.loads(self.report.read_text(encoding="utf-8"))
        data["repeats_completed"] = json.loads(self._original_report_text())["repeats_completed"]
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with patch.object(analyzer, "MAX_CAPTURE_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "bound"):
                analyzer.load_and_validate_report(self.report)

    def test_merges_consistent_stage_and_input_metadata_from_save_stages(self):
        from scripts.isolate_gpu_variance import save_stages

        data = json.loads(self.report.read_text(encoding="utf-8"))
        input_shape = (2, 1, 3, 2, 3)
        tensors = []
        for index, repeat in enumerate(data["repeats_completed"], 1):
            input_array = np.full(input_shape, index, dtype=np.float32)
            capture = save_stages(
                self.run, index,
                {"normalized_model_depth": self.arrays[index - 1], "preprocessed_input_0": input_array},
                [{"stage": "preprocessed_input_0", "name": "images", "tensor_shape": [1, 3, 2, 3]}],
            )
            repeat["capture"] = capture
            tensors.append(input_array)
        self.report.write_text(json.dumps(data), encoding="utf-8")

        _report, loaded = analyzer.load_and_validate_report(self.report)
        self.assertIn("preprocessed_input_0", loaded[0]["arrays"])
        self.assertTrue(np.array_equal(loaded[0]["arrays"]["preprocessed_input_0"], tensors[0]))

        data["repeats_completed"][0]["capture"]["input_tensors"][0]["sha256"] = "0" * 64
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inconsistent duplicate metadata"):
            analyzer.load_and_validate_report(self.report)

    def test_cpu_fast_encoder_profile_is_valid_without_nvenc(self):
        report, _arrays = analyzer.load_and_validate_report(self.report)
        arguments = analyzer.validate_encoder_profile(report, {"libx264"})
        self.assertEqual(arguments["libx264"], analyzer.FAST_X264_ARGS)

    def test_server_guard_fails_closed_on_unreachable_or_unknown_jobs(self):
        with patch.object(analyzer, "urlopen", side_effect=OSError("connection reset")):
            with self.assertRaisesRegex(RuntimeError, "Could not verify local API 8768"):
                analyzer.server_snapshot()

        def fake_urlopen(url, timeout):
            if url.endswith("/api/status"):
                return io.BytesIO(b"{}")
            return io.BytesIO(b'[{"id":"other-job","state":"stopping"}]')

        with patch.object(analyzer, "urlopen", side_effect=fake_urlopen):
            with self.assertRaisesRegex(RuntimeError, "active or unknown jobs"):
                analyzer.server_snapshot()

    def test_rejects_tiny_npy_member_with_giant_declared_shape_before_array_load(self):
        data = json.loads(self.report.read_text(encoding="utf-8"))
        path = self.run / "captures" / "repeat-1.npz"
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            "descr": np.dtype("<f4").str, "fortran_order": False, "shape": (10**12,)})
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("normalized_model_depth.npy", header.getvalue())
        record = data["repeats_completed"][0]["capture"]
        record["sha256"], record["size_bytes"] = analyzer.sha256_file(path), path.stat().st_size
        self.report.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "NPY header declares"):
            analyzer.load_and_validate_report(self.report)

    def _original_report_text(self):
        return json.dumps({
            "status": "completed_diagnostic", "repeats_max": 3,
            "repeats_completed": [
                {"repeat": i, "capture": {"path": str(self.run / "captures" / f"repeat-{i}.npz"),
                 "sha256": analyzer.sha256_file(self.run / "captures" / f"repeat-{i}.npz"),
                 "size_bytes": (self.run / "captures" / f"repeat-{i}.npz").stat().st_size,
                 "stages": {"normalized_model_depth": {"shape": [2, 3], "dtype": "<f4", "nbytes": 24,
                   "sha256": _array_hash(self.arrays[i - 1])}}}} for i in (1, 2, 3)],
            "settings": {"fps": 12, "frames": 2},
            "encoder_experiment": {
                "quality_profile": "fast",
                "profile": {
                    "video_args": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"],
                    "software_fallback_args": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22"],
                },
                "encoders": {"libx264": {"runs": [{"command": ["ffmpeg", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p"]}]}},
            },
        })

    def test_saved_encoder_arguments_must_match_fast_allowlist(self):
        report, arrays = analyzer.load_and_validate_report(self.report)
        report["encoder_experiment"]["profile"]["video_args"][-1] = "-f"
        with self.assertRaisesRegex(ValueError, "production fast"):
            analyzer.validate_encoder_profile(report, {"libx264"})


if __name__ == "__main__":
    unittest.main()
