import hashlib
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import depth_cache_identity as cache_identity


def key(**overrides):
    values = {
        "source_digest": "a" * 64,
        "model_id": "depth-model",
        "model_sha256": "b" * 64,
        "input_size": 518,
        "execution_mode": "cpu",
        "working_size": (640, 360),
        "fps": 24.0,
        "invert_depth": False,
        "temporal_smoothing": 0.25,
        "depth_interval": 1,
        "decoder_identity": {"opencv": "4.10", "codec": {"name": "h264"}},
        "runtime_identity": {"onnxruntime": "1.20", "graph_version": 3},
        "pipeline_version": "native-depth-v1",
    }
    values.update(overrides)
    return cache_identity.build_cache_key(**values)


class BuildCacheKeyTests(unittest.TestCase):
    def test_identity_is_stable_and_dict_order_independent(self):
        self.assertEqual(
            key(decoder_identity={"opencv": "4.10", "codec": {"name": "h264"}}),
            key(decoder_identity={"codec": {"name": "h264"}, "opencv": "4.10"}),
        )
        self.assertRegex(key(), r"^[0-9a-f]{64}$")

    def test_every_native_depth_dependency_invalidates_identity(self):
        changes = (
            {"source_digest": "c" * 64},
            {"model_id": "other-model"},
            {"model_sha256": "d" * 64},
            {"input_size": 384},
            {"execution_mode": "directml"},
            {"working_size": (1280, 720)},
            {"fps": 30.0},
            {"invert_depth": True},
            {"temporal_smoothing": 0.5},
            {"depth_interval": 2},
            {"decoder_identity": {"opencv": "4.11", "codec": {"name": "h264"}}},
            {"runtime_identity": {"onnxruntime": "1.21", "graph_version": 3}},
            {"pipeline_version": "native-depth-v2"},
        )
        baseline = key()
        for change in changes:
            with self.subTest(change=change):
                self.assertNotEqual(baseline, key(**change))

    def test_json_dict_order_is_canonical_at_every_depth(self):
        self.assertEqual(
            key(runtime_identity={"a": 1, "nested": {"x": 2, "y": 3}}),
            key(runtime_identity={"nested": {"y": 3, "x": 2}, "a": 1}),
        )

    def test_rejects_malformed_hashes_and_wrong_types(self):
        bad = (
            {"source_digest": "A" * 64},
            {"source_digest": "0" * 63},
            {"model_sha256": "not-a-hash"},
            {"input_size": True},
            {"input_size": 0},
            {"working_size": (640, True)},
            {"fps": True},
            {"fps": float("nan")},
            {"fps": float("inf")},
            {"invert_depth": 1},
            {"depth_interval": False},
            {"decoder_identity": []},
            {"runtime_identity": {"version": float("nan")}},
            {"runtime_identity": {1: "bad key"}},
            {"model_id": ""},
            {"pipeline_version": None},
        )
        for change in bad:
            with self.subTest(change=change), self.assertRaises((TypeError, ValueError)):
                key(**change)

    def test_rejects_cyclic_nested_identity(self):
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(ValueError):
            key(runtime_identity=cyclic)

    def test_float_values_are_not_rounded_into_same_key(self):
        self.assertNotEqual(key(fps=24.0), key(fps=24.000000000000004))


class SourceHashTests(unittest.TestCase):
    def test_hashes_file_streaming_and_detects_mutation_during_hash(self):
        path = Path(self.id().replace(".", "_") + ".bin")
        try:
            data = b"a" * (1024 * 1024 + 17)
            path.write_bytes(data)
            expected = hashlib.sha256(data).hexdigest()
            self.assertEqual(cache_identity.hash_source(path), expected)

            real_open = open
            changed = False

            class MutatingReader:
                def __init__(self, wrapped):
                    self.wrapped = wrapped

                def __enter__(self):
                    self.wrapped.__enter__()
                    return self

                def __exit__(self, *args):
                    return self.wrapped.__exit__(*args)

                def fileno(self):
                    return self.wrapped.fileno()

                def read(self, size=-1):
                    nonlocal changed
                    block = self.wrapped.read(size)
                    if block and not changed:
                        changed = True
                        path.write_bytes(b"replacement" * 200000)
                    return block

            def open_and_mutate(*args, **kwargs):
                return MutatingReader(real_open(*args, **kwargs))

            with mock.patch("builtins.open", side_effect=open_and_mutate):
                with self.assertRaises((OSError, ValueError)):
                    cache_identity.hash_source(path)
        finally:
            path.unlink(missing_ok=True)

    def test_cancellation_raises_interrupted_error(self):
        path = Path(self.id().replace(".", "_") + ".bin")
        try:
            path.write_bytes(b"source")
            cancelled = threading.Event()
            cancelled.set()
            with self.assertRaises(InterruptedError):
                cache_identity.hash_source(path, cancelled)
        finally:
            path.unlink(missing_ok=True)


class FrameDigestTests(unittest.TestCase):
    def test_hash_changes_with_shape_and_pixel_values(self):
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        baseline = cache_identity.frame_digest(frame)
        changed_pixels = frame.copy()
        changed_pixels[0, 0, 0] = 1
        changed_shape = np.zeros((5, 4, 3), dtype=np.uint8)
        self.assertNotEqual(baseline, cache_identity.frame_digest(changed_pixels))
        self.assertNotEqual(baseline, cache_identity.frame_digest(changed_shape))
        self.assertEqual(len(baseline), 64)

    def test_rejects_invalid_frame_layout_and_type(self):
        invalid = (
            np.zeros((4, 5, 3), dtype=np.float32),
            np.zeros((4, 5), dtype=np.uint8),
            np.zeros((4, 5, 4), dtype=np.uint8),
            np.zeros((4, 5, 3), dtype=np.uint8)[:, ::2, :],
            [[0, 0, 0]],
        )
        for frame in invalid:
            with self.subTest(frame_type=type(frame).__name__), self.assertRaises((TypeError, ValueError)):
                cache_identity.frame_digest(frame)


if __name__ == "__main__":
    unittest.main()
