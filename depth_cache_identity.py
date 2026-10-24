"""Conservative identities for the native depth stage cache."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import struct
from typing import Any


_KEY_SCHEMA = "stereolab-native-depth-cache-v1"
_SOURCE_CHUNK_SIZE = 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _require_text(value: Any, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest")
    return value


def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _require_finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_json_value(value: Any, path: str, active: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool):
        return
    if value_type is int:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return
    if value_type not in (dict, list):
        raise TypeError(f"{path} contains a non-JSON value of type {value_type.__name__}")

    identity = id(value)
    if identity in active:
        raise ValueError(f"{path} contains a cyclic JSON value")
    active.add(identity)
    try:
        if value_type is dict:
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(f"{path} contains a non-string dictionary key")
                _validate_json_value(item, f"{path}.{key}", active)
        else:
            for index, item in enumerate(value):
                _validate_json_value(item, f"{path}[{index}]", active)
    finally:
        active.remove(identity)


def build_cache_key(
    source_digest: str,
    *,
    model_id: str,
    model_sha256: str,
    input_size: int,
    execution_mode: str,
    working_size: tuple[int, int],
    fps: float,
    invert_depth: bool,
    temporal_smoothing: float,
    depth_interval: int,
    decoder_identity: dict,
    runtime_identity: dict,
    pipeline_version: str,
) -> str:
    """Return a stable SHA-256 key for all inputs to native temporal depth."""
    source_digest = _require_sha256(source_digest, "source_digest")
    model_id = _require_text(model_id, "model_id")
    model_sha256 = _require_sha256(model_sha256, "model_sha256")
    input_size = _require_int(input_size, "input_size")
    execution_mode = _require_text(execution_mode, "execution_mode")
    if type(working_size) is not tuple or len(working_size) != 2:
        raise TypeError("working_size must be a pair of positive integers")
    width = _require_int(working_size[0], "working_size width")
    height = _require_int(working_size[1], "working_size height")
    fps = _require_finite_number(fps, "fps", minimum=0)
    if fps == 0:
        raise ValueError("fps must be greater than zero")
    if type(invert_depth) is not bool:
        raise TypeError("invert_depth must be a boolean")
    temporal_smoothing = _require_finite_number(
        temporal_smoothing, "temporal_smoothing", minimum=0
    )
    depth_interval = _require_int(depth_interval, "depth_interval", minimum=0)
    pipeline_version = _require_text(pipeline_version, "pipeline_version")

    for name, identity in (
        ("decoder_identity", decoder_identity),
        ("runtime_identity", runtime_identity),
    ):
        if type(identity) is not dict:
            raise TypeError(f"{name} must be a dictionary")
        _validate_json_value(identity, name, set())

    document = {
        "schema": _KEY_SCHEMA,
        "source_digest": source_digest,
        "model": {"id": model_id, "sha256": model_sha256},
        "inference": {
            "input_size": input_size,
            "execution_mode": execution_mode,
        },
        "working_size": [width, height],
        "preprocessing": {"invert_depth": invert_depth},
        "temporal": {
            "fps": fps,
            "smoothing": temporal_smoothing,
            "depth_interval": depth_interval,
        },
        "decoder_identity": decoder_identity,
        "runtime_identity": runtime_identity,
        "pipeline_version": pipeline_version,
    }
    try:
        canonical = json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("cache identity is not valid finite JSON") from error
    return hashlib.sha256(canonical).hexdigest()


def _stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_nlink,
        getattr(info, "st_file_attributes", 0),
    )


def _validate_source_info(info: os.stat_result, name: str) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"source must be an ordinary regular file: {name}")
    if info.st_nlink != 1:
        raise ValueError(f"source must not have hard links: {name}")
    if getattr(info, "st_file_attributes", 0) & _REPARSE_POINT:
        raise ValueError(f"source must not be a reparse point: {name}")


def hash_source(path: os.PathLike[str] | str, cancel_event: Any = None) -> str:
    """Hash an ordinary source file in bounded memory and detect concurrent changes."""
    try:
        source_path = os.fspath(path)
    except TypeError as error:
        raise TypeError("path must be a filesystem path") from error
    if type(source_path) is not str or not source_path:
        raise TypeError("path must be a non-empty string path")

    if cancel_event is not None and not callable(getattr(cancel_event, "is_set", None)):
        raise TypeError("cancel_event must provide is_set()")

    def check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("source hashing was cancelled")

    check_cancelled()
    before = os.lstat(source_path)
    _validate_source_info(before, source_path)
    digest = hashlib.sha256()

    with open(source_path, "rb") as source:
        opened = os.fstat(source.fileno())
        _validate_source_info(opened, source_path)
        if _stat_identity(opened) != _stat_identity(before):
            raise OSError("source changed while it was being opened")
        while True:
            check_cancelled()
            chunk = source.read(_SOURCE_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        check_cancelled()
        opened_after = os.fstat(source.fileno())

    after = os.lstat(source_path)
    _validate_source_info(after, source_path)
    identity = _stat_identity(before)
    if _stat_identity(opened_after) != identity or _stat_identity(after) != identity:
        raise OSError("source changed while it was being hashed")
    return digest.hexdigest()


def frame_digest(frame: Any) -> str:
    """Hash a contiguous uint8 BGR frame, including dimensions and exact pixels."""
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - project runtime requires NumPy
        raise RuntimeError("NumPy is required to hash video frames") from error

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a NumPy array")
    if frame.dtype != np.dtype(np.uint8):
        raise TypeError("frame must have dtype uint8")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must have shape (height, width, 3) in BGR order")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ValueError("frame dimensions must be positive")
    if not frame.flags.c_contiguous:
        raise ValueError("frame must be C-contiguous")

    digest = hashlib.sha256()
    digest.update(b"stereolab-bgr-frame-v1\0")
    digest.update(struct.pack(">QQQ", *frame.shape))
    digest.update(memoryview(frame).cast("B"))
    return digest.hexdigest()
