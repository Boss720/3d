"""Pinned local CPU/DirectML depth models. No media is uploaded or approximated.

The ONNX exporters are Hugging Face's onnx-community and Xenova, rather than the
original model authors. Revisions, LFS hashes and preprocessing were checked via
the Hugging Face model API on 2026-10-18. Config provenance is exposed in catalog.
"""
from __future__ import annotations

import hashlib
import copy
import importlib
import json
import os
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from gpu_hardware import probe_hardware


CPU_PROVIDER = 'CPUExecutionProvider'
GPU_PROVIDER = 'DmlExecutionProvider'
EXECUTION_MODES = ('auto', 'cpu', 'gpu', 'hybrid')


MODEL_SPECS = {
    "depth-anything-v2-small": {
        "name": "Depth Anything V2 Small", "repo": "onnx-community/depth-anything-v2-small",
        "revision": "4472b7362082ad9968fee890ca0f1e5aca36b93d",
        "sha256": "afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c",
        "size": 99060839, "license": "Apache-2.0", "recommended": True,
        "upstream": "depth-anything/Depth-Anything-V2-Small",
        "description": "Recommended starting point for CPU or DirectML GPU. FP32 relative inverse depth; about 99 MB.",
    },
    "depth-anything-v2-base": {
        "name": "Depth Anything V2 Base", "repo": "onnx-community/depth-anything-v2-base",
        "revision": "dd4557d492cd7b563738ac8d9ccff9094620983c",
        "sha256": "dab40f67d5ead5e0ed52acfae41a717a708b02bc57f6cb761c5de828f26a7289",
        "size": 388917010, "license": "CC-BY-NC-4.0 (non-commercial)", "recommended": False,
        "upstream": "depth-anything/Depth-Anything-V2-Base",
        "description": "Higher memory use on CPU or DirectML GPU; automatic CPU fallback if GPU fails. Non-commercial weights.",
    },
    "depth-anything-v2-large": {
        "name": "Depth Anything V2 Large", "repo": "onnx-community/depth-anything-v2-large",
        "revision": "1fa1591c7b080e98da9655827c3a33a3972a4a83",
        "sha256": "a93aa89b5e92e30e0afbe0f7c3ec692b35cfca791ae9004a190fb0ca2010e905",
        "size": 1336922232, "license": "CC-BY-NC-4.0 (non-commercial)", "recommended": False,
        "upstream": "depth-anything/Depth-Anything-V2-Large",
        "description": "About 1.34 GB download; high memory use. CPU only on GPUs with 4 GB or less VRAM. Non-commercial weights.",
    },
    "dpt-hybrid-midas": {
        "name": "DPT Hybrid / MiDaS", "repo": "Xenova/dpt-hybrid-midas",
        "revision": "8af5a62e326ba3e842759aa27e13008c1c758db5",
        "sha256": "2eca68239006c64af94bdfa68464f34d7627c3dca7bb02932636e616c55a39ff",
        "size": 533061339, "license": "Apache-2.0 (Intel upstream model)", "recommended": False,
        "upstream": "Intel/dpt-hybrid-midas",
        "description": "MiDaS 3.0 DPT Hybrid; fixed 384 px input, CPU or DirectML GPU with automatic CPU fallback. Relative inverse depth.",
    },
}


class ModelManager:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._states: dict[str, dict[str, Any]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._session: Any = None
        self._session_id: str | None = None
        self._session_key: tuple[str, str, int | None] | None = None
        self._gpu_failures: dict[tuple[str, str, int | None], str] = {}
        self._gpu_verified = False
        self._last_execution: dict[str, Any] = {
            'requested_mode': None, 'active_provider': None, 'device': None,
            'fallback_reason': None, 'gpu_verified': False,
        }

    @staticmethod
    def _spec(model_id: str) -> dict[str, Any]:
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown depth model: {model_id}")
        return MODEL_SPECS[model_id]

    def _path(self, model_id: str) -> Path:
        self._spec(model_id)
        return self.models_dir / model_id / "model.onnx"

    def _receipt(self, model_id: str) -> Path:
        return self._path(model_id).with_suffix(".accepted.json")

    @staticmethod
    def _ort() -> Any:
        try:
            return importlib.import_module("onnxruntime")
        except (ImportError, OSError) as exc:
            raise RuntimeError("ONNX Runtime is unavailable. Install the requirements and restart the server. " + str(exc)) from exc

    def runtime_info(self) -> dict[str, Any]:
        hardware = probe_hardware()
        adapter = hardware['selected_adapter']
        last = self.execution_info()
        try:
            ort = self._ort()
            providers = ort.get_available_providers()
            available = CPU_PROVIDER in providers
            gpu_available = GPU_PROVIDER in providers and adapter is not None
            gpu_error = (None if gpu_available else hardware['error'] or
                         ('No physical DXGI graphics adapter was found.' if adapter is None else
                          'DirectML is unavailable in this ONNX Runtime. Use the GPU launcher/runtime.'))
            return {'available': available, 'providers': providers, 'active_provider': last['active_provider'],
                    'version': ort.__version__, 'device': last['device'],
                    'error': None if available else 'CPUExecutionProvider is unavailable.',
                    'gpu_available': gpu_available, 'gpu_verified': self._gpu_verified,
                    'gpu_name': adapter['name'] if adapter else None, 'gpu_device': copy.deepcopy(adapter),
                    'gpu_error': gpu_error, 'adapters': copy.deepcopy(hardware['adapters']),
                    'last_execution': last,
                    'gpu_memory_policy': 'Best effort: bounded input dimensions; DirectML has no configured hard VRAM limit. CPU nodes may remain.'}
        except Exception as exc:
            return {'available': False, 'providers': [], 'active_provider': last['active_provider'],
                    'device': last['device'], 'version': None, 'error': str(exc),
                    'gpu_available': False, 'gpu_verified': self._gpu_verified,
                    'gpu_name': adapter['name'] if adapter else None, 'gpu_device': copy.deepcopy(adapter),
                    'gpu_error': str(exc), 'adapters': copy.deepcopy(hardware['adapters']), 'last_execution': last}

    def execution_info(self) -> dict[str, Any]:
        """Snapshot of the most recent inference, including any CPU fallback reason."""
        with self._lock:
            return copy.deepcopy(self._last_execution)

    def _execution_plan(self, model_id: str, execution_mode: str, runtime: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f'Unknown execution mode: {execution_mode}. Choose auto, cpu, gpu or hybrid.')
        if execution_mode == 'cpu':
            return CPU_PROVIDER, None, None
        adapter = runtime.get('gpu_device')
        reason = None
        if not runtime.get('gpu_available') or adapter is None:
            reason = runtime.get('gpu_error') or 'GPU DirectML backend is unavailable; using CPU.'
        elif model_id == 'depth-anything-v2-large' and adapter.get('memory_mb', 0) <= 4096:
            reason = 'Depth Anything V2 Large GPU inference is disabled on adapters with 4 GB or less VRAM; use Small/Base or CPU.'
        if reason:
            if execution_mode == 'gpu':
                raise RuntimeError(reason)
            return CPU_PROVIDER, None, reason
        return GPU_PROVIDER, adapter, None

    def _accepted(self, model_id: str) -> bool:
        spec = self._spec(model_id)
        try:
            receipt = json.loads(self._receipt(model_id).read_text(encoding="utf-8"))
            return (isinstance(receipt, dict) and receipt.get("sha256") == spec["sha256"] and receipt.get("revision") == spec["revision"]
                    and self._path(model_id).stat().st_size == spec["size"])
        except (OSError, ValueError):
            return False

    def _public(self, model_id: str) -> dict[str, Any]:
        spec = self._spec(model_id)
        source = f"https://huggingface.co/{spec['repo']}"
        with self._lock:
            state = dict(self._states.get(model_id, {}))
        if not state or state.get("state") == "ready":
            accepted = self._accepted(model_id)
            state = {"state": "ready" if accepted else "not_installed", "progress": 100 if accepted else 0, "error": None}
        return {"id": model_id, "name": spec["name"], "description": spec["description"],
                "license": spec["license"], "source_url": source, "size_mb": round(spec["size"] / 1_000_000, 1),
                "recommended": spec["recommended"], "revision": spec["revision"], "sha256": spec["sha256"],
                "upstream_url": f"https://huggingface.co/{spec['upstream']}",
                "preprocessor_url": f"{source}/blob/{spec['revision']}/preprocessor_config.json", **state}

    def catalog(self) -> dict[str, Any]:
        return {"models": [self._public(model_id) for model_id in MODEL_SPECS], "runtime": self.runtime_info()}

    def available(self, model_id: str) -> bool:
        self._spec(model_id)
        return self._accepted(model_id) and self.runtime_info()["available"]

    def validate(self, model_id: str, execution_mode: str = 'auto') -> None:
        self._spec(model_id)
        if execution_mode not in EXECUTION_MODES:
            raise ValueError(f'Unknown execution mode: {execution_mode}. Choose auto, cpu, gpu or hybrid.')
        runtime = self.runtime_info()
        if not runtime["available"]:
            raise RuntimeError(runtime["error"])
        if not self._accepted(model_id):
            raise RuntimeError(f"{self._spec(model_id)['name']} is not installed and validated. Install it in the model panel.")
        self._execution_plan(model_id, execution_mode, runtime)

    def install(self, model_id: str) -> dict[str, Any]:
        self._spec(model_id)
        if not self.runtime_info()["available"]:
            raise RuntimeError(self.runtime_info()["error"])
        with self._lock:
            if self._accepted(model_id) or self._states.get(model_id, {}).get("state") == "downloading":
                return self._public(model_id)
            self._states[model_id] = {"state": "downloading", "progress": 0, "error": None}
            thread = threading.Thread(target=self._download, args=(model_id,), daemon=True, name=f"install-{model_id}")
            self._threads[model_id] = thread
            thread.start()
            return self._public(model_id)

    def _quarantine(self, path: Path) -> None:
        """Keep rejected bytes out of the accepted cache without deleting user files."""
        if path.exists():
            target = self.models_dir / "_quarantine" / uuid.uuid4().hex / path.relative_to(self.models_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, target)

    @staticmethod
    def _hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _download(self, model_id: str) -> None:
        spec = self._spec(model_id)
        destination = self._path(model_id)
        partial = destination.with_name(f"model.{uuid.uuid4().hex}.onnx.part")
        receipt_temp = destination.with_name(f"accepted.{uuid.uuid4().hex}.part")
        try:
            with self._lock:
                self._states[model_id] = {"state": "downloading", "progress": 0, "error": None}
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/onnx/model.onnx"
            request = urllib.request.Request(url, headers={"User-Agent": "LocalStereoConverter/2"})
            total = 0
            digest = hashlib.sha256()
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("xb") as output:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != spec["size"]:
                    raise RuntimeError("Model download has an unexpected Content-Length.")
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > spec["size"]:
                        raise RuntimeError("Model download exceeded its pinned size.")
                    output.write(chunk)
                    digest.update(chunk)
                    with self._lock:
                        self._states[model_id]["progress"] = min(95, round(total / spec["size"] * 95, 1))
                output.flush()
                os.fsync(output.fileno())
            if total != spec["size"] or digest.hexdigest() != spec["sha256"]:
                raise RuntimeError("Model download failed pinned size/SHA256 verification.")
            # A real inference must succeed before downloaded bytes become accepted.
            # Serialize validation and normal inference so only one model is resident.
            with self._inference_lock:
                self._session = None
                self._session_id = None
                self._session_key = None
                session = self._new_session(partial)
                self._predict_with_session(session, np.zeros((64, 64, 3), dtype=np.uint8), model_id, 518)
                del session
                receipt_temp.write_text(json.dumps({"sha256": spec["sha256"], "revision": spec["revision"]}), encoding="utf-8")
                if destination.exists():
                    self._quarantine(destination)
                os.replace(partial, destination)
                os.replace(receipt_temp, self._receipt(model_id))
            with self._lock:
                self._states[model_id] = {"state": "ready", "progress": 100, "error": None}
        except Exception as exc:
            for path in (partial, receipt_temp):
                try:
                    self._quarantine(path)
                except OSError:
                    pass
            with self._lock:
                self._states[model_id] = {"state": "failed", "progress": 0, "error": str(exc)}

    def _new_session(self, path: Path, provider: str = CPU_PROVIDER, device_id: int | None = None) -> Any:
        ort = self._ort()
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == GPU_PROVIDER:
            options.enable_mem_pattern = False
            options.enable_cpu_mem_arena = False
            providers = [(GPU_PROVIDER, {'device_id': int(device_id)}), CPU_PROVIDER]
        else:
            providers = [CPU_PROVIDER]
        session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
        # ORT can retry a failed run with a different provider unless disabled.
        if hasattr(session, 'disable_fallback'):
            session.disable_fallback()
        if provider not in session.get_providers():
            raise RuntimeError(f'{provider} was requested but the created session fell back to {session.get_providers()}.')
        return session

    @staticmethod
    def _input_size(session: Any, height: int, width: int, model_id: str, input_size: int, max_dimension: int | None = None) -> tuple[int, int]:
        inputs = session.get_inputs()
        if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].type != "tensor(float)":
            raise RuntimeError("Depth model must accept one float32 NCHW image tensor.")
        shape = inputs[0].shape
        if isinstance(shape[0], int) and shape[0] != 1 or isinstance(shape[1], int) and shape[1] != 3:
            raise RuntimeError("Depth model must accept a batch of one RGB image.")
        if model_id == "dpt-hybrid-midas":
            out_h = out_w = 384
        else:
            if not 128 <= input_size <= 1024:
                raise ValueError("Depth input size must be between 128 and 1024 pixels.")
            scale_h, scale_w = input_size / height, input_size / width
            # HF DPTImageProcessor keep_aspect_ratio: choose least change in scale.
            scale = scale_w if abs(1 - scale_w) < abs(1 - scale_h) else scale_h
            out_h, out_w = max(14, round(height * scale / 14) * 14), max(14, round(width * scale / 14) * 14)
            if max_dimension is not None and max(out_h, out_w) > max_dimension:
                scale = max_dimension / max(out_h, out_w)
                out_h, out_w = (max(14, int(size * scale / 14) * 14) for size in (out_h, out_w))
        out_h = shape[2] if isinstance(shape[2], int) else out_h
        out_w = shape[3] if isinstance(shape[3], int) else out_w
        if max_dimension is not None and max(out_h, out_w) > max_dimension:
            raise RuntimeError('Fixed model input dimensions exceed the GPU memory policy.')
        if not 1 <= out_h <= 4096 or not 1 <= out_w <= 4096:
            raise RuntimeError("Depth model input dimensions exceed the supported size.")
        return out_h, out_w

    def _predict_with_session(self, session: Any, frame: np.ndarray, model_id: str, input_size: int, max_dimension: int | None = None) -> np.ndarray:
        height, width = frame.shape[:2]
        out_h, out_w = self._input_size(session, height, width, model_id, input_size, max_dimension)
        midas = model_id == "dpt-hybrid-midas"
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR if midas else cv2.INTER_CUBIC)
        values = rgb.astype(np.float32) / 255.0
        mean = np.array([0.5] * 3 if midas else [0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.5] * 3 if midas else [0.229, 0.224, 0.225], dtype=np.float32)
        blob = np.ascontiguousarray(np.transpose((values - mean) / std, (2, 0, 1))[None])
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        if not outputs:
            raise RuntimeError("Depth model returned no output.")
        prediction = np.asarray(outputs[0], dtype=np.float32)
        # Remove batch/channel axes only, preserving singleton spatial dimensions.
        if prediction.ndim == 4 and prediction.shape[:2] == (1, 1):
            prediction = prediction[0, 0]
        elif prediction.ndim == 3 and prediction.shape[0] == 1:
            prediction = prediction[0]
        if prediction.ndim != 2 or not prediction.size or not np.isfinite(prediction).all():
            raise RuntimeError("Depth model must return one finite 2D depth map.")
        depth = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_CUBIC)
        minimum, maximum = float(depth.min()), float(depth.max())
        # Both catalog families return relative inverse depth: high values are near.
        depth = (depth - minimum) / max(maximum - minimum, 1e-6)
        return np.ascontiguousarray(np.clip(depth, 0, 1), dtype=np.float32)

    def predict(self, frame_bgr: np.ndarray, model_id: str, input_size: int = 518, execution_mode: str = 'auto') -> np.ndarray:
        self.validate(model_id, execution_mode)
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or not frame_bgr.shape[0] or not frame_bgr.shape[1] or frame_bgr.dtype != np.uint8:
            raise ValueError("Depth inference needs a nonempty uint8 BGR image.")
        with self._inference_lock:
            provider, adapter, reason = self._execution_plan(model_id, execution_mode, self.runtime_info())
            gpu_key = (model_id, GPU_PROVIDER, adapter['id'] if adapter else None)
            if provider == GPU_PROVIDER and execution_mode != 'gpu' and gpu_key in self._gpu_failures:
                provider, adapter, reason = CPU_PROVIDER, None, self._gpu_failures[gpu_key]
            with self._lock:
                self._last_execution = {'requested_mode': execution_mode, 'model_id': model_id,
                                        'active_provider': None, 'device': None,
                                        'fallback_reason': reason, 'gpu_verified': False}
            try:
                try:
                    depth = self._run(frame_bgr, model_id, input_size, provider, adapter)
                except Exception as exc:
                    if provider != GPU_PROVIDER:
                        raise
                    reason = f'GPU DirectML inference failed: {exc}'
                    self._gpu_failures[gpu_key] = reason
                    self._session = None
                    self._session_key = None
                    self._session_id = None
                    with self._lock:
                        self._last_execution['fallback_reason'] = reason
                    if execution_mode == 'gpu':
                        raise RuntimeError(reason) from exc
                    provider, adapter = CPU_PROVIDER, None
                    depth = self._run(frame_bgr, model_id, input_size, provider, adapter)
                if provider == GPU_PROVIDER:
                    self._gpu_verified = True
                    self._gpu_failures.pop(gpu_key, None)
                with self._lock:
                    self._last_execution.update({
                        'active_provider': provider, 'device': adapter['name'] if adapter else 'CPU',
                        'device_id': adapter['id'] if adapter else None, 'fallback_reason': reason,
                        'gpu_verified': provider == GPU_PROVIDER,
                        'cpu_nodes_possible': provider == GPU_PROVIDER,
                        'input_size_limit': 924 if provider == GPU_PROVIDER else None,
                    })
                return depth
            except Exception as exc:
                with self._lock:
                    self._last_execution['error'] = str(exc)
                raise RuntimeError(f"{self._spec(model_id)['name']} inference failed: {exc}") from exc

    def predict_with_info(self, frame_bgr: np.ndarray, model_id: str, input_size: int = 518,
                          execution_mode: str = 'auto') -> tuple[np.ndarray, dict[str, Any]]:
        """Return depth and its execution snapshot atomically across concurrent requests."""
        with self._inference_lock:
            depth = self.predict(frame_bgr, model_id, input_size, execution_mode)
            return depth, self.execution_info()

    def _run(self, frame: np.ndarray, model_id: str, input_size: int, provider: str, adapter: dict[str, Any] | None) -> np.ndarray:
        key = (model_id, provider, adapter['id'] if adapter else None)
        if self._session_key != key:
            self._session = None
            self._session_id = None
            self._session_key = None
            if self._hash(self._path(model_id)) != self._spec(model_id)["sha256"]:
                self._quarantine(self._receipt(model_id))
                with self._lock:
                    self._states[model_id] = {"state": "failed", "progress": 0, "error": "Installed model failed SHA256 verification. Reinstall the model."}
                raise RuntimeError("Installed model failed SHA256 verification. Reinstall the model.")
            self._session = (self._new_session(self._path(model_id), provider=provider, device_id=adapter['id'])
                             if adapter else self._new_session(self._path(model_id)))
            if hasattr(self._session, 'disable_fallback'):
                self._session.disable_fallback()
            if provider == GPU_PROVIDER and provider not in self._session.get_providers():
                self._session = None
                raise RuntimeError(f'{provider} is absent from the created session; ONNX Runtime silently fell back to CPU.')
            self._session_id = model_id
            self._session_key = key
        result = (self._predict_with_session(self._session, frame, model_id, input_size, 924)
                  if adapter else self._predict_with_session(self._session, frame, model_id, input_size))
        if provider == GPU_PROVIDER and provider not in self._session.get_providers():
            raise RuntimeError(f'{provider} is no longer active after inference.')
        out_h, out_w = self._input_size(self._session, *frame.shape[:2], model_id, input_size, 924 if adapter else None)
        with self._lock:
            self._last_execution.update({'input_width': out_w, 'input_height': out_h})
        return result
