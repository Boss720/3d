"""Exact code identity shared by source execution and frozen builds."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


SOURCE_COMPONENTS = (
    'main.py', 'depth_engine.py', 'stereo.py', 'video_encoder.py',
    'video_pipeline.py', 'preview_cache.py', 'telemetry.py', 'preflight.py',
    'calibration.py', 'resource_sampling.py', 'preview_policy.py', 'preview_clip.py',
    'job_queue.py', 'job_settings.py', 'output_naming.py', 'job_history.py',
    'local_workspace.py', 'video_depth.py', 'video_decoder.py', 'onnx_optimization.py',
    'depth_cache.py', 'depth_cache_identity.py', 'build_identity.py', 'app_launcher.py',
    'media_process.py', 'upload_admission.py', 'gpu_hardware.py',
)
MANIFEST_NAME = 'stereolab-build.json'


def _fingerprint(components: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(components, sort_keys=True,
                                     separators=(',', ':')).encode('ascii')).hexdigest()


def build_manifest(root: Path) -> dict:
    components = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                  for name in SOURCE_COMPONENTS}
    return {'schema_version': 1, 'components': components,
            'fingerprint': _fingerprint(components)}


def build_fingerprint(root: Path, *, frozen: bool = False) -> str | None:
    """Return None for missing/incomplete identity; never reuse unknown builds."""
    try:
        if not frozen:
            return build_manifest(root)['fingerprint']
        path = root / MANIFEST_NAME
        if path.stat().st_size > 16384:
            return None
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (type(manifest) is not dict or manifest.get('schema_version') != 1
                or type(manifest.get('components')) is not dict):
            return None
        components = manifest['components']
        if set(components) != set(SOURCE_COMPONENTS) or any(
                type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None
                for value in components.values()):
            return None
        actual = _fingerprint(components)
        return actual if manifest.get('fingerprint') == actual else None
    except (OSError, ValueError, TypeError):
        return None
