"""Profile actual DirectML variants; no synthetic throughput claims or downloads.

Example: .venv-gpu\\Scripts\\python scripts\\benchmark_directml_variants.py
Optional --fp16-path points to the pinned downloaded model_fp16.onnx.
"""
from pathlib import Path
import argparse
import gc
import json
import statistics
import sys
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import onnxruntime as ort
from depth_engine import ModelManager, MODEL_SPECS, GPU_PROVIDER
from onnx_optimization import prepare_directml_model, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, default=ROOT / '.runtime' / 'models')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.runtime' / 'directml-variants' / uuid.uuid4().hex)
    parser.add_argument('--fp16-path', type=Path)
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manager = ModelManager(args.models_dir)
    runtime = manager.runtime_info()
    if not runtime.get('gpu_available'):
        raise RuntimeError(runtime)
    adapter = runtime['gpu_device']
    frame = np.random.default_rng(42).integers(0, 256, (240, 400, 3), np.uint8)
    sources = [('depth-anything-v2-small', manager._path('depth-anything-v2-small'))]
    if args.fp16_path:
        sources.append(('depth-anything-v2-small-fp16', args.fp16_path))
    original_options = ort.SessionOptions
    results = []
    baseline = None
    for model_id, source in sources:
        if sha256(source) != MODEL_SPECS[model_id]['sha256']:
            raise RuntimeError(f'Unverified source: {source}')
        derived, info = prepare_directml_model(source, MODEL_SPECS[model_id]['sha256'], args.output_dir / 'graphs')
        from types import SimpleNamespace
        descriptor = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(**info['input'])])
        h, w = manager._input_size(descriptor, *frame.shape[:2], model_id, args.input_size, 924)
        dimensions = {name: size for name, size in zip(info['input']['shape'], (1, 3, h, w)) if isinstance(name, str)}
        for variant, path, overrides in [('dynamic-original', source, None), ('static-original', source, dimensions), ('static-linear', derived, dimensions)]:
            def options():
                result = original_options()
                result.enable_profiling = True
                result.profile_file_prefix = str(args.output_dir / f'{model_id}-{variant}')
                return result
            with patch.object(ort, 'SessionOptions', options):
                start = time.perf_counter()
                session = manager._new_session(path, GPU_PROVIDER, adapter['id'], input_dimensions=overrides)
                initialization = time.perf_counter() - start
            timings = []
            for index in range(args.repeats + 1):
                start = time.perf_counter()
                depth = manager._predict_with_session(session, frame, model_id, args.input_size, 924, native=True)
                timings.append(time.perf_counter() - start)
            profile = Path(session.end_profiling())
            counts, cpu_ops = {}, {}
            for event in json.loads(profile.read_text()):
                details = event.get('args', {})
                provider = details.get('provider')
                if provider:
                    counts[provider] = counts.get(provider, 0) + 1
                    if provider == 'CPUExecutionProvider':
                        op = details.get('op_name', '?')
                        cpu_ops[op] = cpu_ops.get(op, 0) + 1
            if not counts.get(GPU_PROVIDER):
                raise RuntimeError('Profile did not observe DirectML execution')
            if baseline is None:
                baseline = depth.copy()
            result = {'model_id': model_id, 'variant': variant, 'shape': [h, w],
                      'initialization_seconds': initialization, 'warm_seconds': timings[1:],
                      'warm_median_seconds': statistics.median(timings[1:]), 'provider_event_counts': counts,
                      'cpu_op_counts': cpu_ops, 'mean_absolute_depth_difference_from_fp32_original': float(np.mean(np.abs(depth - baseline))),
                      'profile': str(profile), 'graph': info if variant == 'static-linear' else None}
            results.append(result)
            print(json.dumps(result), flush=True)
            del session
            gc.collect()
    report = {'runtime': runtime, 'results': results, 'notes': 'Synthetic input; measured inference pipeline includes preprocessing/normalization. Profile events are observations, not a guarantee of zero CPU work.'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(str(args.output_dir / 'report.json'), flush=True)


if __name__ == '__main__':
    main()
