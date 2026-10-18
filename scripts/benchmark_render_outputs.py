"""Compare complete decode-to-PNG paths with/without unused eye synthesis.

This measures run_image (including approximation, refinement and PNG writing),
not API upload, queue wait, preflight or browser download. Sources are synthetic.
"""
from pathlib import Path
import hashlib
import json
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import main
from stereo import make_eye_pair, pack_output


def baseline_render(frame, depth, strength, convergence, layout):
    left, right = make_eye_pair(frame, depth, strength, convergence)
    return pack_output(frame, depth, left, right, layout)


def benchmark():
    folder = main.RUNTIME / 'render-benchmark'
    folder.mkdir(exist_ok=True)
    y, x = np.indices((1080, 1920))
    frame = np.stack(((x % 256).astype('uint8'), (y % 256).astype('uint8'),
                      ((x + y) % 256).astype('uint8')), axis=2)
    source = folder / 'source.png'
    cv2.imwrite(str(source), frame)
    optimized = main.render_output
    results = []
    try:
        for layout in ('DEPTH', 'RGBD'):
            samples = {'baseline': [], 'optimized': []}
            hashes = set()
            for repeat in range(5):
                # Reverse order every pair to reduce warmup/order bias.
                order = ('baseline', 'optimized') if repeat % 2 else ('optimized', 'baseline')
                for variant in order:
                    main.render_output = baseline_render if variant == 'baseline' else optimized
                    job = main.Job(id='benchmark-' + layout, filename='render-benchmark.png', input_path=source,
                                   media_type='image', layout=layout, strength=6, convergence=.5,
                                   depth_mode='balanced', input_size=256, resolution='source', quality='fast',
                                   invert_depth=False)
                    started = time.perf_counter()
                    output = main.run_image(job, None)
                    elapsed = time.perf_counter() - started
                    hashes.add(hashlib.sha256(output.read_bytes()).hexdigest())
                    if repeat:  # Discard one warmup from each variant.
                        samples[variant].append(elapsed)
            if len(hashes) != 1:
                raise AssertionError('Optimized output differs from baseline PNG bytes.')
            results.append({'layout': layout, 'exact_png_match': True, 'output_sha256': hashes.pop(),
                            'samples_seconds': samples,
                            'median_seconds': {key: statistics.median(values) for key, values in samples.items()},
                            'range_seconds': {key: [min(values), max(values)] for key, values in samples.items()}})
    finally:
        main.render_output = optimized
    report = {'schema_version': 1, 'scope': __doc__, 'source_dimensions': [1920, 1080],
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'environment': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__},
              'results': results}
    (folder / 'alternating.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    benchmark()
