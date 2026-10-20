"""Matched synthetic CPU renderer benchmark; no inference or encoding."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from stereo import StereoRenderer, make_eye_pair, pack_output, render_output
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
from test_render_buffers import legacy_xreal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=24)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rng = np.random.default_rng(123)
    results = []
    for h, w in ((720,1280), (1080,1920)):
        frame = rng.integers(0,256,(h,w,3),dtype=np.uint8)
        depth = rng.random((h,w),dtype=np.float32)
        for layout in ('SBS','XREAL_SBS'):
            context = StereoRenderer()
            def old():
                left,right = make_eye_pair(frame,depth,5,.5)
                return legacy_xreal(left,right) if layout == 'XREAL_SBS' else pack_output(frame,depth,left,right,layout)
            def new():
                return context.render(frame,depth,5,.5,layout)
            np.testing.assert_array_equal(old(),new())
            samples = {'old':[], 'new':[]}
            for _ in range(3): old(); new()
            for i in range(args.iterations):
                for name,fn in ((('old',old),('new',new)) if i%2 == 0 else (('new',new),('old',old))):
                    start=time.perf_counter(); result=fn(); elapsed=time.perf_counter()-start
                    samples[name].append(elapsed*1000)
            old_ms,new_ms = (statistics.median(samples[name]) for name in ('old','new'))
            results.append(dict(size=[w,h],layout=layout,old_median_ms=old_ms,new_median_ms=new_ms,speedup=old_ms/new_ms,iterations=args.iterations,pixel_equal=True))
    print(json.dumps(dict(opencv=cv2.__version__,threads=cv2.getNumThreads(),results=results),indent=2))

if __name__ == '__main__': main()
