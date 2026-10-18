"""Verify actual CPU/GPU/hybrid exports against the running GPU-enabled app."""
import json
from pathlib import Path
import sys
import time

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'.runtime/gpu-verification'
DEST.mkdir(parents=True, exist_ok=True)
client = httpx.Client(base_url=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:8766',timeout=90)
source = (ROOT/'.runtime/verification/source.mp4').read_bytes()
reports = []
decoded = {}
for mode in ('cpu','gpu','hybrid'):
    start = time.perf_counter()
    job = client.post('/api/jobs',data={'depth_mode':'depth-anything-v2-small','execution_mode':mode,
        'layout':'HALFSBS','strength':'6','quality':'fast','resolution':'source'},
        files={'file':('device-check.mp4',source)}).raise_for_status().json()
    while time.perf_counter()-start < 120:
        job = next(j for j in client.get('/api/jobs').raise_for_status().json() if j['id']==job['id'])
        if job['state'] in ('completed','failed','canceled'): break
        time.sleep(.1)
    assert job['state']=='completed',job
    expected = 'CPUExecutionProvider' if mode=='cpu' else 'DmlExecutionProvider'
    assert job['execution']['active_provider']==expected,job
    assert job['pipeline']==('overlapped' if mode=='hybrid' else 'sequential'),job
    path = DEST/f'{mode}-video.mp4'
    path.write_bytes(client.get(f"/api/jobs/{job['id']}/download").raise_for_status().content)
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok,frame = capture.read()
        if not ok: break
        frames.append(frame)
    capture.release()
    assert len(frames)==8 and frames[0].shape==(94,160,3)
    decoded[mode] = np.stack(frames)
    report = {'mode':mode,'seconds':round(time.perf_counter()-start,3),'execution':job['execution'],
              'pipeline':job['pipeline'],'frames':len(frames)}
    reports.append(report)
    print(json.dumps(report),flush=True)
# Identical GPU inference with/without stage overlap must produce ordered identical frames.
np.testing.assert_array_equal(decoded['gpu'],decoded['hybrid'])
cpu_gpu_difference = float(np.abs(decoded['cpu'].astype(float)-decoded['gpu']).mean())
assert cpu_gpu_difference<1,cpu_gpu_difference
reports.append({'gpu_hybrid_video_pixels_identical':True,'cpu_gpu_video_mean_absolute_difference':cpu_gpu_difference})
(DEST/'devices.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
print(json.dumps(reports[-1]),flush=True)
