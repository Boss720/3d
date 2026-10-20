"""Real CPU/DirectML verification. Uses installed Small weights; downloads nothing."""
from pathlib import Path
import json
import statistics
import subprocess
import sys
import threading
import time
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import onnxruntime as ort
from depth_engine import ModelManager

DEST = ROOT / '.runtime' / 'gpu-verification'
DEST.mkdir(parents=True, exist_ok=True)
OriginalOptions = ort.SessionOptions

class ProfiledManager(ModelManager):
    def _new_session(self, path, provider='CPUExecutionProvider', device_id=None, input_dimensions=None,
                     model_precision=None):
        # Add profiling to the production factory without changing its policy.
        def options():
            value = OriginalOptions()
            value.enable_profiling = True
            value.profile_file_prefix = str(DEST / f'{provider}-{uuid.uuid4().hex}')
            return value
        with patch.object(ort, 'SessionOptions', options):
            return super()._new_session(path, provider=provider, device_id=device_id,
                                        input_dimensions=input_dimensions, model_precision=model_precision)

m = ProfiledManager(ROOT/'.runtime/models')
frame = cv2.imread(str(ROOT/'.runtime/verification/source.png'))
if frame is None:
    frame = np.random.default_rng(42).integers(0,256,(240,400,3),np.uint8)
print({'runtime':m.runtime_info()},flush=True)
times = []
for _ in range(4):
    start = time.perf_counter()
    cpu = m.predict(frame,'depth-anything-v2-small',518,'cpu')
    times.append(time.perf_counter()-start)
cpu_profile = m._session.end_profiling()
cpu_median = statistics.median(times[1:])
print({'cpu_warm_seconds':cpu_median},flush=True)

samples = []
stopped = threading.Event()
def monitor():
    while not stopped.is_set():
        try:
            data = subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,
                                  creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            used, utilization = [int(x.strip()) for x in data.stdout.splitlines()[0].split(',')]
            samples.append({'memory_mib':used,'gpu_utilization_percent':utilization})
        except Exception as error:
            samples.append({'error':str(error)})
        stopped.wait(.5)
watch = threading.Thread(target=monitor,name='gpu-benchmark-monitor')
watch.start()
times = []
try:
    for index in range(101):
        start = time.perf_counter()
        gpu = m.predict(frame,'depth-anything-v2-small',518,'gpu')
        times.append(time.perf_counter()-start)
        assert m.execution_info()['active_provider']=='DmlExecutionProvider',m.execution_info()
        if index % 20 == 0:
            print({'gpu_frame':index,'seconds':round(times[-1],4)},flush=True)
finally:
    stopped.set()
    watch.join()
gpu_profile = m._session.end_profiling()
events = json.loads(Path(gpu_profile).read_text())
counts = {}
for event in events:
    provider = event.get('args',{}).get('provider')
    if provider:
        counts[provider] = counts.get(provider,0)+1
assert counts.get('DmlExecutionProvider',0)>0,counts
difference = float(np.mean(np.abs(gpu-cpu)))
assert difference < .02, difference
memory = [sample['memory_mib'] for sample in samples if 'memory_mib' in sample]
assert memory,'No GPU memory samples were collected'
report = {'execution':m.execution_info(),'cpu_warm_median_seconds':cpu_median,
          'gpu_warm_median_seconds':statistics.median(times[1:]),'gpu_frames':100,
          'cpu_gpu_mean_absolute_depth_difference':difference,'profile_provider_events':counts,
          'gpu_peak_device_memory_mib':max(memory),'gpu_memory_samples':samples,
          'cpu_profile':cpu_profile,'gpu_profile':gpu_profile}
(DEST/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({key:value for key,value in report.items() if key!='gpu_memory_samples'},indent=2),flush=True)
