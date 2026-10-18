"""Exercise a running local server with real image/video/model inference.

Run after installing Small: .venv/Scripts/python.exe scripts/smoke.py
Keeps synthetic fixtures and reports in .runtime/verification; never downloads models.
"""
from pathlib import Path
import base64
import json
import subprocess
import time
import sys

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / '.runtime' / 'verification'
DEST.mkdir(parents=True, exist_ok=True)
client = httpx.Client(base_url=sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8766', timeout=90)
status = client.get('/api/status').raise_for_status().json()
assert status['runtime']['available'], status
catalog = client.get('/api/models').raise_for_status().json()
assert next(m for m in catalog['models'] if m['id'] == 'depth-anything-v2-small')['state'] == 'ready'

# A known synthetic composition, free of personal media.
frame = np.zeros((240, 400, 3), np.uint8)
for row in range(240):
    frame[row] = (int(160-row*.25), int(75+row*.35), int(35+row*.22))
cv2.circle(frame, (310, 58), 24, (130, 220, 255), -1, cv2.LINE_AA)
cv2.fillPoly(frame, [np.array([(0,185),(110,55),(250,240),(0,240)])], (100,65,80))
cv2.fillPoly(frame, [np.array([(100,240),(280,80),(400,200),(400,240)])], (80,55,58))
cv2.rectangle(frame, (175,146), (219,227), (42,119,230), -1)
cv2.circle(frame, (197,129), 26, (80,170,245), -1, cv2.LINE_AA)
cv2.imwrite(str(DEST/'source.png'), frame)
png = (DEST/'source.png').read_bytes()
settings = {'depth_mode':'depth-anything-v2-small', 'input_size':'518', 'strength':'6', 'convergence':'.5',
            'resolution':'source', 'quality':'fast', 'temporal_smoothing':'.35', 'edge_refinement':'true'}
results = []

def submit(name, payload, layout):
    response = client.post('/api/jobs', data={**settings, 'layout':layout}, files={'file':(name,payload)})
    response.raise_for_status()
    job = response.json()
    start = time.monotonic()
    while time.monotonic()-start < 120:
        job = next(j for j in client.get('/api/jobs').raise_for_status().json() if j['id'] == job['id'])
        if job['state'] in {'completed','failed','canceled'}: break
        time.sleep(.2)
    assert job['state'] == 'completed', job
    output = client.get(f"/api/jobs/{job['id']}/download").raise_for_status().content
    path = DEST/job['output_name']
    path.write_bytes(output)
    print({'layout':layout, 'file':path.name, 'state':job['state'], 'seconds':round(time.monotonic()-start,2)},flush=True)
    return path

preview = client.post('/api/preview', data={**settings,'layout':'SBS'}, files={'file':('source.png',png)}).raise_for_status().json()
shown_bytes = base64.b64decode(preview['image'].split(',',1)[1])
depth_bytes = base64.b64decode(preview['depth'].split(',',1)[1])
(DEST/'preview.png').write_bytes(shown_bytes)
(DEST/'depth.png').write_bytes(depth_bytes)
image = submit('source.png', png, 'SBS')
shown = cv2.imdecode(np.frombuffer(shown_bytes,np.uint8),cv2.IMREAD_COLOR)
exported = cv2.imread(str(image))
difference = cv2.absdiff(exported, shown)
comparison = {'max_channel_difference': int(difference.max()),
              'mean_channel_difference': float(difference.mean()),
              'different_channels': int(np.count_nonzero(difference))}
if preview.get('execution', {}).get('active_provider') == 'DmlExecutionProvider':
    # Separate DirectML inference calls can straddle a remap rounding boundary.
    # Preserve the measured error; this permits only sparse one-level variation.
    assert comparison['max_channel_difference'] <= 1 and comparison['mean_channel_difference'] <= .001, comparison
else:
    np.testing.assert_array_equal(exported, shown)
assert shown.shape == (240,800,3)
assert np.std(cv2.imread(str(DEST/'depth.png'))) > 5
assert np.any(shown[:,:400] != shown[:,400:])
results.append({'neural_preview_export_exact_match': not bool(difference.any()),
                'preview_export_comparison': comparison, 'execution': preview.get('execution'),
                'shape':list(shown.shape)})
for layout, shape in [('DEPTH',(240,400,3)),('RGBD',(240,800,3))]:
    path = submit('source.png',png,layout)
    assert cv2.imread(str(path)).shape == shape

ffmpeg = status['ffmpeg_path']
assert ffmpeg, 'FFmpeg required for this smoke'
video = DEST/'source.mp4'
subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-f','lavfi','-i','testsrc2=size=162x94:rate=8',
                '-f','lavfi','-i','sine=frequency=440:sample_rate=44100','-t','1','-c:v','libx264',
                '-pix_fmt','yuv420p','-c:a','aac',str(video)],check=True,timeout=30)
converted = submit('source.mp4',video.read_bytes(),'HALFSBS')
probe = str(Path(ffmpeg).with_name('ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
raw = subprocess.run([probe,'-v','error','-show_entries','stream=codec_type,width,height,nb_frames,r_frame_rate:format=duration',
                      '-of','json',str(converted)],capture_output=True,text=True,check=True,timeout=15)
metadata = json.loads(raw.stdout)
v = next(s for s in metadata['streams'] if s['codec_type'] == 'video')
assert (v['width'],v['height']) == (160,94), metadata
assert int(v['nb_frames']) == 8 and v['r_frame_rate'] == '8/1', metadata
assert any(s['codec_type']=='audio' for s in metadata['streams']), metadata
assert abs(float(metadata['format']['duration'])-1) < .15, metadata
results.append({'neural_video_audio_dimensions_frames_fps':metadata})

# Real cancellation of a queued neural conversion must never finish as failed.
queued = client.post('/api/jobs',data={**settings,'layout':'SBS'},files={'file':('cancel.mp4',video.read_bytes())}).raise_for_status().json()
client.post(f"/api/jobs/{queued['id']}/cancel").raise_for_status()
for _ in range(100):
    canceled = next(j for j in client.get('/api/jobs').json() if j['id']==queued['id'])
    if canceled['state'] in {'completed','failed','canceled'}: break
    time.sleep(.1)
assert canceled['state']=='canceled', canceled
results.append({'cancellation':canceled['state']})
(DEST/'report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2),flush=True)
