"""Verify preview reuse and opt-in calibration against an idle local API."""
from pathlib import Path
import argparse
import base64
import json
import sys

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarking import validate_local_api_url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:8769')
    parser.add_argument('--device', choices=('cpu', 'gpu', 'hybrid', 'auto'), default='cpu')
    args = parser.parse_args()
    base = validate_local_api_url(args.base_url)
    folder = ROOT / '.runtime' / 'roadmap-verification'
    folder.mkdir(exist_ok=True)
    frame = np.zeros((180, 320, 3), np.uint8)
    frame[:, :, 0] = np.arange(320, dtype=np.uint16)[None, :] % 256
    cv2.rectangle(frame, (100, 35), (170, 150), (30, 150, 240), -1)
    payload = cv2.imencode('.png', frame)[1].tobytes()
    settings = {'depth_mode': 'depth-anything-v2-small', 'execution_mode': args.device,
                'input_size': '256', 'strength': '2', 'layout': 'SBS'}
    with httpx.Client(base_url=base, timeout=120) as client:
        jobs = client.get('/api/jobs').raise_for_status().json()
        if any(job['state'] in ('queued', 'processing') for job in jobs):
            raise RuntimeError('Wait for local jobs before running isolated verification.')
        client.post('/api/runtime/gpu/reset').raise_for_status()
        def preview(**changes):
            return client.post('/api/preview', data={**settings, **changes},
                               files={'file': ('synthetic.png', payload)}).raise_for_status().json()
        first = preview()
        changed = preview(strength='8', convergence='.3', layout='TB')
        assert not first['cache']['hit'] and changed['cache']['hit']
        different = preview(input_size='384')
        assert not different['cache']['hit']
        client.post('/api/runtime/gpu/reset').raise_for_status()
        fresh = preview(strength='8', convergence='.3', layout='TB')
        def pixels(result):
            raw = base64.b64decode(result['image'].split(',', 1)[1])
            return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        difference = cv2.absdiff(pixels(changed), pixels(fresh))
        if args.device == 'cpu':
            assert not difference.any()
        else:
            assert difference.max() <= 1 and difference.mean() <= .001
        calibration = client.post('/api/calibration', data={'depth_mode': settings['depth_mode'],
                                  'execution_mode': args.device, 'budget_seconds': '15'},
                                  files={'file': ('synthetic.png', payload)}).raise_for_status().json()
        assert calibration['schema_version'] == 1
        assert calibration['sufficient_evidence'], calibration
        assert all(len(candidate['samples_seconds']) == 2 for candidate in calibration['candidates'])
        report = {'base_url': base, 'device': args.device, 'cache_hit_after_presentation_edit': True,
                  'detail_change_invalidated': True, 'reset_invalidated': True,
                  'cached_vs_fresh_max_difference': int(difference.max()),
                  'cached_vs_fresh_mean_difference': float(difference.mean()),
                  'execution': fresh['execution'], 'calibration': calibration}
        path = folder / f'{args.device}.json'
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'report': str(path), 'provider': fresh['execution']['active_provider'],
                          'fastest_measured_input': calibration['fastest_measured_input_size'],
                          'calibration_seconds': calibration['total_elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
