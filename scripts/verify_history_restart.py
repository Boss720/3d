"""Isolated CPU/history acceptance; never connects to existing StereoLab servers.

Uses generated PNGs and an owned subprocess with private data directories. One
image execution is held until cancellation to prove shutdown with queued work.
All actual admission, processing, persistence, shutdown and download code is the
application's production implementation. Evidence is retained, never swept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serve(folder: Path, port: int, instance: str) -> None:
    import main
    import uvicorn

    # Keep static source files, isolate every writable app directory.
    main.DATA_ROOT = folder
    main.RUNTIME = folder / '.runtime'
    main.UPLOADS = main.RUNTIME / 'uploads'
    main.MODELS = main.RUNTIME / 'models'
    main.MODEL_PATH = main.MODELS / 'depth.onnx'
    main.OUTPUTS = folder / 'outputs'
    main.CLIP_PROXIES = main.RUNTIME / 'clip-proxies'
    for directory in (main.UPLOADS, main.MODELS, main.OUTPUTS, main.CLIP_PROXIES):
        directory.mkdir(parents=True, exist_ok=True)
    main.DEPTH_ENGINE = main.ModelManager(main.MODELS)
    main.CALIBRATIONS = main.CalibrationStore(main.RUNTIME / 'calibrations.json')
    original = main.run_image

    def controlled_image(job, net):
        if job.filename == 'hold.png':
            (folder / 'active-image.marker').write_text(job.id, encoding='utf-8')
            if not job.cancel_event.wait(45):
                raise RuntimeError('Acceptance hold exceeded its deadline.')
            raise main.JobCancelled()
        return original(job, net)

    main.run_image = controlled_image
    os.environ['STEREOLAB_HISTORY_INSTANCE'] = instance
    server = uvicorn.Server(uvicorn.Config(main.app, host='127.0.0.1', port=port, log_level='warning'))

    def watch_stop():
        while not server.should_exit:
            if (folder / f'stop-{port}.marker').exists():
                server.should_exit = True
                return
            time.sleep(.05)

    threading.Thread(target=watch_stop, daemon=True).start()
    server.run()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def verify(folder: Path, keep_server: bool = False) -> dict:
    import cv2
    import httpx
    import numpy as np

    folder.mkdir(parents=True, exist_ok=False)
    instance = 'acceptance'
    processes = []
    logs = []

    def start():
        port = free_port()
        log = (folder / f'server-{port}.log').open('w', encoding='utf-8')
        logs.append(log)
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve',
                                    '--folder', str(folder), '--port', str(port), '--instance', instance],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((process, port))
        return process, port, httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=15, trust_env=False)

    def ready(process, client):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(f'Owned server exited {process.returncode}; inspect retained logs.')
            try:
                response = client.get('/api/status')
                response.raise_for_status()
                return response.json()
            except httpx.TransportError:
                time.sleep(.1)
        raise AssertionError('Owned server did not become ready.')

    def stop(process, port):
        (folder / f'stop-{port}.marker').touch()
        process.wait(timeout=20)
        assert process.returncode == 0, process.returncode

    frame = np.zeros((72, 96, 3), dtype=np.uint8)
    frame[:, :, 1] = np.arange(96, dtype=np.uint8)
    cv2.rectangle(frame, (23, 18), (67, 57), (220, 100, 30), -1)
    payload = cv2.imencode('.png', frame)[1].tobytes()
    (folder / 'source.png').write_bytes(payload)
    settings = {'depth_mode': 'balanced', 'execution_mode': 'cpu', 'layout': 'SBS',
                'resolution': 'source', 'output_name_template': '{name}_{layout}'}

    def submit(client, filename):
        response = client.post('/api/jobs', data=settings, files={'file': (filename, payload, 'image/png')})
        response.raise_for_status()
        return response.json()

    def wait_job(client, job_id):
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            job = next(j for j in client.get('/api/jobs').raise_for_status().json() if j['id'] == job_id)
            if job['state'] in {'completed', 'failed', 'canceled'}:
                return job
            time.sleep(.05)
        raise AssertionError('Generated still did not finish.')

    report = {'schema_version': 1, 'mode': 'CPU approximation, generated PNGs',
              'interruption_control': 'hold.png run_image waits for production cancellation', 'checks': {}}
    retained = None
    try:
        first, port1, client = start()
        status = ready(first, client)
        assert status['history_capabilities']['state'] == 'ready', status
        assert client.get('/api/history').raise_for_status().json()['items'] == []
        completed = wait_job(client, submit(client, 'complete.png')['id'])
        assert completed['state'] == 'completed', completed
        output = client.get(f"/api/jobs/{completed['id']}/download").raise_for_status().content
        assert cv2.imdecode(np.frombuffer(output, np.uint8), cv2.IMREAD_COLOR).shape == (72, 192, 3)
        (folder / 'completed.png').write_bytes(output)
        held = submit(client, 'hold.png')
        deadline = time.monotonic() + 10
        while not (folder / 'active-image.marker').exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert (folder / 'active-image.marker').exists(), 'Controlled active job never started.'
        pending = submit(client, 'pending.png')
        assert client.get('/api/history').raise_for_status().json()['items'] == [], 'Current boot leaked into history.'
        stop(first, port1)
        client.close()

        second, port2, client = start()
        status = ready(second, client)
        assert status['history_capabilities']['state'] == 'ready', status
        assert client.get('/api/jobs').raise_for_status().json() == [], 'Old jobs were reconstructed.'
        records = client.get('/api/history').raise_for_status().json()['items']
        rows = {row['job_id']: row for row in records}
        assert rows[completed['id']]['state'] == 'completed', rows
        assert rows[pending['id']]['state'] == 'interrupted', rows
        assert rows[held['id']]['state'] in {'canceled', 'interrupted'}, rows
        for row in records:
            assert row['source_sha256'] == hashlib.sha256(payload).hexdigest(), row
            assert row['source_size_bytes'] == len(payload), row
            assert row['source_available'] is False and row['requires_source_reselection'] is True, row
        restored = client.get(f"/api/history/{completed['id']}/download").raise_for_status().content
        assert restored == output, 'Restart download bytes changed.'
        assert client.get('/api/history?limit=0').status_code == 400
        assert client.get('/api/history?before=invalid!').status_code == 400
        page1 = client.get('/api/history?limit=1').raise_for_status().json()
        page2 = client.get('/api/history', params={'limit': 1, 'before': page1['next_cursor']}).raise_for_status().json()
        assert page1['items'][0]['job_id'] != page2['items'][0]['job_id']

        contender, contender_port, contender_client = start()
        contender.wait(timeout=15)
        assert contender.returncode != 0, 'Second owner unexpectedly started.'
        contender_client.close()
        assert client.get('/api/history').raise_for_status().json()['items'] == records
        report['checks'].update(restart_history=True, no_requeue=True, source_sha256=True,
                                output_bytes_identical=True, paging_and_validation=True,
                                active_shutdown_and_pending_interruption=True, second_owner_rejected=True)
        report.update(completed_job_id=completed['id'], interrupted_job_id=pending['id'],
                      held_job_id=held['id'], records=records, output_sha256=hashlib.sha256(output).hexdigest())
        if keep_server:
            retained = second
            report.update(retained_server={'pid': second.pid, 'port': port2, 'url': f'http://127.0.0.1:{port2}',
                                          'stop_marker': str(folder / f'stop-{port2}.marker')})
        else:
            stop(second, port2)
        client.close()
        report['passed'] = True
        return report
    finally:
        for process, port in processes:
            if process.poll() is None and process is not retained:
                (folder / f'stop-{port}.marker').touch()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    # Only this script's exact Popen child, never an existing service.
                    process.terminate()
                    process.wait(timeout=10)
        for log in logs:
            log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serve', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--port', type=int)
    parser.add_argument('--instance', default='acceptance')
    parser.add_argument('--keep-server', action='store_true', help='Keep the restarted owned server for browser QA.')
    args = parser.parse_args()
    if args.serve:
        serve(args.folder.resolve(), args.port, args.instance)
    else:
        folder = (args.folder or ROOT / '.runtime' / 'history-restart-verification' / uuid.uuid4().hex).resolve()
        result = verify(folder, args.keep_server)
        (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps({'passed': result['passed'], 'report': str(folder / 'report.json'),
                          'checks': result['checks'], 'retained_server': result.get('retained_server')}, indent=2))
