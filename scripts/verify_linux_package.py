"""Launch a frozen Linux package read-only in isolated XDG directories and verify shutdown."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.error import URLError
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def free_port() -> int:
    with socket.socket() as server:
        server.bind(('127.0.0.1', 0))
        return int(server.getsockname()[1])


def fetch(url: str) -> tuple[int, bytes]:
    return request(url, timeout=2)


def request(url: str, *, method: str = 'GET', body: bytes | None = None,
            headers: dict[str, str] | None = None, timeout: float = 15) -> tuple[int, bytes]:
    req = Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()


def multipart(fields: dict[str, str], filename: str, content_type: str,
              payload: bytes) -> tuple[bytes, str]:
    boundary = f'----StereoLab-{uuid.uuid4().hex}'
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((f'--{boundary}\r\n'.encode(),
                       f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                       value.encode('utf-8'), b'\r\n'))
    chunks.extend((f'--{boundary}\r\n'.encode(),
                   f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
                   f'Content-Type: {content_type}\r\n\r\n'.encode(), payload, b'\r\n',
                   f'--{boundary}--\r\n'.encode()))
    return b''.join(chunks), boundary


def submit_video_job(base: str, fields: dict[str, str], filename: str, payload: bytes) -> str:
    body, boundary = multipart(fields, filename, 'video/mp4', payload)
    status, response = request(base + '/api/jobs', method='POST', body=body,
                              headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
                              timeout=30)
    if status != 200:
        raise RuntimeError(f'video job admission returned HTTP {status}: {response[:1000]!r}')
    job_id = json.loads(response).get('id')
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError('video job admission returned no job ID')
    return job_id


def wait_for_job(base: str, job_id: str, process, timeout: float, *, ready=None) -> dict:
    deadline = time.monotonic() + timeout
    job = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('packaged process exited while waiting for a job')
        status, payload = fetch(base + '/api/jobs')
        if status != 200:
            raise RuntimeError(f'job catalog returned HTTP {status}')
        job = next((item for item in json.loads(payload) if item.get('id') == job_id), None)
        if job is None:
            raise RuntimeError(f'job {job_id} disappeared')
        terminal = job.get('state') in {'completed', 'failed', 'canceled'}
        reached = ready(job) if ready is not None else terminal
        if reached:
            return job
        if terminal:
            raise RuntimeError(f'job finished before the required running state: {job}')
        time.sleep(.1)
    raise TimeoutError(f'job {job_id} exceeded {timeout:g}s: {job}')


def queue_failed_then_valid(base: str, fields: dict[str, str], source: Path, process,
                           timeout: float) -> tuple[str, str]:
    status, payload = request(base + '/api/queue/control', method='POST',
                              body=b'{"paused":true}', headers={'Content-Type': 'application/json'})
    if status != 200 or json.loads(payload).get('paused') is not True:
        raise RuntimeError('could not pause queue for failed-job recovery acceptance')
    failed_id = submit_video_job(base, fields, 'invalid-input.mp4', b'not a media container')
    valid_id = submit_video_job(base, fields, source.name, source.read_bytes())
    status, payload = fetch(base + '/api/jobs')
    queued = {item['id']: item.get('state') for item in json.loads(payload)} if status == 200 else {}
    if queued.get(failed_id) != 'queued' or queued.get(valid_id) != 'queued':
        raise RuntimeError('paused queue started or lost an admitted acceptance job')
    status, payload = request(base + '/api/queue/control', method='POST',
                              body=b'{"paused":false}', headers={'Content-Type': 'application/json'})
    if status != 200 or json.loads(payload).get('paused') is not False:
        raise RuntimeError('could not resume queue for failed-job recovery acceptance')
    failed = wait_for_job(base, failed_id, process, timeout)
    if failed.get('state') != 'failed' or not failed.get('error'):
        raise RuntimeError(f'invalid video did not fail with an actionable error: {failed}')
    if fetch(base + f'/api/jobs/{failed_id}/download')[0] != 404:
        raise RuntimeError('failed job exposed a downloadable partial output')
    return failed_id, valid_id


def group_members(group_id: int) -> list[int]:
    """Inspect this acceptance process group without relying on a ps command."""
    members = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if os.getpgid(int(entry.name)) == group_id:
                members.append(int(entry.name))
        except ProcessLookupError:
            continue
    return sorted(members)


def assert_history_ready(base: str) -> None:
    status, payload = fetch(base + '/api/status')
    history = json.loads(payload).get('history_capabilities', {}) if status == 200 else {}
    if history.get('enabled') is not True or history.get('state') != 'ready':
        raise RuntimeError(f'history degraded after a job boundary: {history}')


def cancel_active_export(base: str, isolated: Path, source: Path, fields: dict[str, str],
                         process, timeout: float) -> dict:
    long_source = isolated / 'cancel-active-export.mp4'
    result = subprocess.run([shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error',
                             '-stream_loop', '119', '-i', str(source), '-t', '120', '-c', 'copy',
                             '-y', str(long_source)], capture_output=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f'could not create cancellation fixture: {result.stderr[-2000:]!r}')
    job_id = submit_video_job(base, fields, long_source.name, long_source.read_bytes())
    running = wait_for_job(base, job_id, process, timeout,
                           ready=lambda job: job.get('state') == 'processing'
                           and (job.get('frames_processed') or 0) > 0)
    status, _ = request(base + f'/api/jobs/{job_id}/cancel', method='POST')
    if status != 200:
        raise RuntimeError(f'active cancellation returned HTTP {status}')
    canceled = wait_for_job(base, job_id, process, timeout)
    if canceled.get('state') != 'canceled':
        raise RuntimeError(f'active export did not reach canceled: {canceled}')
    if fetch(base + f'/api/jobs/{job_id}/download')[0] != 404:
        raise RuntimeError('canceled job exposed a downloadable partial output')
    deadline = time.monotonic() + 5
    while True:
        survivors = [pid for pid in group_members(process.pid) if pid != process.pid]
        if not survivors:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'owned media children survived cancellation: {survivors}')
        time.sleep(.1)
    recovery_id = submit_video_job(base, fields, 'after-cancel.mp4', source.read_bytes())
    recovery = wait_for_job(base, recovery_id, process, timeout)
    execution = recovery.get('execution', {})
    if (recovery.get('state') != 'completed' or execution.get('active_provider') != 'CPUExecutionProvider'
            or execution.get('inference_succeeded') is not True):
        raise RuntimeError(f'queue failed to recover after active cancellation: {recovery}')
    if fetch(base + f'/api/jobs/{recovery_id}/download')[0] != 200:
        raise RuntimeError('post-cancellation successful job has no downloadable output')
    assert_history_ready(base)
    return {'canceled_job_id': job_id, 'frames_before_cancel': running['frames_processed'],
            'terminal_state': canceled['state'], 'surviving_media_children': survivors,
            'recovery_job_id': recovery_id, 'recovery_state': recovery['state']}


def run_linux_model_smoke(base: str, isolated: Path, process: subprocess.Popen[bytes],
                          model_timeout: float, job_timeout: float) -> dict:
    """Install the pinned Small model and export a real, generated video on Linux."""
    model_id = 'depth-anything-v2-small'
    status, payload = request(f'{base}/api/models/{model_id}/install', method='POST')
    if status != 200:
        raise RuntimeError(f'model installation request returned HTTP {status}: {payload[:1000]!r}')

    deadline = time.monotonic() + model_timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'packaged process exited during model installation ({process.returncode})')
        status, payload = fetch(base + '/api/models')
        if status != 200:
            raise RuntimeError(f'model catalog returned HTTP {status} during installation')
        catalog = json.loads(payload)
        model = next((item for item in catalog.get('models', []) if item.get('id') == model_id), None)
        if model is None:
            raise RuntimeError(f'pinned model {model_id} is missing from the runtime catalog')
        if model.get('state') == 'ready':
            break
        if model.get('state') in {'failed', 'error'}:
            raise RuntimeError(f'pinned model installation failed: {model.get("error")}')
        time.sleep(1)
    else:
        raise TimeoutError(f'pinned model installation exceeded {model_timeout:g}s; last state: {model}')

    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('ffmpeg and ffprobe are required for the Linux media acceptance smoke test')
    source = isolated / 'linux-model-smoke-input.mp4'
    generated = subprocess.run(
        [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i',
         'testsrc2=size=320x180:rate=8:duration=1', '-f', 'lavfi', '-i',
         'sine=frequency=440:sample_rate=48000:duration=1', '-c:v', 'libx264',
         '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', '-y', str(source)],
        capture_output=True, text=True, timeout=120)
    if generated.returncode != 0 or not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f'could not generate the Linux acceptance video: {generated.stderr[-2000:]}')

    fields = {
        'layout': 'SBS', 'strength': '7.5', 'convergence': '0.5',
        'depth_mode': model_id, 'input_size': '256', 'resolution': 'source',
        'quality': 'fast', 'temporal_smoothing': '0', 'edge_refinement': 'false',
        'execution_mode': 'cpu', 'purpose': 'export', 'video_decode': 'software',
    }
    failed_id, job_id = queue_failed_then_valid(base, fields, source, process, job_timeout)
    job = wait_for_job(base, job_id, process, job_timeout)
    if job.get('state') != 'completed':
        raise RuntimeError(f'Linux model smoke job did not complete: {job}')
    execution = job.get('execution', {})
    if (execution.get('model_id') != model_id or execution.get('active_provider') != 'CPUExecutionProvider'
            or execution.get('inference_succeeded') is not True or execution.get('fallback_reason')):
        raise RuntimeError(f'Linux model smoke did not prove named CPU inference: {execution}')

    status, output = fetch(base + f'/api/jobs/{job_id}/download')
    if status != 200 or not output:
        raise RuntimeError(f'Linux model smoke output download failed with HTTP {status}')
    destination = isolated / 'linux-model-smoke-output.mp4'
    destination.write_bytes(output)
    probe = subprocess.run(
        [ffprobe, '-v', 'error', '-count_frames', '-show_entries',
         'format=duration:stream=codec_type,width,height,nb_read_frames', '-of', 'json', str(destination)],
        capture_output=True, text=True, timeout=60)
    if probe.returncode != 0:
        raise RuntimeError(f'ffprobe rejected Linux model smoke output: {probe.stderr[-2000:]}')
    media = json.loads(probe.stdout)
    streams = media.get('streams', [])
    video = next((item for item in streams if item.get('codec_type') == 'video'), None)
    audio = next((item for item in streams if item.get('codec_type') == 'audio'), None)
    if not video or not audio or (video.get('width'), video.get('height')) != (640, 180):
        raise RuntimeError(f'Linux model smoke output has invalid stereo geometry or no audio: {media}')
    if int(video.get('nb_read_frames') or 0) != 8:
        raise RuntimeError(f'Linux model smoke output has an unexpected decoded frame count: {video}')
    try:
        duration = float(media.get('format', {}).get('duration'))
    except (TypeError, ValueError):
        duration = float('nan')
    if not 0.8 <= duration <= 1.2:
        raise RuntimeError(f'Linux model smoke output duration is outside tolerance: {duration}')
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(destination), '-f', 'null', '-'],
                             capture_output=True, text=True, timeout=120)
    if decoded.returncode != 0:
        raise RuntimeError(f'full decode of Linux model smoke output failed: {decoded.stderr[-2000:]}')
    assert_history_ready(base)
    cancellation = cancel_active_export(base, isolated, source, fields, process, job_timeout)
    return {
        'model_id': model_id, 'model_state': model.get('state'),
        'provider': execution.get('active_provider'), 'inference_succeeded': True,
        'job_id': job_id, 'output_bytes': len(output),
        'output_sha256': hashlib.sha256(output).hexdigest(),
        'video': video, 'audio_present': True, 'duration_seconds': duration,
        'full_decode_succeeded': True,
        'failed_predecessor_job_id': failed_id, 'cancellation': cancellation,
    }


def set_package_read_only(package: Path) -> None:
    """Make an extracted CI candidate unwritable before launching it."""
    paths = [package, *package.rglob('*')]
    for path in paths:
        if path.is_symlink():
            target = path.resolve()
            if target != package and package not in target.parents:
                raise RuntimeError(f'package symlink escapes the extracted directory: {path} -> {target}')
            continue
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~0o222)


def assert_package_read_only(package: Path) -> None:
    for path in (package, *package.rglob('*')):
        if path.is_symlink():
            target = path.resolve()
            if target != package and package not in target.parents:
                raise RuntimeError(f'package symlink escapes the extracted directory: {path} -> {target}')
            continue
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            raise RuntimeError(f'package path remains writable: {path}')


def assert_clean_shutdown(returncode: int, server_log: str) -> None:
    # Uvicorn re-raises the captured signal after lifespan shutdown completes.
    if returncode not in (0, -signal.SIGTERM):
        raise RuntimeError(f'unexpected shutdown exit status: {returncode}')
    if 'StereoLab shutdown complete.' not in server_log:
        raise RuntimeError('shutdown has no evidence that worker/history cleanup completed')


def assert_bundled_ui(package: Path) -> None:
    # PyInstaller 6 one-directory builds place collected data in _internal.
    if not (package / '_internal' / 'static' / 'index.html').is_file():
        raise RuntimeError('bundled _internal/static/index.html is missing')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path, help='extracted PyInstaller StereoLab package directory')
    parser.add_argument('--timeout', type=float, default=45)
    parser.add_argument('--model-smoke', action='store_true',
                        help='download the pinned Small model and export a generated video with real CPU inference')
    parser.add_argument('--model-timeout', type=float, default=1200,
                        help='maximum seconds for the pinned model install (default: 1200)')
    parser.add_argument('--job-timeout', type=float, default=600,
                        help='maximum seconds for the generated neural media job (default: 600)')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--make-read-only', action='store_true',
                        help='remove write bits from this disposable extracted package before launch')
    args = parser.parse_args()
    if min(args.timeout, args.model_timeout, args.job_timeout) <= 0:
        parser.error('startup, model and job timeouts must all be positive')
    if not sys.platform.startswith('linux'):
        parser.error('the frozen Linux verifier must run on Linux')

    package = args.package.resolve()
    if args.make_read_only:
        try:
            set_package_read_only(package)
        except OSError as exc:
            parser.error(f'could not make the extracted package read-only: {exc}')
    try:
        assert_package_read_only(package)
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    executable = package / 'StereoLab'
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f'package executable is missing or not executable: {executable}')
    try:
        assert_bundled_ui(package)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not (package / 'stereolab').is_file():
        parser.error('the stereolab launcher is missing')
    try:
        build_report = json.loads((package / 'build-report.json').read_text(encoding='utf-8'))
        native_report = json.loads((package / 'native-dependencies.json').read_text(encoding='utf-8'))
        notice_report = json.loads((package / 'native-notices' / 'NATIVE_NOTICES.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        parser.error(f'package build or native-dependency report is missing or invalid: {exc}')
    if build_report.get('architecture') != 'x86_64':
        parser.error(f"package architecture is not x86_64: {build_report.get('architecture')!r}")
    if not build_report.get('build_fingerprint'):
        parser.error('package build report has no source fingerprint')
    lock_path = package / 'requirements-linux-lock.txt'
    if (not lock_path.is_file() or hashlib.sha256(lock_path.read_bytes()).hexdigest()
            != build_report.get('dependency_lock_sha256')):
        parser.error('packaged dependency lock does not match the build report')
    if native_report.get('architecture') != 'x86_64' or not native_report.get('elf_count'):
        parser.error('native dependency report does not describe an x86_64 ELF package')
    if native_report.get('missing_dependencies'):
        parser.error(f"package has unresolved ELF dependencies: {native_report['missing_dependencies']}")
    if (notice_report.get('elf_count') != native_report.get('elf_count')
            or not notice_report.get('packages')):
        parser.error('native notice inventory does not cover the scanned ELF files or has no system package provenance')

    with tempfile.TemporaryDirectory(prefix='stereolab-linux-acceptance-') as temporary:
        isolated = Path(temporary)
        data_home, state_home, cache_home = (isolated / name for name in ('data', 'state', 'cache'))
        for path in (data_home, state_home, cache_home):
            path.mkdir()
        port = free_port()
        environment = os.environ.copy()
        environment.update({
            'HOME': str(isolated / 'home'),
            'XDG_DATA_HOME': str(data_home),
            'XDG_STATE_HOME': str(state_home),
            'XDG_CACHE_HOME': str(cache_home),
            'STEREOLAB_HISTORY_INSTANCE': 'linux-acceptance',
        })
        (isolated / 'home').mkdir()
        log_path = isolated / 'server.log'
        result = {'package': str(package), 'port': port, 'checks': []}
        process = None
        try:
            with log_path.open('wb') as log:
                process = subprocess.Popen([str(package / 'stereolab'), '--port', str(port)],
                                           cwd=package, env=environment, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                result['checks'].append('launch')
                base = f'http://127.0.0.1:{port}'
                deadline = time.monotonic() + args.timeout
                last_error = None
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f'packaged process exited early with {process.returncode}')
                    try:
                        status, html = fetch(base + '/')
                        api_status, payload = fetch(base + '/api/status')
                        if status == 200 and api_status == 200:
                            break
                    except (OSError, URLError) as exc:
                        last_error = str(exc)
                    time.sleep(.25)
                else:
                    raise TimeoutError(f'packaged server did not become ready: {last_error}')
                state = json.loads(payload)
                if b'StereoLab' not in html or not state.get('runtime', {}).get('available'):
                    raise RuntimeError(f'UI/runtime health check failed: {state}')
                runtime = state['runtime']
                if runtime.get('supported_execution_modes') != ['auto', 'cpu'] or runtime.get('gpu_available') is not False:
                    raise RuntimeError(f'Linux package advertised unsupported GPU modes: {runtime}')
                if state.get('ffmpeg_found') is not True:
                    raise RuntimeError('Linux package could not find the required external FFmpeg installation')
                if state.get('build_fingerprint') != build_report['build_fingerprint']:
                    raise RuntimeError('running application build fingerprint does not match the packaged build report')
                output_dir = Path(state['output_dir']).resolve()
                expected_output = (data_home / 'stereolab').resolve()
                if expected_output not in output_dir.parents and output_dir != expected_output:
                    raise RuntimeError(f'output directory escaped isolated XDG data root: {output_dir}')
                expected_models = data_home / 'stereolab' / 'models'
                expected_uploads = cache_home / 'stereolab' / 'uploads'
                if not expected_models.is_dir() or not expected_uploads.is_dir():
                    raise RuntimeError('models or upload storage was not created under its isolated XDG root')
                expected_history = state_home / 'stereolab' / '.runtime' / 'history' / 'instance-linux-acceptance'
                if not state.get('history_capabilities', {}).get('enabled') or not expected_history.is_dir():
                    raise RuntimeError('application did not create history under the isolated XDG state root')
                models_status, models_payload = fetch(base + '/api/models')
                if models_status != 200:
                    raise RuntimeError(f'model catalog returned HTTP {models_status}')
                catalog = json.loads(models_payload)
                catalog_runtime = catalog.get('runtime', {})
                if catalog_runtime.get('supported_execution_modes') != ['auto', 'cpu'] or catalog_runtime.get('gpu_available') is not False:
                    raise RuntimeError(f'Linux model catalog advertises unsupported GPU modes: {catalog_runtime}')
                for model in catalog.get('models', []):
                    if re.search(r'(?i)DirectML|GPU|falls? back', model.get('description', '')):
                        raise RuntimeError(f"Linux model catalog contains GPU-specific wording for {model.get('id')}")
                result['checks'].extend(['http-ui', 'runtime-status', 'cpu-only-capabilities',
                                         'cpu-only-model-catalog', 'build-fingerprint',
                                         'elf-dependency-report', 'native-notice-inventory',
                                         'dependency-lock', 'xdg-isolation', 'read-only-install'])
                result['build_fingerprint'] = build_report['build_fingerprint']
                result['output_dir'] = str(output_dir)
                result['ffmpeg_found'] = state['ffmpeg_found']
                result['native_notice_inventory'] = {
                    'elf_count': notice_report['elf_count'],
                    'system_package_count': len(notice_report['packages']),
                    'unmapped_elf_count': sum(row.get('ownership_status') == 'unmapped'
                                              for row in notice_report.get('files', [])),
                    'errors': notice_report.get('errors', []),
                }
                if args.model_smoke:
                    result['model_smoke'] = run_linux_model_smoke(
                        base, isolated, process, args.model_timeout, args.job_timeout)
                    result['checks'].extend(['pinned-small-model-download-and-inference',
                                             'generated-video-stereo-export',
                                             'output-audio-frame-count-and-full-decode',
                                             'failed-job-queue-recovery', 'queue-pause-resume',
                                             'active-export-cancellation', 'cancellation-child-cleanup',
                                             'post-cancellation-neural-export'])
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=20)
                assert_clean_shutdown(process.returncode, log_path.read_text(encoding='utf-8', errors='replace'))
                result['shutdown_returncode'] = process.returncode
                result['checks'].append('sigterm-shutdown')
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            result['server_log'] = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
            error = sys.exc_info()[1]
            result['success'] = error is None
            if error is not None:
                result['error'] = str(error)
            print(json.dumps(result, indent=2, sort_keys=True))
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
