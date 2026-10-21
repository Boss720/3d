"""Exercise one packaged StereoLab process across a sustained wall-clock run.

Uses the existing packaged verifier's identity-checked isolated launcher,
generates only a short test pattern, submits recurring uncached neural jobs,
and samples the owned server's health and resources. It never touches the
repository's user-facing dist executable or any process it did not launch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import threading
import time
import traceback
from types import SimpleNamespace
import uuid

import httpx

import verify_packaged_release as package_verifier

ROOT = package_verifier.ROOT
MODEL = package_verifier.MODEL
MODEL_SHA256 = package_verifier.MODEL_SHA256
MODEL_REVISION = package_verifier.MODEL_REVISION
Verifier = package_verifier.Verifier
digest = package_verifier.digest
ffmpeg_run = package_verifier.ffmpeg_run
require = package_verifier.require
utc = package_verifier.utc


def sample_resources(verifier: Verifier, stop: threading.Event,
                     interval: float, started: float, log_path: Path,
                     errors: list[str]) -> None:
    identity = verifier.owned[-1]
    url = f"http://127.0.0.1:{verifier.port}"
    with httpx.Client(base_url=url, timeout=15, trust_env=False) as client:
        while not stop.wait(interval):
            row = {'elapsed_seconds': round(time.monotonic() - started, 3), 'at': utc()}
            try:
                status = client.get('/api/status')
                status.raise_for_status()
                resources = client.get('/api/diagnostics/resources')
                resources.raise_for_status()
                jobs = client.get('/api/jobs')
                jobs.raise_for_status()
                resource_data = resources.json()
                require(resource_data.get('pid') == identity['pid'],
                        f"Server PID changed: expected {identity['pid']}, got {resource_data.get('pid')}")
                row['status_ok'] = True
                row['resources'] = {key: resource_data.get(key) for key in (
                    'pid', 'working_set_bytes', 'peak_working_set_bytes', 'cpu_time_seconds',
                    'memory_method', 'memory_error', 'cpu_time_error')}
                rows = jobs.json()
                row['jobs'] = {'total': len(rows),
                               'active': sum(job.get('state') in {'queued', 'processing'} for job in rows),
                               'failed': sum(job.get('state') == 'failed' for job in rows)}
            except Exception as exc:  # retained and treated as a failed observation
                row['error'] = f'{type(exc).__name__}: {exc}'
                errors.append(row['error'])
            with log_path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
                stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True,
                        help='New isolated LOCALAPPDATA root; must not exist.')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT / '.runtime/models')
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-fingerprint', required=True)
    parser.add_argument('--duration-seconds', type=int, default=7200,
                        help='At least 600 seconds; defaults to two hours.')
    parser.add_argument('--sample-interval', type=float, default=15)
    parser.add_argument('--job-interval', type=float, default=300)
    parser.add_argument('--job-timeout', type=float, default=600)
    args = parser.parse_args()

    for name in ('exe', 'data', 'report', 'ffmpeg', 'models'):
        setattr(args, name, getattr(args, name).resolve())
    require(os.name == 'nt', 'This verifier launches a Windows packaged application.')
    require(args.duration_seconds >= 600, 'Wall-clock observation must last at least 600 seconds.')
    require(args.sample_interval > 0 and args.job_interval > 0 and args.job_timeout > 0,
            'Sampling, job cadence and job timeout must be positive.')
    require(args.exe.is_file() and args.ffmpeg.is_file(), 'EXE and FFmpeg must exist.')
    require(args.ffmpeg.with_name('ffprobe.exe').is_file(), 'Matching ffprobe.exe must be beside FFmpeg.')
    require(not args.data.exists(), '--data must be a new private LOCALAPPDATA directory.')
    require(not args.report.exists(), '--report must be new.')
    require(digest(args.exe) == args.expected_sha256, 'Candidate EXE SHA-256 does not match the requested identity.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(os.environ.get('TEMP', str(args.report.parent))).resolve()
    temp = temp_root / f"StereoLab-wallclock-{uuid.uuid4().hex}"
    temp.mkdir()

    report = {
        'schema_version': 1,
        'started_at': utc(),
        'passed': False,
        'scope': (f"Same-process {args.duration_seconds}-second wall-clock observation of the actual frozen EXE, "
                  'with generated-only short video jobs using strict named Small DirectML and a restricted child PATH.'),
        'limitations': [
            'Same Windows host; this is not a clean-machine installation test.',
            'Generated test pattern and tone only; no natural-scene quality or real-footage AV-sync claim.',
            'The external FFmpeg helper is local and is not bundled in StereoLab.',
            'The exact owned process is forcibly stopped at the end; graceful shutdown is not tested.',
            'Physical XREAL 1S playback remains unverified.',
        ],
        'exe': {'path': str(args.exe), 'bytes': args.exe.stat().st_size, 'sha256': digest(args.exe)},
        'expected_fingerprint': args.expected_fingerprint,
        'isolated_temp': str(temp),
        'checks': {},
        'launches': [],
        'jobs': [],
        'sample_log': str(args.report.parent / 'resource-samples.jsonl'),
        'resource_sample_errors': [],
    }
    verifier_args = SimpleNamespace(
        exe=args.exe, data=args.data, report=args.report, ffmpeg=args.ffmpeg,
        models=args.models, expected_fingerprint=args.expected_fingerprint,
        layout='SBS', job_timeout=args.job_timeout, clean_path=True, temp=temp,
        node=None, playwright=None, chrome=Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),
    )
    verifier = Verifier(verifier_args, report)
    sampler_stop = threading.Event()
    sampler = None
    failure = None
    try:
        model = args.models / MODEL / 'model.onnx'
        receipt_path = model.with_suffix('.accepted.json')
        require(digest(model) == MODEL_SHA256, 'Source model does not match the pinned accepted Small model.')
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        require(receipt.get('sha256') == MODEL_SHA256 and receipt.get('revision') == MODEL_REVISION,
                'Source model acceptance receipt does not match the pinned revision.')
        model_target = args.data / 'StereoLab' / '.runtime' / 'models' / MODEL
        model_target.mkdir(parents=True)
        shutil.copy2(model, model_target / model.name)
        shutil.copy2(receipt_path, model_target / receipt_path.name)
        report['model_copy'] = {'sha256': digest(model_target / 'model.onnx'),
                                'revision': receipt['revision'], 'copied_without_download': True}

        verifier.source = args.report.parent / 'generated-uptime-source.mp4'
        ffmpeg_run(args.ffmpeg, ['-f', 'lavfi', '-i', 'testsrc2=size=192x108:rate=4:duration=2',
                                 '-f', 'lavfi', '-i', 'sine=frequency=660:sample_rate=48000:duration=2',
                                 '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30',
                                 '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '32k', '-shortest',
                                 '-y', str(verifier.source)])
        report['source'] = package_verifier.media_record(args.ffmpeg, verifier.source)

        status = verifier.start()
        require(status.get('build_fingerprint') == args.expected_fingerprint,
                'Started package source fingerprint does not match the requested identity.')
        report['checks']['exact_candidate_isolated_launch'] = True
        started = time.monotonic()
        deadline = started + args.duration_seconds
        report['observation_started_at'] = utc()
        report['observation_target_seconds'] = args.duration_seconds
        sample_errors: list[str] = report['resource_sample_errors']
        sampler = threading.Thread(target=sample_resources,
                                   args=(verifier, sampler_stop, args.sample_interval, started,
                                         Path(report['sample_log']), sample_errors),
                                   name='packaged-wallclock-sampler', daemon=True)
        sampler.start()

        next_job = started
        report['job_checks'] = []
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_job:
                job_name = f"wallclock-{len(report['job_checks']) + 1:03d}"
                job, media = verifier.submit(job_name, use_depth_cache='false')
                execution = job.get('execution') or {}
                require(execution.get('active_provider') == 'DmlExecutionProvider'
                        and execution.get('gpu_verified') is True
                        and not execution.get('fallback_reason'),
                        f"{job_name} did not use strict named Small DirectML: {execution}")
                video = next(row for row in media['probe']['streams'] if row['codec_type'] == 'video')
                audio = next(row for row in media['probe']['streams'] if row['codec_type'] == 'audio')
                require((int(video['width']), int(video['height']), int(video['nb_read_frames'])) == (384, 108, 8),
                        f"{job_name} output geometry/frame count changed: {video}")
                require(int(audio['sample_rate']) == 48000 and float(media['probe']['format']['duration']) >= 1.9,
                        f"{job_name} audio/duration validation failed: {audio}")
                report['job_checks'].append({'name': job_name, 'completed_at': utc(),
                                             'state': job['state'], 'execution': execution,
                                             'output_sha256': media['encoded_sha256'],
                                             'decoded_video_sha256': media['decoded_video_sha256'],
                                             'decoded_audio_sha256': media['decoded_audio_sha256']})
                next_job = max(next_job + args.job_interval, time.monotonic() + args.job_interval)
                verifier.save()
            time.sleep(min(1, max(0.1, deadline - time.monotonic()),
                           max(0.1, next_job - time.monotonic())))

        final_status = verifier.request('GET', '/api/status').json()
        require(final_status.get('build_fingerprint') == args.expected_fingerprint,
                'Build fingerprint changed during the wall-clock run.')
        rows = verifier.request('GET', '/api/jobs').json()
        require(not any(row.get('state') in {'queued', 'processing'} for row in rows),
                'An application job remained active after the observation period.')
        require(not any(row.get('state') == 'failed' for row in rows),
                'A packaged job failed during the wall-clock run.')
        report['observation_completed_at'] = utc()
        report['observation_seconds'] = round(time.monotonic() - started, 3)
        report['final_job_states'] = {'total': len(rows),
                                      'completed': sum(row.get('state') == 'completed' for row in rows),
                                      'failed': sum(row.get('state') == 'failed' for row in rows)}
        report['checks']['same_process_wallclock_health_and_repeated_neural_jobs'] = True
    except Exception:
        failure = traceback.format_exc()
        report['error'] = failure
    finally:
        sampler_stop.set()
        if sampler:
            sampler.join(timeout=max(30, args.sample_interval + 5))
        if verifier.owned:
            try:
                verifier.stop()
            except Exception:
                report['stop_error'] = traceback.format_exc()
        report['resource_sample_errors'] = sample_errors if 'sample_errors' in locals() else []
        report['ended_at'] = utc()
        report['passed'] = failure is None and not report['resource_sample_errors'] and not report.get('stop_error')
        verifier.save()
    if failure:
        print(failure)
        return 1
    require(report['passed'], f"Wall-clock soak did not pass cleanly: {report.get('stop_error') or report['resource_sample_errors']}")
    with Path(report['sample_log']).open(encoding='utf-8') as stream:
        sample_count = sum(1 for _ in stream)
    print(json.dumps({'passed': report['passed'], 'report': str(args.report),
                      'observation_seconds': report.get('observation_seconds'),
                      'completed_jobs': len(report.get('job_checks', [])),
                      'resource_samples': sample_count,
                      'resource_sample_errors': report['resource_sample_errors']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
