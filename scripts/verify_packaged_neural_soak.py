"""Run a generated multi-hour neural export through a private frozen EXE.

Uses the packaged-release verifier's exact-process and HTTP helpers, a fresh
private LOCALAPPDATA, a pinned accepted Small model copy, and a locally supplied
FFmpeg executable. The generated output is decoded in streaming chunks so the
full synthetic soak does not require buffering raw video/audio in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
import urllib.request

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


def stream_command(ffmpeg: Path, arguments: list[str], log_path: Path, timeout: float) -> tuple[str, int]:
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', *arguments]
    with log_path.open('wb') as error_log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=error_log, creationflags=subprocess.CREATE_NO_WINDOW)
        output_hash = hashlib.sha256()
        output_bytes = 0

        def drain() -> None:
            nonlocal output_bytes
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(1024 * 1024)
                if not chunk:
                    return
                output_hash.update(chunk)
                output_bytes += len(chunk)

        reader = threading.Thread(target=drain, name='ffmpeg-stream-hash', daemon=True)
        reader.start()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            reader.join()
            raise TimeoutError(f"FFmpeg exceeded {timeout:.0f}s: {command[0]} {arguments[:4]}")
        reader.join()
        if process.returncode != 0:
            detail = log_path.read_text(encoding='utf-8', errors='replace')
            raise RuntimeError(f"FFmpeg decode failed ({process.returncode}): {detail[:2000]}")
    return output_hash.hexdigest(), output_bytes


def make_streaming_media_record(work: Path, timeout: float):
    serial = 0

    def record(ffmpeg: Path, media_path: Path) -> dict:
        nonlocal serial
        serial += 1
        video_hash, video_bytes = stream_command(
            ffmpeg, ['-i', str(media_path), '-map', '0:v:0', '-an', '-pix_fmt', 'rgb24',
                     '-f', 'rawvideo', '-'], work / f'media-decode-{serial}-video.stderr.log', timeout)
        audio_hash, audio_bytes = stream_command(
            ffmpeg, ['-i', str(media_path), '-map', '0:a:0', '-vn', '-f', 's16le',
                     '-acodec', 'pcm_s16le', '-'], work / f'media-decode-{serial}-audio.stderr.log', timeout)
        ffprobe = ffmpeg.with_name('ffprobe.exe')
        probe = subprocess.run([str(ffprobe), '-v', 'error', '-count_frames', '-show_streams',
                                '-show_format', '-of', 'json', str(media_path)],
                               capture_output=True, text=True, check=True, timeout=timeout,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        require(video_bytes > 0 and audio_bytes > 0, f'Empty decoded audio/video in {media_path}')
        return {'encoded_sha256': digest(media_path), 'bytes': media_path.stat().st_size,
                'decoded_video_sha256': video_hash, 'decoded_video_bytes': video_bytes,
                'decoded_audio_sha256': audio_hash, 'decoded_audio_bytes': audio_bytes,
                'probe': json.loads(probe.stdout)}

    return record


def generate_source(ffmpeg: Path, output: Path, duration: int, fps: int, timeout: float) -> None:
    command = [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-f', 'lavfi', '-i', f'testsrc2=size=192x108:rate={fps}:duration={duration}',
               '-f', 'lavfi', '-i', f'sine=frequency=440:sample_rate=48000:duration={duration}',
               '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30', '-pix_fmt', 'yuv420p',
               '-c:a', 'aac', '-b:a', '32k', '-shortest', '-y', str(output)]
    result = subprocess.run(command, capture_output=True, timeout=timeout,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    require(result.returncode == 0,
            f'FFmpeg generated-source command failed: {result.stderr.decode(errors="replace")[:2000]}')


def sample_resources(base_url: str, job_id: str | None, stop: threading.Event,
                     interval: float, started: float, log_path: Path) -> None:
    def get_json(route: str):
        request = urllib.request.Request(base_url + route, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.load(response)

    while not stop.wait(interval):
        sample = {'elapsed_seconds': round(time.monotonic() - started, 3), 'at': utc()}
        try:
            resources = get_json('/api/diagnostics/resources')
            rows = get_json('/api/jobs')
            if job_id:
                job = next((row for row in rows if row.get('id') == job_id), None)
            else:
                job = next((row for row in rows if row.get('state') in {'queued', 'processing'}), None)
            sample['resources'] = {key: resources.get(key) for key in (
                'pid', 'working_set_bytes', 'peak_working_set_bytes', 'cpu_time_seconds',
                'memory_method', 'memory_error', 'cpu_time_error')}
            sample['job'] = ({key: job.get(key) for key in (
                'state', 'phase', 'progress', 'frames_processed', 'total_frames',
                'processing_fps', 'elapsed_seconds')} if job else None)
        except Exception as exc:
            sample['error'] = f'{type(exc).__name__}: {exc}'
        with log_path.open('a', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(sample, sort_keys=True) + '\n')
            stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True,
                        help='Fresh isolated LOCALAPPDATA root; must not exist.')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True,
                        help='Locally supplied FFmpeg; matching ffprobe.exe must be beside it.')
    parser.add_argument('--models', type=Path, default=ROOT / '.runtime/models')
    parser.add_argument('--expected-fingerprint', required=True)
    parser.add_argument('--duration-seconds', type=int, default=7200)
    parser.add_argument('--fps', type=int, default=1)
    parser.add_argument('--layout', choices=('SBS', 'XREAL_SBS'), default='SBS')
    parser.add_argument('--job-timeout', type=float, default=1800)
    parser.add_argument('--decode-timeout', type=float, default=1800)
    parser.add_argument('--sample-interval', type=float, default=15)
    args = parser.parse_args()

    for name in ('exe', 'data', 'report', 'ffmpeg', 'models'):
        setattr(args, name, getattr(args, name).resolve())
    require(os.name == 'nt', 'This verifier launches a Windows packaged application.')
    require(args.exe.is_file() and args.ffmpeg.is_file(), 'EXE and FFmpeg must exist.')
    require(args.ffmpeg.with_name('ffprobe.exe').is_file(), 'The matching ffprobe.exe must sit beside FFmpeg.')
    require(args.duration_seconds >= 600 and args.fps >= 1,
            'A soak must cover at least ten minutes and use at least one frame per second.')
    require(args.job_timeout > 0 and args.decode_timeout > 0 and args.sample_interval > 0,
            'Timeouts and resource sampling interval must be positive.')
    require(not args.data.exists(), 'The isolated data root must be new.')
    require(not args.report.exists(), 'Report path must be new.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.report.parent / 'isolated-temp'
    temp_path.mkdir()
    args.temp = temp_path
    args.clean_path = True

    ffmpeg_info = subprocess.run([str(args.ffmpeg), '-version'], capture_output=True,
                                 text=True, check=True, timeout=15,
                                 creationflags=subprocess.CREATE_NO_WINDOW).stdout
    report = {
        'schema_version': 1,
        'started_at': utc(),
        'passed': False,
        'scope': 'Actual frozen EXE; generated long-duration test pattern and sine audio; '
                 'pinned Small GPU neural inference; isolated LOCALAPPDATA; child PATH restricted '
                 'to Windows directories and the explicit external FFmpeg directory.',
        'limitations': [
            'Generated test pattern and tone only; no natural-scene quality or real-footage sync claim.',
            'Same Windows host; this does not prove a clean-machine installation.',
            'The external FFmpeg helper is supplied from a local install and is not part of the StereoLab EXE.',
            'The verifier forcibly stops its exact owned process after the run; graceful shutdown is not tested.',
            'Resource samples report process CPU/working set, not per-process GPU memory.',
        ],
        'ffmpeg': {'path': str(args.ffmpeg), 'sha256': digest(args.ffmpeg),
                   'version_first_lines': ffmpeg_info.splitlines()[:4]},
        'checks': {},
        'launches': [],
    }
    package_verifier.media_record = make_streaming_media_record(args.report.parent, args.decode_timeout)
    verifier = Verifier(args, report)
    samples_stop = threading.Event()
    samples_thread = None
    verifier.source = args.report.parent / 'generated-soak-source.mp4'
    report['exe'] = {'path': str(args.exe), 'bytes': args.exe.stat().st_size,
                     'sha256': digest(args.exe)}
    report['expected_fingerprint'] = args.expected_fingerprint
    report['source_spec'] = {'type': 'testsrc2 plus sine', 'width': 192, 'height': 108,
                             'fps': args.fps, 'duration_seconds': args.duration_seconds,
                             'expected_frames': args.fps * args.duration_seconds,
                             'audio_sample_rate': 48000, 'audio_bitrate': '32k'}
    report['resource_sample_log'] = str(args.report.parent / 'resource-samples.jsonl')

    try:
        verifier.save()
        require(digest(args.models / MODEL / 'model.onnx') == MODEL_SHA256,
                'Source Small model bytes do not match the pinned hash.')
        receipt = args.models / MODEL / 'model.accepted.json'
        accepted = json.loads(receipt.read_text(encoding='utf-8'))
        require(accepted.get('sha256') == MODEL_SHA256 and accepted.get('revision') == MODEL_REVISION,
                'Source Small model receipt is not the pinned accepted receipt.')
        target = args.data / 'StereoLab' / '.runtime' / 'models' / MODEL
        target.mkdir(parents=True)
        shutil.copy2(args.models / MODEL / 'model.onnx', target / 'model.onnx')
        shutil.copy2(receipt, target / receipt.name)
        report['model_copy'] = {'sha256': digest(target / 'model.onnx'), 'receipt': accepted,
                                'download_performed': False}

        generate_source(args.ffmpeg, verifier.source, args.duration_seconds, args.fps,
                        args.decode_timeout)
        report['source'] = package_verifier.media_record(args.ffmpeg, verifier.source)
        source_probe = report['source']['probe']
        source_video = next(s for s in source_probe['streams'] if s['codec_type'] == 'video')
        source_audio = next(s for s in source_probe['streams'] if s['codec_type'] == 'audio')
        expected_frames = args.duration_seconds * args.fps
        require(int(source_video['nb_read_frames']) == expected_frames,
                f'Generated source frame count mismatch: {source_video}')
        require(abs(float(source_probe['format']['duration']) - args.duration_seconds) <= 1.0,
                f'Generated source duration mismatch: {source_probe["format"]}')
        require(int(source_audio['sample_rate']) == 48000,
                f'Generated source sample rate mismatch: {source_audio}')
        verifier.check('generated_source_duration_frames_audio')

        verifier.start()
        require(verifier.request('GET', '/api/jobs').json() == [], 'Fresh isolated app has unexpected jobs.')
        require(verifier.request('GET', '/api/depth-cache').json()['entries'] == [],
                'Fresh isolated app has unexpected depth cache.')
        static_hashes = {}
        for source in sorted((ROOT / 'static').rglob('*')):
            if source.is_file():
                route = '/static/' + source.relative_to(ROOT / 'static').as_posix()
                content = verifier.request('GET', route).content
                actual = hashlib.sha256(content).hexdigest()
                require(actual == digest(source), f'Packaged static asset mismatch: {route}')
                static_hashes[route] = actual
        verifier.check('static_current_source_hashes', static_hashes)
        require(verifier.request('GET', '/').content ==
                (ROOT / 'static/index.html').read_text(encoding='utf-8').encode('utf-8'),
                'Packaged home page differs from current source.')
        verifier.check('packaged_home_matches_source')

        report['soak_started_at'] = utc()
        soak_started = time.monotonic()
        samples_thread = threading.Thread(
            target=sample_resources,
            args=(f'http://127.0.0.1:{verifier.port}', None, samples_stop,
                  args.sample_interval, soak_started, Path(report['resource_sample_log'])),
            name='packaged-resource-sampler', daemon=True)
        samples_thread.start()
        job, media = verifier.submit('synthetic-neural-soak', use_depth_cache='false')
        samples_stop.set()
        samples_thread.join(timeout=25)
        if samples_thread.is_alive():
            raise RuntimeError('Resource sampler did not stop after the neural export completed.')
        samples_thread = None
        if verifier.client is not None:
            verifier.client.close()
            verifier.client = None
        video = next(s for s in media['probe']['streams'] if s['codec_type'] == 'video')
        audio = next(s for s in media['probe']['streams'] if s['codec_type'] == 'audio')
        expected_geometry = {'SBS': (384, 108), 'XREAL_SBS': (3840, 1080)}[args.layout]
        require((video['width'], video['height'], int(video['nb_read_frames'])) ==
                (*expected_geometry, expected_frames), f'Output geometry/frame count mismatch: {video}')
        output_duration = float(media['probe']['format']['duration'])
        audio_duration = float(audio['duration'])
        require(abs(output_duration - args.duration_seconds) <= 1.0,
                f'Output duration mismatch: {output_duration}')
        require(abs(audio_duration - args.duration_seconds) <= 1.0,
                f'Output audio duration mismatch: {audio_duration}')
        video_start = float(video.get('start_time') or 0)
        audio_start = float(audio.get('start_time') or 0)
        require(abs(video_start - audio_start) <= 0.05,
                f'Output audio/video start offset exceeds 50ms: video={video_start}, audio={audio_start}')
        execution = job['execution']
        require(execution.get('active_provider') == 'DmlExecutionProvider'
                and execution.get('gpu_verified') is True
                and not execution.get('fallback_reason'),
                f'Neural job did not prove strict DirectML GPU use: {execution}')
        require(execution.get('inference_succeeded') is True,
                f'Neural inference did not complete successfully: {execution}')
        stats = execution.get('video_depth_stats') or {}
        require(int(stats.get('neural_frames', -1)) == expected_frames,
                f'Expected {expected_frames} neural frames but observed {stats}')

        output_path = args.report.parent / 'synthetic-neural-soak-output.mp4'
        report['soak'] = {
            'job_id': job['id'],
            'state': job['state'],
            'wall_seconds': round(time.monotonic() - soak_started, 3),
            'media_duration_seconds': output_duration,
            'audio_duration_seconds': audio_duration,
            'frames_expected': expected_frames,
            'frames_decoded': int(video['nb_read_frames']),
            'neural_frames': int(stats['neural_frames']),
            'audio_video_start_offset_seconds': round(audio_start - video_start, 6),
            'output_path': str(output_path),
            'output_sha256': media['encoded_sha256'],
            'decoded_video_sha256': media['decoded_video_sha256'],
            'decoded_audio_sha256': media['decoded_audio_sha256'],
            'decoded_video_bytes': media['decoded_video_bytes'],
            'decoded_audio_bytes': media['decoded_audio_bytes'],
            'execution': execution,
            'depth_cache': job['depth_cache'],
        }
        resource_path = Path(report['resource_sample_log'])
        if resource_path.exists():
            samples = [json.loads(line) for line in resource_path.read_text(encoding='utf-8').splitlines() if line]
        else:
            samples = []
        rss = [sample.get('resources', {}).get('working_set_bytes') for sample in samples
               if isinstance(sample.get('resources', {}).get('working_set_bytes'), int)]
        report['resource_sampling'] = {
            'count': len(samples),
            'errors': sum('error' in sample for sample in samples),
            'interval_seconds': args.sample_interval,
            'working_set_bytes_min': min(rss) if rss else None,
            'working_set_bytes_max': max(rss) if rss else None,
            'samples': samples,
        }
        require(len(samples) >= 2, f'Expected resource trend samples during the soak, got {len(samples)}.')
        require(not any('error' in sample for sample in samples),
                'One or more periodic resource samples failed.')
        verifier.check('multi_hour_generated_neural_export_full_decode_audio')
        verifier.check('periodic_resource_trend_samples', len(samples))
        report['passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc),
                             'traceback': traceback.format_exc()}
        report['passed'] = False
    finally:
        samples_stop.set()
        if samples_thread is not None:
            samples_thread.join(timeout=25)
        report['finished_at'] = utc()
        if verifier.client is not None:
            verifier.client.close()
            verifier.client = None
        try:
            verifier.stop()
        except Exception as exc:
            report['passed'] = False
            report['cleanup_failure'] = f'{type(exc).__name__}: {exc}'
        verifier.save()

    print(json.dumps({'passed': report['passed'], 'report': str(args.report),
                      'failure': report.get('failure'),
                      'cleanup_failure': report.get('cleanup_failure'),
                      'soak': report.get('soak'),
                      'resource_sample_count': report.get('resource_sampling', {}).get('count')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
