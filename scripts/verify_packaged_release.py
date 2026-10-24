"""Exercise a real frozen StereoLab over HTTP in a fresh private data directory.

No source imports, mocks, model downloads, existing-server mutations or public
release writes. Requires httpx in the verifier Python and installed FFmpeg.
Retains generated media, outputs, process identities, logs and a JSON report on
success or failure. Windows only; starts the actual EXE hidden via Start-Process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import uuid

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'depth-anything-v2-small'
MODEL_SHA256 = 'afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c'
MODEL_REVISION = '4472b7362082ad9968fee890ca0f1e5aca36b93d'


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_execution(execution, execution_mode, *, cached=False):
    expected_provider = {'gpu': 'DmlExecutionProvider', 'cuda': 'CUDAExecutionProvider'}[execution_mode]
    if cached:
        require(execution.get('device') == 'cpu' and execution.get('active_provider') is None
                and execution.get('gpu_verified') is False and execution.get('depth_cache_hit') is True
                and execution.get('cached_execution', {}).get('active_provider') == expected_provider,
                f'Cached execution provenance is inaccurate for {execution_mode}: {execution}')
    else:
        require(execution.get('active_provider') == expected_provider and execution.get('gpu_verified') is True
                and not execution.get('fallback_reason'),
                f'Strict Small {execution_mode} not verified: {execution}')


def quote_ps(value):
    return "'" + str(value).replace("'", "''") + "'"


def canonical_launcher_identity(identity, expected_exe):
    expected_exe = Path(expected_exe).resolve()
    require(isinstance(identity.get('pid'), int) and identity['pid'] > 0
            and bool(identity.get('started')), 'Launcher PID or creation time is missing.')
    # Start-Process can return a null Path immediately for a one-folder EXE.
    # Its FilePath argument is known; retain the returned PID and creation time
    # so stop() still independently checks both path and time before termination.
    observed_path = identity.get('path')
    require(observed_path is None or Path(observed_path).resolve() == expected_exe,
            f'Launcher is not the requested executable: {identity}')
    return {**identity, 'path': str(expected_exe)}


def powershell(script):
    # Start-Process children can retain the outer PowerShell pipe handles even
    # with their own redirected streams. Real files keep communicate from
    # waiting forever after PowerShell exits, and retain every helper failure.
    logs = ROOT / '.runtime/production-package/powershell'
    logs.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stdout, stderr = logs / f'{token}.stdout.log', logs / f'{token}.stderr.log'
    with stdout.open('wb') as out, stderr.open('wb') as err:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                                 "$ErrorActionPreference='Stop'; " + script],
                                stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW)
    output = stdout.read_text(encoding='utf-8-sig', errors='replace')
    error = stderr.read_text(encoding='utf-8-sig', errors='replace')
    require(result.returncode == 0, f'PowerShell failed: {error or output}')
    return output.strip()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def process_identity(pid):
    return json.loads(powershell(
        f"$p=Get-Process -Id {int(pid)}; $c=Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}'; "
        "@{pid=$p.Id; path=$p.Path; started=$p.StartTime.ToUniversalTime().ToString('o'); "
        "parent_pid=$c.ParentProcessId}|ConvertTo-Json -Compress"))


def ffmpeg_run(ffmpeg, arguments):
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', *arguments],
                            capture_output=True, timeout=120,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    require(result.returncode == 0, f'FFmpeg failed: {result.stderr.decode(errors="replace")}')
    return result.stdout


def media_record(ffmpeg, path):
    video = ffmpeg_run(ffmpeg, ['-i', str(path), '-map', '0:v:0', '-an', '-pix_fmt', 'rgb24',
                               '-f', 'rawvideo', '-'])
    audio = ffmpeg_run(ffmpeg, ['-i', str(path), '-map', '0:a:0', '-vn', '-f', 's16le',
                               '-acodec', 'pcm_s16le', '-'])
    ffprobe = ffmpeg.with_name('ffprobe.exe')
    probe = subprocess.run([str(ffprobe), '-v', 'error', '-count_frames', '-show_streams',
                            '-show_format', '-of', 'json', str(path)],
                           capture_output=True, text=True, check=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW)
    require(video and audio, f'Empty decoded audio/video in {path}')
    return {'path': str(path), 'encoded_sha256': digest(path), 'bytes': path.stat().st_size,
            'decoded_video_sha256': hashlib.sha256(video).hexdigest(),
            'decoded_video_bytes': len(video), 'decoded_audio_sha256': hashlib.sha256(audio).hexdigest(),
            'decoded_audio_bytes': len(audio), 'probe': json.loads(probe.stdout)}


def compare_repeat_media(ffmpeg, first, repeat, *, allow_hardware_variation=False):
    """Keep exact audio/geometry/timing; optionally measure lossy encode variation.

    A hardware encoder can vary despite identical raw input. The opt-in bound
    is 60 dB PSNR for EVERY frame (RMS <= 0.255 in 8-bit RGB channels), rather
    than an aggregate score that could hide a changed frame. This is a media
    similarity check, not proof of identical raw rendered frames.
    """
    require(first['decoded_audio_sha256'] == repeat['decoded_audio_sha256']
            and first['decoded_audio_bytes'] == repeat['decoded_audio_bytes'],
            'Repeated output audio changed.')
    videos = [next(s for s in item['probe']['streams'] if s['codec_type'] == 'video')
              for item in (first, repeat)]
    for field in ('width', 'height', 'nb_read_frames', 'avg_frame_rate', 'start_time', 'duration'):
        require(videos[0].get(field) == videos[1].get(field), f'Repeated video {field} changed.')
    audios = [next((s for s in item['probe']['streams'] if s['codec_type'] == 'audio'), None)
              for item in (first, repeat)]
    require(all(audio is not None for audio in audios), 'Repeated output audio stream is missing.')
    for field in ('start_time', 'duration', 'sample_rate', 'channels', 'channel_layout'):
        require(audios[0].get(field) == audios[1].get(field), f'Repeated audio {field} changed.')
    require(first['probe'].get('format', {}).get('duration') == repeat['probe'].get('format', {}).get('duration'),
            'Repeated container duration changed.')
    require(first['decoded_video_bytes'] == repeat['decoded_video_bytes'], 'Repeated video length changed.')
    encoded_equal = first['encoded_sha256'] == repeat['encoded_sha256']
    video_equal = first['decoded_video_sha256'] == repeat['decoded_video_sha256']
    result = {'encoded_identical': encoded_equal, 'decoded_video_identical': video_equal,
              'decoded_audio_identical': True}
    if encoded_equal and video_equal:
        return result
    require(allow_hardware_variation, 'Repeated encoding differs; hardware variation was not enabled.')
    if video_equal:
        return result
    decoded = [ffmpeg_run(ffmpeg, ['-i', item['path'], '-map', '0:v:0', '-an',
                                  '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
               for item in (first, repeat)]
    require(len(decoded[0]) == len(decoded[1]) == first['decoded_video_bytes'],
            'Repeated decoded video size changed during comparison.')
    frame_bytes = int(videos[0]['width']) * int(videos[0]['height']) * 3
    require(frame_bytes > 0 and len(decoded[0]) == frame_bytes * int(videos[0]['nb_read_frames']),
            'Repeated decoded frame geometry is invalid.')
    scores = []
    for offset in range(0, len(decoded[0]), frame_bytes):
        arrays = [np.frombuffer(data, dtype=np.uint8, count=frame_bytes, offset=offset)
                  for data in decoded]
        difference = arrays[0].astype(np.float32) - arrays[1]
        mse = float(np.mean(difference * difference, dtype=np.float64))
        score = None if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        require(score is None or score >= 60, f'Repeated frame {len(scores)} exceeds hardware variation bound: {score} dB')
        scores.append(score)
    result.update({'per_frame_psnr_db': scores, 'minimum_psnr_db': 60,
                   'scope': 'Lossy output similarity only; raw render identity requires separate evidence.'})
    return result


class Verifier:
    def __init__(self, args, report):
        self.args, self.report = args, report
        self.owned = []
        self.client = None
        self.port = None
        self.work = args.report.parent

    def save(self):
        self.args.report.write_text(json.dumps(self.report, indent=2), encoding='utf-8')

    def check(self, name, value=True):
        self.report['checks'][name] = value
        self.save()

    def request(self, method, route, **kwargs):
        response = self.client.request(method, route, **kwargs)
        require(response.is_success, f'{method} {route}: HTTP {response.status_code}: {response.text[:1000]}')
        return response

    def start(self):
        self.port = free_port()
        number = len(self.report['launches']) + 1
        stdout, stderr = self.work / f'exe-{number}.stdout.log', self.work / f'exe-{number}.stderr.log'
        temp_dir = getattr(self.args, 'temp', None)
        temp_env = (f"$env:TEMP={quote_ps(temp_dir)}; $env:TMP={quote_ps(temp_dir)}; "
                    if temp_dir is not None else '')
        clean_path_env = ''
        if getattr(self.args, 'clean_path', False):
            system_root = Path(os.environ.get('SystemRoot', r'C:\Windows'))
            child_path = os.pathsep.join((str(system_root / 'System32'), str(system_root),
                                          str(self.args.ffmpeg.parent)))
            clean_path_env = f"$env:PATH={quote_ps(child_path)}; "
        command = (
            f"$env:LOCALAPPDATA={quote_ps(self.args.data)}; "
            f"$env:VIDEO3D_FFMPEG={quote_ps(self.args.ffmpeg)}; "
            f"{clean_path_env}"
            f"{temp_env}"
            "$env:STEREOLAB_HISTORY_INSTANCE='packaged-acceptance'; "
            f"$p=Start-Process -FilePath {quote_ps(self.args.exe)} -ArgumentList @('--port','{self.port}') "
            f"-WorkingDirectory {quote_ps(self.work)} -WindowStyle Hidden -PassThru "
            f"-RedirectStandardOutput {quote_ps(stdout)} -RedirectStandardError {quote_ps(stderr)}; "
            "@{pid=$p.Id; path=$p.Path; started=$p.StartTime.ToUniversalTime().ToString('o'); "
            "parent_pid=$PID}|ConvertTo-Json -Compress")
        root = canonical_launcher_identity(json.loads(powershell(command)), self.args.exe)
        self.owned.append(root)
        launch = {'started_at': utc(), 'port': self.port, 'launcher': root,
                  'stdout': str(stdout), 'stderr': str(stderr)}
        if getattr(self.args, 'clean_path', False):
            launch['child_path_policy'] = 'Windows system directories plus explicit FFmpeg directory'
        self.report['launches'].append(launch)
        self.save()
        self.client = httpx.Client(base_url=f'http://127.0.0.1:{self.port}', timeout=30, trust_env=False)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if not powershell(f"[bool](Get-Process -Id {root['pid']} -ErrorAction SilentlyContinue)").endswith('True'):
                raise AssertionError('Packaged launcher exited before HTTP readiness; inspect EXE logs.')
            try:
                response = self.client.get('/api/status')
                if response.is_success:
                    break
            except httpx.TransportError:
                pass
            time.sleep(.2)
        else:
            raise AssertionError('Packaged server did not become ready; inspect EXE logs.')
        pid = self.request('GET', '/api/diagnostics/resources').json()['pid']
        server = process_identity(pid)
        require(Path(server['path']).resolve() == self.args.exe, 'HTTP server is not the requested executable.')
        require(server['pid'] == root['pid'] or server['parent_pid'] == root['pid'],
                'HTTP server is not this launch or its direct child.')
        if server['pid'] != root['pid']:
            self.owned.append(server)
        launch['server'] = server
        launch['ready_at'] = utc()
        status = response.json()
        launch['status'] = status
        require(Path(status['output_dir']).resolve() == self.args.data / 'StereoLab' / 'outputs',
                'Packaged writes are not in isolated LOCALAPPDATA.')
        require(status['ffmpeg_found'], 'Packaged app could not find explicitly configured FFmpeg.')
        require(status['build_fingerprint'], 'Frozen build fingerprint is missing.')
        require(status['history_capabilities']['state'] == 'ready', 'Frozen history did not initialize.')
        if self.args.expected_fingerprint:
            require(status['build_fingerprint'] == self.args.expected_fingerprint, 'Unexpected packaged source fingerprint.')
        self.save()
        return status

    def stop(self):
        if self.client:
            self.client.close()
            self.client = None
        errors = []
        for identity in list(reversed(self.owned)):
            # Windows process start time + EXE path prevents terminating a reused PID.
            script = (f"$p=Get-Process -Id {identity['pid']} -ErrorAction SilentlyContinue; if($p){{ "
                      f"if($p.Path -ne {quote_ps(identity['path'])} -or "
                      f"$p.StartTime.ToUniversalTime().ToString('o') -ne {quote_ps(identity['started'])})"
                      "{throw 'Owned process identity changed; refusing termination'}; "
                      "Stop-Process -InputObject $p -Force; $p.WaitForExit(10000)|Out-Null }")
            try:
                powershell(script)
                self.owned.remove(identity)
            except Exception as exc:
                # A frozen EXE may exit during startup (for example, if its
                # one-file runtime cannot extract). Treat an already absent
                # exact PID as stopped; the independent port check below must
                # still prove that its listener closed.
                try:
                    alive = powershell(
                        f"[bool](Get-Process -Id {identity['pid']} -ErrorAction SilentlyContinue)")
                    if alive.endswith('False'):
                        self.owned.remove(identity)
                    else:
                        errors.append(str(exc))
                except Exception as check_exc:
                    errors.append(f'{exc}; process exit could not be verified: {check_exc}')
        if self.port:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', self.port)) != 0:
                        break
                time.sleep(.1)
            else:
                errors.append(f'Owned port {self.port} remains open')
            self.report.setdefault('stops', []).append({'at': utc(), 'port': self.port,
                'method': 'forced exact EXE PID and creation time; simulates process interruption', 'errors': errors})
        self.save()
        require(not errors, '; '.join(errors))

    def submit(self, name, **overrides):
        settings = {'depth_mode': MODEL, 'execution_mode': self.args.execution_mode, 'layout': self.args.layout,
                    'input_size': '256', 'resolution': 'source', 'quality': 'fast',
                    'temporal_smoothing': '0.35', 'use_depth_cache': 'true', 'video_decode': 'software',
                    'output_name_template': '{name}_{layout}', **overrides}
        require(not any(j['state'] in {'queued', 'processing'} for j in self.request('GET', '/api/jobs').json()),
                'Owned API already has active jobs.')
        with self.source.open('rb') as stream:
            admitted = self.request('POST', '/api/jobs', data=settings,
                                    files={'file': (name + '.mp4', stream, 'video/mp4')}).json()
        self.report.setdefault('admissions', []).append(admitted)
        self.save()
        deadline = time.monotonic() + self.args.job_timeout
        while time.monotonic() < deadline:
            job = next(j for j in self.request('GET', '/api/jobs').json() if j['id'] == admitted['id'])
            if job['state'] in {'completed', 'failed', 'canceled'}:
                self.report.setdefault('jobs', []).append(job)
                self.save()
                require(job['state'] == 'completed', f'Packaged job failed: {job}')
                output = self.work / (name + '-output.mp4')
                output.write_bytes(self.request('GET', f"/api/jobs/{job['id']}/download").content)
                media = media_record(self.args.ffmpeg, output)
                copy_directory = self.work / f'{name}-copy-target'
                copy_directory.mkdir()
                copied = self.request('POST', f"/api/jobs/{job['id']}/copy",
                                      json={'directory': str(copy_directory)}).json()
                copied_path = Path(copied['path'])
                require(copied_path.resolve().parent == copy_directory.resolve(),
                        f'Packaged save-copy escaped its selected destination: {copied_path}')
                require(copied_path.read_bytes() == output.read_bytes(),
                        'Packaged save-copy bytes differ from the completed output.')
                collision = self.client.post(f"/api/jobs/{job['id']}/copy",
                                             json={'directory': str(copy_directory)})
                require(collision.status_code == 409,
                        f'Packaged save-copy overwrote an existing file: HTTP {collision.status_code}')
                require(copied_path.read_bytes() == output.read_bytes(),
                        'Packaged save-copy changed the existing destination after collision.')
                self.report.setdefault('output_copies', []).append({
                    'job_id': job['id'], 'path': str(copied_path),
                    'sha256': digest(copied_path), 'matches_download': True,
                    'existing_file_overwrite_refused': True})
                self.check('packaged_save_copy_preserves_bytes_and_refuses_overwrite')
                return job, media
            time.sleep(.2)
        raise AssertionError(f'Packaged job {admitted["id"]} exceeded deadline')

    def browser(self):
        if not self.args.node or not self.args.playwright:
            self.report['browser'] = {'performed': False, 'reason': 'Pass --node and --playwright for screenshot validation.'}
            return
        browser_script = self.work / 'browser-check.cjs'
        browser_script.write_text('''const fs=require('fs'),assert=require('assert/strict');
const {chromium}=require(process.argv[2]);
(async()=>{const browser=await chromium.launch({headless:true,executablePath:process.argv[5]});
try { const page=await browser.newPage({viewport:{width:1280,height:960}}),errors=[];
page.on('pageerror',e=>errors.push(e.message));
await page.goto(process.argv[3],{waitUntil:'networkidle'});
const previewStatus=page.locator('#previewState'),queueStatus=page.locator('#queueLiveStatus');
assert.equal(await previewStatus.getAttribute('role'),'status');assert.equal(await previewStatus.getAttribute('aria-live'),'polite');
assert.equal(await previewStatus.evaluate(e=>e.closest('details')),null);assert.equal(await previewStatus.isVisible(),true);
assert.equal(await queueStatus.getAttribute('role'),'status');assert.equal(await queueStatus.getAttribute('aria-live'),'polite');
assert.equal(await queueStatus.getAttribute('aria-atomic'),'true');assert.equal(await queueStatus.evaluate(e=>e.closest('details')),null);
const dropZone=page.locator('#dropZone'),browse=page.locator('#browseButton');
assert.equal(await dropZone.getAttribute('role'),null);assert.equal(await dropZone.getAttribute('tabindex'),null);
assert.equal(await dropZone.evaluate(e=>e.tabIndex),-1);assert.equal(await dropZone.locator('button').count(),1);
assert.equal(await browse.evaluate(e=>e.tagName),'BUTTON');
await page.locator('#helpButton').focus();await page.keyboard.press('Enter');
const dialog=page.locator('#aboutDialog');assert.equal(await dialog.isVisible(),true);
assert.equal(await dialog.getAttribute('aria-labelledby'),'aboutTitle');
assert.match(await dialog.ariaSnapshot(),/A different dimension/);
await page.keyboard.press('Escape');assert.equal(await page.locator('#helpButton').evaluate(e=>document.activeElement===e),true);
const chooserEvent=page.waitForEvent('filechooser');await browse.focus();await page.keyboard.press('Enter');
const chooser=await chooserEvent;await chooser.setFiles(process.argv[6]);
await page.waitForFunction(()=>{const image=document.querySelector('#sourceImage');return image&&!image.hidden&&image.complete&&image.naturalWidth>0;});
await page.waitForFunction(()=>{const button=document.querySelector('#generatePreview');return button&&!button.disabled;});
let previewRoute=null;await page.route('**/api/preview',route=>{previewRoute=route;});
await page.locator('#generatePreview').click();
await page.waitForFunction(()=>document.querySelector('#previewState')?.textContent==='Processing one frame locally…');
assert.equal(await previewStatus.isVisible(),true);assert.equal(await page.locator('#generatePreview').isVisible(),true);
assert.equal(await page.locator('#generatePreview').isDisabled(),true);
assert.equal(await page.locator('#generatePreview').getAttribute('data-preview-busy'),'true');
assert.ok(previewRoute,'Generate preview must issue the intercepted preview request');await previewRoute.abort();
await page.waitForFunction(()=>document.querySelector('#previewState')?.textContent!=='Processing one frame locally…');
await page.unroute('**/api/preview');
await page.locator('#settingsAdvanced > summary').click();
await page.locator('summary').filter({hasText:/^Fine tuning$/}).click();
assert.equal(await page.locator('#useDepthCache').isChecked(),false);
await page.locator('#useDepthCache').check();
await page.locator('#storage').scrollIntoViewIfNeeded();
await page.locator('#refreshDepthCache').click();
await page.waitForFunction(()=>document.querySelectorAll('#depthCacheEntries input').length>=1);
const widths=[];for(const width of [1280,768,390,320]){
await page.setViewportSize({width,height:960});
const scrollWidth=await page.evaluate(()=>document.documentElement.scrollWidth);
widths.push({width,scrollWidth});assert.ok(scrollWidth<=width);}
await page.setViewportSize({width:1280,height:960});await page.locator('#storage').scrollIntoViewIfNeeded();
await page.screenshot({path:process.argv[4]+'/packaged-desktop.png'});
await page.setViewportSize({width:390,height:960});await page.locator('#storage').scrollIntoViewIfNeeded();
await page.screenshot({path:process.argv[4]+'/packaged-mobile.png'});assert.deepEqual(errors,[]);
console.log(JSON.stringify({performed:true,widths,errors,default_cache_off:true,cache_toggle:true,accessible_contracts:{persistent_statuses:true,single_native_import_control:true,dialog_escape_focus_return:true,preview_busy_visible:true}}));
}finally{await browser.close();}})().catch(e=>{console.error(e);process.exitCode=1});''', encoding='utf-8')
        result = subprocess.run([str(self.args.node), str(browser_script), str(self.args.playwright),
                                 f'http://127.0.0.1:{self.port}/', str(self.work), str(self.args.chrome),
                                 str(ROOT / 'static/screenshots/stereolab-studio.png')],
                                capture_output=True, text=True, timeout=120,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        (self.work / 'browser.stdout.log').write_text(result.stdout, encoding='utf-8')
        (self.work / 'browser.stderr.log').write_text(result.stderr, encoding='utf-8')
        require(result.returncode == 0, f'Packaged browser validation failed: {result.stderr}')
        self.report['browser'] = json.loads(result.stdout)
        self.check('actual_packaged_browser')

    def run(self):
        require(os.name == 'nt', 'This verifier launches Windows packaged applications.')
        require(self.args.exe.is_file() and self.args.ffmpeg.is_file(), 'EXE and FFmpeg must exist.')
        require(not self.args.data.exists(), '--data must be a fresh private LOCALAPPDATA directory.')
        self.args.data.mkdir(parents=True)
        target = self.args.data / 'StereoLab' / '.runtime' / 'models' / MODEL
        target.mkdir(parents=True)
        model = self.args.models / MODEL / 'model.onnx'
        receipt = model.with_suffix('.accepted.json')
        require(digest(model) == MODEL_SHA256, 'Source Small model bytes do not match the pinned model.')
        accepted = json.loads(receipt.read_text(encoding='utf-8'))
        require(accepted.get('sha256') == MODEL_SHA256 and accepted.get('revision') == MODEL_REVISION,
                'Source Small model receipt does not match the accepted pinned model.')
        for source in (model, receipt):
            shutil.copy2(source, target / source.name)
        self.report['model_copy'] = {'source': str(model), 'sha256': digest(target / 'model.onnx'),
                                     'receipt': accepted, 'no_download': True}
        self.report['exe'] = {'path': str(self.args.exe), 'bytes': self.args.exe.stat().st_size,
                              'sha256': digest(self.args.exe)}
        self.source = self.work / 'generated-source.mp4'
        ffmpeg_run(self.args.ffmpeg, ['-f', 'lavfi', '-i', 'testsrc2=size=192x108:rate=8:duration=1',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=48000:duration=1',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', '-y', str(self.source)])
        self.report['source'] = media_record(self.args.ffmpeg, self.source)
        self.start()
        require(self.request('GET', '/api/jobs').json() == [], 'Fresh packaged API has unexpected jobs.')
        require(self.request('GET', '/api/depth-cache').json()['entries'] == [], 'Fresh data contains depth cache.')
        static_hashes = {}
        for source in sorted((ROOT / 'static').rglob('*')):
            if source.is_file():
                route = '/static/' + source.relative_to(ROOT / 'static').as_posix()
                downloaded = self.request('GET', route).content
                actual = hashlib.sha256(downloaded).hexdigest()
                require(actual == digest(source), f'Packaged static asset differs from current source: {route}')
                static_hashes[route] = actual
        require(self.request('GET', '/').content == (ROOT / 'static/index.html').read_text(encoding='utf-8').encode('utf-8'),
                'Packaged home page differs from current source.')
        self.check('static_current_source_hashes', static_hashes)
        first, first_media = self.submit('first')
        video = next(stream for stream in first_media['probe']['streams'] if stream['codec_type'] == 'video')
        audio = next(stream for stream in first_media['probe']['streams'] if stream['codec_type'] == 'audio')
        expected_geometry = {'SBS': (384, 108), 'XREAL_SBS': (3840, 1080)}[self.args.layout]
        require((video['width'], video['height'], int(video['nb_read_frames'])) == (*expected_geometry, 8),
                f'Unexpected generated {self.args.layout} geometry or decoded frame count: {video}')
        self.report['layout_validation'] = {
            'layout': self.args.layout, 'expected_width': expected_geometry[0],
            'expected_height': expected_geometry[1], 'actual_width': video['width'],
            'actual_height': video['height'], 'decoded_frames': int(video['nb_read_frames'])}
        require(abs(float(first_media['probe']['format']['duration']) - 1.0) <= .15
                and int(audio['sample_rate']) == 48000, 'Output audio/video timing or sample rate changed.')
        self.check('stereo_geometry_frame_count_audio_timing')
        execution = first['execution']
        require(first['depth_cache']['state'] == 'stored', f'First cache was not stored: {first["depth_cache"]}')
        validate_execution(execution, self.args.execution_mode)
        self.check('named_small_strict_gpu')
        second, second_media = self.submit('same-settings-reupload', expected_source_sha256=digest(self.source))
        require(second['depth_cache']['state'] == 'hit', 'Same-settings reupload missed cache.')
        execution = second['execution']
        validate_execution(execution, self.args.execution_mode, cached=True)
        hardware_variation = (self.args.allow_hardware_encode_variation
                              and first.get('encoder') == second.get('encoder') == 'NVIDIA NVENC')
        comparison = compare_repeat_media(self.args.ffmpeg, first_media, second_media,
                                          allow_hardware_variation=hardware_variation)
        self.report['media_comparison'] = {'first': first_media, 'cached_repeat': second_media}
        self.check('cached_cpu_provenance_honest')
        self.check('cached_repeat_media', comparison)
        self.check('verified_source_reupload')
        _, variant_media = self.submit('second-cache-entry', temporal_smoothing='0.20')
        inventory = self.request('GET', '/api/depth-cache').json()
        require(len(inventory['entries']) == 2 and not inventory['errors'], f'Two real entries missing: {inventory}')
        self.report['inventory_before_restart'] = inventory
        self.browser()
        fingerprint = self.report['launches'][0]['status']['build_fingerprint']
        self.stop()
        status = self.start()
        require(status['build_fingerprint'] == fingerprint, 'Fingerprint changed across same EXE restart.')
        require(self.request('GET', '/api/jobs').json() == [], 'Restart reconstructed old active jobs.')
        records = self.request('GET', '/api/history').json()['items']
        self.report['history_after_restart'] = records
        row = next(r for r in records if r['job_id'] == first['id'])
        require(row['state'] == 'completed' and row['source_sha256'] == digest(self.source)
                and row['requires_source_reselection'] is True, f'History identity or state invalid: {row}')
        saved = self.request('GET', f"/api/history/{first['id']}/download").content
        require(hashlib.sha256(saved).hexdigest() == first_media['encoded_sha256'], 'History output changed.')
        require(self.request('GET', '/api/depth-cache').json() == inventory, 'Cache inventory did not survive restart.')
        replay, replay_media = self.submit('after-restart-reupload', expected_source_sha256=digest(self.source))
        require(replay['depth_cache']['state'] == 'hit', 'Restart reupload did not reuse saved cache.')
        comparison = compare_repeat_media(
            self.args.ffmpeg, first_media, replay_media,
            allow_hardware_variation=(self.args.allow_hardware_encode_variation
                                      and first.get('encoder') == replay.get('encoder') == 'NVIDIA NVENC'))
        self.check('restart_repeat_media', comparison)
        self.check('exact_owned_exe_process_restart')
        self.check('history_output_source_identity_persist')
        self.check('depth_cache_restart_reuse')
        selected = inventory['entries'][0]['key']
        unselected = inventory['entries'][1]['key']
        cleanup = self.request('POST', '/api/depth-cache/cleanup', json={'keys': [selected]}).json()
        require(cleanup['deleted'] == [selected] and not cleanup['errors'], f'Selected cleanup failed: {cleanup}')
        after = self.request('GET', '/api/depth-cache').json()
        require([entry['key'] for entry in after['entries']] == [unselected], 'Cleanup changed unselected cache.')
        self.report['selected_cleanup'] = {'selection': selected, 'preserved': unselected,
                                           'result': cleanup, 'after': after}
        self.check('selected_cache_cleanup_preserves_unselected')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Fresh isolated LOCALAPPDATA root; never reused.')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--models', type=Path, default=ROOT / '.runtime/models')
    parser.add_argument('--ffmpeg', type=Path, required=True)
    parser.add_argument('--expected-fingerprint')
    parser.add_argument('--execution-mode', choices=('gpu', 'cuda'), default='gpu',
                        help='Strict inference backend to verify: gpu requires DirectML; cuda requires CUDA.')
    parser.add_argument('--layout', choices=('SBS', 'XREAL_SBS'), default='SBS',
                        help='Packaged output layout to validate; XREAL_SBS is full 3840x1080 SBS.')
    parser.add_argument('--job-timeout', type=float, default=600)
    parser.add_argument('--clean-path', action='store_true',
                        help='Launch the EXE with a minimal child PATH plus the explicit FFmpeg directory.')
    parser.add_argument('--allow-hardware-encode-variation', action='store_true',
                        help='For NVENC only, allow lossy variation at >=60 dB PSNR per frame; audio/timing remain exact.')
    parser.add_argument('--node', type=Path)
    parser.add_argument('--playwright', type=Path)
    parser.add_argument('--chrome', type=Path, default=Path('C:/Program Files/Google/Chrome/Application/chrome.exe'))
    return parser


def main():
    args = build_parser().parse_args()
    for name in ('exe', 'data', 'report', 'models', 'ffmpeg', 'node', 'playwright', 'chrome'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema_version': 1, 'started_at': utc(), 'passed': False, 'checks': {}, 'launches': [],
              'execution_mode': args.execution_mode,
              'scope': f'Actual frozen EXE; generated one-second audio video; {args.layout} layout; pinned Small; isolated data; HTTP',
              'limitations': ['Synthetic media only; installed external FFmpeg required.',
                              'Restart uses forced exact owned process termination; no graceful shutdown claim.',
                              'No device utilization or per-process VRAM claim.']}
    verifier = Verifier(args, report)
    try:
        verifier.save()
        verifier.run()
        report['passed'] = True
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    finally:
        try:
            verifier.stop()
        except Exception as exc:
            report['passed'] = False
            report['cleanup_failure'] = str(exc)
        report['finished_at'] = utc()
        verifier.save()
    print(json.dumps({'passed': report['passed'], 'report': str(args.report), 'checks': report['checks'],
                      'failure': report.get('failure'), 'cleanup_failure': report.get('cleanup_failure')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
