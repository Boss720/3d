# StereoLab

A local Windows workspace for converting photos and videos into stereoscopic 3D. This version adds downloadable neural depth models, real processed-frame previews, depth presets, temporal smoothing, and depth/RGB-D exports to the original clean-room prototype.

## Run

Python 3.10+ is required. From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\run.ps1
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The launcher prefers this project's `.venv-gpu` when installed, then `.venv`. If an older server is already using that port, stop that server or use another port:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8766
```

## Windows executable

Build the standalone, 64-bit Windows executable with the DirectML runtime:

```powershell
.\setup-gpu.ps1
.\build-exe.ps1
.\dist\StereoLab.exe
```

The executable runs the local web app; open [http://127.0.0.1:8765](http://127.0.0.1:8765).
Depth models download on demand and are stored under `%LOCALAPPDATA%\StereoLab`;
they are not bundled into the executable. Video conversion also needs FFmpeg. The
app detects FFmpeg on `PATH`, `VIDEO3D_FFMPEG`, or an already installed local helper.
The frozen app stores uploads and outputs under the same StereoLab local-data folder.
One-file extraction can make the first launch slower, and Windows may show a warning
for an unsigned PyInstaller executable.

## NVIDIA GTX GPU and CPU cooperation

For Windows GPU acceleration, run `.\setup-gpu.ps1` once, then restart with `.\run.ps1`.
The setup creates a separate `.venv-gpu` environment containing ONNX Runtime DirectML;
it preserves the CPU environment and model cache. Do not install `onnxruntime`,
`onnxruntime-gpu` and `onnxruntime-directml` into the same environment because they
share a Python module. GPU setup uses `requirements-gpu.txt`, rather than the CPU file.
To use the GPU environment on the currently open port explicitly:

```powershell
.\.venv-gpu\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8766
```

The GTX 1650 Ti works through DirectML/DirectX 12; RTX hardware, TensorRT and a CUDA
toolkit are not required. DXGI enumeration selects the actual NVIDIA adapter: this
machine has AMD graphics at index 0 and the GTX at index 1. ONNX Runtime's registered
provider is a capability signal; the UI reports verified inference separately.

Processing device choices:

- **Auto** uses GPU depth if usable and overlaps the CPU video stages. If the GPU
  cannot run the model, it uses CPU depth and reports the reason.
- **GPU + CPU / hybrid** explicitly requests the same cooperative video pipeline:
  CPU decode/resize, GPU inference, then CPU refinement, temporal filtering, stereo
  rendering and FFmpeg encoding. Two bounded queues preserve frame order.
- **NVIDIA GPU** requires a usable GPU for depth and reports failures. The remaining
  processing is CPU, sequential around inference. Unsupported ONNX operators may
  still run on CPU within the GPU session.
- **CPU only** uses CPU inference and sequential media processing.

Only one model/session is resident in each server, with a single inference call at
a time. All built-in named models can attempt DirectML, including Large on a 4 GB
card. GPU input dimensions are normally capped at 924 pixels on the longer side;
Large is capped at 518 pixels on adapters with 4 GB or less to reduce its activation
memory. Start Large at 256 px neural detail and close other GPU-heavy apps. VRAM use
remains best effort because DirectML has no hard memory cap. Auto/Hybrid automatically
retry on CPU if a GPU session or inference fails; explicit NVIDIA GPU mode reports
the failure. Small remains the recommended 4 GB model for speed. Approximation and
custom ONNX remain CPU engines.

The pipeline overlaps different stages, rather than running duplicate CPU/GPU
depth predictions. Decoder/inference callbacks finish their current operation on
cancellation; encoder termination releases pipe writes, and pipeline workers are
joined before the job ends. Temporal smoothing remains ordered on the CPU.

Video exports automatically use NVIDIA NVENC when the configured FFmpeg completes
a small local encode check for the chosen quality profile. That result is cached for
the current FFmpeg binary; unavailable or failed NVENC falls back to libx264, and a
later NVENC export failure retries the video with the CPU encoder. The queue shows
which encoder actually produced each export. `Fast` favors shorter encoding time;
`High quality` takes longer and can produce larger output.

The strongest speed controls are the depth model, neural input detail, source
resolution, and video length. Start with Depth Anything V2 Small; lower input detail
often reduces inference work, while 518 can preserve more fine depth detail. The
fastest size depends on source shape and runtime: on this GTX 1650 Ti, 15 warmed
Small-model runs on a 240x400 source measured GPU medians of 0.133/0.181/0.263 s at
256/384/518 px, while CPU medians were 0.277/0.205/0.401 s. Treat these as one
workload, not a universal ranking. `Up to 1080p` caps large sources; `Match source`
keeps their full dimensions and takes more work. More video frames mean proportionally
more processing. The encoder preset also matters, while 3D strength, convergence and
depth inversion have negligible cost. Edge refinement and temporal smoothing add
smaller CPU work per frame. Custom ONNX models may have a fixed input shape, in which
case their model's required size takes precedence.

Encoder-only medians on this GTX with a generated 90-frame 1080p clip were 0.952 s
for the previous libx264 `medium` preset, 0.470 s for the new CPU `veryfast` preset,
and 0.690 s for NVENC `p4`. NVENC offloads encoding and beats the old preset in this
test, but the faster libx264 preset won; total depth conversion time also includes
inference and stereo rendering.

For videos, the queue begins estimating after it has observed at least five frames
over two seconds, then reports remaining time, measured FPS and elapsed time from the
actual job. The estimate adapts as the conversion runs; short clips, still images and
videos without a reliable frame count do not get a fabricated remaining time.

On this GTX 1650 Ti, a 100-frame Small-model test measured warmed median inference
of .214 s on GPU versus .414 s on CPU, for one 400x240 synthetic source (518x308
model input). Profiling contained DirectML and CPU kernel events. The highest
sampled whole-device memory reading was 302 MiB; this is not a universal peak or
memory guarantee. Cold graph compilation, model loading, larger inputs, video
processing and other GPU applications can change timing and memory use.

## Convert

1. Install a model from **Depth models**, or choose **Visual approximation** to work without a download.
2. Add photos or videos; select a file in the queue to preview it.
3. Choose a preset, output format, strength and convergence. Strength 0 gives identical eye views. Convergence selects the zero-parallax depth plane.
4. Use **Generate preview** to process the current photo/video frame. Inspect Stereo or Depth map. Strength, convergence, layout, inversion and refinement edits reuse valid inferred depth; changing the frame, engine, input detail or execution settings invalidates it. Temporal smoothing only applies during video export.
5. Convert the queue and download the results. Completed files remain in `outputs/`.

Media stays on the local server. Only model installation contacts Hugging Face. Model weights are stored in `.runtime/models/`. Uploaded conversion sources are removed when processing finishes. The queue survives browser refresh while the server is running; source files must be added again for previews after refresh. Queue state is not retained across server restarts; completed files remain on disk. Clearing the browser's finished list does not delete exports.

## Depth engines

| Engine | Download | License | Runtime |
|---|---:|---|---|
| Depth Anything V2 Small | 99 MB | Apache-2.0 | CPU/DirectML GPU; recommended for 4 GB cards |
| Depth Anything V2 Base | 389 MB | CC-BY-NC-4.0 | CPU/DirectML GPU; non-commercial weights |
| Depth Anything V2 Large | 1.34 GB | CC-BY-NC-4.0 | DirectML attempted at 4 GB with 518 px cap; Auto/Hybrid can fall back to CPU |
| DPT Hybrid / MiDaS | 533 MB | Apache-2.0, Intel upstream | CPU/DirectML GPU; 384 px inference |
| Custom ONNX | Your file | Your model's license | OpenCV DNN CPU |
| Visual approximation | None | App code | Image cues; **not neural AI depth** |

Downloads honor configured HTTPS proxies. If a loopback proxy refuses the connection, StereoLab retries the pinned request directly over HTTPS; any installed file still must pass its pinned size, SHA-256, graph-load, and inference checks. Rejected partial files are retained under `.runtime/models/_quarantine/` for inspection. Only one neural session is kept resident, and inference is serialized to bound memory use.

Named models use their own preprocessing. Depth Anything uses RGB/ImageNet normalization with aspect-preserving, multiple-of-14 sizing (nominal 518); MiDaS uses its 384 px/0.5 normalization. Both produce relative inverse depth: brighter means nearer. Custom models must accept a float32 RGB NCHW batch with square 256/384/518 input, ImageNet normalization, and return one depth map; not every ONNX graph matches this convention.

Model provenance: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Small ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-small), [MiDaS ONNX export](https://huggingface.co/Xenova/dpt-hybrid-midas), [MiDaS upstream](https://huggingface.co/Intel/dpt-hybrid-midas). The ONNX conversions are published by ONNX Community/Xenova, rather than the original model authors.

## Formats and controls

Full/half SBS, full/half top-bottom, cross-eye, red/cyan anaglyph, depth-only, and color-plus-depth side-by-side are available. Videos export H.264/AAC MP4; photos export PNG. Depth exports are normalized 8-bit grayscale, not metric depth or a lossless depth archive. Resolution limits apply per eye; full SBS doubles the encoded width. Video exports preserve decoded frame count and nominal frame rate, and include original audio where available. Variable-frame-rate timing is converted to a constant rate; HDR, subtitle tracks and spatial metadata are not preserved.

Fine tuning offers inversion, luminance-guided edge refinement, and motion-gated temporal smoothing with scene-cut resets. These are practical filters, not a learned video-depth model. Inverse warping can stretch occlusion edges; it cannot reconstruct hidden surfaces. Similar-color motion can leave smoothing lag. Default browser previews use a 960x540 JPEG proxy. Full-detail stills preserve the decoded source within the limits below; neural input remains independently capped. Containers/codecs unsupported by the browser can still be decoded for export by FFmpeg.

## FFmpeg

Video exports need FFmpeg with `libx264`. Detection checks `VIDEO3D_FFMPEG`, then PATH, then an already-installed Owl3D FFmpeg utility as a local convenience. The utility is not copied or redistributed. To use your own:

```powershell
$env:VIDEO3D_FFMPEG = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
.\run.ps1
```

## Roadmap runtime improvements

GPU capabilities are versioned in `runtime.capabilities`; the frontend does not
infer support from prose. Unknown adapter memory follows the conservative Large
518 px cap. After a failed GPU attempt, Auto/Hybrid suppress repeated attempts for
the same model, adapter and input settings for 60 seconds. **Retry GPU** clears that
suppression and the preview cache after any active inference finishes. GPU-only
mode continues to report failures strictly. Actual model dimensions, input cap,
provider and successful inference are reported separately.

Preview depth uses a process-local LRU with at most eight entries and 32 MiB of
depth buffers; metadata and temporary images use additional memory. It stores raw
depth before inversion/refinement. Model identity, preprocessing, frame pixels,
input settings and runtime identity participate in cache keys. Transient neural
CPU fallback results are not cached. Cached tuning remains available during an
export; new neural preview requests report busy while a neural export is active.

Conversion preflight checks readable media/first video frame, output dimensions,
writable destination, disk reserve and video libx264 availability before loading
the depth model. Disk estimates and warnings appear in the queue and do not
guarantee that an export will fit. Later corrupt frames or encoder failures can
still fail a conversion. Existing active jobs are not restarted by diagnostics.

Video telemetry separates initialization, conversion and finalization, records
encoder restart reasons and bases ETA on up to 32 recent frame observations.
Observed variability can produce an approximate range; stalls affect the estimate.
Finalization has no invented remaining time. This is an active-job estimate, not
a queue completion estimate. Stage service and queue-wait times can overlap and
must not be added to obtain elapsed wall time. Queue waits include operation overhead.

**Performance presets** provide Fast, Balanced, Quality and Conservative 4 GB
starting settings, with the exact values visible. Reset restores the selected
preset; edits show an unsaved indicator. Export/import settings uses schema v1 JSON
and rejects unknown fields, obsolete schemas and invalid values before applying.
The 4 GB preset is a conservative starting point, not a memory guarantee.

**Measure input detail** is opt-in for named Depth Anything models. Choose a 5,
15 or 30 second budget. It warms each input size, collects two comparable samples,
and reports the fastest measured depth input only when evidence is complete.
Decode, setup and warmup count against the budget; a running inference cannot be
interrupted and may overrun it. Results never change settings automatically.
**Apply measured input detail** requires an explicit click and unchanged context.
Lower detail can lose fine edges. This does not calibrate models, encoders or whole
exports, and MiDaS has fixed input dimensions. Up to 16 reports are saved atomically
in `.runtime/calibrations.json`; no old measurement is applied automatically.

Preview zoom/pan is shared across views (1Ã—â€“4Ã—). Focus the preview and use +/âˆ’,
arrow keys and 0 to reset, or drag a zoomed view. **Source / Depth** reveals two
images from the exact processing frame sent to inference and labels its dimensions.
It never compares a processed result against a newly sought live video frame.
**Still detail → Full-detail still** sends the original image file instead of a
JPEG capture: at most 12,000,000 decoded pixels, 8192 pixels per side and 24 MiB.
The full source is retained for rendering and Source/Depth comparison; this does
not raise the model's hardware input cap. Dimension limits are checked after
decode, so they are not a guarantee of peak process memory. Large raw-depth
arrays can exceed the cache budget and are then deliberately not retained.

**Processed clip preview** explicitly uploads the original selected video and
prepares a 1–5 second interval from its current timestamp. Preparation uses a
separate H.264/AAC proxy of at most 960x540, 24 fps and 120 frames, followed by
the normal export pipeline and selected refinement/temporal settings. History
starts fresh at the interval boundary. Proxy re-encoding changes source pixels;
the result does not promise pixel identity with a full-source movie export.
Uploads and preparation can dominate the time for large files. Clips queue
behind existing work, can be canceled, and stale source/settings/timestamp/length
results are hidden and canceled where still active. The output player is muted
and does not start automatically. Codec support is needed in the browser to
select a timestamp. Clip jobs use a separate preview area, and the API explicitly
advertises support so an older server cannot silently export the whole movie.
Very slow sources use at least 1 fps in the proxy. Sparse source frames, seeking
or the end of a file can yield a shorter playable clip; actual frames/FPS are
reported separately from the requested source interval.

Queue rows show their exact settings snapshot, actual device/provider, encoder,
phase and fallback. Before upload, files use the visible controls unless you
explicitly freeze their settings; **Apply current settings to ready files** freezes
the batch. Move ready files up/down to choose submission order. Supported servers
also allow pending exports to move, receive current settings, or be removed.
An export already taken by the worker rejects queued edits, and queued preview
clips keep their place when exports move. A changed queue prompts a refresh.

**Use settings** loads a snapshot into the controls without starting work.
Retry and duplicate prepare a new ready item only when the original File remains
in this browser tab. After refresh, reselect the original source; server jobs do
not retain completed uploads or match files automatically by name. Active jobs
are unaffected by changes to the visible controls. Queue state is still in memory
and does not survive a server restart. New outputs use the full job UUID and an
exclusive reservation; a preexisting target is preserved and reported as a conflict.
Existing exports keep their names.

The server accepts loopback hosts and rejects browser writes from foreign origins
or a different service port. Local CLI calls without an Origin remain supported.
This is a local app boundary, not authentication for a public deployment.

DEPTH/RGB-D no longer synthesize unused stereo eyes. An alternating four-sample
1080p approximation-to-PNG benchmark measured DEPTH medians 0.321 â†’ 0.277 seconds
and RGB-D 0.335 â†’ 0.313 seconds, with identical PNG bytes. This measures the image
processing path, excluding upload, queue, preflight and download; it is not a
universal speedup. Reproduce with `scripts/benchmark_render_outputs.py`.

The reproducible API benchmark uses synthetic fixtures (not visual-quality ground
truth), validates output and reports comparable-run medians/ranges:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/benchmark_roadmap.py --base-url http://127.0.0.1:8768 --model balanced --device cpu --repeats 3
```

Reports and generated fixtures are kept under ignored `.runtime/` directories.
Use `--sample-resources` to opt in to server process CPU/working-set snapshots and
bounded whole-device NVIDIA memory sampling. Process working-set peaks cover the
server lifetime; sampled device memory includes other applications. Neither is an
exact per-job VRAM allocation peak, and diagnostics add measurement overhead.
Named models must already be installed. Use `--model depth-anything-v2-small
--device gpu` or `--device hybrid` for matched hardware runs. Missing measurements
remain explicit; short synthetic clips do not establish sustained speed or peak VRAM.

`scripts/verify_roadmap.py --base-url http://127.0.0.1:8770 --device gpu` checks
cache reuse/invalidation and opt-in calibration against an idle running server.
`scripts/compare_renderer_quality.py` and `scripts/compare_temporal_quality.py`
retain unpromoted quality experiments under `.runtime/quality-verification/`.
Measured costs and mixed synthetic evidence do not support selecting them by default.

`scripts/stress_roadmap.py --base-url http://127.0.0.1:8770 --sample-resources`
generates a deterministic 30-second pan/thin-detail/cut/low-light/audio fixture,
then runs all four installed models sequentially on explicit GPU at requested
detail 518. MiDaS remains fixed at 384x384. The local GTX run completed all four
360-frame exports and separate cancellations; reports are incremental under
`.runtime/stress-verification/`. This generated workload does not establish
representative real-footage quality, multi-hour stability or an exact VRAM peak.
The runner requires an idle server; API cancellation/download checks alone do
not establish OS encoder-process cleanup.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pip install httpx
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
node --check static/app.js
```

The semantic tests use fake model sessions/downloads plus real approximation image processing. For a live integration check, install Small, run the server on 8766, then run `.\.venv\Scripts\python.exe scripts/smoke.py`. That check performs real Small inference, matches preview/export pixels for an identical frame, verifies image/depth/RGB-D outputs, and checks video dimensions, frames, frame rate, audio and cancellation. Synthetic fixtures and a report are retained in `.runtime/verification/`.

GPU checks use `.\.venv-gpu\Scripts\python.exe scripts/benchmark_gpu.py` (CPU/GPU
comparison, 100 GPU inferences, provider profiling and sampled nvidia-smi memory).
After the smoke check creates a source video, `scripts/verify_devices.py` exercises
CPU/GPU/hybrid exports and compares decoded frames. Install `httpx` into the chosen
verification environment if needed. Reports remain in `.runtime/gpu-verification/`.

StereoLab is independent of Owl3D. It does not reuse proprietary Owl3D weights, processor code or account protocols, and does not claim matching output quality. Spatial/MV-HEVC, live desktop conversion, VR pairing and learned video-depth models remain outside this version. See `COMPATIBILITY_SPEC.md` and `PROGRESS.md` for scope and evidence.
