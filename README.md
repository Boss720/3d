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
a time. GPU input dimensions are capped at 924 pixels on the longer side. Small is
the recommended 4 GB model. Large is routed to CPU on adapters with 4 GB or less,
and explicit GPU mode rejects it. Base/MiDaS may still exceed available memory;
Auto/hybrid can fall back. DirectML has no configured hard VRAM cap, so this is a
best-effort memory policy. Approximation and custom ONNX remain CPU engines.

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
4. Use **Generate preview** to process the current photo/video frame. Inspect Stereo or Depth map. Regenerate after changing settings or seeking. This is actual selected-model inference; temporal smoothing only applies during video export.
5. Convert the queue and download the results. Completed files remain in `outputs/`.

Media stays on the local server. Only model installation contacts Hugging Face. Model weights are stored in `.runtime/models/`. Uploaded conversion sources are removed when processing finishes. The queue survives browser refresh while the server is running; source files must be added again for previews after refresh. Queue state is not retained across server restarts; completed files remain on disk. Clearing the browser's finished list does not delete exports.

## Depth engines

| Engine | Download | License | Runtime |
|---|---:|---|---|
| Depth Anything V2 Small | 99 MB | Apache-2.0 | CPU/DirectML GPU; recommended for 4 GB cards |
| Depth Anything V2 Base | 389 MB | CC-BY-NC-4.0 | CPU/DirectML GPU; non-commercial weights |
| Depth Anything V2 Large | 1.34 GB | CC-BY-NC-4.0 | CPU on 4 GB cards; high memory use |
| DPT Hybrid / MiDaS | 533 MB | Apache-2.0, Intel upstream | CPU/DirectML GPU; 384 px inference |
| Custom ONNX | Your file | Your model's license | OpenCV DNN CPU |
| Visual approximation | None | App code | Image cues; **not neural AI depth** |

Downloads honor configured HTTPS proxies. If a loopback proxy refuses the connection, StereoLab retries the pinned request directly over HTTPS; any installed file still must pass its pinned size, SHA-256, graph-load, and inference checks. Rejected partial files are retained under `.runtime/models/_quarantine/` for inspection. Only one neural session is kept resident, and inference is serialized to bound memory use.

Named models use their own preprocessing. Depth Anything uses RGB/ImageNet normalization with aspect-preserving, multiple-of-14 sizing (nominal 518); MiDaS uses its 384 px/0.5 normalization. Both produce relative inverse depth: brighter means nearer. Custom models must accept a float32 RGB NCHW batch with square 256/384/518 input, ImageNet normalization, and return one depth map; not every ONNX graph matches this convention.

Model provenance: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Small ONNX export](https://huggingface.co/onnx-community/depth-anything-v2-small), [MiDaS ONNX export](https://huggingface.co/Xenova/dpt-hybrid-midas), [MiDaS upstream](https://huggingface.co/Intel/dpt-hybrid-midas). The ONNX conversions are published by ONNX Community/Xenova, rather than the original model authors.

## Formats and controls

Full/half SBS, full/half top-bottom, cross-eye, red/cyan anaglyph, depth-only, and color-plus-depth side-by-side are available. Videos export H.264/AAC MP4; photos export PNG. Depth exports are normalized 8-bit grayscale, not metric depth or a lossless depth archive. Resolution limits apply per eye; full SBS doubles the encoded width. Video exports preserve decoded frame count and nominal frame rate, and include original audio where available. Variable-frame-rate timing is converted to a constant rate; HDR, subtitle tracks and spatial metadata are not preserved.

Fine tuning offers inversion, luminance-guided edge refinement, and motion-gated temporal smoothing with scene-cut resets. These are practical filters, not a learned video-depth model. Inverse warping can stretch occlusion edges; it cannot reconstruct hidden surfaces. Similar-color motion can leave smoothing lag. Browser previews are limited to 960x540 source frames and JPEG capture, so full-resolution exports may differ in detail. Containers/codecs unsupported by the browser can still be decoded for export by FFmpeg.

## FFmpeg

Video exports need FFmpeg with `libx264`. Detection checks `VIDEO3D_FFMPEG`, then PATH, then an already-installed Owl3D FFmpeg utility as a local convenience. The utility is not copied or redistributed. To use your own:

```powershell
$env:VIDEO3D_FFMPEG = 'C:\Tools\ffmpeg\bin\ffmpeg.exe'
.\run.ps1
```

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
