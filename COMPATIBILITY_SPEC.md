# StereoLab Compatibility Spec (v1)

> Historical design record. Current source and README runtime contracts take
> precedence over restrictions below. Large may attempt DirectML on 4 GB or unknown
> memory with a 518 px cap; unknown memory is conservative. NVENC is used after a
> local capability encode succeeds, with libx264 fallback. See ROADMAP.md and
> PROGRESS.md for the current backlog and dated verification evidence.

## GPU addendum (2026-10-18; expands the v2 CPU-only boundary)

Named models support CPU and Windows DirectML on the GTX 1650 Ti. Auto/hybrid fall
back to CPU with an explanation; explicit GPU fails clearly. Use DXGI ordinals,
not nvidia-smi indices. Provider registration and successful GPU inference are
separate signals; unsupported ONNX operators can remain on CPU. Per-result device
snapshots are captured atomically with inference.

For videos Auto/hybrid overlap bounded CPU decoding and CPU rendering/encoding
with one GPU inference stage. Preserve ordering, temporal history, frame count,
audio, cancellation and error propagation. No shared session is run concurrently.
Small is the 4 GB baseline; bound GPU input size, keep Large on CPU for <=4 GB,
and report failures without claiming a hard DirectML VRAM ceiling.

Verify real GTX inference using profiling, CPU numerical comparison, 100-frame
sampled memory evidence, actual CPU/GPU/hybrid exports and worker cleanup tests.
NVENC, CUDA-specific kernels, TensorRT/RTX, and learned depth fusion are not added.

## v2 addendum (2026-10-18, supersedes the v1 boundary below)

The user's improvement request expands v1 to named neural models and an Owl3D-like
conversion workflow. The v1 investigation is retained below as historical evidence.

- Add CPU ONNX Runtime inference for pinned Depth Anything V2 Small/Base/Large and
  DPT Hybrid MiDaS exports. Publish export/upstream URLs, licenses, download sizes,
  readiness, progress and actionable failures. Downloads require internet;
  conversion does not transmit media. Only verified downloads are accepted.
- Add explicit single-frame preview using the selected engine and the same renderer
  as exports. Browser JPEG capture and preview downscaling may change fine details;
  the UI must not represent it as a completed video or temporally smoothed preview.
- Strength is monotonic 0..10, with neutral eyes at zero. Convergence, depth inversion,
  luminance-guided refinement and scene/motion-gated temporal smoothing are functional.
- Add DEPTH and RGBD layouts (normalized near-white 8-bit depth). Keep all v1 layouts,
  photo PNG and H.264/AAC video with audio when present.
- Add readable responsive UI, presets, model library, selected-file previews,
  browser-session queue restoration and download/cancel/error handling.
- Do not claim learned video consistency, hidden-surface reconstruction, GPU
  acceleration, metric depth, spatial/MV-HEVC, HDR/VFR preservation, VR pairing,
  live display conversion, or Owl3D depth-quality parity.

Acceptance: independently pass semantic model/render/endpoint tests and syntax
checks; exercise a real Small model install/inference and photo/video conversion;
verify preview/export agreement for identical input and video audio/dimensions;
inspect UI at desktop and narrow breakpoints and confirm model-readiness gating.
Other downloadable model families may remain explicitly unverified on hardware.

## Goal
Build a clean-room, local Windows app that converts ordinary video and photo files into standard stereoscopic outputs. Compatibility means the generated media can be opened by ordinary SBS/TB-capable players and headsets; it does not mean reusing Owl3D internals, model files, account state, or its proprietary job/cache protocol.

## Evidence from Owl3D Desktop 2.3.0
- Shortcut resolves to `C:\Users\Hp\AppData\Local\owl3d-desktop-app\Owl3D.exe`; app version is 2.3.0 and its signature is valid.
- The installed Electron renderer/preload starts a separate `process_desktop_main.exe`, writes one JSON payload to stdin, and consumes line-oriented progress/output/error messages. The main app downloads the processor, PyTorch, and model payloads. This establishes local processing with downloaded model/runtime components; the downloaded processor itself was not present in the install directory inspected.
- Renderer exposes video/photo input, depth estimation settings, parallax/convergence controls, queue handling, and exports including SBS, half-SBS, TB, half-TB, cross-eye, anaglyph, RGB-D, depth-only, YouTube, and Spatial.
- The installed UI includes model modes and advanced features beyond this v1. Original media chooser filters include common video formats and JPEG/PNG/WebP/TIFF/JP2/EXR.
- Official product material describes local AI conversion and common stereo outputs. It is current product context, not proof that every current feature exists in version 2.3.0.

## v1 user workflow
1. Launch a local web UI bound only to `127.0.0.1`.
2. Select or drop one or more supported video/photo files; process jobs sequentially and show per-job progress, success, failure, and cancel.
3. Choose full SBS, half-SBS, top-bottom, half-top-bottom, cross-eye, or anaglyph.
4. Adjust 3D strength and convergence and choose Balanced or Precision depth.
5. Export H.264/AAC MP4 for video and PNG for stills; keep video audio and use the original frame rate/duration.
6. Provide a download link and keep completed outputs in the project's `outputs/` folder.

## Input support
- Video: MP4, WebM, MKV, MOV, AVI, WMV, FLV, TS, MTS, M2TS, and M4V, subject to the available FFmpeg decoder.
- Still images: JPG/JPEG, PNG, and WebP.

## Depth behavior and honesty
- Balanced runs an offline visual-cue depth approximation so the app works without a model download.
- Precision may use a user-supplied, compatible ONNX monocular-depth model through OpenCV DNN. No proprietary Owl3D model or weights are bundled or copied.
- The UI must identify when the approximation is active and must not label it AI depth. The optional ONNX path must explain that supported tensor shape/preprocessing are model-dependent.
- This is a functional compatibility prototype, not depth-quality parity with Owl3D's downloaded inference runtime.

## Runtime and privacy
- Python/FastAPI local server, browser UI, OpenCV/NumPy image processing, and FFmpeg/FFprobe for video/audio.
- No conversion API, account, telemetry, or external network calls.
- FFmpeg discovery: explicit `VIDEO3D_FFMPEG` path, PATH, then the already-installed Owl3D FFmpeg as a local convenience fallback. The fallback is not copied or redistributed; the README will explain how to supply a standalone FFmpeg path.
- Uploads are temporary local files. Completed outputs persist in `outputs/`.

## Out of scope for v1
Shift/live desktop streaming, VR device pairing, display topology, model/photo viewers, subtitle conversion, custom depth-map import, RGB-D/Depth-only exports, Spatial/MV-HEVC, HDR, ProRes, GPU encoding, advanced Fusion/DepthGen/StereoGen models, paid-account/activation behavior, and Owl3D queue/cache compatibility.

## Acceptance criteria
- A user can launch the app locally, import a supported file, see the approximation/model mode and settings, and start/cancel a conversion.
- Each listed v1 stereo layout is assembled from a left/right pair; stills export as PNG and video outputs contain the selected layout plus audio where present.
- The app surfaces missing FFmpeg/model/input failures clearly and makes no network requests for conversion.
- No Owl3D JavaScript, binary processor, or model weights are copied into this project.
