# Owl3D Compatibility Reverse Engineering

## GPU integration wave (2026-10-18)

User explicitly requests GTX 1650 Ti (not RTX) GPU support plus CPU/GPU cooperation.
Live nvidia-smi: GeForce GTX 1650 Ti, 4096 MiB, driver 610.47. Existing ONNX models
and cache are retained. DirectML is the first Windows GPU backend; CPU remains usable.

Frozen scope: per-request auto/cpu/gpu/hybrid execution; correct NVIDIA DXGI adapter
selection; actual session-provider/fallback reporting; conservative Small-first
4 GB policy; optional GPU runtime environment; bounded ordered pipeline overlapping
CPU decode/render/encode with GPU inference in hybrid mode. No RTX/TensorRT dependency.
Do not claim speedups or GPU execution until real inference/profiling confirms them.

| Lane | Owner | Exclusive files | Acceptance |
|---|---|---|---|
| GPU depth | runtime worker | depth_engine.py, gpu_hardware.py, tests/test_depth_engine.py, tests/test_gpu_hardware.py | DirectML adapter selection, CPU/auto/gpu/hybrid policy, honest fallback, VRAM guards |
| Ordered pipeline | pipeline worker | video_pipeline.py, tests/test_video_pipeline.py | Bounded 3-stage ordered CPU/GPU/CPU overlap, cancellation and error propagation without thread leaks |
| GPU controls | UI worker | static/index.html, static/app.js, static/style.css | Execution selector, GTX/provider/fallback status, request snapshots |
| Integration/setup | coordinator | main.py, requirements-gpu.txt, setup-gpu.ps1, run.ps1, README.md, COMPATIBILITY_SPEC.md, PROGRESS.md, scripts, tests/test_app.py | Wire policy/pipeline, install runtime, verify real CPU/GPU/hybrid and memory |

Validation: independent suite/syntax/source review, real GPU provider profiling,
CPU comparison and same-frame quality check; video audio/frame-count/order/cancel;
sample nvidia-smi memory over a 100-frame run. Keep prior untracked work, no commits.

### GPU wave outcome and independent evidence

- All lanes completed within their allowlists. Source reviewed and combined integration
  independently checked. Existing untracked files retained; no commits requested/made.
- Added `.venv-gpu` with ONNX Runtime DirectML 1.24.4 and application dependencies;
  CPU `.venv` remains usable. `setup-gpu.ps1`, requirements-gpu.txt and launcher
  preference support repeatable setup without mixing ORT packages in one environment.
- Real DXGI hardware enumeration selected GTX 1650 Ti at adapter **1**, while AMD
  integrated graphics is adapter 0. nvidia-smi reports 4096 MiB; DXGI reports 3935 MiB
  dedicated memory. Software adapters are excluded. No RTX/CUDA toolkit is needed.
- Both environments independently passed **61 semantic tests**: GPU environment
  1.581s, CPU environment 1.516s. Tests include provider selection/failure/fallback,
  VRAM/model policy, atomic per-request execution metadata, endpoint controls,
  bounded pipeline overlap/order/errors/cancellation and complete worker joins.
  Python compilation, JS syntax, PowerShell parser checks and git diff --check passed.
  Inherited TestClient httpx deprecation warning is informational; no tests skipped.
- `.venv-gpu/Scripts/python.exe scripts/benchmark_gpu.py` ran real CPU and 100 GPU
  Small inferences. Warmed median CPU .413982s vs GPU .213996s for a 400x240 synthetic
  source / 518x308 model input. GPU graph creation/warmup excluded from median.
  Mean absolute normalized CPU/GPU depth difference 0.0000013843.
- Profiling recorded **42,521 DirectML** and **13,029 CPU** provider kernel events.
  This establishes actual GPU-assisted execution with CPU graph work, not all-GPU
  inference. Highest sampled whole-device memory reading was **302 MiB** across
  this workload; no hard process memory/VRAM ceiling is claimed.
- Live GPU-server smoke passed actual preview/export equality, image/depth/RGBD,
  video dimensions/8 frames/8 fps/audio/duration and cancellation. Explicit CPU,
  GPU and hybrid video exports each succeeded. GPU and hybrid decoded frames were
  exactly identical; CPU-vs-GPU mean absolute decoded-channel difference .03237/255.
  Hybrid reported `pipeline=overlapped`; GPU/CPU sequential modes reported sequential.
  Their short-job times include different cache/graph warmup and are not a fair
  hybrid-vs-GPU performance benchmark. Reports: `.runtime/gpu-verification/benchmark.json`,
  `devices.json`, provider profiles, and `.runtime/verification/report.json`.
- A real CPU-only environment Auto inference returned CPU with DirectML-unavailable
  reason. Its explicit GPU request rejected clearly. No heuristic model substitution.
- Restarted the verified idle old CPU server on **8766** into the GPU environment,
  preserving outputs and the older 8765 service. Temporary GPU verification server
  on 8767 was stopped after completion to release its session/resources.
- Browser checked the actual updated controls, selected GPU + CPU, imported a synthetic
  photo and generated a neural preview. UI displayed GTX 1650 Ti inference verified,
  DmlExecutionProvider + CPUExecutionProvider, and per-preview GPU depth + CPU processing.
  Browser warning/error logs were empty. App remains open and running on 8766.
- Limits: Small alone has real GPU proof here; Base/MiDaS remain hardware-unverified,
  Large is CPU on <=4GB. FP32 ONNX weights retained. Memory policy is best effort;
  CPU graph nodes remain possible, and callbacks complete current operations on cancel.
  NVENC/CUDA/TensorRT/RTX and learned CPU/GPU fusion were not part of this wave.

## v2 improvement wave (2026-10-18)

User request: bring the existing converter closer to Owl3D and include named depth models, using army.
Existing files are all untracked on master; preserve and improve them without committing.
No applicable AGENTS.md was found in the workspace or its ancestors.

Frozen scope: downloadable local ONNX Depth Anything V2 Small/Base/Large and MiDaS model support;
model provenance, license, download state and errors; real selected-model frame preview;
monotonic strength, convergence, depth inversion, edge-aware refinement and scene-aware temporal
smoothing; depth-only and RGB-D layouts; refreshed conversion workspace, presets and queue recovery.
CPU inference is the baseline. GPU acceleration, spatial/MV-HEVC, learned video consistency and
Owl3D quality parity are outside this wave. Model setup contacts Hugging Face; media stays local.

| Lane | Owner | Exclusive files | Acceptance |
|---|---|---|---|
| Model runtime | model worker | depth_engine.py, requirements.txt, tests/test_depth_engine.py | Real ONNX catalog/install/inference with model-specific preprocessing and atomic installs |
| Conversion UI | UI worker | static/index.html, static/style.css, static/app.js | Models, actual processed preview, presets, depth controls, usable responsive workspace |
| Stereo processing | rendering worker | stereo.py, tests/test_stereo.py | Monotonic strength, layouts, refinement and smoothing with scene resets |
| Integration | coordinator | main.py, README.md, COMPATIBILITY_SPEC.md, PROGRESS.md, verification scripts | Wire contracts, run independent checks, exercise image/video/model paths |

Verification: inspect every changed file; run semantic unit tests and Python/JS syntax checks;
exercise HTTP endpoints, image/video export and cancellation; download and run Small if network
permits; inspect browser UI and document exact hardware/model limitations. Worker evidence alone
does not establish success. No commits requested.

### v2 outcome and independent evidence

- All three lanes completed within their exclusive file ownership. Coordinator integrated
  `main.py`, launcher, docs and endpoint/live verification; no commits or unrelated deletions.
- Model runtime: four ONNX source revisions, file sizes and SHA-256 values are pinned.
  ONNX Runtime 1.31.0 was installed in project `.venv`; the launcher now prefers it.
  Small was downloaded (99,060,839 bytes), checksum verified and actually inferred.
  The install plus validation completed in about 31 seconds; a separate 320x240 random
  frame returned finite 0..1 depth with standard deviation .2403 in 1.06 seconds on CPU.
- ` .venv/Scripts/python.exe -m unittest discover -s tests -v`: **34 tests passed**
  in .891 seconds. Tests cover corrupt/interrupted downloads and cache/session behavior,
  model-specific preprocessing, depth polarity, monotonic strength, all layout sizes,
  guided refinement, temporal motion/cut behavior, endpoint validation, all-layout
  preview/export pixel identity, and cancellation during inference.
- `node --check static/app.js`, Python compilation of main/depth_engine/stereo, and
  `git diff --check` passed. The inherited Starlette TestClient emitted an httpx
  deprecation warning; there were no test failures.
- ` .venv/Scripts/python.exe scripts/smoke.py` against the running server succeeded:
  real Small neural preview exactly matched the exported identical PNG source; stereo
  eyes differed; depth had measurable variation; DEPTH and RGBD PNG dimensions matched.
  A real 8-frame neural half-SBS export was 160x94 at 8 fps, contained audio, and had
  duration 1.020998 seconds. Real cancellation returned `canceled`. Evidence/fixtures:
  `.runtime/verification/report.json`, preview.png, depth.png, synthetic source.png/mp4.
- Browser validation through CUA: synthetic file import, actual Small processed preview,
  Stereo/Depth tabs, preset changes and stale-preview messages, Base install gating,
  real queue submission/progress/download state and responsive layouts were inspected.
  At the 390px override the DOM reported content width equal to viewport width (376px).
  Browser error/warning logs were empty. Desktop was inspected at 1440px.
- Integration review caught/fixed cancellation turning a killed encoder pipe into a
  failed job, aligned API preview/export input defaults, and fixed hidden custom input
  size accidentally selecting 256 for named models. Named engines now submit 518.
- The older server still occupies 8765. Updated server was started on **127.0.0.1:8766**
  for this session, without terminating the older process. To launch normally later,
  stop the old server and run run.ps1, or use the documented 8766 uvicorn command.
- Evidence boundaries: CPU inference only. Small alone was downloaded/run on hardware;
  Base/Large/MiDaS have verified download manifests and tested adapters but remain
  hardware-unverified until installed. No physical headset, long movie, GPU, learned
  video-depth, hidden-surface reconstruction or Owl3D quality-parity claim is made.

## Objective
Map the installed Owl3D Desktop 2.3.0 behavior and build a clean-room compatible alternative in this repository. The delivered v1 converts photos and videos into common stereoscopic layouts.

## Repository state
- Root: `D:\Users\Hp\Documents\ChatGPT\3d`
- Branch: `master`, no commits yet
- Initial worktree: empty except `.git`
- Existing plan/source: none found
- Reference install: `C:\Users\Hp\AppData\Local\owl3d-desktop-app\app-2.3.0`
- Analysis copy: `%TEMP%\owl3d-re-analysis` (temporary, read-only to workers)

## Analysis wave

| Lane | Owner | Scope | Acceptance |
|---|---|---|---|
| Main process | /root/main_analysis | Extracted `.webpack/main/index.js` only | Report app services, conversion orchestration, remote endpoints, local commands, persistence, and evidence strings/functions |
| Renderer UI | /root/renderer_analysis | Extracted `.webpack/renderer/main_window/index.js` only | Report user flows, controls, source/output formats, settings, and evidence strings/components |
| Preload/packaged tools | /root/preload_analysis | Extracted preload bundle, package metadata, unpacked native modules, and bundled media tools | Report IPC surface, file/device integration, bundled tool capabilities, and evidence paths/strings |

All lanes are read-only and must not alter repository or installed files. No code execution of the installed Owl3D application is planned.

## Intended verification
- Coordinator independently inspect all worker findings against extracted bundle evidence.
- Freeze a bounded compatibility spec before implementation.
- Do not add or run tests unless requested; verify the delivered artifact by direct inspection and an appropriate build/smoke check only if needed for implementation correctness.

## Status
- Shortcut target/version/signature identified.
- Electron ASAR inventory extracted to a temporary analysis directory.
- Analysis wave complete: all three read-only lanes returned findings; no worker changed files or ran the app/tests.
- Coordinator independently decoded the main bundle string table and checked the core local processing boundary, formats, and bridge methods against the extracted bundles.
- `COMPATIBILITY_SPEC.md` froze the v1 scope before implementation.
- Coordinator implemented the local FastAPI/OpenCV/FFmpeg pipeline, browser UI, queue and cancel flow, preview simulation, optional ONNX model upload, Windows launcher, dependencies, ignore rules, and run/limitations documentation.
- Python source compile check and browser JavaScript syntax check passed. No conversion run or automated tests were performed.

### Analysis wave outcome
- Main process, renderer, and preload/tool lanes completed read-only analysis; no worker changed files, ran the app, or ran tests.
- Coordinator independently decoded the main-process string table and verified the Electron/ASAR inventory, renderer/preload bridge names, conversion modes, and local-process boundary against bundle evidence.
- Significant implementation evidence: `startProcessVideo`, `startProcessImages`, `startProcessStereoVideo` spawn downloaded `process_desktop_main.exe` with JSON stdin; payload supplies model/workspace/FFmpeg paths and output modes; renderer consumes progress/output/error lines.
- No app-specific conversion service endpoint was found in the main-process lane; current official product material also describes local AI conversion. This does not prove every app service is offline (auth/update/telemetry exist in the installed product).
- Bounded spec frozen in `COMPATIBILITY_SPEC.md`; implementation ownership is coordinator-only, files to be set by the spec.
- Wave review evidence: `git status --short` showed only the untracked `PROGRESS.md` before the spec was added; `git diff --stat` and `git diff` were empty. Both planning documents were then inspected in full. Worker changed-file reports were all empty.
- Completion boundary: the compatible alternative targets standard media outputs. It does not reproduce Owl3D's neural model quality, downloader/account flow, live display/VR integrations, or proprietary formats and job protocol.

## Model download repair (2026-10-18)

Objective: fix model installs failing through the active local server. Coordinator owns `depth_engine.py`, `README.md`, and `PROGRESS.md` because the diagnosis and downloader behavior are sequential.

Finding: `/api/models` reported Base and Large failed with WinError 10061. This process inherited HTTP(S)_PROXY=`http://127.0.0.1:9`; direct read-only HEAD to the pinned Small URL succeeded with HTTP 200 and matching manifest size/hash headers after removing proxy variables for that command.

Acceptance: when an HTTPS loopback proxy refuses a connection, retry that pinned model request without the proxy, preserve normal configured proxies, keep existing hash/session acceptance checks, and show both causes if direct access also fails. Inspect source/docs and verify the actual install path where safe; avoid the 1.34 GB Large model and full suite for this focused repair.

Outcome: added the loopback-refused direct retry and documented it. Through the actual downloader, Base transferred 388,917,010 bytes, passed its pinned SHA-256 (`dab40f67…26a7289`), loaded and completed CPU inference, and the restarted server reports Small and Base ready. `ast.parse` and `git diff --check` passed. The live server is running at `http://127.0.0.1:8766/`; Large and MiDaS remain available to install on demand. Full tests were not run for this focused fix.

## Conversion speed and ETA wave (2026-10-18)

User request: improve conversion speed, identify settings with the largest speed effect,
show remaining time, and make other useful changes. The existing GTX 1650 Ti DirectML
pipeline already overlaps CPU decode/render/encode with GPU depth inference for Auto
and hybrid. Baseline: all 61 unit tests pass. The active server reports the GTX 1650 Ti
and downloaded Small/Base/MiDaS models. Local FFmpeg advertises `h264_nvenc`, and a
30-frame 320x180 smoke encode completed successfully on the NVIDIA device.

Frozen acceptance: expose model input detail for dynamic neural models, clearly label
the strongest speed controls and settings with little/no speed cost, provide a live
video ETA only when a meaningful frame-rate estimate exists, and prefer NVENC only
after the configured FFmpeg passes a local capability smoke test; otherwise retain
libx264. Do not claim a speedup without measuring actual conversion/encoding. Keep
audio, output layout and CPU-only operation working.

| Lane | Owner | Exclusive files | Acceptance |
|---|---|---|---|
| ETA telemetry | ETA worker | main.py, tests/test_app.py | Monotonic elapsed/ETA/FPS fields; no fabricated estimate for stills, unknown frame counts or initial warmup; semantic endpoint/job tests |
| Speed controls and queue display | UI worker | static/index.html, static/app.js, static/style.css | Dynamic model input detail; clear relative speed impact per setting; ETA/estimating/unavailable states; readable actual encoder status |
| Encoder selection | encoder worker | video_encoder.py, tests/test_video_encoder.py | Probe NVENC with configured FFmpeg and a tiny local encode; use profile appropriate arguments only on success; safe libx264 fallback; mock failure cases |
| Integration and verification | coordinator | PROGRESS.md, README.md, main.py after ETA lane, live/smoke evidence | Wire encoder and ETA contract, run full suites and real image/video/encoder checks, inspect browser state; preserve unrelated worktree content |

Wave 1 handoff: the ETA lane added video-only frame metrics and a conservative estimate
window; its focused app suite reports 11 passing tests. The UI lane added the detail
selector, speed impact notes and queue metrics; its JS syntax and UI contract audit
report passing. The encoder lane added a cached quality-specific NVENC smoke probe;
its 7 unit tests pass, and local runs on the GTX 1650 Ti report NVENC working for all
quality profiles. Worker reports are claims pending coordinator review and combined
verification. All lanes released their assigned files.

Wave 2 completed by the coordinator: independently reviewed the combined source and
tests, integrated the quality-specific encoder profile and truthful encoder reporting
into video jobs, made NVENC failure retry with libx264, and corrected the user-facing
speed guidance to avoid claiming a universal fastest input size.

Verification completed:

- Full unit suite: 73 passing in `.venv` and 73 passing in `.venv-gpu`.
- `py_compile` for the app, encoder helper and app tests; `node --check static/app.js`;
  and `git diff --check` all passed.
- Real hybrid export through the GPU environment: Depth Anything Small used the
  verified `DmlExecutionProvider` on the GTX 1650 Ti, video stages overlapped, NVENC
  produced the H.264 output, and the queue observed `estimating`, `available`, and
  `complete` ETA states. The 48-frame 320x180 source became a 640x180 SBS MP4 with
  all 48 frames and AAC audio; measured job elapsed time was 6.05 seconds.
- Real encoder-only, three-run medians on generated clips: for 90 frames at 1080p,
  old libx264 medium was 0.952 s, new veryfast was 0.470 s, and NVENC p4 was 0.690 s.
  For 60 frames at 3840x1080, medians were 1.442 s, 0.683 s and 0.863 s. NVENC beat
  the old preset in these tests but was slower than current veryfast x264; it is not
  described as universally fastest. A PSNR check on the generated 1080p pattern was
  47.83 dB for medium, 45.43 dB for veryfast and 48.56 dB for NVENC p4.
- Warmed Small-model inference on a 240x400 source, 15 samples each: GPU medians
  were 0.133/0.181/0.263 s at 256/384/518 input detail; CPU medians were
  0.277/0.205/0.401 s. This workload demonstrates that input detail affects speed,
  while the fastest size depends on the source shape and backend.
- NVENC probe/profile and CPU retry paths have unit coverage; a GTX 1650 Ti smoke
  encode succeeded for all three quality profiles. Final worktree review and
  whitespace checks passed. Changes remain uncommitted.
