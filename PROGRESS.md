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

## DirectML on 4 GB GPUs (2026-10-18)

Request: allow every built-in named depth model, including Depth Anything V2 Large,
to attempt DirectML on a 4 GB GPU. Removed the Large hard CPU route/rejection and
added a 518 px maximum long-side input cap for Large on adapters with 4 GB or less.
Auto/Hybrid retains the existing one-time CPU retry after a GPU session/inference
failure; explicit NVIDIA GPU mode reports the failure. The UI recommends 256 px
detail and detects an old running backend that has not picked up this policy yet.

Verification: new mocked DirectML tests confirm Large is routed to the GPU at 4 GB,
uses the reduced input cap, and falls back to CPU after an out-of-memory failure.
The live server was restarted only after its queue contained canceled jobs and no
active conversion or completed output. Its `/api/status` now reports the new policy,
all four built-in models are ready, and `/api/jobs` is empty. Real previews on the
GTX 1650 Ti (3,935 MiB reported by the adapter) verified Large at both 256 detail
(252x196 model input) and the full 518 detail cap (518x280 model input) with
`DmlExecutionProvider`, no fallback, and GPU verification true. MiDaS also completed
a DirectML preview. A separate post-preview `nvidia-smi` sample showed 1,907 MiB
device memory in use and 2,029 MiB free; this is resident usage after inference,
not peak allocation. Earlier live Small and Base runs also showed DirectML. The
Large 518 cap and OOM retry behavior are additionally covered by mocked tests.

Final verification: all 75 unit tests pass in both `.venv` and `.venv-gpu`;
`py_compile`, `node --check static/app.js`, and `git diff --check` pass. The local
server is running at `http://127.0.0.1:8766/`. The implementation and its independent
critique brief are included in the follow-up commit after this verification.

## Roadmap continuation — 18 October 2026

Current baseline: branch `master`; existing modified `.gitignore`, `README.md`,
`main.py` and untracked `ROADMAP.md`, `StereoLab.spec`, `build-exe.ps1` are retained.
The roadmap is the authorized backlog; earlier evidence remains historical.

Frozen first batch: versioned GPU capabilities and conservative unknown-memory
policy; bounded GPU failure suppression and explicit retry; reproducible benchmark
records; bounded preview-depth reuse; stage timing and phased ETA integration.
Accept only semantic tests and independently verified results. Do not claim long
exports, exact VRAM peaks, or visual superiority from mocked tests.

Exclusive ownership for the first wave:
- Runtime worker: `depth_engine.py`, `tests/test_depth_engine.py` only. Capability
  schema v1, retry/reset method, bounded failure suppression, unknown-memory tests.
- Frontend worker: `static/app.js`, `static/index.html`, `static/style.css`,
  `tests/test_frontend_contract.py` only. Consume capability v1, GPU retry, truthful
  stale-backend handling; keep current preview response revision guard.
- Benchmark worker: `benchmarking.py`, `scripts/benchmark_roadmap.py`,
  `tests/test_benchmarking.py`, `benchmarks/manifest.json` only. Reproducible local
  fixture identity, repeats/median/range, output verification and environment data.
- Coordinator: `main.py`, new cache/telemetry helpers and tests, docs. Integrate retry,
  timing, preview reuse and benchmark schema; run CPU/GPU suites and live checks.

Capability contract agreed before dispatch: `runtime.capabilities` contains
`schema_version: 1`, `named_models_directml: true`, `gpu_max_input_dimension: 924`,
`large_limited_gpu_max_input_dimension: 518`, `limited_gpu_memory_mb: 4096`,
`unknown_memory_policy: "limited"`, `gpu_failure_cooldown_seconds: 60`, and
`gpu_retry_supported: true`. Retry endpoint: POST `/api/runtime/gpu/reset`, returns
`{"reset": true}`. Reset clears suppression without replacing a running inference.
Benchmark consumes `/api/jobs` timing fields when present and explicitly records
missing values. Initial manifest is synthetic and not representative visual proof.

### First batch integration checkpoint

Coordinator independently ran CPU and GPU environment suites: 126 tests pass in
both, JS syntax and whitespace checks pass. This is the current test checkpoint,
not the final handoff gate. The live GPU smoke on isolated port 8768 passed image,
depth/RGB-D, eight-frame video dimensions/FPS/audio/duration and cancellation.
First smoke comparison differed in one channel of one pixel by one 8-bit level;
rerun matched exactly. Smoke now reports actual differences and allows only sparse
one-level DirectML rounding (max <=1, mean <=0.001); CPU comparison stays exact.

Independent review found and fixed a preview/export admission race by gating the
processing transition through the preview lock. The engine already serialized
inference. Non-NVIDIA DirectML adapter cap display was aligned with backend policy.
Queue waits now have separate counters/timers; they include operation overhead
and overlap stage work. An initial test TempDirectory failed with sandbox ACL
WinError 5; ordinary inherited-ACL test directories work. Inaccessible generated
`.runtime/tmpbuy6m3w4` is left untouched.

Measured Phase 1.2 slice: DEPTH/RGB-D skip unused eye synthesis via render_output.
Alternating four retained samples per variant, 1920x1080 synthetic approximation,
full decode-to-PNG path: DEPTH medians 0.320607 vs 0.277169 s; RGB-D 0.335090 vs
0.312955 s. Exact PNG SHA256 agrees for all variants. Report:
`.runtime/render-benchmark/alternating.json`; runner:
`scripts/benchmark_render_outputs.py`. Does not measure API/queue/preflight/download.
Other allocation/grid optimizations remain unimplemented pending profiling.

Next bounded work: finish reproducible benchmark live runs and independent runtime
regressions, then Phase 1.3 calibration foundation and Phase 3.2 validated presets.
Calibration must be opt-in with a visible finite time budget, never change quality
settings silently, separate depth-input evidence from whole-export/encoder claims,
and report budget overruns if a single uninterruptible inference takes too long.
Presets must display exact settings and schema-validate imports/exports; no invented
speed rankings. Quality algorithm changes remain gated by matched fixtures/review.

## Final roadmap continuation handoff — 18 October 2026

Completed this bounded run near the usage limit; the remaining roadmap is explicit
in HANDOFF.md and the ROADMAP.md execution-status table. No commits or staging.
Preserved all earlier packaging changes; existing dist executable was not rebuilt.

Integrated: capabilities/retry/cooldown; raw preview LRU and export-admission gate;
preflight; stage/queue/session/resource telemetry; recent ETA/finalization/retry;
measured DEPTH/RGB-D optimization; budgeted opt-in detail calibration/history;
strict presets/import/export; exact-frame split comparison and zoom/pan; local
Host/Origin mutation boundary. Quality renderer and temporal-alignment experiments
were tested, measured and visually reviewed but deliberately not promoted.

Final independent suites: CPU 200/200 (3.585 s), GPU environment 200/200 (3.662 s).
Raw logs `.runtime/roadmap-verification/final-{cpu,gpu}-tests.log`. Compilation,
JS syntax and whitespace checks pass. Deprecation/CRLF warnings remain informational.
Final latest-server DirectML smoke passed image/depth/RGB-D, eight-frame H.264/AAC
video dimensions/FPS/duration and cancellation. One channel in one pixel differed
by one 8-bit level between independent preview/export inference calls (recorded,
not claimed exact). CPU image comparison remains exact. Calibration/cache report
`.runtime/roadmap-verification/gpu.json` verified reuse, invalidation and real GPU
inference; last calibration elapsed 1.782 s for its synthetic frame.

Benchmark reports: approximation CPU 15 verified runs; Small CPU, GPU and hybrid
15 each; another 15 CPU runs in the same ORT 1.24.4 GPU runtime to avoid conflating
ORT 1.31.0 with device selection. Five fixtures, three repeats. Resource-enabled
hybrid run verified three outputs, process snapshots and seven whole-device memory
samples (207–375 MiB). Not process VRAM peaks or long-export proof. See HANDOFF.md
for exact paths, matched-runtime caveats and earlier-build report boundaries.

Browser verified preset/dirty behavior, GPU retry, exact 400x240 source/depth A/B,
125% synchronized zoom, correct side labels, complete calibration status and no
input change before explicit Apply (256 stayed selected until Apply 384). Saved
screenshot `.runtime/roadmap-verification/browser-preview.png`. Broad narrow-layout,
keyboard and physical-display acceptance is still open.

Experiments retained outside runtime: visibility renderer ~97–166 ms vs 2.9–3.8 ms;
motion alignment ~28–30 ms vs 2.1–2.2 ms with mixed synthetic errors. No demonstrated
real-video improvement; baseline renderer/filter remain selected. Source-aware raw
normalization, full temporal clip preview, long all-model exports, full calibration,
queue source-retention and durable projects/caches are next gated work.

Live latest GPU app: http://127.0.0.1:8770, PID 12464, session 31019. Port 8768 and
its 37 completed records/download URLs were preserved after a user export; it runs
older backend code. Own idle CPU verification server 8769 was stopped. All worker
lanes are released. See HANDOFF.md before continuing; do not restart active jobs.

Final handoff usage check: 89% five-hour / 29% weekly. Untracked-file whitespace
check found one extra EOF blank line in benchmarking.py; removed and rechecked.

## Resumed roadmap wave — 18 October 2026

Resume authorization retains the prior until-near-usage-limit/handoff objective.
Starting usage: 0% five-hour / 30% weekly. Branch master / HEAD0c08d1a and all dirty
packaging/roadmap changes preserved. Ports8768 and8770 live, idle; 8768 olderbackend
37 completedrecords preserved, 8770 current backend five completedrecords. Other
old ports unavailable. No restart of either existing server is authorized by this
plan while it has active jobs or unaccounted outputs.

Frozen wave: extend Phase0.1 with longer synthetic all-model exports + cancellation,
and Phase3.1 with an explicit full-detail still check (bounded decoded dimensions,
current proxy default retained, exact source comparison and current settings).
This is not evidence of representative real-footage quality or multi-hour stability.
A text question for a user-selected footage path is pending; synthetic work proceeds.

Exclusive lanes:
- Stress worker: scripts/stress_roadmap.py, benchmarks/stress-manifest.json,
  tests/test_stress_roadmap.py; generated artifacts onlyunder .runtime/stress-verification.
  Own all real GPU benchmark scheduling against idle8770 in this wave. Other lanes
  must not run realGPU inference concurrently. Preserve user8768 outputrecords.
- Preview backend worker: main.py, new preview_policy.py, tests/test_full_preview.py.
  Add optional preview_detail proxy(default)/full; source fullimage capped12MP,
  maxdimension8192, 24MiB upload. Full means no source downscale, not higher neural
  input than hardware cap. Return original source dimensions, processing dimensions,
  preview_detail, is_proxy; include policy in rawdepth cachekey. Keep defaultcallers,
  admission/cancel/model/runtime contracts and CPUfallback truthful. Reject oversize
  fullstills with actionable400/413 before expensive inference. Existing running
  server stays untouched; unit verification uses mocked sessions/CPUapproximation.
- Preview frontend worker: static/app.js,index.html,style.css,
  tests/test_frontend_contract.py. Visible Proxy/Full-detail still selector; fullonly
  decoded images, explicit12MP/8192 bound and GPUinputcap explanation; uploadoriginal
  File forfullstill, JPEGcanvasforproxy. Return dimensions clearly. Detail changes
  invalidate preview+calibration context; stale responses cannot replace current.
- Coordinator: PROGRESS/HANDOFF/ROADMAP/README, reviewintegration and independent
  CPU/GPUenv suites, idle-server scheduling, final evidence and next bounded spec.

Contract preview fields: request preview_detail='proxy'|'full'; response source_width/
source_height remain ACTUAL processingframe forA/B; original_width/original_height
report decodedsource, preview_detail string, is_proxy bool. Fullvideo frames rejected
by frontend; backend receives decodedstillframe so has no authority to infer its
origin beyond uploadedimage. Fullsourcebudgetlimits do not impose a hard process
memorypeak; neural dimensions are reported separately. Full entries may exceedthe
32MiB cachebudget and then are intentionally not retained.

Full-still wave integration: independent full suites pass in both environments:
224 tests / CPU5.285s and GPUenv4.977s; node syntax and tracked whitespace pass.
LiveCPU server8771 verifies original1200x640 vsproxy960x512 with approximationCPU.
No GPU work was run by coordinator; stress lane retains exclusive inference use.

Next frozen wave: plans/processed-clip-preview.md. Backend helper lane owns new
preview_clip.py and tests/test_preview_clip.py only. Frontend lane owns static/*
and tests/test_frontend_contract.py. Coordinator owns main.py integration,
tests/test_clip_jobs.py, liveCPU validation and documentation. Stress lane retains
its prior file and realGPU ownership. Extraction helper API frozen below:
prepare_preview_clip(ffmpeg, source, destination, start_seconds, duration_seconds,
source_fps, cancel_event, on_process=None, timeout_seconds=30.0) -> Path.
ClipCancelled distinct from RuntimeError; helper registers Popen or None via
on_process. Invalid interval/paths reject without mutation; unique destination
must not preexist; only owned partial destination may be removed. Bound proxy
960x540/24fps/120frames, optionalaudio, CPUlibx264. Main integrates purpose and
explicit capability, same pipeline and source/proxy cleanup; UI requires capability
before upload and rejects wrong-purpose response. FIFO, no priority/preemption.

Clip integration checkpoint: 247 tests pass independently in each environment
(CPU7.426s/GPUenv7.378s). Coordinator liveCPU8771 validates nonzero9s start,
3s/36frames12fps640x180SBS with AACaudio and exactdecoded equality to ordinary
export of matched extractedproxy (same smoothing.65/refinement/settings). Invalid
40sstart failsbeforeinference. SmallCPU clip canceledafter2frames; bothclipvideo
anddownload404. CurrentCPU serverbuild101dc031; generated scratchreports in
.runtime/resume-verification. UI auditfixed durationchange stalecheck and status
loaderror, with furtherpollrecovery auditunderway. Fullstillbrowser12MPbounded
source/depthcomparison was checked and screenshot retained.

Verification lane after stressrelease: new scripts/verify_clip_preview.py and
new tests/test_clip_verification.py; generated .runtime/clip-verification only.
Verify latestsource server8772(.venv-gpu,PID15152) explicitSmallGPU finiteclip,
matchedproxyoutput/frame/audio check, realextraction andinferencecancellation,
oldbackendrejection. No otherrealGPU lane active. Preserve8770/8768 unchanged.
Coordinator owns allmain changesanddocs,frontendowns pollfixes untilrelease.

Clip review correction: preflight now preserves finite positive sub-1 FPS and
reports fps_provisional. Clip start validation does not reject timestamps using
provisional30FPS fallback; extraction uses explicit24FPS fallback. Output proxy
FPS must be measured before inference. Redtest .5FPS was30 and missingprovisional;
15preflighttests+8clipjobtests green. GPU8772verification isstillloaded101dc031,
prior tothis sourcecorrection; reportthatfingerprint truthfully.

Next frozenqueuewave: plans/queue-workflow.md. Backendworkerexclusive main.py,
newjob_queue.py,newjob_settings.py,tests/test_queue_jobs.py (namesmayomitunneeded
helper). Frontendworkerexclusive static/app.js/index.html/style.css,
tests/test_frontend_contract.py. Clipverificationworkerretains prior runner/tests
andGPU8772. Coordinatoronlydocs/plans, independentchecks andnewlivequeueartifacts;
no main/static edits untilownersrelease. Source retention remains explicitlyabsent.
Queueorder/queuedsettings must reject worker-dequeue races andpreserveclip slots.

Stresswave independentacceptance: coordinator decoded allfourretainedMP4s:
360frames,640x180,12fps each. Report success4models and canceledstates verified;
outputhashes independentlyrecomputed. Originalreportbuild1fd77b5a (priorroadmap
baseline, not new fullstill/clip source). Modelwall Small75.02/Base133.11/
MiDaS53.15/Large263.12seconds, requested518 with actual518x294exceptfixedMiDaS384.
Device-widesamplemax566/938/1182/2025MiB; notprocessVRAM. Allrunframesare generated,
notrepresentativerealfootage. Sourceuserpathquestion unanswered; no personalfiles
searched. Successfulreport .runtime/stress-verification/20261018T110251Z-f0392917/report.json.

Low-FPS extraction followup: generated.5FPS source showed shorter seeked video,
and.1FPS produced an empty-stream container. Helper now uses minimum1FPS for
sparse sources while keeping24FPS/120frame maximum; actualproxyframes/FPS/duration
remain separatelyreported and maybeshorter thanrequested interval. Added real
slowFPS case +argvexpectation: semanticred2failures, final13helpertests pass1.580s.
No forcedtailpadding; no claimthatrequestedlength always equalsplayablelength.

Queuewave integration checkpoint: completeCPU/GPUenv suites285tests green
8.714s/8.853s, JSsyntax andtrackedwhitespace clean. Coordinatorreviewed queue/
settings helpers, handlerlocking, modelreadiness/dequeue recheck andworkerstartup
lock. Browser verified2readyfiles reorder, batchfreezestrength5, globalstrength0
changesdidnotmutatesnapshots, submittedjobssnapshot5; duplicatenewreadyitemwith
current0 andNOsubmission. RefreshdropsbrowserFiles andshowsreselectionrequired;
validserversnapshotsremain. Live screenshot caughtmetadata squeezedbywideactions;
frontendCSSfixunderway, notyetacceptedvisually. Liveatomicqueuemutationtestpending
GPUverificationlane release (CPUvideoencodewoulduseNVENCconcurrently).

GPUclipverification functionalmedia checks pass butencodedpixelgateFAILED;
max40/mean.0024837 across12,441,600channels,10,576changed. Strictthresholdnotrelaxed.
Workerdiagnosingordinaryrepeatvarianceandcompletingcancelcase independently.
CPUmatchedproxyclip/export remains exactlyequal36decodedframes. Preservefailed
reportandkeepGPUencodedidentityacceptanceexplicitlyopen.

Final queue/output acceptance: correctedliveAPIrunner passes whileanordinarySmallCPU
360frameexport runs tocompletion. Reorder C/A/B preservedclip slot2; activeedit+
remove409, pendingBstrength0/refinementoff snapshotcommitted, Aremoved canceled+
sourcefalse/download404, staleorder409. FinalBPNGexactlyduplicatessource. Rawreport
.runtime/resume-verification/queue-live.json (latestbuild608134272b78059e5a4b5db83db00cd75ed3c08a4a8a242a2f8d2c019ca7ec23).
First runnerusedPOSTondownloadGETroute andgot405afterothercheckspassed; corrected
method, retainedfirst-attemptrecords andreranfullcheck successfully.

Outputownership reviewfoundold8charIDcollisions andexistingtargetcleanup risk.
Rootadded3semantic tests (red2fails), fullUUIDhexnames, exclusiveoutputreservation,
andcleanupguardforownedtargets only; focused3+clip9green, backendindependent
read-onlyreviewfoundnoregression. Existingoutputs neverrewritten/renamed. No naming
templateUIyet. Final complete suites288tests: CPU9.055s/GPUenv9.130s, bothOK.

QueueCSS initialappendhadliteralbackslash-n anddidnotparse; rootbrowsercomputedstyle
caughtit. Correctedactualnewlines, reloadedandvisuallyaccepted530pxmetadatawidth+
actionsbelow (35px/530px two-column grid at1265pxviewport); screenshotqueue-browser-fixed.png.
Broadphysical/narrow-layout acceptance stillopen. Oldfailedscreenshotretained.

GPUclipdiagnosis: secondreport20261018T113023Z-3a3457b7 recordsfunctionalmedia/
cancelchecks; strictpixelgateoverallnonpassing/inconclusive, toleranceNOTrelaxed.
Ordinary-vs-ordinary repeatedGPUH264 max29/mean.0016760706/5913changedchannels.
Clipmatchessecondordinary EXACTacross36frames. Coordinatorindependentlydecodedall
threeandverifiedthesemetrics (gpu-clip-independent.json). OwnedextractionFFmpegexit
observed byworkerPID25844rc1+partialremoved+sourcehashpreserved; APIinferencecancel
at23frames/canceled/bothroutes404 doesNOTproveOSencoderexit. Firstfailurepreserved.
Reportloadedbuild101dc031, predatesqueue/outputownership/lowFPSfixes.

LatestfreshGPU8772build608134272b... passesfull1200x640SmallDMLstill with exactsource
pixels andseparatemodelinput252x140, plus1s12frame640x180clipwithAACaudioandstrictGPU
nofallback. Reportlatest-gpu-smoke.json, PID14348/session36321. FinalCPUlow.5fps
source nonzerostart clipcompletes asproxy1fps2frames; actual2s vsrequested3s explicitly
reported (lowfps-live.json). No wholemovie/multihourqualityclaim.


Closing resumed run: 85% five-hour / 43% weekly usage observed; no reset used.
HANDOFF.md rewritten with completed slices, nonpassing GPU pixel gate, exact
commands/reports, server ownership and next bounded work. All workers released.
Own idle CPU8771 stopped after accounting for 11 terminal records. Latest GPU8772
PID14348/session36321 retained with one completed clip, no active jobs, and source
fingerprint independently confirmed to match loaded backend. User8768 (39 records
at last check) and baseline8770 (16 records) preserved. Browser tab4 retained at
latest8772. No commits/staging, executable rebuild, or broad cleanup performed.
This until-near-limit-and-handoff run is complete; the roadmap remains unfinished.


## User-requested roadmap commit

User requested a commit after the handoff. Staged 58 roadmap source, test, fixture
and documentation files. Preserved the pre-existing executable packaging edits
unstaged: .gitignore, StereoLab.spec, build-exe.ps1 and packaging-only README/main
hunks. Working files were not reverted. Exported the exact index to an isolated
.runtime/commit-verification directory and independently ran both full suites:
CPU 288 tests / 7.366s OK; GPU environment 288 tests / 7.258s OK. JavaScript syntax
and staged whitespace pass. Generated media, weights, runtime state and binaries
are excluded. This is a local source commit, with no push or EXE rebuild.
