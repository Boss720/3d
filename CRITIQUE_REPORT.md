# StereoLab independent critique brief

**Prepared:** 18 October 2026
**Repository:** `D:\Users\Hp\Documents\ChatGPT\3d`
**Current HEAD:** `79379af` — `Improve conversion speed and ETA reporting`
**Working tree:** six implementation/documentation files have uncommitted changes for 4 GB DirectML support. This report is an evidence-based handoff for critique; it does not authorize edits or a commit.

## Request for the reviewing LLM

Please critique the current implementation and propose a prioritized set of changes. Treat the repository as the source of truth if you can inspect it. Focus on correctness, reliability, performance, UI clarity, and whether the claims match the evidence. Separate confirmed defects from plausible risks and product tradeoffs. For each proposed change, explain the user impact, likely files/components, a small acceptance test, and any tradeoff. Prefer targeted, testable improvements over a rewrite.

The user wants a polished local stereo photo/video converter with a workflow closer to Owl3D, more depth-model choices, NVIDIA GPU support on a GTX 1650 Ti with approximately 4 GB VRAM, and CPU/GPU cooperation where useful. Do not assume access to Owl3D internals or claim feature/quality parity. This is a clean-room project; critique similarity at the level of user workflow and documented capabilities.

Please explicitly assess:

1. Whether the 4 GB GPU policy is safe and honest, especially for Large and for adapters whose dedicated-memory value is missing or inaccurate.
2. Whether Auto, Hybrid, explicit GPU, and CPU-only behavior is understandable and correct after GPU failures, including the per-model/per-adapter/per-input-size failure cache and retry at lower detail.
3. Whether DirectML execution reporting distinguishes a GPU session from GPU-only execution, given that CPU operators may remain in the ONNX graph.
4. Whether custom ONNX models should remain CPU-only, or whether there is a feasible, safe DirectML path that respects arbitrary model input/output contracts.
5. Whether the bounded video pipeline, cancellation behavior, ETA calculation, and NVENC fallback have gaps or misleading edge cases.
6. What additional real-hardware benchmarks, memory measurements, quality comparisons, or UI tests would most reduce uncertainty.
7. Which user-facing improvements would most help the app feel complete, without copying proprietary assets, code, model files, or protocols.

Do not make source changes. Return: (a) the five highest-priority findings, ordered by severity and confidence; (b) suggested changes with tests/acceptance criteria; (c) evidence that is missing or contradictory; and (d) any questions whose answers materially change the recommendations.

## Product and user context

StereoLab is a Windows-first, local FastAPI service with a browser UI. Users add photos/videos, choose a depth engine and stereo layout, preview one frame, adjust stereo controls, convert a queue, then download local outputs. Its goal is a useful independent alternative with a familiar model-library/preview/queue workflow, not a clone of Owl3D's internals. Current supported exports include SBS, half-SBS, top-bottom, half-top-bottom, cross-eye, anaglyph, depth-only, and RGB-D. Video is H.264/AAC MP4 when FFmpeg supports the required codecs; photos are PNG.

The user's recent requests, in order, were to improve Owl3D-like workflow/model coverage, enable NVIDIA acceleration on a GTX 1650 Ti (not RTX-only CUDA), allow CPU/GPU cooperation, fix model downloads, improve conversion speed and surface settings with large speed impact plus a time remaining estimate, and now allow all built-in models to attempt GPU inference with 4 GB VRAM. The earlier download issue was traced to an inherited refusing loopback proxy; the pinned download path now retries directly only for that specific refusal and still checks the model's pinned size/hash and load/inference acceptance.

Conversion is local. Installing models contacts Hugging Face; uploaded media is processed by the local service. The project explicitly avoids Owl3D proprietary weights, processor code, account behavior, and job/cache protocol.

## Current architecture and behavior

- **Depth engines:** Depth Anything V2 Small, Base, Large, and DPT Hybrid/MiDaS are pinned ONNX exports managed by `depth_engine.py`. The backend uses ONNX Runtime with Windows DirectML and CPU providers. Model revisions, licenses, sizes, and hashes are recorded. A single session is kept resident and inference is serialized.
- **GPU selection:** `gpu_hardware.py` enumerates DXGI adapters and prefers a physical NVIDIA adapter. This machine has AMD integrated graphics first and a GTX 1650 Ti second; the backend selected DXGI adapter 1. Provider availability and successful inference are reported separately. A DirectML session can still contain CPU-assigned ONNX operations.
- **Execution choices:** Auto and Hybrid use DirectML when available and fall back to CPU after session/inference failure. For neural video conversions, those modes overlap bounded CPU decode/resize and CPU render/write stages around one inference stage. Explicit NVIDIA GPU mode requires the GPU and surfaces failure; it does not silently become CPU mode. CPU-only runs sequentially. Approximation and uploaded custom ONNX are separate CPU paths.
- **4 GB policy:** Ordinary GPU inputs have a maximum long-side dimension of 924 px. On an adapter reporting 4,096 MiB or less, Large has a 518 px maximum long-side dimension. The UI recommends starting Large at 256 px neural detail. This cap controls input size and expected activation demand; it is not a VRAM reservation or guarantee. DirectML has no configured hard memory ceiling, and available memory can vary with other applications.
- **GPU failure behavior:** Auto/Hybrid retry the failing frame on CPU and cache the failure for that model, GPU adapter, and requested input detail to avoid repeating a failing GPU attempt on every video frame. A different input detail gets a fresh GPU attempt. Explicit GPU mode reports the failure. The session is invalidated after a GPU inference error.
- **Video processing:** `video_pipeline.py` uses bounded queues (default size two), preserves order, and keeps one GPU inference call active at a time while CPU stages overlap. Temporal smoothing stays ordered in the CPU consumer. Errors and cancellation join workers; an in-flight native decoder, inference call, or blocked callback cannot be forcibly interrupted by Python.
- **Encoding:** `video_encoder.py` probes the configured FFmpeg with a small NVENC encode per quality profile and caches the result against the FFmpeg binary. It selects NVENC only after success and has libx264 fallback, including retry after a later NVENC failure. NVENC is not assumed to be universally faster.
- **Speed and ETA:** The UI labels model, neural input detail, output resolution, video length, and encoder quality as high-impact controls. Strength/convergence/inversion have negligible cost; refinement and temporal smoothing add smaller per-frame CPU work. Video ETA is based on observed frame throughput and withheld until there is enough signal (at least five frames and two seconds); stills and unknown frame counts do not get an invented ETA.
- **Preview:** The preview processes one browser-selected frame and downsizes large sources to at most 960x540 before depth estimation. It is not a temporally smoothed video preview, and its JPEG/browser capture can differ from full-resolution export detail.

Useful source locations: `depth_engine.py` (model catalog, provider planning, input caps, fallback), `gpu_hardware.py` (adapter enumeration), `main.py` (API, preview, job ETA and video route), `video_pipeline.py` (bounded overlap/cancellation), `video_encoder.py` (NVENC probe/fallback), `static/app.js` and `static/index.html` (settings and explanations), and `tests/test_depth_engine.py`, `tests/test_app.py`, `tests/test_video_pipeline.py`, `tests/test_video_encoder.py`.

## Latest 4 GB GPU change under review

The previous Large-on-4-GB CPU-only rejection was removed. All four built-in named neural models can now request DirectML. Large uses the smaller 518 px long-side cap on adapters reporting no more than 4 GB. Lowering input detail after an Auto/Hybrid GPU failure permits another GPU attempt. The UI detects an older backend policy and tells the user to restart after active conversions; it distinguishes this from the new Large policy.

The current uncommitted changes are in `depth_engine.py`, `tests/test_depth_engine.py`, `static/app.js`, `static/index.html`, `README.md`, and `PROGRESS.md`. New semantic tests cover Large GPU routing at 4 GB, reduced input sizing, CPU fallback after a mocked OOM, and a new GPU attempt at lower detail. Custom ONNX and visual approximation remain CPU-only; the phrase “all models” in this implementation means all four built-in named neural models.

## Verification and evidence boundaries

**Automated checks:** 75 tests pass in `.venv`; 75 pass in `.venv-gpu`. Python byte compilation for the changed engine/test, `node --check static/app.js`, and `git diff --check` pass. The test suite emits a non-fatal Starlette/httpx deprecation warning.

**Live adapter:** The local `/api/status` reports ONNX Runtime 1.24.4, DirectML plus CPU providers, verified NVIDIA inference, and a GTX 1650 Ti adapter with 3,935 MiB dedicated memory. `/api/models` reported all four built-in models ready; `/api/jobs` was empty. The local UI server was running at `http://127.0.0.1:8766/` at the last check.

**Real inference:** A live Large preview on that GTX succeeded at 256 detail (reported model input 252x196) and at the full 518 detail cap (518x280); both reported `DmlExecutionProvider`, `gpu_verified=true`, and no fallback. A MiDaS preview also ran through DirectML. Earlier Small preview inference used DirectML. A Base hybrid video job showed DirectML while processing but was later canceled, so that run is not evidence of a completed Base export. A post-preview `nvidia-smi` sample showed 1,907 MiB used and 2,029 MiB free on the whole device; it is a resident post-run sample, not a peak and not necessarily all app-attributable.

**Performance evidence:** Existing README benchmarks include warmed Small inference medians on a 240x400 source: GPU 0.133/0.181/0.263 s at 256/384/518 detail and CPU 0.277/0.205/0.401 s. A separate 100-frame Small-model test reports warmed median inference of 0.214 s GPU versus 0.414 s CPU at 518x308. These are workload-specific and use different sampling contexts; do not combine them as one benchmark. A real 48-frame 320x180 video export previously completed in 6.05 s with GPU depth, NVENC H.264, AAC audio, and observed ETA states. Encoder-only synthetic benchmarks found libx264 `veryfast` faster than NVENC p4 in the tested clips; NVENC beat the older `medium` profile in those samples. None of these measurements guarantees performance on other footage, resolutions, drivers, or background GPU loads.

## Known limitations and open review questions

- The 4 GB policy is a best-effort model-input heuristic. It does not query current free VRAM or enforce an allocation limit. Large was proven on this adapter at the 518 cap for a small preview, but a longer full-resolution export or concurrent GPU workload may behave differently.
- The cap is only applied when the adapter reports a positive memory amount at or below 4,096 MiB. Assess whether unknown/zero memory should receive a conservative cap instead of the normal 924 px cap.
- The server/frontend compatibility check currently looks for a phrase inside the `gpu_memory_policy` status string. Assess whether an explicit policy/version field would be safer than relying on text.
- GPU verification means a DirectML provider session completed inference; it does not guarantee every graph operation ran on the GPU or establish a particular utilization percentage. Review whether the UI/API should report more granular placement evidence.
- Auto and Hybrid currently choose the same neural inference provider; their video distinction is primarily the overlapped versus sequential pipeline. Assess whether those names/labels communicate that distinction clearly.
- The GPU-failure cache is process-local and keyed by requested detail. Assess whether it needs expiry, clearing on device/session changes, or visibility to the user; also make sure it does not suppress a useful retry after transient resource pressure.
- The current ETA is an observed average-throughput estimate. Assess behavior when processing speed changes substantially by scene, model warmup, CPU/GPU contention, or encoder bottlenecks.
- Custom ONNX is CPU-only through OpenCV DNN. Generic ONNX input/output contracts are not reliably discoverable, so enabling DirectML may need explicit compatibility validation and must not imply arbitrary graph support.
- The app does not claim metric depth, learned temporal consistency, hidden-surface reconstruction, HDR/spatial/MV-HEVC preservation, live desktop conversion, VR pairing, or Owl3D quality parity. Assess which additions would be most useful next and how each can be evaluated honestly.

Please ground any external claims about current Owl3D capabilities in current public sources if you choose to compare products. Do not treat the compatibility spec's inspection of an installed 2.3.0 build as proof of every present-day Owl3D feature.
