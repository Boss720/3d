# StereoLab development roadmap

Prepared 18 October 2026 against commit `0c08d1a`.

## Execution status — 18 October 2026

This is a reconciled backlog. Completed slices do not imply their entire phase is
finished. See PROGRESS.md and HANDOFF.md for dated commands, reports and limitations.

| Item | Current outcome | Remaining acceptance |
|---|---|---|
| 0.1 | Benchmark runner, repeated CPU/GPU/hybrid Small checks; all four models completed generated 30s/360-frame GPU exports and separate cancellations, with sampled resources | Representative real footage, multi-hour reliability and exact process VRAM/encoder cleanup evidence |
| 0.2 | Capability schema v1, conservative unknown memory, bounded cooldown, visible GPU retry, actual execution metadata | Broader adapter/device validation beyond local GTX and mocked policies |
| 0.3 | Preflight decode/first frame, dimensions, codec inventory, writable output/disk budget; historical docs labeled | Full-stream corruption and export-time failures remain possible |
| 1.1 | Bounded raw preview-depth LRU; presentation edits reuse depth; precise invalidation/reset | Continue real-media interaction validation |
| 1.2 | DEPTH/RGB-D skip unused eye synthesis, with measured PNG-path gain and exact outputs | Other allocation/grid changes require profiles |
| 1.3 | Opt-in budgeted input-detail measurements, one warmup and two samples per candidate, explicit apply and bounded history | Model/encoder and complete-export calibration comparisons; no Auto policy change |
| 1.4 | Recent-frame estimate/range, phased finalization, encoder-restart reason and deterministic tests | Comparable-history queue estimates remain deferred |
| 2.1 | Motion-aligned history prototype and matched synthetic clips evaluated, not adopted | No consistent improvement demonstrated; raw-scale normalization remains unsolved |
| 2.2 | Visibility-order prototype with hole masks and matched fixtures evaluated, not adopted | Large CPU cost and no real-video superiority established |
| 3.1 | Exact-frame Source/Depth split, bounded zoom/pan, bounded full-detail stills and 1–5s processed proxy clips using export settings | GPU encoded pixel gate inconclusive from ordinary-export self-variation; deeper stereo-run A/B, real footage and broader responsive/keyboard/display review |
| 3.2 | Four exact performance presets, reset/dirty state, strict versioned JSON import/export | Broader accessibility and real-device validation |
| 3.3 | Atomic pending-export reorder/remove/settings, immutable snapshots, ready-file order/batch settings, source-aware retry/duplicate preparation | Naming templates, notification opt-in, more shortcuts/accessibility; durable retry sources remain Phase 4 |
| Delivery API review | Loopback Host validation and same-service browser write-origin checks, CLI access retained | This local boundary is not public deployment authentication |

Quality prototypes are reproducible experiments outside the supported app path.
The existing renderer/filter remain selected; neither experiment passed promotion.

## Direction

Build a dependable local photo/video-to-stereo workstation: quick to preview, predictable to run, easy to tune, and capable of visibly better stereo. The primary hardware target remains Windows with a GTX 1650 Ti and 4 GB VRAM, with a useful CPU-only path. Familiar conversion workflows are a design reference; matching another product's depth quality is not an established capability.

This is a proposed backlog, not a promise that experimental techniques will be faster or better. Deliver phases in order, in small independently verifiable changes. Effort below is relative: S = a bounded change, M = several connected changes, L = substantial implementation and validation, XL = research or a new subsystem. These are not delivery dates.

## Suggested LLM workflow

Use the **army skill** (`C:\Users\Hp\.codex\skills\army\SKILL.md`) to organize substantial roadmap work: first size the task and choose the least expensive suitable model, then define independent lanes with concrete outputs, acceptance criteria, file allowlists, dependencies, and verification commands. Keep a root `PROGRESS.md` handoff for multi-milestone work. Prefer isolated worktrees; when agents share a checkout, assign exclusive file ownership. Treat worker reports as claims and have the coordinator inspect the combined diff and independently run the relevant verification before accepting a phase.

Within that workflow, use **GPT-6 Luna Max** for bounded evidence gathering and implementation experiments in Phases 0, 2, and 6. Use **GPT-6.1 Sol** as the phase lead: set acceptance criteria before work starts, review the evidence and changes, resolve tradeoffs, and decide what is ready to proceed. Sol should own architecture decisions and final integration for all three phases; Luna's output is a proposal or scoped change for review, not independent approval.

- **Phase 0:** Luna Max can build the benchmark inventory, collect repeatable measurements, and implement clearly scoped instrumentation. Sol defines what counts as comparable evidence and reviews the measurement method and runtime contracts.
- **Phase 2:** Luna Max can prototype one renderer or temporal-quality change at a time against agreed fixtures. Sol selects the experiment, checks visual and motion evidence, reviews regressions, and decides whether it becomes a supported option.
- **Phase 6:** Luna Max can investigate candidate models, runtimes, precision formats, and acceleration paths. It should report verified support separately from assumptions and include license, hardware, setup, memory, and benchmark evidence. Sol reviews the sources and decides whether a prototype is warranted.

Keep concurrent assignments on separate files and interfaces. Have Luna finish and report its scoped work before Sol integrates or edits those files. Do not run two writers on the same files at the same time. Include the army worker contract in each assignment, including the file allowlist, no-commit instruction unless requested, and raw verification evidence. Model availability depends on the host; if Luna Max is unavailable, use an available suitable model for the bounded lane and retain Sol for review. Model tiering never replaces verification.

## Baseline and reconciliation

Already implemented:

- Four downloadable built-in ONNX models: Depth Anything V2 Small/Base/Large and DPT Hybrid/MiDaS, with pinned downloads and acceptance checks.
- DirectML GPU inference, CPU fallback for Auto/Hybrid, explicit GPU failure reporting, and one resident inference session.
- A 518-pixel long-side cap for Large on adapters reporting at most 4 GB; other GPU inputs normally cap at 924 pixels. Custom ONNX and approximation are CPU-only.
- Bounded overlapping CPU decode and rendering around inference, NVENC probing with software fallback, per-job FPS/elapsed/ETA, and speed-impact explanations.
- Single-frame preview, stereo controls, edge refinement, motion-gated temporal smoothing, eight layouts, queue cancellation and downloads.

The older `COMPATIBILITY_SPEC.md` retains superseded restrictions on Large and NVENC. `CRITIQUE_REPORT.md` describes the pre-commit working tree; its implementation changes are now committed. Treat source at the baseline commit as authoritative and preserve older measurements as historical evidence. Phase 0 updates the documentation hierarchy.

Recorded verification at this baseline includes 75 passing tests in each environment and successful Large GPU previews on the local GTX. These were recorded previously, not rerun for this roadmap. Previews and short tests do not establish long-video reliability, peak VRAM, or sustained speed. Source inspection for this roadmap confirms CPU inverse warping, appearance-gated temporal filtering without optical-flow alignment, in-memory job state, and an existing bounded pipeline.

## Priority overview

| Phase | Outcome | Priority | Effort | Depends on |
|---|---|---|---|---|
| 0 | Reliable measurements and explicit runtime contracts | P0 | M | Current baseline |
| 1 | Faster repeat previews and evidence-based conversion tuning | P1 | L | Phase 0 measurements |
| 2 | Better edges, occlusions, and temporal stability | P1 | L | Quality fixtures from Phase 0 |
| 3 | Comfortable preview, settings, and queue workflow | P1 | L | Stable job/settings contracts |
| 4 | Durable jobs, reusable projects, and depth caching | P2 | L | Phases 0 and 3 |
| 5 | Advanced editing, model adapters, and export fidelity | P2 | XL | Phases 2 and 4 |
| 6 | Optional acceleration and research models | P3 | XL | Benchmark and quality gates |

P0 addresses correctness and evidence gaps; P1 improves everyday use; P2 extends capability; P3 is gated experimentation. UI and quality work can proceed in parallel after interfaces and exclusive file ownership are agreed.

For model responsibilities, use the **Suggested LLM workflow** above: Luna Max handles scoped research/experiments, while GPT-6.1 Sol owns design gates and final review.

## Phase 0 — Establish a trustworthy baseline

### 0.1 Reproducible performance and quality suite — P0 / M

Create a local benchmark runner and a small redistributable or user-supplied clip manifest. Include static fine detail, thin objects/hair, fast movement, camera pans, hard cuts, low light, animation, and foreground objects revealing background. Include landscape and portrait inputs, plus audio/timing fixtures.

Record source identity, settings, model hash, runtime/provider, adapter/driver, requested and actual input dimensions, output dimensions, cold initialization, warmed timings, total wall time, stage timings, throughput, CPU/RAM and sampled device memory where available. Explain that sampled device memory is not an exact process allocation peak. Record queue stalls separately: overlapping stage times cannot simply be summed to calculate wall time.

**Acceptance:** repeat a representative run three times; save machine-readable results and median/range; verify frame count, duration, audio, layout and successful decoding. Compare CPU, explicit GPU and cooperative processing on identical settings. Complete a longer representative export on each built-in model, including Large at its cap, while observing memory and cancellation. Mark missing hardware evidence explicitly.

**Likely files:** `scripts/benchmark_gpu.py`, `scripts/verify_devices.py`, new benchmark fixtures/results schema, tests.

### 0.2 Explicit capability and fallback contracts — P0 / M

Replace frontend phrase-matching of memory-policy text with versioned machine-readable capability fields. Report requested device, actual provider, adapter, input cap, fallback cause, and whether inference succeeded. Preserve the distinction between a successful DirectML session and all operators executing on GPU.

Give unknown adapter memory a conservative documented policy. Add a visible retry/reset action and bounded lifetime for transient GPU-failure suppression; avoid repeated failing attempts on every frame. Keep explicit GPU mode strict. Clarify that Auto currently enables the cooperative pipeline too.

**Acceptance:** tests cover unknown memory, changed adapter/settings, transient failure, retry, and stale frontend/backend combinations. Existing Large-on-4-GB support remains available. Capability text and behavior agree.

### 0.3 Conversion preflight and documentation — P0 / S–M

Check model readiness, codec availability, dimensions, writable destination and estimated disk demand before expensive inference. Treat disk estimates as estimates. Document output timing/color limitations and reconcile the compatibility spec and critique snapshot with the current release.

**Acceptance:** failures identify a recovery action; unsupported inputs fail before model work where detectable; active jobs are not interrupted by diagnostics.

## Phase 1 — Improve speed where measurements justify it

### 1.1 Reuse preview depth — P1 / M

Cache a selected frame's inferred depth independently from stereo rendering. Strength, convergence and layout changes should reuse valid depth. Cache keys include source/frame identity, preprocessing, model hash, actual inference dimensions and relevant execution settings. Bound RAM use and invalidate precisely.

**Acceptance:** changing only stereo presentation runs zero additional inference calls; model/detail changes invalidate the cache; cached and uncached renders agree. Report measured interaction latency rather than promising a universal target.

### 1.2 Remove avoidable per-frame work — P1 / M

Profile buffer allocations, array copies, remap-grid construction, color conversion, resizing, and unused eye synthesis for DEPTH/RGBD output. Reuse shape-dependent buffers where ownership is safe. Keep queues bounded and preserve ordered temporal state.

**Acceptance:** an optimization is retained only with a repeatable end-to-end gain on an affected workload, no unexplained quality difference, and bounded memory. Confirm cancellation and ordering tests still pass.

### 1.3 Calibrate Auto mode and encoder choice — P1 / M

Offer an opt-in short calibration on representative frames, with a visible time budget. Compare useful model/detail/encoder combinations rather than assuming GPU or NVENC always wins. Store results by hardware/runtime/model and invalidate them when dependencies change. Show which stage limits throughput and let users override recommendations.

**Acceptance:** calibration includes its own overhead, never silently reduces output quality, and reports when there is insufficient evidence. Compare complete exports: encoder-only wins are not conversion wins.

### 1.4 Improve ETA — P1 / M

Separate initialization, conversion and finalization. Use a smoothed recent throughput estimate with warmup handling; display an approximate range when speed varies. Include retry/restart state and distinguish active-job ETA from queue ETA. Queue estimates can use historical comparable jobs but must show their uncertainty.

**Acceptance:** deterministic timing tests cover warmup, stalls, fallback, encoder retry, unknown counts and completion. No negative times or fabricated precision. A retry clearly explains why progress or remaining time changed.

### 1.5 Experimental adaptive depth sampling — P3 / L

Investigate inferring fewer frames and propagating depth with motion compensation and confidence checks. Reset at cuts and re-infer on uncertain motion. Keep disabled by default until evaluation succeeds; simple repeated depth maps can introduce trails and incorrect moving edges.

**Acceptance:** publish speed/quality comparisons on cuts, pans and fast objects. A speed gain without acceptable temporal output does not qualify for release as a recommended preset.

## Phase 2 — Improve stereo quality

### 2.1 Stable depth scale and temporal behavior — P1 / L

Evaluate scene-aware depth normalization to reduce frame-to-frame depth pumping. Add optional motion-aligned history with confidence rejection, disocclusion handling and cut resets. Compare against the existing appearance-gated filter; do not remove the cheaper path without evidence.

**Acceptance:** synthetic motion/cut tests validate history rejection; identical static scenes remain stable; visual review shows reduced flicker without persistent trails. Measure added CPU/GPU cost and memory.

### 2.2 Visibility-aware rendering and hole treatment — P1 / L

Add an optional higher-quality renderer using explicit visibility ordering and disocclusion masks. Compare forward splatting or another depth-aware method against the existing inverse warp. Provide bounded background-aware hole filling, edge handling and crop/border options. Start with deterministic methods before learned inpainting.

**Acceptance:** foreground/background fixtures verify visibility ordering; holes and edge stretching are measured and visually inspected; both eyes remain coherent. Missing surfaces are estimated, never described as recovered ground truth. Keep the fast renderer selectable.

### 2.3 Better refinement and stereo controls — P1 / M

Expose restrained depth-edge controls with useful defaults. Add near/far compression, a zero-parallax picker and a disparity visualization. Warn through an optional visual overlay when settings exceed the user's selected disparity budget. Avoid implying a universal medical comfort threshold.

**Acceptance:** zero strength produces identical eyes; eye order and near/far directions remain correct; clipping and odd dimensions are covered; controls have clear reset behavior. Evaluate actual display playback before claiming device suitability.

## Phase 3 — Make tuning and daily use pleasant

### 3.1 Preview workspace — P1 / L

Provide original/depth/stereo views, A/B comparison, zoom/pan, synchronized crops, selectable timestamps, and a short processed clip preview that uses export temporal settings. Clearly label proxy resolution and provide a full-detail still check. Cancel or supersede stale preview requests; avoid silently starving an active export of the single inference session.

**Acceptance:** A/B uses the same source frame; old responses cannot replace newer settings; clip preview exercises cut handling and motion. Test keyboard interaction and narrow layouts.

### 3.2 Settings and presets — P1 / M

Separate Basic and Advanced controls. Ship named Fast, Balanced and Quality starting points, plus a conservative 4 GB profile, with exact settings visible. Add per-setting speed/quality explanations, actual effective input size, reset-to-preset, unsaved-change indicators, and user preset import/export with schema validation.

**Acceptance:** presets round-trip; invalid or obsolete fields are handled explicitly; users can see when hardware caps alter requested settings. Avoid unexplained star ratings or invented speed multipliers.

### 3.3 Queue and quality of life — P1 / M

Add reorder/remove queued items, retry failed jobs, duplicate with changed settings, per-job settings snapshots, batch apply, and output naming templates with collision protection. Make active device, encoder, stage and fallback reason readable without expanding technical logs. Add completion notification opt-in, keyboard shortcuts, accessible focus/labels and non-color status cues.

**Acceptance:** queued edits never mutate an active job; retries preserve source/settings and use distinct output paths; selection survives refresh; cancel remains reachable on narrow screens. Clarify browser download versus opening a local output folder, which may require a desktop helper.

## Phase 4 — Preserve work and avoid repeated inference

### 4.1 Durable job/project records — P2 / L

Persist job metadata and validated settings, with a versioned project format and source fingerprints. Restore history after restart, marking interrupted work honestly. Add pause at safe boundaries; implement true resume only after chunk/checkpoint design is proven.

**Acceptance:** crash/restart cannot mark partial output complete; missing/changed sources are identified; output paths cannot escape intended destinations. Test migration and corrupted state recovery.

### 4.2 Optional reusable depth cache — P2 / L

Store depth separately from presentation so users can render multiple layouts or stereo strengths without repeating inference. Define precision, normalization metadata, frame timestamps, compression, cache version and invalidation rules. Include storage estimates, a quota and explicit cleanup controls.

**Acceptance:** rerender avoids inference; stale or partial caches are rejected; cached output is compared with uncached output. Quantify disk cost before enabling by default. Cache keys must include temporal/preprocessing choices appropriate to the stored stage.

### 4.3 Chunked recovery — P2 / L

Checkpoint at validated segment boundaries. Preserve timestamps, audio alignment and temporal context through overlap or saved state; finalize output atomically. Avoid claiming arbitrary frame-level resume for a monolithic MP4 stream.

**Acceptance:** interrupted/resumed output matches uninterrupted processing within documented tolerances, with no duplicated frames, boundary flashes or audio discontinuities. Restart and cancellation tests leave no orphan encoder processes.

## Phase 5 — Customization, models and export fidelity

| Feature | Priority / effort | Scope and release gate |
|---|---|---|
| Timeline settings | P2 / L | Keyframes for strength/convergence and per-scene overrides; deterministic interpolation, cut behavior and project round-trip tests. |
| External depth input | P2 / M | Import depth images/sequences with explicit polarity, range and alignment; reject mismatched dimensions/timestamps; add higher-precision depth export. |
| Local depth editing | P2 / L | Masks, depth painting and protected regions; undo/redo and saved edits; prove alignment after resize/crop. |
| Custom ONNX GPU adapters | P2 / L | Explicit preprocessing, shape, tensor and depth-output contracts; validate on a sample before accepting a model; preserve CPU compatibility and strict GPU failure behavior. No arbitrary-ONNX compatibility claim. |
| Model-library improvements | P1 / M | Actionable download errors, retry/cancel, verified partial-download resume where server validators permit, disk checks, installed size/license/source, and repair/revalidation. Hash-check complete files before activation. |
| Model comparison | P2 / M | Same-frame A/B and short-clip evaluation with timing, actual dimensions and memory evidence; recommendations tied to the user's hardware. |
| Media fidelity | P2 / L | Audit VFR, rotation, aspect ratio, color range/primaries, audio stream selection and subtitle handling. Explicitly preserve, convert or reject each case; test with ffprobe and playback. |
| More output profiles | P2 / L | Evaluate HEVC, image sequences and high-quality intermediates against installed FFmpeg capability; show compatibility/file-size tradeoffs and verify actual output. |
| Device profiles | P2 / M | Editable dimensions, stereo layout and eye order; validate on the target player/display before advertising a verified profile. |
| Portable Windows delivery | P2 / L | Setup diagnostics, reliable launcher, runtime isolation, dependency/license inventory and clean-machine test. Model downloads remain explicit and verified. |

HDR, MV-HEVC/spatial export and subtitle depth placement each need a separate bounded design and real playback validation. They are not implied by adding an output label.

## Phase 6 — Gated research and acceleration

- **Alternate GPU backends:** compare CUDA ONNX Runtime or PyTorch only if profiling identifies a worthwhile bottleneck and a supported GTX-compatible runtime can be installed independently. Keep DirectML working. Include setup size, memory, CPU fallback, maintenance cost and end-to-end output comparisons.
- **Precision and graph optimization:** evaluate validated FP16/quantized exports per model and backend. Require numerical and visual quality checks plus measured speed/VRAM benefit; do not assume a lower precision format accelerates every GTX operation.
- **GPU rendering or decode:** prototype only after measuring CPU rendering/decode pressure. Account for host/device copies, synchronization and shared memory pressure with inference/NVENC. A faster isolated kernel is insufficient.
- **New model families:** evaluate temporally consistent video depth and stronger image-depth candidates through the adapter contract. Verify current licenses, weight provenance, runtime support and hardware requirements at selection time. Keep heavyweight options optional and label unsupported local configurations.
- **Learned hole filling:** require temporally stable, eye-consistent results and an explicit quality/speed/memory comparison; avoid per-eye hallucination differences.

Do not adopt unconditional FP16, periodic cache purging, extra concurrent GPU sessions, or tiled inference as generic speed fixes. Each can change behavior or add overhead and needs backend-specific evidence. In particular, splitting an image can lose global depth consistency and introduce seams.

## Delivery gates

Every implementation slice should include a bounded spec, affected files, semantic acceptance criteria, relevant automated checks and a short changelog. Keep benchmark outputs and environment details sufficient to reproduce claims. Run both CPU and GPU environment suites for shared inference/runtime changes and real hardware checks when making GPU claims.

Quality evaluation should combine synthetic geometry tests, matched visual comparisons and motion inspection. Where ground-truth depth is unavailable, do not present an image-similarity score as proof of correct depth. Compare speed at matched settings and report quality changes when settings differ.

For a release candidate, verify long conversion, queue operations, cancellation, model failure, encoder fallback, restart behavior, audio/timing, accessible UI, and output playback. Bound in-memory queues and cache sizes. Keep the service local by default; include request-origin/upload/path handling in API reviews and never put media or personal paths into diagnostics without explicit user action.

## Recommended first implementation batch

1. Add stage timings and reproducible benchmark output (0.1).
2. Replace policy-string checks; implement conservative unknown-memory handling and a visible GPU retry action (0.2).
3. Cache preview depth so strength/convergence/layout adjustments avoid inference (1.1).
4. Add phased ETA and clear fallback/retry status (1.4).
5. Build quality fixtures and compare the current renderer/filter before choosing the next quality algorithm (2.1–2.2).

This batch makes subsequent speed and quality decisions measurable while improving the frequent preview-and-adjust workflow. Start the larger UI and persistence work after these contracts settle. Reprioritize from observed bottlenecks and the user's most common footage rather than committing to speculative speedup percentages.
