# GPU output-variance isolation — bounded diagnostic experiment

Frozen diagnostic-only lane; no production pipeline/model/encoder default changes.
Follow strict pixel gate from prior failed/inconclusive report, never loosen it.
Use existing generated 3s36frame320x180/12FPS/audio matched proxy, pinned Small at256,
explicit GPU, max3 repeats, one owned model session. Check known local servers
8768/8770/8772/8774 idle before phases; if another job starts, stop new work and report
interference, never cancel/restart another server. Other processes are not excluded.

Capture earliest feasible observable tensors: optional session.run raw ONNX output
through diagnostic-local session wrapper if safe, plus normalized engine depth,
refined/temporally filtered depth and raw rendered BGR frames. Reuse production
ModelManager/refine_depth/TemporalDepthFilter/render_output; fresh temporal state
for each repetition. If raw ONNX capture is impractical, explicitly label earliest
stage normalized model depth (not raw ONNX output). Validate finite arrays and
provider/gpu_verified/no fallback, exact identical source-frame digest and settings.
Record stage hashes, exact-equality, changed values, max/mean absolute differences,
per-frame/sample metrics and source/model/runtime/hardware fingerprint.

Encode ONE FIXED rendered frame sequence twice with CPUlibx264 and, only if current
FFmpeg probe supports it, twice with NVENC using production fast profiles. Compare
raw frame bytes and decoded outputs per encoder separately. No cross-codec exact
claim. Fixed input variants isolate encoder repeatability; independently varied
inference outputs isolate earlier drift. Resource sampling optional and bounded.

Keep artifacts under .runtime/variance-isolation/unique run. No model download,
no main.py instrumentation/API changes, no unbounded capture, no neural parallelism.
Report conclusions as observed stage localization, not proof all GPU nodes or all
hardware/codecs are deterministic. Preserve raw failed gate. CLI defaults CPU,
requires explicit --device gpu for GPU; timeout subprocesses and clean only owned
partials. Unit tests semantic CPU utility/capture/metrics/guard behavior. Diagnostic
runner must permit independent stages to finish after nonidentity and record result.

## Follow-up measurement contract

Add preprocessed session.run input capture before delegating to ORT, preserving
input names/shapes/dtypes/values with finite numeric and per-frame/total byte bounds.
Compare all repeat pairs, including2-vs3, rather than comparing only to repeat1.
Add explicit diagnostic --warmup-frames0/1 (default0): warmup is separately counted,
provider-verified and excluded from the36-frame temporal/export sequence. It never
changes production ModelManager/session defaults or hides cold-start evidence.
Run fresh cold and one-frame-warmed processes sequentially, with CPU control and
known8774 activity included in guards. No concurrent neural experiment.

An independent artifact analyzer verifies report/file/stage hashes before trusting
captures, compares exact tensors and all later stages, and optionally encodes EACH
captured rendered sequence under the saved per-codec fast args. Cross-inference
encodes are compared using the unchanged GPU max<=1,mean<=.001 decoded policy;
fixed-input encoder evidence and cross-codec comparisons stay distinct. Bounded
inputs (three captures, <=36frames320x180->640x180, byte limits), new unique output
folder, finite subprocess timeouts and exact owned cleanup. A diagnostic result is
not a production processed-clip acceptance result. Preserve prior failed gate.

## Follow-up outcome

CPU control `followup-cpu/20261018T221813Z-9c4215cd`, cold GPU
`followup-cold-gpu/20261018T221849Z-cb7c88c7`, and warmed GPU
`followup-warm-gpu/20261018T221919Z-cab0494f` live under
`.runtime/variance-isolation/`. All exact preprocessed inputs matched across
repeat pairs. CPU stages were exact. Cold GPU raw maximum was2.861e-6 with up to9
rendered color values changing by1; warmed GPU still varied in repeat3, raw maximum
2.623e-6 and29 rendered changes by1. Warmup is not promoted to production.

Independent per-repeat encode analyses are under `.runtime/variance-capture-analysis/`:
CPU `20261018T222038Z-fedac2e7`, cold `20261018T221953Z-2d17dda0`,
warm `20261018T222013Z-bde0bd39`, legacy `20261018T222030Z-d1046a73`.
All tested NVENC decoded pairs were exact. Some libx264 cold/warm pairs exceeded
the unchanged strict policy (max10/19; warmed mean.005891). This demonstrates
encoding amplification in these experiments, not the cause of the old NVENCmax29.

The corrected API verifier now requires all three pixel pairs, retains inconclusive
self-repeat precedence and fails without a public execution snapshot. Fresh
private GPU API verification passed all three 36-frame comparisons exactly:
`.runtime/clip-verification/consistency-followup/20261018T222230Z-0e510fba/report.json`.
Full source/clip/ordinary exports and cancellation were exercised; the generated
fixture, API cancellation and metadata-only proxy correspondence retain their
existing limitations. Owned API stopped after5 terminal records were saved.
Original failed report was not overwritten. Production files/defaults unchanged.
