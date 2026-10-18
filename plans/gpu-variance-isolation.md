# GPU output-variance isolation — bounded diagnostic experiment

Frozen diagnostic-only lane; no production pipeline/model/encoder default changes.
Follow strict pixel gate from prior failed/inconclusive report, never loosen it.
Use existing generated 3s36frame320x180/12FPS/audio matched proxy, pinned Small at256,
explicit GPU, max3 repeats, one owned model session. Check known local servers
8768/8770/8772 idle before phases; if another job starts, stop new work and report
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
