# StereoLab review fixes - 20 October 2026

All five findings from the review are implemented. Existing XREAL/UI edits,
benchmarking changes, renderer changes and saved mockups were preserved.

- Cleanup failures now retain the affected paths and show a queue warning without
  changing a completed, failed or canceled conversion into another outcome. An
  unexpected job exception terminalizes that job and the worker continues.
- History capacity checks perform reads only. Eviction and admission remain in
  one SQLite transaction, so a rejected upload retains prior history/downloads.
- Preflight and video export share finite metadata normalization. Unknown frame
  counts remain unknown, and the capture is released even if metadata or encoder
  setup fails.
- Removing a selected source invalidates pending preview and calibration state.
  Stale success/error responses cannot reappear after removal.
- Failed queue restoration retries with backoff capped at 30 seconds. Refresh
  status also refreshes jobs; successful recovery restores the normal empty state.

Independent review also found an existing FFmpeg setup leak when starting its
stderr reader failed. The encoder is now stopped on that setup failure. A new
test failed before the guard and passes afterward.

Verification against the final source:

- CPU environment: 435 tests in 13.060 seconds, OK with two existing skips.
- GPU-runtime environment: 435 tests in 12.764 seconds, OK with two existing skips.
  This is environment compatibility coverage, not live GPU inference.
- Regression coverage includes a real Windows open-upload handle, continued queue
  processing, accurate terminal outcomes, retained cleanup warnings, a failed
  upload at history capacity with an unchanged downloadable prior output,
  transactional admission rollback, unknown video totals and encoder setup errors.
- Seven frontend lifecycle cases execute the real application closure with
  controlled network/timers and a DOM fixture. Rendered-browser behavior was not
  reverified in this implementation slice.
- Dedicated XREAL_SBS tests cover 3840x1080 packing, eye order, aspect-preserving
  centering, padding, input preservation and up/downscaling.
- Python syntax passed for 69 files, both JavaScript syntax checks passed, and
  git whitespace checks passed.

A generated real CPU/libx264 export used actual decoding/rendering/encoding with
NaN frame-count metadata injected around the real decoder. All six frames decoded
at 128x48 and 6 FPS; audio decoded successfully, the source upload was removed,
and unknown total/ETA remained unset. No model download or neural/GPU workload
was needed. The subsequent stderr-setup guard was covered by both final suites.

Full suite logs:

- `.runtime/review-fixes-verification/cpu-tests-final.log`
- `.runtime/review-fixes-verification/gpu-env-tests-final.log`
- `.runtime/review-fixes-verification/cpu-video-8b303b0bcb2f49549f260aea024b8417/report.json`

Original snapshots are retained under
`.runtime/review-fixes-baseline-266f4dfbb424440e98be6a2dd5d7710b/`.
No commit or executable rebuild was performed.
