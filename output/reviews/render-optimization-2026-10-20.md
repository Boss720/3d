# Rendering allocation optimization — 2026-10-20



Implemented in `stereo.py` while preserving the existing XREAL dimensions, aspect fit, eye order, rounding, interpolation, and black padding.



- `StereoRenderer.render(frame, depth, strength, convergence, layout)` owns coordinate grids, disparity, and map scratch space per job. Scratch storage replaces itself when source dimensions change. One instance must belong to one processing thread.

- Returned images remain independent allocations; rendering a later frame cannot overwrite earlier results. Public stateless helpers remain available.

- XREAL packing allocates one combined output canvas and resizes directly into the appropriate eye region. It avoids two eye canvases plus concatenation and intermediate resized arrays.



## Integration



Import `StereoRenderer` in `main.py`. Construct it once in the video job processing scope and use its `render(...)` method at the video rendering call. Keep still/preview callers on existing `render_output` unless they have a clearly owned single-thread context. This slice does not edit `main.py`.



## Verification



`.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_render_buffers.py`: 3 tests passed in 2.785 seconds. Coverage includes every layout, strength 0/4.5/10, nonfinite depth, source shape changes, scalar/depth validation, unmodified source/depth arrays, and output retention after subsequent renders. The same 3 regressions passed in `.venv-gpu` in 2.697 seconds; the existing 14 stereo tests passed in `.venv` in 0.708 seconds. Scoped `git diff --check` passed. Independent legacy XREAL packing matched exactly for tiny, odd, portrait, ultrawide, and downscaled source dimensions.



`.venv\Scripts\python.exe -B scripts/benchmark_render_buffers.py --iterations 24`: OpenCV 5.0.0, explicitly one CPU thread, deterministic synthetic arrays, three warmups and alternating old/new timed order. Full output pixel equality checked before timing. Medians below are renderer-only measurements, not model inference or complete exports.



| Source | Layout | Previous ms | Optimized ms | Ratio |

|---|---|---:|---:|---:|

| 1280×720 | SBS | 27.389 | 23.674 | 1.157× |

| 1280×720 | XREAL_SBS | 50.702 | 38.141 | 1.329× |

| 1920×1080 | SBS | 70.900 | 59.693 | 1.188× |

| 1920×1080 | XREAL_SBS | 68.862 | 56.419 | 1.221× |



The benchmark retains the previous eye-warp implementation as `make_eye_pair`; it uses an independent copy of legacy XREAL packing from the regression module. Results include both grid reuse and XREAL packing changes where applicable. Scratch float arrays remain allocated for the job lifetime, trading bounded retained memory for less allocation churn. No claim is made about GPU inference speed, whole-export speed, peak RSS, or other machines/thread configurations. No EXE was built.





## Real CPU export comparison



Ran `scripts/verify_render_exports.py` through actual `main.run_video` six times on the same generated 12-frame 1280×720, 12fps testsrc2 clip with sine/AAC audio. Used balanced approximation depth, temporal smoothing 0.35, edge refinement, strength 5, convergence 0.5, XREAL 3840×1080 output, CPU libx264 ultrafast CRF22, and one OpenCV/encoder thread. Forced software profile avoids NVENC probing. The old path explicitly uses `make_eye_pair` and independent legacy XREAL packing. The new path constructs one `StereoRenderer` per job through the real video implementation.



| Trial | Order | Old wall seconds | New wall seconds | Old render seconds | New render seconds |

|---|---|---:|---:|---:|---:|

| 0 | old, new | 2.955776 | 2.368541 | 0.608161 | 0.418172 |

| 1 | new, old | 2.076937 | 2.087966 | 0.480053 | 0.380387 |

| 2 | old, new | 2.299687 | 1.980479 | 0.508931 | 0.365011 |



Median full `run_video` time decreased from 2.299687 to 2.087966 seconds (1.1014× throughput, 9.21% lower wall time in this sample). One pair slightly regressed, and the first pair contains startup effects. Three short pairs establish only a bounded local observation, not a robust whole-film or neural/GPU speed claim. Median render-stage time was 0.508931 versus 0.380387 seconds. Wall timings include capture metadata/ffprobe, decode, approximation depth, refinement/temporal filtering, rendering, encoding, audio and encoder completion; they exclude model loading, HTTP upload, history admission and process_job cleanup.



All six MP4 files were byte-identical. Every file independently decoded to 12 frames at 3840×1080, and audio decoded successfully to 96,256 PCM bytes. File SHA256: `01088d1848c02d040121f5998c559a93e601f8910a55d0e577f04541528c7ce1`; decoded pixel SHA256: `1fd309876359710224df5e838f1dadf97460dd40d1c308f915ee6669a99a3447`; decoded audio SHA256: `b2f66d33815b74977ac292ceaae7457f09966318e087dd4c1c33bf4c8f9157f4`.



Raw JSON, source, and all six exports retained in `.runtime/render-export-verification-c2e01f03d52d4b7987d4f35a01af0199/`. No live server, GPU, model downloads, or deletion were used.



## Coordinator independent repeat

Repeated `scripts/verify_render_exports.py` after both full test suites finished.
Median old/new complete run_video wall time: 2.073524/1.942392 seconds
(6.32% lower wall time in this separate short CPU sample).
All six encoded files, decoded frames and decoded audio matched exactly.
Evidence: `D:\Users\Hp\Documents\ChatGPT\3d\.runtime\render-export-verification-c301c1ee49be44bb9530e7d354f795df` and `.runtime/qol-render-exports-independent.log`.
This confirms a local repeat improvement for this generated approximation/XREAL
workload; it does not establish whole-film, neural-inference or GPU gains.
