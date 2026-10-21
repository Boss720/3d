# Production release readiness

Updated 21 October 2026. **Release readiness is not yet established.** The public
v0.1.0 remains a beta. This checklist records release requirements separately
from optional roadmap features.

The published v0.1.0 release points to commit `71c6d6e` and its Windows EXE was
published before the root MIT license was added. The build spec embeds
`LICENSE`, and the build script copies it beside the EXE. A private replacement
candidate was built at `.runtime/mit-license-packaging-20261021-01/dist/`:
98,572,977 bytes, SHA-256
`6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45`. The exact
1,064-byte MIT file is embedded and hash-matched; the extraction comparison is
retained at `.runtime/mit-license-packaging-20261021-01/license-package-audit.json`.
Packaged generated XREAL SBS
acceptance passed on the exact candidate; report:
`.runtime/mit-license-packaging-20261021-01/acceptance-retry-01/report.json`.
The public v0.1.0 binary remains unchanged; this private candidate is not yet
approved for publication.

## Verified in the current source wave

- Latest full-suite rerun against current source: CPU `617` tests, `10` skips;
  GPU `617` tests, `6` skips, both with `ResourceWarning` treated as an error.
  The runs were serial; their private test port was closed afterward. Logs:
  `.runtime/vfr-wave/resume-cpu-current-20261021.log` and
  `.runtime/vfr-wave/resume-gpu-current-20261021.log`. The existing CPU `.venv`
  remains on Starlette 1.3.1; separate fresh/overlay runs above cover the
  declared Starlette 1.8.0 floor and newer runtime resolution.
- Frozen builds carry a manifest with the same code fingerprint as source.
  Missing or invalid build identity disables persistent depth reuse.
- Pending jobs, application upload-copy reservations and session job records
  have limits of 32, 4 and 512. Rejection occurs before the application copies
  the upload. No completed record or output is silently evicted.
- Pipeline errors stop the owned encoder before joining a blocked render worker.
  Production software decode and preflight use owned FFmpeg processes with
  bounded startup/read and cancellation paths. Decoder cleanup failure now
  raises instead of reporting cancellation or starting a second fallback.
- Upload admission runs before multipart parsing. Both requirement files now
  require Starlette >=1.8.0 so disconnect/cancellation closes parser spools.
- A nondefault `--port` is supported; binding remains `127.0.0.1`.
- Full suites with `ResourceWarning` treated as an error: CPU 613 tests, 10
  skips on the unchanged `.venv`; GPU 613 tests, 6 skips; CPU 613 tests, 9 skips
  with an isolated Starlette 1.8.0 overlay. Logs:
  `.runtime/vfr-wave/full-cpu-final-current.log`,
  `.runtime/vfr-wave/full-gpu-final-current.log`, and
  `.runtime/vfr-wave/full-cpu-starlette-floor.log`. CPU and GPU suites ran
  serially because tests share `.runtime` data.
- A fresh CPU virtual environment resolved the declared runtime requirements,
  including Starlette 1.9.0. With test-only clients from `requirements-test.txt`,
  its complete suite passed: 613 tests, 6 skips, with `ResourceWarning` treated
  as an error. Log: `.runtime/vfr-wave/full-cpu-fresh.log`. The venv was created
  without bundled pip because `ensurepip` failed in this Python installation;
  the existing pip installed into that isolated interpreter. This still tests a
  fresh runtime dependency set. The original `.venv` remains unchanged.
- The repository CPU `.venv` still has Starlette 1.3.1. The official 1.8.0 wheel
  was unpacked into an isolated import overlay and the complete CPU suite passed
  against it, including parser spool cleanup. `pip --target` could not write to
  its Windows temporary directory (WinError 5), so this is a compatibility run,
  not a full fresh dependency installation. The GPU environment at 1.8.0 also
  passed the parser integration and full suite.
- Generated mixed-rate media preserves interior marker timing: source ordinal
  40 at 4.000 seconds appears at output frame 60 at 4.000 seconds in both eyes;
  measured maximum correspondence error is 66.7 ms. Audio/video start offset
  drift is 22 ms. A separate 2 FPS fixture preserves a 250 ms video start after
  audio. Report: `.runtime/production-media/20261021T100451Z-1e4bc3ac/report.json`.
- A generated 300.1-second CPU approximation export fully decoded to 3,001
  frames at 10 FPS, with 300.117 seconds of audio. This is not a neural,
  representative-footage, or wall-clock soak test.
- A private packaged EXE with the current Python source fingerprint passed a sequential failure-recovery
  run: it accepted a custom ONNX file, failed that first job during real DNN
  inference on an out-of-range Gather index, then completed an accepted Small
  CPU inference queued behind it. The 64x48 generated PNG produced a decodable
  128x48 output; the queue emptied and the same server PID stayed alive. Report:
  `.runtime/vfr-wave/neural-failure-recovery-20261021-retry4/report.json`.
  This proves recovery from the deliberate custom-model failure fixture, not
  from naturally occurring model corruption or sustained video processing.
- The exact current EXE archive was inspected and recorded in
  `RELEASE_DEPENDENCIES.md`. It contains 182 CArchive entries, 975 PYZ modules,
  and 97 native DLL/PYD entries. Its 22 named license/notice files belong to
  five archive owners; no separately named OpenCV/FFmpeg notice was found.
  The bundled OpenCV video-I/O DLL is distinct from the external `ffmpeg.exe`.
  The archive findings identify notice and provenance work; they are not a
  legal determination.
- The accessibility source review led to always-visible preview status,
  background queue completion/failure announcements, a single native import
  control, and a named About dialog. The focused frontend markup/lifecycle run
  passed 21 tests, including the new status and interaction contracts. These
  source checks do not prove screen-reader behavior. On the latest private
  candidate, Chromium passed the rendered checks for persistent status regions,
  a single native import control, About-dialog naming and Escape focus return,
  keyboard Browse activation, visible preview status while busy, and no page
  errors or horizontal overflow at 1280, 768, 390 and 320 pixels. Report:
  `.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-accessibility/report.json`.
  Packaged `Open output folder` and `Reveal file` actions have since been
  exercised on the exact candidate; both opened the isolated output folder and
  the generated file was visible. The Reveal selection highlight was not
  confirmed, and screen-reader behavior remains open. Save-copy byte parity and
  overwrite refusal have passed on the package.
  full `.venv-gpu` suite passed 617 tests with 6 skips before the final
  cache-token-only HTML update; the focused frontend suite and Node syntax
  checks passed after it.
- The exact packaged candidate also passed an `XREAL_SBS` run using generated
  media: 8 decoded frames at 3840x1080 (1920x1080 per eye), with audio. Report:
  `.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-xreal/report.json`.
  The generated sample is `acceptance-xreal/first-output.mp4`. This checks the
  export dimensions and decode; playback through physical XREAL 1S glasses is
  still unverified. XREAL's [One Series 3D Mode guide](https://tutorials.xreal.com/docs/glasses/one-series/osd/3d-mode/)
  specifies Full SBS as 3840x1080 without horizontal squeeze.
- The exact candidate passed an isolated idle Ctrl+C shutdown probe in normal
  Windows context. A real `CTRL_C_EVENT` reached the packaged app in a shared
  private console; it exited with code 0, the server process and listener
  closed, the private PyInstaller extraction directory was removed, and no
  forced cleanup was needed. The three pre-existing StereoLab process
  identities were unchanged. Report:
  `.runtime/graceful-ctrlc-20261021T165240Z-de8c94a9/report.json`. This closes
  the same-machine idle graceful-shutdown subgate, not clean-account setup.
- The same exact EXE completed a generated two-hour neural export: 7,200 frames
  at 1 FPS, all 7,200 inferred by strict Small DirectML GPU, fully decoded with
  two-hour audio, and zero audio/video start offset. It completed in 349.5 wall
  seconds. Twenty-three process samples had no errors and measured a 438.7–447.4
  MB working set. Report:
  `.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-synthetic-soak-normal/report.json`.
  This is a two-hour media timeline check. A separate two-hour same-process
  wall-clock soak on the exact candidate is recorded below. Natural-scene image
  quality remains unverified.

The earlier 1.333-second interior VFR drift was reproduced and fixed for the
tested mixed-rate fixture by timestamp-aware CFR resampling. The metadata
fallback unit test passes (provisional 30 FPS, unknown duration and frame count,
format-start fallback). A generated H.264 Annex B elementary-stream fixture
with no packet PTS or DTS also passes the production software decode and export:
12/12 frames, 12 FPS, 1 second, and complete decoder cleanup. Software and AMD
D3D11VA each decoded all 12 frames and cleaned up successfully. Their generated
test-pattern pixels differ slightly (mean absolute error 2.83/255; p95 15/255),
so bit-exact decoder parity is not claimed. The fixture has no audio and adds no
AV-sync evidence. Report:
`.runtime/vfr-wave/missing-pts-release-check/20261021T104229Z-fd20374f/report.json`.
Wider real-footage codec/driver coverage remains open. The retained failing report at
`.runtime/production-media/20261021T052646Z-f4f9cbce/report.json` is historical.

## Remaining correctness and resource gates

| Requirement | Required evidence / next action |
| --- | --- |
| VFR timing and audio alignment | Generated mixed-rate, nonzero-start, low-rate offset, CFR and missing-metadata checks pass. The no-PTS/no-DTS H.264 fixture passes 12-frame software export; software and AMD D3D11VA each decode all 12 frames. Real-footage AV-sync and broader codec/driver review remain open. |
| Stalled decoder and queued successor | The worker-level regression processed one frame, timed out on a stalled owned decoder, closed decoder/encoder processes and pipes, failed that job, and completed the next queued image job. A separate HistoryStore shutdown test proves its lease stays held while an active writer exits; the worker-level stalled test uses history disabled. |
| Multipart parser cleanup | Disconnect/cancellation spool cleanup passes under GPU Starlette 1.8.0, CPU Starlette 1.8.0 overlay, and a fresh CPU install at Starlette 1.9.0. The original repository `.venv` stays on 1.3.1. |
| Encoder and model failure recovery | Encoder cleanup and stalled-decoder next-job recovery pass. A preceding frozen EXE with the current Python source fingerprint also passed a deliberate custom-ONNX inference failure followed by queued named Small CPU inference; report: `.runtime/vfr-wave/neural-failure-recovery-20261021-retry4/report.json`. Recovery from naturally occurring model corruption remains unverified; generated sustained video results are recorded below. |
| Generated sustained neural work | A generated two-hour test-pattern video plus tone completed on the exact private EXE: 7,200/7,200 Small DirectML neural frames, full video/audio decode, 7,200-second output and audio, 0.000-second start offset, and 23 error-free process-resource samples. Wall time was 349.5 seconds; working set ranged from 438.7 to 447.4 MB. Report and output are under `acceptance-synthetic-soak-normal/`. A separate same-process two-hour wall-clock run passed on this machine: 7,200.047 seconds, 24/24 verified Small DirectML jobs, and 478 clean health/resource samples; report: `.runtime/wallclock-soak-normal-2h-retry-20261021/report.json`. Together these verify generated work and two-hour uptime on this host; natural-scene depth quality and real-footage AV sync remain unverified under the owner's generated-only scope. |
| Clean Windows installation | Same-host empty-profile startup and first-run Small model installation now pass on the exact Python 3.12 candidate. The fresh private profile started without a model root, the catalog reported not installed, and the install API downloaded 99,060,839 bytes in 24.046 seconds. Independent SHA-256 and pinned-revision receipt checks passed, then generated CPU neural inference produced a decoded 192x64 SBS PNG. Report: `.runtime/first-run-model-download-20261021-a1/probe/report.json`. No candidate process or private listener remained and extraction temp was empty afterward. Ctrl+Break returned code 3, so this probe does not establish clean application exit; the separate idle Ctrl+C code-0 check remains valid. Clean-account/machine installation remains open. These results do not transfer to the separate Python 3.14 candidate. |
| Accessibility and native actions | Source contracts and the current candidate's Chromium checks pass for status regions, keyboard Browse, About-dialog naming/focus return, and visible busy preview status. Packaged save-copy passed byte-equality and overwrite-refusal checks. On the exact candidate, `Open output folder` and `Reveal file` both opened Explorer to the isolated output directory, where the generated output was visible. Evidence: `.runtime/release-explorer-ui-20261021-normal2/report.json` and `.runtime/release-explorer-ui-20261021-normal2/explorer-visible.png`. The Reveal selection highlight was not confirmed; screen-reader behavior remains open. The generated `XREAL_SBS` export has the expected 3840x1080 geometry; physical playback on the owner-selected XREAL 1S remains open. |
| Project and third-party licensing | The owner directed broad public reuse; the repository now licenses original StereoLab source and documentation under MIT in the root `LICENSE`. This resolves the project source-license selection, within the rights held by the project. It does not grant rights to third-party files or models and does not approve executable redistribution. Exact inspected-candidate audit: all 22 archived notice files match their installed sources; 31 additional package/runtime texts were originally mapped. The draft has review view r4 with 54 unique byte-verified payloads and all 62 source aliases preserved at `output/release-notices-deduplicated-review-2026-10-21-r4.zip` (SHA-256 `c593d4317079f6195f8fa085718d42ae2ad4d1227abc733541a4cc7d772b0778`). Its aliases refer to 60 distinct paths: two OpenSSL/libffi aliases point to the CPython aggregate copy, and one FFmpeg-LGPL 2.1 preamble excerpt comes from OpenCV's pinned source. Earlier r1-r3 ZIPs remain unchanged. Neither artifact is approved or in the build. Ten setuptools vendored aliases now preserve eight package/version labels from local dist-info metadata; module roots appear in the candidate PYZ but frozen-code version equivalence remains unverified. The draft includes `setuptools/config/_validate_pyproject/NOTICE`, which names upstream components for generated validator modules present in the candidate PYZ. The separate parent `setuptools/config/NOTICE` names schema JSON files absent as standalone archive entries; whether generated code also requires that parent notice remains open. All 97 native DLL/PYD entries match local source copies. The pinned model pages were rechecked on 2026-10-21: Small is listed as Apache-2.0; Base/Large as CC-BY-NC-4.0; Intel's upstream DPT-Hybrid/MiDaS card says Apache 2.0, while the Xenova conversion page does not visibly declare its own license. Owner/reviewer must confirm conversion terms and whether exposing non-commercial models fits the intended delivery model. Remaining: Microsoft runtime source-channel/entitlement; OpenSSL and libffi upstream build inputs and notice applicability (24 native entries now match the published CPython 3.12.10 ZIP and the existing aggregate license matches its official copy; the earlier local-only package evidence is superseded for that subset); OpenCV/FFmpeg/GCC/Winpthreads object attribution and remaining notice-scope review; DirectML source/notice review: the candidate DLL matches the published PyPI wheel and its `RECORD`, but differs from the NuGet 1.15.4 DLL; the wheel notice files omit a DirectML mention, so applicability of DirectML terms/notices needs owner/reviewer resolution; exact native transitive attribution; owner review of the pinned model terms and Xenova conversion; and approval of a complete notice bundle. SQLite's embedded source ID matches the official 3.49.1 release record and its candidate bytes match the official CPython ZIP. Upstream says its deliverable code is public domain; historical acquisition and upstream build inputs remain unverified. See `RELEASE_DEPENDENCIES.md`; no legal conclusion is made. |
| Release operations | Rebuild from the final reviewed source, record tool versions, checksum the exact tested artifact, document support limitations and reproducible commands. Publish only with release authorization. |

The candidate-specific runtime review has now confirmed the exact local byte
matches for all 48 root VC/UCRT/API-set files, plus NumPy's separately bundled
MSVC runtime and DirectML 1.15.4. Authenticode signatures are valid on the
matched local copies, and the DirectML 1.15.4 NuGet terms have been located.
These checks identify the binaries; they do not establish original Microsoft
package provenance, applicable VC 2026/UCRT terms, or the owner's distribution
entitlement. The project's license and the intended distribution model remain
owner decisions. See `RELEASE_DEPENDENCIES.md` for hashes, versions, source
references and the remaining attribution review.

A targeted local provenance recheck on 2026-10-21 independently matched all
48 candidate root CRT/API-set archive entries to the Codex `native/libheif`
runtime files. The installed VC redistributable is 14.44.35211.00, not the
candidate's 14.51.36247.0 VC files. The installed SDK redist has 44 matching
names, but its 10.0.26100.7705 files do not hash-match the candidate's
10.0.26100.4654 UCRT/API files. Local Codex metadata does not bind those bytes
to a Microsoft package channel or agreement. Microsoft source-channel and
redistribution entitlement therefore remain open.

## Current artifacts

The current candidate's review notes, support boundaries, re-run commands and
remaining manual acceptance actions are assembled in
`output/release-review-2026-10-21/REVIEW.md`. Its adjacent `evidence-index.json`
records independently checked candidate/source/static identities and hashes of
the retained acceptance reports. These materials remain pending owner approval.
The adjacent private source snapshot preserves the source/build/test inputs
with per-file hashes. The runtime manifest covers 26 declared source files;
`gpu_hardware.py` is also preserved and its compiled code matches the archived
module after build-path filename normalization. This additional comparison
does not change the runtime manifest's coverage.

The current private Windows candidate is
`.runtime/mit-license-packaging-20261021-01/dist/StereoLab.exe`,
98,572,977 bytes, SHA-256
`6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45`.
Its embedded Python-source fingerprint matches:
`8e56afce8aa2240e94a07f27bc093186939dd3ed0302c72976a51baa76d9e2fe`. That
fingerprint covers the Python components listed in `build_identity.py`, not
`static/`; the separate packaged acceptance compared all nine static asset
hashes with current source.

The latest same-machine package report
`.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance/report.json`
passes current static asset hashes, strict named Small DirectML inference,
truthful cached CPU provenance, exact cached encoded/decoded/audio parity,
source reupload, forced exact-owned-process restart with history/output/source
identity, depth-cache restart reuse, and selected cleanup preserving another
entry. The generated source is 8 frames with audio. The report has no cleanup
errors. A separate Chromium run of the same candidate passed the browser checks
listed above, with no page errors or overflow and no cleanup errors. A second
isolated package run also proved that save-copy writes bytes identical to the
completed output and refuses to overwrite an existing destination; report:
`.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-copy/report.json`.
The candidate's owned test processes and loopback ports were cleaned up.

The same candidate also passed a generated `XREAL_SBS` export with strict named
Small DirectML inference and the package checks recorded in
`acceptance-xreal/report.json`. The output is 3840x1080, contains eight decoded
frames and audio, and is available at
`.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-xreal/first-output.mp4`.
This is software/file evidence only; no physical XREAL 1S playback has been
observed.

The exact candidate completed the two-hour synthetic neural export recorded at
`.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-synthetic-soak-normal/report.json`.
It processed 7,200 neural frames with strict DirectML GPU, decoded the complete
384x108 SBS video and two-hour audio, and retained a 113,751,452-byte output at
`.runtime/vfr-wave/frontend-a11y-final-20261021/acceptance-synthetic-soak-normal/synthetic-neural-soak-output.mp4`.
Output audio and video both span 7,200 seconds and start together. The run took
349.5 seconds and recorded 23 process resource samples with no sampling errors;
working set ranged from 438,710,272 to 447,377,408 bytes. The child PATH contained
only Windows system directories and the explicitly configured FFmpeg directory.

The preceding private fingerprint-matched EXE also passed the isolated model-failure
recovery report cited above. Its first custom-model image job failed during
inference; the queued accepted Small CPU job then succeeded and its PNG decoded.
The EXE was launched with a private LOCALAPPDATA and temp directory, and the
owned process and port were stopped after the run.

These are same-machine packaged checks with external configured FFmpeg and
copied accepted model data, not clean-machine proof. The exact candidate later
passed the two-hour same-process wall-clock soak below. Its teardown forcibly
terminated the exact owned process identity, so that run did not prove graceful
shutdown. A separate idle Ctrl+C probe now passes on this host as recorded above.
Natural-scene quality and real-footage AV sync remain unverified by the owner's
generated-only scope. The initial sandbox extraction failed on
`MSVCP140_1.dll`; the normal Windows-context retry passed. A first retry safely
refused a changed process identity; reinspection found the recorded PIDs
absent, and a fresh isolated retry passed.

The exact candidate has now also passed a separate 600.031-second same-process
wall-clock smoke in normal Windows context: five uncached strict Small DirectML
jobs, 119 clean health/resource samples, no failed jobs, and no cleanup errors.
Working set ranged from 446,898,176 to 448,122,880 bytes; sampled process peak
was 449,167,360 bytes. Report:
`.runtime/wallclock-soak-normal-smoke-20261021/report.json`. A distinct
The exact candidate then passed a 7,200.047-second same-process wall-clock run
in normal Windows context at
`.runtime/wallclock-soak-normal-2h-retry-20261021/report.json`. All 24 uncached
generated strict Small DirectML jobs verified NVIDIA GTX 1650 Ti inference;
the 478 health/resource samples had no errors. Sampled working set ranged from
416.6 to 425.0 MB (sampled process peak: 426.1 MB). All output-copy byte checks
and overwrite refusals passed, and exact-process cleanup reported no errors.
The first attempt was blocked before app startup by the sandbox's
`MSVCP140_1.dll` extraction permission error; that failed sandbox attempt is
recorded separately and is not counted as a candidate runtime failure. This
closes the two-hour same-process uptime gate on this machine only; it does not
prove clean-machine installation, screen-reader behavior,
physical XREAL 1S playback, natural-scene depth quality, or real-footage AV sync.

Do not overwrite the running user EXE or republish v0.1.0 as part of a check.
The existing saved-project and depth-cache changes remain uncommitted. The
release goal is still open; do not replace the user EXE or publish v0.1.0 during
validation.

## Private CPython 3.14.9 candidate review

A separate compatibility candidate passed a full GPU-environment suite (617 tests run, 6 skipped) with `ResourceWarning` treated as an error. Its packaged smoke used only isolated data/temp roots and generated media; the exact EXE identity stayed unchanged. Generated image output decoded at 3840x1080. A generated one-second video export completed with four 3840x1080 frames and 1.002-second audio. It used OpenCV CPU visual approximation and reports `gpu_verified=false`; NVENC was the video encoder only. Evidence: `.runtime/python314-compatibility-review/launch-smoke-20261021-3a912cf2/generated-video-smoke-report.json` and `image-smoke-report.json`. This is generated pipeline evidence, not natural-scene quality or physical glasses playback.

The private EXE is 99,782,837 bytes, SHA-256 `e5afd208495e9b1690fb0bd75cefb692e367195f2e9c8761060595cbf79b190e`; its source manifest fingerprint matches the checkout. It is not promoted over the recorded Python 3.12 candidate. The candidate-specific notice delta is `output/release-notices-deduplicated-review-2026-10-21-python314-delta.zip` (SHA-256 `2e813c2ab8fcd300a8a32bf20bb5c3c2314b9f9070c7bfe965ff66911136806`). It is explicitly incomplete and nests the prior r4 review ZIP as a reference. All 22 named notices matched r4 by name/hash, but Python 3.14 changes the native inventory to 98 entries (17 added, 16 removed, 26 changed, 55 unchanged) and needs a full refreshed component/notice review. Its exact CPython 3.14.9 license is included in the delta package; `_zstd.pyd` is newly present.

Remaining candidate-specific release blockers include refreshed native and package source/version mapping; DirectML terms/notice applicability; Microsoft CRT/UCRT provenance and redistribution entitlement; exact OpenSSL/libffi and OpenCV/FFmpeg/GCC/Winpthreads attribution; model terms and delivery-model choice; allowed-path first-run model acceptance/download; clean-account installation; screen-reader behavior; physical XREAL 1S playback; and an owner-reviewed final release artifact, checksum, notes, and authorization. The targeted generated video smoke exited after Ctrl+Break with code 3; process/listener and extraction-temp cleanup passed, but clean application exit is not claimed. The Python 3.12 candidate's previous gates do not automatically transfer to this compatibility candidate.
