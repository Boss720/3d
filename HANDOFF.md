# StereoLab handoff — 24 October 2026

## Current beta.6 package checkpoint

The current-source beta.6 source/build/test changes were committed as
`2391cf590a3c397b4e9e88450d5169e781163cbc` and pushed to
`origin/codex/linux-cpu-build`. The remote branch head was verified against this
commit. The commit contains 44 source, build, release-documentation, and test
files. `PROGRESS.md` and existing `output/` reports/assets were deliberately
left local and uncommitted.

Both Windows packages were rebuilt in fresh ignored directories from source
fingerprint
`459dbc29681f15b5b264911e0b86494e10f53fea641faf9618143a23a2d27451`:

| Artifact | Local path | Bytes | SHA-256 |
|---|---|---:|---|
| DirectML EXE | `.runtime/windows-directml-2026-10-24/dist/StereoLab.exe` | 98,650,993 | `278e0845f08270a6be96cb959d7fc1ae0b1f89b3a430f0823678feec5a57d162` |
| DirectML installer | `.runtime/windows-directml-2026-10-24/installer/StereoLab-0.2.0-beta.6-Setup.exe` | 99,844,534 | `6824aa3e422cbd71f021725ffdddb2d12ea9e99672c62b37e38e3b18c183905f4` |
| CUDA EXE (one-folder package) | `.runtime/windows-cuda-2026-10-24/dist/StereoLab/StereoLab.exe` | 8,703,055 | `fcc7f13f014d1b836ec9f033068c98871dee610cd1aac688bbdf703381c07e0c` |
| CUDA installer | `.runtime/windows-cuda-2026-10-24/installer/StereoLab-CUDA-0.2.0-beta.6-Setup.exe` | 1,356,881,500 | `f2f5ffc7a123f23e70631eff954fe14194663273d58c839f8f3aa4617888cd5b` |

Current-source tests reported 774 passed, 10 skipped, and 530 subtests passed.
The DirectML installer passed installed-path packaged acceptance, matched all
3 payload hashes, and uninstalled cleanly. The CUDA installer passed strict
CUDA packaged acceptance, matched all 200 payload hashes, and uninstalled
cleanly. Both were verified on a Windows 11 GTX 1650 Ti; other GPUs and
physical XREAL playback were not tested. Evidence is retained under each
matching `.runtime/windows-*-2026-10-24/` directory.

The source commit is pushed, but no GitHub release/tag or binary assets were
published. `gh auth status` reports invalid CLI credentials; Git push succeeded
through the configured Git transport. If binary publication is still wanted,
reauthenticate GitHub CLI and create the beta.6 release from these verified
local assets. Check dependency redistribution requirements before uploading the
CUDA installer.

The separate Phase 0 report is
`output/reviews/stereolab-performance-baseline-2026-10-24.md`; it remains local.
It recommends profiling full-resolution `refine_depth` first and records that
the benchmark clips were synthetic. No Phase 1 performance optimization was
implemented in this packaging pass.

---

## Earlier handoff — 21 October 2026

## Current beta.2 review checkpoint

This section supersedes the older beta.1 publication status and candidate identities below. Live GitHub inspection confirmed beta.1 was already public. The owner has now authorized review/fixes, rebuilt binaries, a scoped commit/push, and a new beta release.

The current wave fixes late pipeline cancellation, custom-model candidate cleanup, and missing GPU hardware code in the build fingerprint. It adds isolated build paths, updates installer defaults to `0.2.0-beta.2`, fixes the FFmpeg 9 rotation fixture, and makes hardware-encoding variation an explicit measured verifier option. See `RELEASE_NOTES.md`.

Evidence is local under `.runtime/review-release-beta2/`: CPU/GPU full suites each run 628 tests (7/3 skips); packaged `acceptance-final/report.json` passes strict Small DirectML, generated XREAL SBS, cache/restart/history, static hashes and browser checks. `final-media-postcheck.json` verifies exact audio timing/format and video timing; the final repeats were byte-identical. `source-parity/report.json` separately proves identical eight-frame fresh/cache raw render hashes. `nvenc-timing-analysis/` records independent intermittent lossy-encoding variation. Installer install, hash comparison, HTTP launch and uninstall passed in `installer-acceptance.json`.

The selected EXE is `.runtime/review-release-beta2/dist/StereoLab.exe`; embedded source fingerprint `26d8375445e80934d4ba230d8cfa2f1b499a650c4e96e857da096905326ef86a` covers 27 modules. Release assets and checksums are assembled in `.runtime/stereolab-release-candidate-0.2.0-beta.2/`. Existing user processes, output review archives and local PROGRESS changes are preserved.

This remains a beta with the previously documented clean-machine, physical-device, screen-reader and dependency review limits. No production-readiness or third-party relicensing claim is made.

The owner has requested a new release and directed broad public reuse; original
StereoLab source and documentation are now under MIT in the root `LICENSE`.
Public binary publication still waits on third-party dependency notices and
redistribution rights, model terms, and remaining release-readiness checks.

The existing public v0.1.0 binary predates the root MIT license and does not
carry that notice. `StereoLab.spec` now embeds `LICENSE`; `build-exe.ps1` also
copies it beside a fresh build. The old public binary has not been replaced.

## Source checkpoint and scope

- Repository: `D:\Users\Hp\Documents\ChatGPT\3d`; `master`; remote: `https://github.com/Boss720/3d.git`.
- Source checkpoint: the commit containing this handoff; use `git log -1` and
  `git log -1 -- HANDOFF.md`. Remote integration base is `origin/master`
  `3c1f09e1dd51646ed4db2b22cc7bdd7ea0302024`, preserving its README support link.
  Push verification compares local HEAD with `git ls-remote --heads origin master`;
  the final verified hash is recorded in chat and local `PROGRESS.md`.
- The coordinator is preserving the existing feature code and committing the
  scoped application, tests, verifier/build files, and root release documents.
  `PROGRESS.md`, `.runtime/`, review ZIPs, and other `output/` artifacts stay local.
  Paths below therefore do not imply that their payloads are available on GitHub.
- Current source adds bounded admission, owned decoder/encoder cancellation and
  cleanup, media-timing fixes, opt-in persistent neural depth reuse, frozen build
  identity, source relinking, and preview/queue accessibility improvements.
- Owner scope is generated media only, targeting XREAL 1S. The owner currently
  has no access to the glasses. The project source license is MIT; model and
  dependency delivery terms remain under review.

Start with `RELEASE_READINESS.md` for acceptance, `RELEASE_DEPENDENCIES.md` for
candidate-specific dependency review, and `README.md` for use. `ROADMAP.md` is
an ongoing backlog; its older open items can be superseded by dated release
results. The former long handoff is retained locally at
`.runtime/final-push-checkpoint-20261021/HANDOFF-before-wrapup.md`.

## Selected private candidate: Python 3.12

| Identity | Value |
| --- | --- |
| EXE | `dist/StereoLab-0.2.0-beta.1.exe` |
| Size | 98,572,977 bytes |
| SHA-256 | `6776ba55a960d817c6e26f3a125dbc49f399f967435633c4e310fd1663943b45` |
| Embedded source fingerprint | `8e56afce8aa2240e94a07f27bc093186939dd3ed0302c72976a51baa76d9e2fe` |

The companion release-candidate files are in `.runtime/stereolab-release-candidate-0.2.0-beta.1/` with checksums in `SHA256SUMS.txt`. The locally built Inno Setup candidate is `.runtime/stereolab-installer/StereoLab-0.2.0-beta.1-Setup.exe` (99,775,508 bytes; SHA-256 `87f6a135b1e5c57f15dbf2fdf33b1b4873372604bcc86c19ac0f8c14a443e260`). It installs per-user, copies the EXE and MIT license, creates Start Menu launch/browser/uninstall shortcuts, and leaves app data in `%LOCALAPPDATA%\StereoLab`. Its isolated install/launch/uninstall evidence is retained under `.runtime/stereolab-installer/acceptance/report.json`; the installed app returned HTTP 200 and its exact EXE and license hashes matched the candidate. This installer is a private candidate only, not an approved public artifact. The old unversioned `dist/StereoLab.exe` remains intact because two processes were running from it; the new hash-verified EXE is the versioned sibling above.

The runtime fingerprint covers the 26 Python files declared in
`build_identity.py`. All nine packaged static assets separately matched source.
`gpu_hardware.py` is preserved in the source snapshot and its compiled code was
separately compared with the candidate after normalizing build-path filenames;
this supplement does not extend the runtime fingerprint.

The Windows x64 console EXE serves a loopback UI. Weights download on demand.
Video needs external FFmpeg with libx264. DirectML availability and NVENC encoding
alone do not prove GPU depth inference: inspect the named model's actual
provider, device, `inference_succeeded`, and `gpu_verified` results.

## Evidence retained for this candidate

These are same-host generated-media checks. Paths beginning `acceptance` are
under `.runtime/vfr-wave/frontend-a11y-final-20261021/`.

| Evidence | Result and retained report |
| --- | --- |
| MIT license package | Exact root license bytes are embedded and match the 1,064-byte source file (SHA-256 `a4de0c3dea2553b4f07daf7b5ace3e299498db724d575da3438f8e7d530a31b2`); sidecar copy matches. Reproducible extraction and hash evidence: `.runtime/mit-license-packaging-20261021-01/license-package-audit.json`. New 183-entry archive adds `LICENSE`; all 97 native entries and 22 third-party notice entries match the prior candidate. The 155 `base_library.zip` members also match by content. Isolated generated XREAL SBS/Small DirectML package acceptance passed: `.runtime/mit-license-packaging-20261021-01/acceptance-retry-01/report.json`. |
| Packaged pipeline/cache/restart | Named Small DirectML inference, cached CPU provenance, encoded/decoded/audio parity, source reupload, history/output identity, cache reuse and selected cleanup: `acceptance/report.json`. |
| Generated XREAL file | 3840x1080 SBS, eight decoded frames and audio: `acceptance-xreal/report.json`; sample: `acceptance-xreal/first-output.mp4`. No physical playback observed. |
| Two-hour generated neural export | 7,200/7,200 strict Small DirectML frames; complete video/audio decode with aligned starts; 349.5 seconds wall time, 23 clean resource samples: `acceptance-synthetic-soak-normal/report.json`. |
| Two-hour same-process uptime | 7,200.047 seconds, 24/24 verified Small DirectML jobs and 478 clean health/resource samples: `.runtime/wallclock-soak-normal-2h-retry-20261021/report.json`. This run used forced owned-process teardown. |
| Accessibility and output actions | Chromium keyboard Browse, status/busy regions, About naming/focus return and responsive widths: `acceptance-accessibility/report.json`. Copy equality and overwrite refusal: `acceptance-copy/report.json`. Explorer opened and showed the file: `.runtime/release-explorer-ui-20261021-normal2/report.json`; Reveal selection highlighting was not confirmed. |
| Model failure recovery | A preceding fingerprint-matched private build failed a custom-model inference job, then completed a queued named Small CPU job: `.runtime/vfr-wave/neural-failure-recovery-20261021-retry4/report.json`. |
| Idle graceful shutdown | Real Ctrl+C, exit 0, closed listener and removed extraction directory, without forced cleanup: `.runtime/graceful-ctrlc-20261021T165240Z-de8c94a9/report.json`. |
| First actual model download | Empty private profile downloaded Small, then completed generated CPU neural inference: `.runtime/first-run-model-download-20261021-a1/probe/report.json`. |

The first-run download was 99,060,839 bytes in 24.046 seconds. Independent hash
and pinned-revision receipt checks passed; model SHA-256:
`afb6a5c28f3b6bf1618c6e43f02073ef9dfdc70e937502d51603e57b0a1df10c`.
Its CPU output decoded as a 192x64 SBS PNG. Cleanup left no candidate process or
listener and empty extraction temp. Ctrl+Break returned code 3, so this probe
does not establish clean application exit. The separate idle Ctrl+C result does.
This is same-host fresh-profile evidence; clean-account/machine acceptance stays open.

Final serial source checks passed: CPU ran 617 tests in 23.111 seconds with
10 skips; GPU environment ran 617 in 21.621 seconds with 6 skips. Both treated
`ResourceWarning` as an error. Five Node syntax checks exited 0. Logs:
`.runtime/final-push-checkpoint-20261021/{cpu-full,gpu-full,node-syntax}.log`.
The CPU `.venv` has Starlette 1.3.1; separate fresh/overlay evidence in
`RELEASE_READINESS.md` covers the 1.8.0 declared minimum.

## Local review packet and dependency evidence

`output/release-review-2026-10-21/REVIEW.md` and `evidence-index.json` record the
candidate, report hashes and limits. The fixed earlier `source-snapshot.zip`
contains 144 source entries plus manifest/environment records; SHA-256:
`6e8204f5152213ad5da47fcb64cc02e184883973ff5c0bbf008b2c43214870d9`.
It predates this wrap-up and is not a replacement for the final Git commit.
Its environment inventory is not a dependency lock or complete frozen SBOM.

All 97 native entries were matched to local copies. Independent comparison with
the official Python 3.12.10 embeddable ZIP found 24 native byte matches among 26
overlapping names; two candidate VC runtime DLLs differ, and 71 other native
entries are outside that comparison. Package checksum/SPDX and aggregate-license
matches are recorded in `output/release-review-2026-10-21/cpython312-package-comparison.json`.
This establishes package-byte identity for the subset, not every upstream build
input or redistribution entitlement. All 48 candidate root CRT/API-set files
match Codex runtime copies; that directory does not identify their source agreement.

The incomplete notice review is
`output/release-notices-deduplicated-review-2026-10-21-r4.zip`, SHA-256
`c593d4317079f6195f8fa085718d42ae2ad4d1227abc733541a4cc7d772b0778`.
Its 62 aliases and 54 verified payloads include 22 embedded notices. The
supplementary draft is unapproved and is not in the EXE.
DirectML matches the published ONNX Runtime DirectML 1.24.4 CPython 3.12 wheel
and RECORD; the same-version NuGet DLL differs. Terms/notice applicability,
Microsoft CRT/UCRT provenance and entitlement, native transitive attribution,
OpenCV/FFmpeg/GCC/Winpthreads object attribution, upstream OpenSSL/libffi build
inputs, setuptools frozen-version/NOTICE scope, and model/conversion terms need
owner/reviewer resolution. Exact source links and qualifications are in
`RELEASE_DEPENDENCIES.md`.

A separate Python 3.14.9 candidate remains unpromoted at
`.runtime/python314-compatibility-review/dist-python314/StereoLab.exe`;
99,782,837 bytes; SHA-256
`e5afd208495e9b1690fb0bd75cefb692e367195f2e9c8761060595cbf79b190e`.
It has separate suite, generated approximation, provenance and notice evidence.
Its smoke reports `gpu_verified=false`; NVENC only encoded the video.
Python 3.12 results do not transfer to it.

## Remaining release gates and resumption

1. Reconcile pinned model terms, including non-commercial Base/Large and Xenova
   conversion rights, against the intended executable/model delivery.
2. Rebuild the exact selected source and verify the packaged MIT license, then
   resolve dependency provenance/notice scope and approve complete delivery
   materials. Current drafts are review inputs.
3. Observe the final artifact on a clean standard Windows account or machine,
   with supplied FFmpeg, fresh model acceptance/download, generated outputs,
   retrieval/restart and shutdown. Local VM/Sandbox discovery did not provide
   a usable clean environment; optional-feature state was not established.
4. Exercise an actual screen reader. Current Chromium/source checks do not prove
   spoken announcements or broad assistive-technology behavior.
5. Play the generated sample through physical XREAL 1S glasses when available.
   File geometry alone does not establish stereo, aspect ratio or audio playback.
6. Freeze approved source/build inputs, rebuild, verify that exact artifact,
   record its checksum/version/notes, then publish once the remaining
   redistribution and release-readiness gates are closed.

Generated-only scope leaves natural-scene depth quality and real-footage AV sync
unverified. Do not replace a running user EXE or control unrelated app instances.
Run source checks serially in PowerShell from the repository root:

```powershell
.\.venv\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -q
.\.venv-gpu\Scripts\python.exe -W error::ResourceWarning -m unittest discover -s tests -q
node --check static/app.js
git diff --check
```

A new build needs a fresh private directory, captured dependencies, its own identity
and candidate-specific acceptance. Some `scripts/` verifiers need local `.runtime/`
fixtures/models absent from GitHub. Final EXE promotion requires the gates above.
