# StereoLab roadmap handoff — 20 October 2026

## Current checkpoint — commit and resume authorized

The user requested committing the process-observation slice, then resuming with
army. Three independent lanes reviewed process identity, stress semantics, and
the private launch procedure. Review found and fixed a Toolhelp/PID-reuse race,
a post-terminal observation window delayed until after download/inspection, and
a stale cancellation status comparison. Semantic regression evidence is retained
under `.runtime/commit-resume/{process-review,stress-review}/`.

Independent final CPU and GPU-environment suites each passed 406 tests, with one
existing real-Windows-symlink privilege skip (16.150s and 16.307s). Python
compilation, Node syntax and whitespace checks passed. These suites precede the
new live GPU gate; no live result is claimed at this checkpoint.

Fresh NVIDIA inventory reports 0 MiB used, 0% utilization and no compute
processes; documented historical service ports are no longer listening. A
run-specific wrapper under `.runtime/commit-resume/private_gpu_gate.py` isolates
writable app directories and disables history before startup. It retains the
accepted model cache and runs unchanged production execution methods. Next:
after the scoped commit, recheck idle state and run the generated all-model GPU
process-observation gate on this newly owned private API, then archive jobs and
stop it gracefully. Representative footage and multi-hour acceptance stay open.

## Previous checkpoint — Phase 0.1 process observation instrumented (19 October)

Repository `D:\Users\Hp\Documents\ChatGPT\3d`, branch `master`, HEAD
`2bc6361`. The current process-cleanup instrumentation changes are uncommitted.
The four requested dark UI mockups under `output/` were already untracked and
remain untouched; no other output artifacts were staged or removed. No commit,
push, or executable build was requested.

The stress runner now requires an explicit loopback URL and expected server PID,
then matches `/api/diagnostics/resources` and the read-only Windows process
snapshot before it can submit jobs. It records verified FFmpeg PID/creation-time
identities on job polls, checks those identities globally after terminal status
to handle reparenting, and distinguishes first-sample absence, visibility then
absence, persistence, PID reuse, and unavailable observations. A timeout or
missing terminal snapshot is unverified and does not start a post-terminal
window. Snapshot-time visibility does not establish exact exit timing.
Cancellation now requires a processing/conversion job with at least five but
fewer than all frames completed, and records phase and frame counts.

Coordinator verification passed **399 tests** in each of `.venv` and `.venv-gpu`,
with the same existing Windows real-symlink privilege skip. Python compilation
passed. The helper lane also passed its 10 focused tests, including an owned
Windows child/nested-child lifecycle. No stress API job or new GPU workload was
run; current services remain untouched and per-process WDDM memory is still
reported as `[N/A]`.

Next gate: use an authorized representative clip and a genuinely idle window to
run a newly owned private GPU API with explicit PID matching, then collect live
process observations. Multi-hour reliability and exact process VRAM need their
own evidence; this instrumentation alone does not close Phase 0.1.

## Previous checkpoint — history integrated; fresh GPU clip gate passed (18 October)

## Prior checkpoint details — history integration and GPU clip gate

Repository D:\Users\Hp\Documents\ChatGPT\3d, branch master, HEAD `20709ff`
(dark UI polish). The Phase 4.1 application integration is implemented and
verified in the working tree, uncommitted. GPU consistency diagnostics and the
strict clip verifier have also been improved and verified. Earlier documentation changes and
output/ mockups were preserved. No push or EXE rebuild was requested.

Opt-in `STEREOLAB_HISTORY_INSTANCE` is now active at ASGI startup. The app owns
the worker lifetime and holds SQLite/lease ownership until cancellation and
worker join finish. Ordinary exports have streamed source hashes, capacity
checks and durable admission/settings/naming/lifecycle boundaries. Status and
read-only prior-launch history/download APIs feed the new history UI. Sources,
queue order and processing state are not restored or automatically requeued.

Latest independent suites: CPU 380 tests in 11.991s and GPU environment 380 in
11.624s, both OK with one existing Windows real-symlink privilege skip. Logs:
.runtime/gpu-consistency-followup/full-cpu.log and full-gpu.log. Node syntax,
Python compilation and whitespace checks passed. The earlier history-only
364-test suites remain in .runtime/history-coordinator-*-final.log.
The initial full-suite fixture failure is retained separately in
.runtime/history-coordinator-cpu-tests.log; its mock now produces an owned output.

Isolated CPU acceptance completed a generated still, canceled a controlled active
image, left a pending record, then gracefully restarted: prior success downloaded
with identical bytes, pending became interrupted, source SHA256/size matched,
current JOBS stayed empty and a concurrent second owner was rejected. Evidence:
.runtime/history-restart-verification/25c0413df33040f08bfa829e36d733e8/report.json.
The controlled image waited for production cancellation; this is not long-video
or GPU execution proof. Browser settings reuse left source empty/convert disabled,
history refresh worked, no console warnings/errors, and 320/391px layouts had no
document overflow. Screenshot history-browser.png is beside the report.

History QA did not contact existing services. GPU follow-up used read-only
status/job guards; none were restarted or stopped. The isolated browser server
was stopped after QA; its evidence persists and its temporary tab is closed.
Older live backends and the existing EXE do not gain the new Python behavior
without a separately accounted-for launch. Historical service inventories below
must be refreshed before any future service action.

GPU consistency follow-up: exact preprocessed tensors matched across all repeat
pairs; CPU stages were exact, while both fresh cold and one-frame-warmed GPU
processes showed small raw-output variation. One warmup is not a proven fix.
Per-repeat encoding showed libx264 amplification up to decoded max19/mean.005891;
the tested NVENC pairs decoded identically. Original failed NVENC max29 was not
reproduced or explained by these diagnostics. Independent output hashes/decodes:
.runtime/gpu-consistency-followup/independent-audit.json.

Verifier fixes require clip-to-both-export comparisons (tolerances are not
transitive) and reject missing public execution evidence. Unchanged GPU bounds
are max1/mean.001. On a NEW private GPU API, full generated source, processed clip,
two matched exports and cancellation completed: all three 36-frame decoded pairs
were exactly equal. Report:
.runtime/clip-verification/consistency-followup/20261018T222230Z-0e510fba/report.json.
Independent decode: .runtime/gpu-consistency-followup/api-independent.json.
Five terminal jobs (four complete, one canceled) were saved before stopping owned
PID34376/port56485. Existing APIs were guarded read-only and job counts unchanged.
The preserved earlier failed report remains historical counterevidence; this
fresh passing synthetic run does not establish universal determinism or real-media
quality. Proxy correspondence was checked by metadata, not exact source pixels.

Next: investigate intermittent raw/codec variance and continue real-media/reliability
validation. Phase 4.1 project/source-retention work, reusable depth (4.2),
checkpoints/resume (4.3), broader customization and clean-machine delivery remain
separate bounded slices. Do not promote either quality prototype without evidence.

The sections below preserve earlier evidence; current integration supersedes the
old standalone-foundation status and earlier next-step instructions.

## Historical running services — preserve existing download URLs

| Port | Ownership and last check |
| --- | --- |
| 8774 | Latest GPU app, PID13868, exec session7200, browser tab5 retained. One completed coordinator export, no active jobs. |
| 8772 | Earlier GPU app, PID14348/session36321. Three terminal records, including two user exports. Do not restart it to load new code. |
| 8770 | Earlier GPU baseline, PID12464. Sixteen terminal records. Left untouched. |
| 8768 | User's earlier service. Thirty-nine terminal records, no build fingerprint. Left untouched. |

Latest8774 backend fingerprint:
65bcb937dc4ccb86c2196d44f3ccd676a7e32a8caff1d06e8c392e70eaac3ae1.
Coordinator independently matched it to current backend source. Static files are
served from the working tree and are not included in that backend fingerprint.
Final safe server summary: .runtime/continuation-verification/final-servers.json.
Recheck active work and record counts before stopping any service.

Own CPU8773/PID25384/session70442 was stopped only after all four terminal records
were saved in .runtime/continuation-verification/cpu-final-jobs.json. Its files
remain, but those temporary in-memory download URLs no longer work. Earlier own
CPU8771 was similarly accounted for in the preceding handoff. No user service,
source/output directory or model cache was cleaned. WMI process details remain
unavailable; PID evidence came from owned launch sessions and HTTP diagnostics.

## Shipped in source during this continuation

Output naming: output_naming.py, main.py and frontend implement the frozen
plans/output-naming.md contract. Ordinary exports accept output_name_template,
default {name}_{layout}; supported fields are name/layout/model/date. Date uses
job creation UTC, not execution time. The limit is120 Unicode codepoints on both
sides; Cc/Cf/Cs controls, paths and formatting expressions are rejected. Final
ASCII stems are bounded to100 characters, with full UUIDhex and automatic PNG/MP4
extension. Windows device names before dots are protected. Exclusive reservation
and owned-output cleanup are retained. Clips keep their own internal names.

Naming capability/schema1 is separate from the exact11 performance settings.
Pending-only POST/api/jobs/{id}/naming is atomic and returns400/404/409 as appropriate.
Ready-file and batch freezes include names. Retry preserves the original template;
duplicate uses the current template. Both prepare a new ready item only when the
browser still holds the original File. Sources are not retained for retry.
Queue readiness now checks each file's frozen/current settings independently.
Older backends hide naming controls and omit the unsupported form field.

Completion notifications are off at every page launch. Explicit tab-only opt-in
requests browser permission only on click. Bounded512-ID tracking ignores clips
and terminal jobs first discovered during refresh; notices are batched per poll.
Alt+P generates a still preview and Alt+Enter submits the ready queue through the
existing enabled buttons. Editable fields, repeats, IME, other modifiers and open
dialogs suppress shortcuts. Both buttons expose aria-keyshortcuts.

Independent live checks used generated media: frozen engine/name submission worked
with an invalid visible template and unloaded visible custom model; queued renaming
worked, invalid edits returned400 and completed edits409. Downloaded PNG640x180
had the expected date/fullUUID name. Browser keyboard preview/submit, editable
suppression, tab arrows/End and panel labels passed. Real browser notification
permission/delivery was never enabled; only policy and failure handling were tested.

Real Small-model DirectML export on latest8774 completed without fallback, requested
256 and actual252x140, custom filename and PNG640x180. Evidence is under
.runtime/continuation-verification/: latest-gpu-naming.json/.png,
browser-naming-job.json, live-naming.json, keyboard-job.json and screenshots.

Responsive QA used a temporary same-origin iframe harness at320/375/768 CSS pixels.
No document horizontal overflow or out-of-bounds buttons/inputs/selects/queue rows;
long filenames and actions wrapped. This is embedded desktop-engine CSS evidence,
not physical phone/touch/display proof. The harness was moved out of static/ into
continuation-verification beside responsive-metrics.json and naming-responsive*.png.

## GPU pixel gate remains inconclusive

Do not relax or overwrite the failed strict clip gate:
.runtime/clip-verification/20261018T113023Z-3a3457b7/report.json.
Two ordinary matched exports themselves differed; the processed clip matched the
second ordinary export exactly. Earlier detailed metrics remain in the archived
handoff and .runtime/resume-verification/gpu-clip-independent.json.

New diagnostic scripts/isolate_gpu_variance.py reuses production stages with a
pinned generated36-frame Small-model proxy, one owned session and fresh temporal
history for each repeat. GPU report:
.runtime/variance-isolation/20261018T161848Z-05a358b7/report.json.
Only the first raw ONNX frame differed between repeat1 and repeats2/3:
max1.9073486328125e-6,34865 values. Temporal effects spanned11 frames; rendered BGR
changed only4 channel values by1. Repeats2/3 matched exactly at all captured stages.
Identical fixed-input libx264 and NVENC encode pairs had identical file hashes and
all36 decoded frames within each codec. Coordinator recomputed arrays/metrics and
rehashed/decoded outputs: continuation-verification/independent-variance.json.

CPU diagnostic20261018T163225Z-f5a4316b matched exactly at every stage in all three
repeats and produced identical libx264 files. A diagnostic CPU-mode bug that still
probed NVENC was fixed with regression tests; CPU now avoids hardware probing.
Independent check: continuation-verification/cpu-variance-independent.json.
These observations do not establish preprocessed tensor identity, universal GPU
determinism, or a production warmup/comparison policy. App defaults were unchanged.

## Historical Phase4.1 foundation — before application integration

At this earlier checkpoint job_history.py was a standalone opt-in SQLite metadata
store. The following describes that checkpoint, superseded above. App jobs then
still disappeared after a server restart; the proposed environment variable/API/UI
in plans/job-history.md are not active app features. No sources are retained,
read or deleted by the module, and no Job/queue objects are reconstructed.

Implemented foundation: strict stable instance namespace; non-inheritable OS-held
lease; schema1/finite bounded metadata and exact settings/naming; atomic admission,
pending edits and terminal records; interrupted prior-boot history; validated
read-only pagination and confined completed-output lookup; metadata-only pruning
(default1000 records/90 days); degraded writes with preserved readable history.
Invalid rows remain preserved and visibly warn. Corrupt/future schemas are preserved.

Review caught and fixed startup-exception lease leaks, dangling database link
creation, and cross-namespace database/lease hardlinks. Real Windows subprocess
contention, owner exit, stale lock contents, crash before/after completion commit,
non-inherited generic exec child and hardlink tests passed. Of25 focused tests,
24 passed and one real symlink-creation test skipped for Windows privilege1314;
a separate mocked dangling-link-before-connect test passes. POSIX locking and real
symlink rejection still need execution evidence. Lookup checks size, not a content
hash; hostile filesystem replacement races are outside this cooperative local scope.

Exact adapter API and deferred work are documented in plans/job-history.md.
Next: design startup/shutdown ownership so a lease cannot be released while a
worker can still mutate records; stream source hashes; pre-upload capacity checks;
durable admission/queued edits before in-memory publication; coarse lifecycle
writes and degraded status mapping; separate prior-boot read-only API/UI. Then
verify an isolated CPU server restart/download end to end. Never revive old JOBS,
automatically requeue, retain sources implicitly or claim arbitrary-frame resume.

## Earlier independent checks

CPU:350 tests in11.661s, OK with one skipped real symlink-creation test.
GPU environment:350 tests in11.766s, same expected skip. No remaining failures.
Logs: .runtime/continuation-verification/final-cpu-tests.log and
final-gpu-env-tests.log. Compileall, Node syntax, git diff--check and changed-text
checks including untracked files pass. Earlier integrated-* logs captured the
history module's in-progress red stubs and are superseded by these final logs.

The suites contain mocked neural sessions plus real CPU/FFmpeg/process checks;
real DirectML evidence is separate above. No new EXE build, clean-machine test,
physical display proof, multi-hour reliability or universal quality claim.
At the previous handoff all worker lanes were released and the20 changed/new
source/docs/tests were ready for the separately documented `cba8716` commit.

## Earlier roadmap gates (first gate now completed)

1. Integrate the standalone history foundation under its frozen failure/ownership
   contract, with a separate read-only history UI and explicit source reselection.
2. Continue GPU variance investigation without changing the failed comparison gate.
3. Test representative authorized footage, longer stress workloads, physical target
   displays, touch/accessibility and actual browser/OS notification delivery.
4. Whole-export/model/encoder calibration still needs its own bounded spec. Current
   calibration measures input detail on one proxy frame only.
5. Projects, reusable depth, pause/checkpoints and true resume remain separate.
   Neither renderer_quality.py nor temporal_quality.py prototype is promoted.
6. Rebuild the EXE only with actual launch/runtime validation when requested.

Useful gates: .venv/Scripts/python.exe -m unittest discover -s tests -q;
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -q;
node --check static/app.js. Do not restart old services merely to refresh source.
