# StereoLab roadmap handoff — 18 October 2026

## Resume here

User asked to commit, then continue the roadmap until approaching the usage limit
and write a handoff. This bounded run is complete; the roadmap is unfinished.
Observed closing usage:83% five-hour/58%weekly. No reset credit was used.

Repository: D:\Users\Hp\Documents\ChatGPT\3d, branch master.
Commits: fbab918 contains the previous roadmap wave; a130916 commits the remaining
Windows packaging setup. The tree was clean immediately after a130916. This
continuation's 20 changed/new files remain uncommitted. Nothing was pushed.
The executable was not rebuilt and does not contain the latest source changes.

Earlier detailed evidence is preserved in:
- .runtime/continuation-verification/handoff-before-continue.md
- .runtime/resume-verification/handoff-before-resume.md
- ROADMAP.md, PROGRESS.md and the plans/ specs.

## Running services — preserve existing download URLs

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

## Phase4.1 storage foundation — NOT wired into the app

job_history.py is a standalone opt-in SQLite metadata store. Current app jobs
still disappear after a server restart; the proposed environment variable/API/UI
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

## Final independent checks

CPU:350 tests in11.661s, OK with one skipped real symlink-creation test.
GPU environment:350 tests in11.766s, same expected skip. No remaining failures.
Logs: .runtime/continuation-verification/final-cpu-tests.log and
final-gpu-env-tests.log. Compileall, Node syntax, git diff--check and changed-text
checks including untracked files pass. Earlier integrated-* logs captured the
history module's in-progress red stubs and are superseded by these final logs.

The suites contain mocked neural sessions plus real CPU/FFmpeg/process checks;
real DirectML evidence is separate above. No new EXE build, clean-machine test,
physical display proof, multi-hour reliability or universal quality claim.
All worker lanes are released. Preserve all20 changed/new source/docs/tests.

## Next roadmap gates

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
