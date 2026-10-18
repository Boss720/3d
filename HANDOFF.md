# StereoLab roadmap handoff — resumed run, 18 October 2026

## Resume here

Continue ROADMAP.md's execution table and the final entries in PROGRESS.md.
This bounded run stops near account usage limits; the roadmap is NOT finished.
Closing observed usage: 85% five-hour / 43% weekly.
No reset credit was consumed. Read the final PROGRESS entry for the closing check.

Repository D:\Users\Hp\Documents\ChatGPT\3d; branch master, roadmap baseline0c08d1a
(Enable DirectML for Large on 4 GB GPUs). The user subsequently requested a local
roadmap commit; see git log and the final PROGRESS entry. Packaging-only edits
remain unstaged, including their README/main hunks. Exact staged source passed
288 tests in each environment; its fingerprint differs from the retained live
working-tree build because the frozen-executable path changes are excluded.
Preserve all dirty/untracked files. Original packaging work (.gitignore, README,
main, ROADMAP, StereoLab.spec, build-exe.ps1) coexists with roadmap edits.
dist/StereoLab.exe was NOT rebuilt and does NOT contain these source changes.
No clean-machine/executable acceptance was performed in this resumed run.
Prior handoff archived at .runtime/resume-verification/handoff-before-resume.md.

## Latest live app and server ownership

- Latest GPU app: http://127.0.0.1:8772, .venv-gpu, PID14348, exec session36321.
  Loaded backend fingerprint608134272b78059e5a4b5db83db00cd75ed3c08a4a8a242a2f8d2c019ca7ec23.
  Latest Small GPU full still + processed clip passed. Browser tab4 retained here.
  One completed clip record, no active jobs at its verification; recheck before stop.
- Port8768 belongs to the user's earlier export session. Last observed39 records,
  no active jobs, older backend with no fingerprint. Count rose from37 during this
  run, so preserve its current user outputs/download URLs. NEVER restart merely
  to pick up current static files. Current frontend capability-gates new APIs.
- Port8770 retained untouched: GPU PID12464,16 terminal records, no active jobs,
  fingerprint1fd77b5a23caae4379a381ef30b2e14a96ed4fbe628912c53f9a39c62c4ca27e.
  It provided the all-model stress evidence and predates this resumed wave.
- Coordinator CPU8771 was stopped ONLY after11 own terminal records were saved
  to .runtime/resume-verification/cpu-final-jobs.json and no job was active.
  Its output files remain; those temporary in-memory download URLs no longer work.
- Previous8772 verification server was stopped idle after its8 terminal records
  were saved in server-8772-before-final.json; reports retain output copies.
- Other services were not stopped. OS process-detail/WMI queries were denied;
  server PIDs came from launch logs/diagnostics, not independent WMI verification.
- All worker lanes released. All generated fixtures/reports stay in ignored
  .runtime/ and outputs/. No recursive cleanup was run. Some empty inaccessible
  OS-temp directories from early test-fixture ACL errors were left untouched.

## Implemented before this resumed run

Capability schema1, conservative unknown GPU memory, Large518px policy,
60s/64-entry GPU failure suppression and explicit retry; requested/actual provider,
input dimensions/session/inference timing; depth-only raw-preview LRU8/32MiB;
shared preview/export inference admission; first-frame/codec/disk preflight;
service/queue/finalization/restart timing and recent-frame ETA; unused-eye skipping
for DEPTH/RGBD; four exact performance presets and JSON import/export;
opt-in ONE-FRAME input-detail calibration with explicit apply and atomic16-report
history; exact processing-frame Source/Depth comparison and bounded zoom/pan;
loopback Host/Origin write boundary; benchmark and opt-in resource tools.

Input calibration does NOT compare whole exports, models, encoders or visual
quality, and it does NOT set an Auto policy. Source/metadata memory is additional
to the raw-depth32MiB budget. DirectML may execute some nodes on CPU.

Unpromoted renderer_quality.py and temporal_quality.py experiments remain outside
runtime: much slower and no consistent synthetic/real-video benefit demonstrated.
Do not select them merely because semantic tests pass. Lost raw scale before
per-frame normalization remains an architectural quality issue.

## New full-detail still preview

preview_policy.py and /api/preview preview_detail='proxy'|'full', defaultproxy.
Proxy keeps960x540; full uploads the original decoded still up to12,000,000pixels,
8192px per side and24MiB. Neural input remains independently hardware-capped.
Checks after imdecode do NOT bound decoder allocation or peak process memory.
Full depth larger than cache budget intentionally bypasses retention.

Response source_width/height are actual inference-source pixels; original_* are
uploaded decoded dimensions. Frontend separately labels selected source dimensions
because JPEG proxy uploads are already downscaled. Legacy full responses reject
rather than mislabeling a proxy. Fullvideo-frame mode is disabled in the UI.

LiveCPU and latestSmallDML verified1200x640 full vs960x512 proxy and exact returned
source pixels. Latest full still used actual neural252x140 at requested256.
Artifacts: .runtime/resume-verification/full-still-live.json, latest-gpu-smoke.json,
full-still-browser.png. Unit tests use mocked neural sessions, not real Large proof.

## New processed clip preview

Frozen spec plans/processed-clip-preview.md implemented. POST/api/jobs supports
purpose='export'(default)|'preview_clip', finite start>=0 and duration1..5seconds.
Explicit status.preview_capabilities schema1 is REQUIRED by UI before upload so
old backends cannot ignore fields and export a whole movie.

Original selected video is uploaded; preview_clip.py extracts its own unique
H264/AAC proxy <=960x540, <=24FPS, <=120frames, nominal <=5.05seconds before
inference. FFmpeg argv/no shell, exclusive destination ownership,30s timeout,
cancel polling, finite terminate/kill waits, bounded stderr tail (spooled to disk).
Proxy and original upload cleanup have distinct owners. LowFPS proxy minimum1FPS;
missing source FPS uses explicit fallback. Actual clip may be shorter than requested
because sparse frames/seeking/EOF; UI separates requested interval and playable
frames/FPS/duration. Preflight preserves positive sub-1 FPS and marks provisional
fallback timing, which cannot justify a hard source-duration cutoff.

Same Job, export refinement/temporal/render pipeline and current settings; fresh
TemporalDepthFilter at interval boundary. Proxy re-encoding changes source pixels;
no full-source pixel-identity claim. FIFO behind existing jobs, no priority/preemption.
Stale source/settings/timestamp/length results hide and cancel. Uploads await returned
job IDs before stale cancellation; transient poll errors recover; malformed responses
are quarantined until terminal state is confirmed. Muted, non-autoplay inline player
GET/api/preview-clips/{id}/video. Clip records excluded from ordinary queue UI.

CPU matched-proxy clip vs ordinary export:36frames exact decoded equality,
640x180 SBS/12FPS/3seconds/AACaudio, start9seconds across a synthetic cut.
CPU Small inference cancellation after2frames: canceled and both routes404.
Latest GPU source build: full still +1second12frame640x180 clip/audio pass.
Live.5FPS nonzero-start source also completes, actual2frames/1FPS vsrequested3s.
Reports: .runtime/resume-verification/processed-clips-live.json,
processed-clip-probe.json, clip-cancel-live.json, lowfps-live.json,
latest-gpu-smoke.json; screenshot processed-clip-browser.png.

## GPU pixel gate is explicitly inconclusive

scripts/verify_clip_preview.py and tests/test_clip_verification.py added.
Report .runtime/clip-verification/20261018T113023Z-3a3457b7/report.json is overall
NONPASSING (strict gate), pixel_gate_conclusion='inconclusive'. Keep this evidence.
First rawfailure20261018T112810Z-af1948cc/report.json also preserved.
Loaded build101dc0311a17978ac1a263111af74a8d03d89dcafaeef54ca9eb738d88de75c5,
before queue/output/lowFPS corrections; normal12FPS path unaffected by those fixes.

All functional jobs verified Small explicit DML/GTX1650Ti, no fallback:360frame
full-source export,36frame processed clip, two36frame ordinary matched exports,
audio/dimensions/rate/duration. Two ordinary exports themselves differ max29,
mean0.0016760706/255,5913 of12,441,600channels. The processed clip matches the
SECOND ordinary export EXACTLY across all36frames. Coordinator independently
recomputed all three comparisons (gpu-clip-independent.json). First run comparison
was max40/mean0.00248368. The strict sparse-one-level threshold was NOT relaxed.
Do not apply PNG rounding bounds as a guarantee for NVENC/H264 decoded output or
claim a deterministic GPU-video pipeline. Further isolation of inference/render/
encoder variation is next gated evidence work, not an automatic quality fix.

Inference cancellation requested after23frames: canceled, inline and download404;
this does NOT prove encoder OS-process exit. Worker directly observed owned
extraction FFmpegPID25844 exit rc1, removed only its partial, source SHA unchanged.
That proof is limited to extraction's own registered Popen.

## New queue workflow and output ownership

Frozen plans/queue-workflow.md implemented. New job_queue.py EditableJobQueue uses
one reentrant queue lock for dequeue and edits plus normal condition/task accounting.
Queue-before-jobs lock order. Worker startup has a separate lock to prevent dual
consumers under concurrent uploads. Dequeued-but-not-yet-processing exports reject
edits. status.queue_capabilities schema1 gates frontend mutation support.

POST/api/queue/reorder {job_ids:[ALL pending ordinary exports]} atomically reorders
only export slots, preserving clip positions. Stale order409, malformed400.
POST/api/jobs/{id}/settings {settings:{11exactfields}} replaces a pending ordinary
export snapshot, strict enums/finite numbers/booleans/noextras; readiness validated
outside locks then membership rechecked; failures preserve old settings.
POST/api/jobs/{id}/remove-queued markscanceled and immediately removes only its
owned upload. Active exports use existing Cancel and reject pending edits.
Public job settings_schema_version1/settings/source_available/queue_position;
legacy top-level fields remain. No source retention introduced.

Browser local ready items reorder; default remains visible settings at submit.
Explicit freeze perfile/batch apply creates snapshots. Use settings loads controls
without starting work. Retry failed/canceled and duplicate create NEW ready items
only if browser-held File remains, never auto-submit. Refresh requires reselecting
source; no filename matching. Verified batchstrength5 remained5 despiteglobal0,
duplicate prepared0 without submitting; source absence explained afterrefresh.
Initial CSS actions squeezed metadata; browser caught literal-backslash-n append
failure too. Corrected real linebreaks and verified35px/530px two-column row with
full-width actions BELOW details at1265pxviewport. queue-browser-fixed.png retained.

Live API check used a slow ordinary360frameSmallCPU export while reorder/edit/remove
occurred. Active settings unchanged and outputcompleted; editedBstrength0PNG exact
source pair; clip slot2 preserved; stale/active edits409; removedA sourcefalse,
canceled/download404. Report .runtime/resume-verification/queue-live.json.
Scratch repeatable live script verify_queue.py is in that same ignored folder.
First attempt used wrong HTTP method for download; fixed and fullcheck reran.

Output names now use fullUUIDhex, rather than8-character prefix, with exclusive
reservation before processing and cleanup only for owned targets. Distinct IDs
sharing oldprefix no longer collide. Existing output/file/symlink is preserved
and fails before inference. Existing export filenames were NOT rewritten.
Naming-template UI, completion notification opt-in and further shortcuts remain open.

## Longer all-model synthetic stress evidence

scripts/stress_roadmap.py + benchmarks/stress-manifest.json generate deterministic
pan/thin-detail/lowlight/cuts/audio30s320x180/12FPS360frames. Allfour explicitGPU
518-request exports and separate >=5frame cancellation tests passed sequentially
against old8770build1fd77b5a. MiDaS actual384x384; others518x294 includingLargecap.
Final report .runtime/stress-verification/20261018T110251Z-f0392917/report.json.
Coordinator decoded allfour640x180/360/12FPS outputs and independently checked
SHA256/audio/30sduration/provider/cancel404 (stress-independent.json).

Small/Base/MiDaS/Large walltimes75.02/133.11/53.15/263.12s. Whole-device sampled
max566/938/1182/2025MiB includes OTHER processes, is not exact per-job VRAM.
Server working-set peak is process-lifetime. Other user server remained available;
target API idleness is not proof no other GPU process was active systemwide.
This generated30s workload is NOT representative real-footage quality or multi-hour
stability. User-selected footage path was asked for but not supplied; personal files
were not searched. Initial runner null-error bug fixed/regression tested; failed
attempt artifacts preserved. Cancellation API does not establish no orphan encoder.

## Independent final gates

.venv Python3.12.10:288tests in9.055s OK.
.venv-gpu:288tests in9.130s OK.
Logs .runtime/resume-verification/final-cpu-tests.log and final-gpu-env-tests.log.
Node syntax, compileall, tracked git diff--check and untracked text whitespace pass.
Tests include fake neural sessions and realCPU/FFmpeg fixtures; liveDML evidence is
separate. Starlette/httpx deprecation and CRLF warnings do not fail tests.
Python3.10 itself remains untested. No complete EXE/release/build claim.

## Next bounded work

1. Isolate GPU encoded-output self-variation with matched raw render/encoder controls;
   retain failed gate until evidence justifies a comparison policy. Do not silently
   relax tolerances or claim the clip implementation is the cause from these data.
2. More3.1/3.3 keyboard, narrow-layout and physical-display QA. Tooling lacked viewport
   resizing; desktop snapshot alone is not responsive proof. Fix concrete issues.
3. Naming templates and opt-in notifications/remaining shortcuts. Preserve complete
   immutable queued settings, exclusive output ownership, clip slots and active jobs.
4. Phase1.3 model/encoder/whole-export calibration still needs a frozen source/budget/
   session comparison contract. Current calibration is input-detail on ONE frame.
5. Representative user/licensed footage and substantially longer stress sessions;
   all-model30s synthetic checks do not close Phase0.1's broader reliability gate.
6. Phase4 durable jobs/projects/reusable depth needs explicit source retention,
   quotas/TTL/cleanup and restart contracts. Current jobs are memory-only; never
   imply retry/resume works after source deletion. MonolithicMP4 frame resume unsolved.
7. Raw-scale temporal quality and occlusion experiments remain gated; neither prior
   prototype is promoted. Rebuild EXE only with actual launch/runtime validation.

Useful gates: .venv/Scripts/python.exe -m unittest discover -s tests -q;
.venv-gpu/Scripts/python.exe -m unittest discover -s tests -q; node --check static/app.js.
Restart only an identified IDLE own server after accounting for its output records.
Never touch user's8768 merely to refresh backend source.
