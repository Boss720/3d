# Processed clip preview — bounded Phase 3.1 specification

Status: implemented; integration checks recorded in PROGRESS.md. Further real-footage and display acceptance remains open.
Existing export behavior, active jobs and deleted-source ownership must remain clear.

## Outcome

The user selects a timestamp in a browser-playable video and explicitly requests a
1–5 second processed preview. This runs the SAME depth/refinement/temporal/render
pipeline and selected settings as export, at a clearly labeled proxy bound. It
exercises motion/cuts within the chosen interval, starts temporal history fresh at
the clip boundary, and does not claim pixel identity with a full-detail export.

## API and scheduling

Reuse local job machinery with purpose='preview_clip' (ordinary jobs default to
'export'). The frontend must require an explicit new status capability before
sending this request, so an old backend cannot ignore fields and export a whole
movie. Validate response purpose before displaying it. Inputs: original video File,
current export settings, clip_start_seconds finite>=0, clip_duration_seconds in1..5.
Images reject clip purpose. Limits: proxy <=960x540, <=24fps, <=120 frames. No silent
model/detail changes. CPU-only remains useful; explicitGPU remains strict.

A server-owned temporary H.264/AAC proxy segment is extracted with FFmpeg, bounded
by a cancellation-aware finite-time helper. Extraction uses argv, no shell, strips
irrelevant tracks, keeps optional audio and reports preparation separately. The
original upload and proxy have separate unique paths and explicit cleanup owners.
The output is unique and finalized by existing encoder checks. Sources delete when
the job ends, as current exports do; this does not establish a reusable source cache.

Queue behind existing work. Superseded preview requests are canceled by their IDs;
an old response can never replace the current selected source/settings/timestamp.
One inference session remains shared; no parallel model sessions are created.
Preview jobs are labeled and displayed in their preview area, not mixed into the
ordinary conversion-list selection/count. Existing exports remain cancelable.
No priority/preemption is promised beyond preserving existing queued work and
canceling obsolete previews. Later queue redesign must preserve these contracts.

## UI

A short-clip duration selector and explicit Generate processed clip action live in
the preview workspace. Show queued/preparing/converting/finalizing/canceled/failed
states, selected source interval, actual proxy dimensions/FPS, model/device and
fresh temporal-history note. A processed video player uses the resulting local URL.
Cancel is always reachable. Source/settings/timestamp change invalidates clip and
cancels its pending request/job; view/zoom changes alone do not. Stale controls cannot
silently restart full conversion. Do not auto-play with audio.

## Acceptance

Semantic tests prove: old API jobs retain behavior; invalid purpose/range/media
rejected before expensive work; exact <=120-frame bound; path/output separation;
extraction failure/cancel cleanup without deleting another job's sources; processing
uses export temporal/refinement settings; stale/superseded requests cancel the old
job; active ordinary exports continue; preview purpose validation blocks oldbackend.

Real generated moving/cut/audio fixture: extract nonzero start, decode processed
output, verify frame count, nominal FPS/duration, audio and expected proxy layout;
observe cut behavior across the interval. Compare at matched proxy inputs with the
ordinary export pipeline, allowing only previously documented DirectML rounding.
Test cancellation during both extraction and inference. Browser-check states and
playback. Physical stereo-display suitability remains unverified.

## Explicit limits

Browser codec support may prevent source timestamp selection while export remains
possible. Uploading an original large file can dominate preview time. Re-encoding a
proxy changes pixels and can affect inferred depth. VFR/HDR/subtitle/spatial metadata
remain subject to the documented export limitations. No arbitrary-frame resume,
full-movie temporal context, hidden-surface recovery or live VR output is implied.
