# Queue workflow — bounded Phase 3.3 slice

Status: implemented and independently verified; see PROGRESS.md. Durable sources/restart recovery remain Phase4.

Ordinary queued exports can be reordered, removed and assigned a complete settings
snapshot. Running, dequeued-for-start and terminal jobs reject these edits (409).
Preview clip positions remain fixed when ordinary exports are reordered. Queue
mutation and worker dequeue share one lock; stale order requests fail atomically.
No source upload retention is introduced. Completed/failed/canceled job sources
require reselection after reload. Browser-held original Files may create a NEW job.

## API

Status queue_capabilities: schema_version1, reorder_queued_exports:true,
edit_queued_exports:true, remove_queued_exports:true, settings_snapshot:true.

POST /api/queue/reorder JSON {job_ids:[...]} lists EVERY currently pending,
uncanceled ordinary export exactly once. Preserve queued clip slots. Invalid
duplicates/type =>400; changed queue =>409. Return {job_ids:[...]} accepted order.

POST /api/jobs/{id}/settings JSON {settings:{...}} replaces all 11 known export
settings, with strict enum/finite-number/boolean validation and no unknown fields.
Only a pending ordinary export may change. Revalidate model/device policy before
commit. Failures leave previous settings intact. Return publicjob. Model readiness
checks may occur before acquiring queue locks, followed by rechecking pending state.

POST /api/jobs/{id}/remove-queued removes an ordinary pending job, immediately
marks canceled and cleans only its owned upload. Unknown404, no-longer-pending409.
The existing cancel endpoint and terminal download behavior stay intact.

Public jobs add settings_schema_version:1, settings:{11 exact values}, source_available
(true only while its owned upload is a readable file), queue_position (1 based over
all active pending work; null once dequeued). Top-level legacy fields remain.
Queue list read and mutations use one consistent queue-before-jobs lock order.
Workers retain FIFO get/task_done semantics and non-busy condition waits.

## Frontend

Require explicit queue capability before server mutations. Up/down on pending
ordinary exports submits the full current export ID order; on409 refresh and
explain that work started or queue changed. Active rows never get queued-edit
buttons. Remove queued uses its endpoint; active still uses Cancel.

Pending browser Files can move up/down before upload. Their default is current
visible settings at conversion time. An explicit Use current settings action freezes
an individual pending snapshot. Apply to ready files freezes all unsubmitted ready
Files. Show snapshot values and whether they are frozen; never silently mutate an
active job when controls change. A per-queued-row Use current settings uses the API.

Use settings copies a valid public snapshot to visible controls, without starting
work. Retry failed/canceled with a browser-held File creates a new ready item with
the original snapshot; Duplicate with current settings creates a new ready item
with current snapshot. Neither automatically submits. If File is unavailable,
show source reselection required and offer Use settings; do not imply server source
retention or match arbitrary files by filename. No automatic retry after GPU failure.

Show readable settings details, actual device/provider, encoder, phase/fallback.
Buttons need contextual accessible names, and keyboard focus should remain useful
after queue redraw. Notification opt-in, naming templates and durable source storage
remain later work. Current unique output IDs preserve collision protection.

## Gates

Semantic concurrency tests prove a dequeued/running job cannot be edited or removed,
atomic stale-order rejection, clip slot preservation, no task-count deadlock, exact
snapshot validation, failed edit rollback, owned-upload cleanup and legacy behavior.
UI contracts cover settings round trip, source absence, ready-order changes and
invalid/old capabilities. Coordinator runs both full suites and live queue operations
with one intentionally slow active CPU job and small queued images. Verify active
settings/output identity stays fixed and queued edits affect only the selected job.
