# Job history — bounded Phase 4.1 slice

Status: storage, backend lifecycle, API and UI integrated in source on top of
`20709ff`. Independent CPU and GPU-environment suites each passed 364 tests
(one Windows symlink-privilege skip). Isolated CPU server restart/download/hash,
pending interruption and owner-contention acceptance passed. Browser settings
reuse leaves the source empty and conversion disabled; 320/391px layouts fit.
Evidence: `.runtime/history-restart-verification/25c0413df33040f08bfa829e36d733e8/`.
Ordinary export metadata only; source retention, automatic restart and frame
resume stay off. Existing servers were left running with their current code.

## Outcome and scope

An explicitly configured server remembers its ordinary exports across launches.
The user can inspect prior results, download a still-available completed output,
and reuse validated settings/naming after explicitly selecting a source again.
Interrupted work is identified honestly and never automatically enqueued.

History records are separate data objects, never reconstructed executable Job
instances or inserted into JOBS/JOB_QUEUE. Processed previews remain memory-only.
This slice does not restore queue order, inference sessions, temporal state,
monotonic timers or ETA. It does not introduce pause, retained-source retry,
projects, reusable depth or chunk/checkpoint recovery.

## Explicit instance opt-in and existing paths

Use one new environment variable, read at ASGI startup:

```powershell
$env:STEREOLAB_HISTORY_INSTANCE = 'desktop'
.\run.ps1
```

Unset or empty disables history and creates no history files. A supplied name
must match `[a-z0-9][a-z0-9_-]{0,31}` exactly. Invalid configuration fails startup
before opening history files. No automatic naming by PID, port or timestamp.
Keep the same name to revisit that instance; choose a different name for an
independent server. Removing the variable disables history without deleting it.

Retain current DATA_ROOT semantics: source runs use the repository; the frozen
EXE uses `%LOCALAPPDATA%\StereoLab`. Store this instance under
`DATA_ROOT/.runtime/history/instance-<name>/`, containing `owner.lock` and
`history.sqlite3`. Source and EXE histories are consequently separate unless a
later explicit migration feature is designed. No custom database-path setting.

The current run.ps1 already inherits environment and starts main.py; external
single-worker uvicorn launches inherit the same variable. No launcher or port
change is required. History-enabled `--reload` and multi-worker operation are
outside this slice; enforce one owner process per instance namespace.

Do not contact, restart, import records from, enumerate sources for, or clean up
the existing services on 8768/8770/8772. Old processes do not participate in this
new namespace. Shared uploads/outputs remain protected by per-job identity and
existing exclusive output reservation. Never infer ownership from a port or
from finding a file in those shared directories.

## Process lease and startup

1. Acquire the instance lease before opening/migrating SQLite or accepting jobs.
   Hold a dedicated descriptor for the server lifespan, including worker shutdown.
2. On Windows, use `msvcrt.locking(fd, LK_NBLCK, 1)` at offset zero. On POSIX,
   use `fcntl.flock(fd, LOCK_EX | LOCK_NB)`. The lock file may contain an initial
   byte; its contents/existence are never the ownership authority.
3. Explicitly keep this descriptor non-inheritable so FFmpeg cannot retain it.
   Never unlink, replace or truncate owner.lock during normal cleanup. Unlock and
   close only after this process has stopped its own worker and closed SQLite.
4. Contention or unsupported locking fails history-enabled startup clearly;
   never fall back silently to another namespace or an unlocked writer. Do not
   terminate the existing owner. Use another instance name for another server.
5. Each successful owner gets a fresh boot UUID. PID/start time may be recorded
   for diagnostics, not takeover decisions. No heartbeat expiry or wall-clock
   timeout authorizes reclaiming a live owner.

Use local storage. Reject UNC paths and Windows drives identified as remote;
network/synchronized filesystem locking and durability are not supported claims.
Acquiring the lease proves exclusive metadata ownership, not that a prior
owner's FFmpeg children exited. Do not kill, resume or reuse their files.

## SQLite schema and bounded storage

Use Python's sqlite3 module with short transactions, parameterized queries,
`journal_mode=DELETE`, `synchronous=FULL`, `foreign_keys=ON` and a 1000 ms busy
timeout. An atomic JSON replacement prevents a torn document but does not itself
coordinate process writers; SQLite provides the chosen transaction boundary.
One instance owner removes the need for WAL or cross-server shared writers here.

Schema version 1 uses explicit sessions and export-record tables. A record stores:

- Canonical full job UUID, owner boot UUID, schema version and UTC timestamps.
- Basename-only source display name, media type, byte count and SHA256 computed
  while streaming the already-required upload. No original absolute source path.
- Validated exact 11-field settings plus their schema, naming template/schema,
  planned output basename and build fingerprint.
- Last durably recorded lifecycle state/phase; bounded terminal error, execution,
  encoder and timing summaries. Exclude arrays, handles, events and raw media.
- Only after verified completion: actual output basename, media type, byte size
  and completion UTC. Store a basename, never a client-selected download path.

Record admission after successful upload and before publishing/enqueuing the job;
persist accepted queued settings/naming replacements too. Record processing and
terminal boundaries. Do not write every frame or persist queue positions.
Lifecycle summaries contain finite numbers; string/detail fields are bounded.

Limit to 1000 records, with each serialized record at most 64 KiB. Prune terminal
metadata older than 90 days, then oldest terminal records to admit new records.
Never prune unfinished records to make room; if no room remains, reject new
ordinary export admission with 429 before application upload staging. Recheck
capacity at commit to cover concurrent uploads. SQLite pages/journals add disk
overhead: these logical limits are not a hard database-byte or disk-fit guarantee.

Pruning removes database rows only. No output/source TTL, file deletion, media
retention or directory sweep is introduced. Existing current-job source cleanup
still occurs as today. Leftover uploads after a crash do not become retained
sources, even if they happen to exist and match a recorded hash.

## Crash, restart and persistence failure

After acquiring the lease, transactionally mark every earlier boot's nonterminal
record interrupted, retaining its last-known state. This includes queued,
initializing, preparing, processing and finalizing. Never start it automatically.
Terminal records retain their terminal state. Graceful shutdown does not convert
unfinished work into success; shutdown must not release the lease while its
worker can still mutate that instance's records.

Persist completed only after the existing encoder/image-write success checks,
output ownership/path checks and nonempty output checks. A crash before the
completion transaction leaves interrupted history, even if an output file looks
finished. Never promote a row by scanning for files or trusting an MP4 header.
Missing or size/type/path-changed completed outputs keep their recorded success
but show output unavailable; an unavailable file cannot be downloaded through
history. Same-size content replacement is not detected by this first slice.

No arbitrary source path is restored or reopened. Source reselection creates a
new ready item and, on explicit submission, a new job UUID. This is a new export
from the beginning. A later same-source restart feature must verify the full
source SHA256 and readiness; filename/stat matches alone are insufficient.

Startup corruption, unsupported schema or lease failure preserves the database
and fails opt-in startup with a recovery message. No automatic reset, destructive
migration, quarantine move or recreation. Version 0 may initialize only an empty
new database. Inspect an existing store's schema before any schema/journal-mode
writes. A structurally readable store with invalid individual records may serve
the valid records: omit invalid records from public results, report their count,
and preserve their rows. They still count toward capacity; never guess their
state or silently prune them. Migration tests cover both corrupt databases and
invalid records.

For runtime write failure, latch history state degraded and report it in status:

- Admission and queued settings/naming edits requiring a durable write return
  503 without publishing/enqueuing or changing the prior snapshot. Remove only
  the failed request's proven-owned newly staged upload, if one was created.
- Existing admitted work continues. Cancel/remove remain available and retain
  current owned-source cleanup behavior; attempt their history write once with
  the bounded timeout, then show that history was not saved if it failed.
- A completed live job remains downloadable through its current in-memory job
  route if its history commit fails. On a later restart it is conservatively
  interrupted unless completion was durably committed. No automatic promotion.
- Queue reorder remains memory-only; this slice never promises durable queue
  order. No unbounded retry thread, queue or poll loop. Recovery of the history
  writer is an explicit restart after current work/output accounting.

Use the existing queue-before-jobs lock order for mutations. A durable edit must
commit before its in-memory snapshot changes; a failed commit leaves it intact.
Keep hashing, filesystem discovery and model readiness outside those locks.
Never enter queue/jobs locks while holding a separate history-read transaction.

## Minimal API and UI

Add independent status.history_capabilities schema 1 with supported, enabled,
instance_name, state (`disabled`, `ready`, `degraded`), records_limit=1000,
source_retention=false, automatic_requeue=false and frame_resume=false. A degraded
state includes a bounded actionable warning without private filesystem paths.

`GET /api/history?limit=50&before=<opaque-cursor>` returns schema_version=1,
items and next_cursor. Limit is an integer 1..100; malformed input/cursor is 400.
Disabled history is 409. Return only prior-boot records, newest first by stable
creation-time/UUID ordering. Current jobs remain solely in the existing queue UI.
Each history item is a validated read-only DTO with job ID, creation/completion
UTC, recorded state/last-known state, basename, source size/hash, settings/naming
schemas and values, actual/planned output name and output_available. No current
queue position, cancel/edit endpoint, executable Job or public source path.

`GET /api/history/{job_id}/download` serves only a durably completed record whose
regular, nonsymlink file is directly inside current OUTPUTS and matches its saved
nonzero size. Reject traversal/absolute names and unavailable/changed files with
404. Size verification is not a cryptographic content-integrity claim. Reuse the
existing local Host/Origin boundary; no history write/delete/restart endpoints.

UI adds a capability-gated History from previous launches section: timestamp,
source basename, state, settings/naming details, available output download and
Use settings and reselect source. This action loads validated values, explains
that source files are not retained and requires explicit source selection/new
submission. Never match files by filename or auto-submit. Interrupted rows say
Interrupted; select source to create a new export, without a Resume button.
Show disabled/degraded and missing-output states truthfully. Older backends keep
their current queue UI without broken history requests.

## Implemented storage API and application adapter

`HistoryStore.from_environment(DATA_ROOT)` reads the opt-in variable without
opening or creating files. `HistoryStore(DATA_ROOT, instance_name)` is the explicit
constructor. `open()` (or the context manager) acquires the lease, validates the
schema, creates a fresh boot identity, and atomically marks valid unfinished rows
from earlier boots interrupted. `close()` releases the connection and OS lease;
it never unlinks the lease file. Failed startup releases ownership for every
exception, preserving unexpected exceptions instead of claiming an operational
database failure. Existing database and lease files must be regular files with
exactly one filesystem link; symlinks, redirected namespaces and hardlinks are
rejected before database connection. This is a local, cooperative ownership
boundary, not protection against an adversarial process racing filesystem edits.

The backend adapter calls these methods under its existing queue/jobs
ordering and publish the in-memory change only after a successful durable write:

| Method | Exact integration contract |
| --- | --- |
| `admit(metadata)` | Insert one ordinary queued export. Exact keys: `job_id`, `created_at`, `filename`, `media_type`, `source_size_bytes`, `source_sha256`, `settings`, `output_name_template`, `build_fingerprint`. UUID is canonical; basename has no paths/control characters; type is image/video; source size is a positive integer at most 8 GiB; SHA256 is lowercase 64 hex; fingerprint is null or SHA256. Created time is stable UTC epoch seconds. Settings are the existing exact 11-field schema; naming uses the existing strict parser. Reject duplicate UUIDs. Pruning and insertion share one transaction. |
| `edit_pending(id, *, settings=..., output_name_template=...)` | Replace either or both validated snapshots atomically, only for this boot's queued record. At least one value is required. Omitted values remain unchanged; explicit null naming is invalid. Caller must separately prove the worker has not dequeued the job. |
| `mark_processing(id, phase='initialization')` | Set processing/phase only for this boot's queued or processing record. Persist coarse transitions, not every frame. |
| `finish(id, state, *, output=None, error=None, execution=None, encoder=None, timings=None)` | States are completed/failed/canceled. Completed requires prior processing and exact output `{name, size_bytes}` matching the planned full-UUID filename and existing regular readable output of that size. Failed/canceled may also finalize queued work; neither exposes an output reference. Application must first finish encoder close/validation and preserve its current output ownership rules. |
| `list_prior(limit=50, before=None)` | Return schema 1, validated prior-boot read-only items, opaque next cursor and invalid-record count. Limit is exact integer 1..100. No executable jobs or sources are reconstructed. |
| `lookup_output(id)` | Return a confined Path only for a valid prior-boot completed output with saved nonzero size; return null when absent/changed/unreadable. Application maps this to a download response and rechecks its ordinary HTTP boundary. |
| `status()` | Return capability/state/limits and a bounded warning. Invalid records remain preserved and explicitly warn even if usable storage otherwise remains ready. |

Mutations return detached validated snapshots, never live Job instances. Exceptions
are `ValueError` for invalid values, `HistoryConflict` for duplicate/ineligible
identities, `HistoryCapacityError` for exhausted metadata quota,
`HistoryLeaseError` for ownership contention, and `HistoryUnavailable` for closed,
disabled or failed storage. SQLite runtime write failures roll back and latch
degraded: further writes fail, prior reads remain best effort. Validation/conflict
failures do not latch degraded. Startup corruption/unsupported structure never
opens a writable connection to the original database. Invalid individual rows
are hidden and preserved in place, count toward capacity and are never silently
pruned. Successful snapshots are limited to 64 KiB; execution/timing JSON is finite,
at most eight nested levels, 64 entries per container and 4096 characters per
string. Production defaults remain 1000 records and 90 days. Constructor quota
and clock overrides exist for bounded tests, not external path configuration.

The adapter includes `check_capacity()` before application upload staging,
streamed source hashing, durable-before-publication admission and queued edits,
failure-to-status mapping, coarse lifecycle hooks and read-only API/UI wiring.
FastAPI parses multipart uploads before the endpoint; this check does not reject
the network body before framework spooling. Capacity is rechecked in admission.
`mark_degraded()` latches adapter persistence failures while live work continues.
Public completed records add `completed_at` from their terminal update time;
page validation uses the store's `invalid_records` count. No source files are
read, retained or deleted by the store. Completed output lookup checks
confinement/readability/size, not cryptographic identity or atomic HTTP file-open
replacement races.

## Acceptance before promotion

Semantic unit tests and isolated process tests must prove:

- Opt-out makes no history files; valid/invalid stable instance names and source
  versus frozen DATA_ROOT paths behave explicitly, without production-server use.
- A second live process cannot acquire the same lease; another namespace can.
  Owner exit releases the OS lease. PID reuse, stale lock-file contents and clock
  changes never permit takeover. Test Windows for real; POSIX claims need its own
  execution evidence. Child FFmpeg does not inherit the lease descriptor.
- Crash before/after admission, queued edit, processing and completion commit
  produces truthful history. No crash point automatically requeues or promotes
  partial output. Current cancellation/source cleanup and output reservation pass.
- Corrupt/unknown-schema data is preserved; invalid row settings, naming, sizes,
  nonfinite values, IDs and paths are rejected from public history/downloads.
- Database failure rolls back admission/edits, keeps active work cancelable and
  exposes degraded status; no false saved/completed claim is made after restart.
- Metadata age/count/row bounds never delete files, unowned uploads, outputs or
  another instance's records. Missing/changed source/output is not guessed back.
- History never enters JOBS/JOB_QUEUE; read-only UI reuses settings with mandatory
  source reselection and never applies old queued controls or automatic retries.

Run complete CPU and GPU-environment suites, migration/corruption tests and an
owned isolated CPU server restart/download check. No real GPU run is needed for
this slice. Preserve legacy live servers and document exact evidence boundaries.

Reference mechanics: [Python Windows locking](https://docs.python.org/3/library/msvcrt.html#msvcrt.locking),
[Python POSIX locking](https://docs.python.org/3/library/fcntl.html#fcntl.flock),
[SQLite synchronization](https://www.sqlite.org/pragma.html#pragma_synchronous).

## Implemented integration sequence (design rationale)

Before this integration the app created its worker lazily in `ensure_worker()`
and it blocked forever in `JOB_QUEUE.get()`. Do not open `HistoryStore` at module import
or simply add `close()` to an ASGI shutdown callback: that could release the
instance lease while a daemon worker is still writing a terminal record. Keep
the standalone foundation intact and implement app integration in these
reviewable steps:

1. **Own worker lifetime.** Add an application lifespan that reads the opt-in
   instance setting, acquires `HistoryStore` before serving requests, and owns a
   stoppable worker. Stop accepting mutations before shutdown begins. Signal an
   active job to cancel, stop its owned encoder process using current behavior,
   and wake the worker if it is waiting on an empty queue. Join the worker before
   closing SQLite or releasing the lease. If processing does not stop, keep the
   store and lease open until the worker exits; never close underneath a live
   writer. Test disabled mode with no history files and opt-in startup contention
   with no second worker started.
2. **Admit before publishing.** Keep streaming the upload into its existing
   uniquely owned staging file while updating SHA256 and byte count. Before the
   upload body is read, reject when durable history cannot admit an export; after
   streaming and validation, transactionally admit the record before inserting
   into `JOBS` or `JOB_QUEUE`. On failure, remove only the staged file proven to
   belong to that request. Processed preview clips remain outside history.
3. **Persist queue edits at the existing lock boundary.** Under queue-before-jobs
   locks, commit settings/naming changes before mutating the in-memory `Job`.
   Persist queued removal as canceled before removing it from the queue. Keep
   expensive model checks and upload hashing outside these locks, then recheck
   pending membership as the current settings endpoint does.
4. **Persist coarse lifecycle boundaries.** Record processing when the worker
   claims an ordinary export, and record completed only after output validation
   and close; record failed/canceled at the existing exception boundary. Keep
   preview clips on their current in-memory path. A history write failure latches
   degraded status while admitted work continues; live completed output remains
   downloadable through `/api/jobs/{id}/download`.
5. **Expose previous-run history.** Add the independent capability object,
   prior-boot listing and confined output download routes, then a capability-
   gated UI for read-only details, available downloads, settings reuse and
   explicit source reselection. Do not reconstruct `Job` objects or refill the
   current queue from history.
6. **Prove restart behavior in an isolated CPU server.** Use a dedicated history
   instance and generated media. Complete one still export, leave one queued or
   active job for graceful shutdown, restart with the same instance and confirm
   terminal success remains available while unfinished work is marked
   interrupted. Verify download bytes and source hashes, then verify a concurrent
   second server fails startup without changing the first server's records.

Each step needs focused regressions before moving to the next. Keep source
retention, automatic requeue, queue-order persistence, project records, depth
caching, pause/checkpoints and frame resume out of this integration slice.
