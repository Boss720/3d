# Phase 0.1 process cleanup observation subgate

## Frozen scope

Add read-only Windows process-tree observation to the existing 30-second stress
runner. The observer may report only descendants of the API PID returned by that
API's `/api/diagnostics/resources` endpoint. It must identify FFmpeg by executable
basename, sample while a video export runs, and report whether an FFmpeg child is
still visible within five seconds after the job reaches terminal status. This
post-terminal observation does not establish the precise exit time. If the last
observed job snapshot is still active, timed out, or unavailable, report the
process result as unverified and do not start a post-terminal window.

Do not signal, terminate, restart, or otherwise manage any process. Do not inspect
or act on descendants of any other API. A live run may target only a newly launched
private server owned for that run. Preserve current CPU/RAM and whole-device
sampling behavior, and report process-tree observation as unverified if the Windows
snapshot API is unavailable or the server PID cannot be validated. Require an
explicit loopback base URL and expected API PID before the stress runner can submit
any work; compare the live diagnostics PID and the observed executable with that
expectation. Do not default the stress runner to the existing 8770 service.

The existing cancellation trigger records a frame threshold but omits phase. Tighten
it to require API state `processing`, phase `conversion`, at least five processed
frames, and fewer processed than total frames. Save all values at the trigger and
describe this as cancellation during conversion; frame progress alone does not
identify a narrower inference substage. A passing cancellation does not prove
cancellation during encoding. Never infer per-process VRAM from `nvidia-smi` values
reported as `[N/A]` under Windows WDDM.

Retain observed process identities through the run. If an observed FFmpeg process
is no longer a descendant after a parent exits, check its PID together with a
creation-time identity; a reused PID or unreadable identity is unverified, not proof
that the original process exited.

## Acceptance

- Pure tests cover root exclusion, direct and nested descendants, unrelated
  processes, stable ordering, and absent/malformed roots. Windows identity checks
  cover PID reuse and a child that remains after its parent exits.
- A Windows integration test observes a test-owned child process and its nested
  child, then confirms disappearance after the test reaps them.
- The stress report records the API PID, sampled FFmpeg child identities, whether
  a child was observed during each completed export, and whether it remained
  visible within five seconds after terminal status. Missing observations, a
  missing terminal snapshot, or unavailable diagnostics remain explicit and
  cannot be described as proof of encoder cleanup.
- Existing stress checks still require an idle server before every submitted job;
  submissions require explicit URL/PID matching, the cancellation snapshot proves
  conversion phase, and API cancellation stays separate from process-tree evidence.
- Focused process/stress tests, both full CPU/GPU-environment suites, compile and
  diff checks pass. Any real server run uses a newly owned private instance and
  never submits to the existing services.

## Deferred evidence

No representative real-footage sample is currently confirmed in the workspace.
The existing stress sources are generated test fixtures. A real-media quality
claim remains deferred until authorized footage is available. The current NVIDIA
device has four existing server PIDs and reports `[N/A]` per-process memory, so no
new GPU workload is authorized by this diagnostic scope. Exact process VRAM and
multi-hour reliability remain open even after this subgate.

## Authorized continuation — 20 October 2026

The user requested commit then resume with army. Fresh NVIDIA inventory now shows
0 MiB used, 0% utilization, and no compute processes. The previous busy-device
restriction above records the earlier checkpoint. During this idle window the
coordinator may launch one new private GPU API, with isolated writable directories
and history disabled, and run the existing generated all-model stress gate.
Existing accepted models are reused without installation or download. Before any
submission, match the API PID and creation identity and confirm an idle queue.
Archive terminal jobs and output evidence, then gracefully stop only this owned
server. Retain before/after inventory and global identity observations. Generated
fixtures do not satisfy representative-footage or multi-hour acceptance.
