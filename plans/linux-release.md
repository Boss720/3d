# StereoLab Linux release plan

Updated 22 October 2026. Status: implementation in progress. Linux CPU portability and native packaging support are on the Linux branch. The first native CI run failed during the Ubuntu 22.04 source suite before packaging; there is still no Linux archive or successful Linux inference/media evidence. Static checks and Windows platform simulations are not Linux release evidence.

## Current implementation snapshot

The working tree now contains XDG path handling, Linux desktop-action fallbacks, CPU-only Linux capability reporting, Linux FFmpeg discovery and version-compatible passthrough handling, a PyInstaller one-directory spec, a native build script, ELF dependency inspection, and an Ubuntu 22.04/24.04 test workflow. Package acceptance checks read-only installation, XDG data/state/cache placement, CPU-only runtime/catalog claims, build identity, unresolved ELF dependencies and graceful shutdown. CI acceptance also downloads the pinned Small model and exports an eight-frame generated audio/video clip, checking the reported CPU provider, output geometry, audio, duration and full decode. Three review findings are fixed locally with regressions: Auto-to-CPU is no longer mislabeled as fallback, stale AMD decode requests normalize to software on Linux, and Linux model descriptions are CPU-only. The first workflow run failed in the Ubuntu 22.04 source suite before packaging; the full Windows-host suite now passes 658 tests with 8 skips. There is still no Linux archive or successful Linux inference/media evidence.

The implementation is on `codex/linux-cpu-build` at `c6a2b5c`, pushed after the user authorized a Linux branch. It includes the then-current `v0.2.0-beta.3` master commit, which was one commit ahead of remote `master` when this branch was created. The first GitHub Actions run started on Ubuntu 22.04 and exposed Linux suite problems before packaging: Windows DirectML policy tests were not simulating Windows, FFmpeg/FFprobe compatibility differed from the Windows build, and one AMD-setting test assumed Windows. The current follow-up working tree isolates those platform tests, selects the installed FFmpeg passthrough option, falls back to bounded default FFprobe stream metadata when old versions reject the side-data selector, and resets preview timestamps. The Windows-host suite now passes 658 tests with 8 skips. The first Linux run has not built an archive; the follow-up commit and a passing Linux rerun are still required.

The next execution slice is to commit and push the verified compatibility fixes, then let CI complete its Ubuntu 22.04/24.04 source suites, native build, checksum validation, named-model CPU inference and generated audio/video conversion against the exact archive. The current host has no usable Linux runtime. Do not publish a GitHub release as part of this slice; third-party and collected native-library notices still need review against the built artifact.

## Release scope

Ship a Linux x86-64 CPU beta first, preserving the existing local browser interface, neural models, conversion queue, projects, history and depth cache. Add an optional NVIDIA CUDA edition after the CPU artifact passes release acceptance. Windows remains supported throughout.

Proposed initial support targets: Ubuntu 22.04 and 24.04 LTS on x86-64. Confirm wheel availability and native-library compatibility before fixing this matrix. Use Python 3.12 as the initial build candidate; pin the exact patch and dependency closure after a clean Linux resolution. Other distributions, ARM64, AMD/Intel inference acceleration, AppImage, Flatpak and distro installers are subsequent work.

First deliverable: `StereoLab-<version>-linux-x86_64-cpu.tar.gz`, containing a PyInstaller one-directory application, launcher, license/third-party notices and Linux instructions. Users should not need a Python installation. Require a separately installed FFmpeg/FFprobe with the supported encoders; do not redistribute an unreviewed media binary. Linux may ship as a beta while broader production-readiness work remains open, provided its limits are explicit.

## Current code and gaps

Source inspection of the current working tree found:

| Area | Current implementation | Linux release gap |
| --- | --- | --- |
| App entry point | `app_launcher.py` binds to `127.0.0.1`; the package launcher supports `--port` | Native start, occupied-port behavior, restart and shutdown are unverified |
| Writable files | `platform_paths.py` resolves frozen Linux XDG data, state and cache locations | Verify defaults, overrides, read-only installs and user-data preservation in a Linux package |
| Neural inference | Linux advertises CPU and Auto; catalog copy is CPU-specific; strict GPU modes fail before inference | Run named-model inference and calibration from the final package on Linux |
| Desktop actions | `local_workspace.py` uses Linux desktop helpers with headless-safe behavior | Test in a real desktop session and on headless systems |
| Media tools | FFmpeg discovery uses the override or PATH and avoids the Windows-only convenience fallback | Verify actual codecs, audio, timing, cancellation and failure recovery on Linux |
| Disk coordination | History and depth cache contain POSIX `fcntl` lease paths | Exercise native multiprocess contention and crash recovery |
| Packaging and verification | Linux PyInstaller spec, build script, ELF scanner, package verifier and two-version Actions workflow exist | Run the native build, inspect complete notices, and accept the same archive on both targets |
| Observability | Unsupported platform process/GPU metrics are reported without claiming Windows-only evidence | Confirm useful diagnostics and process shutdown on Linux |

The checkout contains unrelated edits and release-review artifacts. Implement this plan in scoped changes after recording the exact source baseline; do not overwrite those files. Existing Windows reports and test counts are historical evidence, not Linux acceptance.

## Milestone 1: establish a Linux source baseline — medium

1. Create a clean Linux build/test environment on the oldest proposed supported target. Record OS, architecture, Python, native libraries and exact package versions. Generate and select `en_US.UTF-8`; the ONNX Runtime Linux package documents this locale as a system requirement.
2. Install the CPU requirements in an isolated virtual environment, plus test dependencies. Record wheel availability and `pip check` results. Investigate GUI-library imports from OpenCV; evaluate `opencv-python-headless` only as a separately tested dependency change, avoiding simultaneous OpenCV distributions.
3. Run the full suite with ResourceWarning promoted to an error. Classify genuine Windows-only skips separately from portable failures. Run suites serially where they share runtime files.
4. Launch from source, fetch the UI and capability endpoints, then complete an approximation image and short audio/video conversion.

**Exit gate:** repeatable source installation and recorded Linux test results; every remaining portable failure has a bounded fix in Milestone 2. Windows-only APIs must not prevent import or CPU launch.

## Milestone 2: close portability gaps — medium/large

### Storage and filesystem behavior

Introduce a small platform-path module used by the server and acceptance harness. Keep Windows paths compatible and make the source-development storage policy explicit. For packaged Linux runs, propose these XDG locations, honoring absolute overrides and ignoring invalid relative values:

- Data: `${XDG_DATA_HOME:-~/.local/share}/stereolab` for models and managed outputs.
- State: `${XDG_STATE_HOME:-~/.local/state}/stereolab` for history, calibration and logs.
- Cache: `${XDG_CACHE_HOME:-~/.cache}/stereolab` for regenerable depth/proxy caches and temporary uploads. History must tolerate unavailable temporary sources after cleanup.
- Configuration: `${XDG_CONFIG_HOME:-~/.config}/stereolab` if persistent preferences are added.

Expose the resolved paths in diagnostics and storage UI. Audit assumptions that all categories share one root; keep atomic temporary writes beside their destination so separate XDG filesystems do not break rename/replace. Include new runtime modules in `build_identity.py` source fingerprints so persistent caches invalidate correctly.

Test spaces, Unicode, case sensitivity, read-only installation directories, insufficient disk space, absolute/relative XDG overrides, symlink rejection and distinct filesystems. Exercise POSIX history/cache leases with separate processes, contention, process death and restart. Test source relinking without assuming Windows path conventions.

### Runtime and desktop behavior

- Keep CPU and Auto usable without DirectML/DXGI. Disable or clearly reject unavailable strict GPU modes before a costly job starts; make UI copy and capability fields platform-aware.
- Implement open-folder using an available Linux desktop opener; attempt file selection through a supported desktop interface, with a documented containing-folder fallback. Handle missing desktop sessions/helpers without breaking downloads or Save copy. Keep subprocess arguments separate and avoid shell interpolation.
- Retain `VIDEO3D_FFMPEG` and PATH discovery. Check FFmpeg/FFprobe execution and required H.264/audio capabilities with actionable errors. Leave NVENC optional and independently probed.
- Verify SIGINT/SIGTERM during decode, inference and encode: stop admission, cancel owned work, terminate/reap media children, join workers, then release database/cache leases. Distinguish process absence from inability to inspect it.
- Preserve loopback binding, Host checks and same-service browser write-origin checks.

**Exit gate:** complete CPU suite passes on Linux and Windows with justified platform skips; real Linux image/video jobs, cancellation, restart, history, cache reuse, storage and desktop fallbacks pass. No Windows feature regressions from shared changes.

## Milestone 3: build the CPU artifact — medium

1. Add `scripts/build-linux.sh`, a Linux PyInstaller spec (or a shared spec with explicit platform branches), and a launcher. Reuse static asset, license and build-manifest collection. Build on Linux; the Windows EXE is not a Linux artifact.
2. Prefer a one-directory tarball initially, keeping native dependency inspection and launch diagnostics straightforward. Set executable permissions in the archive and support extraction/launch under an arbitrary directory without writing into that directory.
3. Pin build tools and runtime dependencies in Linux-specific lock/constraint files. Record source commit, dirty-tree status, fingerprint, architecture, OS, Python and resolved dependency versions in build evidence. Public candidates must correspond to a reviewable committed baseline.
4. Inspect the actual collected ELF libraries and transitive dependencies. Document the glibc/native baseline and required system libraries; exclude host GPU drivers. Treat compatibility on newer targets as a test requirement.
5. Regenerate third-party inventory and full notices from the actual Linux artifact. Include Python, ONNX Runtime, OpenCV and collected native components. Keep model weights downloadable, with their existing provenance and usage restrictions visible.
6. Add an optional user-local `.desktop` entry and icon with correct quoting. Provide terminal launch instructions and the URL if browser opening is unavailable. Do not auto-install a service or require root.

**Exit gate:** the artifact launches on clean target machines without Python or development environments, serves its bundled UI, uses only user-writable storage and passes Milestone 4. The archive contains complete release materials and a SHA-256 checksum.

## Milestone 4: qualify and publish the Linux CPU beta — large

Port the packaged acceptance harness with Linux process identity checks and isolated XDG roots. Run the exact final archive on both supported distributions; use a real desktop session for browser/file-manager checks and a clean VM for installation isolation. WSL/container results can support source/CI checks but do not establish desktop or physical GPU support.

Required acceptance evidence:

- Cold launch, bundled assets, alternate port, occupied port, repeat launch, graceful stop and restart.
- Approximation and a real named Small-model image/video conversion; deliberate failed job followed by a successful queued job.
- Preview, queue edits, pause/resume, cancellation, downloads, Save copy, project relinking, opt-in history and persistent depth-cache reuse/invalidation.
- Full output decode, frame count, expected layout/dimensions, audio presence and timing; include mixed-rate, portrait and representative real footage. Record existing media-fidelity limitations.
- At least one representative long neural export, with wall-clock duration, memory/disk observations and cancellation checks. Agree the workload and acceptance limits before the run; a short synthetic clip does not satisfy this gate.
- Missing FFmpeg, unavailable model/download failure, unwritable/full destination, non-ASCII paths, headless desktop actions and shutdown during active encoding.
- Named-model inference with the documented `en_US.UTF-8` locale; verify that missing locale data produces clear setup guidance rather than a silent runtime failure.
- Upgrade to a replacement build preserves user data and invalidates incompatible cache entries; removal of application files preserves exports/models unless explicitly deleted by the user.

Save artifact hashes, environment details, logs, test reports and decoded-media measurements together. Update README and release readiness with the supported Linux matrix, install/remove instructions, FFmpeg setup, paths and known limits. Reconcile stale release-status statements rather than copying Windows release claims.

**Publication gate:** all mandatory checks pass for the exact candidate, notices are complete, and any remaining limitations are explicitly recorded. Upload the versioned archive, checksums and release notes through the normal release workflow when publication is authorized. Keep a prior candidate and rollback instructions.

## Milestone 5: optional NVIDIA CUDA edition — large, after CPU beta

Use a separate GPU dependency environment/artifact; never install CPU, CUDA and DirectML ONNX Runtime distributions together. Select and pin ONNX Runtime/CUDA/cuDNN versions from the upstream compatibility matrix, checking Python wheels, minimum driver and the intended GTX/RTX hardware. Do not promise GTX 1650 Ti support until that exact stack executes successfully.

Refactor provider selection and session options so Windows DirectML and Linux CUDA have separate implementations. Adapter IDs, memory discovery, provider configuration and graph-optimization settings must be provider-specific. Preserve one resident session, serialized inference, conservative unknown-memory policy, Auto CPU fallback with its reason, strict GPU failure behavior and cache/calibration identity separation by backend/runtime/device.

Validate named-model inference and exports on physical Linux NVIDIA hardware, including Small and the existing Large memory policy, low-memory failures, recovery, cancellation and sustained jobs. Record actual provider and inference evidence; a registered CUDA provider, `nvidia-smi`, or a working NVENC encoder alone is insufficient. Include a no-GPU/missing-library test and any redistributed CUDA component notice review.

**Exit gate:** publish a separate CUDA beta only after real inference, artifact isolation and hardware-specific acceptance pass. CPU beta delivery does not depend on CUDA availability. AMD/Intel acceleration requires its own later feasibility and qualification plan.

## Delivery sequence and review

The local implementation is present for XDG storage, desktop actions, Linux CPU capability/UI behavior, FFmpeg discovery, a one-directory PyInstaller build, ELF inspection, and isolated-package launch verification. Focused platform-simulation tests and both full Windows environments pass. This does not establish a Linux source baseline or a Linux package.

Remaining execution order:

1. Run the full source suite on Ubuntu 22.04 and 24.04; record skips and fix Linux-specific failures.
2. Resolve and pin the Linux Python dependency closure on Ubuntu 22.04, build the one-directory archive, inspect native dependencies and produce versioned package evidence.
3. Verify the archive checksum and launch that exact archive on both Ubuntu versions. The CI smoke now installs the pinned Small model and converts a generated short clip; native cancellation/recovery, representative real footage and longer exports remain separate acceptance work.
4. Complete native/third-party notice review and desktop-session checks; update supported-target instructions and release readiness from measured evidence.
5. Publish the CPU beta only after the acceptance and notice gates pass and publication is authorized. Plan CUDA as a separate later artifact.

Keep Linux source tests ahead of each package build. Re-run the Windows suite for shared runtime changes and Linux artifact acceptance after packaging changes. Do not run conversion benchmarks concurrently on the same GPU. No reliable calendar estimate is available until the first native Linux baseline is established.

## Primary references

Consulted 22 October 2026; recheck when selecting the build stack:

- [PyInstaller Linux compatibility guidance](https://pyinstaller.org/en/stable/usage.html): it does not bundle glibc, so build on the oldest supported Linux version; Python 3.12 is within the supported Python range.
- [ONNX Runtime install requirements](https://onnxruntime.ai/docs/install/): includes the Linux `en_US.UTF-8` locale requirement and separate CPU/CUDA packages.
- [ONNX Runtime compatibility guidance](https://onnxruntime.ai/docs/reference/compatibility/).
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/).
