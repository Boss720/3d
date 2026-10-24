# StereoLab v0.2.0-beta.6

This Windows x64 beta broadens NVIDIA GPU support:

- DirectML setup covers DX12-capable NVIDIA adapters, including GTX 600-series and newer cards, subject to compatible Windows graphics drivers.
- CUDA device selection now treats detected NVIDIA models uniformly and prefers the GPU with the most VRAM instead of hard-coding a Tesla V100 preference.
- CUDA device IDs now come from the CUDA runtime, respecting visibility masks and device ordering. Physical `nvidia-smi` indices are diagnostic metadata only.
- The CUDA edition pins cuDNN 9.10.2.21 to retain older Maxwell/Pascal/Volta kernels; build preflight rejects a different cuDNN version. Devices below compute capability 5.2 or with unknown capability are excluded from CUDA selection, with guidance to use DirectML or CPU.
- Auto preserves separate failure cooldowns for CUDA and DirectML; a successful fallback no longer clears the failed backend's cooldown.
- The DirectML installer takes its launcher from the selected package and refuses an incomplete package.
- CUDA packaging excludes obsolete pip backup DLLs, keeping the selected runtime version intact.
- Linux CUDA policy tests use explicit device inventories, covering CPU-only and CUDA-capable environments without depending on host hardware.

GPU execution is reported as verified only after a named-model inference succeeds. CUDA compatibility depends on the specific card, driver, and runtime package; this release has not been validated on every NVIDIA model.

The rebuilt DirectML and CUDA editions each passed strict named Small inference, generated XREAL SBS video/audio export, cache provenance, restart/history, and save-copy acceptance on a GeForce GTX 1650 Ti. Both installers passed full acceptance from installed paths containing spaces, with all payload hashes matching (3 DirectML files and 200 CUDA files), followed by successful uninstall and registration cleanup. CUDA also passed with a restricted PATH to exercise its bundled runtime. Full CPU and DirectML suites each ran 709 tests successfully, with 9 and 5 skips respectively; the CUDA engine suite passed 58 tests. NVENC acceptance permits the verifier's explicit 60 dB per-frame tolerance while requiring exact audio, geometry, frame count, and timing. These checks cover one NVIDIA GPU model and do not establish compatibility with every NVIDIA card. See `RELEASE_READINESS.md` for exact artifact hashes and remaining qualification limits.

---

# StereoLab v0.2.0-beta.3

This Windows x64 beta fixes issues found during review:

- Input-detail calibration now warms and measures each dimension-keyed DirectML session before changing input size, so repeated samples can be collected.
- Video preflight detects HDR transfer functions and non-square pixel aspect ratios and explains the supported SDR color path and how to convert unsupported inputs.
- The queue refreshes on window focus and tab return, with a 15-second idle refresh for work submitted from other clients.
- Queue rendering no longer duplicates depth-cache labels or warnings; removed queue items release retained thumbnails, and filmstrip arrows are identified as sampled-thumbnail navigation.

FFprobe now collects sample aspect ratio, color range, matrix, transfer, and primaries. SDR video is decoded to 8-bit BGR and exported as limited-range YUV420. HDR and anamorphic video are rejected during preflight with conversion guidance.

Validation: full suites passed in both environments (633 tests each; 7 CPU skips and 3 DirectML skips), with `ResourceWarning` treated as an error. The packaged EXE completed strict Small DirectML inference and generated XREAL SBS video at 3840×1080 with audio, then passed cache, restart, history, and save-copy checks. The beta.3 installer was installed into an isolated directory, its EXE and license hashes matched the candidate, the installed EXE passed the same packaged acceptance, and uninstall removed the installed files. Reports and SHA-256 checksums are recorded in `RELEASE_READINESS.md` and the release assets.

This remains a beta. Generated-media and same-host checks do not establish clean-machine installation, physical XREAL 1S playback, or screen-reader usability. Original StereoLab code is MIT licensed; third-party software and model terms remain separate. See RELEASE_READINESS.md and RELEASE_DEPENDENCIES.md for the remaining review gaps.

---

# StereoLab v0.2.0-beta.2

This Windows x64 beta fixes issues found during code review:

- Canceling during the final encoder write now stops owned I/O while the conversion workers finish, avoiding a stuck cancellation.
- Canceled custom-model uploads and failed model replacement now remove their temporary files without deleting an existing model.
- Build fingerprints now include GPU hardware selection code, so changes to that code invalidate persistent depth reuse.
- The build script accepts an explicit Python environment and separate output/work directories, and copies the launcher and MIT license beside the EXE.

The regression suite also creates rotated video with an explicit display matrix so current FFmpeg versions actually exercise rotation.

The packaged verifier keeps byte identity as its default. Its explicit hardware-variation option compares each decoded NVENC frame at a minimum 60 dB PSNR while retaining exact audio, geometry, frame count, and timing checks. Independent identical-input encoding reproduced small variations in the external FFmpeg/NVENC stack; a separate real DirectML source run confirmed identical fresh/cache raw render hashes.

Validation: 628 tests in each CPU and DirectML environment; generated 3840x1080 XREAL SBS conversion, cache/restart/history, browser checks at 320–1280 pixels, and isolated installer install/launch/uninstall. The final packaged cache/restart repeats also had identical encoded, decoded video, and audio hashes.

The setup installer installs per user. The standalone EXE serves the local browser interface. Video conversion requires separately installed FFmpeg; model weights download separately.

This remains a beta. Generated-media and same-host checks do not establish clean-machine installation, physical XREAL 1S playback, or screen-reader usability. Original StereoLab code is MIT licensed; third-party software and model terms remain separate. See RELEASE_READINESS.md and RELEASE_DEPENDENCIES.md for the remaining review gaps.
