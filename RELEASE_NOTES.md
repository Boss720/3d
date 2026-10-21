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
