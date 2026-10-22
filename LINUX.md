# StereoLab for Linux

The planned Linux CPU build targets x86-64 with Ubuntu 22.04 as the glibc compatibility baseline. Ubuntu 24.04 is also included in the configured test matrix. These targets have not yet passed native package acceptance. The app runs as a local web app and listens only on `127.0.0.1`; open the printed address in a browser. Images and videos stay on this computer. Neural models download only after you choose **Install**.

The CPU package does not require Python. Install FFmpeg and FFprobe from your distribution to convert video. For Ubuntu, run `sudo apt install ffmpeg language-pack-en`, then generate and select the locale required by ONNX Runtime:

```sh
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
```

Start a new login session after changing the system locale. The FFmpeg build must include `libx264` and AAC support. The application reports an actionable error when a required media tool is unavailable.

Download and extract the `StereoLab-<version>-linux-x86_64-cpu.tar.gz` archive, then start it from a terminal:

```sh
tar -xzf StereoLab-<version>-linux-x86_64-cpu.tar.gz
cd StereoLab
./stereolab
```

Use `./stereolab --port 8766` to choose another local port. Press Ctrl+C to stop the server. If no browser opens automatically, visit the printed `http://127.0.0.1:<port>` address.

The frozen Linux app uses the XDG base directories: models and outputs under `$XDG_DATA_HOME/stereolab` (default `~/.local/share/stereolab`), history under `$XDG_STATE_HOME/stereolab/.runtime/history` (default `~/.local/state/stereolab/.runtime/history`), calibration under the state root, and temporary/cache files under `$XDG_CACHE_HOME/stereolab` (default `~/.cache/stereolab`). Removing the extracted application directory leaves these personal files in place.

This CPU package does not provide Linux GPU inference. Choose **CPU** or **Auto** in execution settings. Windows DirectML acceleration remains a separate Windows build. A Linux CUDA build requires a separately qualified NVIDIA package and is not included here.

The project source uses the MIT License in the `LICENSE` file. Python packages, native libraries and downloadable model files carry separate terms. `build-report.json`, `requirements-resolved.txt` and `native-dependencies.json` record the builder, installed Python package versions, and collected ELF dependency links; they are not a substitute for reviewing third-party license texts and native library notices before redistribution. Base and Large depth models are marked non-commercial. Physical headset playback has not been verified.
