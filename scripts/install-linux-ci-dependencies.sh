#!/usr/bin/env bash
# GitHub-hosted Ubuntu x86-64 setup. Keep stalled mirrors from consuming a job.
set -Eeuo pipefail

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This setup script requires an Ubuntu x86-64 CI runner.' >&2
    exit 2
fi

# Hosted runners use mirror+file source lists. The Azure mirror stalled during
# the native build; select the official HTTPS endpoints for these apt lists.
if [[ -f /etc/apt/apt-mirrors.txt ]]; then
    printf '%s\n' 'https://archive.ubuntu.com/ubuntu' | sudo tee /etc/apt/apt-mirrors.txt >/dev/null
fi
if [[ -f /etc/apt/apt-security-mirrors.txt ]]; then
    printf '%s\n' 'https://security.ubuntu.com/ubuntu' | sudo tee /etc/apt/apt-security-mirrors.txt >/dev/null
fi

APT_OPTIONS=(-o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
sudo apt-get "${APT_OPTIONS[@]}" update
sudo apt-get "${APT_OPTIONS[@]}" install --yes --no-install-recommends \
    ffmpeg language-pack-en libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
sudo locale-gen en_US.UTF-8
sudo update-locale LANG=en_US.UTF-8
