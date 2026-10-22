"""Platform-specific writable paths for StereoLab."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping


@dataclass(frozen=True)
class PlatformPaths:
    data_root: Path
    state_root: Path
    cache_root: Path
    runtime: Path
    uploads: Path
    models: Path
    outputs: Path
    clip_proxies: Path
    depth_cache: Path
    calibration_file: Path
    history_root: Path


def resolve_paths(*, frozen: bool, root: Path, platform_name: str | None = None,
                  environ: Mapping[str, str] | None = None,
                  home: Path | None = None) -> PlatformPaths:
    """Resolve app storage roots while retaining the established source/Windows layout."""
    env = os.environ if environ is None else environ
    platform_name = sys.platform if platform_name is None else platform_name
    root = Path(root)

    if frozen and platform_name.startswith("linux"):
        # Keep concrete filesystem methods on Linux. A pure path is useful only
        # when simulating Linux on Windows, where Path follows Windows syntax.
        path = PurePosixPath if os.name == "nt" else Path
        home_value = str(Path.home() if home is None else home)
        if os.name == "nt":
            home_value = home_value.replace("\\", "/")
        home = path(home_value)

        def xdg(key: str, default: Path) -> Path:
            value = env.get(key)
            candidate = path(value) if value else None
            return candidate if candidate is not None and candidate.is_absolute() else default

        data = xdg("XDG_DATA_HOME", home / ".local" / "share") / "stereolab"
        state = xdg("XDG_STATE_HOME", home / ".local" / "state") / "stereolab"
        cache = xdg("XDG_CACHE_HOME", home / ".cache") / "stereolab"
        runtime = cache
        return PlatformPaths(
            data_root=data,
            state_root=state,
            cache_root=cache,
            runtime=runtime,
            uploads=cache / "uploads",
            models=data / "models",
            outputs=data / "outputs",
            clip_proxies=cache / "clip-proxies",
            depth_cache=cache / "depth-cache",
            calibration_file=state / "calibrations.json",
            history_root=state,
        )

    if frozen:
        data = Path(env.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "StereoLab"
    else:
        data = root
    runtime = data / ".runtime"
    return PlatformPaths(
        data_root=data,
        state_root=data,
        cache_root=runtime,
        runtime=runtime,
        uploads=runtime / "uploads",
        models=runtime / "models",
        outputs=data / "outputs",
        clip_proxies=runtime / "clip-proxies",
        depth_cache=runtime / "depth-cache",
        calibration_file=runtime / "calibrations.json",
        history_root=data,
    )
