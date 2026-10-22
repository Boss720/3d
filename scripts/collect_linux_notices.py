#!/usr/bin/env python3
"""Copy installed Python license/notice files into a release directory.

This inventories files present in installed distribution metadata. It does not
determine whether the resulting inventory is legally complete.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import sys
import sysconfig
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


MANIFEST_NAME = "THIRD_PARTY_NOTICES.json"
FLATBUFFERS_FALLBACK_VERSION = "25.12.19"
FLATBUFFERS_FALLBACK_URL = "https://raw.githubusercontent.com/google/flatbuffers/v25.12.19/LICENSE"
FLATBUFFERS_FALLBACK_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
NOTICE_BASENAME = re.compile(
    r"^(?:licen[cs]e|copying|notice|copyright|third[-_.]?party[-_.]?notices?)(?:[-_.].*)?$",
    re.IGNORECASE,
)


def _safe_relative_path(value: object) -> PurePosixPath:
    """Validate a metadata path before using it as a source or output path."""
    raw = str(value)
    windows = PureWindowsPath(raw)
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or windows.is_absolute() or windows.drive or windows.root:
        raise ValueError(f"unsafe distribution path: {raw!r}")
    # importlib.metadata returns pathlib.Path objects, whose normal string form
    # uses the host separator. Normalize only after checking absolute/drive forms.
    path = PurePosixPath(raw.replace("\\", "/"))
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe distribution path: {raw!r}")
    return path


def _inside(path: Path, root: Path, *, error: str) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(error) from exc
    return resolved


def _distribution_identity(distribution: object) -> tuple[str, str]:
    metadata = getattr(distribution, "metadata", {})
    name = metadata.get("Name") or getattr(distribution, "name", None) or "unknown"
    version = str(getattr(distribution, "version", "unknown"))
    return str(name), version


def _is_notice_file(relative: PurePosixPath) -> bool:
    return bool(NOTICE_BASENAME.match(relative.name))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_record(source: Path, destination: Path, relative: str, bundle: Path) -> dict[str, str]:
    safe_destination = _inside(destination, bundle, error="output path escapes requested root")
    if not source.is_file():
        raise FileNotFoundError(f"listed notice file is missing: {source}")
    safe_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, safe_destination)
    return {
        "source_relative_path": relative,
        "source_path": str(source.resolve()),
        "copied_to": safe_destination.relative_to(bundle.resolve()).as_posix(),
        "sha256": _sha256(safe_destination),
    }


def _stdlib_license_path() -> Path | None:
    stdlib = sysconfig.get_path("stdlib")
    roots = [Path(stdlib)] if stdlib else []
    roots.extend(Path(prefix) for prefix in dict.fromkeys((sys.base_prefix, sys.prefix)))
    for root in roots:
        for name in ("LICENSE", "LICENSE.txt", "COPYING"):
            candidate = root / name
            if candidate.is_file():
                return candidate
    return None


def _flatbuffers_fallback_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "third_party" / "linux" / "flatbuffers" / FLATBUFFERS_FALLBACK_VERSION / "LICENSE"
    )


def collect_notices(
    output_dir: str | Path,
    *,
    distributions: Iterable[object] | None = None,
    stdlib_license: str | Path | None = None,
    selected_distributions: Iterable[str] | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Copy license/notice files and write a provenance/hash manifest.

    ``distributions`` and ``stdlib_license`` are injectable for deterministic
    fixture use. Selection matches installed distribution names, case-insensitively,
    after normalizing runs of hyphens, underscores, and dots.
    """
    requested_root = Path(output_root) if output_root is not None else Path(output_dir)
    bundle = _inside(Path(output_dir), requested_root, error="output path escapes requested root")
    bundle.mkdir(parents=True, exist_ok=True)
    # Recheck after mkdir to catch an existing symlink at the destination.
    bundle = _inside(bundle, requested_root, error="output path escapes requested root")

    def normalize(value: str) -> str:
        return re.sub(r"[-_.]+", "-", value).casefold()

    selected = {normalize(str(name)) for name in selected_distributions or ()}
    all_distributions = list(distributions if distributions is not None else importlib.metadata.distributions())
    rows: list[dict[str, object]] = []
    matching_names: set[str] = set()
    for distribution in sorted(all_distributions, key=lambda item: tuple(map(str.casefold, _distribution_identity(item)))):
        name, version = _distribution_identity(distribution)
        normalized_name = normalize(name)
        if selected and normalized_name not in selected:
            continue
        matching_names.add(normalized_name)
        files = getattr(distribution, "files", None) or ()
        candidates: list[tuple[PurePosixPath, Path]] = []
        for item in files:
            # Distribution metadata can list non-notice files outside its
            # package root. Ignore them; validate paths before using candidates.
            lexical = PurePosixPath(str(item).replace("\\", "/"))
            if not _is_notice_file(lexical):
                continue
            relative = _safe_relative_path(item)
            candidates.append((relative, Path(distribution.locate_file(item))))

        records: list[dict[str, object]] = []
        prefix = f"distributions/{len(rows):04d}-{normalized_name}-{version}"
        for relative, source in sorted(candidates, key=lambda pair: pair[0].as_posix().casefold()):
            records.append(
                _copy_record(source, bundle.joinpath(prefix, *relative.parts), relative.as_posix(), bundle)
            )
        if not records and normalized_name == "flatbuffers" and version == FLATBUFFERS_FALLBACK_VERSION:
            fallback = _flatbuffers_fallback_path()
            if not fallback.is_file() or _sha256(fallback) != FLATBUFFERS_FALLBACK_SHA256:
                raise ValueError(
                    "vendored FlatBuffers 25.12.19 license is missing or does not match its recorded SHA256"
                )
            fallback_record = _copy_record(
                fallback,
                bundle.joinpath(prefix, "LICENSE"),
                "LICENSE",
                bundle,
            )
            fallback_record["provenance"] = {
                "source_url": FLATBUFFERS_FALLBACK_URL,
                "source_version": FLATBUFFERS_FALLBACK_VERSION,
                "sha256": FLATBUFFERS_FALLBACK_SHA256,
            }
            records.append(fallback_record)
        rows.append({
            "name": name,
            "version": version,
            "files": records,
            "missing_notice": not bool(records),
        })

    missing_selection = sorted(selected - matching_names)
    if missing_selection:
        raise ValueError("requested distributions not installed: " + ", ".join(missing_selection))

    if stdlib_license is None:
        stdlib_license = _stdlib_license_path()
    stdlib_record: dict[str, object] = {"missing_notice": True, "files": []}
    if stdlib_license is not None and Path(stdlib_license).is_file():
        source = Path(stdlib_license)
        item = _copy_record(source, bundle / "python-stdlib" / source.name, source.name, bundle)
        stdlib_record = {"missing_notice": False, **item}

    manifest: dict[str, object] = {
        "format_version": 1,
        "generated_by": "scripts/collect_linux_notices.py",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "scope_note": "Inventory of notice files found in installed distribution metadata; legal completeness is not assessed.",
        "distributions": rows,
        "stdlib": stdlib_record,
    }
    manifest_path = _inside(bundle / MANIFEST_NAME, bundle, error="output path escapes requested root")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="directory for copied notices and manifest")
    parser.add_argument(
        "--distribution", action="append", dest="distributions",
        help="only include this installed distribution name (repeatable)",
    )
    args = parser.parse_args(argv)
    try:
        manifest = collect_notices(args.output, selected_distributions=args.distributions)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    missing = sum(bool(row["missing_notice"]) for row in manifest["distributions"])
    print(f"Wrote {args.output / MANIFEST_NAME}; distributions={len(manifest['distributions'])}; missing={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
