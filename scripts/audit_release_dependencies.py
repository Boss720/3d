"""Write a metadata-based dependency inventory for the frozen release build."""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

from packaging.requirements import Requirement
from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_identity import build_manifest


RUNTIME_REQUIREMENTS = ROOT / "requirements-gpu.txt"
BUILD_ROOTS = ("PyInstaller",)
DEFAULT_EXE = ROOT / ".runtime/vfr-wave/production-package-decoder-recovery/dist/StereoLab.exe"


def requirement_roots(path: Path) -> tuple[str, ...]:
    roots = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.split("#", 1)[0].strip()
        if value:
            roots.append(Requirement(value).name)
    return tuple(roots)


def closure(roots: tuple[str, ...]) -> dict[str, metadata.Distribution]:
    found: dict[str, metadata.Distribution] = {}
    pending = list(roots)
    while pending:
        requested = pending.pop()
        try:
            dist = metadata.distribution(requested)
        except metadata.PackageNotFoundError:
            found[requested.lower()] = None  # type: ignore[assignment]
            continue
        name = dist.metadata.get("Name", requested)
        key = name.casefold().replace("_", "-")
        if key in found:
            continue
        found[key] = dist
        for raw in dist.requires or ():
            dependency = Requirement(raw)
            if dependency.marker is None or dependency.marker.evaluate({"extra": ""}):
                pending.append(dependency.name)
    return found


def license_label(dist: metadata.Distribution | None) -> str:
    if dist is None:
        return "NOT INSTALLED"
    expr = dist.metadata.get("License-Expression")
    if expr:
        return expr
    legacy = (dist.metadata.get("License") or "").strip()
    classifiers = [value.rsplit(" :: ", 1)[-1] for value in dist.metadata.get_all("Classifier", ())
                   if value.startswith("License :: OSI Approved ::")]
    if classifiers:
        return "; ".join(sorted(set(classifiers)))
    return legacy or "No SPDX license field in installed distribution metadata"


def markdown_table(title: str, items: dict[str, metadata.Distribution | None]) -> list[str]:
    lines = [f"## {title}", "", "| Distribution | Version | License metadata |", "| --- | --- | --- |"]
    for key, dist in sorted(items.items()):
        if dist is None:
            lines.append(f"| `{key}` | missing | NOT INSTALLED |")
            continue
        name = dist.metadata.get("Name", key)
        version = dist.version
        license_text = license_label(dist).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{name}` | `{version}` | {license_text} |")
    lines.append("")
    return lines


def archive_inventory(path: Path) -> dict:
    """Read frozen archive contents without extracting or executing the EXE."""
    archive = CArchiveReader(str(path))
    pyz = archive.open_embedded_archive("PYZ.pyz")
    manifest_name = next((name for name in archive.toc
                          if Path(name).name.casefold() == "stereolab-build.json"), None)
    manifest = json.loads(archive.extract(manifest_name)) if manifest_name else None
    components = manifest.get("components") if isinstance(manifest, dict) else None
    recomputed = (hashlib.sha256(json.dumps(components, sort_keys=True, separators=(",", ":"))
                                .encode("ascii")).hexdigest()
                  if isinstance(components, dict) else None)
    notice_files = sorted(name for name in archive.toc
                          if re.search(r"(?i)(license|licence|copying|notice)", name))
    owners = sorted({name.split("\\", 1)[0].split("/", 1)[0] for name in notice_files})
    pyz_names = set(pyz.toc)
    cv2_ffmpeg_name = next((name for name in archive.toc
                            if Path(name).name.casefold() == "opencv_videoio_ffmpeg500_64.dll"), None)
    cv2_ffmpeg = archive.extract(cv2_ffmpeg_name) if cv2_ffmpeg_name else None
    bundled_codec_strings = ([name for name in ("libopenh264", "libvpx", "libaom")
                              if name.encode("ascii") in cv2_ffmpeg]
                             if cv2_ffmpeg is not None else [])
    native_entries = [name for name in archive.toc
                      if Path(name).suffix.casefold() in {".dll", ".pyd"}]
    ffmpeg_exe_present = any(Path(name).name.casefold() == "ffmpeg.exe" for name in archive.toc)
    flat_closure = ["flatbuffers", "mpmath", "sympy"]
    extra_frozen = [name for name in sorted(pyz_names)
                    if name == "setuptools" or name.startswith("setuptools._vendor.")]
    return {
        "carchive_entries": len(archive.toc),
        "pyz_modules": len(pyz.toc),
        "native_dll_pyd_entries": len(native_entries),
        "named_license_notice_entries": len(notice_files),
        "named_license_notice_owners": owners,
        "named_notice_file_entries": notice_files,
        "third_party_notices_file_present": any(
            Path(name).name.casefold() == "third_party_notices.txt" for name in archive.toc),
        "external_ffmpeg_exe_present": ffmpeg_exe_present,
        "opencv_ffmpeg_videoio_dll": ({
            "path": cv2_ffmpeg_name,
            "bytes": len(cv2_ffmpeg),
            "sha256": hashlib.sha256(cv2_ffmpeg).hexdigest(),
            "contains_libswscale_license_string": b"libswscale license:" in cv2_ffmpeg,
            "embedded_codec_configuration_strings": bundled_codec_strings,
        } if cv2_ffmpeg is not None else None),
        "installed_closure_packages_absent_from_pyz": {
            name: not any(module == name or module.startswith(name + ".") for module in pyz_names)
            for name in flat_closure
        },
        "setuptools_and_vendored_modules_present": bool(extra_frozen),
        "setuptools_vendor_roots": sorted({
            name.split(".")[2] for name in extra_frozen
            if name.startswith("setuptools._vendor.") and len(name.split(".")) > 2
        }),
        "embedded_manifest_present": manifest is not None,
        "embedded_manifest_fingerprint": manifest.get("fingerprint") if isinstance(manifest, dict) else None,
        "embedded_manifest_recomputes": bool(manifest and manifest.get("fingerprint") == recomputed),
        "embedded_manifest_matches_current_source": bool(
            manifest and manifest.get("fingerprint") == build_manifest(ROOT)["fingerprint"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "RELEASE_DEPENDENCIES.md")
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                        help="Frozen EXE to inspect for this artifact's archive contents and identity.")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.exe = args.exe.resolve()
    if not args.exe.is_file():
        parser.error(f"Frozen EXE does not exist: {args.exe}")
    frozen = archive_inventory(args.exe)
    exe_sha256 = hashlib.sha256(args.exe.read_bytes()).hexdigest()
    exe_bytes = args.exe.stat().st_size
    runtime_roots = requirement_roots(RUNTIME_REQUIREMENTS)
    runtime = closure(runtime_roots)
    build = closure(BUILD_ROOTS)
    lines = [
        "# Release dependency and license inventory",
        "",
        f"Generated {date.today().isoformat()} using `{Path(sys.executable)}` distribution metadata.",
        f"Python: `{platform.python_version()}` ({platform.platform()}).",
        "",
        "This combines installed-package metadata from the interpreter running this script with a read-only inspection of the selected PyInstaller archive. It is not a lock file, a complete frozen SBOM, or a legal opinion. Verify full upstream license texts, notices, native component provenance, and delivery materials before redistribution.",
        "",
        "## Direct runtime requirements",
        "",
        "The version ranges come from `requirements-gpu.txt`; resolved versions below describe the interpreter running this script, not necessarily every file frozen into the selected EXE.",
        "",
        "| Requirement | Resolved version |",
        "| --- | --- |",
    ]
    for root in runtime_roots:
        key = root.casefold().replace("_", "-")
        dist = runtime.get(key)
        version = dist.version if dist is not None else "missing"
        lines.append(f"| `{root}` | `{version}` |")
    lines.append("")
    lines += markdown_table("Runtime dependency closure", runtime)
    lines += markdown_table("Build-time dependency closure", build)
    lines += [
        "## External runtime and model terms",
        "",
        f"- The standalone `ffmpeg.exe` helper is {'present in' if frozen['external_ffmpeg_exe_present'] else 'absent from'} the inspected archive. The archive's OpenCV video-I/O component and its embedded libswscale license-string check are recorded below. No separately named OpenCV/FFmpeg notice file was found by the archive filename audit. This is an evidence gap, not a legal conclusion.",
        "- The separately tested external helper reported FFmpeg 6.0-era build `2aec86695a`, configured with `--enable-gpl` and `libx264`; `ffmpeg -L` reports GPL version 3 or later. Users supply their own FFmpeg or use an already-installed local helper. Keep this external executable distinct from the bundled OpenCV video-I/O DLL when describing distribution contents.",
        "- Depth model weights are downloaded on demand and are not embedded in the EXE. README model terms list Small/Small FP16 as Apache-2.0, Base/Large as CC-BY-NC-4.0 (non-commercial), DPT Hybrid/MiDaS as Apache-2.0 for the Intel upstream model, and custom ONNX under the user's license. Recheck each selected model card and attribution before a release that promotes model use.",
        "- No project-level `LICENSE`, `COPYING`, `NOTICE`, or third-party notices file was present at generation time. The repository owner must choose and add the project's own license before public source redistribution; that choice is not inferred here.",
        "",
        "## Frozen archive inspection",
        "",
        f"Inspected `{args.exe.relative_to(ROOT).as_posix() if args.exe.is_relative_to(ROOT) else args.exe}` — {exe_bytes:,} bytes; SHA-256 `{exe_sha256}`.",
        f"CArchive entries: {frozen['carchive_entries']}; PYZ modules: {frozen['pyz_modules']}; native DLL/PYD entries: {frozen['native_dll_pyd_entries']}.",
        (f"Bundled OpenCV video-I/O DLL: `{frozen['opencv_ffmpeg_videoio_dll']['path']}`; "
         f"{frozen['opencv_ffmpeg_videoio_dll']['bytes']:,} bytes; SHA-256 "
         f"`{frozen['opencv_ffmpeg_videoio_dll']['sha256']}`; "
         f"libswscale license string present: `{frozen['opencv_ffmpeg_videoio_dll']['contains_libswscale_license_string']}`; "
         f"codec configuration strings: {', '.join(frozen['opencv_ffmpeg_videoio_dll']['embedded_codec_configuration_strings']) or 'none detected'}."
         if frozen["opencv_ffmpeg_videoio_dll"] else "Bundled OpenCV video-I/O DLL was not found in the inspected archive."),
        f"Named license/notice entries: {frozen['named_license_notice_entries']} across {len(frozen['named_license_notice_owners'])} archive owners ({', '.join(frozen['named_license_notice_owners']) or 'none'}). A standalone third-party notices file is {'present' if frozen['third_party_notices_file_present'] else 'absent'}.",
        f"Embedded build manifest fingerprint: `{frozen['embedded_manifest_fingerprint'] or 'missing'}`; recomputes correctly: `{frozen['embedded_manifest_recomputes']}`; matches current source: `{frozen['embedded_manifest_matches_current_source']}`.",
        "The archive notice-name scan found named license files only for the listed owners. FastAPI, Starlette, Uvicorn, python-multipart, ONNX Runtime, OpenCV, pydantic_core, and setuptools have no corresponding named license/notice files in the archive scan. Native archive components also include Python, OpenSSL, libffi, SQLite, DirectML, and Microsoft runtime DLLs without separately named notices found by this scan. These results require upstream notice and provenance review; they do not determine compliance.",
        "The installed runtime closure lists `flatbuffers`, `mpmath`, and `sympy`, which are absent from the inspected PYZ; the archive instead contains setuptools and vendored jaraco, more_itertools, tomli, wheel, and backports modules. Treat the installed closure and frozen archive as separate inventories.",
        "",
        "## Current artifact identity",
        "",
        f"Artifact: `{args.exe}` — {exe_bytes:,} bytes; SHA-256 `{exe_sha256}`; embedded source fingerprint `{frozen['embedded_manifest_fingerprint'] or 'missing'}`.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
