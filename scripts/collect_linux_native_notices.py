#!/usr/bin/env python3
"""Inventory collected ELF origins and installed Debian copyright evidence.

This records available evidence, not license compliance or source completeness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import subprocess


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _relative(value: object) -> str:
    raw = str(value)
    path = PurePosixPath(raw)
    win = PureWindowsPath(raw)
    if (not raw or "\\" in raw or path.is_absolute() or win.drive
            or any(part in (".", "..", "") for part in raw.split("/"))):
        raise ValueError(f"unsafe destination: {raw!r}")
    return path.as_posix()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["dpkg-query", *arguments], capture_output=True,
                          text=True, check=False, timeout=15)


def _copy(source: Path, destination: Path, output: Path) -> dict:
    target = _inside(destination, output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _inside(destination, output)
    shutil.copyfile(source.resolve(), target)
    return {"source_path": str(source), "resolved_source_path": str(source.resolve()),
            "copied_to": target.relative_to(output.resolve()).as_posix(),
            "sha256": _hash(target)}


def collect(origins: dict, package: Path, output: Path, *,
            doc_root: Path = Path("/usr/share/doc"),
            common_root: Path = Path("/usr/share/common-licenses")) -> dict:
    if origins.get("schema_version") != 1 or not isinstance(origins.get("files"), list):
        raise ValueError("origins must use schema_version 1 and a files list")
    package = package.resolve()
    output = output.resolve()
    indexed = {}
    # Validate the entire input before making output directories or copying.
    for entry in origins["files"]:
        destination = _relative(entry["destination"])
        source = Path(entry["source"])
        if not source.is_absolute() or entry["kind"] not in ("BINARY", "EXTENSION"):
            raise ValueError("origins require absolute source and BINARY/EXTENSION kind")
        _inside(package / "_internal" / destination, package)
        if destination in indexed:
            raise ValueError(f"duplicate origin destination: {destination}")
        indexed[destination] = entry
    if not package.is_dir():
        raise ValueError("package directory does not exist")
    if output == package or package in output.parents:
        raise ValueError("output must be outside package")
    output.mkdir(parents=True, exist_ok=True)
    rows, packages, errors = [], {}, []
    for binary in sorted(package.rglob("*")):
        if not binary.is_file():
            continue
        _inside(binary, package)
        with binary.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        relative = binary.relative_to(package).as_posix()
        entry = indexed.get(relative.removeprefix("_internal/")) if relative.startswith("_internal/") else None
        row = {"path": relative, "sha256": _hash(binary), "ownership_status": "unmapped",
               "owners": []}
        if entry:
            row["origin_mapping"] = {"method": "direct_destination", "destination": entry["destination"]}
        elif binary.is_symlink():
            resolved_binary = _inside(binary, package)
            resolved_relative = resolved_binary.relative_to(package).as_posix()
            resolved_entry = (indexed.get(resolved_relative.removeprefix("_internal/"))
                              if resolved_relative.startswith("_internal/") else None)
            if resolved_entry:
                entry = resolved_entry
                row["origin_mapping"] = {
                    "method": "collected_symlink_target",
                    "destination": entry["destination"],
                    "resolved_collected_path": resolved_relative,
                }
        if entry:
            source = Path(entry["source"])
            row.update(source_path=str(source), kind=entry["kind"],
                       resolved_source_path=str(source.resolve()))
            row["source_classification"] = ("wheel_or_site_packages" if "site-packages" in source.parts
                                            else "stdlib" if "lib-dynload" in source.parts
                                            else "other_unresolved")
            if source.is_file():
                row["source_sha256"] = _hash(source)
                row["source_matches_collected"] = row["sha256"] == row["source_sha256"]
            else:
                row["source_matches_collected"] = None
                row["source_missing"] = True
            candidates = list(dict.fromkeys([str(source), str(source.resolve())]))
            attempts = []
            owners = set()
            for candidate in candidates:
                try:
                    result = _query(["-S", candidate])
                    attempts.append({"path": candidate, "exit_code": result.returncode,
                                     "stdout": result.stdout, "stderr": result.stderr})
                    if result.returncode == 0:
                        for line in result.stdout.splitlines():
                            if ": " in line:
                                owner, owned_path = line.rsplit(": ", 1)
                                if owned_path == candidate:
                                    owners.update(owner.split(", "))
                except (OSError, subprocess.TimeoutExpired) as exc:
                    attempts.append({"path": candidate, "error": str(exc)})
                if owners:
                    break
            row["ownership_queries"] = attempts
            row["owners"] = sorted(owners)
            if owners:
                row["ownership_status"] = "dpkg_owned"
                row["source_classification"] = "dpkg_owned"
            else:
                row["ownership_status"] = "no_dpkg_owner"
            for owner in sorted(owners):
                if owner in packages:
                    continue
                record = {"name": owner, "copyright": None}
                packages[owner] = record
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9+.:~-]*", owner):
                    record["error"] = "unsafe dpkg package name"
                    continue
                try:
                    result = _query(["-W", "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\\n", owner])
                    record["details_exit_code"] = result.returncode
                    fields = result.stdout.rstrip("\n").split("\t")
                    if result.returncode == 0 and len(fields) == 4:
                        record.update(zip(("binary_package", "version", "source_package", "source_version"), fields))
                    else:
                        record["details_error"] = result.stderr or result.stdout
                except (OSError, subprocess.TimeoutExpired) as exc:
                    record["details_error"] = str(exc)
                copyright_path = doc_root / owner.split(":", 1)[0] / "copyright"
                if copyright_path.is_file():
                    try:
                        _inside(copyright_path, doc_root)
                        record["copyright"] = _copy(copyright_path, output / "packages" / owner.replace(":", "_") / "copyright", output)
                    except (OSError, ValueError) as exc:
                        record["copyright_error"] = str(exc)
                else:
                    record["copyright_error"] = "installed copyright file not found"
        rows.append(row)
    common = []
    if common_root.is_dir():
        for source in sorted(common_root.iterdir()):
            if source.is_file():
                try:
                    _inside(source, common_root)
                    common.append(_copy(source, output / "common-licenses" / source.name, output))
                except (OSError, ValueError) as exc:
                    errors.append({"path": str(source), "error": str(exc)})
    report = {"schema_version": 1, "scope_note": "Installed native provenance and available notice evidence only; no compliance or corresponding-source closure claim.",
              "elf_count": len(rows), "files": rows, "packages": list(packages.values()),
              "common_licenses_scope": "All available installed common-license files; shared supporting texts, not inferred component license assignments.",
              "common_licenses": common, "errors": errors}
    target = _inside(output / "NATIVE_NOTICES.json", output)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origins", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = collect(json.loads(args.origins.read_text(encoding="utf-8")), args.package, args.output)
    print(json.dumps({"elf_count": report["elf_count"], "package_count": len(report["packages"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
