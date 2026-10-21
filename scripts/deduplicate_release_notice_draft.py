"""Create a review-only content-deduplicated view of a release notice draft.

The original draft is preserved. This script groups exact byte-identical texts
by SHA-256 and writes an alias row for every original notice manifest entry.
It does not decide component scope or legal sufficiency, and the output is not
an approved distribution bundle.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "release-notices-draft-2026-10-21"
DESTINATION = ROOT / "output" / "release-notices-deduplicated-review-2026-10-21.zip"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    manifest_path = SOURCE / "source-manifest.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing source manifest: {manifest_path}")
    if DESTINATION.exists():
        raise SystemExit(f"Refusing to overwrite existing review output: {DESTINATION}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        required = {"component", "version", "source_kind", "source", "copied_to",
                    "bytes", "sha256", "already_in_candidate"}
        if not rows or not required.issubset(reader.fieldnames or []):
            raise SystemExit("Source manifest is empty or has an unexpected schema.")

    payloads: dict[str, bytes] = {}
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_payloads: list[tuple[dict[str, str], str]] = []
    for row in rows:
        draft_path = SOURCE / Path(row["copied_to"])
        data = draft_path.read_bytes()
        actual_hash = digest(data)
        if actual_hash != row["sha256"].lower():
            raise SystemExit(f"Source copy hash mismatch: {draft_path}")
        if len(data) != int(row["bytes"]):
            raise SystemExit(f"Source copy length mismatch: {draft_path}")
        prior = payloads.setdefault(actual_hash, data)
        if prior != data:
            raise SystemExit(f"Distinct payload bytes share SHA-256: {actual_hash}")
        grouped[actual_hash].append(row)
        row_payloads.append((row, actual_hash))

    unique_rows: list[dict[str, str]] = []
    payload_members: dict[str, bytes] = {}
    for content_hash in sorted(payloads):
        payload_path = f"payloads/{content_hash}.txt"
        data = payloads[content_hash]
        payload_members[payload_path] = data
        aliases = grouped[content_hash]
        unique_rows.append({
            "sha256": content_hash,
            "payload": payload_path,
            "bytes": str(len(data)),
            "alias_count": str(len(aliases)),
            "candidate_embedded_aliases": str(sum(r["already_in_candidate"].lower() == "true" for r in aliases)),
            "component_version_labels": " | ".join(
                f"{r['component']} {r['version']}" for r in aliases),
        })

    alias_rows: list[dict[str, str]] = []
    for row, content_hash in row_payloads:
        alias_rows.append({
            "component": row["component"],
            "version": row["version"],
            "source_kind": row["source_kind"],
            "source": row["source"],
            "original_draft_path": row["copied_to"],
            "bytes": row["bytes"],
            "sha256": content_hash,
            "already_in_candidate": row["already_in_candidate"],
            "deduplicated_payload": f"payloads/{content_hash}.txt",
        })

    def csv_bytes(fieldnames: list[str], data_rows: list[dict[str, str]]) -> bytes:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(data_rows)
        return stream.getvalue().encode("utf-8")

    source_manifest_hash = digest(manifest_path.read_bytes())
    archive_sources = {
        row["source"].split("!", 1)[0]
        for row in rows if row["source_kind"] == "candidate CArchive entry"
    }
    candidate_lines = []
    for exe in sorted(archive_sources):
        candidate_path = Path(exe)
        candidate_lines.append(
            f"- `{candidate_path}` — SHA-256 `{digest(candidate_path.read_bytes())}`"
            if candidate_path.is_file() else f"- `{candidate_path}` — source no longer present")
    readme = (
        "# Mechanically deduplicated release-notice review copy\n\n"
        "This is a review aid generated from the original notice draft. It "
        "contains one exact-byte payload per SHA-256 and preserves one alias "
        "row for every source-manifest entry. It does not decide which notices "
        "apply, add missing component texts, establish redistribution rights, "
        "or constitute an approved distribution bundle.\n\n"
        f"- Source draft: `{SOURCE.relative_to(ROOT).as_posix()}`\n"
        f"- Source manifest SHA-256: `{source_manifest_hash}`\n"
        f"- Source rows: {len(rows)}\n"
        f"- Unique payloads: {len(payloads)}\n"
        f"- Rows already embedded in the inspected candidate: "
        f"{sum(row['already_in_candidate'].lower() == 'true' for row in rows)}\n"
        f"- Supplementary source rows: "
        f"{sum(row['already_in_candidate'].lower() != 'true' for row in rows)}\n\n"
        "Every payload was checked against the original draft copy's recorded "
        "byte count and SHA-256, then checked again after writing. `aliases.csv` "
        "maps all original component labels, source paths, and draft paths to "
        "the retained payload. No source file or prior draft was changed.\n\n"
        "Candidate archive source identity:\n\n"
        + ("\n".join(candidate_lines) if candidate_lines else "- No CArchive source rows were recorded.")
        + "\n"
    )

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)
        archive.writestr("aliases.csv", csv_bytes(list(alias_rows[0]), alias_rows))
        archive.writestr("payload-manifest.csv", csv_bytes(list(unique_rows[0]), unique_rows))
        for name, data in sorted(payload_members.items()):
            archive.writestr(name, data)

    # Exclusive creation prevents accidentally replacing an earlier review artifact.
    with DESTINATION.open("xb") as stream:
        stream.write(archive_buffer.getvalue())

    with zipfile.ZipFile(DESTINATION) as archive:
        names = set(archive.namelist())
        if len(names) != len(archive.namelist()):
            raise SystemExit("Duplicate file names in deduplicated review archive.")
        for name, data in payload_members.items():
            written = archive.read(name)
            if written != data or digest(written) != name.removeprefix("payloads/").removesuffix(".txt"):
                raise SystemExit(f"Archive payload verification failed: {name}")
        if len([name for name in names if name.startswith("payloads/")]) != len(payloads):
            raise SystemExit("Unexpected payload count in deduplicated review archive.")

    print(f"source_rows={len(rows)} unique_payloads={len(payloads)}")
    print(f"embedded_aliases={sum(r['already_in_candidate'].lower() == 'true' for r in rows)}")
    print(f"supplementary_aliases={sum(r['already_in_candidate'].lower() != 'true' for r in rows)}")
    print(f"output={DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
