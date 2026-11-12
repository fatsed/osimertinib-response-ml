from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ACCESSION_RE = re.compile(r"^GSE\d+$")
GSM_RE = re.compile(r"^GSM\d+$")
BLOCK_SIZE = 1024 * 1024
MAX_LINE_BYTES = 4 * 1024 * 1024  # bounded preview for extremely wide count matrices

# Only explicitly provided sample-characteristic keys are used here.
# Values are copied without translating, categorizing, or inferring them.
KEYS = {
    "explicit_cell_line": {"cell line", "cell_line", "cellline"},
    "explicit_treatment": {"treatment", "drug treatment", "drug", "treatment condition", "compound"},
    "explicit_exposure_time": {"treatment duration", "exposure time", "treatment time", "timepoint", "time point", "duration"},
    "explicit_state": {"state", "cell state", "treatment state", "condition", "phenotype"},
}

FILE_COLS = ["accession", "filename", "file_type", "size_bytes", "sha256", "manifest_sha256",
             "integrity", "preview", "notes"]
SERIES_COLS = ["accession", "soft_file", "soft_status", "geo_sample_records",
               "unique_gsm_records", "characteristic_keys", "total_local_files", "review_status", "notes"]
SAMPLE_COLS = ["accession", "gsm", "title", "source_name_ch1", "organism_ch1",
               "characteristics_json", "treatment_protocol_ch1", "description", "explicit_cell_line",
               "explicit_treatment", "explicit_exposure_time", "explicit_state", "review_status"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(BLOCK_SIZE), b""):
            h.update(b)
    return h.hexdigest()


def read_candidates(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate list is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "geo_accession" not in rows[0]:
        raise ValueError(f"Expected column geo_accession in {path}")
    acc = sorted({row["geo_accession"].strip().upper() for row in rows})
    if not all(ACCESSION_RE.fullmatch(x) for x in acc):
        raise ValueError(f"Invalid accessions in {path}")
    return acc


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict):
        raise ValueError("Download report does not contain a 'files' dictionary")
    return files


def normalize_key(key: str) -> str:
    return re.sub(r"\s+", " ", key.strip().lower().replace("-", " "))


def get_explicit(characteristics: list[str], accepted: set[str]) -> str:
    """Extract only unambiguous values from explicitly named GEO characteristics."""
    accepted_normalized = {normalize_key(k) for k in accepted}
    matches: set[str] = set()
    for item in characteristics:
        if ":" not in item:
            continue
        label, val = item.split(":", 1)
        val = val.strip()
        if normalize_key(label) in accepted_normalized and val:
            matches.add(val)
    return next(iter(matches)) if len(matches) == 1 else ""


def parse_soft(path: Path, accession: str) -> tuple[list[dict[str, str]], Counter[str]]:
    """Stream GEO SOFT; ignore platform/sample expression tables, preserve provided sample fields.

    SOFT sample tags are repeated and order-dependent. Values are never interpreted
    as eligibility, ground-truth cell-cycle labels, or independent replicates.
    """
    records: list[dict[str, str]] = []
    key_counts: Counter[str] = Counter()
    current: dict[str, list[str]] | None = None
    current_id = ""
    in_sample_table = False

    def commit() -> None:
        if current is None or not GSM_RE.fullmatch(current_id):
            return
        fields = {k: v[0] if v else "" for k, v in current.items()}
        ch: list[str] = []
        for key, values in current.items():
            if key.startswith("characteristics_"):
                ch.extend(values)
        for s in ch:
            if ":" in s:
                key_counts[normalize_key(s.split(":", 1)[0])] += 1
        records.append({
            "accession": accession,
            "gsm": current_id,
            "title": fields.get("title", ""),
            "source_name_ch1": fields.get("source_name_ch1", ""),
            "organism_ch1": fields.get("organism_ch1", ""),
            "characteristics_json": json.dumps(ch, ensure_ascii=False),
            "treatment_protocol_ch1": " | ".join(current.get("treatment_protocol_ch1", [])),
            "description": " | ".join(current.get("description", [])),
            **{k: get_explicit(ch, keys) for k, keys in KEYS.items()},
            "review_status": "pending_manual_review",
        })

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for original in f:
            line = original.rstrip("\r\n")
            line_stripped = line.strip()
            if line_stripped.lower() in ("!sample_table_begin", "!platform_table_begin"):
                in_sample_table = True
                continue
            if line_stripped.lower() in ("!sample_table_end", "!platform_table_end"):
                in_sample_table = False
                continue
            if line.startswith("^"):
                # Sample and platform record boundaries terminate the prior sample.
                if current is not None:
                    commit()
                    current = None
                    current_id = ""
                in_sample_table = False
                if line.upper().startswith("^SAMPLE") and "=" in line:
                    sample_id = line.split("=", 1)[1].strip().upper()
                    if GSM_RE.fullmatch(sample_id):
                        current_id = sample_id
                        current = {}
                continue
            if current is None or in_sample_table:
                continue
            if line.startswith("!Sample_") and "=" in line:
                left, val = line.split("=", 1)
                key = left.removeprefix("!Sample_").strip().lower()
                current.setdefault(key, []).append(val.strip())
        if current is not None:
            commit()
    return records, key_counts


def gzip_preview(path: Path, deep_check: bool) -> tuple[str, dict[str, Any]]:
    """Peek at a bounded number of lines. Optional deep check reads the entire gzip/CRC."""
    with path.open("rb") as raw:
        if raw.read(2) != b"\x1f\x8b":
            raise ValueError("invalid gzip magic bytes")
    result: dict[str, Any] = {}
    with gzip.open(path, "rb") as f:
        lines = []
        truncated = False
        for _ in range(4):
            line = f.readline(MAX_LINE_BYTES + 1)
            if not line:
                break
            if len(line) > MAX_LINE_BYTES:
                truncated = True
                lines.append(line[:180].decode("utf-8", errors="replace"))
                break
            if line.strip():
                lines.append(line.decode("utf-8", errors="replace").rstrip("\r\n"))
        if lines:
            sep = "\t" if lines[0].count("\t") >= lines[0].count(",") else ","
            result["delimiter_preview"] = "tab" if sep == "\t" else "comma"
            result["first_line_fields"] = None if truncated else lines[0].count(sep) + 1
            result["first_fields"] = [x[:85] for x in lines[0].split(sep, 8)[:8]]
            result["first_line_truncated"] = truncated
            result["nonempty_lines_previewed"] = len(lines)
        if deep_check:
            decompressed_bytes = sum(len(chunk) for chunk in iter(lambda: f.read(BLOCK_SIZE), b""))
            result["deep_check"] = "gzip_stream_completed_crc_checked"
            result["bytes_read_after_preview"] = decompressed_bytes
    return ("gzip_full_ok" if deep_check else "gzip_header_readable"), result


def tar_preview(path: Path) -> tuple[str, dict[str, Any]]:
    with tarfile.open(path, "r:") as archive:
        members = archive.getmembers()  # reads headers; never extracts files
    regular = [m for m in members if m.isfile()]
    suspicious = [m.name for m in members if m.issym() or m.islnk() or m.name.startswith("/") or ".." in Path(m.name).parts]
    return "tar_headers_ok", {
        "member_count": len(members),
        "regular_file_count": len(regular),
        "sum_declared_member_bytes": sum(m.size for m in regular),
        "example_member_names": [m.name[:180] for m in regular[:8]],
        "suspicious_paths_or_links_count": len(suspicious),
        "note": "Archive listed, NOT extracted; payload integrity NOT tested",
    }


def inspect_file(path: Path, accession: str, manifest: dict[str, Any], deep_check: bool) -> dict[str, str]:
    name = path.name
    kind = "geo_soft" if name.endswith("_family.soft.gz") else (
        "tar_archive" if name.lower().endswith(".tar") else
        "gzip_file" if name.lower().endswith(".gz") else "other")
    key = f"{accession}/{name}"
    entry = manifest.get(key, {})
    if not isinstance(entry, dict):
        entry = {}
    expected_hash = str(entry.get("sha256") or "")
    actual_hash = sha256(path)
    notes = []
    if expected_hash and actual_hash != expected_hash:
        integrity = "SHA256_MISMATCH"
        notes.append("Manifest SHA-256 does not match local file")
    else:
        integrity = "hash_match" if expected_hash else "hash_recorded_no_reference"
    preview: dict[str, Any] = {}
    try:
        if kind == "tar_archive":
            tar_integrity, preview = tar_preview(path)
        elif kind in ("geo_soft", "gzip_file"):
            tar_integrity, preview = gzip_preview(path, deep_check)
        else:
            tar_integrity = "not_analyzed"
        if integrity == "SHA256_MISMATCH":
            notes.append(f"Format check: {tar_integrity}")
        else:
            integrity = f"{integrity};{tar_integrity}"
    except (OSError, EOFError, ValueError, tarfile.TarError) as exc:
        integrity = "FORMAT_ERROR" + (";SHA256_MISMATCH" if expected_hash and actual_hash != expected_hash else "")
        notes.append(f"Cannot inspect file: {type(exc).__name__}: {str(exc)[:180]}")
    return {"accession": accession, "filename": name, "file_type": kind,
            "size_bytes": str(path.stat().st_size), "sha256": actual_hash,
            "manifest_sha256": expected_hash, "integrity": integrity,
            "preview": json.dumps(preview, ensure_ascii=False, separators=(",", ":")),
            "notes": " | ".join(notes)}


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_audit(raw_root: Path, report_dir: Path, candidates: list[str],
              manifest: dict[str, Any], deep_check: bool) -> dict[str, Any]:
    inventories: list[dict[str, str]] = []
    samples: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    warnings: list[str] = []
    for accession in candidates:
        folder = raw_root / accession
        paths = sorted(p for p in folder.iterdir() if p.is_file() and not p.name.endswith(".part")) if folder.is_dir() else []
        incomplete = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name.endswith(".part")) if folder.is_dir() else []
        if incomplete:
            warnings.append(f"{accession}: incomplete download(s): {', '.join(incomplete[:8])}")
        expected = {key.split("/", 1)[1] for key in manifest if key.startswith(accession + "/")}
        for expected_filename in sorted(expected):
            if not (folder / expected_filename).is_file():
                warnings.append(f"{accession}: file recorded in download manifest is missing: {expected_filename}")
        soft = folder / f"{accession}_family.soft.gz"
        soft_status = "not_found"
        extracted: list[dict[str, str]] = []
        characteristic_counts: Counter[str] = Counter()
        if not soft.is_file():
            warnings.append(f"{accession}: missing SOFT family metadata")
        print(f"[{accession}] {len(paths)} local files", flush=True)
        for path in paths:
            rec = inspect_file(path, accession, manifest, deep_check)
            inventories.append(rec)
            print(f"  - {path.name}: {rec['integrity']}", flush=True)
            if "MISMATCH" in rec["integrity"] or "ERROR" in rec["integrity"]:
                warnings.append(f"{accession}/{path.name}: {rec['integrity']} {rec['notes']}")
            if path == soft and "SHA256_MISMATCH" in rec["integrity"]:
                soft_status = "hash_mismatch_not_parsed"
            elif path == soft and "FORMAT_ERROR" not in rec["integrity"]:
                try:
                    extracted, characteristic_counts = parse_soft(path, accession)
                    soft_status = "parsed" if extracted else "parsed_zero_gsm"
                except (OSError, EOFError, ValueError) as exc:
                    soft_status = "parse_error"
                    warnings.append(f"{accession}: SOFT parsing failed: {exc}")
        if soft_status == "parsed_zero_gsm":
            warnings.append(f"{accession}: SOFT parsed but no GSM sample records found")
        if soft_status == "not_found" and soft.is_file():
            soft_status = "unreadable"
        observed_ids = [s["gsm"] for s in extracted]
        if len(observed_ids) != len(set(observed_ids)):
            warnings.append(f"{accession}: duplicate GSM identifiers in SOFT")
        samples.extend(extracted)
        summaries.append({"accession": accession, "soft_file": soft.name,
                          "soft_status": soft_status, "geo_sample_records": str(len(extracted)),
                          "unique_gsm_records": str(len(set(observed_ids))),
                          "characteristic_keys": " | ".join(sorted(characteristic_counts)),
                          "total_local_files": str(len(paths)),
                          "review_status": "pending_manual_review", "notes": "GSM records are not replicate counts"})

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "geo_file_inventory.csv", inventories, FILE_COLS)
    write_csv(report_dir / "geo_series_summary.csv", summaries, SERIES_COLS)
    write_csv(report_dir / "geo_samples_unreviewed.csv", samples, SAMPLE_COLS)
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "checked_accessions": candidates, "local_files_inspected": len(inventories),
              "soft_gsm_records": len(samples), "deep_gzip_check": deep_check,
              "all_labels_pending_manual_review": True,
              "warnings": warnings,
              "interpretation_limits": [
                  "GEO GSM record counts are not cell numbers or independent biological replicate counts.",
                  "Expression file column previews do not establish sample alignment, gene identifiers, or scales.",
                  "Bulk and single-cell datasets cannot be pooled without a defined, validated mapping.",
                  "This script does not derive response states from filenames or sample titles.",
                  "Tar archives were listed, not extracted; their payloads remain unaudited.",
              ],
              "outputs": ["geo_file_inventory.csv", "geo_series_summary.csv", "geo_samples_unreviewed.csv", "audit_summary.json"]}
    tmp = report_dir / "audit_summary.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(report_dir / "audit_summary.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "metadata" / "audit")
    parser.add_argument("--candidate-file", type=Path, default=ROOT / "configs" / "dataset_candidates.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "metadata" / "geo_downloads.json")
    parser.add_argument("--only", action="append", metavar="GSEXXXXXX", help="Limit to accession (repeatable)")
    parser.add_argument("--deep-check", action="store_true", help="Read full .gz stream and verify gzip CRC (slower)")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when warnings are reported")
    args = parser.parse_args(argv)
    try:
        candidates = read_candidates(args.candidate_file)
        if args.only:
            chosen = {x.upper() for x in args.only}
            unknown = chosen - set(candidates)
            if unknown:
                parser.error(f"Unknown accession(s): {', '.join(sorted(unknown))}")
            candidates = [c for c in candidates if c in chosen]
        manifest = read_manifest(args.manifest)
        result = run_audit(args.raw_root, args.output_dir, candidates, manifest, args.deep_check)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        return 2
    print("\nReports in:", args.output_dir)
    for file in result["outputs"]:
        print("  ", file)
    print(f"Audited {result['local_files_inspected']} files, listed {result['soft_gsm_records']} GEO GSM records.")
    print(f"Warnings: {len(result['warnings'])}. All samples require manual scientific review.")
    if result["warnings"]:
        for warning in result["warnings"][:12]:
            print("WARNING:", warning)
        if len(result["warnings"]) > 12:
            print(f"... plus {len(result['warnings']) - 12} additional warnings in audit_summary.json")
    return 1 if args.strict and result["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
