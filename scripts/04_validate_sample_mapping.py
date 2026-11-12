from __future__ import annotations

import argparse
import csv
import gzip
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_CSV = ROOT / "data" / "metadata" / "review" / "sample_review_queue.csv"
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "metadata" / "mapping"
GSM_PATTERN = re.compile(r"GSM\d+", re.IGNORECASE)
SAMPLE_SUFFIX = re.compile(r"\s*[\[(]\s*RNA[-_ ]?seq\s*[\])]\s*$", re.IGNORECASE)
NON_SAMPLE_COLUMNS = {
    "gene", "geneid", "gene_id", "gene name", "gene symbol", "symbol",
    "ensembl", "ensembl id", "ensembl_gene_id", "gene stable id",
    "probe", "chromosome", "start", "end", "probe strand", "feature",
    "id", "description", "length", "gene length", "gene biotype",
    "feature strand", "type", "feature orientation", "distance",
}
MAPPING_FIELDS = [
    "accession", "expression_file", "source_type", "column_or_member", "position",
    "gsm", "sample_title", "candidate_group", "mapping_status", "note",
]
SUMMARY_FIELDS = [
    "accession", "expression_file", "source_type", "items_inspected",
    "exact_gsm", "exact_title", "alias_candidate", "unresolved", "ambiguous",
    "not_sample_level", "known_gsm_records", "note",
]


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", SAMPLE_SUFFIX.sub("", value).lower())



def alias_key(accession: str, title: str) -> str:
    title = SAMPLE_SUFFIX.sub("", title).strip()

    if accession == "GSE193258":
        match = re.fullmatch(
            r"([A-Za-z0-9]+)_(long|short)_washout_(?:\d+[dh]_)?(\d+)", title, re.IGNORECASE
        )
        if match:
            cell, kind, repeat = match.groups()
            return normalized(f"{cell}_{kind}_wash_{repeat}")

    if accession == "GSE254485":
        match = re.fullmatch(
            r"(.+?)_RNA_Seq_(?:(0d|28d|35d|56d)_osi_)?rep(\d+)",
            title,
            re.IGNORECASE,
        )
        if match:
            cell, day, repeat = match.groups()
            return normalized(f"{cell.replace('-OR', 'OR')}_{(day + '_') if day else ''}rep{repeat}")

    if accession == "GSE202859":
        match = re.fullmatch(r"(.+?) cells(?: (.*?))? rep (\d+)", title, re.IGNORECASE)
        if not match:
            return ""
        cell, condition, repeat = match.groups()
        cell = cell.replace("-", "").upper()
        if condition is None:
            return normalized(f"{cell}.{repeat}")
        condition = condition.lower()
        match_condition = re.fullmatch(
            r"(untreated|osimertinib)(?: (scramble shrna|smarca4 #1 shrna|smarca4 #2 shrna))?",
            condition,
        )
        if not match_condition:
            return ""
        drug, knockdown = match_condition.groups()
        drug_tag = "UNT" if drug == "untreated" else "OSI"
        knockdown_tag = {
            None: "",
            "scramble shrna": "SCR",
            "smarca4 #1 shrna": "B1A",
            "smarca4 #2 shrna": "B1B",
        }[knockdown]
        return normalized(f"{cell}.{knockdown_tag}.{drug_tag}.{repeat}")

    return ""

def read_review(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cohort review output: {path}")
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "gsm", "title", "candidate_group"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Review file is missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for record in reader:
            if record["accession"] and record["gsm"]:
                result[record["accession"]].append(record)
    return result


def write_csv(path: Path, records: list[dict[str, str | int]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def expression_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    files = []
    for path in sorted(folder.iterdir()):
        lower = path.name.lower()
        if not path.is_file() or "family.soft" in lower or "metadata_with_lineage" in lower:
            continue
        if lower.endswith((".gz", ".tar", ".tsv", ".txt", ".csv")):
            files.append(path)
    return files


def matrix_header(path: Path) -> list[str]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
        line = handle.readline()
    if not line:
        raise ValueError("empty expression matrix")
    delimiter = "\t" if line.count("\t") >= line.count(",") else ","
    return next(csv.reader([line], delimiter=delimiter))


def row(accession: str, filename: str, source_type: str, item: str, pos: int,
        sample: dict[str, str] | None, status: str, note: str = "") -> dict[str, str | int]:
    return {
        "accession": accession,
        "expression_file": filename,
        "source_type": source_type,
        "column_or_member": item,
        "position": pos,
        "gsm": sample["gsm"] if sample else "",
        "sample_title": sample["title"] if sample else "",
        "candidate_group": sample["candidate_group"] if sample else "",
        "mapping_status": status,
        "note": note,
    }


def inspect_matrix(path: Path, samples: list[dict[str, str]], accession: str):
    columns = matrix_header(path)
    title_lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    alias_lookup: dict[str, list[dict[str, str]]] = defaultdict(list)
    for sample in samples:
        title_lookup[normalized(sample["title"])].append(sample)
        alias = alias_key(accession, sample["title"])
        if alias:
            alias_lookup[alias].append(sample)

    if accession == "GSE150949" and "pc9_count_matrix" in path.name.lower():
        record = row(accession, path.name, "cell_matrix", f"{len(columns) - 1} data columns", 0,
                     None, "cell_barcodes_need_metadata_join",
                     "Per-cell columns are not GEO GSM records; review cell metadata first")
        return [record], len(columns) - 1, "cell_matrix; no GSM-level matching attempted"

    records = []
    for index, column in enumerate(columns, start=1):
        label = column.strip()
        if not label or label.lower() in NON_SAMPLE_COLUMNS:
            continue
        key = normalized(label)
        matches = title_lookup.get(key, [])
        aliases = alias_lookup.get(key, []) if not matches else []
        if len(matches) == 1:
            records.append(row(accession, path.name, "matrix_column", label, index,
                               matches[0], "exact_title", "Title match only; verify sample correspondence"))
        elif len(matches) > 1 or len(aliases) > 1:
            records.append(row(accession, path.name, "matrix_column", label, index,
                               None, "ambiguous", "Multiple GEO records match the same column"))
        elif len(aliases) == 1:
            records.append(row(accession, path.name, "matrix_column", label, index,
                               aliases[0], "alias_candidate",
                               "Dataset-specific naming convention; manual confirmation required"))
        else:
            records.append(row(accession, path.name, "matrix_column", label, index,
                               None, "unresolved", "No exact GSM/title match; do not assign a label"))

    counts = Counter(x["column_or_member"] for x in records)
    gsm_counts = Counter(rec["gsm"] for rec in records if rec["gsm"])
    for rec in records:
        if counts[rec["column_or_member"]] > 1 or (rec["gsm"] and gsm_counts[rec["gsm"]] > 1):
            rec["mapping_status"] = "ambiguous"
            rec["gsm"] = ""
            rec["sample_title"] = ""
            rec["candidate_group"] = ""
            rec["note"] = "Duplicate matrix column name"
    return records, len(columns), "Bulk matrix columns checked by exact sample title"


def inspect_tar(path: Path, samples: list[dict[str, str]], accession: str):
    by_gsm = {x["gsm"].upper(): x for x in samples}
    records = []
    with tarfile.open(path, "r:") as archive:
        for index, member in enumerate(archive, start=1):
            if not member.isfile():
                continue
            gsm = GSM_PATTERN.search(Path(member.name).name)
            if not gsm:
                records.append(row(accession, path.name, "tar_member", member.name, index,
                                   None, "unresolved", "No GSM identifier in member filename"))
                continue
            sample = by_gsm.get(gsm.group().upper())
            status = "exact_gsm" if sample else "unresolved"
            note = "GSM in filename only; matrix contents not checked" if sample else "GSM not found in review CSV"
            records.append(row(accession, path.name, "tar_member", member.name, index,
                               sample, status, note))
    return records, len(records), "TAR member filenames checked; member contents not extracted"


def summarize(accession: str, name: str, source: str, records: list[dict[str, str | int]],
              inspected: int, known: int, note: str) -> dict[str, str | int]:
    counts = Counter(str(r["mapping_status"]) for r in records)
    return {
        "accession": accession,
        "expression_file": name,
        "source_type": source,
        "items_inspected": inspected,
        "exact_gsm": counts["exact_gsm"],
        "exact_title": counts["exact_title"],
        "alias_candidate": counts["alias_candidate"],
        "unresolved": counts["unresolved"],
        "ambiguous": counts["ambiguous"],
        "not_sample_level": counts["cell_barcodes_need_metadata_join"],
        "known_gsm_records": known,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="GEO accession, e.g. GSE193258")
    args = parser.parse_args()
    samples_by_accession = read_review(REVIEW_CSV)
    accessions = sorted(samples_by_accession)
    if args.only:
        target = args.only.strip().upper()
        if target not in samples_by_accession:
            parser.error(f"GEO accession is absent from the review file: {target}")
        accessions = [target]

    mappings: list[dict[str, str | int]] = []
    summaries: list[dict[str, str | int]] = []
    for accession in accessions:
        files = expression_files(RAW_DIR / accession)
        if not files:
            print(f"[{accession}] No expression files found in data/raw/{accession}")
            continue
        for file in files:
            try:
                if file.name.lower().endswith(".tar"):
                    records, inspected, note = inspect_tar(file, samples_by_accession[accession], accession)
                    source = "tar_member"
                else:
                    records, inspected, note = inspect_matrix(file, samples_by_accession[accession], accession)
                    source = "matrix"
            except (OSError, UnicodeError, ValueError, tarfile.TarError, csv.Error) as exc:
                records = []
                inspected = 0
                note = f"ERROR: {type(exc).__name__}: {exc}"
                source = "unknown"
            mappings.extend(records)
            summary = summarize(accession, file.name, source, records, inspected,
                                len(samples_by_accession[accession]), note)
            summaries.append(summary)
            unique_gsm = len({str(r["gsm"]) for r in records if r["gsm"]})
            print(f"[{accession}] {file.name}: {summary['exact_gsm']} GSM-linked files, "
                  f"{summary['exact_title']} title matches, {summary['alias_candidate']} alias candidates, "
                  f"{unique_gsm} unique GSM, {summary['unresolved']} unresolved, "
                  f"{summary['ambiguous']} ambiguous, "
                  f"{summary['not_sample_level']} cell-level matrices")
            if note.startswith("ERROR"):
                print(f"  {note}")

    if not summaries:
        raise SystemExit("No files inspected. Check the data/raw folder.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "sample_matrix_mapping.csv", mappings, MAPPING_FIELDS)
    write_csv(OUTPUT_DIR / "mapping_summary.csv", summaries, SUMMARY_FIELDS)
    print(f"Reports saved in: {OUTPUT_DIR}")
    print("Alias candidates are naming-based suggestions, not validated sample labels.")


if __name__ == "__main__":
    main()
