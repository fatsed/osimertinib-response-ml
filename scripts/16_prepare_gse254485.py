from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE254485"
RAW_FILE = "GSE254485_RNAseq_featureCounts_final.txt.gz"
DAYS = (0, 28, 35, 56)
EXPECTED = {f"GSM{8043046 + 2 * i + rep - 1}": (day, rep)
            for i, day in enumerate(DAYS) for rep in (1, 2)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in {path}")
        return list(reader)


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def validate_mapping(root: Path) -> list[dict]:
    reviewed = read_csv(
        root / "data/metadata/review/sample_review_queue.csv",
        {"accession", "gsm", "title", "cell_line", "time_point",
         "candidate_group", "analysis_role", "review_status", "treatment"},
    )
    mappings = read_csv(
        root / "data/metadata/mapping/sample_matrix_mapping.csv",
        {"accession", "expression_file", "source_type", "column_or_member",
         "gsm", "mapping_status"},
    )
    selected = [row for row in reviewed if row["accession"] == ACCESSION
                and row["analysis_role"] == "timecourse_candidate"]
    if len(selected) != 8 or {r["gsm"] for r in selected} != set(EXPECTED):
        raise ValueError("Expected eight specific GSE254485 PC9 time-course GSM records")

    linked = [row for row in mappings if row["accession"] == ACCESSION
              and row["expression_file"] == RAW_FILE
              and row["gsm"] in EXPECTED]
    if len(linked) != 8 or {r["gsm"] for r in linked} != set(EXPECTED):
        raise ValueError("Eight unique mapping entries are required")
    by_gsm = {row["gsm"]: row for row in linked}
    records = []
    for gsm, (day, rep) in EXPECTED.items():
        sample = next(r for r in selected if r["gsm"] == gsm)
        match = by_gsm[gsm]
        expected_title = f"PC9_RNA_Seq_{day}d_osi_rep{rep}"
        expected_column = f"PC9_{day}d_rep{rep}"
        expected_drug = "No treatment" if day == 0 else "Osimertinib"
        if (sample["title"] != expected_title
                or sample["cell_line"].upper() != "PC9"
                or sample["time_point"].lower() != f"{day}d"
                or sample["candidate_group"] != f"Day_{day}"
                or sample["treatment"].lower() != expected_drug.lower()):
            raise ValueError(f"GEO metadata conflict for {gsm}; review the source")
        if (match["source_type"] != "matrix_column"
                or match["mapping_status"] not in {"alias_candidate", "exact_title"}
                or normalize(match["column_or_member"]) != normalize(expected_column)):
            raise ValueError(f"Unapproved or inconsistent expression-column alias for {gsm}")
        records.append({
            "gsm": gsm,
            "cell_line": "PC9",
            "day": day,
            "replicate_label": rep,
            "treatment": sample["treatment"],
            "geo_title": sample["title"],
            "matrix_column": match["column_or_member"],
            "mapping_status": match["mapping_status"],
            "review_status": sample["review_status"],
            "scientifically_verified": False,
        })
    if len({r["matrix_column"] for r in records}) != 8:
        raise ValueError("A matrix column was assigned to multiple GEO records")
    return records


def matrix_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            header = next(csv.reader([line], delimiter="\t"))
            if len(header) < 9 or len(header) != len(set(header)):
                raise ValueError("Unexpected/duplicate featureCounts header")
            return header
    raise ValueError("The expression file has no header")


def prepare_counts(path: Path, records: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    header = matrix_header(path)
    gene_candidates = [item for item in header if item.lower() in {"geneid", "gene_id", "gene"}]
    if len(gene_candidates) != 1:
        raise ValueError("Exactly one Geneid column is required")
    gene_col = gene_candidates[0]
    wanted = [r["matrix_column"] for r in records]
    missing = set(wanted) - set(header)
    if missing:
        raise ValueError(f"Selected columns absent from count matrix: {sorted(missing)}")
    table = pd.read_csv(
        path, compression="gzip", sep="\t", comment="#", usecols=[gene_col] + wanted,
        dtype=str, keep_default_na=False,
    )
    if len(table) < 10000:
        raise ValueError("Unexpectedly few gene rows; check the matrix format")
    genes = table[gene_col].astype(str).str.strip()
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError("Missing or duplicate gene IDs")
    values = table[wanted].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and non-negative")
    if not np.equal(values, np.floor(values)).all() or values.max() > np.iinfo(np.int64).max:
        raise ValueError("Expected non-negative integer read counts, not TPM or normalized values")
    values = values.astype(np.int64)
    library_sizes = values.sum(axis=0)
    if (library_sizes <= 0).any():
        raise ValueError("One or more samples have zero total reads")

    matrix = pd.DataFrame(values, columns=[r["gsm"] for r in records])
    matrix.insert(0, "gene_id", genes.to_numpy())
    stats = []
    for i, record in enumerate(records):
        stats.append({
            **record,
            "total_assigned_counts": int(library_sizes[i]),
            "genes_with_nonzero_counts": int(np.count_nonzero(values[:, i])),
        })
    return matrix, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data/raw" / ACCESSION / RAW_FILE
    if not source.is_file():
        raise FileNotFoundError(f"Run stage 01 to obtain {source}")
    records = validate_mapping(root)
    matrix, stats = prepare_counts(source, records)

    folder = root / "data/processed" / ACCESSION
    folder.mkdir(parents=True, exist_ok=True)
    counts_path = folder / "counts_timecourse_candidate.tsv.gz"
    metadata_path = folder / "samples_timecourse_candidate.csv"
    matrix.to_csv(counts_path, sep="\t", index=False, compression="gzip")
    pd.DataFrame(stats).to_csv(metadata_path, index=False)

    report = {
        "analysis": "GSE254485_timecourse_candidate_mapping_and_counts",
        "source": ACCESSION,
        "source_file": str(source.relative_to(root)),
        "source_sha256": sha256(source),
        "processed_counts": str(counts_path.relative_to(root)),
        "processed_counts_sha256": sha256(counts_path),
        "candidate_metadata": str(metadata_path.relative_to(root)),
        "candidate_metadata_sha256": sha256(metadata_path),
        "units": "featureCounts read counts; no normalization performed",
        "genes": len(matrix),
        "samples": len(records),
        "group_sizes": dict(sorted(Counter(str(r["day"]) for r in records).items())),
        "samples_qc": stats,
        "training_or_modeling_performed": False,
        "scientific_label_review_complete": False,
        "interpretation": [
            "Eight unique metadata aliases were checked against the matrix columns.",
            "Name-based aliases and replicate numbers do not prove sample provenance or biological independence.",
            "Day 0 samples are recorded as no treatment; later days are osimertinib treated.",
            "Long-exposure samples must not be relabeled DTP or resistant without source evidence.",
            "This count matrix is not directly comparable with stage 08 log2TPM expression.",
        ],
    }
    output = root / "results/gse254485_timecourse_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"GSE254485: {len(records)} candidate PC9 samples, {len(matrix)} genes")
    for row in stats:
        print(f"{row['gsm']} day={row['day']} rep={row['replicate_label']} "
              f"counts={row['total_assigned_counts']:,} genes>0={row['genes_with_nonzero_counts']:,}")
    print(f"Saved: {output}")
    print("Technical alias checks passed. Biological sample review remains pending.")
    print("No model training or DTP/resistance labeling was performed.")


if __name__ == "__main__":
    main()
