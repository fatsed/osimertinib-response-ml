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
ACCESSION = "GSE335846"
SOURCE = "GSE335846_RNA-seq_for_GEO.txt.gz"
SAMPLE_TITLES = {
    "GSM9822015": "Day 0 G1 clone 1",
    "GSM9822016": "Day 0 G1 clone 7",
    "GSM9822017": "Day 0 S clone 1",
    "GSM9822018": "Day 0 S clone 7",
    "GSM9822019": "Day 0 G2 clone 1",
    "GSM9822020": "Day 28 G1 clone 1",
    "GSM9822021": "Day 28 G1 clone 7",
    "GSM9822022": "Day 28 S clone 1",
    "GSM9822023": "Day 28 G2 clone 1",
}
IDENTIFIER_NAMES = (
    "ensembl_gene_id", "ensembl", "gene stable id", "geneid", "gene_id",
    "gene id", "gene symbol", "gene name", "symbol", "gene", "feature", "id",
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local metadata: {path}; rerun stages 03 and 04")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return list(reader)


def normalize(value: str) -> str:
    text = re.sub(r"\s*[\[(]\s*RNA[-_ ]?seq\s*[\])]\s*$", "", value, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def verified_candidates(root: Path) -> list[dict]:
    review = read_csv(
        root / "data/metadata/review/sample_review_queue.csv",
        {"accession", "gsm", "title", "cell_line", "time_point", "treatment",
         "candidate_group", "analysis_role", "review_status"},
    )
    mapping = read_csv(
        root / "data/metadata/mapping/sample_matrix_mapping.csv",
        {"accession", "expression_file", "source_type", "column_or_member",
         "gsm", "mapping_status"},
    )
    cohort = [r for r in review if r["accession"] == ACCESSION]
    links = [r for r in mapping if r["accession"] == ACCESSION and r["expression_file"] == SOURCE]
    if len(cohort) != 9 or len(links) != 9:
        raise ValueError("Expected nine GEO records and nine source-matrix mappings")
    if Counter(r["gsm"] for r in cohort) != Counter(SAMPLE_TITLES.keys()):
        raise ValueError("Unexpected or repeated GEO GSM records")
    if Counter(r["gsm"] for r in links) != Counter(SAMPLE_TITLES.keys()):
        raise ValueError("Missing, repeated, or unexpected matrix-to-GSM correspondence")

    samples = {r["gsm"]: r for r in cohort}
    mapped = {r["gsm"]: r for r in links}
    selected = []
    for gsm, title in SAMPLE_TITLES.items():
        meta, link = samples[gsm], mapped[gsm]
        match = re.fullmatch(r"Day (0|28) (G1|S|G2) clone (1|7)", title)
        day, phase, clone = int(match[1]), match[2], int(match[3])
        if (normalize(meta["title"]) != normalize(title)
                or meta["cell_line"].upper() != "PC9"
                or meta["analysis_role"] != "cell_cycle_candidate"
                or meta["candidate_group"].lower() != f"day_{day}_{phase}".lower()
                or normalize(meta["time_point"]) != normalize(f"Day {day} {phase}")):
            raise ValueError(f"Review metadata does not match GEO title for {gsm}")
        if (link["source_type"] != "matrix_column"
                or link["mapping_status"] != "exact_title"
                or normalize(link["column_or_member"]) != normalize(title)):
            raise ValueError(f"Expression column does not match GEO title for {gsm}")
        selected.append({
            "gsm": gsm, "geo_title": title, "matrix_column": link["column_or_member"],
            "cell_line": meta["cell_line"], "day": day, "phase": phase,
            "clone": clone, "treatment": meta["treatment"],
            "mapping_status": link["mapping_status"],
            "review_status": meta["review_status"],
            "scientifically_verified": False,
        })
    if len({s["matrix_column"] for s in selected}) != 9:
        raise ValueError("A matrix column is assigned to multiple samples")
    return selected


def load_expression(path: Path, samples: list[dict]):
    if not path.is_file():
        raise FileNotFoundError(f"Missing downloaded GEO file: {path}; rerun stage 01")
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        first = stream.readline().rstrip("\r\n")
    delimiter = "\t" if first.count("\t") >= first.count(",") else ","
    header = next(csv.reader([first], delimiter=delimiter))
    if len(header) != len(set(header)) or not {"Probe", "ID"}.issubset(header):
        raise ValueError("Expected distinct columns 'Probe' and 'ID' in GSE335846 source")

    wanted = [sample["matrix_column"] for sample in samples]
    if not set(wanted).issubset(header):
        raise ValueError(f"Missing sample columns: {sorted(set(wanted) - set(header))}")

    annotations = [column for column in ("Feature", "Type", "Chromosome", "Start", "End")
                   if column in header]
    table = pd.read_csv(
        path, sep=delimiter, compression="gzip",
        usecols=["Probe", "ID"] + annotations + wanted,
        dtype=str, keep_default_na=False, low_memory=False,
    )
    if len(table) < 1000:
        raise ValueError("Unexpectedly few rows in the source expression table")

    probes = table["Probe"].str.strip()
    genes = table["ID"].str.strip()
    missing_probes = probes.eq("")
    duplicate_probes = probes.ne("") & probes.duplicated(keep=False)
    missing_genes = genes.eq("")
    repeated_genes = genes.ne("") & genes.duplicated(keep=False)

    if missing_probes.mean() > 0.1:
        raise ValueError("More than 10% of source Probe identifiers are missing")
    if genes[~missing_genes].nunique() < 1000:
        raise ValueError("Too few distinct Ensembl gene identifiers")

    raw_values = table[wanted].apply(pd.to_numeric, errors="raise")
    values = raw_values.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Source expression values contain NaN or Infinity")
    if (np.ptp(values, axis=0) <= 1e-12).any():
        raise ValueError("One or more sample columns are constant")

    output = pd.DataFrame({
        "feature_id": [f"SOURCE_ROW_{row:07d}" for row in range(1, len(table) + 1)],
        "probe_id": probes.to_numpy(),
        "source_gene_id": genes.to_numpy(),
    })
    if "Feature" in table:
        output["feature_name"] = table["Feature"].to_numpy()
    if "Type" in table:
        output["feature_type"] = table["Type"].to_numpy()
    for index, sample in enumerate(samples):
        output[sample["gsm"]] = values[:, index]

    identifier_qc = {
        "source_row_column": "Probe",
        "gene_annotation_column": "ID",
        "source_rows": len(table),
        "nonempty_probe_ids": int((~missing_probes).sum()),
        "unique_nonempty_probe_ids": int(probes[~missing_probes].nunique()),
        "missing_probe_rows": int(missing_probes.sum()),
        "repeated_probe_rows": int(duplicate_probes.sum()),
        "distinct_annotated_gene_ids": int(genes[~missing_genes].nunique()),
        "missing_gene_annotation_rows": int(missing_genes.sum()),
        "rows_belonging_to_repeated_gene_ids": int(repeated_genes.sum()),
        "gene_rows_are_independent_genes": False,
        "source_rows_aggregated": False,
        "feature_id_meaning": "Stable source-row key; probe_id and source_gene_id are kept separately",
    }

    stats = []
    for index, sample in enumerate(samples):
        column = values[:, index]
        negative = column[column < 0]
        if len(negative):
            distinct, counts = np.unique(negative, return_counts=True)
            most = int(np.argmax(counts))
            common = float(distinct[most])
            common_count = int(counts[most])
        else:
            distinct = np.empty(0)
            common = None
            common_count = 0
        stats.append({
            **sample,
            "features_with_positive_values": int(np.count_nonzero(column > 0)),
            "features_with_zero_values": int(np.count_nonzero(column == 0)),
            "features_with_negative_values": int(len(negative)),
            "minimum_value": float(column.min()),
            "maximum_value": float(column.max()),
            "distinct_negative_values": int(len(distinct)),
            "most_common_negative_value": common,
            "most_common_negative_value_frequency": common_count,
        })
    fractional = bool(np.any(values != np.floor(values)))
    return output, stats, identifier_qc, fractional


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    samples = verified_candidates(root)
    source = root / "data/raw" / ACCESSION / SOURCE
    matrix, stats, identifier_qc, fractional = load_expression(source, samples)

    directory = root / "data/processed" / ACCESSION
    directory.mkdir(parents=True, exist_ok=True)
    expression_path = directory / "expression_phase_candidate.tsv.gz"
    metadata_path = directory / "samples_phase_candidate.csv"
    matrix.to_csv(expression_path, sep="\t", index=False, compression="gzip", float_format="%.17g")
    pd.DataFrame(stats).to_csv(metadata_path, index=False)
    group_counts = Counter(f"Day_{s['day']}_{s['phase']}" for s in samples)
    report = {
        "analysis": "GSE335846_cell_cycle_phase_data_preparation",
        "source": ACCESSION,
        "geo_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335846",
        "source_file": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": sha256(source),
        "processed_expression": str(expression_path.relative_to(root)).replace("\\", "/"),
        "processed_expression_sha256": sha256(expression_path),
        "candidate_metadata": str(metadata_path.relative_to(root)).replace("\\", "/"),
        "candidate_metadata_sha256": sha256(metadata_path),
        "source_row_identifier_column": "Probe",
        "gene_annotation_column": "ID",
        "identifier_quality": identifier_qc,
        "features": len(matrix),
        "distinct_annotated_genes": identifier_qc["distinct_annotated_gene_ids"],
        "samples": len(samples),
        "group_sizes": dict(sorted(group_counts.items())),
        "fractional_expression_values_observed": fractional,
        "expression_units": "Submitter-exported processed values; GEO describes size-factor normalisation, exact numeric scale unverified",
        "numeric_qc": {
            "all_values_finite": True,
            "minimum_value": float(matrix[[s["gsm"] for s in samples]].to_numpy().min()),
            "maximum_value": float(matrix[[s["gsm"] for s in samples]].to_numpy().max()),
            "negative_values_total": sum(s["features_with_negative_values"] for s in stats),
            "samples_with_negative_values": [
                s["gsm"] for s in stats if s["features_with_negative_values"] > 0
            ],
            "negative_values_preserved_unchanged": True,
            "numeric_scale_requires_source_verification": True,
        },
        "sample_qc": stats,
        "labels_verified": False,
        "supervised_model_trained": False,
        "limitations": [
            "The original matrix is described as size-factor normalised; do not assume raw read counts.",
            "Negative finite source values were preserved; the exact export transformation and baseline offset are not verified.",
            "Do not clip, exponentiate, apply log2, or treat source values as raw counts without documentation.",
            "No additional CPM or DESeq2 size-factor normalisation was applied.",
            "G1, S and G2 have unequal sample numbers across days.",
            "Clone labels do not prove independent biological replicates or longitudinal matching.",
            "No DTP-specificity or drug-response classification is inferred from these groups.",
            "SOURCE_ROW feature IDs are positional keys and do not identify independent genes.",
            "Probe identifies the source measurement; ID gives its gene annotation.",
            "Multiple probes may annotate the same Ensembl gene; no probe rows were averaged or collapsed.",
            "Probe-level observations cannot be used as independent biological replicates.",
            "Gene-set matching, cross-study merging or gene-level PCA need a validated gene-level summarisation rule.",
        ],
    }
    output = root / "results/gse335846_phase_preparation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{ACCESSION}: {len(samples)} mapped samples; {len(matrix)} source probe rows")
    print(f"Distinct annotated Ensembl genes: {identifier_qc['distinct_annotated_gene_ids']}")
    print(f"Unique Probe IDs: {identifier_qc['unique_nonempty_probe_ids']} | "
          f"missing Probe rows: {identifier_qc['missing_probe_rows']} | "
          f"duplicate Probe rows: {identifier_qc['repeated_probe_rows']}")
    print(f"Rows with repeated gene annotation: {identifier_qc['rows_belonging_to_repeated_gene_ids']} | "
          f"missing gene annotation rows: {identifier_qc['missing_gene_annotation_rows']}")
    print(f"Fractional expression values: {fractional}")
    print(f"Finite source-value range: {report['numeric_qc']['minimum_value']:.9f} "
          f"to {report['numeric_qc']['maximum_value']:.9f}")
    print(f"Negative source values preserved: {report['numeric_qc']['negative_values_total']}")
    for row in stats:
        if row["features_with_negative_values"]:
            print(f"  {row['geo_title']}: {row['features_with_negative_values']} negatives, "
                  f"{row['distinct_negative_values']} distinct, "
                  f"most common={row['most_common_negative_value']:.9f} "
                  f"(n={row['most_common_negative_value_frequency']})")
    for key, count in sorted(group_counts.items()):
        print(f"{key}: {count}")
    print(f"Saved: {output}")
    print("Technical title matching passed; biological verification still pending.")
    print("Numeric scale remains unverified; no log transform, clipping or model fitting was performed.")


if __name__ == "__main__":
    main()
