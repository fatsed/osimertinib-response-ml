from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE193258"
GROUPS = ("Control", "Acute", "DTP")
SOURCES = {
    "estimated_counts": "GSE193258_RNAseq_estimated_counts.tsv.gz",
    "log2tpm": "GSE193258_RNAseq_log2TPM_abundance.tsv.gz",
}
REVIEW = ROOT / "data/metadata/review/sample_review_queue.csv"
MAPPING = ROOT / "data/metadata/mapping/sample_matrix_mapping.csv"
RAW = ROOT / "data/raw" / ACCESSION
OUTPUT = ROOT / "data/processed" / ACCESSION


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_samples() -> pd.DataFrame:
    review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    samples = review.loc[
        (review["accession"] == ACCESSION)
        & (review["candidate_cohort"] == "GSE193258_3state")
        & (review["candidate_group"].isin(GROUPS))
        & (review["analysis_role"] == "three_state_candidate")
    ].copy()

    if len(samples) != 36 or samples["gsm"].duplicated().any():
        raise ValueError("Expected 36 distinct GSM records for Control, Acute and DTP")
    if samples["cell_line"].nunique() != 4:
        raise ValueError("Expected four cell lines")
    for _, group in samples.groupby(["cell_line", "candidate_group"]):
        if len(group) != 3:
            raise ValueError("Each cell-line/state combination needs three records")
    if len(samples.groupby(["cell_line", "candidate_group"])) != 12:
        raise ValueError("Incomplete cell-line/state combinations")
    if samples[["gsm", "cell_line", "candidate_group", "title"]].eq("").any().any():
        raise ValueError("Required sample metadata are missing")

    return samples.sort_values(["cell_line", "candidate_group", "gsm"]).reset_index(drop=True)


def load_mapping(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
    mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    mapping = mapping.loc[mapping["accession"] == ACCESSION]
    wanted = set(samples["gsm"])
    result = {}

    for kind, filename in SOURCES.items():
        rows = mapping.loc[
            (mapping["expression_file"] == filename)
            & (mapping["source_type"] == "matrix_column")
            & (mapping["gsm"].isin(wanted))
        ].copy()
        if len(rows) != 36 or set(rows["gsm"]) != wanted:
            raise ValueError(f"Sample mapping is incomplete for {filename}")
        if rows["gsm"].duplicated().any() or rows["column_or_member"].duplicated().any():
            raise ValueError(f"Duplicate GSM or expression column in {filename}")
        if not rows["mapping_status"].eq("exact_title").all():
            raise ValueError(f"Non-exact mapping among selected samples in {filename}")
        expected_group = dict(zip(samples["gsm"], samples["candidate_group"]))
        if any(row["candidate_group"] != expected_group[row["gsm"]] for _, row in rows.iterrows()):
            raise ValueError(f"Group labels disagree for {filename}")
        result[kind] = rows.set_index("gsm").loc[samples["gsm"]].reset_index()

    return result


def prepare_matrix(path: Path, sample_map: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict]:
    if not path.is_file():
        raise FileNotFoundError(path)

    header = pd.read_csv(path, sep="\t", compression="gzip", nrows=0).columns.tolist()
    if not header:
        raise ValueError(f"No columns in {path.name}")
    columns = sample_map["column_or_member"].tolist()
    if header[0] in columns or any(col not in header for col in columns):
        raise ValueError(f"Missing or invalid mapped columns in {path.name}")

    frame = pd.read_csv(
        path, sep="\t", compression="gzip", dtype=str,
        keep_default_na=False, usecols=[header[0], *columns], low_memory=False,
    )
    frame = frame.loc[:, [header[0], *columns]]
    features = frame[header[0]].astype(str).str.strip()
    if features.eq("").any():
        raise ValueError(f"Empty gene identifiers in {path.name}")

    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Missing, nonnumeric or infinite values in {path.name}")
    if kind == "estimated_counts" and (numeric < 0).any():
        raise ValueError("Negative values found in estimated counts")

    names = sample_map["gsm"].tolist()
    values.columns = names
    prepared = pd.concat([features.rename("gene_id"), values], axis=1)
    summary = {
        "source_file": path.name,
        "source_sha256": checksum(path),
        "original_gene_column": header[0],
        "n_features": len(features),
        "n_selected_samples": len(names),
        "duplicate_gene_id_rows": int(features.duplicated().sum()),
        "zero_values": int((numeric == 0).sum()),
        "zero_fraction": float((numeric == 0).mean()),
        "minimum_value": float(numeric.min()),
        "maximum_value": float(numeric.max()),
    }
    if kind == "estimated_counts":
        totals = values.sum(axis=0)
        summary["estimated_count_totals_by_gsm"] = {gsm: float(total) for gsm, total in totals.items()}
        summary["fractional_value_count"] = int(np.count_nonzero(numeric != np.floor(numeric)))
    return prepared, summary


def main() -> None:
    samples = load_samples()
    maps = load_mapping(samples)
    matrices = {}
    qc = {}

    for kind, filename in SOURCES.items():
        matrices[kind], qc[kind] = prepare_matrix(RAW / filename, maps[kind], kind)
        print(f"{kind}: {qc[kind]['n_features']} features, 36 samples")

    genes_counts = matrices["estimated_counts"]["gene_id"]
    genes_tpm = matrices["log2tpm"]["gene_id"]
    if not genes_counts.equals(genes_tpm):
        raise ValueError("Gene identifiers/order differ between counts and log2TPM; no files written")

    sample_columns = samples["gsm"].tolist()
    if any(list(matrix.columns[1:]) != sample_columns for matrix in matrices.values()):
        raise ValueError("Sample column orders disagree")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {
        "estimated_counts": "estimated_counts_36samples.tsv.gz",
        "log2tpm": "log2tpm_36samples.tsv.gz",
    }
    for kind, filename in files.items():
        matrices[kind].to_csv(OUTPUT / filename, sep="\t", index=False, compression="gzip")

    samples_out = samples[["gsm", "title", "cell_line", "candidate_group", "treatment", "time_point", "review_status"]].copy()
    samples_out.insert(0, "accession", ACCESSION)
    samples_out["counts_column"] = maps["estimated_counts"]["column_or_member"].tolist()
    samples_out["log2tpm_column"] = maps["log2tpm"]["column_or_member"].tolist()
    samples_out["mapping_status"] = "exact_title"
    samples_out.to_csv(OUTPUT / "samples_36.csv", index=False)

    report = {
        "accession": ACCESSION,
        "status": "technical_preparation_only_not_biological_validation",
        "selected_groups": list(GROUPS),
        "n_samples": len(samples),
        "sample_counts_by_cell_line_and_group": {
            f"{cell_line}:{group}": int(count)
            for (cell_line, group), count in samples.groupby(["cell_line", "candidate_group"]).size().items()
        },
        "review_sha256": checksum(REVIEW),
        "mapping_sha256": checksum(MAPPING),
        "matrices": qc,
        "cautions": [
            "Sample groups are candidates until the original study protocol is reviewed.",
            "No filtering, normalization, feature selection or scaling has been fitted.",
            "Washout and alias_candidate samples are excluded from these outputs.",
            "Estimated counts may be fractional and are not automatically rounded.",
        ],
    }
    (OUTPUT / "qc_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Prepared {len(samples)} samples across four cell lines")
    print(f"Files saved in: {OUTPUT}")
    print("No model has been trained; scientific label review remains open")


if __name__ == "__main__":
    main()
