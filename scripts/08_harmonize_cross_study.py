from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPEATED_GENE = re.compile(r"^(.+)_\1$")
PSEUDOCOUNT = 0.01
GROUPS = ("Control", "Acute", "DTP")


def normalize_gene_id(value: str) -> str:
    value = value.strip()
    match = REPEATED_GENE.fullmatch(value)
    return match.group(1) if match else value


def load_cohorts(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    prepared = root / "data/processed/GSE193258/samples_36.csv"
    reviewed = root / "data/metadata/review/sample_review_queue.csv"
    mapping = root / "data/metadata/mapping/sample_matrix_mapping.csv"
    source = pd.read_csv(prepared, dtype=str, keep_default_na=False)
    review = pd.read_csv(reviewed, dtype=str, keep_default_na=False)
    links = pd.read_csv(mapping, dtype=str, keep_default_na=False)

    target = review.loc[
        (review["accession"] == "GSE255958")
        & (review["candidate_cohort"] == "GSE255958_PC9_4state")
        & (review["analysis_role"] == "four_state_candidate")
    ].sort_values("gsm").copy()

    if len(source) != 36 or source["gsm"].nunique() != 36:
        raise ValueError("Source must contain 36 unique samples")
    if source["candidate_group"].value_counts().to_dict() != {g: 12 for g in GROUPS}:
        raise ValueError("Source groups do not match the reviewed cohort")
    if len(target) != 12 or target["gsm"].nunique() != 12:
        raise ValueError("Target must contain 12 unique samples")
    if target["candidate_group"].value_counts().to_dict() != {
        "Control": 3, "Acute": 3, "DTP": 3, "Resistant": 3
    }:
        raise ValueError("Target groups do not match the reviewed cohort")
    if not target["cell_line"].eq("PC9").all():
        raise ValueError("Target cohort contains non-PC9 samples")
    if source["gsm"].duplicated().any() or set(source["gsm"]) & set(target["gsm"]):
        raise ValueError("Duplicate sample identifiers")

    archive_name = "GSE255958_RAW.tar"
    links = links.loc[
        (links["accession"] == "GSE255958")
        & (links["expression_file"] == archive_name)
        & (links["source_type"] == "tar_member")
        & (links["mapping_status"] == "exact_gsm")
        & (links["gsm"].isin(target["gsm"]))
    ]
    if len(links) != 12 or links["gsm"].nunique() != 12 or links["column_or_member"].nunique() != 12:
        raise ValueError("Target GSM-to-TAR mapping is incomplete or non-unique")
    return source, target, dict(zip(links["gsm"], links["column_or_member"]))


def load_source(path: Path, source: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip", keep_default_na=False)
    wanted = source["gsm"].tolist()
    if "gene_id" not in frame or any(gsm not in frame for gsm in wanted):
        raise ValueError("Source gene or sample columns are missing")
    if len(frame) != 19712 or frame["gene_id"].isna().any():
        raise ValueError("Unexpected source gene count or missing identifiers")
    ids = frame["gene_id"].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Invalid source gene identifiers")
    values = frame[wanted].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Source matrix has non-finite expression values")
    result = pd.DataFrame(values, index=ids, columns=wanted)
    result.index.name = "gene_id"
    return result


def read_target(archive: tarfile.TarFile, member: tarfile.TarInfo) -> tuple[pd.Series, dict]:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read TAR member {member.name}")
    with stream, gzip.GzipFile(fileobj=stream) as unpacked:
        table = pd.read_csv(unpacked, sep="\t", usecols=["gene_id", "TPM"],
                            dtype={"gene_id": str}, keep_default_na=False)

    raw_ids = table["gene_id"].astype(str).str.strip()
    if table.empty or raw_ids.eq("").any() or raw_ids.duplicated().any():
        raise ValueError(f"Invalid RSEM gene identifiers in {member.name}")
    values = pd.to_numeric(table["TPM"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"Invalid RSEM TPM values in {member.name}")

    normalized = pd.Series([normalize_gene_id(x) for x in raw_ids])
    duplicates = normalized.duplicated(keep=False).to_numpy()
    genes = normalized.loc[~duplicates].tolist()
    retained = values[~duplicates]
    series = pd.Series(retained, index=genes)
    if not series.index.is_unique:
        raise ValueError(f"Non-unique normalized genes in {member.name}")
    return series, {
        "original_genes": int(len(table)),
        "collision_rows_excluded": int(duplicates.sum()),
        "original_tpm_sum": float(values.sum()),
        "zero_tpm_count": int(np.count_nonzero(values == 0)),
    }


def build_sample_table(source: pd.DataFrame, target: pd.DataFrame,
                       members: dict[str, str]) -> pd.DataFrame:
    rows = []
    for row in source.itertuples(index=False):
        rows.append({
            "accession": "GSE193258", "gsm": row.gsm, "cell_line": row.cell_line,
            "state": row.candidate_group, "treatment": row.treatment,
            "time_point": row.time_point, "title": row.title,
            "source_file": "log2tpm_36samples.tsv.gz",
            "evaluation_role": "source_candidate",
            "label_review": row.review_status,
        })
    for row in target.itertuples(index=False):
        rows.append({
            "accession": "GSE255958", "gsm": row.gsm, "cell_line": row.cell_line,
            "state": row.candidate_group, "treatment": row.treatment,
            "time_point": row.time_point, "title": row.title,
            "source_file": members[row.gsm],
            "evaluation_role": "target_candidate" if row.candidate_group in GROUPS else "exploratory_only",
            "label_review": row.review_status,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    archive_path = root / "data/raw/GSE255958/GSE255958_RAW.tar"
    source_path = root / "data/processed/GSE193258/log2tpm_36samples.tsv.gz"
    report_path = root / "results/gse255958_compatibility.json"
    for path in (archive_path, source_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_samples, target_samples, file_map = load_cohorts(root)
    source = load_source(source_path, source_samples)

    target_series = {}
    target_qc = {}
    gene_set = None
    with tarfile.open(archive_path, "r") as archive:
        files = {}
        for item in archive.getmembers():
            if item.isfile():
                filename = Path(item.name).name
                if filename in files:
                    raise ValueError(f"Duplicate filename inside TAR: {filename}")
                files[filename] = item
        for sample in target_samples.itertuples(index=False):
            name = file_map[sample.gsm]
            if name not in files:
                raise FileNotFoundError(f"Missing TAR member: {name}")
            series, qc = read_target(archive, files[name])
            current = set(series.index)
            if gene_set is None:
                gene_set = current
            elif current != gene_set:
                raise ValueError("Target samples do not have the same normalized gene set")
            target_series[sample.gsm] = series
            target_qc[sample.gsm] = qc

    common = [gene for gene in source.index if gene in gene_set]
    if len(common) < 10000:
        raise ValueError(f"Too few shared genes: {len(common)}. Check gene identifiers")
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        expected = previous.get("genes_shared_by_all_12_and_source")
        if expected and expected != len(common):
            raise ValueError(f"Shared-gene count differs from step 07: {expected} vs {len(common)}")

    target = pd.DataFrame({gsm: data.loc[common].to_numpy(dtype=float)
                           for gsm, data in target_series.items()}, index=common)
    target = np.log2(target + PSEUDOCOUNT)
    source = source.loc[common]
    target = target.loc[common, target_samples["gsm"].tolist()]
    if not np.isfinite(source.to_numpy()).all() or not np.isfinite(target.to_numpy()).all():
        raise ValueError("Non-finite values found after transformation")

    sample_table = build_sample_table(source_samples, target_samples, file_map)
    source_min = float(source.min().min())
    reference_floor = float(np.log2(PSEUDOCOUNT))
    floor_consistent = bool(abs(source_min - reference_floor) < 0.0001)
    report = {
        "status": "provisional_alignment_not_training_ready",
        "source_study": "GSE193258",
        "target_study": "GSE255958",
        "shared_genes": len(common),
        "source_samples": source.shape[1],
        "target_samples": target.shape[1],
        "target_shared_state_samples": int(target_samples["candidate_group"].isin(GROUPS).sum()),
        "target_resistant_samples_exploratory_only": int((target_samples["candidate_group"] == "Resistant").sum()),
        "source_scale": "GEO-provided log2TPM (transformation formula not independently verified)",
        "target_scale": "candidate log2(RSEM gene TPM + 0.01)",
        "target_pseudocount": PSEUDOCOUNT,
        "source_minimum_after_intersection": source_min,
        "source_minimum_consistent_with_log2_0_01_floor": floor_consistent,
        "target_tpm_quality": target_qc,
        "conditions": {
            "gene_order_aligned": list(source.index) == list(target.index),
            "labels_from_review_csv_not_independently_verified": True,
            "source_and_target_protocols_not_equivalent": True,
            "target_values_not_used_for_scaling_or_feature_selection": True,
        },
        "remaining_checks": [
            "Verify the source study's exact log2TPM transformation before pooling expression scales.",
            "Confirm the acute/DTP definitions, osimertinib dose, and exposure durations in both studies.",
            "Compare cross-study distributions and library/quantification effects without fitting to held-out data.",
            "All model feature selection, imputation and scaling must be fit on training data only.",
        ],
    }
    out = root / "data/processed/cross_study"
    out.mkdir(parents=True, exist_ok=True)
    source.to_csv(out / "gse193258_common_log2tpm.tsv.gz", sep="\t", compression="gzip",
                  index_label="gene_id")
    target.to_csv(out / "gse255958_common_log2tpm_candidate.tsv.gz", sep="\t", compression="gzip",
                  index_label="gene_id")
    sample_table.to_csv(out / "samples.csv", index=False)
    (out / "scale_review.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Shared genes: {len(common)}")
    print(f"GSE193258: {source.shape[1]} samples | GSE255958: {target.shape[1]} samples")
    print(f"Cross-study shared states: {report['target_shared_state_samples']} target samples")
    print(f"Source log2TPM floor consistent with +0.01: {floor_consistent}")
    print(f"Saved: {out}")
    print("Provisional alignment only; no model trained or test data used for scaling.")


if __name__ == "__main__":
    main()
