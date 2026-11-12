from __future__ import annotations

import gzip
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE255958"
REVIEW = ROOT / "data/metadata/review/sample_review_queue.csv"
MAPPING = ROOT / "data/metadata/mapping/sample_matrix_mapping.csv"
ARCHIVE = ROOT / "data/raw/GSE255958/GSE255958_RAW.tar"
REFERENCE = ROOT / "data/processed/GSE193258/log2tpm_36samples.tsv.gz"
OUTPUT = ROOT / "results/gse255958_compatibility.json"
GROUPS = {"Control", "Acute", "DTP", "Resistant"}
REPEATED_NAME = re.compile(r"^(.+)_\1$")


def normalize_gene_id(value: str) -> str:
    value = value.strip()
    match = REPEATED_NAME.fullmatch(value)
    return match.group(1) if match else value


def selected_samples() -> pd.DataFrame:
    review = pd.read_csv(REVIEW, dtype=str, keep_default_na=False)
    selected = review.loc[
        (review["accession"] == ACCESSION)
        & (review["candidate_cohort"] == "GSE255958_PC9_4state")
        & (review["analysis_role"] == "four_state_candidate")
    ].copy()

    if len(selected) != 12 or selected["gsm"].nunique() != 12:
        raise ValueError("Expected 12 distinct PC9 GEO records")
    if selected["candidate_group"].value_counts().to_dict() != {group: 3 for group in GROUPS}:
        raise ValueError("Expected three samples in each of four groups")
    if not selected["cell_line"].eq("PC9").all():
        raise ValueError("Unexpected cell line in selected samples")
    return selected.sort_values("gsm").reset_index(drop=True)


def selected_members(selected: pd.DataFrame) -> dict[str, str]:
    mapping = pd.read_csv(MAPPING, dtype=str, keep_default_na=False)
    rows = mapping.loc[
        (mapping["accession"] == ACCESSION)
        & (mapping["expression_file"] == ARCHIVE.name)
        & (mapping["source_type"] == "tar_member")
        & (mapping["mapping_status"] == "exact_gsm")
        & (mapping["gsm"].isin(selected["gsm"]))
    ]
    if len(rows) != 12 or rows["gsm"].nunique() != 12 or rows["column_or_member"].nunique() != 12:
        raise ValueError("Missing or duplicated GSM-to-file mappings")
    return dict(zip(rows["gsm"], rows["column_or_member"]))


def read_rsem_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> pd.DataFrame:
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Cannot read archive member: {member.name}")
    with stream, gzip.GzipFile(fileobj=stream, mode="rb") as unzipped:
        table = pd.read_csv(unzipped, sep="\t", usecols=["gene_id", "TPM"], dtype={"gene_id": str})

    raw_ids = table["gene_id"].astype("string").str.strip()
    if table.empty or raw_ids.isna().any() or raw_ids.eq("").any() or raw_ids.duplicated().any():
        raise ValueError(f"Empty or repeated original gene IDs: {member.name}")

    tpm = pd.to_numeric(table["TPM"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(tpm).all() or (tpm < 0).any():
        raise ValueError(f"Invalid TPM values: {member.name}")

    table["gene_id"] = raw_ids.astype(str)
    table["TPM"] = tpm
    return table


def inspect(reference: Path, archive_path: Path, selected: pd.DataFrame, member_names: dict[str, str]) -> dict:
    source = pd.read_csv(reference, sep="\t", compression="gzip", usecols=["gene_id"], dtype=str)
    source_ids = source["gene_id"].astype("string").str.strip()
    if source_ids.isna().any() or source_ids.eq("").any() or source_ids.duplicated().any():
        raise ValueError("Invalid reference gene identifiers")
    source_set = set(source_ids.astype(str))

    report = {
        "source_study": "GSE193258",
        "target_study": ACCESSION,
        "source_genes": len(source_set),
        "target_samples": len(selected),
        "groups": selected["candidate_group"].value_counts().sort_index().to_dict(),
        "normalization_rule": "X_X becomes X only when both parts are exactly identical",
        "source_gene_id_examples": sorted(source_set)[:6],
        "samples": [],
    }

    overlap_all = None
    raw_gene_sets = []
    normalized_gene_sets = []
    with tarfile.open(archive_path, "r") as archive:
        members = {}
        for member in archive.getmembers():
            if member.isfile():
                name = Path(member.name).name
                if name in members:
                    raise ValueError(f"Repeated filename in TAR: {name}")
                members[name] = member

        for sample in selected.itertuples(index=False):
            name = member_names[sample.gsm]
            if name not in members:
                raise FileNotFoundError(f"Missing file in TAR: {name}")
            table = read_rsem_member(archive, members[name])
            original = table["gene_id"].tolist()
            normalized = [normalize_gene_id(gene) for gene in original]
            repeated = pd.Series(normalized).duplicated(keep=False).to_numpy()
            valid_set = {gene for gene, duplicate in zip(normalized, repeated) if not duplicate}
            shared = valid_set & source_set
            overlap_all = shared if overlap_all is None else overlap_all & shared
            raw_gene_sets.append(set(original))
            normalized_gene_sets.append(valid_set)

            changed = [(before, after) for before, after in zip(original, normalized) if before != after]
            report["samples"].append({
                "gsm": sample.gsm,
                "group": sample.candidate_group,
                "file": name,
                "target_genes": len(original),
                "ids_normalized": len(changed),
                "ids_excluded_due_to_normalization_collisions": int(repeated.sum()),
                "overlap_before_normalization": len(set(original) & source_set),
                "overlap_with_source": len(shared),
                "zero_tpm_genes": int((table["TPM"] == 0).sum()),
            })
            if "target_gene_id_examples" not in report:
                report["target_gene_id_examples"] = original[:6]
                report["normalization_examples"] = [
                    {"before": before, "after": after} for before, after in changed[:6]
                ]

    common = overlap_all if overlap_all is not None else set()
    report["genes_shared_by_all_12_and_source"] = len(common)
    report["source_to_target_shared_fraction"] = round(len(common) / len(source_set), 4)
    report["all_target_gene_sets_identical"] = all(ids == raw_gene_sets[0] for ids in raw_gene_sets[1:])
    report["all_normalized_gene_sets_identical"] = all(ids == normalized_gene_sets[0] for ids in normalized_gene_sets[1:])
    report["status"] = "gene_names_overlap_found" if common else "no_overlap_after_safe_normalization"
    report["interpretation"] = (
        "Matching gene names alone does not verify expression-scale compatibility, sample labels, "
        "treatment comparability, independence, or cross-study prediction validity. "
        "Any identifiers that collide after normalization are excluded, not merged."
    )
    return report


def main() -> None:
    selected = selected_samples()
    members = selected_members(selected)
    for path in (REFERENCE, ARCHIVE):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = inspect(REFERENCE, ARCHIVE, selected, members)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    first = report["samples"][0]
    print(f"Samples inspected: {report['target_samples']}")
    print(f"Source genes: {report['source_genes']}")
    print(f"Target genes in first sample: {first['target_genes']}")
    print(f"Normalized IDs in first sample: {first['ids_normalized']}")
    print(f"Colliding IDs excluded in first sample: {first['ids_excluded_due_to_normalization_collisions']}")
    print(f"Shared genes before normalization: {first['overlap_before_normalization']}")
    print(f"Shared genes across source and all 12 samples: {report['genes_shared_by_all_12_and_source']}")
    print(f"All target gene sets identical: {report['all_target_gene_sets_identical']}")
    print(f"Report: {OUTPUT}")
    print("No model trained; cross-study label and expression-scale review remains open.")


if __name__ == "__main__":
    main()
