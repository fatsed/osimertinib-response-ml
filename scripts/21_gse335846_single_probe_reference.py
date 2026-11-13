"""Stage 21: build an annotation-only singleton-gene *sensitivity* reference.

This is NOT a complete gene-level matrix, statistical test, or a validated
expression scale. It never collapses probes, imputes, normalizes or logs values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE19 = Path("results/gse335846_phase_preparation.json")
STAGE20 = Path("results/gse335846_scale_audit.json")
DESTINATION = Path("data/processed/GSE335846/single_probe_gene_reference.tsv.gz")
REPORT = Path("results/gse335846_single_probe_reference.json")
ANNO_COLUMNS = ("feature_id", "probe_id", "source_gene_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required report: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_inputs(root: Path):
    previous = read_json(root / STAGE19)
    audit = read_json(root / STAGE20)
    if previous.get("analysis") != "GSE335846_cell_cycle_phase_data_preparation":
        raise ValueError("Wrong stage 19 input report")
    if audit.get("analysis") != "GSE335846_stage20_scale_and_probe_audit":
        raise ValueError("Wrong stage 20 input report")
    if previous.get("labels_verified") is not False or audit.get("approved_for_gene_level_analysis") is not False:
        raise ValueError("Unexpected scientific approval state; review previous stages")
    if audit.get("input_numeric_values_changed") is not False:
        raise ValueError("Stage 20 changed source numeric values")

    matrix_path = root / previous["processed_expression"]
    metadata_path = root / previous["candidate_metadata"]
    checks = [
        (matrix_path, previous["processed_expression_sha256"]),
        (metadata_path, previous["candidate_metadata_sha256"]),
    ]
    for path, expected in checks:
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed stage 19 input: {path}")
        rel = path.relative_to(root).as_posix()
        if audit.get("stage19_input_sha256", {}).get(rel) != expected:
            raise ValueError(f"Stage 20 checksum disagrees for {rel}")
    if audit.get("n_probe_rows") != previous.get("features") or audit.get("n_samples") != 9:
        raise ValueError("Stage 19/20 dimensions disagree")

    meta = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if len(meta) != 9 or meta["gsm"].duplicated().any() or meta["gsm"].eq("").any():
        raise ValueError("Expected nine unique GSM samples in original order")
    gsms = meta["gsm"].tolist()
    source = pd.read_csv(matrix_path, sep="\t", compression="gzip", dtype=str,
                         keep_default_na=False, low_memory=False)
    if (source.columns.duplicated().any() or not set(ANNO_COLUMNS + tuple(gsms)).issubset(source.columns)
            or len(source) != previous["features"]):
        raise ValueError("Unexpected matrix columns or row count")
    if source["feature_id"].duplicated().any() or source["feature_id"].eq("").any():
        raise ValueError("Source-row identifiers must be unique and nonempty")
    genes = source["source_gene_id"].str.strip()
    probe = source["probe_id"].str.strip()
    if int(genes.ne("").sum()) != len(source) - audit["identifier_audit"]["rows_missing_gene_annotation"]:
        raise ValueError("Stage 20 missing-gene count mismatch")
    if genes[genes.ne("")].nunique() != audit["identifier_audit"]["distinct_annotated_genes"]:
        raise ValueError("Stage 20 unique gene count mismatch")
    if int((probe.ne("") & probe.duplicated(keep=False)).sum()) != audit["identifier_audit"]["rows_with_duplicate_probe_id"]:
        raise ValueError("Stage 20 repeated-probe count mismatch")
    return previous, audit, source, meta, gsms, genes, probe


def select_singletons(source: pd.DataFrame, gsms: list[str], genes: pd.Series,
                      probes: pd.Series):
    occurrences = genes.loc[genes.ne("")].value_counts()
    single_gene = genes.isin(occurrences[occurrences.eq(1)].index)
    probe_occurrences = probes.loc[probes.ne("")].value_counts()
    unique_probe = probes.isin(probe_occurrences[probe_occurrences.eq(1)].index)
    singleton = single_gene & unique_probe
    result = source.loc[singleton, list(ANNO_COLUMNS) + gsms].copy()
    if result.empty:
        raise ValueError("No unique-annotation genes available")
    if (result["source_gene_id"].duplicated().any() or
            result["probe_id"].duplicated().any() or
            result["probe_id"].eq("").any()):
        raise ValueError("Singleton gene subset still has duplicate identifiers")
    numbers = result[gsms].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numbers).all():
        raise ValueError("Nonfinite source values in selected rows")
    # Keep strings (including negative values) unchanged: never transform numerical data.
    if not result[gsms].equals(source.loc[singleton, gsms]):
        raise AssertionError("Subset must preserve source values verbatim")
    return result, occurrences, singleton


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    prev, audit, source, meta, gsms, genes, probes = verify_inputs(root)
    selected, counts, mask = select_singletons(source, gsms, genes, probes)

    dest = root / DESTINATION
    dest.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(dest, sep="\t", index=False, compression="gzip")
    # Read output back and verify it preserves order, labels and numeric strings.
    roundtrip = pd.read_csv(dest, sep="\t", compression="gzip", dtype=str,
                            keep_default_na=False, low_memory=False)
    if not roundtrip.equals(selected.reset_index(drop=True)):
        dest.unlink(missing_ok=True)
        raise AssertionError("Round-trip validation failed; output removed")

    n_unique_genes = int(counts.size)
    n_singletons = len(selected)
    n_multiprobe_rows = int(genes.ne("").sum() - counts[counts.eq(1)].sum())
    n_singleton_excluded_bad_probe = int(counts.eq(1).sum() - n_singletons)
    report = {
        "analysis": "GSE335846_stage21_annotation_only_single_probe_reference",
        "dataset": "GSE335846",
        "source_stage19_sha256": prev["processed_expression_sha256"],
        "source_stage20_report": STAGE20.as_posix(),
        "input_rows": len(source),
        "distinct_annotated_genes": n_unique_genes,
        "retained_singleton_genes": n_singletons,
        "fraction_of_annotated_genes_retained": n_singletons / n_unique_genes,
        "rows_excluded_due_to_multi_probe_gene": n_multiprobe_rows,
        "single_gene_rows_excluded_due_to_missing_or_repeated_probe_id": n_singleton_excluded_bad_probe,
        "rows_with_missing_gene_annotation": int(genes.eq("").sum()),
        "n_samples": len(gsms),
        "sample_order": gsms,
        "input_scale": audit["scale_classification"],
        "output_expression": DESTINATION.as_posix(),
        "output_sha256": sha256(dest),
        "values_transformed": False,
        "probe_rows_collapsed": False,
        "source_data_edited": False,
        "supervised_model_trained": False,
        "approved_for_primary_gene_level_inference": False,
        "approved_for_external_validation": False,
        "intended_use": "Optional exploratory/sensitivity check after scale review; not a full gene-expression reference",
        "limitations": [
            "Singleton-gene annotation selection excludes all genes covered by multiple probe rows; coverage is biased and incomplete.",
            "GEO describes size-factor normalization but does not establish the exported numeric transformation or comparability.",
            "Negative finite values have been kept without clipping or log transformation.",
            "Gene-level analysis of the full dataset requires justified probe handling or re-quantification from raw reads.",
            "Nine samples are unbalanced across day/phase/clone, and biological independence is not verified.",
            "No PCA, gene-set inference, differential expression testing, or training was performed.",
        ],
        "metadata_reference": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE335846",
        "sample_processing_example": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM9822019",
    }
    path = root / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Source: {len(source)} probe rows; {n_unique_genes} annotated genes")
    print(f"Singleton-probe genes: {n_singletons} ({100 * n_singletons/n_unique_genes:.1f}% of annotated genes)")
    print(f"Excluded annotated multi-probe rows: {n_multiprobe_rows}")
    print(f"Singleton genes omitted for missing/repeated Probe IDs: {n_singleton_excluded_bad_probe}")
    print(f"Samples: {len(gsms)}; original numeric strings preserved: yes")
    print(f"Saved: {path}")
    print(f"Local matrix: {dest}")
    print("Sensitivity-only subset; exact numeric scale still unverified. No PCA or modeling.")


if __name__ == "__main__":
    main()
