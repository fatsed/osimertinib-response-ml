"""Stage 34: source-only gene-universe correction for exploratory nested LOCO.

Replays the *unchanged* Stage-31 candidate grid, split rules, and selection
policy on the original Stage-05 GSE193258 gene universe (19,712 genes), NOT
on the Stage-08 target-intersected 18,013-gene matrix.

This is a sensitivity/reproducibility experiment on previously inspected data,
not an independent test or a new model/parameter search. No GSE255958
expression or outcome data are read. Historic Stage-32/33 results are unchanged.

Usage:
  python scripts/34_source_only_rebaseline.py --check-only
  python scripts/34_source_only_rebaseline.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREPARED = Path("data/processed/GSE193258/log2tpm_36samples.tsv.gz")
METADATA = Path("data/processed/GSE193258/samples_36.csv")
PREP_QC = Path("data/processed/GSE193258/qc_summary.json")
SOURCE_RAW = Path("data/raw/GSE193258/GSE193258_RNAseq_log2TPM_abundance.tsv.gz")
REVIEW = Path("data/metadata/review/sample_review_queue.csv")
MAPPING = Path("data/metadata/mapping/sample_matrix_mapping.csv")
PROTOCOL = Path("results/dcsl_stage31_evaluation_protocol.json")
PRIOR = Path("results/dcsl_stage32_nested_loco.json")
OLD_SOURCE = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
BASELINE = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/dcsl_stage34_source_only_nested_loco.json")
FIGURE = Path("figures/dcsl_stage34_source_only_nested_loco_dtp_recall.png")


def import_stage32(root: Path):
    file = root / "scripts/32_run_nested_loco.py"
    spec = importlib.util.spec_from_file_location("stage32_frozen_for_stage34", file)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_cohort(meta: pd.DataFrame, protocol: dict, *, expected_count: int = 36) -> None:
    required = {"gsm", "cell_line", "candidate_group", "review_status", "mapping_status"}
    if not required.issubset(meta) or len(meta) != expected_count or meta.gsm.duplicated().any():
        raise ValueError("Invalid Stage-05 sample metadata")
    if not meta["mapping_status"].eq("exact_title").all():
        raise ValueError("Source GSM mappings are not exact-title matches")
    if not meta["review_status"].eq("PENDING_MANUAL_AND_MATRIX_CHECK").all():
        raise ValueError("Unexpected scientific review status: do not promote labels automatically")
    allowed = {"Control", "Acute", "DTP"}
    if set(meta.candidate_group) != allowed:
        raise ValueError("Unexpected source classes")
    expected_gsms = set()
    expected_line = {}
    for fold in protocol["nested_loco_protocol"]["folds"]:
        line = fold["outer_test_cell_line"]
        for gsm in fold["outer_test_gsm"]:
            if gsm in expected_gsms:
                raise ValueError("Repeated GSM in frozen protocol")
            expected_gsms.add(gsm)
            expected_line[gsm] = line
    if set(meta.gsm) != expected_gsms:
        raise ValueError("Source-only cohort does not match frozen Stage-31 GSM set")
    if any(row.cell_line != expected_line[row.gsm] for row in meta.itertuples()):
        raise ValueError("Cell-line metadata disagree with frozen protocol")
    counts = meta.groupby(["cell_line", "candidate_group"]).size()
    if len(counts) != 12 or not counts.eq(3).all():
        raise ValueError("Need 4 cell lines, 3 states and 3 samples per cell")


def validate_matrix(frame: pd.DataFrame, gsm_order: list[str], expected_genes: int = 19712):
    if ("gene_id" not in frame or len(frame) != expected_genes or
            frame.columns.duplicated().any() or set(frame.columns) != {"gene_id", *gsm_order}):
        raise ValueError("Unexpected original source-only expression matrix")
    ids = frame["gene_id"].astype(str)
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Duplicate or blank gene IDs")
    x = frame[gsm_order].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64).T
    if not np.isfinite(x).all():
        raise ValueError("Non-finite source-only expression data")
    return ids.to_numpy(), x


def verify_stage05_qc(root: Path, stage32) -> dict:
    qc = json.loads((root / PREP_QC).read_text(encoding="utf-8"))
    mat = qc.get("matrices", {}).get("log2tpm", {})
    if (qc.get("accession") != "GSE193258" or qc.get("n_samples") != 36
            or mat.get("n_features") != 19712
            or qc.get("status") != "technical_preparation_only_not_biological_validation"):
        raise ValueError("Stage-05 technical preparation report is not as expected")
    for path, key in ((REVIEW, "review_sha256"), (MAPPING, "mapping_sha256")):
        if stage32.sha256(root / path) != qc.get(key):
            raise ValueError(f"Stage-05 provenance hash mismatch: {path}")
    if stage32.sha256(root / SOURCE_RAW) != mat.get("source_sha256"):
        raise ValueError("Source-only matrix raw-data provenance hash mismatch")
    return {"source_raw_sha256": mat["source_sha256"],
            "prepared_matrix_sha256": stage32.sha256(root / PREPARED),
            "prepared_metadata_sha256": stage32.sha256(root / METADATA),
            "stage05_qc_sha256": stage32.sha256(root / PREP_QC)}


def verify_old_common_is_exact_subset(root: Path, gene_ids: np.ndarray,
                                      x: np.ndarray, meta: pd.DataFrame) -> int:
    """Compare source expression values only; never load any target expression."""
    old = pd.read_csv(root / OLD_SOURCE, sep="\t", compression="gzip")
    old_ids = old["gene_id"].astype(str)
    if len(old) != 18013 or old_ids.duplicated().any():
        raise ValueError("Unexpected historical shared-gene matrix")
    lookup = pd.Index(gene_ids).get_indexer(old_ids)
    if np.any(lookup < 0):
        raise ValueError("Historical genes missing from source-only matrix")
    old_x = old[meta.gsm.tolist()].to_numpy(dtype=np.float64).T
    if not np.allclose(x[:, lookup], old_x, rtol=1e-12, atol=1e-12):
        raise ValueError("Historical common-gene values differ from Stage-05 source matrix")
    return len(old_ids)


def run(root: Path, check_only: bool = False) -> dict:
    stage32 = import_stage32(root)
    protocol = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    prior = json.loads((root / PRIOR).read_text(encoding="utf-8"))
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    stage32.verify_manifest(protocol, baseline, root)
    if prior.get("protocol_sha256") != stage32.sha256(root / PROTOCOL):
        raise ValueError("Locked Stage-31 / Stage-32 protocol differs")
    if prior.get("independent_validation") is not False:
        raise ValueError("Historical Stage-32 provenance status changed")
    provenance = verify_stage05_qc(root, stage32)
    original_meta = pd.read_csv(root / METADATA, dtype=str, keep_default_na=False)
    validate_cohort(original_meta, protocol)
    meta = original_meta.rename(columns={"candidate_group": "state"})
    matrix = pd.read_csv(root / PREPARED, sep="\t", compression="gzip")
    gene_ids, x = validate_matrix(matrix, meta.gsm.astype(str).tolist())
    overlap = verify_old_common_is_exact_subset(root, gene_ids, x, meta)
    stage32.validate_folds(protocol["nested_loco_protocol"]["folds"], meta)
    specs = protocol["candidate_configurations_for_future_implementation"]
    stage32.validate_candidate_specs(specs)
    report = {
        "analysis": "stage34_source_only_gene_universe_sensitivity_nested_loco_exploratory",
        "source": "GSE193258", "n_samples": int(len(meta)),
        "n_source_only_genes": int(len(gene_ids)),
        "old_target_intersected_gene_count": overlap,
        "genes_restored_to_source_only_universe": int(len(gene_ids) - overlap),
        "provenance": provenance,
        "frozen_stage31_sha256": stage32.sha256(root / PROTOCOL),
        "historical_stage32_sha256": stage32.sha256(root / PRIOR),
        "candidate_grid_and_inner_selection_policy_unchanged": True,
        "source_only_gene_universe_not_selected_from_target_study": True,
        "target_expression_or_target_labels_read": False,
        "source_labels_still_provisional": True,
        "historically_inspected_folds_independent_validation": False,
        "comparison_status": "SENSITIVITY_ANALYSIS_NOT_NEW_VALIDATION_OR_MODEL_TUNING",
        "check_only": bool(check_only),
        "limitations": [
            "Prior inspection of all source folds prevents independent validation claims.",
            "Gene-universe repair addresses target-dependent intersection, not treatment-duration confounding.",
            "No proof of biological replicate independence or manual label certification is supplied.",
            "Only four cell lines and three replicates per state; uncertainty remains substantial.",
            "No target study expression, outcome or feature filtering is used for this run.",
            "Do not select a new method based on this post-hoc comparison to Stage 32.",
        ],
    }
    if not check_only:
        folds = []
        for fold in protocol["nested_loco_protocol"]["folds"]:
            result = stage32.evaluate_outer_fold(x, meta, fold, specs)
            folds.append(result)
            print("Completed source-only outer fold:", result["outer_test_cell_line"])
        report["outer_folds"] = folds
        report["families"] = stage32.summarize_families(folds)
        stage32.plot_heatmap(report["families"], root / FIGURE)
        print("Saved exploratory plot:", root / FIGURE)
    out = root / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved report:", out)
    print(f"Source-only: {len(gene_ids)} genes; previous: {overlap}; restored: {len(gene_ids)-overlap}")
    print("Exploratory repair only; no new method, independence or clinical validation claim.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
