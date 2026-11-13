"""Stage 29: reproducible, descriptive cross-study provenance and scale audit.

Not a normalization procedure, label certification, model or independent validation.
The nine PC9 samples from GSE255958 have previously been inspected; any
cross-study comparisons here are post-hoc diagnostics, not model selection.
Requires the stage-08 processed files and stage-09 fixed input hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

INPUT_DIR = Path("data/processed/cross_study")
BASELINE_REPORT = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/dcsl_stage29_provenance_scale_audit.json")
FIGURE = Path("figures/dcsl_stage29_scale_audit.png")
INPUT_FILES = (
    "gse193258_common_log2tpm.tsv.gz",
    "gse255958_common_log2tpm_candidate.tsv.gz",
    "samples.csv",
)
FLOOR = float(np.log2(0.01))
STATES = ("Control", "Acute", "DTP")
KNOWN_STATUS = "PENDING_MANUAL_AND_MATRIX_CHECK"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check_input_hashes(root: Path) -> dict:
    """Fails closed if any input differs from the recorded stage-09 version."""
    previous = json.loads((root / BASELINE_REPORT).read_text(encoding="utf-8"))
    expected = previous["input_sha256"]
    actual = {}
    for filename in INPUT_FILES:
        path = root / INPUT_DIR / filename
        current = digest(path)
        if current != expected.get(filename):
            raise ValueError(f"Stage-09 checksum mismatch for {filename}")
        actual[filename] = current
    return actual


def read_expression(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", compression="gzip", keep_default_na=False)
    if "gene_id" not in frame or frame.columns.duplicated().any():
        raise ValueError("Expression matrix needs gene_id and unique columns")
    genes = frame.pop("gene_id").astype(str).str.strip()
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError("Blank or duplicated gene IDs")
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Expression contains non-finite entries")
    matrix = pd.DataFrame(values, index=genes, columns=frame.columns)
    matrix.index.name = "gene_id"
    return matrix


def check_metadata(meta: pd.DataFrame, source: pd.DataFrame, target: pd.DataFrame) -> dict:
    required = {"accession", "gsm", "cell_line", "state", "treatment", "title", "label_review", "evaluation_role"}
    if not required.issubset(meta.columns) or meta["gsm"].duplicated().any():
        raise ValueError("Incomplete or repeated sample metadata")
    if len(meta) != 48 or len(source.columns) != 36 or len(target.columns) != 12:
        raise ValueError("Unexpected dataset sample counts")
    src = meta.loc[meta.accession == "GSE193258"].copy()
    tgt = meta.loc[meta.accession == "GSE255958"].copy()
    if set(src.gsm) != set(source.columns) or set(tgt.gsm) != set(target.columns):
        raise ValueError("GSM-to-expression mapping does not match matrices")
    if len(src) != 36 or len(tgt) != 12 or set(src.cell_line) != {"H1975", "HCC2935", "HCC827", "PC9"}:
        raise ValueError("Unexpected sample cell lines")
    if (src.groupby(["cell_line", "state"]).size() != 3).any() or len(src.groupby(["cell_line", "state"])) != 12:
        raise ValueError("Unexpected source group/state replicates")
    if tgt.state.value_counts().to_dict() != {"Control": 3, "Acute": 3, "DTP": 3, "Resistant": 3}:
        raise ValueError("Unexpected target group counts")
    if not tgt.cell_line.eq("PC9").all():
        raise ValueError("Target comparison is not PC9 only")
    if not src.evaluation_role.eq("source_candidate").all():
        raise ValueError("Unexpected source roles")
    if not tgt.loc[tgt.state != "Resistant", "evaluation_role"].eq("target_candidate").all():
        raise ValueError("Unexpected target roles")
    if not tgt.loc[tgt.state == "Resistant", "evaluation_role"].eq("exploratory_only").all():
        raise ValueError("Resistant samples must remain excluded")
    statuses = meta.groupby(["accession", "label_review"]).size()
    status_counts = {f"{a}:{s}": int(n) for (a, s), n in statuses.items()}
    return {
        "source": src,
        "target": tgt,
        "label_status_counts": status_counts,
        "every_label_still_pending_review": bool(meta.label_review.eq(KNOWN_STATUS).all()),
    }


def sample_summary(matrix: pd.DataFrame, meta: pd.DataFrame) -> list[dict]:
    annotations = meta.set_index("gsm")
    out = []
    for sample in matrix.columns:
        v = matrix[sample].to_numpy(dtype=float)
        out.append({
            "gsm": sample,
            "accession": str(annotations.loc[sample, "accession"]),
            "cell_line": str(annotations.loc[sample, "cell_line"]),
            "state": str(annotations.loc[sample, "state"]),
            "median": float(np.median(v)),
            "q10": float(np.quantile(v, .1)),
            "q90": float(np.quantile(v, .9)),
            "minimum": float(v.min()),
            "fraction_at_candidate_0_01_floor": float(np.mean(np.isclose(v, FLOOR, atol=1e-8, rtol=0))),
        })
    return out


def compare_state_centroids(source: pd.DataFrame, target: pd.DataFrame,
                            srcmeta: pd.DataFrame, tgtmeta: pd.DataFrame) -> dict:
    """Descriptive comparison of *unpaired* PC9 state centroids, no adaptation."""
    if not source.index.equals(target.index):
        raise ValueError("Common genes differ in order")
    out = {}
    for state in STATES:
        src_ids = srcmeta.loc[(srcmeta.cell_line == "PC9") & (srcmeta.state == state), "gsm"].tolist()
        tgt_ids = tgtmeta.loc[tgtmeta.state == state, "gsm"].tolist()
        if len(src_ids) != 3 or len(tgt_ids) != 3:
            raise ValueError(f"Expected 3 PC9 replicates for {state}")
        a = source[src_ids].to_numpy(dtype=float).mean(axis=1)
        b = target[tgt_ids].to_numpy(dtype=float).mean(axis=1)
        delta = np.abs(a - b)
        corr = spearmanr(a, b).statistic
        out[state] = {
            "source_n": 3, "target_n": 3,
            "median_abs_gene_mean_difference_in_log2_units": float(np.median(delta)),
            "q90_abs_gene_mean_difference_in_log2_units": float(np.quantile(delta, .9)),
            "gene_mean_spearman_rho": float(corr),
            "interpretation": "Unpaired post-hoc contrast; confounds processing, dose, time, batch, and biology",
        }
    return out


def group_floor_stats(samples: list[dict], study: str, state: str) -> dict:
    values = np.array([s["fraction_at_candidate_0_01_floor"] for s in samples
                       if s["accession"] == study and s["state"] == state], dtype=float)
    if not len(values):
        raise ValueError("No sample values in group")
    return {"n": len(values), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def plot_report(summary: dict, path: Path) -> None:
    values = summary["sample_scale_summaries"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    groups = [(study, state) for study in ("GSE193258", "GSE255958") for state in STATES]
    data = [[s["fraction_at_candidate_0_01_floor"] * 100 for s in values
             if s["accession"] == study and s["state"] == state] for study, state in groups]
    axes[0].boxplot(data, tick_labels=[f"{study[-3:]}\n{state}" for study, state in groups],
                    showmeans=True)
    axes[0].set_title("Values at candidate floor, by study/state")
    axes[0].set_ylabel("Fraction of genes at log2(0.01), %")
    axes[0].grid(axis="y", alpha=.2)
    differences = summary["pc9_unpaired_state_comparisons"]
    states = list(STATES)
    axes[1].bar(states, [differences[s]["median_abs_gene_mean_difference_in_log2_units"] for s in states])
    axes[1].set_title("PC9 state centroids across studies")
    axes[1].set_ylabel("Median absolute difference in log2 units")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Stage 29: descriptive QC only — no scale harmonization or label validation", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(root: Path) -> dict:
    input_hashes = check_input_hashes(root)
    source = read_expression(root / INPUT_DIR / INPUT_FILES[0])
    target = read_expression(root / INPUT_DIR / INPUT_FILES[1])
    meta = pd.read_csv(root / INPUT_DIR / INPUT_FILES[2], dtype=str, keep_default_na=False)
    if not source.index.equals(target.index) or len(source.index) != 18013:
        raise ValueError("Gene identity/order mismatch")
    reviewed = check_metadata(meta, source, target)
    all_samples = sample_summary(source, reviewed["source"]) + sample_summary(target, reviewed["target"])
    group_stats = {study: {state: group_floor_stats(all_samples, study, state)
                           for state in STATES} for study in ("GSE193258", "GSE255958")}
    pc9 = compare_state_centroids(source, target, reviewed["source"], reviewed["target"])
    output = {
        "analysis": "stage29_provenance_and_expression_scale_audit",
        "source": "GSE193258", "target": "GSE255958",
        "source_n": 36, "target_n": 12, "target_candidate_n": 9,
        "genes": len(source.index), "input_sha256": input_hashes,
        "all_48_label_review_counts": reviewed["label_status_counts"],
        "all_48_labels_pending_manual_review": reviewed["every_label_still_pending_review"],
        "protocol_documentation_from_GEO": {
            "source": {"url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258",
                       "control": "DMSO 24 h following 21 days drug-free",
                       "acute": "500 nM osimertinib 24 h",
                       "DTP": "500 nM osimertinib 21 days"},
            "target": {"url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255958",
                       "control": "0.1% DMSO 48 h",
                       "acute": "osimertinib 48 h; candidate PC9 metadata list 2 uM",
                       "DTP": "osimertinib >=9 days; candidate PC9 metadata list 2 uM and 9 d"},
        },
        "scale_provenance": {
            "source": "GEO-supplied log2TPM; exact transform/pseudocount not independently established",
            "target": "Stage-08 candidate log2(RSEM gene TPM + 0.01)",
            "candidate_floor_log2_0_01": FLOOR,
            "source_minimum": float(source.to_numpy().min()),
            "target_minimum": float(target.to_numpy().min()),
            "same_floor_does_not_prove_same_scale": True,
        },
        "sample_scale_summaries": all_samples,
        "candidate_floor_fraction_by_study_and_state": group_stats,
        "pc9_unpaired_state_comparisons": pc9,
        "data_usage": {"new_model_fitted": False,
                       "normalization_or_batch_correction_performed": False,
                       "labels_used_for_posthoc_descriptive_comparison_only": True,
                       "target_has_been_inspected_previously": True,
                       "independent_validation": False},
        "unresolved_before_cross_study_claims": [
            "Manually review/record provenance of all sample state labels and biological replicates",
            "Verify exact source expression transformation and gene annotation versions",
            "Reconcile treatment dose, durations, sequencing/quantification protocols and batch",
            "Do not use post-hoc target diagnostics to select a new model or tune preprocessing",
        ],
        "decision": "KEEP_CROSS_STUDY_VALIDATION_PROVISIONAL",
    }
    output_path = root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_report(output, root / FIGURE)
    print("Stage 29: 36 source + 12 target samples; 18,013 shared genes")
    print(f"All 48 label_review entries pending: {reviewed['every_label_still_pending_review']}")
    for study in ("GSE193258", "GSE255958"):
        stats = group_stats[study]
        print(f"{study} candidate-floor median: " + ", ".join(
            f"{s}={100*stats[s]['median']:.1f}%" for s in STATES))
    for state in STATES:
        print(f"PC9 {state}: median abs gene-mean shift={pc9[state]['median_abs_gene_mean_difference_in_log2_units']:.3f}; "
              f"rho={pc9[state]['gene_mean_spearman_rho']:.3f}")
    print(f"Saved report: {output_path}")
    print(f"Saved figure: {root / FIGURE}")
    print("Provisional QC only. No model training, batch correction, or label certification.")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
