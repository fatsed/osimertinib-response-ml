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

ROOT = Path(__file__).resolve().parents[1]
COUNTS = Path("data/processed/GSE254485/counts_timecourse_candidate.tsv.gz")
SAMPLES = Path("data/processed/GSE254485/samples_timecourse_candidate.csv")
PREP_REPORT = Path("results/gse254485_timecourse_validation.json")
EDA_REPORT = Path("results/gse254485_timecourse_eda.json")
OUTPUT = Path("results/gse254485_timecourse_trends.json")
FIGURE = Path("figures/gse254485_timecourse_distances.png")
DAYS = (0, 28, 35, 56)
CONTRASTS = ((0, 28), (28, 35), (35, 56), (28, 56))
MIN_CPM = 1.0
MIN_SAMPLES = 2
TOP_GENES = 10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(root: Path):
    prep = json.loads((root / PREP_REPORT).read_text(encoding="utf-8"))
    eda = json.loads((root / EDA_REPORT).read_text(encoding="utf-8"))
    for path, key in ((COUNTS, "processed_counts_sha256"),
                      (SAMPLES, "candidate_metadata_sha256")):
        full = root / path
        if not full.is_file() or sha256(full) != prep.get(key):
            raise ValueError(f"Stage-16 input is missing or changed: {path}")
        if sha256(full) != eda.get("input_sha256", {}).get(path.as_posix()):
            raise ValueError(f"Stage-17 input hash differs: {path}")

    samples = pd.read_csv(root / SAMPLES, dtype=str, keep_default_na=False)
    needed = {"gsm", "day", "replicate_label", "cell_line", "treatment",
              "total_assigned_counts", "genes_with_nonzero_counts", "scientifically_verified"}
    if needed - set(samples.columns):
        raise ValueError(f"Missing metadata columns: {sorted(needed - set(samples.columns))}")
    if len(samples) != 8 or samples["gsm"].duplicated().any():
        raise ValueError("Expected 8 distinct GSM samples")
    samples["day"] = pd.to_numeric(samples["day"], errors="raise").astype(int)
    samples["replicate_label"] = pd.to_numeric(samples["replicate_label"], errors="raise").astype(int)
    samples = samples.sort_values(["day", "replicate_label"]).reset_index(drop=True)
    expected_pairs = {(day, rep) for day in DAYS for rep in (1, 2)}
    actual_pairs = set(zip(samples["day"], samples["replicate_label"]))
    if actual_pairs != expected_pairs or not samples["cell_line"].eq("PC9").all():
        raise ValueError("Unexpected days, replicate labels or cell line")
    if not samples.loc[samples["day"] == 0, "treatment"].eq("No treatment").all():
        raise ValueError("Day 0 must be untreated")
    if not samples.loc[samples["day"] != 0, "treatment"].eq("Osimertinib").all():
        raise ValueError("Later samples must be osimertinib-treated")

    table = pd.read_csv(root / COUNTS, sep="\t", compression="gzip", dtype={"gene_id": str})
    if "gene_id" not in table or table.columns.duplicated().any():
        raise ValueError("Missing gene_id or duplicate matrix columns")
    gene_ids = table.pop("gene_id")
    if gene_ids.isna().any() or gene_ids.str.strip().eq("").any() or gene_ids.duplicated().any():
        raise ValueError("Invalid or duplicate gene identifiers")
    if set(table.columns) != set(samples["gsm"]):
        raise ValueError("Matrix columns do not match the sample metadata")
    values = table[samples["gsm"].tolist()].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Expected finite, nonnegative integer counts")
    if values.max() > np.iinfo(np.int64).max:
        raise ValueError("Count overflow")
    counts = values.astype(np.int64)
    libraries = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)
    if (libraries <= 0).any():
        raise ValueError("All sample library sizes must be positive")
    if not np.array_equal(libraries, samples["total_assigned_counts"].astype(np.int64)):
        raise ValueError("Library sizes differ from stage 16")
    if not np.array_equal(detected, samples["genes_with_nonzero_counts"].astype(np.int64)):
        raise ValueError("Detected gene counts differ from stage 16")
    if len(gene_ids) != prep.get("genes") or len(samples) != prep.get("samples"):
        raise ValueError("Matrix dimensions differ from stage 16")
    if eda.get("normalization") != "library-size CPM followed by log2(CPM + 1); exploratory, not TMM/DESeq2":
        raise ValueError("Stage-17 normalization differs; check before comparing")
    return counts, gene_ids.to_numpy(), samples, libraries, prep, eda


def summarize(counts: np.ndarray, gene_ids: np.ndarray,
              samples: pd.DataFrame, libraries: np.ndarray, eda: dict) -> dict:
    cpm = counts.astype(float) / libraries[None, :] * 1e6
    keep = (cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES
    if int(keep.sum()) != eda.get("n_genes_passing_cpm_filter"):
        raise ValueError("Expression filter does not reproduce stage 17")
    genes = gene_ids[keep]
    expression = np.log2(cpm[keep] + 1.0)
    day_indices = {day: np.flatnonzero(samples["day"].to_numpy() == day) for day in DAYS}
    group_mean = {day: expression[:, idx].mean(axis=1) for day, idx in day_indices.items()}

    within = {}
    all_within = []
    for day, idx in day_indices.items():
        difference = np.abs(expression[:, idx[0]] - expression[:, idx[1]])
        corr = float(np.corrcoef(expression[:, idx[0]], expression[:, idx[1]])[0, 1])
        if not np.isclose(corr, eda["within_day_logcpm_pearson"][str(day)], atol=1e-10, rtol=0):
            raise ValueError(f"Day-{day} correlation differs from stage 17")
        within[str(day)] = {
            "median_abs_replicate_difference_log2cpm": float(np.median(difference)),
            "rms_replicate_difference_log2cpm": float(np.sqrt(np.mean(difference ** 2))),
            "pearson": corr,
        }
        all_within.append(float(np.median(difference)))

    contrasts = []
    for start, end in CONTRASTS:
        delta = group_mean[end] - group_mean[start]
        a = expression[:, day_indices[start]]
        b = expression[:, day_indices[end]]
        nonoverlap = (a.max(axis=1) < b.min(axis=1)) | (b.max(axis=1) < a.min(axis=1))
        strong = nonoverlap & (np.abs(delta) >= 1.0)
        indices = np.flatnonzero(strong)
        ranking = indices[np.argsort(-np.abs(delta[indices]), kind="stable")[:TOP_GENES]]
        top = [{
            "gene_id": str(genes[i]),
            "mean_difference_log2cpm": float(delta[i]),
            "mean_log2cpm_start": float(group_mean[start][i]),
            "mean_log2cpm_end": float(group_mean[end][i]),
            "direction": "higher_at_end" if delta[i] > 0 else "lower_at_end",
        } for i in ranking]
        contrasts.append({
            "start_day": start,
            "end_day": end,
            "comparison_type": "untreated_vs_treated" if start == 0 else "within_treated_timepoints",
            "median_abs_mean_difference_log2cpm": float(np.median(np.abs(delta))),
            "rms_mean_difference_log2cpm": float(np.sqrt(np.mean(delta ** 2))),
            "genes_with_abs_mean_difference_at_least_1": int((np.abs(delta) >= 1.0).sum()),
            "genes_with_nonoverlapping_replicate_ranges": int(nonoverlap.sum()),
            "exploratory_candidates_with_both_criteria": int(strong.sum()),
            "top_10_by_abs_mean_difference_among_candidates": top,
        })

    return {
        "analysis": "GSE254485_exploratory_timecourse_expression_shifts",
        "n_samples": len(samples),
        "n_genes_filtered": len(genes),
        "cpm_filter": {"minimum_cpm": MIN_CPM, "in_at_least_n_samples": MIN_SAMPLES},
        "expression_scale": "log2(CPM + 1) from library-size-normalized featureCounts; no composition correction",
        "sample_days_and_treatments": {str(day): {
            "samples": samples.loc[samples.day == day, "gsm"].tolist(),
            "treatment": "No treatment" if day == 0 else "Osimertinib",
        } for day in DAYS},
        "within_day_replicate_differences": within,
        "median_within_day_replicate_difference_reference": float(np.median(all_within)),
        "contrasts": contrasts,
        "labels_verified": False,
        "supervised_model_trained": False,
        "inferential_statistics_performed": False,
        "limitations": [
            "Sample aliases and biological replicate independence have not been manually verified.",
            "Only two candidate samples per day; selected genes are exploratory, not statistically significant DEGs.",
            "Day 0 is untreated but later days are treated; treatment and elapsed time are confounded.",
            "Comparisons are unpaired; replicate numbers across days are not confirmed matched longitudinal samples.",
            "CPM may retain RNA-composition biases; differences in mean log2(CPM+1) are not formal log2 fold changes.",
            "Selection by effect size and non-overlapping two-replicate ranges is descriptive, not a hypothesis test.",
            "No DTP or resistance phenotypes are inferred from these time points.",
        ],
    }


def plot_shift(report: dict, output: Path) -> None:
    rows = report["contrasts"]
    labels = [f"Day {r['start_day']} to {r['end_day']}" for r in rows]
    values = [r["median_abs_mean_difference_log2cpm"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(labels, values)
    reference = report["median_within_day_replicate_difference_reference"]
    ax.axhline(reference, linestyle="--", linewidth=1.3,
               label="Median within-day replicate difference (descriptive)")
    ax.set_title("GSE254485: descriptive expression shifts")
    ax.set_ylabel("Median |difference of group-mean log2(CPM + 1)|")
    ax.set_xlabel("Comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    counts, genes, samples, libraries, prep, eda = load_inputs(root)
    report = summarize(counts, genes, samples, libraries, eda)
    report["input_sha256"] = {p.as_posix(): sha256(root / p) for p in (COUNTS, SAMPLES)}
    report["stage_16_source_sha256"] = prep.get("source_sha256")
    report["figure"] = FIGURE.as_posix()

    image_path = root / FIGURE
    image_path.parent.mkdir(parents=True, exist_ok=True)
    plot_shift(report, image_path)
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Samples: {report['n_samples']} | filtered genes: {report['n_genes_filtered']}")
    for row in report["contrasts"]:
        print(f"Day {row['start_day']} -> {row['end_day']}: "
              f"median abs difference={row['median_abs_mean_difference_log2cpm']:.3f}; "
              f"exploratory candidates={row['exploratory_candidates_with_both_criteria']}")
    print(f"Saved: {output}")
    print(f"Saved: {image_path}")
    print("Descriptive comparisons only; no differential-expression test or resistance labeling.")


if __name__ == "__main__":
    main()
