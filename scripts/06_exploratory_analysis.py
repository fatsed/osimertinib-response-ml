from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "GSE193258"
RESULT = ROOT / "results" / "eda_gse193258_summary.json"
FIGURES = ROOT / "figures"
FEATURE_LIMIT = 2000


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(INPUT / "samples_36.csv", dtype=str, keep_default_na=False)
    expression = pd.read_csv(INPUT / "log2tpm_36samples.tsv.gz", sep="\t", compression="gzip")

    columns = {"gsm", "cell_line", "candidate_group"}
    if not columns.issubset(samples.columns) or "gene_id" not in expression.columns:
        raise ValueError("Expected sample metadata or gene_id column is missing")
    if len(samples) != 36 or samples["gsm"].duplicated().any():
        raise ValueError("Expected 36 unique sample IDs")
    if samples["cell_line"].nunique() != 4 or set(samples["candidate_group"]) != {"Control", "Acute", "DTP"}:
        raise ValueError("Unexpected cell lines or response groups")
    group_sizes = samples.groupby(["cell_line", "candidate_group"]).size()
    if len(group_sizes) != 12 or not group_sizes.eq(3).all():
        raise ValueError("Expected three samples in every cell-line and response-state group")
    if expression["gene_id"].isna().any() or expression["gene_id"].duplicated().any():
        raise ValueError("Missing or repeated gene IDs")

    sample_ids = samples["gsm"].tolist()
    if len(expression.columns) != 37 or set(expression.columns[1:]) != set(sample_ids):
        raise ValueError("Expression columns do not match the selected GSM IDs")
    expression = expression.set_index("gene_id").loc[:, sample_ids]
    expression = expression.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise ValueError("Missing or non-finite expression values")
    return samples, expression


def pair_mean(correlation: np.ndarray, labels: list[str], same: bool) -> float:
    values = [
        correlation[i, j]
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
        if (labels[i] == labels[j]) == same
    ]
    return float(np.mean(values)) if values else float("nan")


def save_pca_plot(coordinates: np.ndarray, labels: pd.Series, title: str, filename: str, percentages: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for group in sorted(labels.unique()):
        selected = labels.eq(group).to_numpy()
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=55, label=group, alpha=0.85)
    ax.set_xlabel(f"PC1 ({percentages[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({percentages[1]:.1f}%)")
    ax.set_title(title)
    ax.legend(title="Group", fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180)
    plt.close(fig)


def main() -> None:
    samples, expression = load_data()
    values = expression.to_numpy(dtype=float)
    variance = values.var(axis=1)
    informative = np.flatnonzero(variance > 0)
    if informative.size < 2:
        raise ValueError("Too few variable genes for PCA")

    selected = informative[np.argsort(variance[informative])[-FEATURE_LIMIT:]]
    x = values[selected, :].T
    if np.any(x.std(axis=1) == 0):
        raise ValueError("A sample has zero variance over selected genes")

    pca = PCA(n_components=2, svd_solver="full")
    coordinates = pca.fit_transform(x)
    percentages = pca.explained_variance_ratio_ * 100
    correlations = np.corrcoef(x)

    report = {
        "accession": "GSE193258",
        "analysis": "exploratory_only_not_external_validation",
        "input_matrix": "log2tpm_36samples.tsv.gz",
        "n_samples": int(len(samples)),
        "n_genes_total": int(len(expression)),
        "n_genes_variable": int(len(informative)),
        "n_genes_in_pca": int(len(selected)),
        "pca_pc1_explained_percent": float(percentages[0]),
        "pca_pc2_explained_percent": float(percentages[1]),
        "mean_correlation_same_cell_line": pair_mean(correlations, samples["cell_line"].tolist(), True),
        "mean_correlation_different_cell_line": pair_mean(correlations, samples["cell_line"].tolist(), False),
        "mean_correlation_same_state": pair_mean(correlations, samples["candidate_group"].tolist(), True),
        "mean_correlation_different_state": pair_mean(correlations, samples["candidate_group"].tolist(), False),
        "note": "All samples are used for descriptive EDA. Do not reuse this PCA or feature selection in held-out model evaluation. Sample labels still require scientific review.",
    }

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_pca_plot(coordinates, samples["cell_line"], "GSE193258: PCA by cell line", "pca_gse193258_cell_line.png", percentages)
    save_pca_plot(coordinates, samples["candidate_group"], "GSE193258: PCA by response state", "pca_gse193258_state.png", percentages)

    print(f"Samples: {len(samples)} | Genes: {len(expression)} | PCA genes: {len(selected)}")
    print(f"PC1: {percentages[0]:.2f}% | PC2: {percentages[1]:.2f}%")
    print(f"Mean correlation, same/different cell line: {report['mean_correlation_same_cell_line']:.3f} / {report['mean_correlation_different_cell_line']:.3f}")
    print(f"Mean correlation, same/different state: {report['mean_correlation_same_state']:.3f} / {report['mean_correlation_different_state']:.3f}")
    print(f"Saved: {RESULT.relative_to(ROOT)} and two PCA figures")


if __name__ == "__main__":
    main()
