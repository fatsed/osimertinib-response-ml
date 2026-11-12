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
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
ACCESSION = "GSE254485"
COUNTS = Path("data/processed/GSE254485/counts_timecourse_candidate.tsv.gz")
SAMPLES = Path("data/processed/GSE254485/samples_timecourse_candidate.csv")
PREPARATION = Path("results/gse254485_timecourse_validation.json")
REPORT = Path("results/gse254485_timecourse_eda.json")
FIGURES = Path("figures")
EXPECTED_DAYS = (0, 28, 35, 56)
MIN_CPM = 1.0
MIN_SAMPLES_EXPRESSED = 2
PCA_TOP_GENES = 2000


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_data(root: Path):
    report_path = root / PREPARATION
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    paths = [root / COUNTS, root / SAMPLES]
    keys = ["processed_counts_sha256", "candidate_metadata_sha256"]
    for path, key in zip(paths, keys):
        if not path.is_file() or digest(path) != prior.get(key):
            raise ValueError(f"Missing or changed stage-16 input: {path}")

    sample = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    required = {
        "gsm", "cell_line", "day", "replicate_label", "treatment",
        "total_assigned_counts", "genes_with_nonzero_counts", "scientifically_verified"
    }
    if not required.issubset(sample.columns):
        raise ValueError(f"Missing sample metadata columns: {sorted(required - set(sample.columns))}")
    if len(sample) != 8 or sample["gsm"].duplicated().any():
        raise ValueError("Expected eight unique GSM records")
    sample["day"] = pd.to_numeric(sample["day"], errors="raise").astype(int)
    sample["replicate_label"] = pd.to_numeric(sample["replicate_label"], errors="raise").astype(int)
    if sample.groupby("day").size().to_dict() != {day: 2 for day in EXPECTED_DAYS}:
        raise ValueError("Expected two samples at each of days 0, 28, 35, and 56")
    if sample.groupby(["day", "replicate_label"]).size().ne(1).any():
        raise ValueError("Replicate labels are duplicated for a time point")
    if sorted(sample["replicate_label"].unique().tolist()) != [1, 2]:
        raise ValueError("Expected replicate labels 1 and 2")
    if not sample["cell_line"].eq("PC9").all():
        raise ValueError("Unexpected cell line")
    if not sample.loc[sample["day"] == 0, "treatment"].eq("No treatment").all():
        raise ValueError("Day 0 treatment metadata changed")
    if not sample.loc[sample["day"] != 0, "treatment"].eq("Osimertinib").all():
        raise ValueError("Treatment metadata changed")
    sample = sample.sort_values(["day", "replicate_label"]).reset_index(drop=True)

    matrix = pd.read_csv(paths[0], sep="\t", compression="gzip", dtype={"gene_id": str})
    if "gene_id" not in matrix or matrix.columns.duplicated().any():
        raise ValueError("Invalid count-matrix columns")
    ids = matrix.pop("gene_id")
    if ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Invalid or repeated gene IDs")
    gsm = sample["gsm"].tolist()
    if set(matrix.columns) != set(gsm):
        raise ValueError("Count matrix samples do not match metadata")
    values = matrix[gsm].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Counts must be finite nonnegative integers")
    if (values > np.iinfo(np.int64).max).any():
        raise ValueError("Count overflow")
    counts = values.astype(np.int64)
    libraries = counts.sum(axis=0)
    detected = (counts > 0).sum(axis=0)
    if (libraries <= 0).any():
        raise ValueError("Some samples have no mapped counts")
    if not np.array_equal(libraries, sample["total_assigned_counts"].astype(np.int64)):
        raise ValueError("Library sizes differ from stage 16")
    if not np.array_equal(detected, sample["genes_with_nonzero_counts"].astype(np.int64)):
        raise ValueError("Detected-gene counts differ from stage 16")
    if len(ids) != prior["genes"] or len(gsm) != prior["samples"]:
        raise ValueError("Matrix dimensions differ from stage 16")
    return counts, ids.to_numpy(), sample, prior, libraries, detected


def calculate(counts: np.ndarray, genes: np.ndarray, libraries: np.ndarray):
    cpm = counts.astype(np.float64) / libraries[None, :] * 1e6
    keep = (cpm >= MIN_CPM).sum(axis=1) >= MIN_SAMPLES_EXPRESSED
    if keep.sum() < 3:
        raise ValueError("Too few expressed genes for PCA")
    log_cpm = np.log2(cpm[keep] + 1.0)
    variation = log_cpm.var(axis=1, ddof=1)
    if not np.isfinite(variation).all() or variation.max() <= 1e-12:
        raise ValueError("No reproducible between-sample variation for PCA")
    top_k = min(PCA_TOP_GENES, len(variation))
    chosen = np.argsort(-variation, kind="stable")[:top_k]
    x = log_cpm[chosen].T
    pca = PCA(n_components=2, svd_solver="full")
    points = pca.fit_transform(x)
    if not np.isfinite(points).all():
        raise ValueError("Invalid PCA coordinates")
    if any(log_cpm[:, index].std() <= 1e-10 for index in range(log_cpm.shape[1])):
        raise ValueError("A sample has insufficient variance for correlation")
    corr = np.corrcoef(log_cpm.T)
    return keep, genes[keep][chosen], points, pca.explained_variance_ratio_, corr


def plot_library_sizes(sample: pd.DataFrame, libraries: np.ndarray, path: Path):
    names = [f"D{int(d)}-R{int(r)}" for d, r in zip(sample["day"], sample["replicate_label"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(names, libraries / 1e6)
    ax.set_ylabel("Assigned gene counts (millions)")
    ax.set_xlabel("Time point and replicate label")
    ax.set_title("GSE254485: library-size QC")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_pca(sample: pd.DataFrame, points: np.ndarray, explained: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for day in EXPECTED_DAYS:
        idx = np.flatnonzero(sample["day"].to_numpy() == day)
        ax.scatter(points[idx, 0], points[idx, 1], s=85, label=f"Day {day}")
        for j in idx:
            ax.annotate(f"R{int(sample.loc[j, 'replicate_label'])}",
                        (points[j, 0], points[j, 1]), xytext=(5, 6),
                        textcoords="offset points", fontsize=9)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title("GSE254485: PCA of variable log2(CPM + 1) genes")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    counts, genes, sample, prior, libraries, detected = load_data(root)
    keep, top_genes, points, explained, corr = calculate(counts, genes, libraries)

    within_day = {}
    for day in EXPECTED_DAYS:
        indices = sample.index[sample["day"] == day].tolist()
        within_day[str(day)] = float(corr[indices[0], indices[1]])

    details = []
    for i, row in sample.iterrows():
        details.append({
            "gsm": row["gsm"],
            "day": int(row["day"]),
            "replicate_label": int(row["replicate_label"]),
            "treatment": row["treatment"],
            "library_size": int(libraries[i]),
            "detected_genes": int(detected[i]),
            "pca_pc1": float(points[i, 0]),
            "pca_pc2": float(points[i, 1]),
        })

    report = {
        "analysis": "exploratory_GSE254485_timecourse_qc_and_pca",
        "source_study": ACCESSION,
        "input_sha256": {
            COUNTS.as_posix(): digest(root / COUNTS),
            SAMPLES.as_posix(): digest(root / SAMPLES),
        },
        "n_samples": len(sample),
        "n_genes_raw": len(genes),
        "n_genes_passing_cpm_filter": int(keep.sum()),
        "filter": {"minimum_cpm": MIN_CPM, "in_at_least_n_samples": MIN_SAMPLES_EXPRESSED},
        "normalization": "library-size CPM followed by log2(CPM + 1); exploratory, not TMM/DESeq2",
        "pca_top_variable_genes": len(top_genes),
        "pca_explained_variance_ratio": [float(v) for v in explained],
        "pca_input": "top-variance filtered logCPM genes, centered by PCA; no class supervision",
        "library_size_max_min_ratio": float(libraries.max() / libraries.min()),
        "within_day_logcpm_pearson": within_day,
        "samples": details,
        "plots": [
            "figures/qc_gse254485_library_sizes.png",
            "figures/pca_gse254485_timecourse.png",
        ],
        "labels_verified": bool(prior.get("scientific_label_review_complete", False)),
        "supervised_model_trained": False,
        "limitations": [
            "Expression-column aliases and biological-replicate independence require manual verification.",
            "Two samples per time point are insufficient for strong inferential claims.",
            "No differential expression tests, response-state labels, or resistance predictions are performed.",
            "Day 0 is untreated while later time points are treated, so treatment and time are confounded.",
            "PCA is descriptive and sensitive to gene selection, normalization, and technical effects.",
            "CPM does not correct RNA composition bias; counts must not be compared directly with log2TPM from another study.",
        ],
    }

    figure_dir = root / FIGURES
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_library_sizes(sample, libraries, figure_dir / "qc_gse254485_library_sizes.png")
    plot_pca(sample, points, explained, figure_dir / "pca_gse254485_timecourse.png")
    output = root / REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Samples: {len(sample)} | genes: {len(genes)} | passed CPM filter: {int(keep.sum())}")
    print(f"Library-size ratio (max/min): {report['library_size_max_min_ratio']:.3f}")
    print(f"PCA variance: PC1={explained[0] * 100:.1f}%, PC2={explained[1] * 100:.1f}%")
    for day, correlation in within_day.items():
        print(f"Day {day} replicate correlation (logCPM): {correlation:.4f}")
    print(f"Saved: {output}")
    print("Saved: figures/qc_gse254485_library_sizes.png")
    print("Saved: figures/pca_gse254485_timecourse.png")
    print("Exploratory analysis only; biological sample verification remains pending.")


if __name__ == "__main__":
    main()
