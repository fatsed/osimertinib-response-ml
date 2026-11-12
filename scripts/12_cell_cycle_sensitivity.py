from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/processed/cross_study")
BASELINE = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/gse193258_cell_cycle_sensitivity.json")
STATES = ("Control", "Acute", "DTP")
N_FEATURES = 500
N_RANDOM = 5
SEED = 42

# Illustrative subset of Hallmark G2M Checkpoint genes reported by St. Jude.
# This is not the complete 200-gene MSigDB Hallmark G2M gene set.
MARKERS = (
    "UBE2C", "CENPF", "TOP2A", "TPX2", "PTTG1", "PLK1", "CENPA", "CDC20",
    "CCNB2", "KPNA2", "HMMR", "NUSAP1", "CDKN3", "NEK2", "MKI67", "CENPE",
    "BIRC5", "CCNA2", "AURKA", "KIF23", "KIF2C", "PBK", "TTK", "TROAP",
    "CKS1B", "CKS2", "NDC80", "CDK1", "BUB1", "PRC1",
)
MARKER_SOURCE = "https://pscb.stjude.org/pscb/GEP_reports/human/GSE137804/gepSummaryKnitr_8.html"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(root: Path):
    expression_path = root / DATA / "gse193258_common_log2tpm.tsv.gz"
    metadata_path = root / DATA / "samples.csv"
    previous = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    for file in (expression_path, metadata_path):
        if checksum(file) != previous["input_sha256"][file.name]:
            raise ValueError(f"Input differs from stage 09: {file.name}")

    frame = pd.read_csv(expression_path, sep="\t", compression="gzip")
    if "gene_id" not in frame:
        raise ValueError("Missing gene_id column")
    gene_ids = frame.pop("gene_id").astype(str).str.strip().to_numpy()
    if not all(gene_ids) or len(np.unique(gene_ids)) != len(gene_ids):
        raise ValueError("Missing or repeated gene IDs")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate sample columns")
    source = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    source = source.loc[source["accession"] == "GSE193258"].copy()
    if len(source) != 36 or source["gsm"].nunique() != 36:
        raise ValueError("Expected 36 source GSMs")
    if set(frame.columns) != set(source["gsm"]) or len(frame.columns) != 36:
        raise ValueError("Expression columns do not match source sample IDs")
    source = source.set_index("gsm").loc[list(frame.columns)].reset_index()
    if source.groupby(["cell_line", "state"]).size().to_dict() != {
        (line, state): 3
        for line in ("PC9", "H1975", "HCC827", "HCC2935")
        for state in STATES
    }:
        raise ValueError("Unexpected cell-line/state design")
    x = frame.to_numpy(dtype=np.float64).T
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values")
    return x, gene_ids, source, previous


def fit_and_predict(x_train, y_train, x_test, selected_ids):
    model = Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)),
    ])
    if x_train.shape[1] < N_FEATURES:
        raise ValueError("Insufficient features for the fixed 500-gene model")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        model.fit(x_train, y_train)
    pred = model.predict(x_test)
    genes_after_filter = selected_ids[model.named_steps["filter"].get_support()]
    chosen = genes_after_filter[model.named_steps["select"].get_support()]
    return pred, set(chosen.tolist())


def scores(actual, predicted):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(
            actual, predicted, labels=list(STATES), average="macro", zero_division=0
        )),
    }


def mean_jaccard(gene_sets):
    pairs = [len(a & b) / len(a | b) for a, b in combinations(gene_sets, 2)]
    return float(np.mean(pairs))


def sample_variance_matched_indices(x_train, marker_idx, excluded_idx, seed):
    variance = x_train.var(axis=0)
    log_var = np.log1p(variance)
    eligible = np.flatnonzero(~np.isin(np.arange(len(variance)), excluded_idx))
    rng = np.random.default_rng(seed)
    remaining = set(eligible.tolist())
    chosen = []
    for index in rng.permutation(marker_idx):
        target = log_var[index]
        options = np.fromiter(remaining, dtype=int)
        nearest = np.abs(log_var[options] - target)
        threshold = np.partition(nearest, min(29, len(nearest) - 1))[min(29, len(nearest) - 1)]
        pool = options[nearest <= threshold]
        candidate = int(rng.choice(pool))
        chosen.append(candidate)
        remaining.remove(candidate)
    return np.array(chosen, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    x, gene_ids, metadata, previous = load_data(root)
    markers_present = sorted(set(MARKERS) & set(gene_ids))
    if len(markers_present) < 20:
        raise ValueError("Too few predefined marker genes in the expression matrix")
    panel_idx = np.flatnonzero(np.isin(gene_ids, markers_present))
    panel_mask = ~np.isin(np.arange(len(gene_ids)), panel_idx)
    y = metadata["state"].to_numpy()
    cell_lines = metadata["cell_line"].to_numpy()

    fold_results = []
    base_true, base_pred, removed_pred = [], [], []
    random_preds = [[] for _ in range(N_RANDOM)]
    base_gene_sets = []
    removed_gene_sets = []
    previous_folds = {
        entry["held_out_cell_line"]: entry["predictions"]
        for entry in previous["models"]["LogisticRegression"]["loco_folds"]
    }

    for fold_number, line in enumerate(sorted(set(cell_lines))):
        testing = cell_lines == line
        training = ~testing
        expected = previous_folds[line]
        if len(expected) != int(testing.sum()):
            raise ValueError("Stage 09 fold size differs")

        base, selected = fit_and_predict(
            x[training], y[training], x[testing], gene_ids
        )
        reproduced = [
            {"gsm": str(gsm), "true_state": str(actual), "predicted_state": str(guess)}
            for gsm, actual, guess in zip(metadata.loc[testing, "gsm"], y[testing], base)
        ]
        if reproduced != expected:
            raise ValueError(f"Baseline predictions differ in {line}")

        removed, selected_without = fit_and_predict(
            x[training][:, panel_mask], y[training],
            x[testing][:, panel_mask], gene_ids[panel_mask]
        )
        random_fold_predictions = []
        for iteration in range(N_RANDOM):
            control_idx = sample_variance_matched_indices(
                x[training], panel_idx, panel_idx, SEED + fold_number * 100 + iteration
            )
            keep = np.ones(len(gene_ids), dtype=bool)
            keep[control_idx] = False
            prediction, _ = fit_and_predict(
                x[training][:, keep], y[training],
                x[testing][:, keep], gene_ids[keep]
            )
            random_fold_predictions.append(prediction.tolist())
            random_preds[iteration].extend(prediction.tolist())

        fold_results.append({
            "held_out_cell_line": line,
            "baseline": scores(y[testing], base),
            "without_marker_panel": scores(y[testing], removed),
            "selected_panel_markers_in_baseline": sorted(set(markers_present) & selected),
            "predictions": [
                {"gsm": str(gsm), "true_state": str(actual),
                 "baseline": str(before), "without_panel": str(after)}
                for gsm, actual, before, after in zip(
                    metadata.loc[testing, "gsm"], y[testing], base, removed
                )
            ],
        })
        base_gene_sets.append(selected)
        removed_gene_sets.append(selected_without)
        base_true.extend(y[testing].tolist())
        base_pred.extend(base.tolist())
        removed_pred.extend(removed.tolist())
        print(f"{line}: {len(set(markers_present) & selected)} panel genes in baseline's top 500")

    controls = [scores(base_true, pred) for pred in random_preds]
    result = {
        "analysis": "exploratory_cell_cycle_marker_ablation",
        "source_dataset": "GSE193258",
        "samples": len(metadata),
        "genes": len(gene_ids),
        "marker_panel": {
            "description": "Fixed 30-gene illustrative subset, not the full Hallmark G2M signature",
            "source_url": MARKER_SOURCE,
            "requested": len(MARKERS),
            "present": len(markers_present),
            "genes": markers_present,
        },
        "evaluation": "leave-one-cell-line-out, source only",
        "baseline_predictions_reproduced": True,
        "baseline_500_genes": scores(base_true, base_pred),
        "without_marker_panel": scores(base_true, removed_pred),
        "variance_matched_random_removals": {
            "iterations": N_RANDOM,
            "removed_genes_per_fold": len(panel_idx),
            "scores": controls,
        },
        "folds": fold_results,
        "selected_gene_stability_mean_jaccard": {
            "baseline": mean_jaccard(base_gene_sets),
            "without_marker_panel": mean_jaccard(removed_gene_sets),
        },
        "limitations": [
            "Only a limited marker panel is tested, not all cell-cycle-associated genes.",
            "Removing marker genes cannot remove all cell-cycle information or establish causality.",
            "Random controls are matched approximately by training-fold gene variance.",
            "The same LOCO samples were inspected in stages 09-11; these are post-hoc diagnostics.",
            "No GSE255958 samples are used; no external generalization claim is made here.",
            "Biological replicates and labels still require manual scientific confirmation.",
        ],
    }
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Baseline LOCO BA:", f"{result['baseline_500_genes']['balanced_accuracy']:.3f}")
    print("Without marker panel LOCO BA:", f"{result['without_marker_panel']['balanced_accuracy']:.3f}")
    print("Random matched removal BAs:", [
        round(item["balanced_accuracy"], 3) for item in controls
    ])
    print("Saved:", output)
    print("Exploratory marker sensitivity only, not causal proof of cell-cycle confounding")


if __name__ == "__main__":
    main()
