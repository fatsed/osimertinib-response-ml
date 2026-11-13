"""Stage 26: training-only feature-selection stability audit for DCSL.

Reproduces the stage-23 ANOVA/stable feature selections in the four LOCO
training folds, checks their stored hashes, and quantifies selection overlap.
No model/hyperparameter is changed and no held-out labels are used to select
features. This is descriptive, NOT an independent validation of biomarkers.

Usage: python scripts/26_feature_stability_audit.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import (  # noqa: E402
    cell_line_contrasts, stable_feature_scores,
)

OUTPUT = Path("results/dcsl_stage26_feature_stability.json")
FIGURE = Path("figures/dcsl_stage26_feature_overlap.png")
STAGE23 = Path("results/dcsl_stage23_exploratory.json")
GENE_MATRIX = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
SELECTORS = ("anova", "stable")
STAGE23_MODELS = {
    "anova": ("ANOVA_CE", "ANOVA_Stability"),
    "stable": ("Stable_CE", "DCSL"),
}


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def overlap(a: set[int], b: set[int]) -> dict:
    common = len(a & b)
    union = len(a | b)
    return {
        "intersection": common,
        "union": union,
        "jaccard": float(common / union) if union else 1.0,
        "selection_fraction_shared": float(common / len(a)) if a else 1.0,
    }


def frequency_summary(sets: dict[str, set[int]], gene_ids: np.ndarray) -> dict:
    if not sets:
        raise ValueError("Expected at least one selection")
    counts = Counter(i for selected in sets.values() for i in selected)
    fold_count = len(sets)
    return {
        "union_genes": len(counts),
        "all_fold_intersection_genes": sum(v == fold_count for v in counts.values()),
        "frequency_distribution": {str(i): sum(v == i for v in counts.values()) for i in range(1, fold_count + 1)},
        "all_fold_gene_ids": sorted(str(gene_ids[i]) for i, n in counts.items() if n == fold_count),
    }


def rank_features(x: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                  selector: str, n_features: int, tau: float) -> tuple[np.ndarray, float]:
    """Replicate stage-23's ranking and scaling exactly, on TRAINING data only."""
    if selector not in SELECTORS:
        raise ValueError("Unknown feature selector")
    if len(x) != len(labels) or len(x) != len(groups):
        raise ValueError("Mismatched training data")
    if n_features < 1 or tau <= 0 or not np.isfinite(x).all():
        raise ValueError("Invalid feature-selection inputs")
    valid = np.var(x, axis=0) > 1e-8
    if int(valid.sum()) < n_features:
        raise ValueError("Insufficient nonconstant training features")
    z = StandardScaler().fit_transform(x[:, valid])
    contrasts = cell_line_contrasts(z, labels, groups)
    if selector == "stable":
        scores = stable_feature_scores(contrasts, tau=tau)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores, _ = f_classif(z, labels)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
    ranked = np.lexsort((np.arange(scores.size), -scores))
    selected_local = np.sort(ranked[:n_features])
    indices = np.flatnonzero(valid)[selected_local]
    selected_contrasts = contrasts[:, selected_local]
    all_same_sign = np.all(selected_contrasts > 0, axis=0) | np.all(selected_contrasts < 0, axis=0)
    return indices, float(np.mean(all_same_sign))


def load_stage23(root: Path):
    p = root / "scripts/23_run_dcsl_experiments.py"
    spec = importlib.util.spec_from_file_location("stage23_selection_audit", p)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Stage 23 script missing: {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_pairwise(sets: dict[str, set[int]]) -> list[dict]:
    return [
        {"fold_a": a, "fold_b": b, **overlap(sets[a], sets[b])}
        for a, b in combinations(sorted(sets), 2)
    ]


def save_figure(data: dict, output: Path) -> None:
    pairs = data["anova"]["pairwise_jaccard"]
    labels = [f"{r['fold_a']} / {r['fold_b']}" for r in pairs]
    anova = [r["jaccard"] for r in pairs]
    stable = [r["jaccard"] for r in data["stable"]["pairwise_jaccard"]]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.bar(x - width / 2, anova, width, label="ANOVA")
    ax.bar(x + width / 2, stable, width, label="Stable")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Jaccard overlap of selected genes")
    ax.set_ylim(0, 1)
    ax.set_title("GSE193258: LOCO fold feature-selection overlap (500 genes)")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(root: Path) -> dict:
    stage23 = load_stage23(root)
    _, x, info, _, _ = stage23.load_stage09(root)  # verifies three input SHA256 digests
    prior = json.loads((root / STAGE23).read_text(encoding="utf-8"))
    config = prior["fixed_parameters_no_heldout_tuning"]
    if prior.get("source") != "GSE193258" or prior.get("train_samples") != 36:
        raise ValueError("Unexpected stage 23 cohort")
    if int(config["features"]) != stage23.K or float(config["tau"]) != stage23.TAU:
        raise ValueError("Stage 23 selection parameters changed")
    for filename in stage23.FILES:
        if prior["input_sha256"].get(filename) != stage23.checksum(root / stage23.INPUT / filename):
            raise ValueError(f"Stage 23 checksum mismatch: {filename}")

    gene_ids = pd.read_csv(root / GENE_MATRIX, sep="\t", compression="gzip", usecols=["gene_id"],
                           dtype=str, keep_default_na=False)["gene_id"].to_numpy()
    if len(gene_ids) != x.shape[1] or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Gene order/identifiers inconsistent with model inputs")
    y = info["state"].to_numpy()
    groups = info["cell_line"].to_numpy()
    folds = sorted(set(groups))
    if len(folds) != 4:
        raise ValueError("Expected four source cell lines")
    report = {
        "analysis": "stage26_training_only_feature_stability_audit",
        "source": "GSE193258",
        "source_samples": len(y),
        "source_genes": len(gene_ids),
        "features_per_fold": int(config["features"]),
        "tau": float(config["tau"]),
        "input_sha256": prior["input_sha256"],
        "no_new_model_training_or_parameter_search": True,
        "heldout_fold_labels_unused_for_feature_selection": True,
        "stage23_feature_hashes_reproduced": True,
        "not_independent_validation": True,
        "interpretation_limitations": [
            "Feature overlaps use highly overlapping LOCO training sets and are descriptive only.",
            "All selected feature signatures are reproduced from the original stage-23 settings.",
            "Selection frequency is not biomarker importance, causal evidence, or proof of robustness.",
            "The analysis is exploratory and cannot be used to tune models against previously inspected PC9.",
        ],
        "selectors": {},
        "same_fold_between_selector_jaccard": [],
    }
    selections: dict[str, dict[str, set[int]]] = {}
    for selector in SELECTORS:
        fold_sets: dict[str, set[int]] = {}
        fold_reports = []
        model_names = STAGE23_MODELS[selector]
        for held_out in folds:
            tr = groups != held_out
            indices, sign_agreement = rank_features(
                x[tr], y[tr], groups[tr], selector, int(config["features"]), float(config["tau"])
            )
            digest = sha256_indices(indices)
            for model_name in model_names:
                match = next((f for f in prior["models"][model_name]["loco_folds"]
                              if f["held_out_cell_line"] == held_out), None)
                if match is None or digest != match["selected_feature_indices_sha256"]:
                    raise ValueError(f"Feature selection mismatch: {selector}, held out {held_out}, {model_name}")
            fold_sets[held_out] = set(map(int, indices))
            fold_reports.append({
                "held_out_cell_line": held_out,
                "train_sample_count": int(tr.sum()),
                "training_cell_lines": sorted(set(groups[tr])),
                "selected_indices_sha256": digest,
                "selected_gene_ids": [str(gene_ids[i]) for i in indices],
                "fraction_selected_genes_same_DTP_Acute_direction_across_train_lines": sign_agreement,
            })
        selections[selector] = fold_sets
        pairwise = build_pairwise(fold_sets)
        freq = frequency_summary(fold_sets, gene_ids)
        report["selectors"][selector] = {
            "folds": fold_reports,
            "pairwise_jaccard": pairwise,
            "average_pairwise_jaccard": float(np.mean([row["jaccard"] for row in pairwise])),
            "frequency": freq,
        }
        print(f"{selector}: mean pairwise Jaccard={report['selectors'][selector]['average_pairwise_jaccard']:.3f}; "
              f"selected in all 4 folds={freq['all_fold_intersection_genes']}; union={freq['union_genes']}")

    for held_out in folds:
        report["same_fold_between_selector_jaccard"].append({
            "held_out_cell_line": held_out,
            **overlap(selections["anova"][held_out], selections["stable"][held_out]),
        })
    print("Same-fold ANOVA / Stable Jaccard:", ", ".join(
        f"{x['held_out_cell_line']}={x['jaccard']:.3f}" for x in report["same_fold_between_selector_jaccard"]
    ))
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_figure({k: report["selectors"][k] for k in SELECTORS}, root / FIGURE)
    print("Saved report:", dest)
    print("Saved figure:", root / FIGURE)
    print("Diagnostic only: do not use PC9 to tune model parameters or claim independent validation.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
