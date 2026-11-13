"""Stage 28: fixed-setting, matched-selector domain generalization benchmark.

Uses source GSE193258 only. Four leave-one-cell-line-out folds.
No hyperparameter selection on PC9 or GSE255958; post-hoc exploratory audit.
Compares existing ERM/DCSL/GroupRobust to established V-REx (Krueger et al.).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier, STATES  # noqa: E402
from osimertinib_ml.group_robust_contrast import GroupRobustDCSLClassifier  # noqa: E402
from osimertinib_ml.vrex_domain_generalization import (  # noqa: E402
    VRExDCSLClassifier, group_risk_variance_and_gradient,
)

STAGE09 = Path("scripts/09_train_baselines.py")
OLD23 = Path("results/dcsl_stage23_exploratory.json")
OLD25 = Path("results/dcsl_stage25_group_robust_exploratory.json")
OUTPUT = Path("results/dcsl_stage28_domain_generalization.json")
FIGURE = Path("figures/dcsl_stage28_domain_generalization.png")
L2, GAMMA, TAU, K = 0.1, 0.05, 0.5, 500
ETA, TEMPERATURE, BETA = 0.2, 0.5, 0.2
CONFIGS = ("Stable_CE", "DCSL", "Stable_GroupRobust", "Stable_VREx")
CELL_LINES = ("H1975", "HCC2935", "HCC827", "PC9")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_source_only(root: Path, previous: dict):
    """Read GSE193258 ONLY; independently verify stage-09 source and metadata hashes.

    The GSE255958 expression file is not opened or needed in stage 28.
    """
    folder = root / "data/processed/cross_study"
    expected = previous["input_sha256"]
    for filename in ("gse193258_common_log2tpm.tsv.gz", "samples.csv"):
        path = folder / filename
        if not path.is_file() or sha256(path) != expected[filename]:
            raise ValueError(f"Changed or missing source/metadata input: {filename}")
    spec = importlib.util.spec_from_file_location("stage28_stage09", root / STAGE09)
    if spec is None or spec.loader is None:
        raise ValueError("Missing stage-09 matrix reader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.read_matrix(folder / "gse193258_common_log2tpm.tsv.gz")
    meta = pd.read_csv(folder / "samples.csv", dtype=str, keep_default_na=False)
    if not {"accession", "gsm", "cell_line", "state"}.issubset(meta.columns):
        raise ValueError("Bad source metadata")
    info = meta.loc[meta["accession"] == "GSE193258"].copy()
    if (len(info) != 36 or info["gsm"].duplicated().any()
            or set(source.columns) != set(info["gsm"])
            or set(info["cell_line"]) != set(CELL_LINES)
            or set(info["state"]) != set(STATES)
            or (info.groupby(["cell_line", "state"]).size() != 3).any()):
        raise ValueError("Unexpected source cohort or missing cell line/state groups")
    info = info.set_index("gsm").loc[list(source.columns)].rename_axis("gsm").reset_index()
    x = source.to_numpy(dtype=np.float64).T
    if x.shape != (36, 18013):
        raise ValueError("Unexpected source matrix dimensions")
    return x, info


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", labels=list(STATES), zero_division=0)),
        "DTP_recall": float(recall_score(actual, predicted, average="macro", labels=["DTP"], zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(STATES)).tolist(),
        "correct": int(np.sum(actual == predicted)), "n": int(len(actual)),
    }


def validate_prior(prior23: dict, prior25: dict) -> None:
    if prior23.get("analysis") != "DCSL_stage23_fixed_ablation_prototype":
        raise ValueError("Stage-23 report mismatch")
    if prior25.get("analysis") != "stage25_fixed_group_robust_DTP_Acute_ablation":
        raise ValueError("Stage-25 report mismatch")
    if prior23.get("input_sha256") != prior25.get("input_sha256"):
        raise ValueError("Mismatched source data hashes")
    if prior23.get("fixed_parameters_no_heldout_tuning") != {"features": K, "l2": L2, "gamma": GAMMA, "tau": TAU}:
        raise ValueError("Unexpected stage-23 constants")
    if prior25.get("fixed_hyperparameters_not_heldout_tuned") != {
        "n_features": K, "l2": L2, "gamma": GAMMA, "stable_tau": TAU,
        "group_robust_weight_eta": ETA, "smoothmax_temperature": TEMPERATURE,
    }:
        raise ValueError("Unexpected stage-25 constants")


def make_model(name: str):
    common = dict(selector="stable", n_features=K, l2=L2, tau=TAU)
    if name == "Stable_CE":
        return DCSLClassifier(gamma=0.0, **common)
    if name == "DCSL":
        return DCSLClassifier(gamma=GAMMA, **common)
    if name == "Stable_GroupRobust":
        return GroupRobustDCSLClassifier(gamma=0.0, eta=ETA, temperature=TEMPERATURE, **common)
    if name == "Stable_VREx":
        return VRExDCSLClassifier(gamma=0.0, beta=BETA, **common)
    raise ValueError(f"Unknown config: {name}")


def assert_reproduction(name: str, fold: dict, clf, prior23: dict, prior25: dict):
    if name == "Stable_VREx":
        return
    previous = prior25 if name == "Stable_GroupRobust" else prior23
    old = next((d for d in previous["models"][name]["loco_folds"]
                if d["held_out_cell_line"] == fold["held_out_cell_line"]), None)
    if old is None:
        raise RuntimeError(f"Missing reference fold: {name}")
    old_preds = [(p["gsm"], p["actual"], p["predicted"]) for p in old["predictions"]]
    new_preds = [(p["gsm"], p["actual"], p["predicted"]) for p in fold["predictions"]]
    if new_preds != old_preds:
        raise RuntimeError(f"Stage 23/25 predictions changed: {name}/{fold['held_out_cell_line']}")
    if not np.isclose(clf.train_margin_variance_, old["training_margin_variance"] if name == "Stable_GroupRobust" else old["train_margin_variance"], rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"Stage 23/25 training loss diagnostic changed: {name}")


def plot_scores(models: dict, out: Path):
    matrix = np.asarray([[next(f for f in models[m]["loco_folds"]
                                  if f["held_out_cell_line"] == c)["metrics"]["DTP_recall"]
                          for c in CELL_LINES] for m in CONFIGS])
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(CONFIGS)), CONFIGS)
    ax.set_xticks(range(len(CELL_LINES)), CELL_LINES)
    ax.set_title("Stage 28: DTP recall by held-out cell line (exploratory)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(round(matrix[i,j]*3))}/3", va="center", ha="center",
                    color="white" if matrix[i,j] > .65 else "black")
    fig.colorbar(im, ax=ax, label="DTP recall")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def run(root: Path):
    p23 = json.loads((root / OLD23).read_text(encoding="utf-8"))
    p25 = json.loads((root / OLD25).read_text(encoding="utf-8"))
    validate_prior(p23, p25)
    x, meta = load_source_only(root, p23)  # reads NO target expression matrix
    y = meta["state"].to_numpy()
    groups = meta["cell_line"].to_numpy()
    if x.shape != (36, 18013) or set(groups) != set(CELL_LINES):
        raise ValueError("Unexpected source cohort")
    report = {
        "analysis": "stage28_matched_domain_generalization_benchmark",
        "source": "GSE193258", "source_n": len(x), "n_features": x.shape[1],
        "input_sha256": p23["input_sha256"],
        "target_study_evaluated": False,
        "target_expression_file_read": False,
        "target_or_heldout_used_for_fitting_selection_scaling_or_tuning": False,
        "protocol": "4-fold source leave-one-cell-line-out with same stable selector, K, L2, optimizer family",
        "fixed_parameters_exploratory_not_tuned": {
            "n_features": K, "l2": L2, "tau": TAU,
            "DCSL_gamma": GAMMA, "GroupRobust_eta": ETA,
            "GroupRobust_temperature": TEMPERATURE, "VREx_beta": BETA,
        },
        "models": {},
        "references": {
            "V_REx": "Krueger et al. (ICML 2021), https://proceedings.mlr.press/v139/krueger21a.html",
            "GroupDRO": "Sagawa et al. (ICLR 2020), https://mlanthology.org/iclr/2020/sagawa2020iclr-distributionally/",
            "DomainBed": "Gulrajani and Lopez-Paz (ICLR 2021), https://mlanthology.org/iclr/2021/gulrajani2021iclr-search/",
        },
        "limitations": [
            "All four cell lines, especially PC9, have been inspected repeatedly; exploratory post-hoc results.",
            "V-REx and GroupRobust are established ideas; no method novelty claim.",
            "36 samples, just 4 cell lines and 3 samples per state/group; uncertainty is substantial.",
            "All hyperparameters are fixed hypotheses, not objectively selected via nested grouped validation.",
            "External study GSE255958 is NOT used. Cross-study numeric scale/labels remain provisional.",
            "Smaller training risk variance does not establish confounder removal or causality.",
        ],
    }
    for name in CONFIGS:
        actual_all, predicted_all, fold_rows = [], [], []
        for heldout in CELL_LINES:
            train_mask = groups != heldout
            held_mask = ~train_mask
            clf = make_model(name).fit(x[train_mask], y[train_mask], groups[train_mask])
            predicted = clf.predict(x[held_mask])
            train_z = clf.scaler_.transform(x[train_mask][:, clf.nonconstant_indices_])[:, clf.selected_local_]
            train_idx = np.asarray([list(STATES).index(label) for label in y[train_mask]], dtype=int)
            parameters = np.concatenate([clf.coef_.ravel(), clf.intercept_])
            risk_variance, _, train_risks = group_risk_variance_and_gradient(
                parameters, train_z, train_idx, groups[train_mask]
            )
            fold = {
                "held_out_cell_line": heldout,
                "train_cell_lines": sorted(set(groups[train_mask])),
                "n_train": int(train_mask.sum()), "n_test": int(held_mask.sum()),
                "metrics": metrics(y[held_mask], predicted),
                "selected_feature_indices_sha256": hashlib.sha256(
                    np.asarray(clf.selected_indices_, dtype="<i8").tobytes()).hexdigest(),
                "train_margin_variance": float(clf.train_margin_variance_),
                "train_risk_variance": float(risk_variance),
                "train_group_multiclass_CE": {
                    str(k): float(v) for k, v in zip(np.unique(groups[train_mask]), train_risks)
                },
                "predictions": [{"gsm": str(gsm), "actual": str(a), "predicted": str(p)}
                                for gsm, a, p in zip(meta.loc[held_mask, "gsm"], y[held_mask], predicted)],
            }
            old_feature_hash = next(f["selected_feature_indices_sha256"] for f in
                p23["models"]["Stable_CE"]["loco_folds"]
                if f["held_out_cell_line"] == heldout)
            if fold["selected_feature_indices_sha256"] != old_feature_hash:
                raise RuntimeError(f"Feature selection does not reproduce stage 23: {heldout}")
            assert_reproduction(name, fold, clf, p23, p25)
            fold_rows.append(fold)
            actual_all.extend(y[held_mask]); predicted_all.extend(predicted)
        pooled = metrics(np.asarray(actual_all), np.asarray(predicted_all))
        report["models"][name] = {
            "loco_pooled": pooled, "loco_folds": fold_rows,
            "worst_fold_balanced_accuracy": float(min(f["metrics"]["balanced_accuracy"] for f in fold_rows)),
            "worst_fold_DTP_recall": float(min(f["metrics"]["DTP_recall"] for f in fold_rows)),
        }
        pc9 = next(f for f in fold_rows if f["held_out_cell_line"] == "PC9")
        print(f"{name:21s} LOCO BA={pooled['balanced_accuracy']:.3f} DTP={pooled['DTP_recall']:.3f} "
              f"worst DTP={report['models'][name]['worst_fold_DTP_recall']:.3f} "
              f"PC9 DTP={pc9['metrics']['DTP_recall']:.3f}")
    out = root / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_scores(report["models"], root / FIGURE)
    print("Saved report:", out)
    print("Saved figure:", root / FIGURE)
    print("Exploratory method benchmarking only. No new-method or untouched-test claim.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT, type=Path)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
