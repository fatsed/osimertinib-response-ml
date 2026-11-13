"""Stage 25: fixed, train-only worst-group DTP/Acute contrast ablation.

No tuning or selection on previously examined held-out PC9 or the target study.
This is exploratory method development, NOT independent validation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.group_robust_contrast import GroupRobustDCSLClassifier  # noqa: E402
from osimertinib_ml.contrast_stability import STATES  # noqa: E402

STAGE23 = Path("scripts/23_run_dcsl_experiments.py")
PREVIOUS = Path("results/dcsl_stage23_exploratory.json")
OUTPUT = Path("results/dcsl_stage25_group_robust_exploratory.json")
# Exploratory fixed design; these are NOT hyperparameters selected on any held-out cell line.
L2, GAMMA, TAU, K = 0.1, 0.05, 0.5, 500
ETA, TEMPERATURE = 0.2, 0.5
# All use stable selection, allowing a direct contrast of the two penalties.
CONFIGS = (
    ("Stable_CE", 0.0, 0.0),
    ("DCSL", GAMMA, 0.0),
    ("Stable_GroupRobust", 0.0, ETA),
    ("DCSL_GroupRobust", GAMMA, ETA),
)


def load_stage23(root: Path):
    path = root / STAGE23
    spec = importlib.util.spec_from_file_location("dcsl_stage23_reuse", path)
    if spec is None or spec.loader is None:
        raise ValueError("Stage 23 script missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "macro_f1": float(f1_score(y_true, prediction, labels=list(STATES), average="macro", zero_division=0)),
        "dtp_recall": float(recall_score(y_true, prediction, labels=["DTP"], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=list(STATES)).tolist(),
        "correct": int(np.sum(y_true == prediction)), "n": int(len(y_true)),
    }


def run(root: Path) -> dict:
    prior = json.loads((root / PREVIOUS).read_text(encoding="utf-8"))
    if prior.get("analysis") != "DCSL_stage23_fixed_ablation_prototype" or prior.get("fixed_parameters_no_heldout_tuning") != {
        "features": K, "l2": L2, "gamma": GAMMA, "tau": TAU,
    }:
        raise ValueError("Stage-23 report or fixed parameters changed")
    stage = load_stage23(root)
    _, x, meta, _, _ = stage.load_stage09(root)  # stage-09 SHA-256 checks input integrity
    y = meta["state"].to_numpy()
    groups = meta["cell_line"].to_numpy()
    if x.shape != (36, 18013) or set(groups) != {"PC9", "H1975", "HCC2935", "HCC827"}:
        raise ValueError("Unexpected source matrix or training groups")
    report = {
        "analysis": "stage25_fixed_group_robust_DTP_Acute_ablation",
        "source": "GSE193258", "source_samples": 36, "folds": 4,
        "input_sha256": prior["input_sha256"],
        "fixed_hyperparameters_not_heldout_tuned": {
            "n_features": K, "l2": L2, "gamma": GAMMA, "stable_tau": TAU,
            "group_robust_weight_eta": ETA, "smoothmax_temperature": TEMPERATURE,
        },
        "objective": "multiclass CE + (l2/2)||W||^2 + gamma Var_c(mean_margin_DTP-Acute) + eta * temperature * logmeanexp_c(DA_binary_CE_c / temperature)",
        "train_only_group_risk": "binary Acute/DTP log loss per training cell line; smooth maximum across training lines",
        "target_study_not_evaluated": True,
        "heldout_groups_excluded_from_selection_scaler_training": True,
        "PC9_has_been_previously_inspected": True,
        "not_untouched_independent_validation": True,
        "not_a_demonstrated_novel_method": True,
        "limitations": [
            "Only 4 biological cell lines; groups have 3 samples per state and results have high uncertainty.",
            "The held-out PC9 failure was previously inspected. This is exploratory, not unbiased selection.",
            "No parameter search on held-out data; preset eta and temperature are design choices, not validated optima.",
            "This risk is a smooth GroupDRO-inspired loss; it is NOT a new group-robust optimization principle.",
            "Study-level effects and sample labels/harmonized expression scale remain provisional.",
        ],
        "models": {},
    }
    for name, gamma, eta in CONFIGS:
        all_true, all_pred, folds = [], [], []
        for heldout in sorted(set(groups)):
            training = groups != heldout
            test = ~training
            clf = GroupRobustDCSLClassifier(
                selector="stable", n_features=K, l2=L2, gamma=gamma,
                tau=TAU, eta=eta, temperature=TEMPERATURE,
            ).fit(x[training], y[training], groups[training])
            predicted = clf.predict(x[test])
            all_true.extend(y[test]); all_pred.extend(predicted)
            folds.append({
                "held_out_cell_line": str(heldout),
                "trained_on": clf.train_groups_,
                "metrics": metrics(y[test], predicted),
                "training_margin_variance": clf.train_margin_variance_,
                "training_worstgroup_DA_logloss": max(clf.group_da_losses_.values()) if clf.group_da_losses_ else None,
                "predictions": [
                    {"gsm": str(gsm), "actual": str(actual), "predicted": str(pred)}
                    for gsm, actual, pred in zip(meta.loc[test, "gsm"], y[test], predicted)
                ],
            })
            if eta == 0:
                prev = next(f for f in prior["models"][name]["loco_folds"] if f["held_out_cell_line"] == heldout)
                old = [(p["gsm"], p["actual"], p["predicted"]) for p in prev["predictions"]]
                new = [(p["gsm"], p["actual"], p["predicted"]) for p in folds[-1]["predictions"]]
                if old != new or not np.isclose(prev["train_margin_variance"], clf.train_margin_variance_, atol=1e-7):
                    raise RuntimeError(f"Stage-23 reproduction mismatch for {name}/{heldout}")
        pooled = metrics(np.asarray(all_true), np.asarray(all_pred))
        report["models"][name] = {"loco_pooled": pooled, "loco_folds": folds}
        pc9 = next(f for f in folds if f["held_out_cell_line"] == "PC9")
        print(f"{name}: LOCO BA={pooled['balanced_accuracy']:.3f}, DTP recall={pooled['dtp_recall']:.3f}; "
              f"held-out PC9 DTP recall={pc9['metrics']['dtp_recall']:.3f}")
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", output)
    print("Exploratory only. No tuning against previously inspected PC9 or the target study.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
