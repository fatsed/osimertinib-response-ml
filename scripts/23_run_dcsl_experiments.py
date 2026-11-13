"""Stage 23: fixed-setting DCSL ablations; exploratory study transfer only.

Files prepared by stage 08 and their hashes are checked against stage 09.
Target labels never enter preprocessing, feature scoring or training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier, STATES  # noqa: E402

# Reuse stage-09 data validation rather than creating a competing loader.
from importlib.util import spec_from_file_location, module_from_spec

INPUT = Path("data/processed/cross_study")
STAGE09 = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/dcsl_stage23_exploratory.json")
FILES = ("gse193258_common_log2tpm.tsv.gz",
         "gse255958_common_log2tpm_candidate.tsv.gz", "samples.csv")
L2, GAMMA, TAU, K = 0.1, 0.05, 0.5, 500
CONFIGS = (
    ("ANOVA_CE", "anova", 0.0),
    ("Stable_CE", "stable", 0.0),
    ("ANOVA_Stability", "anova", GAMMA),
    ("DCSL", "stable", GAMMA),
)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stage09(root: Path):
    report = json.loads((root / STAGE09).read_text(encoding="utf-8"))
    if report.get("source") != "GSE193258" or report.get("target") != "GSE255958":
        raise ValueError("Unexpected stage 09 report")
    if report.get("supervised_feature_count") != K:
        raise ValueError("Stage 09 feature count differs")
    for name in FILES:
        path = root / INPUT / name
        if not path.is_file() or checksum(path) != report["input_sha256"].get(name):
            raise ValueError(f"Stage-09 input changed or missing: {name}")
    spec = spec_from_file_location("stage09_loader", root / "scripts/09_train_baselines.py")
    if spec is None or spec.loader is None:
        raise ValueError("Stage 09 loader not available")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    xs, source, xt, target, n_genes = module.load_data(root)
    if n_genes != 18013 or len(xs) != 36 or len(xt) != 9:
        raise ValueError("Unexpected cohort size")
    if set(source["state"]) != set(STATES) or set(target["state"]) != set(STATES):
        raise ValueError("Unexpected state labels")
    return report, xs, source, xt, target


def evaluate(true: np.ndarray, pred: np.ndarray):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=list(STATES), average="macro", zero_division=0)),
        "DTP_recall": float(recall_score(true, pred, labels=["DTP"], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(true, pred, labels=list(STATES)).tolist(),
        "correct": int(np.sum(true == pred)), "n": int(len(true)),
    }


def run(root: Path):
    prior, x, info, ext_x, ext_info = load_stage09(root)
    y = info["state"].to_numpy()
    groups = info["cell_line"].to_numpy()
    target_y = ext_info["state"].to_numpy()
    report = {
        "analysis": "DCSL_stage23_fixed_ablation_prototype",
        "source": "GSE193258", "target": "GSE255958",
        "train_samples": 36, "target_samples": 9,
        "preprocessing": "variance filter > 1e-8, standardization, then training-only feature ranking",
        "input_sha256": {name: prior["input_sha256"][name] for name in FILES},
        "fixed_parameters_no_heldout_tuning": {"features": K, "l2": L2, "gamma": GAMMA, "tau": TAU},
        "objective": "mean multiclass CE + (l2/2)*||W||^2 + gamma*Var_train_cell_line(margin DTP-Acute)",
        "selection_contrast": "mean standardized expression DTP minus Acute per training cell line",
        "evaluation": "4-fold leave-one-cell-line-out on GSE193258; reused GSE255958 exploratory diagnostic",
        "target_used_in_selection_or_training": False,
        "not_untouched_external_validation": True,
        "not_clinically_validated": True,
        "not_a_demonstrated_novel_algorithm": True,
        "comparability_note": "ANOVA_CE uses a different softmax L2 convention from stage-09 sklearn LogisticRegression; comparison is descriptive, not exact reproduction.",
        "limitations": [
            "DTP labels, biological independence and exact harmonized expression scale remain provisional.",
            "Only 4 source cell lines and 3 samples per state/cell line; uncertainty is high.",
            "The same target study has been seen during earlier development; no untouched test claim.",
            "Study and cell-line/protocol changes are entangled in the external comparison.",
            "Neither stable feature ranking nor variance penalty alone proves confounder removal or novelty.",
            "Fixed hyperparameters are exploratory, not selected by unbiased nested CV.",
        ],
        "models": {},
    }
    for label, selector, gamma in CONFIGS:
        folds = []
        all_y, all_pred = [], []
        for held_out in sorted(set(groups)):
            tr, te = groups != held_out, groups == held_out
            model = DCSLClassifier(selector=selector, gamma=gamma, l2=L2,
                                   n_features=K, tau=TAU).fit(x[tr], y[tr], groups[tr])
            pred = model.predict(x[te])
            all_y.extend(y[te]); all_pred.extend(pred)
            folds.append({
                "held_out_cell_line": held_out,
                "training_cell_lines": model.train_groups_,
                "metrics": evaluate(y[te], pred),
                "train_margin_variance": model.train_margin_variance_,
                "optimizer_iterations": model.n_iter_,
                "selected_feature_indices_sha256": hashlib.sha256(
                    model.selected_indices_.astype("<i8").tobytes()).hexdigest(),
                "predictions": [
                    {"gsm": str(gsm), "actual": str(actual), "predicted": str(guess)}
                    for gsm, actual, guess in zip(info.loc[te, "gsm"], y[te], pred)
                ],
            })
        full = DCSLClassifier(selector=selector, gamma=gamma, l2=L2,
                              n_features=K, tau=TAU).fit(x, y, groups)
        external_pred = full.predict(ext_x)
        report["models"][label] = {
            "loco_pooled": evaluate(np.asarray(all_y), np.asarray(all_pred)),
            "loco_folds": folds,
            "external_exploratory": {
                "metrics": evaluate(target_y, external_pred),
                "predictions": [
                    {"gsm": str(gsm), "actual": str(actual), "predicted": str(guess)}
                    for gsm, actual, guess in zip(ext_info["gsm"], target_y, external_pred)
                ],
            },
            "full_train_margin_variance": full.train_margin_variance_,
            "full_train_selected_features_sha256": hashlib.sha256(
                full.selected_indices_.astype("<i8").tobytes()).hexdigest(),
            "max_optimization_iterations": max(full.n_iter_, *(f["optimizer_iterations"] for f in folds)),
        }
        loco = report["models"][label]["loco_pooled"]
        ext = report["models"][label]["external_exploratory"]["metrics"]
        print(f"{label}: LOCO BA={loco['balanced_accuracy']:.3f} DTP recall={loco['DTP_recall']:.3f}; "
              f"reused target BA={ext['balanced_accuracy']:.3f} DTP recall={ext['DTP_recall']:.3f}")

    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", dest)
    print("Exploratory prototype. Do NOT use these results to claim independent validation or method novelty.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
