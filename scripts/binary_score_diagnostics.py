"""Diagnostic-only ranking and decision-threshold audit of frozen Stage-35 LR.

Run from the repository root:
    python scripts/binary_score_diagnostics.py --self-test
    python scripts/binary_score_diagnostics.py

No hyperparameter search, test-label fitting, or model selection. The source
cell-line folds are historically inspected; this is NOT independent validation.
The original Stage-35 JSON and all fitted model source files remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STAGE35 = Path("scripts/35_dtp_vs_acute_benchmark.py")
STAGE31 = Path("results/dcsl_stage31_evaluation_protocol.json")
STAGE35_REPORT = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
OUTPUT = Path("results/binary_score_diagnostics.json")


def load_stage35(root: Path):
    script = root / STAGE35
    spec = importlib.util.spec_from_file_location("stage35_for_score_audit", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize_scores(y: np.ndarray, scores: np.ndarray) -> dict:
    """Evaluate ranking without selecting/changing a decision threshold."""
    y = np.asarray(y, dtype=str)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid labels or scores")
    if set(y) != {"Acute", "DTP"}:
        raise ValueError("Exactly Acute and DTP labels are required")
    acute = scores[y == "Acute"]
    dtp = scores[y == "DTP"]
    auc = roc_auc_score(y == "DTP", scores)
    if not 0 <= auc <= 1:
        raise ValueError("Invalid ROC-AUC")
    return {
        "roc_auc": float(auc),
        "acute_mean_score": float(np.mean(acute)),
        "dtp_mean_score": float(np.mean(dtp)),
        "acute_max_score": float(np.max(acute)),
        "dtp_min_score": float(np.min(dtp)),
        "rank_separation_gap_dtp_min_minus_acute_max": float(np.min(dtp) - np.max(acute)),
        "acute_above_fixed_threshold_zero": int(np.count_nonzero(acute >= 0)),
        "dtp_below_fixed_threshold_zero": int(np.count_nonzero(dtp < 0)),
    }


def self_test():
    example_y = np.array(["Acute"] * 3 + ["DTP"] * 3)
    example_scores = np.array([-4.0, -5.0, -3.5, -0.05, -0.07, -0.01])
    s = summarize_scores(example_y, example_scores)
    assert s["roc_auc"] == 1.0
    assert s["dtp_below_fixed_threshold_zero"] == 3
    assert s["rank_separation_gap_dtp_min_minus_acute_max"] > 0
    flipped = summarize_scores(example_y, -example_scores)
    assert flipped["roc_auc"] == 0.0
    print("Self-test passed: ranking and classification can disagree")


def run(root: Path) -> dict:
    stage35 = load_stage35(root)
    # Reuse existing source-only provenance and cohort guards; do NOT change Stage-35 results.
    stage35.run(root, check_only=True)
    prior = json.loads((root / STAGE35_REPORT).read_text(encoding="utf-8"))
    protocol = json.loads((root / STAGE31).read_text(encoding="utf-8"))
    if (prior.get("analysis") != "stage35_binary_dtp_acute_source_only_loco_exploratory"
        or prior.get("n_binary_samples") != 24
        or prior.get("fixed_models", {}).get("LogisticRegression") != {"C": 0.01, "k": 500}
        or prior.get("frozen_stage31_sha256") != stage35.sha256(root / STAGE31)
        or prior.get("not_independent_validation") is not True):
        raise ValueError("Frozen Stage-35 reference is inconsistent")

    stage34 = stage35.import_stage34(root)
    raw_meta = pd.read_csv(root / stage34.METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(raw_meta, protocol)
    meta = raw_meta.rename(columns={"candidate_group": "state"})
    folds = protocol["nested_loco_protocol"]["folds"]
    stage35.validate_binary_cohort(meta, folds)
    matrix = pd.read_csv(root / stage34.PREPARED, sep="\t", compression="gzip")
    _genes, x = stage34.validate_matrix(matrix, meta.gsm.tolist())
    y = meta.state.to_numpy(dtype=str)
    idx = {str(gsm): i for i, gsm in enumerate(meta.gsm)}
    old_folds = {row["held_out_line"]: row for row in prior["folds_by_family"]["LogisticRegression"]}
    if len(old_folds) != 4:
        raise ValueError("Missing frozen Stage-35 folds")

    records = []
    for fold in folds:
        line = fold["outer_test_cell_line"]
        train_ids, test_ids, _inner = stage35.select_frozen_rows(fold, meta)
        tr = np.asarray([idx[gsm] for gsm in train_ids], dtype=int)
        te = np.asarray([idx[gsm] for gsm in test_ids], dtype=int)
        if len(tr) != 18 or len(te) != 6 or set(meta.cell_line.iloc[tr]) & set(meta.cell_line.iloc[te]):
            raise ValueError("Held-out cell-line leakage or incorrect fold")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            model = stage35.make_model("LogisticRegression").fit(x[tr], y[tr])
        classes = model.named_steps["classifier"].classes_.tolist()
        if classes != ["Acute", "DTP"]:
            raise ValueError("Score sign not guaranteed to mean DTP")
        scores = np.asarray(model.decision_function(x[te]), dtype=np.float64)
        predictions = np.asarray(model.predict(x[te]), dtype=str)
        expected = old_folds[line]["predictions"]
        actual = [{"gsm": gsm, "actual": str(yi), "predicted": str(pi)}
                  for gsm, yi, pi in zip(test_ids, y[te], predictions)]
        if actual != expected:
            raise ValueError(f"Refitted baseline predictions differ from Stage 35: {line}")
        threshold_preds = np.where(scores >= 0, "DTP", "Acute")
        if not np.array_equal(threshold_preds, predictions):
            raise ValueError("Decision scores disagree with model predictions")

        diagnostics = summarize_scores(y[te], scores)
        row = {
            "held_out_line": line,
            "n_test": int(len(te)),
            "frozen_balanced_accuracy": float(old_folds[line]["outer_metrics"]["balanced_accuracy"]),
            "frozen_dtp_recall": float(old_folds[line]["outer_metrics"]["dtp_recall"]),
            **diagnostics,
            "sample_scores": [
                {"gsm": gsm, "actual": str(yi), "predicted": str(pi), "raw_logit_DTP": float(sc)}
                for gsm, yi, pi, sc in zip(test_ids, y[te], predictions, scores)
            ],
        }
        records.append(row)
        print(f"{line}: BA={row['frozen_balanced_accuracy']:.3f}, "
              f"ROC-AUC={row['roc_auc']:.3f}, "
              f"gap={row['rank_separation_gap_dtp_min_minus_acute_max']:+.3f}, "
              f"missed DTP={row['dtp_below_fixed_threshold_zero']}/3")

    report = {
        "analysis": "diagnostic_only_frozen_stage35_binary_score_audit",
        "source": "GSE193258",
        "protocol_sha256": stage35.sha256(root / STAGE31),
        "stage35_report_sha256": stage35.sha256(root / STAGE35_REPORT),
        "no_hyperparameter_search": True,
        "no_threshold_tuning": True,
        "test_labels_used_only_for_posthoc_ranking_metrics": True,
        "historically_inspected_folds": True,
        "independent_validation": False,
        "limitations": [
            "Only 3 Acute and 3 DTP samples per held-out cell line: ROC-AUC is highly uncertain.",
            "A high ROC-AUC does not license choosing a threshold from held-out labels.",
            "Cell-line scores come from different fitted models; do not pool them as calibrated scores.",
            "Acute 24h and DTP 21d are confounded with exposure duration.",
            "This audit neither improves the classifier nor establishes an original method.",
        ],
        "folds": records,
    }
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", dest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.root.resolve())


if __name__ == "__main__":
    main()
