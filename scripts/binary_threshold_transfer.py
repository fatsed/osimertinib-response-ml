"""Frozen Stage-35 binary LogisticRegression: decision-threshold transfer audit.

This is a pre-specified, exploratory comparison on four *previously inspected*
GSE193258 cell lines. Nothing is optimized against outer-test labels.

Three variants share one fitted classifier and one fixed feature-selection rule:
  FixedZero: original logistic threshold at 0, no target adaptation.
  SourceInnerLOCO: single threshold selected from training-cell-line cross-fit
      scores and training labels only; applied unchanged to the outer test line.
  UnlabeledMedianAlign: threshold = median(outer-test scores) -
      median(outer-train scores), without viewing target labels. This is
      TRANSDUCTIVE and requires the full unlabeled target batch. It assumes
      similar score-mixture composition in source and target; a 3/3 curated
      test set is not representative of unconstrained clinical class priors.

Do not interpret performance here as independent validation, calibration,
new methodological novelty, or evidence independent of exposure duration.

From the repository root:
  python scripts/binary_threshold_transfer.py --self-test
  python scripts/binary_threshold_transfer.py --check-only
  python scripts/binary_threshold_transfer.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
STAGE35_SCRIPT = Path("scripts/35_dtp_vs_acute_benchmark.py")
STAGE35_REPORT = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
SCORE_REPORT = Path("results/binary_score_diagnostics.json")
STAGE31_REPORT = Path("results/dcsl_stage31_evaluation_protocol.json")
OUTPUT = Path("results/binary_threshold_transfer_development.json")
VARIANTS = ("FixedZero", "SourceInnerLOCO", "UnlabeledMedianAlign")


def load_stage35(root: Path):
    path = root / STAGE35_SCRIPT
    spec = importlib.util.spec_from_file_location("stage35_for_threshold_audit", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scores(model, data: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_.tolist()
    if classes != ["Acute", "DTP"]:
        raise ValueError(f"Unexpected sign of logistic scores: {classes}")
    values = np.asarray(model.decision_function(data), dtype=np.float64)
    if values.shape != (len(data),) or not np.isfinite(values).all():
        raise ValueError("Expected one finite DTP score per sample")
    return values


def predict_at(scores: np.ndarray, threshold: float) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim != 1 or not np.isfinite(z).all() or not np.isfinite(threshold):
        raise ValueError("Expected finite 1D decision scores and threshold")
    return np.where(z >= threshold, "DTP", "Acute")


def choose_source_threshold(inner_scores: list[np.ndarray], inner_labels: list[np.ndarray]) -> tuple[float, dict]:
    """Select a single raw-score threshold by equal-weight mean inner-line BA.

    All score/label arrays come from *outer-training* lines predicted by models
    fitted on the other outer-training lines. Inherently scale-sensitive: inner
    models use fewer training lines than the final model.
    """
    if len(inner_scores) != 3 or len(inner_labels) != 3:
        raise ValueError("Need the three frozen inner training-line folds")
    scores, labels = [], []
    for s, y in zip(inner_scores, inner_labels):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=str)
        if s.shape != y.shape or s.ndim != 1 or not np.isfinite(s).all() or set(y) != {"Acute", "DTP"}:
            raise ValueError("Invalid inner fold scores or class labels")
        scores.append(s)
        labels.append(y)
    unique = np.unique(np.concatenate(scores))
    mid = (unique[:-1] + unique[1:]) / 2
    candidates = np.unique(np.concatenate(([0.0], unique, mid)))
    # Cross-fit BA is evaluated per cell line and averaged, not pooled by sample.
    best_ba = -1.0
    best_t = 0.0
    for t in candidates:
        ba = float(np.mean([balanced_accuracy_score(y, predict_at(s, float(t)))
                            for s, y in zip(scores, labels)]))
        # Exact tie-breaking favors the original zero threshold, then lower t.
        if ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12
                and (abs(float(t)), float(t)) < (abs(best_t), best_t)):
            best_ba, best_t = ba, float(t)
    zero_ba = float(np.mean([balanced_accuracy_score(y, predict_at(s, 0.0))
                             for s, y in zip(scores, labels)]))
    return best_t, {"n_candidates": int(len(candidates)), "inner_mean_ba_at_selected": best_ba,
                    "inner_mean_ba_at_zero": zero_ba}


def unlabeled_median_threshold(source_scores: np.ndarray, target_unlabeled_scores: np.ndarray) -> float:
    """Label-free, batch-transductive median-location alignment.

    No number of Acute/DTP samples or their labels enter this function.
    Warning: this simple heuristic can fail under class-prior shift or if a
    target batch contains only one state; 6-sample groups are noisy.
    """
    a = np.asarray(source_scores, dtype=np.float64)
    b = np.asarray(target_unlabeled_scores, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected two finite score batches with at least two samples")
    return float(np.median(b) - np.median(a))


def self_test() -> None:
    source = np.array([-2., -1., 1., 2.])
    target = source - 3.5
    assert np.isclose(unlabeled_median_threshold(source, target), -3.5)
    np.testing.assert_array_equal(predict_at(target, -3.5), predict_at(source, 0.0))
    # A one-class target batch also produces a threshold: it MUST NOT be
    # interpreted as valid binary calibration without its mixture assumption.
    assert np.isfinite(unlabeled_median_threshold(source, [-10., -9.]))
    ys = [np.array(["Acute", "Acute", "DTP", "DTP"]) for _ in range(3)]
    ss = [np.array([-2.0, -1.0, 1.0, 2.0]) for _ in range(3)]
    t, diagnostic = choose_source_threshold(ss, ys)
    assert t == 0.0 and diagnostic["inner_mean_ba_at_zero"] == 1.0
    # Labels for an outer-test line cannot influence the methods: no outer-test
    # labels are arguments to either threshold function.
    assert np.array_equal(predict_at(np.array([-1., 0., 1.]), 0), ["Acute", "DTP", "DTP"])
    for invalid in ([0., float("nan")], []):
        try:
            unlabeled_median_threshold(source, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid target score batch not rejected")
    print("Self-test OK: source-only and label-free transductive thresholds")


def load_inputs(root: Path, stage35):
    baseline = json.loads((root / STAGE35_REPORT).read_text(encoding="utf-8"))
    audit = json.loads((root / SCORE_REPORT).read_text(encoding="utf-8"))
    protocol = json.loads((root / STAGE31_REPORT).read_text(encoding="utf-8"))
    if (baseline.get("analysis") != "stage35_binary_dtp_acute_source_only_loco_exploratory"
            or baseline.get("fixed_models", {}).get("LogisticRegression") != {"C": 0.01, "k": 500}
            or baseline.get("n_binary_samples") != 24
            or baseline.get("frozen_stage31_sha256") != stage35.sha256(root / STAGE31_REPORT)
            or baseline.get("not_independent_validation") is not True
            or audit.get("analysis") != "diagnostic_only_frozen_stage35_binary_score_audit"
            or audit.get("stage35_report_sha256") != stage35.sha256(root / STAGE35_REPORT)
            or audit.get("protocol_sha256") != stage35.sha256(root / STAGE31_REPORT)
            or audit.get("independent_validation") is not False):
        raise ValueError("Stage-35 baseline or binary score audit changed; refuse to compare")
    stage35.run(root, check_only=True)  # existing Stage-05, -31, -34 provenance checks
    stage34 = stage35.import_stage34(root)
    metadata_raw = pd.read_csv(root / stage34.METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(metadata_raw, protocol)
    metadata = metadata_raw.rename(columns={"candidate_group": "state"})
    folds = protocol["nested_loco_protocol"]["folds"]
    stage35.validate_binary_cohort(metadata, folds)
    old = {r["held_out_line"]: r for r in baseline["folds_by_family"]["LogisticRegression"]}
    diagnostic = {r["held_out_line"]: r for r in audit["folds"]}
    if set(old) != set(diagnostic) or len(old) != 4:
        raise ValueError("Different held-out cell-line sets in frozen reports")
    return stage34, metadata, folds, old, diagnostic


def run(root: Path, check_only: bool = False) -> dict:
    stage35 = load_stage35(root)
    stage34, metadata, folds, old, diagnostic = load_inputs(root, stage35)
    if check_only:
        print("Preflight OK: four frozen previously-seen cell lines, score audit verified")
        print("Comparisons: FixedZero, SourceInnerLOCO, UnlabeledMedianAlign")
        return {"check_only": True}
    matrix = pd.read_csv(root / stage34.PREPARED, sep="\t", compression="gzip")
    _, x = stage34.validate_matrix(matrix, metadata.gsm.tolist())
    idx = {str(gsm): i for i, gsm in enumerate(metadata.gsm)}
    y = metadata.state.to_numpy(dtype=str)
    per_variant = {name: [] for name in VARIANTS}
    for fold in folds:
        line = fold["outer_test_cell_line"]
        train_ids, test_ids, inner = stage35.select_frozen_rows(fold, metadata)
        tr = np.asarray([idx[g] for g in train_ids], dtype=int)
        te = np.asarray([idx[g] for g in test_ids], dtype=int)
        if len(tr) != 18 or len(te) != 6 or set(metadata.cell_line.iloc[tr]) & set(metadata.cell_line.iloc[te]):
            raise ValueError("Outer training and test domain overlap or wrong sample count")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            final = stage35.make_model("LogisticRegression").fit(x[tr], y[tr])
        train_scores = _scores(final, x[tr])
        target_scores = _scores(final, x[te])
        expected = old[line]["predictions"]
        actual = [{"gsm": gsm, "actual": str(label), "predicted": str(pred)}
                  for gsm, label, pred in zip(test_ids, y[te], predict_at(target_scores, 0.0))]
        if actual != expected:
            raise ValueError(f"Frozen Stage-35 predictions changed: {line}")
        prior_score_rows = diagnostic[line]["sample_scores"]
        if [r["gsm"] for r in prior_score_rows] != test_ids or not np.allclose(
                [r["raw_logit_DTP"] for r in prior_score_rows], target_scores, rtol=1e-9, atol=1e-8):
            raise ValueError(f"Binary score audit not reproduced: {line}")
        inner_scores = []
        inner_labels = []
        for subtrain_ids, val_ids in inner:
            a = np.asarray([idx[g] for g in subtrain_ids], dtype=int)
            b = np.asarray([idx[g] for g in val_ids], dtype=int)
            if len(a) != 12 or len(b) != 6 or set(metadata.cell_line.iloc[a]) & set(metadata.cell_line.iloc[b]):
                raise ValueError("Inner LOCO domain leakage")
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                inner_model = stage35.make_model("LogisticRegression").fit(x[a], y[a])
            inner_scores.append(_scores(inner_model, x[b]))
            inner_labels.append(y[b])  # inner labels are *outer-training* only
        source_threshold, inner_detail = choose_source_threshold(inner_scores, inner_labels)
        target_threshold = unlabeled_median_threshold(train_scores, target_scores)
        fixed_auc = float(roc_auc_score(y[te] == "DTP", target_scores))  # descriptive only
        for variant, threshold in (("FixedZero", 0.0), ("SourceInnerLOCO", source_threshold),
                                   ("UnlabeledMedianAlign", target_threshold)):
            preds = predict_at(target_scores, threshold)
            metrics = stage35.metrics(y[te], preds)  # held-out labels used only for evaluation
            per_variant[variant].append({
                "held_out_line": line,
                "threshold": float(threshold),
                "metrics": metrics,
                "roc_auc_unchanged": fixed_auc,
                "inner_training_only_diagnostics": inner_detail if variant == "SourceInnerLOCO" else None,
                "source_train_score_median": float(np.median(train_scores)) if variant == "UnlabeledMedianAlign" else None,
                "unlabeled_target_score_median": float(np.median(target_scores)) if variant == "UnlabeledMedianAlign" else None,
                "predictions": [
                    {"gsm": gsm, "actual": str(label), "predicted": str(pred), "raw_DTP_logit": float(score)}
                    for gsm, label, pred, score in zip(test_ids, y[te], preds, target_scores)
                ],
            })
        print(f"{line}: AUC={fixed_auc:.3f}; BA fixed={per_variant['FixedZero'][-1]['metrics']['balanced_accuracy']:.3f}, "
              f"source={per_variant['SourceInnerLOCO'][-1]['metrics']['balanced_accuracy']:.3f}, "
              f"unlabeled={per_variant['UnlabeledMedianAlign'][-1]['metrics']['balanced_accuracy']:.3f}; "
              f"t(source)={source_threshold:+.3f}, t(unlabeled)={target_threshold:+.3f}")
    summary = {}
    for variant in VARIANTS:
        records = per_variant[variant]
        rows = [p for f in records for p in f["predictions"]]
        if len(rows) != 24 or len({p["gsm"] for p in rows}) != 24:
            raise ValueError("Expected 24 unique held-out predictions per variant")
        pooled = stage35.metrics(np.array([p["actual"] for p in rows]),
                                 np.array([p["predicted"] for p in rows]))
        summary[variant] = {
            "pooled": pooled,
            "mean_outer_balanced_accuracy": float(np.mean([f["metrics"]["balanced_accuracy"] for f in records])),
            "worst_line_balanced_accuracy": float(min(f["metrics"]["balanced_accuracy"] for f in records)),
        }
        print(f"{variant}: BA={pooled['balanced_accuracy']:.3f}, DTP recall={pooled['dtp_recall']:.3f}, "
              f"worst={summary[variant]['worst_line_balanced_accuracy']:.3f}")
    report = {
        "analysis": "source_only_and_unlabeled_transductive_threshold_development",
        "source": "GSE193258",
        "original_protocol_sha256": stage35.sha256(root / STAGE31_REPORT),
        "reference_stage35_report_sha256": stage35.sha256(root / STAGE35_REPORT),
        "reference_score_audit_sha256": stage35.sha256(root / SCORE_REPORT),
        "variants": {
            "FixedZero": "Original LR threshold at score zero (inductive)",
            "SourceInnerLOCO": "One fixed threshold selected from inner source-line CV scores and labels (inductive)",
            "UnlabeledMedianAlign": "Median target-minus-source score shift; full unlabeled target batch required (transductive)",
        },
        "parameter_search_on_outer_test": False,
        "outer_test_labels_used_for_threshold_selection": False,
        "unlabeled_target_batch_used_in_one_variant": True,
        "class_proportions_must_be_similar_for_median_alignment": True,
        "same_frozen_classifier_and_feature_selection": True,
        "historically_inspected_folds": True,
        "independent_validation": False,
        "limitations": [
            "Six test samples per cell line (3 Acute, 3 DTP); ROC-AUC and BA have substantial uncertainty.",
            "The target-median method uses the entire unlabeled target batch and is not an inductive single-sample classifier.",
            "An equal-class test design favors location matching; performance can collapse when target class mix differs.",
            "Cross-fit inner scores come from 2-line fits, final scores from 3-line fits; thresholds need not transfer.",
            "All four held-out line results had already been examined before this study; this is developmental.",
            "Acute (24h) versus DTP (21d) is confounded with exposure duration.",
            "Threshold adaptation, median alignment, and training-only CV are known methods; no novelty claimed.",
        ],
        "summary": summary,
        "folds_by_variant": per_variant,
    }
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", dest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.self_test and args.check_only:
        parser.error("Choose only one of --self-test and --check-only")
    if args.self_test:
        self_test()
    else:
        run(args.root.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
