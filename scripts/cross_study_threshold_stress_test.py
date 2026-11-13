"""Exploratory GSE193258 -> GSE255958 Acute/DTP threshold-transfer stress test.

Run from the existing repository root:
  python scripts/cross_study_threshold_stress_test.py --self-test
  python scripts/cross_study_threshold_stress_test.py --check-only
  python scripts/cross_study_threshold_stress_test.py

Uses existing Stage-08 common-gene matrices and Stage-09 checksum manifest.
Never downloads, changes, or overwrites previous project files.

PREDECLARED development settings (GSE255958 was previously inspected):
  A. Train on three source lines, excluding PC9 (18 Acute/DTP samples).
  B. Train on all four source lines, including PC9 (24 samples).
  - Stage-35 Logistic Regression hyperparameters: C=0.01, ANOVA k=500,
    variance filter and scaler fitted only on source-training observations.
  - Threshold 0 (inductive); inner-cell-line source-CV threshold (inductive);
    unlabeled-median alignment (transductive; needs full target batch).
  - A separate post-hoc, label-defined subset sensitivity study enumerates
    all nonempty Acute/DTP subsets: it is NOT an independent test.

CAVEATS: The shared 18,013-gene feature universe was formed using gene IDs
from the target study in Stage 08. Source / target log2TPM transformations,
treatment timing and doses are not independently harmonized. The target six
samples were selected by GEO group metadata, 3 per state. Median alignment
uses no labels WITHIN this selected batch, but relies on cohort composition.
No paper-level independent generalization, clinical use, or novelty claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from osimertinib_ml.models.baselines import make_binary_model
from osimertinib_ml.evaluation.metrics import binary_metrics

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = Path("data/processed/cross_study")
BASELINE_REPORT = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/cross_study_threshold_stress_test.json")
INPUT_NAMES = (
    "gse193258_common_log2tpm.tsv.gz",
    "gse255958_common_log2tpm_candidate.tsv.gz",
    "samples.csv",
)
# Fixed before fitting. Excludes Control/Resistant from the six-sample batch.
TARGET_GSMS = tuple(f"GSM{i}" for i in range(8083377, 8083383))
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
STATES = ("Acute", "DTP")
VARIANTS = ("FixedZero", "SourceInnerLOCO", "UnlabeledMedianAlign")
SETTINGS = ("PC9_excluded", "PC9_included")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_existing(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_threshold_from_cv(score_folds: list[np.ndarray], label_folds: list[np.ndarray]):
    """Original Stage-38/threshold-audit tie rule, generalized to 3 or 4 source lines.

    No target samples or target labels are accepted by this function.
    """
    if len(score_folds) not in (3, 4) or len(label_folds) != len(score_folds):
        raise ValueError("Expected 3 or 4 source-cell-line validation folds")
    scores = []
    labels = []
    for raw_s, raw_y in zip(score_folds, label_folds):
        s, y = np.asarray(raw_s, dtype=float), np.asarray(raw_y, dtype=str)
        if s.ndim != 1 or s.shape != y.shape or set(y) != set(STATES) or not np.isfinite(s).all():
            raise ValueError("Invalid source CV scores or labels")
        scores.append(s)
        labels.append(y)
    unique = np.unique(np.concatenate(scores))
    candidates = np.unique(np.concatenate(([0.0], unique, (unique[:-1] + unique[1:]) / 2)))
    best_t, best_ba = 0.0, -1.0
    for t in candidates:
        score = float(np.mean([
            balanced_accuracy_score(y, np.where(s >= t, "DTP", "Acute"))
            for s, y in zip(scores, labels)
        ]))
        if score > best_ba + 1e-12 or (abs(score - best_ba) <= 1e-12 and
                (abs(float(t)), float(t)) < (abs(best_t), best_t)):
            best_t, best_ba = float(t), score
    return best_t, {"source_cv_folds": len(scores), "candidate_thresholds": int(len(candidates)),
                    "inner_mean_balanced_accuracy": best_ba}


def threshold_from_unlabeled(source_scores: np.ndarray, target_scores: np.ndarray) -> float:
    """Batch-transductive location alignment, using ONLY scores, never labels."""
    a = np.asarray(source_scores, dtype=float)
    b = np.asarray(target_scores, dtype=float)
    if (a.ndim != 1 or b.ndim != 1 or len(a) < 2 or len(b) < 2
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Expected two finite score batches with >= 2 samples each")
    return float(np.median(b) - np.median(a))


def predict(scores: np.ndarray, t: float) -> np.ndarray:
    a = np.asarray(scores, dtype=float)
    if a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(t):
        raise ValueError("Nonfinite scores or threshold")
    return np.where(a >= t, "DTP", "Acute")


def fit_model(x: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        model = make_binary_model("LogisticRegression")
        model.fit(x, y)
    if model.named_steps["classifier"].classes_.tolist() != list(STATES):
        raise ValueError("Expected decision scores positive for DTP")
    if model.named_steps["anova"].k != 500:
        raise ValueError("Unexpected feature count")
    return model


def scores(model, x: np.ndarray) -> np.ndarray:
    out = np.asarray(model.decision_function(x), dtype=float)
    if out.shape != (len(x),) or not np.isfinite(out).all():
        raise ValueError("Malformed fitted model scores")
    if not np.array_equal(model.predict(x), predict(out, 0.0)):
        raise ValueError("Score orientation does not match model.predict")
    return out


def posthoc_subsets(source_scores: np.ndarray, target_scores: np.ndarray,
                    labels: np.ndarray, fixed_threshold: float) -> dict:
    """Exhaustive, LABEL-SELECTED sensitivity cohorts, for analysis only.

    All 3 Acute and 3 DTP are used in the full experiment. In this separate
    descriptive sensitivity analysis, labels select subsets; they NEVER enter
    the median estimator, which sees only the chosen sample score vector.
    """
    a = np.flatnonzero(labels == "Acute")
    d = np.flatnonzero(labels == "DTP")
    if len(a) != 3 or len(d) != 3:
        raise ValueError("Sensitivity expects 3 Acute and 3 DTP samples")
    results = defaultdict(list)
    for na in (1, 2, 3):
        for nd in (1, 2, 3):
            for ai in combinations(a, na):
                for di in combinations(d, nd):
                    chosen = np.asarray((*ai, *di), dtype=int)
                    score = target_scores[chosen]
                    y = labels[chosen]
                    med_t = threshold_from_unlabeled(source_scores, score)
                    median_ba = float(balanced_accuracy_score(y, predict(score, med_t)))
                    fixed_ba = float(balanced_accuracy_score(y, predict(score, fixed_threshold)))
                    results[(na, nd)].append((median_ba, fixed_ba))
    summary = []
    for (na, nd), b in sorted(results.items()):
        arr = np.asarray(b, dtype=float)
        summary.append({"acute_n": na, "dtp_n": nd, "subsets": len(arr),
                        "median_align_ba_mean": float(arr[:, 0].mean()),
                        "median_align_ba_min": float(arr[:, 0].min()),
                        "median_align_ba_max": float(arr[:, 0].max()),
                        "fixed_zero_ba_mean": float(arr[:, 1].mean())})
    if sum(z["subsets"] for z in summary) != 49:
        raise AssertionError("Expected all 49 nonempty class-mixture subsets")
    return {"selection_uses_target_labels": True, "for_posthoc_sensitivity_only": True,
            "all_subsets_count": 49, "by_class_counts": summary}


def load_checked_inputs(root: Path):
    prior_file = root / BASELINE_REPORT
    baseline = json.loads(prior_file.read_text(encoding="utf-8"))
    expected = baseline.get("input_sha256", {})
    if baseline.get("source") != "GSE193258" or baseline.get("target") != "GSE255958":
        raise ValueError("Old Stage-09 study labels unexpectedly changed")
    digests = {}
    for name in INPUT_NAMES:
        path = root / INPUT_DIR / name
        got = sha256(path)
        if got != expected.get(name):
            raise ValueError(f"Stage-09 input SHA256 mismatch: {name}. Do not overwrite old matrices")
        digests[name] = got
    stage09 = import_existing(root / "scripts/09_train_baselines.py", "stage09_for_cross_study_stress")
    x_source, source_meta, x_target9, target_meta, n_genes = stage09.load_data(root)
    if n_genes != 18013 or x_source.shape != (36, n_genes) or x_target9.shape != (9, n_genes):
        raise ValueError("Unexpected Stage-08 aligned expression matrix dimensions")
    if set(source_meta.cell_line) != set(LINES) or not source_meta.groupby(["cell_line", "state"]).size().eq(3).all():
        raise ValueError("Source cell-line/state replication mismatch")
    if set(TARGET_GSMS) - set(target_meta.gsm):
        raise ValueError("Expected six GSE255958 target GSMs missing")
    target_pos = {gsm: i for i, gsm in enumerate(target_meta.gsm)}
    target_indices = [target_pos[g] for g in TARGET_GSMS]
    y_target = target_meta.iloc[target_indices].state.to_numpy(dtype=str)
    if np.count_nonzero(y_target == "Acute") != 3 or np.count_nonzero(y_target == "DTP") != 3:
        raise ValueError("Fixed target GSMs no longer map to three Acute and three DTP")
    if not target_meta.iloc[target_indices].cell_line.eq("PC9").all():
        raise ValueError("Non-PC9 target samples")
    source_mask = source_meta.state.isin(STATES).to_numpy()
    if int(source_mask.sum()) != 24:
        raise ValueError("Source binary sample count changed")
    return (x_source[source_mask], source_meta.loc[source_mask].reset_index(drop=True),
            x_target9[target_indices], y_target, digests, sha256(prior_file))


def run(root: Path, check_only: bool = False) -> dict:
    x_source, meta, x_target, y_target, checksums, prior_sha = load_checked_inputs(root)
    if check_only:
        print("Preflight OK: Stage-09 checksums, 18,013 aligned genes, 24 source binary, 6 target binary")
        print("Train source only: 18 (no PC9) or 24 (with PC9); no external labels for fitting")
        print("Target batch has 3 Acute and 3 DTP by GEO design; alignment uses no class labels")
        return {"check_only": True}
    y_source = meta.state.to_numpy(dtype=str)
    cell_lines = meta.cell_line.to_numpy(dtype=str)
    result = {}
    for setting in SETTINGS:
        train_mask = cell_lines != "PC9" if setting == "PC9_excluded" else np.ones(len(meta), dtype=bool)
        x_train, y_train, line_train = x_source[train_mask], y_source[train_mask], cell_lines[train_mask]
        n_expected = 18 if setting == "PC9_excluded" else 24
        if len(x_train) != n_expected or len(set(line_train)) != (3 if n_expected == 18 else 4):
            raise ValueError("Unexpected source training lines or samples")
        model = fit_model(x_train, y_train)
        train_scores, target_scores = scores(model, x_train), scores(model, x_target)
        inner_scores, inner_labels = [], []
        for heldout in sorted(np.unique(line_train)):
            train = line_train != heldout
            val = ~train
            if len(set(line_train[train])) != len(set(line_train)) - 1:
                raise ValueError("Invalid source inner domain split")
            inner_model = fit_model(x_train[train], y_train[train])
            inner_scores.append(scores(inner_model, x_train[val]))
            inner_labels.append(y_train[val])
        source_threshold, inner_cv = source_threshold_from_cv(inner_scores, inner_labels)
        # No external label is used in either of these threshold estimators.
        unlabeled_threshold = threshold_from_unlabeled(train_scores, target_scores)
        thresholds = {"FixedZero": 0.0, "SourceInnerLOCO": source_threshold,
                      "UnlabeledMedianAlign": unlabeled_threshold}
        y_score_auc = float(roc_auc_score(y_target == "DTP", target_scores))
        variants = {}
        for name, t in thresholds.items():
            guessed = predict(target_scores, t)
            variants[name] = {"threshold": float(t), "metrics": binary_metrics(y_target, guessed),
                              "predictions": [{"gsm": gsm, "actual": str(y), "predicted": str(p),
                                               "raw_DTP_logit": float(s)}
                                              for gsm, y, p, s in zip(TARGET_GSMS, y_target, guessed, target_scores)]}
        result[setting] = {
            "source_training_n": len(x_train),
            "source_training_lines": sorted(set(line_train)),
            "target_n": len(y_target), "external_target_cell_line": "PC9",
            "roc_auc_common_to_all_thresholds": y_score_auc,
            "source_inner_validation": inner_cv,
            "source_train_score_median": float(np.median(train_scores)),
            "target_unlabeled_score_median": float(np.median(target_scores)),
            "variants": variants,
            "posthoc_sensitivity": posthoc_subsets(train_scores, target_scores, y_target, 0.0),
        }
        b = {name: variants[name]["metrics"]["balanced_accuracy"] for name in VARIANTS}
        print(f"{setting}: ROC-AUC={y_score_auc:.3f}; BA fixed={b['FixedZero']:.3f}, "
              f"source={b['SourceInnerLOCO']:.3f}, unlabeled={b['UnlabeledMedianAlign']:.3f}")
    report = {
        "analysis": "exploratory_cross_study_threshold_transfer_not_independent_validation",
        "source": "GSE193258", "target": "GSE255958", "n_aligned_genes": x_source.shape[1],
        "stage09_common_gene_input_sha256": checksums,
        "stage09_reference_report_sha256": prior_sha,
        "training_settings": {"PC9_excluded": "18 source Acute/DTP from 3 non-PC9 lines",
                              "PC9_included": "24 source Acute/DTP from all 4 lines"},
        "model_settings": {"family": "LogisticRegression", "C": 0.01, "k": 500,
                           "variance_threshold": 1e-8, "solver": "lbfgs", "seed": 42},
        "variants": {"FixedZero": "no adaptation, threshold=0",
                     "SourceInnerLOCO": "source-inner-cell-line-CV threshold; 3 folds if PC9 excluded, 4 if included",
                     "UnlabeledMedianAlign": "median(target batch scores) - median(training scores), no target labels"},
        "test_labels_used_for_fitting_feature_selection_or_threshold": False,
        "target_cohort_selected_by_known_GEO_state_GSMs": True,
        "target_batch_balanced_by_study_design": True,
        "median_alignment_requires_entire_unlabeled_batch": True,
        "posthoc_subsets_selected_using_target_labels_only_for_sensitivity": True,
        "independent_validation": False,
        "unseen_target_study": False,
        "limitations": [
            "GSE255958 has been inspected in previous stages; all results are developmental.",
            "Source and target share 18,013 genes determined via target-informed gene ID intersection in Stage 08.",
            "Existing Stage-08 scales are provisional (GEO source log2TPM vs log2(RSEM TPM + 0.01) target).",
            "Not the identical 19,712-gene source-only feature universe used in Stage 35.",
            "Acute and DTP durations/doses differ within and between the source and target studies.",
            "Six target observations (three per state) yield very uncertain BA and ROC-AUC.",
            "Batch-transductive median alignment benefits from balanced selected class composition.",
            "Sensitivity subsets are chosen post hoc with labels and do not represent new validation cohorts.",
            "No new algorithm novelty, clinical use, or independent generalization demonstrated.",
        ],
        "settings_results": result,
    }
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Saved:", dest)
    return report


def self_test() -> None:
    # Toy score ranking: full-batch transductive shift works, but changing the
    # target mixture can change the estimated threshold without any labels used.
    base = np.asarray([-3., -2., 2., 3.])
    shifted = base - 4.0
    assert np.isclose(threshold_from_unlabeled(base, shifted), -4.0)
    assert predict(shifted, -4.0).tolist() == ["Acute", "Acute", "DTP", "DTP"]
    a = [np.asarray([-2., -1., 1., 2.])] * 3
    y = [np.asarray(["Acute", "Acute", "DTP", "DTP"])] * 3
    t, summary = source_threshold_from_cv(a, y)
    assert t == 0.0 and summary["inner_mean_balanced_accuracy"] == 1.0
    a.append(np.asarray([-2., -1., 1., 2.])); y.append(y[0])
    t, _ = source_threshold_from_cv(a, y)
    assert t == 0.0
    toy_scores = np.asarray([-6., -5., -4., -1.3, -1.1, -1.])
    toy_labels = np.asarray(["Acute"] * 3 + ["DTP"] * 3)
    sub = posthoc_subsets(np.asarray([-3., -1., 1., 3.]), toy_scores, toy_labels, 0.0)
    assert sub["all_subsets_count"] == 49
    # Ensure the original Stage-35 factory signature is preserved.
    assert make_binary_model("LogisticRegression").named_steps["anova"].k == 500
    print("Self-test OK: threshold calculations, 3/4 source CV folds, 49 sensitivity subsets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.check_only and args.self_test:
        parser.error("Use only one of --self-test or --check-only")
    if args.self_test:
        self_test()
    else:
        run(args.root.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
