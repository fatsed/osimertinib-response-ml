"""Stage 35 — fixed binary DTP-versus-Acute, source-only LOCO benchmark.

This is exploratory analysis on the already-inspected GSE193258 source data.
The 24h Acute and 21d DTP arms differ in exposure duration: no duration-
independent or causal DTP biology is established by this classifier.

Run:
    python scripts/35_dtp_vs_acute_benchmark.py
    python scripts/35_dtp_vs_acute_benchmark.py --check-only
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
STAGE34_SCRIPT = Path("scripts/34_source_only_rebaseline.py")
STAGE34_REPORT = Path("results/dcsl_stage34_source_only_nested_loco.json")
STAGE31_REPORT = Path("results/dcsl_stage31_evaluation_protocol.json")
OUTPUT = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
STATES = ("Acute", "DTP")
FAMILIES = ("Dummy", "LogisticRegression", "LinearSVM")
SEED = 42
N_FEATURES = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def import_stage34(root: Path):
    script = root / STAGE34_SCRIPT
    spec = importlib.util.spec_from_file_location("stage34_for_stage35", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_binary_cohort(meta: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """The binary experiment must be an exact subset of Stage 31's source GSMs."""
    if not {"gsm", "cell_line", "state"}.issubset(meta) or meta.gsm.duplicated().any():
        raise ValueError("Invalid source metadata")
    if len(folds) != 4 or len({f["outer_test_cell_line"] for f in folds}) != 4:
        raise ValueError("Expected four distinct frozen outer folds")
    expected = {gsm for f in folds for gsm in f["outer_test_gsm"]}
    if len(expected) != 36 or set(meta.gsm) != expected:
        raise ValueError("Stage-31 source GSM identity mismatch")
    subset = meta.loc[meta.state.isin(STATES)].copy()
    if len(subset) != 24 or set(subset.state) != set(STATES):
        raise ValueError("Expected 24 Acute/DTP samples")
    if set(subset.cell_line) != {f["outer_test_cell_line"] for f in folds}:
        raise ValueError("Cell-line identities do not match frozen folds")
    count = subset.groupby(["cell_line", "state"]).size()
    if len(count) != 8 or not count.eq(3).all():
        raise ValueError("Need exactly 3 Acute and 3 DTP samples per cell line")
    return subset


def select_frozen_rows(fold: dict, metadata: pd.DataFrame) -> tuple[list[str], list[str], list[tuple[list[str], list[str]]]]:
    lookup = metadata.set_index("gsm")["state"].to_dict()
    def keep(ids):
        if any(g not in lookup for g in ids):
            raise ValueError("GSM absent from source-only metadata")
        return [g for g in ids if lookup[g] in STATES]
    train, test = keep(fold["outer_train_gsm"]), keep(fold["outer_test_gsm"])
    if len(train) != 18 or len(test) != 6 or set(train) & set(test):
        raise ValueError("Invalid binary outer split")
    inners = [(keep(f["train_gsm"]), keep(f["validation_gsm"])) for f in fold["inner_folds"]]
    if len(inners) != 3:
        raise ValueError("Expected exactly three inner folds")
    for a, b in inners:
        if len(a) != 12 or len(b) != 6 or set(a) & set(b) or (set(a) | set(b)) != set(train):
            raise ValueError("Inner-fold overlap, missing rows or outer-test leakage")
    return train, test, inners


def make_model(family: str):
    if family == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "LogisticRegression":
        classifier = LogisticRegression(C=0.01, solver="lbfgs", max_iter=3000, random_state=SEED)
    elif family == "LinearSVM":
        classifier = LinearSVC(C=0.001, max_iter=20000, dual=True, random_state=SEED)
    else:
        raise ValueError("Unknown frozen baseline")
    return Pipeline([
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("anova", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("standardize", StandardScaler()),
        ("classifier", classifier),
    ])


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, dtype=str), np.asarray(pred, dtype=str)
    if len(y) != len(pred) or not len(y) or set(y) != set(STATES) or not set(pred).issubset(STATES):
        raise ValueError("Binary scores require both true states and valid predictions")
    return {
        "n": len(y),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=STATES, average="macro", zero_division=0)),
        "dtp_recall": float(recall_score(y, pred, pos_label="DTP")),
        "acute_recall": float(recall_score(y, pred, pos_label="Acute")),
        "confusion_matrix_Acute_DTP": confusion_matrix(y, pred, labels=STATES).tolist(),
    }


def evaluate(x: np.ndarray, metadata: pd.DataFrame, folds: list[dict]) -> dict:
    if x.ndim != 2 or len(x) != len(metadata) or not np.isfinite(x).all():
        raise ValueError("Invalid source-only expression matrix")
    validate_binary_cohort(metadata, folds)
    idx = {str(gsm): i for i, gsm in enumerate(metadata.gsm)}
    y = metadata.state.to_numpy(dtype=str)
    result = {k: [] for k in FAMILIES}
    for fold in folds:
        tr, te, inner = select_frozen_rows(fold, metadata)
        for family in FAMILIES:
            inner_results = []
            for train_ids, val_ids in inner:
                a = np.array([idx[g] for g in train_ids])
                b = np.array([idx[g] for g in val_ids])
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    clf = make_model(family).fit(x[a], y[a])
                inner_results.append(metrics(y[b], clf.predict(x[b])))
            a = np.array([idx[g] for g in tr])
            b = np.array([idx[g] for g in te])
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                final = make_model(family).fit(x[a], y[a])
            pred = final.predict(x[b])
            result[family].append({
                "held_out_line": fold["outer_test_cell_line"],
                "train_n": len(a), "test_n": len(b),
                "inner_mean_balanced_accuracy_diagnostic_only": float(np.mean([m["balanced_accuracy"] for m in inner_results])),
                "outer_metrics": metrics(y[b], pred),
                "predictions": [{"gsm": gsm, "actual": str(yt), "predicted": str(yp)}
                                for gsm, yt, yp in zip(te, y[b], pred)],
            })
        print(f"Finished binary LOCO: {fold['outer_test_cell_line']}")
    summary = {}
    for family in FAMILIES:
        rows = [p for fold in result[family] for p in fold["predictions"]]
        if len(rows) != 24 or len({r["gsm"] for r in rows}) != 24:
            raise ValueError("Expected 24 unique outer predictions")
        pooled = metrics(np.asarray([r["actual"] for r in rows]), np.asarray([r["predicted"] for r in rows]))
        summary[family] = {
            "pooled": pooled,
            "mean_outer_balanced_accuracy": float(np.mean([f["outer_metrics"]["balanced_accuracy"] for f in result[family]])),
            "mean_outer_dtp_recall": float(np.mean([f["outer_metrics"]["dtp_recall"] for f in result[family]])),
            "per_line": {f["held_out_line"]: f["outer_metrics"] for f in result[family]},
        }
    return {"folds_by_family": result, "summary": summary}


def run(root: Path, check_only: bool = False) -> dict:
    stage34 = import_stage34(root)
    previous = json.loads((root / STAGE34_REPORT).read_text(encoding="utf-8"))
    protocol = json.loads((root / STAGE31_REPORT).read_text(encoding="utf-8"))
    if (previous.get("check_only") is not False or
            previous.get("comparison_status") != "SENSITIVITY_ANALYSIS_NOT_NEW_VALIDATION_OR_MODEL_TUNING" or
            previous.get("target_expression_or_target_labels_read") is not False or
            previous.get("n_source_only_genes") != 19712 or
            previous.get("frozen_stage31_sha256") != sha256(root / STAGE31_REPORT)):
        raise ValueError("Stage-34 full source-only report or frozen protocol mismatch")
    provenance = stage34.verify_stage05_qc(root, stage34.import_stage32(root))
    if provenance != previous.get("provenance"):
        raise ValueError("Source-only input hashes changed since Stage 34")
    meta_original = pd.read_csv(root / stage34.METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(meta_original, protocol)
    metadata = meta_original.rename(columns={"candidate_group": "state"})
    folds = protocol["nested_loco_protocol"]["folds"]
    validate_binary_cohort(metadata, folds)
    output = {
        "analysis": "stage35_binary_dtp_acute_source_only_loco_exploratory",
        "source": "GSE193258",
        "n_source_samples": 36,
        "n_binary_samples": 24,
        "features_source_only": 19712,
        "fixed_models": {"LogisticRegression": {"C": 0.01, "k": 500},
                         "LinearSVM": {"C": 0.001, "k": 500},
                         "Dummy": {"strategy": "most_frequent"}},
        "source_provenance": provenance,
        "stage34_report_sha256": sha256(root / STAGE34_REPORT),
        "frozen_stage31_sha256": sha256(root / STAGE31_REPORT),
        "target_study_read": False,
        "reusing_historically_inspected_folds": True,
        "not_independent_validation": True,
        "treatment_duration_confounding_unresolved": True,
        "new_hyperparameter_search_or_test_based_selection": False,
        "limitations": [
            "Acute=24 h and DTP=21 d in this cohort: exposure duration is confounded with class.",
            "Source labels remain provisional; biological replication requires provenance review.",
            "Only four previously examined cell lines, three replicates per class/line.",
            "Parameters were inherited from prior three-class development, not prospectively registered.",
            "No new independent target cohort was evaluated.",
        ],
    }
    if not check_only:
        frame = pd.read_csv(root / stage34.PREPARED, sep="\t", compression="gzip")
        _genes, x = stage34.validate_matrix(frame, metadata.gsm.tolist())
        output.update(evaluate(x, metadata, folds))
        (root / OUTPUT).parent.mkdir(parents=True, exist_ok=True)
        (root / OUTPUT).write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Saved:", root / OUTPUT)
        for name, values in output["summary"].items():
            print(f"{name}: BA={values['pooled']['balanced_accuracy']:.3f}, DTP recall={values['pooled']['dtp_recall']:.3f}")
    else:
        print("Stage 35 preflight OK: 24 source-only Acute/DTP samples, 4 frozen cell-line folds")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), args.check_only)


if __name__ == "__main__":
    main()
