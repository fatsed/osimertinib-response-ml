"""Exploratory four-line Acute/DTP ablation, using frozen Stage-35 source rows.

Do NOT use this script to select a method for an independent test or claim
novelty. The source folds, including PC9, have already been inspected.
No GSE243565 or other target-study expression is read.

Run from repository root:
    python -m pytest -q tests/test_contrast_pairwise.py
    python scripts/38_pairwise_development.py --check-only
    python scripts/38_pairwise_development.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_pairwise import ContrastPairwiseClassifier  # noqa: E402

REFERENCE_SCRIPT = Path("scripts/35_dtp_vs_acute_benchmark.py")
REFERENCE_REPORT = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
REPORT = Path("results/contrast_pairwise_development.json")

# Chosen for a *single, un-tuned development demonstration*, NOT optimized
# against the known held-out labels. L2=.1 mirrors earlier DCSL prototypes;
# alpha/beta=.2 and temperature=.5 mirror previous group-robust baselines.
FIXED = {"n_features": 500, "l2": 0.1, "alpha": 0.2, "beta": 0.2,
         "temperature": 0.5}

VARIANTS = {
    "PlainBCE_L2": {"alpha": 0.0, "pairwise": "none", "beta": 0.0},
    "PairwiseMean": {"alpha": FIXED["alpha"], "pairwise": "mean", "beta": 0.0},
    "PairwiseSoftWorst": {"alpha": FIXED["alpha"], "pairwise": "soft_worst", "beta": 0.0},
    "GroupBCE_SoftWorst": {"alpha": 0.0, "pairwise": "none", "beta": FIXED["beta"]},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage35(root: Path):
    script = root / REFERENCE_SCRIPT
    spec = importlib.util.spec_from_file_location("stage35_for_pairwise", script)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def new_feature_pipeline():
    """Fitted on outer training rows only, then frozen for the held-out line."""
    return Pipeline([
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("anova", SelectKBest(score_func=f_classif, k=FIXED["n_features"])),
        ("standardize", StandardScaler()),
    ])


def run(root: Path, check_only: bool = False):
    stage35 = load_stage35(root)
    # Re-use existing provenance and frozen-cohort guards instead of rewriting.
    stage35.run(root, check_only=True)
    old = json.loads((root / REFERENCE_REPORT).read_text(encoding="utf-8"))
    if old.get("analysis") != "stage35_binary_dtp_acute_source_only_loco_exploratory":
        raise ValueError("Unexpected Stage-35 baseline reference")
    stage34 = stage35.import_stage34(root)
    protocol = json.loads((root / stage35.STAGE31_REPORT).read_text(encoding="utf-8"))
    meta = pd.read_csv(root / stage34.METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(meta, protocol)
    metadata = meta.rename(columns={"candidate_group": "state"})
    folds = protocol["nested_loco_protocol"]["folds"]
    stage35.validate_binary_cohort(metadata, folds)
    if check_only:
        print("Preflight OK: four already-inspected outer folds; training-only feature selection")
        return None

    matrix = pd.read_csv(root / stage34.PREPARED, sep="\t", compression="gzip")
    _genes, x = stage34.validate_matrix(matrix, metadata.gsm.tolist())
    y = metadata.state.to_numpy(dtype=str)
    group = metadata.cell_line.to_numpy(dtype=str)
    idx = {str(g): i for i, g in enumerate(metadata.gsm)}
    models = ["LogisticRegression_Stage35", *VARIANTS]
    per_model = {m: [] for m in models}

    for fold in folds:
        train_ids, test_ids, _inner = stage35.select_frozen_rows(fold, metadata)
        tr = np.asarray([idx[g] for g in train_ids], dtype=int)
        te = np.asarray([idx[g] for g in test_ids], dtype=int)
        if set(group[tr]) & set(group[te]) or len(tr) != 18 or len(te) != 6:
            raise ValueError("Held-out cell line leakage or unexpected samples")

        # Frozen Stage-35 logistic baseline; compare its predictions against
        # the previously saved reference before running any prototype model.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            baseline = stage35.make_model("LogisticRegression").fit(x[tr], y[tr])
        base_pred = baseline.predict(x[te])
        previous_fold = next((f for f in old["folds_by_family"]["LogisticRegression"]
                              if f["held_out_line"] == fold["outer_test_cell_line"]), None)
        if previous_fold is None or [p["gsm"] for p in previous_fold["predictions"]] != test_ids or \
                [p["predicted"] for p in previous_fold["predictions"]] != base_pred.tolist():
            raise ValueError("Frozen Stage-35 baseline no longer reproduces")
        per_model["LogisticRegression_Stage35"].append({
            "held_out_line": fold["outer_test_cell_line"],
            "metrics": stage35.metrics(y[te], base_pred),
            "predictions": [dict(gsm=g, actual=str(a), predicted=str(b))
                            for g, a, b in zip(test_ids, y[te], base_pred)],
        })

        # No preprocess step ever reads the test row or its label. Pairwise
        # comparisons occur only within one TRAINING line.
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            pre = new_feature_pipeline().fit(x[tr], y[tr])
            z_train = pre.transform(x[tr])
            z_test = pre.transform(x[te])
        for name, settings in VARIANTS.items():
            model = ContrastPairwiseClassifier(
                l2=FIXED["l2"], temperature=FIXED["temperature"],
                max_iter=1500, **settings,
            ).fit(z_train, y[tr], group[tr])
            predicted = model.predict(z_test)
            per_model[name].append({
                "held_out_line": fold["outer_test_cell_line"],
                "metrics": stage35.metrics(y[te], predicted),
                "training_pairwise_risks": model.train_group_pairwise_risks_,
                "predictions": [dict(gsm=g, actual=str(a), predicted=str(b))
                                for g, a, b in zip(test_ids, y[te], predicted)],
            })
        print(f"Completed existing LOCO fold: {fold['outer_test_cell_line']}")

    summary = {}
    for name, records in per_model.items():
        preds = [p for row in records for p in row["predictions"]]
        if len(preds) != 24 or len({p["gsm"] for p in preds}) != 24:
            raise ValueError("Expected precisely 24 unique held-out predictions")
        summary[name] = {
            "pooled": stage35.metrics(np.asarray([p["actual"] for p in preds]),
                                      np.asarray([p["predicted"] for p in preds])),
            "per_line": {row["held_out_line"]: row["metrics"] for row in records},
            "worst_line_balanced_accuracy": min(row["metrics"]["balanced_accuracy"] for row in records),
        }
    report = {
        "analysis": "exploratory_previously_inspected_binary_source_loco",
        "purpose": "Objective and ablation comparison, not method selection or independent validation",
        "fixed_settings_no_search": FIXED,
        "variant_definitions": VARIANTS,
        "source": "GSE193258",
        "n_binary_source_samples": 24,
        "previous_stage35_report_sha256": sha256(root / REFERENCE_REPORT),
        "target_expression_or_labels_read": False,
        "previously_inspected_folds": True,
        "not_independent_validation": True,
        "no_test_based_tuning": True,
        "limitations": [
            "Existing PC9 and other outer-fold outcomes were already inspected before this method was proposed.",
            "Hyperparameters are illustrative and NOT a fair model-family search or proof of superiority.",
            "Only four cell lines, with 3 biological-label replicates per arm and line.",
            "Within-line pairwise samples share replicates; do not count pairs as independent observations.",
            "Acute=24 h versus DTP=21 d; exposure time remains confounded with treatment state.",
            "Group robustness and RankNet-style losses are pre-existing techniques.",
            "No external dataset or new cell line was used.",
        ],
        "summary": summary,
        "folds_by_model": per_model,
    }
    output = root / REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", output)
    for name, stats in summary.items():
        m = stats["pooled"]
        print(f"{name}: balanced_accuracy={m['balanced_accuracy']:.3f}; "
              f"DTP_recall={m['dtp_recall']:.3f}; "
              f"worst_line_BA={stats['worst_line_balanced_accuracy']:.3f}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
