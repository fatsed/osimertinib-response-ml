from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("data/processed/cross_study")
BASELINE = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/gse255958_pc9_excluded_transfer.json")
STATES = ["Control", "Acute", "DTP"]
MODEL_NAMES = ("Dummy", "LogisticRegression", "LinearSVM")
FILES = (
    "gse193258_common_log2tpm.tsv.gz",
    "gse255958_common_log2tpm_candidate.tsv.gz",
    "samples.csv",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_expression(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", compression="gzip", keep_default_na=False)
    if "gene_id" not in table.columns or table.columns.duplicated().any():
        raise ValueError(f"Invalid expression columns in {path}")
    ids = table.pop("gene_id").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Invalid gene IDs in {path}")
    matrix = table.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite expression in {path}")
    return pd.DataFrame(matrix, index=ids, columns=table.columns)


def load_inputs(root: Path):
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    if baseline.get("source") != "GSE193258" or baseline.get("target") != "GSE255958":
        raise ValueError("The stage-09 baseline is not the expected study pair")
    if baseline.get("supervised_feature_count") != 500 or baseline.get("fixed_seed") != 42:
        raise ValueError("Stage-09 configuration changed; do not compare unlike experiments")

    for name in FILES:
        path = root / INPUT / name
        if not path.is_file() or digest(path) != baseline["input_sha256"].get(name):
            raise ValueError(f"Missing or modified stage-09 input: {name}")

    source = read_expression(root / INPUT / FILES[0])
    target = read_expression(root / INPUT / FILES[1])
    if not source.index.equals(target.index) or len(source) != baseline.get("gene_count"):
        raise ValueError("Source and target gene identities/orders do not match stage 09")

    metadata = pd.read_csv(root / INPUT / FILES[2], dtype=str, keep_default_na=False)
    needed = {"accession", "gsm", "cell_line", "state", "evaluation_role"}
    if not needed.issubset(metadata.columns) or metadata["gsm"].duplicated().any():
        raise ValueError("Invalid sample metadata")
    train_info = metadata.loc[metadata["accession"] == "GSE193258"].copy()
    target_info = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & metadata["evaluation_role"].eq("target_candidate")
    ].copy()
    excluded = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & metadata["evaluation_role"].eq("exploratory_only")
    ]
    if len(train_info) != 36 or len(target_info) != 9 or len(excluded) != 3:
        raise ValueError("Unexpected study sample counts")
    if not excluded["state"].eq("Resistant").all():
        raise ValueError("Target excluded samples are not all resistant")
    if (train_info.groupby(["cell_line", "state"]).size() != 3).any():
        raise ValueError("Unequal source cell-line/state groups")
    if train_info["cell_line"].nunique() != 4 or set(train_info["state"]) != set(STATES):
        raise ValueError("Unexpected source cell lines or classes")
    if target_info["state"].value_counts().to_dict() != {s: 3 for s in STATES}:
        raise ValueError("Unexpected target state groups")
    if not target_info["cell_line"].eq("PC9").all():
        raise ValueError("The target cohort is not PC9 only")
    if set(source.columns) != set(train_info["gsm"]):
        raise ValueError("Source GSM matrix mismatch")
    if set(target.columns) != set(metadata.loc[metadata["accession"] == "GSE255958", "gsm"]):
        raise ValueError("Target GSM matrix mismatch")
    if set(source.columns) & set(target.columns):
        raise ValueError("GSM overlap across source and target studies")

    train_info = train_info.set_index("gsm").loc[list(source.columns)].reset_index()
    test_order = [gsm for gsm in target.columns if gsm in set(target_info["gsm"])]
    target_info = target_info.set_index("gsm").loc[test_order].reset_index()
    x_source = source.to_numpy(dtype=float).T
    x_target = target.loc[:, test_order].to_numpy(dtype=float).T
    source_mask = train_info["cell_line"].ne("PC9").to_numpy()
    if int(source_mask.sum()) != 27 or int((~source_mask).sum()) != 9:
        raise ValueError("Expected exactly 27 non-PC9 training and nine PC9 held-out samples")
    if set(train_info.loc[source_mask, "cell_line"]) != {"H1975", "HCC2935", "HCC827"}:
        raise ValueError("Non-PC9 training cell lines changed")
    if set(train_info.loc[~source_mask, "state"]) != set(STATES):
        raise ValueError("Incomplete held-out PC9 classes")
    return baseline, train_info, target_info, x_source, x_target, source_mask


def build_model(name: str):
    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    if name == "LogisticRegression":
        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)
    elif name == "LinearSVM":
        clf = LinearSVC(C=0.01, dual=True, max_iter=20000, random_state=42)
    else:
        raise ValueError(f"Unknown model {name}")
    return Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=500)),
        ("scale", StandardScaler()),
        ("classifier", clf),
    ])


def evaluation(y_true, pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=STATES, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=STATES).tolist(),
        "correct": int(np.count_nonzero(y_true == pred)),
        "total": int(len(y_true)),
    }


def individual_results(info: pd.DataFrame, preds) -> list[dict]:
    return [
        {"gsm": str(gsm), "true_state": str(actual), "predicted_state": str(pred),
         "correct": bool(actual == pred)}
        for gsm, actual, pred in zip(info["gsm"], info["state"], preds)
    ]


def check_stage09_fold(baseline: dict, name: str, predictions: list[dict]):
    folds = [f for f in baseline["models"][name]["loco_folds"]
             if f["held_out_cell_line"] == "PC9"]
    if len(folds) != 1:
        raise ValueError(f"Stage-09 PC9 fold missing or duplicated for {name}")
    expected = {p["gsm"]: (p["true_state"], p["predicted_state"])
                for p in folds[0]["predictions"]}
    actual = {p["gsm"]: (p["true_state"], p["predicted_state"])
              for p in predictions}
    if actual != expected:
        raise ValueError(f"Stage-09 held-out PC9 predictions not reproduced for {name}")


def run(root: Path) -> dict:
    baseline, train_info, external, x_source, x_target, train_mask = load_inputs(root)
    y_source = train_info["state"].to_numpy()
    y_external = external["state"].to_numpy()
    source_pc9 = train_info.loc[~train_mask].copy()
    results = {}

    for name in MODEL_NAMES:
        model = build_model(name)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            model.fit(x_source[train_mask], y_source[train_mask])
        source_predictions = model.predict(x_source[~train_mask])
        external_predictions = model.predict(x_target)
        pc9_rows = individual_results(source_pc9, source_predictions)
        check_stage09_fold(baseline, name, pc9_rows)
        results[name] = {
            "source_pc9_held_out": {
                "metrics": evaluation(y_source[~train_mask], source_predictions),
                "predictions": pc9_rows,
                "stage09_fold_reproduced": True,
            },
            "external_study_pc9": {
                "metrics": evaluation(y_external, external_predictions),
                "predictions": individual_results(external, external_predictions),
            },
            "previous_target_ba_with_pc9_training": baseline["models"][name]["target_exploratory"]["balanced_accuracy"],
        }
        a = results[name]["source_pc9_held_out"]["metrics"]
        b = results[name]["external_study_pc9"]["metrics"]
        print(f"{name}: held-out source PC9 {a['correct']}/9 (BA={a['balanced_accuracy']:.3f}); "
              f"external PC9 {b['correct']}/9 (BA={b['balanced_accuracy']:.3f})")

    report = {
        "analysis": "GSE255958_PC9_excluded_cross_study_transfer",
        "source": "GSE193258",
        "target": "GSE255958",
        "original_stage09_inputs_verified": True,
        "input_sha256": {name: baseline["input_sha256"][name] for name in FILES},
        "feature_selection_and_scaling": "trained on 27 non-PC9 source samples only",
        "training_cell_lines": sorted(train_info.loc[train_mask, "cell_line"].unique().tolist()),
        "excluded_from_training_cell_line": "PC9",
        "training_samples": 27,
        "held_out_source_pc9_samples": 9,
        "external_study_pc9_samples": 9,
        "target_used_for_training_or_tuning": False,
        "feature_count": 500,
        "seed": 42,
        "classes": STATES,
        "study_reuse_warning": "GSE255958 labels were already examined in earlier exploratory experiments; not an untouched prospective validation set",
        "limitations": [
            "The source PC9 fold is a replication of stage-09 LOCO, not a new independent test.",
            "GSE255958 was used earlier for evaluation; do not claim an untouched test set.",
            "Cell-line and study effects are both shifted relative to training; they cannot be separately attributed.",
            "Acute/DTP labels, time, dose, protocol and biological replicate independence require review.",
            "The two external training configurations differ in PC9 inclusion; comparing scores is exploratory, not a formal causal test.",
            "The source and target log2TPM scale alignment was provisional in stage 08.",
            "No model tuning, model ranking, calibration, or gene-set selection used target labels.",
        ],
        "models": results,
    }
    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {destination}")
    print("Exploratory transfer only; held-out study has previously been inspected.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
