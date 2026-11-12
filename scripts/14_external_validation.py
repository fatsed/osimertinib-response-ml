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
OUTPUT = Path("results/gse255958_external_diagnostics.json")
STATES = ("Control", "Acute", "DTP")
FILES = (
    "gse193258_common_log2tpm.tsv.gz",
    "gse255958_common_log2tpm_candidate.tsv.gz",
    "samples.csv",
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_matrix(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", compression="gzip", keep_default_na=False)
    if "gene_id" not in table.columns or table.columns.duplicated().any():
        raise ValueError(f"Missing gene_id or duplicate sample columns: {path.name}")
    ids = table.pop("gene_id").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"Invalid gene IDs: {path.name}")
    values = table.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite expression values: {path.name}")
    return pd.DataFrame(values, index=ids, columns=table.columns)


def load_inputs(root: Path):
    previous = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    for name in FILES:
        path = root / INPUT / name
        if checksum(path) != previous["input_sha256"].get(name):
            raise ValueError(f"Stage-09 input hash mismatch: {name}")

    source = read_matrix(root / INPUT / FILES[0])
    target = read_matrix(root / INPUT / FILES[1])
    if not source.index.equals(target.index) or len(source) != 18013:
        raise ValueError("Source and target genes or gene ordering differ")

    metadata = pd.read_csv(root / INPUT / FILES[2], dtype=str, keep_default_na=False)
    required = {"accession", "gsm", "cell_line", "state", "treatment", "evaluation_role", "label_review"}
    if not required.issubset(metadata.columns) or metadata["gsm"].duplicated().any():
        raise ValueError("Missing metadata fields or duplicated GSM IDs")
    train = metadata.loc[metadata["accession"] == "GSE193258"].copy()
    test = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & (metadata["evaluation_role"] == "target_candidate")
    ].copy()
    excluded = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & (metadata["evaluation_role"] == "exploratory_only")
    ].copy()
    if len(train) != 36 or len(test) != 9 or len(excluded) != 3:
        raise ValueError("Unexpected source, target or excluded sample counts")
    if train.groupby(["cell_line", "state"]).size().to_dict() != {
        (line, state): 3
        for line in ("PC9", "H1975", "HCC827", "HCC2935")
        for state in STATES
    }:
        raise ValueError("Unexpected source cohort design")
    if test["state"].value_counts().to_dict() != {state: 3 for state in STATES}:
        raise ValueError("Unexpected target state counts")
    if not test["cell_line"].eq("PC9").all() or not excluded["state"].eq("Resistant").all():
        raise ValueError("Target study inclusion/exclusion mismatch")
    if set(source.columns) != set(train["gsm"]) or set(target.columns) != set(
        metadata.loc[metadata["accession"] == "GSE255958", "gsm"]
    ):
        raise ValueError("Matrix sample columns do not match metadata")
    if set(source.columns) & set(target.columns):
        raise ValueError("Overlapping GSM IDs across studies")

    train = train.set_index("gsm").loc[list(source.columns)].rename_axis("gsm").reset_index()
    test = test.set_index("gsm").loc[
        [gsm for gsm in target.columns if gsm in set(test["gsm"])]
    ].rename_axis("gsm").reset_index()
    x_train = source.to_numpy(dtype=float).T
    x_target = target[test["gsm"].tolist()].to_numpy(dtype=float).T
    return previous, train, test, excluded, x_train, x_target


def make_model(name: str):
    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    if name == "LogisticRegression":
        classifier = LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)
    elif name == "LinearSVM":
        classifier = LinearSVC(C=0.01, dual=True, max_iter=20000, random_state=42)
    else:
        raise ValueError(f"Unexpected model name: {name}")
    return Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=500)),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=list(STATES), average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(STATES)).tolist(),
        "correct": int(np.sum(y_true == y_pred)),
        "total": int(len(y_true)),
    }


def check_baseline_predictions(previous, name, test, predictions):
    expected = previous["models"][name]["target_exploratory"]["predictions"]
    actual = [
        {"gsm": str(gsm), "true_state": str(truth), "predicted_state": str(pred)}
        for gsm, truth, pred in zip(test["gsm"], test["state"], predictions)
    ]
    if actual != expected:
        raise ValueError(f"Stage-09 target predictions could not be reproduced: {name}")


def groups_and_protocol(metadata: pd.DataFrame):
    return {
        state: sorted(metadata.loc[metadata["state"] == state, "treatment"].unique().tolist())
        for state in STATES
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    previous, train, test, excluded, x_train, x_target = load_inputs(root)
    truth = test["state"].to_numpy()
    results = {}
    fitted = {}
    for name in ("Dummy", "LogisticRegression", "LinearSVM"):
        model = make_model(name)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            model.fit(x_train, train["state"].to_numpy())
        predictions = model.predict(x_target)
        check_baseline_predictions(previous, name, test, predictions)
        fitted[name] = model
        results[name] = {"metrics": metrics(truth, predictions), "predictions": predictions.tolist()}

    lr = fitted["LogisticRegression"]
    raw_prob = lr.predict_proba(x_target)
    order = lr.named_steps["classifier"].classes_.tolist()
    probabilities = np.column_stack([raw_prob[:, order.index(state)] for state in STATES])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid logistic class probabilities")
    if not np.array_equal(np.array(STATES)[probabilities.argmax(axis=1)],
                          np.asarray(results["LogisticRegression"]["predictions"])):
        raise ValueError("Logistic predicted labels and class probabilities disagree")

    rows = []
    for i, item in test.iterrows():
        p = probabilities[i]
        predictions = {name: results[name]["predictions"][i] for name in results}
        rows.append({
            "gsm": item["gsm"],
            "true_state": item["state"],
            "treatment": item["treatment"],
            "sample_title": item["title"],
            "label_review": item["label_review"],
            "predictions": predictions,
            "logistic_correct": bool(predictions["LogisticRegression"] == item["state"]),
            "logistic_probabilities_uncalibrated": {
                state: float(p[j]) for j, state in enumerate(STATES)
            },
            "logistic_max_probability_uncalibrated": float(p.max()),
            "logistic_top_two_probability_gap": float(np.sort(p)[-1] - np.sort(p)[-2]),
            "logistic_dtp_minus_acute_probability": float(
                p[STATES.index("DTP")] - p[STATES.index("Acute")]
            ),
            "lr_svm_agree": bool(
                predictions["LogisticRegression"] == predictions["LinearSVM"]
            ),
        })

    baseline_pc9 = next(
        fold for fold in previous["models"]["LogisticRegression"]["loco_folds"]
        if fold["held_out_cell_line"] == "PC9"
    )
    per_state = {
        state: {
            "n": 3,
            "logistic_correct": int(sum(
                row["logistic_correct"] for row in rows if row["true_state"] == state
            )),
            "mean_dtp_minus_acute_probability": float(np.mean([
                row["logistic_dtp_minus_acute_probability"]
                for row in rows if row["true_state"] == state
            ])),
        }
        for state in STATES
    }
    report = {
        "analysis": "post_hoc_held_out_study_prediction_diagnostics",
        "source": "GSE193258",
        "target": "GSE255958",
        "classes_in_order": list(STATES),
        "input_sha256": previous["input_sha256"],
        "sample_counts": {"source_training": 36, "target_evaluated": 9,
                          "target_resistant_excluded": int(len(excluded))},
        "reproduces_stage_09_target_predictions": True,
        "model_metrics": {name: value["metrics"] for name, value in results.items()},
        "logistic_source_pc9_loco_reference": {
            "balanced_accuracy": baseline_pc9["balanced_accuracy"],
            "macro_f1": baseline_pc9["macro_f1"],
            "n": baseline_pc9["test_n"],
            "note": "Different training design: PC9 excluded from training in this fold",
        },
        "lr_svm_agreement": {
            "agree": int(sum(row["lr_svm_agree"] for row in rows)),
            "total": 9,
        },
        "by_true_state": per_state,
        "source_protocol_as_recorded": groups_and_protocol(train),
        "target_protocol_as_recorded": groups_and_protocol(test),
        "samples": rows,
        "limitations": [
            "These nine target samples were already examined in stage 09; this is diagnostic follow-up, not a new untouched external test.",
            "All target samples are PC9, a cell line represented in full source training.",
            "The protocol and time on osimertinib differ between studies; nominal labels may not be biologically equivalent.",
            "LogisticRegression.predict_proba values are uncalibrated; high scores do not establish reliable probabilities.",
            "The study contains only three target samples per class and biological independence is not established.",
            "Sample label_review entries have not been manually scientifically validated.",
            "No target labels or expression values were used for model fitting or supervised feature selection.",
            "No models were selected, calibrated or tuned based on target results.",
            "DTP-specific biology cannot be inferred from these classifications.",
        ],
    }
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Stage-09 external predictions reproduced: yes")
    for name, result in results.items():
        print(f"{name}: {result['metrics']['correct']}/9 correct, "
              f"BA={result['metrics']['balanced_accuracy']:.3f}")
    print(f"Logistic/SVM agreement: {report['lr_svm_agreement']['agree']}/9")
    print("Target sample details (logistic probabilities are NOT calibrated):")
    for row in rows:
        print(f"  {row['gsm']} {row['true_state']:7s} "
              f"LR={row['predictions']['LogisticRegression']:7s} "
              f"SVM={row['predictions']['LinearSVM']:7s} "
              f"p_DTP={row['logistic_probabilities_uncalibrated']['DTP']:.3f} "
              f"p_DTP-p_Acute={row['logistic_dtp_minus_acute_probability']:+.3f}")
    print(f"Saved: {dest}")
    print("Post-hoc external diagnostics only; not independent confirmation of DTP specificity.")


if __name__ == "__main__":
    main()
