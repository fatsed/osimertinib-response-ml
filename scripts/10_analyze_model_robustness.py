from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("data/processed/cross_study")
BASELINE_FILE = Path("results/baseline_gse193258_loco.json")
OUTPUT_FILE = Path("results/gse193258_robustness.json")
STATES = ["Control", "Acute", "DTP"]
FEATURE_COUNTS = [50, 200, 500, 1000]
BASELINE_FEATURES = 500


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(root: Path):
    folder = root / DATA_DIR
    matrix_path = folder / "gse193258_common_log2tpm.tsv.gz"
    metadata_path = folder / "samples.csv"
    report_path = root / BASELINE_FILE

    baseline = json.loads(report_path.read_text(encoding="utf-8"))
    expected = baseline.get("input_sha256", {})
    if expected.get(matrix_path.name) != file_sha256(matrix_path):
        raise ValueError("Source expression matrix differs from the baseline experiment")
    if expected.get(metadata_path.name) != file_sha256(metadata_path):
        raise ValueError("Sample metadata differ from the baseline experiment")

    matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip")
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    if "gene_id" not in matrix.columns:
        raise ValueError("Gene identifiers are missing")
    genes = matrix.pop("gene_id").astype(str).str.strip()
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError("Gene identifiers are missing or duplicated")

    source = metadata.loc[metadata["accession"] == "GSE193258"].copy()
    if len(source) != 36 or source["gsm"].duplicated().any():
        raise ValueError("Expected 36 unique source samples")
    if set(source["gsm"]) != set(matrix.columns) or len(matrix.columns) != 36:
        raise ValueError("Sample identifiers do not match the expression columns")
    if source["state"].value_counts().to_dict() != {state: 12 for state in STATES}:
        raise ValueError("Unexpected source response-state groups")
    if source["cell_line"].nunique() != 4:
        raise ValueError("Expected four distinct cell lines")
    if len(source.groupby(["cell_line", "state"])) != 12:
        raise ValueError("Missing cell-line/state combinations")
    if not source.groupby(["cell_line", "state"]).size().eq(3).all():
        raise ValueError("Expected three samples per cell-line/state combination")

    source = source.set_index("gsm").loc[list(matrix.columns)].reset_index()
    x = matrix.to_numpy(dtype=float).T
    if not np.isfinite(x).all():
        raise ValueError("Expression matrix contains invalid numbers")
    return x, genes.to_numpy(), source, baseline


def make_model(k: int):
    return Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)),
    ])


def score(y_true, y_pred, classes):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=classes
        ).tolist(),
    }


def evaluate(x, genes, source, classes, feature_counts):
    eligible = source["state"].isin(classes).to_numpy()
    x = x[eligible]
    info = source.loc[eligible].reset_index(drop=True)
    y = info["state"].to_numpy()
    lines = info["cell_line"].to_numpy()

    per_count = {}
    feature_sets = {}
    for k in feature_counts:
        folds = []
        all_true, all_pred = [], []
        selected_per_fold = {}
        for line in sorted(np.unique(lines)):
            test = lines == line
            train = ~test
            model = make_model(k)
            model.fit(x[train], y[train])
            predicted = model.predict(x[test])
            fold = score(y[test], predicted, classes)
            fold["held_out_cell_line"] = line
            fold["predictions"] = [
                {"gsm": gsm, "true_state": actual, "predicted_state": pred}
                for gsm, actual, pred in zip(
                    info.loc[test, "gsm"], y[test], predicted
                )
            ]
            folds.append(fold)
            all_true.extend(y[test].tolist())
            all_pred.extend(predicted.tolist())

            kept = model.named_steps["filter"].get_support()
            chosen = model.named_steps["select"].get_support()
            selected_per_fold[line] = set(genes[kept][chosen].tolist())

        result = score(all_true, all_pred, classes)
        result["folds"] = folds
        per_count[str(k)] = result
        feature_sets[k] = selected_per_fold

    return per_count, feature_sets


def mean_jaccard(sets):
    values = []
    for first, second in combinations(sorted(sets), 2):
        left, right = sets[first], sets[second]
        values.append(len(left & right) / len(left | right))
    return float(np.mean(values)) if values else None


def validate_baseline(results, baseline):
    saved = {
        fold["held_out_cell_line"]: fold
        for fold in baseline["models"]["LogisticRegression"]["loco_folds"]
    }
    actual = results[str(BASELINE_FEATURES)]["folds"]
    if set(saved) != {fold["held_out_cell_line"] for fold in actual}:
        raise ValueError("LOCO fold identities differ from the previous experiment")
    for fold in actual:
        prior = saved[fold["held_out_cell_line"]]["predictions"]
        if fold["predictions"] != prior:
            raise ValueError(
                "500-gene predictions do not reproduce the stage-09 baseline "
                f"for {fold['held_out_cell_line']}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    x, genes, source, baseline = load_source(root)
    multi, features = evaluate(x, genes, source, STATES, FEATURE_COUNTS)
    validate_baseline(multi, baseline)
    exposed, _ = evaluate(
        x, genes, source, ["Acute", "DTP"], [BASELINE_FEATURES]
    )

    sensitivity = {
        k: {
            "balanced_accuracy": multi[str(k)]["balanced_accuracy"],
            "macro_f1": multi[str(k)]["macro_f1"],
        }
        for k in FEATURE_COUNTS
    }
    jaccard = {
        str(k): mean_jaccard(features[k])
        for k in FEATURE_COUNTS
    }
    errors = [
        {
            "held_out_cell_line": fold["held_out_cell_line"],
            **item,
        }
        for fold in multi[str(BASELINE_FEATURES)]["folds"]
        for item in fold["predictions"]
        if item["true_state"] != item["predicted_state"]
    ]

    report = {
        "analysis": "post_baseline_source_only_robustness_diagnostics",
        "source": "GSE193258",
        "samples": len(source),
        "genes": len(genes),
        "baseline_predictions_reproduced": True,
        "multiclass_logistic_regression": {
            "feature_count_sensitivity": sensitivity,
            "selected_gene_jaccard_between_training_folds": jaccard,
            "baseline_500_gene_errors": errors,
        },
        "acute_vs_dtp_only": {
            "meaning": "Drug-exposed groups only; no untreated controls",
            "feature_count": BASELINE_FEATURES,
            **exposed[str(BASELINE_FEATURES)],
        },
        "cautions": [
            "These analyses revisit LOCO test folds after observing baseline results.",
            "Feature-count comparisons are sensitivity analyses, not hyperparameter tuning.",
            "Do not choose the best feature count using these results and then report "
            "its LOCO score as an untouched test result.",
            "No GSE255958 expression values or labels are used in this script.",
            "Acute-versus-DTP classification does not establish DTP-specific biology "
            "because treatment duration and other factors differ.",
            "Biological replicate independence and treatment labels require study review.",
        ],
    }
    output = root / OUTPUT_FILE
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("Stage-09 500-gene LOCO predictions reproduced: yes")
    for k, values in sensitivity.items():
        print(
            f"{k} genes: LOCO BA={values['balanced_accuracy']:.3f}, "
            f"macro-F1={values['macro_f1']:.3f}, "
            f"mean feature Jaccard={jaccard[str(k)]:.3f}"
        )
    print(
        "Acute vs DTP (LOCO): "
        f"BA={report['acute_vs_dtp_only']['balanced_accuracy']:.3f}, "
        f"macro-F1={report['acute_vs_dtp_only']['macro_f1']:.3f}"
    )
    print(f"Saved: {output}")
    print("Exploratory diagnostics only; this does not establish DTP specificity.")


if __name__ == "__main__":
    main()
