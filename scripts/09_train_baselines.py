from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("data/processed/cross_study")
OUTPUT = Path("results/baseline_gse193258_loco.json")
STATES = ["Control", "Acute", "DTP"]
SEED = 42
N_FEATURES = 500


def read_matrix(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, sep="\t", compression="gzip", keep_default_na=False)
    if "gene_id" not in matrix.columns:
        raise ValueError(f"Missing gene_id in {path.name}")
    genes = matrix.pop("gene_id").astype(str).str.strip()
    if genes.eq("").any() or genes.duplicated().any():
        raise ValueError(f"Invalid or duplicate gene IDs in {path.name}")
    if matrix.columns.duplicated().any():
        raise ValueError(f"Duplicate sample columns in {path.name}")
    values = matrix.apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite expression values in {path.name}")
    matrix = pd.DataFrame(values, index=genes, columns=matrix.columns)
    matrix.index.name = "gene_id"
    return matrix


def load_data(root: Path):
    folder = root / INPUT
    source = read_matrix(folder / "gse193258_common_log2tpm.tsv.gz")
    target = read_matrix(folder / "gse255958_common_log2tpm_candidate.tsv.gz")
    metadata = pd.read_csv(folder / "samples.csv", dtype=str, keep_default_na=False)

    required = {"accession", "gsm", "cell_line", "state", "evaluation_role"}
    if not required.issubset(metadata.columns) or metadata["gsm"].duplicated().any():
        raise ValueError("Sample metadata are missing fields or have duplicated GSM IDs")
    if not source.index.equals(target.index) or len(source) < 10000:
        raise ValueError("Source and target gene sets/orders are inconsistent")

    train_info = metadata.loc[metadata["accession"] == "GSE193258"].copy()
    test_info = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & (metadata["evaluation_role"] == "target_candidate")
    ].copy()
    other = metadata.loc[
        (metadata["accession"] == "GSE255958")
        & (metadata["evaluation_role"] == "exploratory_only")
    ]
    if len(train_info) != 36 or len(test_info) != 9 or len(other) != 3:
        raise ValueError("Expected 36 source, 9 target and 3 excluded resistant samples")
    if train_info["state"].value_counts().to_dict() != {s: 12 for s in STATES}:
        raise ValueError("Unexpected source state counts")
    if test_info["state"].value_counts().to_dict() != {s: 3 for s in STATES}:
        raise ValueError("Unexpected target state counts")
    if len(train_info["cell_line"].unique()) != 4:
        raise ValueError("Expected four source cell lines")
    if not test_info["cell_line"].eq("PC9").all():
        raise ValueError("Target study samples should all be PC9")
    if not other["state"].eq("Resistant").all():
        raise ValueError("Unexpected excluded sample states")
    if (train_info.groupby(["cell_line", "state"]).size() != 3).any():
        raise ValueError("Source cell-line/state combinations must each have 3 samples")
    if set(source.columns) != set(train_info["gsm"]) or set(target.columns) != set(
        metadata.loc[metadata["accession"] == "GSE255958", "gsm"]
    ):
        raise ValueError("Expression columns do not match the sample metadata")
    if set(source.columns) & set(target.columns):
        raise ValueError("GSM identifiers overlap between studies")

    train_info = train_info.set_index("gsm").loc[list(source.columns)].rename_axis("gsm").reset_index()
    test_info = test_info.set_index("gsm").loc[
        [gsm for gsm in target.columns if gsm in set(test_info["gsm"])]
    ].rename_axis("gsm").reset_index()
    x_source = source.to_numpy(dtype=np.float64).T
    x_target = target.loc[:, test_info["gsm"].tolist()].to_numpy(dtype=np.float64).T
    return x_source, train_info, x_target, test_info, int(len(source))


def make_model(name: str):
    if name == "Dummy":
        return DummyClassifier(strategy="prior")
    if name == "LogisticRegression":
        classifier = LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)
    elif name == "LinearSVM":
        classifier = LinearSVC(C=0.01, dual=True, max_iter=20000, random_state=SEED)
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("scale", StandardScaler()),
        ("classifier", classifier),
    ])


def metrics(y_true, y_pred):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=STATES, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=STATES).tolist(),
    }


def run_experiment(root: Path):
    x, source, x_external, external, genes = load_data(root)
    y = source["state"].to_numpy()
    lines = source["cell_line"].to_numpy()
    y_external = external["state"].to_numpy()
    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    report = {
        "input_sha256": {
            name: digest(root / INPUT / name)
            for name in (
                "gse193258_common_log2tpm.tsv.gz",
                "gse255958_common_log2tpm_candidate.tsv.gz",
                "samples.csv",
            )
        },
        "scikit_learn_version": sklearn.__version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fixed-parameter exploratory baselines, not validated clinical prediction",
        "source": "GSE193258",
        "target": "GSE255958",
        "gene_count": genes,
        "source_samples": len(source),
        "external_target_samples": len(external),
        "classes_in_order": STATES,
        "fixed_seed": SEED,
        "supervised_feature_count": N_FEATURES,
        "source_evaluation": "leave-one-cell-line-out; train on 3 cell lines, test on 1",
        "target_evaluation": "exploratory held-out-study transfer with PC9 represented in source",
        "notes": [
            "All supervised feature selection, variance filtering, and scaling are fit inside each training fold.",
            "No PCA features from step 06 are used.",
            "Hyperparameters are fixed in advance; there is no tuning on test folds or target labels.",
            "Source and target treatment dose/duration differ; matching state names do not establish biological equivalence.",
            "Sample labels and biological replicate independence remain pending source-study review.",
            "The target study contains PC9, also present in training; this is not novel-cell-line validation.",
            "Only Control, Acute and DTP are evaluated; Resistant samples remain excluded.",
            "Common gene identities were aligned using both studies before training, not using target outcome labels.",
        ],
        "models": {},
    }

    for name in ("Dummy", "LogisticRegression", "LinearSVM"):
        folds = []
        all_true, all_pred = [], []
        for held_out in sorted(np.unique(lines)):
            test = lines == held_out
            train = ~test
            if len(np.unique(y[train])) != 3 or len(np.unique(y[test])) != 3:
                raise ValueError(f"Incomplete classes in fold: {held_out}")
            model = make_model(name)
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                model.fit(x[train], y[train])
            pred = model.predict(x[test])
            item = metrics(y[test], pred)
            item.update({
                "held_out_cell_line": str(held_out),
                "train_n": int(train.sum()),
                "test_n": int(test.sum()),
                "predictions": [
                    {"gsm": str(gsm), "true_state": str(actual), "predicted_state": str(guess)}
                    for gsm, actual, guess in zip(source.loc[test, "gsm"], y[test], pred)
                ],
            })
            folds.append(item)
            all_true.extend(y[test].tolist())
            all_pred.extend(pred.tolist())

        full_model = make_model(name)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            full_model.fit(x, y)
        external_pred = full_model.predict(x_external)
        external_stats = metrics(y_external, external_pred)
        external_stats["predictions"] = [
            {"gsm": str(gsm), "true_state": str(actual), "predicted_state": str(guess)}
            for gsm, actual, guess in zip(external["gsm"], y_external, external_pred)
        ]
        report["models"][name] = {
            "loco_folds": folds,
            "loco_pooled": metrics(all_true, all_pred),
            "loco_mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in folds])),
            "loco_mean_macro_f1": float(np.mean([f["macro_f1"] for f in folds])),
            "target_exploratory": external_stats,
        }
        print(f"{name}: LOCO BA={report['models'][name]['loco_mean_balanced_accuracy']:.3f}, "
              f"Macro-F1={report['models'][name]['loco_mean_macro_f1']:.3f}, "
              f"Target BA={external_stats['balanced_accuracy']:.3f}")

    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {destination}")
    print("Exploratory baselines only. No publication claim or model selection from these results.")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run_experiment(args.root.resolve())


if __name__ == "__main__":
    main()
