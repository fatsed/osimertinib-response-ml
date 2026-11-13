"""Fixed, exploratory reference-assisted DTP/Acute transfer comparison.

Run from project root:
  python scripts/reference_conditioned_development.py --self-test
  python scripts/reference_conditioned_development.py --check-only
  python scripts/reference_conditioned_development.py

Uses the Stage-31 frozen cell-line folds, Stage-35 binary cohort/model settings,
and Stage-05 source-only GSE193258 matrix. Comparison:
  raw: Stage-35 Logistic Regression, no target-control access
  shift: raw fitted model with held-out line control-mean shift correction
  centered: Logistic Regression trained on within-line control-centered data

Both 'shift' and 'centered' use three labelled Control samples from the held-out
cell line as a reference. None uses held-out Acute or DTP labels for fitting,
feature selection or preprocessing. The held-out folds were previously seen:
results are developmental, NOT independent validation or new method novelty.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STAGE35 = Path("scripts/35_dtp_vs_acute_benchmark.py")
PREVIOUS = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
PREPARED = Path("data/processed/GSE193258/log2tpm_36samples.tsv.gz")
METADATA = Path("data/processed/GSE193258/samples_36.csv")
PROTOCOL = Path("results/dcsl_stage31_evaluation_protocol.json")
OUTPUT = Path("results/reference_conditioned_development.json")
VARIANTS = ("Raw_Stage35", "ControlShift_Raw", "ControlCentered_Train")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_stage35(root: Path):
    p = root / STAGE35
    spec = importlib.util.spec_from_file_location("reference_development_stage35", p)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_transform(x: np.ndarray, line_names: np.ndarray, refs: dict[str, np.ndarray]) -> np.ndarray:
    """Subtract each sample's own line-specific Control centroid (no fitted parameters)."""
    x = np.asarray(x, dtype=np.float64)
    names = np.asarray(line_names, dtype=str)
    if x.ndim != 2 or len(names) != len(x) or not np.isfinite(x).all():
        raise ValueError("Invalid reference-transform inputs")
    if not set(names).issubset(refs):
        raise ValueError("Control reference missing for a cell line")
    out = np.empty_like(x)
    for line in np.unique(names):
        r = np.asarray(refs[line], dtype=np.float64)
        if r.shape != (x.shape[1],) or not np.isfinite(r).all():
            raise ValueError(f"Invalid control reference for {line}")
        out[names == line] = x[names == line] - r
    return out


def get_control_refs(x: np.ndarray, meta: pd.DataFrame, idx: dict[str, int],
                     lines: set[str]) -> dict[str, np.ndarray]:
    refs = {}
    for line in sorted(lines):
        selected = meta.loc[(meta.cell_line == line) & (meta.state == "Control"), "gsm"].tolist()
        if len(selected) != 3 or len(set(selected)) != 3:
            raise ValueError(f"Need three independent Control samples for {line}")
        refs[line] = x[[idx[s] for s in selected]].mean(axis=0)
    return refs


def load_inputs(root: Path, stage35, previous: dict):
    # Recheck Stage-34 hashes and frozen Stage-31 data before running anything.
    preflight = stage35.run(root, check_only=True)
    if (previous.get("source") != "GSE193258" or
        previous.get("n_binary_samples") != 24 or
        previous.get("fixed_models", {}).get("LogisticRegression") != {"C": 0.01, "k": 500} or
        previous.get("source_provenance") != preflight.get("source_provenance") or
        previous.get("frozen_stage31_sha256") != sha256(root / PROTOCOL)):
        raise ValueError("Stage-35 baseline and source provenance disagree")
    proto = json.loads((root / PROTOCOL).read_text(encoding="utf-8"))
    folds = proto["nested_loco_protocol"]["folds"]
    orig = pd.read_csv(root / METADATA, dtype=str, keep_default_na=False)
    meta = orig.rename(columns={"candidate_group": "state"})
    stage35.validate_binary_cohort(meta, folds)
    if (len(meta) != 36 or len(set(meta.cell_line)) != 4 or
            set(meta.state) != {"Control", "Acute", "DTP"} or
            not meta.groupby(["cell_line", "state"]).size().eq(3).all()):
        raise ValueError("Reference experiment needs 4 lines with 3 controls and 6 binary samples each")
    if sha256(root / PREPARED) != previous["source_provenance"]["prepared_matrix_sha256"]:
        raise ValueError("Prepared source-only matrix changed since Stage 35")
    return folds, meta


def run(root: Path, check_only: bool = False) -> dict:
    stage35 = load_stage35(root)
    previous = json.loads((root / PREVIOUS).read_text(encoding="utf-8"))
    folds, meta = load_inputs(root, stage35, previous)
    if check_only:
        print("OK: 36 source-only samples; 12 Control references; 24 Acute/DTP; 4 frozen LOCO folds")
        print("Compared: original raw, ControlShift_Raw and ControlCentered_Train; fixed Stage-35 LR settings")
        print("No study outside GSE193258 is read; no independent validation or method selection")
        return {"check_only": True}

    stage34 = stage35.import_stage34(root)
    frame = pd.read_csv(root / PREPARED, sep="\t", compression="gzip")
    _genes, x = stage34.validate_matrix(frame, meta.gsm.tolist())
    idx = {gsm: i for i, gsm in enumerate(meta.gsm)}
    y = meta.state.to_numpy(dtype=str)
    records = {variant: [] for variant in VARIANTS}
    old_folds = {f["held_out_line"]: f for f in previous["folds_by_family"]["LogisticRegression"]}
    lines = set(meta.cell_line)
    for fold in folds:
        held = fold["outer_test_cell_line"]
        train_gsm, test_gsm, _inner = stage35.select_frozen_rows(fold, meta)
        tr = np.array([idx[gsm] for gsm in train_gsm], dtype=int)
        te = np.array([idx[gsm] for gsm in test_gsm], dtype=int)
        train_lines = meta.iloc[tr].cell_line.to_numpy(dtype=str)
        target_lines = meta.iloc[te].cell_line.to_numpy(dtype=str)
        if set(train_lines) != lines - {held} or set(target_lines) != {held}:
            raise ValueError("Frozen outer training and held-out lines disagree")
        train_refs = get_control_refs(x, meta, idx, set(train_lines))
        test_refs = get_control_refs(x, meta, idx, {held})
        if set(test_gsm) & set(meta.loc[meta.state.eq("Control"), "gsm"]):
            raise ValueError("Target Control samples entered binary held-out test labels")

        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            raw = stage35.make_model("LogisticRegression").fit(x[tr], y[tr])
        raw_pred = raw.predict(x[te]).tolist()
        historical = {p["gsm"]: p["predicted"] for p in old_folds[held]["predictions"]}
        if dict(zip(test_gsm, raw_pred)) != historical:
            raise ValueError(f"Original Stage-35 predictions not reproduced for {held}")

        # Matched-access baseline: fit the unchanged raw training model, but
        # relocate held-out samples onto the pooled training-control baseline.
        pooled_train_control = np.vstack(list(train_refs.values())).mean(axis=0)
        # All source lines have exactly 3 Control replicates, so averaging
        # control centroids equals the 9-sample pooled Control mean.
        shifted_test = x[te] - test_refs[held] + pooled_train_control
        shift_pred = raw.predict(shifted_test).tolist()

        # Reference-conditioned model: both training and target groups are
        # centered using only their own Control samples. All feature learning
        # (ANOVA, scaling, LR) happens on transformed TRAINING arms only.
        centered_train = reference_transform(x[tr], train_lines, train_refs)
        centered_test = reference_transform(x[te], target_lines, test_refs)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            centered = stage35.make_model("LogisticRegression").fit(centered_train, y[tr])
        centered_pred = centered.predict(centered_test).tolist()

        for variant, pred in zip(VARIANTS, (raw_pred, shift_pred, centered_pred)):
            arr = np.asarray(pred, dtype=str)
            records[variant].append({
                "held_out_line": held,
                "train_n_binary": len(tr),
                "train_n_controls_reference_only": 9,
                "test_n_binary": len(te),
                "heldout_n_controls_reference_only": 0 if variant == VARIANTS[0] else 3,
                "metrics": stage35.metrics(y[te], arr),
                "predictions": [
                    {"gsm": gsm, "actual": str(a), "predicted": str(p)}
                    for gsm, a, p in zip(test_gsm, y[te], pred)
                ],
            })
        print(f"{held}: raw={stage35.metrics(y[te], np.asarray(raw_pred))['balanced_accuracy']:.3f}; "
              f"shift={stage35.metrics(y[te], np.asarray(shift_pred))['balanced_accuracy']:.3f}; "
              f"centered={stage35.metrics(y[te], np.asarray(centered_pred))['balanced_accuracy']:.3f}")

    summary = {}
    for variant, entries in records.items():
        rows = [p for f in entries for p in f["predictions"]]
        if len(rows) != 24 or len(set(p["gsm"] for p in rows)) != 24:
            raise ValueError("Binary held-out samples missing or duplicated")
        pooled = stage35.metrics(np.asarray([p["actual"] for p in rows]),
                                 np.asarray([p["predicted"] for p in rows]))
        summary[variant] = {
            "pooled": pooled,
            "mean_fold_balanced_accuracy": float(np.mean([f["metrics"]["balanced_accuracy"] for f in entries])),
            "mean_fold_dtp_recall": float(np.mean([f["metrics"]["dtp_recall"] for f in entries])),
            "worst_fold_balanced_accuracy": float(min(f["metrics"]["balanced_accuracy"] for f in entries)),
        }

    report = {
        "analysis": "fixed_reference_assisted_development_only",
        "source": "GSE193258",
        "stage35_source_sha256": previous["source_provenance"],
        "stage35_reference_report_sha256": sha256(root / PREVIOUS),
        "frozen_protocol_sha256": sha256(root / PROTOCOL),
        "settings": {"classifier": "LogisticRegression", "C": 0.01, "k": 500,
                     "seed": 42, "folds": "frozen Stage-31 outer LOCO", "n_lines": 4,
                     "n_binary": 24, "n_control": 12, "normalization": "log2TPM as supplied"},
        "variants": {
            "Raw_Stage35": "No held-out references; exactly reproduces Stage 35 LR",
            "ControlShift_Raw": "Original raw LR with held-out Control centroid shifted to pooled training Control centroid",
            "ControlCentered_Train": "Subtract same-line Control centroid in train/test BEFORE training-only ANOVA and standardization",
        },
        "no_hyperparameter_search": True,
        "heldout_acute_dtp_labels_not_used_for_training": True,
        "heldout_control_references_used": ["ControlShift_Raw", "ControlCentered_Train"],
        "historically_inspected_outer_folds": True,
        "independent_validation": False,
        "limitations": [
            "This is a reference-assisted transfer setting: access to 3 Control samples from the held-out line is required.",
            "All fold outcomes have been previously examined; results are exploratory, not unbiased generalization estimates.",
            "Acute=24h versus DTP=21d remains confounded with treatment duration.",
            "Three Control replicates may produce a noisy baseline estimate.",
            "Labels/source grouping remain provisional pending biological protocol review.",
            "Control centering and target-domain baseline alignment are known techniques; no novelty claimed.",
        ],
        "summary": summary,
        "folds_by_variant": records,
    }
    out = root / OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, obj in summary.items():
        print(f"{name}: mean fold BA={obj['mean_fold_balanced_accuracy']:.3f}, "
              f"worst={obj['worst_fold_balanced_accuracy']:.3f}, "
              f"DTP recall={obj['pooled']['dtp_recall']:.3f}")
    print("Saved:", out)
    return report


def self_test() -> None:
    # Two expression profiles with the SAME treatment-associated change but
    # DIFFERENT per-line baseline expression must yield identical deltas.
    controls = {"A": np.array([1., 4.]), "B": np.array([20., -3.])}
    values = np.array([[2., 6.], [21., -1.], [1., 4.], [20., -3.]])
    lines = np.array(["A", "B", "A", "B"])
    z = reference_transform(values, lines, controls)
    assert np.allclose(z[0], [1., 2.])
    assert np.allclose(z[0], z[1])
    assert np.allclose(z[2:], 0.)
    # Missing target reference must be a hard failure, not replaced with
    # unseen Acute or DTP labels/centroids.
    try:
        reference_transform(values, lines, {"A": controls["A"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Missing reference was silently accepted")
    print("Self-test OK: baseline shifts cancel; missing references rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.root.resolve(), check_only=args.check_only)


if __name__ == "__main__":
    main()
