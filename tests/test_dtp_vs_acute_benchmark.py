"""Lightweight integrity tests for Stage 35 binary DTP/Acute evaluation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/35_dtp_vs_acute_benchmark.py"
spec = importlib.util.spec_from_file_location("stage35_binary", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_data():
    lines = ("H1975", "HCC2935", "HCC827", "PC9")
    rows = []
    for line in lines:
        for state in ("Control", "Acute", "DTP"):
            for rep in range(3):
                rows.append({"gsm": f"{line}-{state}-{rep}", "cell_line": line, "state": state})
    meta = pd.DataFrame(rows)
    folds = []
    for line in lines:
        test = meta.loc[meta.cell_line == line, "gsm"].tolist()
        train = meta.loc[meta.cell_line != line, "gsm"].tolist()
        inner = []
        for val_line in lines:
            if val_line == line:
                continue
            tr = meta.loc[(meta.cell_line != line) & (meta.cell_line != val_line), "gsm"].tolist()
            val = meta.loc[meta.cell_line == val_line, "gsm"].tolist()
            inner.append({"train_gsm": tr, "validation_gsm": val})
        folds.append({"outer_test_cell_line": line, "outer_train_gsm": train,
                      "outer_test_gsm": test, "inner_folds": inner})
    return meta, folds


def test_binary_subset_and_folds():
    meta, folds = fixture_data()
    sub = m.validate_binary_cohort(meta, folds)
    assert len(sub) == 24
    tr, te, inner = m.select_frozen_rows(folds[-1], meta)
    assert len(tr) == 18 and len(te) == 6
    assert all(len(a) == 12 and len(b) == 6 for a, b in inner)
    assert set(tr).isdisjoint(te)


def test_tampered_inner_split_rejected():
    meta, folds = fixture_data()
    fold = folds[-1]
    fold["inner_folds"][0]["train_gsm"][0] = fold["outer_test_gsm"][3]
    with pytest.raises(ValueError):
        m.select_frozen_rows(fold, meta)


def test_incomplete_source_or_duplicate_gsm_rejected():
    meta, folds = fixture_data()
    with pytest.raises(ValueError):
        m.validate_binary_cohort(meta.iloc[:-1], folds)
    bad = meta.copy()
    bad.loc[1, "gsm"] = bad.loc[0, "gsm"]
    with pytest.raises(ValueError):
        m.validate_binary_cohort(bad, folds)


def test_binary_scoring():
    y = np.array(["Acute"] * 3 + ["DTP"] * 3)
    pred = np.array(["Acute"] * 3 + ["Acute"] * 2 + ["DTP"])
    v = m.metrics(y, pred)
    assert v["confusion_matrix_Acute_DTP"] == [[3, 0], [2, 1]]
    assert v["dtp_recall"] == pytest.approx(1 / 3)
    assert v["balanced_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        m.metrics(np.array(["Acute"] * 3), pred[:3])


def test_frozen_models_and_invalid_name():
    assert m.FAMILIES == ("Dummy", "LogisticRegression", "LinearSVM")
    assert m.make_model("LogisticRegression").named_steps["anova"].k == 500
    assert m.make_model("LinearSVM").named_steps["classifier"].C == 0.001
    with pytest.raises(ValueError):
        m.make_model("Unknown")


def test_synthetic_run_no_outer_leakage(monkeypatch):
    meta, folds = fixture_data()
    n = len(meta)
    # Each line has a distinct offset, but the same label signal; 510 features.
    rng = np.random.default_rng(42)
    x = rng.normal(0, .15, size=(n, 510))
    for i, row in enumerate(meta.itertuples()):
        x[i, :10] += {"Control": -2, "Acute": 0, "DTP": 2}[row.state]
        x[i, 10:20] += ("H1975", "HCC2935", "HCC827", "PC9").index(row.cell_line) * .15
    seen = []
    original_make = m.make_model
    def make_logged(family):
        model = original_make(family)
        if family != "Dummy":
            original_fit = model.fit
            def checked_fit(xx, yy, **kwargs):
                seen.append((len(xx), len(yy)))
                assert len(xx) in (12, 18)
                assert set(yy) == set(m.STATES)
                return original_fit(xx, yy, **kwargs)
            model.fit = checked_fit
        return model
    monkeypatch.setattr(m, "make_model", make_logged)
    report = m.evaluate(x, meta, folds)
    assert set(report["summary"]) == set(m.FAMILIES)
    assert all(v["pooled"]["n"] == 24 for v in report["summary"].values())
    assert len(seen) == 4 * 2 * 4
    assert all(length in (12, 18) for length, _ in seen)
