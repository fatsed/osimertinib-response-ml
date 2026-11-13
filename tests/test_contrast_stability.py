"""Numerical and leakage-oriented unit tests for the stage-23 prototype."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from osimertinib_ml.contrast_stability import (  # noqa: E402
    DCSLClassifier, cell_line_contrasts, loss_and_gradient, stable_feature_scores,
)


def toy():
    rng = np.random.default_rng(71)
    x, y, g = [], [], []
    for c in ("A", "B", "C"):
        for state, effect in (("Control", -0.2), ("Acute", 0.2), ("DTP", 1.6)):
            for _ in range(4):
                row = rng.normal(0, 1, 12)
                row[:3] += effect
                row[3:6] += {"A": -2, "B": 0, "C": 2}[c]
                x.append(row); y.append(state); g.append(c)
    return np.array(x), np.array(y), np.array(g)


def test_analytic_gradient_matches_finite_differences():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(11, 4))
    target = rng.integers(0, 3, size=11)
    contrast = rng.normal(size=(3, 4))
    w = rng.normal(size=4 * 3 + 3)
    fun = lambda t: loss_and_gradient(t, x, target, contrast, 0.13, 0.07)[0]
    numeric = approx_derivative(fun, w, method="3-point")
    analytic = loss_and_gradient(w, x, target, contrast, 0.13, 0.07)[1]
    np.testing.assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-6)


def test_training_only_and_prediction_shapes():
    x, y, g = toy()
    model = DCSLClassifier(n_features=7, selector="stable", gamma=0.03).fit(x, y, g)
    assert len(np.unique(model.selected_indices_)) == 7
    assert set(model.predict(x)) <= {"Control", "Acute", "DTP"}
    np.testing.assert_allclose(model.predict_proba(x).sum(axis=1), 1, atol=1e-12)
    # If evaluation labels are modified, fitting on the unchanged training subset is identical.
    source_mask = g != "C"
    first = DCSLClassifier(n_features=5, gamma=0.03).fit(x[source_mask], y[source_mask], g[source_mask])
    modified_labels = y.copy(); modified_labels[~source_mask] = "Control"
    second = DCSLClassifier(n_features=5, gamma=0.03).fit(x[source_mask], modified_labels[source_mask], g[source_mask])
    np.testing.assert_array_equal(first.selected_indices_, second.selected_indices_)
    np.testing.assert_array_equal(first.predict(x[~source_mask]), second.predict(x[~source_mask]))


def test_invalid_groups_fail_closed():
    x, y, g = toy()
    missing_dtp = (g == "B") & (y == "DTP")
    with pytest.raises(ValueError, match="Acute and DTP"):
        DCSLClassifier(n_features=5).fit(x[~missing_dtp], y[~missing_dtp], g[~missing_dtp])


def test_stability_score_and_margin():
    d = np.array([[2., 1., -1.], [2.1, -1., -1.2], [1.9, 1., -0.8]])
    score = stable_feature_scores(d)
    assert score[0] > score[1]
    assert score[0] > 0
    x, y, g = toy()
    c = cell_line_contrasts(x, y, g)
    assert c.shape == (3, 12)


def test_anova_ablation_fits():
    x, y, g = toy()
    m = DCSLClassifier(selector="anova", gamma=0, n_features=6).fit(x, y, g)
    assert np.isfinite(m.train_loss_)
