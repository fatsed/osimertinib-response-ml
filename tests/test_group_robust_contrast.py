"""Tests for stage-25 training-only smooth worst-group DTP-vs-Acute loss."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from osimertinib_ml.group_robust_contrast import (
    GroupRobustDCSLClassifier,
    combined_loss_and_gradient,
    group_da_risk_and_gradient,
)
from osimertinib_ml.contrast_stability import DCSLClassifier, cell_line_contrasts


def toy():
    rng = np.random.default_rng(123)
    data, labels, groups = [], [], []
    for group, offset in (("A", 0.0), ("B", 1.3), ("C", -0.6)):
        for state, contrast in (("Control", -1.0), ("Acute", 0.0), ("DTP", 1.8)):
            for _ in range(4):
                x = rng.normal(0, .7, 9)
                x[:3] += contrast
                x[3:6] += offset
                data.append(x); labels.append(state); groups.append(group)
    return np.array(data), np.array(labels), np.array(groups)


def test_exact_group_risk_gradient():
    x, y, groups = toy()
    rng = np.random.default_rng(7)
    p = x.shape[1]
    flat = rng.normal(size=3 * p + 3)
    f = lambda w: group_da_risk_and_gradient(w, x, y, groups, .5)[0]
    estimated = approx_derivative(f, flat, method="3-point")
    actual = group_da_risk_and_gradient(flat, x, y, groups, .5)[1]
    np.testing.assert_allclose(estimated, actual, rtol=1e-6, atol=1e-7)


def test_combined_objective_gradient():
    x, y, groups = toy()
    rng = np.random.default_rng(8)
    flat = rng.normal(size=3 * x.shape[1] + 3)
    indices = np.array([("Control", "Acute", "DTP").index(v) for v in y])
    contrasts = cell_line_contrasts(x, y, groups)
    f = lambda w: combined_loss_and_gradient(w, x, indices, contrasts, y, groups, .1, .05, .2, .5)[0]
    actual = combined_loss_and_gradient(flat, x, indices, contrasts, y, groups, .1, .05, .2, .5)[1]
    np.testing.assert_allclose(approx_derivative(f, flat, method="3-point"), actual, rtol=1e-6, atol=1e-7)


def test_zero_eta_exactly_reproduces_stage23():
    x, y, groups = toy()
    a = DCSLClassifier(n_features=5, selector="stable", gamma=.05).fit(x, y, groups)
    b = GroupRobustDCSLClassifier(n_features=5, selector="stable", gamma=.05, eta=0).fit(x, y, groups)
    np.testing.assert_array_equal(a.selected_indices_, b.selected_indices_)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    np.testing.assert_array_equal(a.coef_, b.coef_)


def test_training_only_heldout_not_read():
    x, y, groups = toy()
    train, heldout = groups != "C", groups == "C"
    first = GroupRobustDCSLClassifier(n_features=5, eta=.2).fit(x[train], y[train], groups[train])
    changed_labels = y.copy()
    changed_labels[heldout] = "Control"
    second = GroupRobustDCSLClassifier(n_features=5, eta=.2).fit(x[train], changed_labels[train], groups[train])
    np.testing.assert_array_equal(first.selected_indices_, second.selected_indices_)
    np.testing.assert_array_equal(first.predict(x[heldout]), second.predict(x[heldout]))
    assert np.isfinite(first.train_loss_)
    assert set(first.group_da_losses_) == {"A", "B"}


def test_reject_invalid_group_or_temperature():
    x, y, groups = toy()
    bad = (groups == "A") & (y == "DTP")
    with pytest.raises(ValueError, match="needs Acute and DTP"):
        group_da_risk_and_gradient(np.zeros(3 * x.shape[1] + 3), x[~bad], y[~bad], groups[~bad], .5)
    with pytest.raises(ValueError, match="temperature"):
        GroupRobustDCSLClassifier(eta=.2, temperature=0)
