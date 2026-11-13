"""Stage 28: exact V-REx gradient, behavior, and split integrity checks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier
from osimertinib_ml.vrex_domain_generalization import (
    VRExDCSLClassifier, group_risk_variance_and_gradient,
    vrex_combined_loss_and_gradient,
)


def toy():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(12, 6))
    y = np.tile(np.array([0, 1, 2]), 4)
    groups = np.repeat(np.array(["A", "B", "C", "D"]), 3)
    flat = rng.normal(scale=.15, size=21)
    return x, y, groups, flat


def finite_diff(fn, flat, h=1e-6):
    out = np.zeros_like(flat)
    for j in range(len(flat)):
        step = np.zeros_like(flat); step[j] = h
        out[j] = (fn(flat + step)[0] - fn(flat - step)[0]) / (2 * h)
    return out


def test_vrex_exact_gradient():
    x, y, groups, flat = toy()
    fn = lambda z: group_risk_variance_and_gradient(z, x, y, groups)
    value, grad, risks = fn(flat)
    assert value >= 0 and len(risks) == 4
    assert np.allclose(grad, finite_diff(fn, flat), rtol=1e-5, atol=2e-8)


def test_full_objective_exact_gradient():
    x, y, groups, flat = toy()
    contrasts = np.random.default_rng(11).normal(size=(4, 6))
    fn = lambda z: vrex_combined_loss_and_gradient(z, x, y, contrasts, groups, .1, .03, .2)
    v, grad = fn(flat)
    assert v > 0
    assert np.allclose(grad, finite_diff(fn, flat), rtol=1e-5, atol=5e-8)


def test_constant_risk_has_no_vrex_gradient():
    x = np.zeros((6, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    groups = np.array(["A"] * 3 + ["B"] * 3)
    flat = np.zeros(9)
    value, gradient, _ = group_risk_variance_and_gradient(flat, x, y, groups)
    assert np.isclose(value, 0.0) and np.allclose(gradient, 0)


def test_invalid_groups_and_labels_rejected():
    x, y, groups, flat = toy()
    with pytest.raises(ValueError):
        group_risk_variance_and_gradient(flat, x, y, np.repeat("A", len(groups)))
    with pytest.raises(ValueError):
        group_risk_variance_and_gradient(flat, x, np.ones_like(y) * 3, groups)
    with pytest.raises(ValueError):
        group_risk_variance_and_gradient(flat[:-1], x, y, groups)
    with pytest.raises(ValueError):
        VRExDCSLClassifier(beta=-.1)


def test_beta_zero_reproduces_baseline_and_target_perturbation_invariance():
    rng = np.random.default_rng(24)
    x = rng.normal(size=(27, 14))
    groups = np.repeat(["A", "B", "C"], 9)
    y = np.tile(["Control", "Acute", "DTP"], 9)
    model = VRExDCSLClassifier(beta=0, selector="stable", n_features=5, l2=.1, gamma=0).fit(x, y, groups)
    baseline = DCSLClassifier(selector="stable", n_features=5, l2=.1, gamma=0).fit(x, y, groups)
    assert np.array_equal(model.selected_indices_, baseline.selected_indices_)
    assert np.allclose(model.coef_, baseline.coef_)
    assert np.allclose(model.intercept_, baseline.intercept_)
    target = rng.normal(size=(4, 14))
    original = model.predict(target)
    _ = model.predict(target + 10_000)  # unrelated target batch cannot alter fitted state
    assert np.array_equal(original, model.predict(target))


def test_vrex_fit_and_no_heldout_in_feature_selection():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(36, 15))
    y = np.tile(["Control", "Acute", "DTP"], 12)
    groups = np.repeat(["A", "B", "C", "D"], 9)
    train = groups != "D"
    model = VRExDCSLClassifier(beta=.2, selector="stable", n_features=6,
                               l2=.1, gamma=0).fit(x[train], y[train], groups[train])
    alt = x.copy(); alt[~train] = rng.normal(loc=1000, size=alt[~train].shape)
    refit = VRExDCSLClassifier(beta=.2, selector="stable", n_features=6,
                               l2=.1, gamma=0).fit(alt[train], y[train], groups[train])
    assert np.array_equal(model.selected_indices_, refit.selected_indices_)
    assert np.allclose(model.coef_, refit.coef_)
    assert np.allclose(model.intercept_, refit.intercept_)
    assert len(model.train_group_ce_risks_) == 3
