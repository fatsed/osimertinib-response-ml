"""Unit checks for the exploratory Acute/DTP group-pairwise prototype."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from osimertinib_ml.contrast_pairwise import (
    ContrastPairwiseClassifier, group_bce_terms, objective_and_gradient,
    pairwise_group_terms,
)


def toy():
    rng = np.random.default_rng(91)
    x, y, groups = [], [], []
    for group, offset in (("H1975", 0.4), ("HCC827", -0.5), ("PC9", 0.8)):
        for label, shift in (("Acute", -0.7), ("DTP", 0.8)):
            for _ in range(3):
                item = rng.normal(0, 0.3, 7)
                item[0] += shift
                item[1] += offset
                x.append(item); y.append(label); groups.append(group)
    return np.array(x), np.array(y), np.array(groups)


@pytest.mark.parametrize("pairwise,alpha,beta", [
    ("none", 0.0, 0.0),
    ("mean", 0.2, 0.0),
    ("soft_worst", 0.2, 0.0),
    ("none", 0.0, 0.2),
])
def test_full_analytic_gradient(pairwise, alpha, beta):
    x, y, groups = toy()
    rng = np.random.default_rng(8)
    w = rng.normal(size=x.shape[1] + 1)
    def fun(u):
        return objective_and_gradient(u, x, y, groups, l2=0.1, alpha=alpha,
                                      pairwise=pairwise, beta=beta, temperature=0.5)
    numerical = approx_derivative(lambda u: fun(u)[0], w, method="3-point")
    np.testing.assert_allclose(fun(w)[1], numerical, rtol=2e-6, atol=2e-7)


def test_pairwise_intercept_invariant():
    x, y, groups = toy()
    w = np.arange(x.shape[1] + 1) / 10
    a, grad = pairwise_group_terms(w, x, y, groups)
    moved = w.copy(); moved[-1] += 100
    b, grad_moved = pairwise_group_terms(moved, x, y, groups)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
    np.testing.assert_allclose(grad, grad_moved, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(grad[:, -1], 0)


def test_group_order_independent():
    x, y, groups = toy()
    w = np.arange(x.shape[1] + 1) / 11
    perm = np.random.default_rng(3).permutation(len(y))
    for risk in (pairwise_group_terms, group_bce_terms):
        a = risk(w, x, y, groups)
        b = risk(w, x[perm], y[perm], groups[perm])
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


def test_ablation_alpha_zero_matches_plain_bce():
    x, y, groups = toy()
    w = np.arange(x.shape[1] + 1) / 13
    a = objective_and_gradient(w, x, y, groups, l2=.1, alpha=0, pairwise="none")
    b = objective_and_gradient(w, x, y, groups, l2=.1, alpha=0, pairwise="soft_worst")
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])


def test_fit_predict_and_no_heldout_labels():
    x, y, groups = toy()
    train = groups != "PC9"
    first = ContrastPairwiseClassifier(alpha=.2, pairwise="soft_worst").fit(x[train], y[train], groups[train])
    changed = y.copy(); changed[~train] = "Acute"
    second = ContrastPairwiseClassifier(alpha=.2, pairwise="soft_worst").fit(x[train], changed[train], groups[train])
    np.testing.assert_array_equal(first.coef_, second.coef_)
    np.testing.assert_array_equal(first.predict(x[~train]), second.predict(x[~train]))
    assert set(first.train_group_pairwise_risks_) == {"H1975", "HCC827"}
    assert first.predict_proba(x[~train]).shape == (6, 2)
    np.testing.assert_allclose(first.predict_proba(x[~train]).sum(axis=1), 1.0)


def test_missing_class_in_group_is_rejected():
    x, y, groups = toy()
    keep = ~((groups == "PC9") & (y == "DTP"))
    with pytest.raises(ValueError, match="must contain Acute and DTP"):
        ContrastPairwiseClassifier().fit(x[keep], y[keep], groups[keep])


def test_bad_settings_and_nonfinite_data_rejected():
    x, y, groups = toy()
    with pytest.raises(ValueError, match="temperature"):
        objective_and_gradient(np.zeros(x.shape[1] + 1), x, y, groups,
                               alpha=.2, pairwise="soft_worst", temperature=0)
    with pytest.raises(ValueError, match="requires a pairwise"):
        ContrastPairwiseClassifier(alpha=.2, pairwise="none").fit(x, y, groups)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ContrastPairwiseClassifier().fit(x, y, groups)
