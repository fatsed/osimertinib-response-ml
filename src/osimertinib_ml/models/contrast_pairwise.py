"""Development-only binary Acute/DTP contrast objectives.

Uses training-cell-line labels and within-line DTP/Acute pairs.  The pairwise
term is a RankNet-style loss; soft worst-group aggregation is a known DRO
surrogate. Their combination is a benchmark hypothesis, not established novelty.

Preprocessing / gene selection must happen on the training split *before*
calling fit.  This estimator does not look at a held-out cohort.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax


def _prepare(x, y, groups):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=str)
    groups = np.asarray(groups, dtype=str)
    if x.ndim != 2 or not all(x.shape) or len(y) != len(x) or len(groups) != len(x):
        raise ValueError("Mismatched or empty x/y/groups")
    if not np.isfinite(x).all() or not set(y).issubset({"Acute", "DTP"}) or set(y) != {"Acute", "DTP"}:
        raise ValueError("Expected finite matrix and both Acute/DTP labels")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two training cell lines are required")
    for group in unique:
        labels = set(y[groups == group])
        if labels != {"Acute", "DTP"}:
            raise ValueError(f"Group {group!r} must contain Acute and DTP")
    return x, (y == "DTP").astype(float), groups, unique


def _smooth_worst(risks: np.ndarray, gradients: np.ndarray, temperature: float):
    """Temperature*(logsumexp(risks/T)-log G), and exact gradient."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    weights = softmax(risks / temperature)
    value = temperature * (logsumexp(risks / temperature) - np.log(len(risks)))
    grad = np.sum(weights[:, None] * gradients, axis=0)
    return float(value), grad


def pairwise_group_terms(flat, x, y, groups):
    """For each training group, average softplus[-w@(DTP-Acute)] and gradient.

    DTP/Acute pairs within a group share samples; they are not independent
    biological replicates. Intercept cancels from every pairwise comparison.
    """
    x, target, groups, unique = _prepare(x, y, groups)
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (x.shape[1] + 1,) or not np.isfinite(flat).all():
        raise ValueError("Invalid parameter vector")
    w = flat[:-1]
    losses = []
    grads = []
    for group in unique:
        dtp = x[(groups == group) & (target == 1)]
        acute = x[(groups == group) & (target == 0)]
        differences = (dtp[:, None, :] - acute[None, :, :]).reshape(-1, x.shape[1])
        margins = differences @ w
        losses.append(float(np.mean(np.logaddexp(0.0, -margins))))
        grad_w = -(differences.T @ expit(-margins)) / len(differences)
        grads.append(np.r_[grad_w, 0.0])
    return np.asarray(losses), np.asarray(grads)


def group_bce_terms(flat, x, y, groups):
    """Group-level pointwise logistic risks and their gradients."""
    x, target, groups, unique = _prepare(x, y, groups)
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (x.shape[1] + 1,) or not np.isfinite(flat).all():
        raise ValueError("Invalid parameter vector")
    losses, grads = [], []
    for group in unique:
        m = groups == group
        xg, yg = x[m], target[m]
        z = xg @ flat[:-1] + flat[-1]
        losses.append(float(np.mean(np.logaddexp(0.0, z) - yg * z)))
        delta = expit(z) - yg
        grads.append(np.r_[xg.T @ delta / len(delta), delta.mean()])
    return np.asarray(losses), np.asarray(grads)


def objective_and_gradient(flat, x, y, groups, *, l2=0.1, alpha=0.0,
                           pairwise="none", beta=0.0, temperature=0.5):
    """BCE + L2 + alpha*pairwise risk + beta*soft worst-group pointwise BCE.

    pairwise: 'none', 'mean', or 'soft_worst'. With alpha=0 the pairwise term
    is disabled. Use beta>0 for a comparable group-BCE/DRO ablation.
    """
    if pairwise not in {"none", "mean", "soft_worst"}:
        raise ValueError("Invalid pairwise mode")
    if any((not np.isfinite(q) or q < 0) for q in (l2, alpha, beta)):
        raise ValueError("l2/alpha/beta must be nonnegative and finite")
    if pairwise == "none" and alpha:
        raise ValueError("alpha>0 requires a pairwise mode")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    x, target, groups, _ = _prepare(x, y, groups)
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (x.shape[1] + 1,) or not np.isfinite(flat).all():
        raise ValueError("Invalid parameter vector")
    z = x @ flat[:-1] + flat[-1]
    sample_delta = expit(z) - target
    loss = float(np.mean(np.logaddexp(0.0, z) - target * z)
                 + 0.5 * l2 * (flat[:-1] @ flat[:-1]))
    grad = np.r_[x.T @ sample_delta / len(x) + l2 * flat[:-1], sample_delta.mean()]
    if alpha:
        losses, gradients = pairwise_group_terms(flat, x, y, groups)
        if pairwise == "mean":
            pair_loss, pair_grad = float(losses.mean()), gradients.mean(axis=0)
        else:
            pair_loss, pair_grad = _smooth_worst(losses, gradients, temperature)
        loss += alpha * pair_loss
        grad += alpha * pair_grad
    if beta:
        losses, gradients = group_bce_terms(flat, x, y, groups)
        group_loss, group_grad = _smooth_worst(losses, gradients, temperature)
        loss += beta * group_loss
        grad += beta * group_grad
    return loss, grad


class ContrastPairwiseClassifier:
    """Deterministic, regularized linear prototype; not a validated new method."""

    def __init__(self, *, l2=0.1, alpha=0.0, pairwise="none", beta=0.0,
                 temperature=0.5, max_iter=1500):
        self.l2 = l2
        self.alpha = alpha
        self.pairwise = pairwise
        self.beta = beta
        self.temperature = temperature
        self.max_iter = max_iter

    def fit(self, x, y, groups):
        x, _, groups, _ = _prepare(x, y, groups)
        if not isinstance(self.max_iter, int) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        args = dict(l2=self.l2, alpha=self.alpha, pairwise=self.pairwise,
                    beta=self.beta, temperature=self.temperature)
        initial = np.zeros(x.shape[1] + 1, dtype=float)
        def fun(w):
            return objective_and_gradient(w, x, y, groups, **args)
        result = minimize(fun, initial, jac=True, method="L-BFGS-B",
                          options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError(f"Contrast optimization failed: {result.message}")
        self.coef_ = result.x[:-1].copy()
        self.intercept_ = float(result.x[-1])
        self.n_features_in_ = x.shape[1]
        self.classes_ = np.asarray(["Acute", "DTP"])
        self.train_loss_ = float(result.fun)
        self.n_iter_ = int(result.nit)
        self.train_group_pairwise_risks_ = dict(zip(np.unique(groups).tolist(),
            pairwise_group_terms(result.x, x, y, groups)[0].tolist()))
        return self

    def decision_function(self, x):
        if not hasattr(self, "coef_"):
            raise ValueError("Call fit before predict")
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_ or not np.isfinite(x).all():
            raise ValueError("Invalid feature matrix")
        return x @ self.coef_ + self.intercept_

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0, "DTP", "Acute")

    def predict_proba(self, x):
        p = expit(self.decision_function(x))
        return np.column_stack([1 - p, p])
