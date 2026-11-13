"""Stage 28: source-domain-only V-REx benchmark (not an original method).

V-REx = variance of multiclass CE risks across training cell lines.
Ref: Krueger et al., ICML 2021. This is a small-sample exploratory benchmark.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from .contrast_stability import DCSLClassifier, cell_line_contrasts, loss_and_gradient


def group_risk_variance_and_gradient(
    flat: np.ndarray, x: np.ndarray, y_idx: np.ndarray, groups: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return Var_group(mean multiclass CE), its exact gradient, group risks.

    Uses population variance (1/G), and equal domain weighting for the variance.
    Group risk is averaged over its training samples. Never accesses a test domain.
    """
    x = np.asarray(x, dtype=np.float64)
    y_idx = np.asarray(y_idx)
    groups = np.asarray(groups)
    flat = np.asarray(flat, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != len(y_idx) or len(y_idx) != len(groups):
        raise ValueError("Mismatched training dimensions")
    n, p = x.shape
    if not n or not p or flat.shape != (3 * p + 3,):
        raise ValueError("Wrong parameter shape or empty data")
    if not np.isfinite(x).all() or not np.isfinite(flat).all():
        raise ValueError("Non-finite data or parameters")
    if y_idx.dtype.kind not in "iu" or np.any((y_idx < 0) | (y_idx >= 3)):
        raise ValueError("Expected integer class indices 0..2")
    unique, group_id = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two training domains required")
    counts = np.bincount(group_id).astype(np.float64)
    w = flat[:3 * p].reshape(p, 3)
    b = flat[3 * p:]
    logits = x @ w + b
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    per_row_risk = -logp[np.arange(n), y_idx]
    group_risks = np.bincount(group_id, weights=per_row_risk, minlength=len(unique)) / counts
    center = group_risks - group_risks.mean()
    variance = float(np.mean(center ** 2))
    # d Var / d ell_i = (2/G) * (L_group(i) - mean_L) / n_group(i)
    row_weight = (2.0 / len(unique)) * center[group_id] / counts[group_id]
    probability_delta = np.exp(logp)
    probability_delta[np.arange(n), y_idx] -= 1.0
    probability_delta *= row_weight[:, None]
    grad_w = x.T @ probability_delta
    grad_b = probability_delta.sum(axis=0)
    grad = np.concatenate([grad_w.ravel(), grad_b])
    return variance, grad, group_risks


def vrex_combined_loss_and_gradient(
    flat: np.ndarray, x: np.ndarray, y_idx: np.ndarray, contrasts: np.ndarray,
    groups: np.ndarray, l2: float, gamma: float, beta: float
) -> tuple[float, np.ndarray]:
    if beta < 0 or not np.isfinite(beta):
        raise ValueError("beta must be finite and nonnegative")
    base, grad = loss_and_gradient(flat, x, y_idx, contrasts, l2, gamma)
    if beta == 0:
        return base, grad
    variance, grad_var, _ = group_risk_variance_and_gradient(flat, x, y_idx, groups)
    return base + beta * variance, grad + beta * grad_var


class VRExDCSLClassifier(DCSLClassifier):
    """Matched-preprocessing, fixed-beta V-REx ablation of stage-23 DCSL.

    fit() invokes the existing training-only preprocessing/selector, then
    optimizes CE + L2 + gamma DCSL penalty + beta V-REx on those same samples.
    beta=0 exactly retains the stage-23 fit (no second optimization).
    """

    def __init__(self, *, beta: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if beta < 0 or not np.isfinite(beta):
            raise ValueError("beta must be finite and nonnegative")
        self.beta = float(beta)

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        super().fit(x, y, groups)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=str)
        groups = np.asarray(groups, dtype=str)
        z = self.scaler_.transform(x[:, self.nonconstant_indices_])[:, self.selected_local_]
        indices = np.array([self.classes_.tolist().index(t) for t in y], dtype=np.int64)
        contrasts = cell_line_contrasts(z, y, groups)
        start = np.concatenate([self.coef_.ravel(), self.intercept_])
        if self.beta:
            opt = minimize(
                vrex_combined_loss_and_gradient, start,
                args=(z, indices, contrasts, groups, self.l2, self.gamma, self.beta),
                method="L-BFGS-B", jac=True,
                options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7},
            )
            if not opt.success:
                raise RuntimeError(f"V-REx optimization failed: {opt.message}")
            self.coef_ = opt.x[:self.n_features * 3].reshape(self.n_features, 3)
            self.intercept_ = opt.x[self.n_features * 3:]
            self.train_loss_ = float(opt.fun)
            self.n_iter_ = int(opt.nit)
        flat = np.concatenate([self.coef_.ravel(), self.intercept_])
        variance, _, risks = group_risk_variance_and_gradient(flat, z, indices, groups)
        self.train_risk_variance_ = variance
        self.train_group_ce_risks_ = {str(k): float(v) for k, v in zip(np.unique(groups), risks)}
        self.train_group_margins_ = contrasts @ (self.coef_[:, 2] - self.coef_[:, 1])
        self.train_margin_variance_ = float(np.var(self.train_group_margins_))
        return self
