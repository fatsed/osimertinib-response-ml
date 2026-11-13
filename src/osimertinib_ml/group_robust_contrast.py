"""Stage 25: training-group-robust DTP-versus-Acute loss (exploratory).

Inspired by group-DRO / smooth worst-group risk. This is a research ablation,
not evidence of an original method, clinical validity or confounder removal.
Stage-23 classifier and its outputs remain untouched.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from .contrast_stability import DCSLClassifier, cell_line_contrasts, loss_and_gradient


def group_da_risk_and_gradient(
    flat: np.ndarray, x: np.ndarray, labels: np.ndarray, groups: np.ndarray,
    temperature: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Smooth max over training-cell-line binary DTP/Acute log losses.

    For each group, L_c = mean[softplus(-t_i * (logit_DTP-logit_Acute))]
    over Acute/DTP samples (t=+1 for DTP, -1 for Acute).
    R = temperature * (logsumexp(L_c/temperature) - log(C)).
    Returns R, exact gradient dR/d[W,b], and group L_c's.
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=str)
    groups = np.asarray(groups, dtype=str)
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive and finite")
    if x.ndim != 2 or len(labels) != len(x) or len(groups) != len(x) or not np.isfinite(x).all():
        raise ValueError("invalid training inputs")
    n, p = x.shape
    flat = np.asarray(flat, dtype=float)
    if flat.shape != (3 * p + 3,) or not np.isfinite(flat).all():
        raise ValueError("invalid parameter vector")
    w = flat[:3 * p].reshape(p, 3)
    b = flat[3 * p:]
    binary = (labels == "Acute") | (labels == "DTP")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("at least two training groups are needed")
    d = x @ (w[:, 2] - w[:, 1]) + b[2] - b[1]
    t = np.where(labels == "DTP", 1.0, -1.0)
    sample_loss = np.logaddexp(0, -t * d)
    sample_grad = -t * expit(-t * d)
    losses = []
    masks = []
    for c in unique_groups:
        mask = (groups == c) & binary
        if not np.any((groups == c) & (labels == "Acute")) or not np.any((groups == c) & (labels == "DTP")):
            raise ValueError(f"training group {c!r} needs Acute and DTP")
        masks.append(mask)
        losses.append(float(sample_loss[mask].mean()))
    losses = np.asarray(losses)
    weights = softmax(losses / temperature)
    risk = temperature * (logsumexp(losses / temperature) - np.log(len(unique_groups)))
    g = np.zeros(len(x), dtype=float)
    for mask, weight in zip(masks, weights):
        g[mask] = weight * sample_grad[mask] / int(mask.sum())
    grad_w = np.zeros((p, 3), dtype=float)
    temp_grad = x.T @ g
    grad_w[:, 1] -= temp_grad
    grad_w[:, 2] += temp_grad
    grad_b = np.array([0.0, -g.sum(), g.sum()])
    return float(risk), np.concatenate([grad_w.ravel(), grad_b]), losses


def combined_loss_and_gradient(
    flat: np.ndarray, x: np.ndarray, y_idx: np.ndarray, contrasts: np.ndarray,
    labels: np.ndarray, groups: np.ndarray, l2: float, gamma: float,
    eta: float, temperature: float,
) -> tuple[float, np.ndarray]:
    base_loss, base_grad = loss_and_gradient(flat, x, y_idx, contrasts, l2, gamma)
    if eta == 0:
        return base_loss, base_grad
    robust_loss, robust_grad, _ = group_da_risk_and_gradient(flat, x, labels, groups, temperature)
    return float(base_loss + eta * robust_loss), base_grad + eta * robust_grad


class GroupRobustDCSLClassifier(DCSLClassifier):
    """Training-only smooth worst-cell-line DTP/Acute risk added to stage-23 CE.

    fit() calls stage-23 fit() first solely to reuse fold-isolated preprocessing,
    feature selection, and warm-start. The robust objective is then optimized
    from those coefficients on the *same training rows*. With eta=0, stage-23
    fit and predictions are returned unchanged (exact ablation).
    """

    def __init__(self, *, eta: float = 0.2, temperature: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        if eta < 0 or not np.isfinite(eta) or temperature <= 0 or not np.isfinite(temperature):
            raise ValueError("eta must be nonnegative and temperature positive, both finite")
        self.eta = float(eta)
        self.temperature = float(temperature)

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        super().fit(x, y, groups)
        if self.eta == 0:
            # Diagnostics only: evaluate group loss without changing stage-23 fit.
            x0 = np.asarray(x, dtype=np.float64)
            y0 = np.asarray(y, dtype=str)
            g0 = np.asarray(groups, dtype=str)
            z0 = self.scaler_.transform(x0[:, self.nonconstant_indices_])[:, self.selected_local_]
            flat0 = np.concatenate([self.coef_.ravel(), self.intercept_])
            risk0, _, losses0 = group_da_risk_and_gradient(flat0, z0, y0, g0, self.temperature)
            self.group_da_risk_ = risk0
            self.group_da_losses_ = {str(c): float(v) for c, v in zip(np.unique(g0), losses0)}
            return self
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=str)
        groups = np.asarray(groups, dtype=str)
        z = self.scaler_.transform(x[:, self.nonconstant_indices_])[:, self.selected_local_]
        contrasts = cell_line_contrasts(z, y, groups)
        indices = np.asarray([self.classes_.tolist().index(label) for label in y], dtype=int)
        start = np.concatenate([self.coef_.ravel(), self.intercept_])
        res = minimize(
            combined_loss_and_gradient, start,
            args=(z, indices, contrasts, y, groups, self.l2, self.gamma, self.eta, self.temperature),
            method="L-BFGS-B", jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7},
        )
        if not res.success:
            raise RuntimeError(f"Group-robust optimization failed: {res.message}")
        p = self.n_features
        self.coef_ = res.x[:p * 3].reshape(p, 3)
        self.intercept_ = res.x[p * 3:]
        self.train_group_margins_ = contrasts @ (self.coef_[:, 2] - self.coef_[:, 1])
        self.train_margin_variance_ = float(np.var(self.train_group_margins_))
        self.train_loss_ = float(res.fun)
        self.n_iter_ = int(res.nit)
        risk, _, losses = group_da_risk_and_gradient(res.x, z, y, groups, self.temperature)
        self.group_da_risk_ = risk
        self.group_da_losses_ = {str(c): float(v) for c, v in zip(np.unique(groups), losses)}
        return self
