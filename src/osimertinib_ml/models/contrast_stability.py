"""DTP-Contrast Stability Learning (DCSL), a research prototype.

Fits a fixed-feature-count linear softmax classifier using training-only
feature selection and optionally penalizes between-cell-line variation in
DTP-minus-Acute logit margins. This is NOT a clinical prediction tool.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

STATES = ("Control", "Acute", "DTP")


def cell_line_contrasts(x: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """One DTP-minus-Acute mean-feature vector per training cell line."""
    diffs = []
    for group in np.unique(groups):
        mask = groups == group
        acute = x[mask & (labels == "Acute")]
        dtp = x[mask & (labels == "DTP")]
        if acute.size == 0 or dtp.size == 0:
            raise ValueError(f"Need Acute and DTP examples in training group {group!r}")
        diffs.append(dtp.mean(axis=0) - acute.mean(axis=0))
    if len(diffs) < 2:
        raise ValueError("At least two distinct training cell lines are required")
    return np.asarray(diffs, dtype=np.float64)


def stable_feature_scores(contrasts: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """Within-training stability score (heuristic; not a validated biomarker score)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    mean = contrasts.mean(axis=0)
    sd = contrasts.std(axis=0, ddof=0)
    same_sign_fraction = np.mean(contrasts * mean[None, :] > 0, axis=0)
    return np.abs(mean) * same_sign_fraction / (sd + tau)


def loss_and_gradient(
    flat: np.ndarray, x: np.ndarray, y_idx: np.ndarray, contrasts: np.ndarray,
    l2: float, gamma: float,
) -> tuple[float, np.ndarray]:
    """Exact gradient of CE + (lambda/2)||W||^2 + gamma Var_c(m_c).

    m_c = (W_DTP - W_Acute)^T (mean(X_DTP,c) - mean(X_Acute,c)).
    Both logits use the same feature scaling; biases cancel in the margin.
    """
    n, p = x.shape
    w = flat[:p * 3].reshape(p, 3)
    b = flat[p * 3:]
    logits = x @ w + b
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    ce = -np.mean(log_probs[np.arange(n), y_idx])
    probs = np.exp(log_probs)
    probs[np.arange(n), y_idx] -= 1
    gw = x.T @ probs / n + l2 * w
    gb = probs.sum(axis=0) / n
    loss = ce + 0.5 * l2 * np.sum(w * w)
    if gamma:
        margin = contrasts @ (w[:, 2] - w[:, 1])
        centered = margin - margin.mean()
        penalty = np.mean(centered * centered)
        loss += gamma * penalty
        v = 2.0 * gamma / contrasts.shape[0] * (contrasts.T @ centered)
        gw[:, 2] += v
        gw[:, 1] -= v
    return float(loss), np.concatenate([gw.ravel(), gb])


class DCSLClassifier:
    """Simple fold-isolated classifier; training groups required by fit().

    Parameters are intentionally fixed before looking at held-out labels.
    selector='anova' uses a standard multiclass ANOVA feature ranking.
    selector='stable' uses per-training-line Acute/DTP contrasts.
    gamma=0 disables the stability penalty, enabling ablations.
    """

    def __init__(self, *, selector: str = "stable", n_features: int = 500,
                 l2: float = 0.1, gamma: float = 0.05, tau: float = 0.5,
                 max_iter: int = 1500):
        if selector not in {"anova", "stable"}:
            raise ValueError("selector must be 'anova' or 'stable'")
        if n_features < 1 or l2 <= 0 or gamma < 0 or tau <= 0 or max_iter <= 0:
            raise ValueError("Invalid hyperparameters")
        self.selector = selector
        self.n_features = n_features
        self.l2 = l2
        self.gamma = gamma
        self.tau = tau
        self.max_iter = max_iter

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=str)
        groups = np.asarray(groups, dtype=str)
        if x.ndim != 2 or len(y) != len(x) or len(groups) != len(x):
            raise ValueError("Training data/label/group dimensions differ")
        if not np.isfinite(x).all() or set(np.unique(y)) != set(STATES):
            raise ValueError("Need finite features and all three training states")
        self.n_input_features_ = int(x.shape[1])
        valid = np.var(x, axis=0) > 1e-8
        if np.count_nonzero(valid) < self.n_features:
            raise ValueError("Not enough nonconstant training genes")
        self.nonconstant_indices_ = np.flatnonzero(valid)
        self.scaler_ = StandardScaler().fit(x[:, valid])
        z = self.scaler_.transform(x[:, valid])
        contrasts_all = cell_line_contrasts(z, y, groups)
        if self.selector == "stable":
            scores = stable_feature_scores(contrasts_all, self.tau)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                scores, _ = f_classif(z, y)
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e12, neginf=0.0)
        # Deterministic tie-breaking by source gene order (never by held-out data).
        ranked = np.lexsort((np.arange(scores.size), -scores))
        self.selected_local_ = np.sort(ranked[:self.n_features])
        self.selected_indices_ = self.nonconstant_indices_[self.selected_local_]
        self.selected_scores_ = scores[self.selected_local_].copy()
        z = z[:, self.selected_local_]
        contrasts = contrasts_all[:, self.selected_local_]
        indices = np.array([STATES.index(label) for label in y], dtype=int)
        start = np.zeros(self.n_features * 3 + 3, dtype=np.float64)
        result = minimize(
            loss_and_gradient, start,
            args=(z, indices, contrasts, self.l2, self.gamma),
            method="L-BFGS-B", jac=True,
            options={"maxiter": self.max_iter, "ftol": 1e-11, "gtol": 1e-7},
        )
        if not result.success:
            raise RuntimeError(f"Softmax optimization failed: {result.message}")
        self.coef_ = result.x[:self.n_features * 3].reshape(self.n_features, 3)
        self.intercept_ = result.x[self.n_features * 3:]
        self.classes_ = np.asarray(STATES)
        self.train_group_margins_ = contrasts @ (self.coef_[:, 2] - self.coef_[:, 1])
        self.train_margin_variance_ = float(np.var(self.train_group_margins_))
        self.train_loss_ = float(result.fun)
        self.n_iter_ = int(result.nit)
        self.train_groups_ = np.unique(groups).tolist()
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise ValueError("Fit the model first")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_input_features_ or not np.isfinite(x).all():
            raise ValueError("Invalid test feature matrix")
        z = self.scaler_.transform(x[:, self.nonconstant_indices_])[:, self.selected_local_]
        return z @ self.coef_ + self.intercept_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.classes_[self.decision_function(x).argmax(axis=1)]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = self.decision_function(x)
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
