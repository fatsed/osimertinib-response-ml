"""Feature preprocessing steps for the frozen binary benchmark.

The returned estimators are UNFITTED. They must be fitted as part of a
Pipeline on the training partition of each fold; never on held-out samples.
"""
from __future__ import annotations

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.preprocessing import StandardScaler


def binary_feature_steps(n_features: int = 500):
    """Return the exact Stage-35 variance/ANOVA/scale step configuration."""
    if not isinstance(n_features, int) or n_features < 1:
        raise ValueError("n_features must be a positive integer")
    return [
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("anova", SelectKBest(score_func=f_classif, k=n_features)),
        ("standardize", StandardScaler()),
    ]
