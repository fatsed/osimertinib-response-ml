"""Frozen Stage-35 binary baseline estimators; no tuning or new method."""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from osimertinib_ml.features.selection import binary_feature_steps

FAMILIES = ("Dummy", "LogisticRegression", "LinearSVM")


def make_binary_model(family: str):
    """Create the exact Stage-35 unfitted model (C, seed, 500 genes unchanged)."""
    if family == "Dummy":
        return DummyClassifier(strategy="most_frequent")
    if family == "LogisticRegression":
        classifier = LogisticRegression(C=0.01, solver="lbfgs", max_iter=3000, random_state=42)
    elif family == "LinearSVM":
        classifier = LinearSVC(C=0.001, max_iter=20000, dual=True, random_state=42)
    else:
        raise ValueError("Unknown frozen baseline")
    return Pipeline(binary_feature_steps(n_features=500) + [("classifier", classifier)])
