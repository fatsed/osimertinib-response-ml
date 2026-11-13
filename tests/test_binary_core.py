"""Regression guards for moving Stage-35 code into the reusable package."""
from __future__ import annotations

import numpy as np
import pytest

from osimertinib_ml.evaluation.metrics import binary_metrics
from osimertinib_ml.features.selection import binary_feature_steps
from osimertinib_ml.models.baselines import make_binary_model


def test_frozen_model_parameters_and_pipeline_order():
    model = make_binary_model("LogisticRegression")
    assert list(model.named_steps) == ["variance", "anova", "standardize", "classifier"]
    assert model.named_steps["anova"].k == 500
    assert model.named_steps["classifier"].C == .01
    assert make_binary_model("LinearSVM").named_steps["classifier"].C == .001
    assert make_binary_model("Dummy").strategy == "most_frequent"
    with pytest.raises(ValueError):
        make_binary_model("wrong")


def test_feature_steps_are_unfitted_and_independent():
    a, b = binary_feature_steps(), binary_feature_steps()
    assert a[0][1] is not b[0][1]
    assert a[1][1] is not b[1][1]
    assert a[2][1] is not b[2][1]
    with pytest.raises(ValueError):
        binary_feature_steps(0)


def test_binary_metrics_match_frozen_stage35_structure():
    actual = np.array(["Acute"] * 3 + ["DTP"] * 3)
    guess = np.array(["Acute"] * 3 + ["Acute"] * 2 + ["DTP"])
    m = binary_metrics(actual, guess)
    assert m["confusion_matrix_Acute_DTP"] == [[3, 0], [2, 1]]
    assert m["dtp_recall"] == pytest.approx(1 / 3)
    assert m["balanced_accuracy"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        binary_metrics(actual[:3], guess[:3])
