"""Stage 33: frozen replay integrity, class-order margins, and leakage checks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
file = ROOT / "scripts/33_diagnose_dcsl_vs_logistic.py"
spec = importlib.util.spec_from_file_location("stage33_posthoc", file)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_margin_uses_class_names_not_index_positions():
    class Model:
        classes_ = np.array(["Acute", "Control", "DTP"])
        def decision_function(self, x):
            return np.array([[3., 100., 5.], [10., 0., 2.]])[:len(x)]
        def predict(self, x):
            return np.array(["Control", "Acute"])[:len(x)]
    p, v = m.sample_scores(Model(), np.zeros((2, 2)))
    assert p.tolist() == ["Control", "Acute"]
    assert v.tolist() == [2.0, -8.0]


def test_confusion_uses_actual_class_and_includes_errors():
    gsms = ["A", "B", "C"]
    truth = ["DTP", "DTP", "Acute"]
    pred = {"LogisticRegression": np.array(["DTP", "Acute", "Acute"]),
            "DCSL": np.array(["Acute", "Acute", "DTP"]),
            "matched_anova_gamma0": np.array(["DTP", "Acute", "Acute"]),
            "matched_stable_gamma0": np.array(["DTP", "Acute", "Acute"]),
            "matched_anova_logistic_l2": np.array(["DTP", "Acute", "Acute"]),
            "matched_stable_logistic_l2": np.array(["DTP", "Acute", "Acute"])}
    margins = {k: np.array([1., 0., -1.]) for k in pred}
    records, counts = m.compare_predictions(gsms, truth, pred, margins)
    assert counts == {"both_correct": 0, "logistic_only_correct": 2,
                      "dcsl_only_correct": 0, "both_wrong": 1,
                      "prediction_disagreements": 2}
    assert records[0]["logistic_correct"] is True
    assert records[0]["dcsl_correct"] is False


def test_bad_classes_and_bad_decision_shape_fail():
    class Invalid:
        classes_ = np.array(["Control", "Acute", "Unknown"])
        def decision_function(self, x): return np.zeros((len(x), 3))
        def predict(self, x): return np.array(["Acute"] * len(x))
    with pytest.raises(ValueError, match="classes"):
        m.sample_scores(Invalid(), np.ones((2, 1)))


def test_feature_overlap_count():
    assert m.compare_sets(np.array([1, 2, 3]), np.array([3, 4, 5])) == {
        "n_first": 3, "n_second": 3, "shared_genes": 1, "jaccard": 0.2}
    with pytest.raises(ValueError, match="duplicated"):
        m.compare_sets(np.array([1, 1]), np.array([1, 2]))


def test_prior_predictions_verify_fail_closed():
    ref = {"family_outer_results": {"DCSL": {
        "selected_feature_indices_sha256": "digest",
        "predictions": [{"gsm": "one", "actual": "DTP", "predicted": "Acute"}]}}}
    m.verify_prior_predictions(ref, "DCSL", ["one"], ["DTP"], np.array(["Acute"]), "digest")
    with pytest.raises(ValueError, match="predictions"):
        m.verify_prior_predictions(ref, "DCSL", ["one"], ["DTP"], np.array(["DTP"]), "digest")
    with pytest.raises(ValueError, match="hash"):
        m.verify_prior_predictions(ref, "DCSL", ["one"], ["DTP"], np.array(["Acute"]), "tampered")


def test_prior_provenance_changed_is_rejected(tmp_path):
    stage32 = m.import_stage32(ROOT)
    prior = json.loads((ROOT / m.STAGE32_JSON).read_text(encoding="utf-8"))
    plan = json.loads((ROOT / m.STAGE31_JSON).read_text(encoding="utf-8"))
    prior["independent_validation"] = True
    with pytest.raises(ValueError, match="provenance"):
        m.assert_stage32_unchanged(prior, plan, stage32, ROOT)
    prior["independent_validation"] = False
    prior["protocol_sha256"] = "fake"
    with pytest.raises(ValueError, match="protocol has changed"):
        m.assert_stage32_unchanged(prior, plan, stage32, ROOT)


def test_predeclared_ablations_are_exactly_four():
    assert m.MATCHED_ABLATIONS == ("matched_anova_gamma0", "matched_stable_gamma0",
                                   "matched_anova_logistic_l2", "matched_stable_logistic_l2")


def test_stage32_locked_inputs_and_source_only_loading():
    stage32 = m.import_stage32(ROOT)
    prior = json.loads((ROOT / m.STAGE32_JSON).read_text(encoding="utf-8"))
    plan = json.loads((ROOT / m.STAGE31_JSON).read_text(encoding="utf-8"))
    m.assert_stage32_unchanged(prior, plan, stage32, ROOT)
    x, meta = stage32.load_source(ROOT, plan)
    assert x.shape == (36, 18013)
    assert len(meta) == 36
    assert set(plan["nested_loco_protocol"]["folds"][3]["outer_test_gsm"]) == set(meta.loc[
        meta["cell_line"] == "PC9", "gsm"].tolist())
