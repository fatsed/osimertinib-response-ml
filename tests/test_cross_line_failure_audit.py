"""Tests for deterministic, descriptive stage-27 diagnostics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/27_cross_line_failure_audit.py"
spec = importlib.util.spec_from_file_location("audit_stage27_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_sign_concordance_and_zero_handling():
    assert module.sign_agreement(np.array([1., -1., 0.]), np.array([2., 3., 9.])) == .5
    assert module.sign_agreement(np.array([0., 0.]), np.array([2., 3.])) is None
    with pytest.raises(ValueError):
        module.sign_agreement(np.ones(2), np.ones(3))


def test_hallmark_union_deduplicated_and_background_rate():
    sets = {"a": {"B", "C"}, "b": {"C", "D"}}
    result = module.cycle_overlap({"A", "B"}, sets, {"A", "B", "C", "D", "E"})
    assert result["G2M_OR_E2F_UNION"]["overlap_gene_count"] == 1
    assert result["G2M_OR_E2F_UNION"]["expected_count_if_random"] == 1.2
    assert np.isclose(result["G2M_OR_E2F_UNION"]["observed_to_expected"], 1 / 1.2)
    with pytest.raises(ValueError):
        module.cycle_overlap({"MISSING"}, sets, {"A", "B", "C", "D", "E"})


def test_train_vs_heldout_shift_only_uses_train_reference():
    y = np.array(["Control", "Acute", "DTP"] * 3)
    groups = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
    z_train = np.array([[-1., -1.], [0., 0.], [1., 1.],
                        [-2., -1.], [0., 1.], [2., 1.],
                        [0., -1.], [1., 0.], [2., 2.]])
    z_hold = np.array([[-1., -1.], [0., 0.], [1., 1.]])
    got = module.standardized_centroid_shift(z_train, y, groups, z_hold, np.array(["Control", "Acute", "DTP"]))
    assert set(got) == set(module.STATES)
    assert all(x["heldout_sample_count"] == 1 for x in got.values())
    # Held-out values may change, but the *training reference* must stay unchanged.
    changed = module.standardized_centroid_shift(z_train, y, groups, z_hold + 10,
                                                  np.array(["Control", "Acute", "DTP"]))
    for state in module.STATES:
        assert got[state]["mean_rms_train_group_vs_other_train_centroids"] == changed[state]["mean_rms_train_group_vs_other_train_centroids"]
        assert got[state]["rms_heldout_vs_training_state_centroid"] != changed[state]["rms_heldout_vs_training_state_centroid"]


def test_missing_state_fails():
    y = np.array(["Control", "Acute", "DTP"] * 3)
    groups = np.array(["A"] * 3 + ["B"] * 3 + ["C"] * 3)
    with pytest.raises(ValueError, match="State absent"):
        module.standardized_centroid_shift(np.ones((9, 2)), y, groups,
                                            np.ones((2, 2)), np.array(["Control", "Acute"]))
