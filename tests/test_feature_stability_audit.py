"""Checks for stage-26 fold-selection stability calculations."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "26_feature_stability_audit.py"
spec = importlib.util.spec_from_file_location("stage26_audit", MODULE_PATH)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_jaccard_overlap():
    assert mod.overlap({1, 2, 3}, {2, 3, 4}) == {
        "intersection": 2, "union": 4, "jaccard": 0.5, "selection_fraction_shared": 2/3
    }


def test_frequency_summary_exact():
    gene_ids = np.array(["A", "B", "C", "D", "E"])
    summary = mod.frequency_summary({
        "F1": {0, 1, 2}, "F2": {0, 2, 3}, "F3": {0, 1, 3}, "F4": {0, 1, 4}
    }, gene_ids)
    assert summary["all_fold_intersection_genes"] == 1
    assert summary["all_fold_gene_ids"] == ["A"]
    assert summary["frequency_distribution"] == {"1": 1, "2": 2, "3": 1, "4": 1}
    assert summary["union_genes"] == 5


def test_pairwise_count():
    rows = mod.build_pairwise({str(i): {i, i+1} for i in range(4)})
    assert len(rows) == 6
    assert rows[0]["jaccard"] == pytest.approx(1/3)


def test_selection_hash_is_order_sensitive():
    assert mod.sha256_indices(np.array([0, 3, 17])) != mod.sha256_indices(np.array([3, 0, 17]))


def test_feature_selection_reproducible_and_non_leaky():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(36, 45))
    y = np.tile(np.repeat(["Control", "Acute", "DTP"], 3), 4)
    groups = np.repeat(["A", "B", "C", "D"], 9)
    x[y == "DTP", 0] += 3
    x[:, 1] += np.repeat([0, 1, 2, 3], 9)
    tr = groups != "D"
    for selector in mod.SELECTORS:
        original, same_sign = mod.rank_features(x[tr], y[tr], groups[tr], selector, 6, 0.5)
        assert len(original) == 6 and len(set(original)) == 6
        assert 0 <= same_sign <= 1
        changed = x.copy()
        changed[~tr] *= 1e6
        shifted_y = y.copy()
        shifted_y[~tr] = "Control"
        rerun, _ = mod.rank_features(changed[tr], shifted_y[tr], groups[tr], selector, 6, 0.5)
        np.testing.assert_array_equal(original, rerun)


def test_reject_invalid_selector():
    with pytest.raises(ValueError, match="Unknown feature selector"):
        mod.rank_features(np.ones((3, 2)), np.array(["A"]*3), np.array(["A"]*3), "wrong", 2, 0.5)
