"""Core numerical checks of the washout contrast calculation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/39_washout_contrast_audit.py"
spec = importlib.util.spec_from_file_location("washout_contrast", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample_groups():
    # Four cells per condition? No: three biological-label replicates per arm.
    a = np.repeat([[0.0, 0.0, 0.0]], 3, axis=0)
    d = np.repeat([[2.0, 0.0, 0.0]], 3, axis=0)
    short = np.repeat([[1.5, 0.0, 0.0]], 3, axis=0)
    long = np.repeat([[0.5, 0.0, 0.0]], 3, axis=0)
    return a, d, short, long


def test_retention_is_centroid_relative_not_probability():
    a, d, short, long = sample_groups()
    result = module.contrast_summary(a, d, short, long, np.array([1.0, 0, 0]))
    assert result["on_drug_DTP_reference"]["retention_along_heldout_DTP_Acute"] == pytest.approx(1)
    assert result["short_washout"]["retention_along_heldout_DTP_Acute"] == pytest.approx(.75)
    assert result["long_washout"]["retention_along_heldout_DTP_Acute"] == pytest.approx(.25)
    assert result["heldout_DTP_Acute_alignment_to_training_direction"] == pytest.approx(1)


def test_retention_can_be_negative_or_above_one():
    a, d, short, long = sample_groups()
    short[:, 0] = 3
    long[:, 0] = -1
    r = module.contrast_summary(a, d, short, long, np.array([1.0, 0, 0]))
    assert r["short_washout"]["retention_along_heldout_DTP_Acute"] == pytest.approx(1.5)
    assert r["long_washout"]["retention_along_heldout_DTP_Acute"] == pytest.approx(-.5)


def test_rejects_invalid_reference_or_replicate_count():
    a, d, short, long = sample_groups()
    with pytest.raises(ValueError, match="essentially zero"):
        module.contrast_summary(a, a, short, long, np.array([1.0, 0, 0]))
    with pytest.raises(ValueError, match="three"):
        module.contrast_summary(a[:2], d, short, long, np.array([1.0, 0, 0]))


def test_train_only_selection_and_direction():
    rng = np.random.default_rng(4)
    x, y, g = [], [], []
    for group in ("A", "B", "C"):
        for state in ("Acute", "DTP"):
            for _ in range(3):
                row = rng.normal(size=9)
                row[:4] += 2.0 * (state == "DTP")
                x.append(row); y.append(state); g.append(group)
    x, y, g = np.array(x), np.array(y), np.array(g)
    pipe, ids, z = module.fit_training_features(x, y, k=4)
    u = module.normalized_direction(z, y, g)
    assert len(ids) == 4 and len(set(ids)) == 4
    assert np.linalg.norm(u) == pytest.approx(1.0)
    heldout_a = rng.normal(size=(3, 9))
    heldout_b = rng.normal(size=(3, 9)) + 20.0
    np.testing.assert_array_equal(ids, module.fit_training_features(x, y, k=4)[1])
    assert pipe.transform(heldout_a).shape == pipe.transform(heldout_b).shape == (3, 4)
