"""Stage 34: source-only cohort, matrix and leakage-related integrity guards."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/34_source_only_rebaseline.py"
spec = importlib.util.spec_from_file_location("stage34_source_only", SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_data():
    meta = pd.DataFrame([
        {"gsm": f"{line}_{state}_{rep}", "cell_line": line,
         "candidate_group": state, "review_status": "PENDING_MANUAL_AND_MATRIX_CHECK",
         "mapping_status": "exact_title"}
        for line in ("H1975", "HCC2935", "HCC827", "PC9")
        for state in ("Control", "Acute", "DTP") for rep in range(3)
    ])
    protocol = {"nested_loco_protocol": {"folds": [
        {"outer_test_cell_line": line,
         "outer_test_gsm": meta.loc[meta.cell_line == line, "gsm"].tolist()}
        for line in ("H1975", "HCC2935", "HCC827", "PC9")
    ]}}
    frame = pd.DataFrame({"gene_id": ["A", "B", "C"], **{
        gsm: [1.0, 2.0, 3.0] for gsm in meta.gsm
    }})
    return meta, protocol, frame


def test_valid_source_only_cohort_and_expression():
    meta, protocol, frame = fake_data()
    m.validate_cohort(meta, protocol)
    genes, x = m.validate_matrix(frame, meta.gsm.tolist(), expected_genes=3)
    assert genes.tolist() == ["A", "B", "C"]
    assert x.shape == (36, 3)


def test_rejects_target_sample_in_protocol():
    meta, protocol, _ = fake_data()
    protocol["nested_loco_protocol"]["folds"][0]["outer_test_gsm"][0] = "GSM_FROM_TARGET"
    with pytest.raises(ValueError, match="GSM"):
        m.validate_cohort(meta, protocol)


def test_rejects_relabeling_and_unverified_mapping():
    meta, protocol, _ = fake_data()
    meta.loc[0, "review_status"] = "VERIFIED"
    with pytest.raises(ValueError, match="review status"):
        m.validate_cohort(meta, protocol)
    meta.loc[0, "review_status"] = "PENDING_MANUAL_AND_MATRIX_CHECK"
    meta.loc[0, "mapping_status"] = "alias_candidate"
    with pytest.raises(ValueError, match="exact-title"):
        m.validate_cohort(meta, protocol)


def test_rejects_duplicate_genes_and_sample_columns():
    meta, _, frame = fake_data()
    bad = frame.copy()
    bad.loc[1, "gene_id"] = "A"
    with pytest.raises(ValueError, match="Duplicate"):
        m.validate_matrix(bad, meta.gsm.tolist(), expected_genes=3)
    with pytest.raises(ValueError, match="matrix"):
        m.validate_matrix(frame.drop(columns=[meta.gsm.iloc[0]]), meta.gsm.tolist(), expected_genes=3)


def test_rejects_nonfinite_expression():
    meta, _, frame = fake_data()
    frame.loc[2, meta.gsm.iloc[3]] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        m.validate_matrix(frame, meta.gsm.tolist(), expected_genes=3)


def test_historical_subset_comparison_does_not_need_target_matrix(tmp_path):
    meta, _, frame = fake_data()
    genes, x = m.validate_matrix(frame, meta.gsm.tolist(), expected_genes=3)
    old = frame.loc[frame.gene_id.isin(["A", "C"])].copy()
    path = tmp_path / m.OLD_SOURCE
    path.parent.mkdir(parents=True)
    old.to_csv(path, sep="\t", compression="gzip", index=False)
    # The production check also enforces the frozen historical 18,013-gene count.
    with pytest.raises(ValueError, match="historical shared-gene"):
        m.verify_old_common_is_exact_subset(tmp_path, genes, x, meta)
