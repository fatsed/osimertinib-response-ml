"""Stage 32: nested validation rules, deterministic selection and leakage guards."""
from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/32_run_nested_loco.py"
STAGE31 = ROOT / "scripts/31_lock_evaluation_protocol.py"


def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = import_file(SCRIPT, "stage32_nested_loco")
p = import_file(STAGE31, "stage31_plan_for_stage32_tests")


def fake_source():
    rows = []
    for line in m.LINES:
        for state in m.STATES:
            for rep in range(3):
                rows.append({"gsm": f"{line}_{state}_{rep}", "cell_line": line, "state": state})
    return pd.DataFrame(rows)


def fold_fixture():
    meta = fake_source()
    rows = [{"accession": "GSE193258", **row} for row in meta.to_dict("records")]
    return meta, p.make_folds(rows)


def test_stage31_exact_candidates_are_accepted():
    m.validate_candidate_specs(p.candidate_specs())
    assert len(p.candidate_specs()) == 11


def test_candidate_change_is_rejected():
    specs = p.candidate_specs()
    specs[1]["parameters"]["C"] = 100.0
    with pytest.raises(ValueError, match="Candidate"):
        m.validate_candidate_specs(specs)


def test_outer_inner_membership_and_tampering():
    meta, folds = fold_fixture()
    m.validate_folds(folds, meta)
    tampered = json.loads(json.dumps(folds))
    heldout_gsm = tampered[0]["outer_test_gsm"][0]
    tampered[0]["inner_folds"][0]["train_gsm"][0] = heldout_gsm
    with pytest.raises(ValueError, match="Inner"):
        m.validate_folds(tampered, meta)


def test_scoring_labels_and_class_specific_recall():
    y = np.asarray(["Control"] * 3 + ["Acute"] * 3 + ["DTP"] * 3)
    pred = np.asarray(["Control"] * 3 + ["Acute"] * 3 + ["Acute"] * 3)
    result = m.scores(y, pred)
    assert result["DTP_recall"] == 0
    assert result["balanced_accuracy"] == pytest.approx(2 / 3)
    assert result["per_class_recall"]["Control"] == 1
    assert result["n_by_class"] == {"Control": 3, "Acute": 3, "DTP": 3}
    assert result["confusion_matrix"] == [[3, 0, 0], [0, 3, 0], [0, 3, 0]]


def test_selection_tie_break_and_global_order():
    def row(name, ba, dtp, f1):
        return {"id": name, "mean_inner_balanced_accuracy": ba,
                "mean_inner_DTP_recall": dtp, "mean_inner_macro_f1": f1,
                "inner_folds": [{}, {}, {}]}
    rows = [row("z", .8, .4, .7), row("a", .8, .5, .4), row("b", .8, .5, .6)]
    assert m.select_candidate(rows)["id"] == "b"
    assert m.select_candidate([row("z", .8, .5, .6), row("a", .8, .5, .6)])["id"] == "a"
    assert m.select_candidate([*rows, row("winner", .9, 0, 0)])["id"] == "winner"
    with pytest.raises(ValueError):
        m.select_candidate([row("same", .2, .2, .2)] * 2)


def test_outer_test_never_passed_to_any_fit():
    meta, folds = fold_fixture()
    # First synthetic coordinate encodes a cell line; it never changes on fitting.
    index_line = {line: i for i, line in enumerate(m.LINES)}
    x = np.array([[index_line[line], i] for i, line in enumerate(meta.cell_line)], dtype=float)
    calls = []

    class AuditedModel:
        def __init__(self, spec):
            self.spec = spec
        def fit(self, xx, yy, groups=None):
            assert len(xx) in (18, 27)
            assert set(yy) == set(m.STATES)
            calls.append((self.spec["id"], set(xx[:, 0].astype(int)), len(xx)))
            return self
        def predict(self, xx):
            return np.asarray(["Control"] * len(xx))

    fold = folds[3]  # PC9 is outer test
    result = m.evaluate_outer_fold(x, meta, fold, p.candidate_specs(),
                                   make_model_fn=lambda spec: AuditedModel(spec))
    assert len(calls) == 3 * 11 + 6
    assert all(3 not in lines for _, lines, _ in calls)
    assert sum(n == 18 for _, _, n in calls) == 33
    assert sum(n == 27 for _, _, n in calls) == 6
    assert result["n_outer_test"] == 9
    assert len(result["selected_by_family_inner_only"]) == 6
    assert len(result["candidates_inner_only"]) == 11
    assert all(v["metrics"]["n"] == 9 for v in result["family_outer_results"].values())


def test_all_36_outer_predictions_appear_once_in_summary():
    meta, folds = fold_fixture()
    outer = []
    for fold in folds:
        family_results = {}
        for family in m.FAMILIES:
            y = [s for g in fold["outer_test_gsm"] for s in [meta.set_index("gsm").loc[g, "state"]]]
            pred = ["Control"] * 9
            family_results[family] = {
                "selected_candidate_id": family,
                "metrics": m.scores(np.array(y), np.array(pred)),
                "predictions": [{"gsm": g, "actual": yy, "predicted": pp}
                                for g, yy, pp in zip(fold["outer_test_gsm"], y, pred)],
            }
        outer.append({"outer_test_cell_line": fold["outer_test_cell_line"],
                      "family_outer_results": family_results})
    summary = m.summarize_families(outer)
    assert all(summary[f]["pooled"]["n"] == 36 for f in m.FAMILIES)
    assert all(summary[f]["mean_outer_DTP_recall"] == 0 for f in m.FAMILIES)


def test_protocol_hash_guard_rejects_changed_inputs(tmp_path):
    (tmp_path / m.SOURCE_FILE).parent.mkdir(parents=True)
    (tmp_path / m.STAGE30).parent.mkdir(parents=True)
    (tmp_path / m.SOURCE_FILE).write_bytes(b"tampered")
    (tmp_path / m.META_FILE).write_bytes(b"tampered")
    (tmp_path / m.STAGE30).write_bytes(b"tampered")
    protocol = json.loads((ROOT / m.PROTOCOL).read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / m.BASELINE).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="hashes differ"):
        m.verify_manifest(protocol, baseline, tmp_path)


def test_target_expression_not_read(monkeypatch):
    protocol = json.loads((ROOT / m.PROTOCOL).read_text(encoding="utf-8"))
    original_read_csv = pd.read_csv
    seen = []
    def guard(path, *args, **kwargs):
        path_str = str(path).lower()
        assert "gse255958_common" not in path_str, "target expression read"
        seen.append(path_str)
        return original_read_csv(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", guard)
    x, meta = m.load_source(ROOT, protocol)
    assert x.shape == (36, 18013)
    assert len(meta) == 36
    assert len(seen) == 2
