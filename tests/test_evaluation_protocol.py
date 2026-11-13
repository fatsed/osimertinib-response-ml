"""Stage-31 tests: nested-LOCO membership, integrity, and independence flags."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'scripts/31_lock_evaluation_protocol.py'
SPEC = importlib.util.spec_from_file_location('stage31_protocol', MODULE_PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def demo_rows():
    rows = []
    for cell in M.CELL_LINES:
        for state in M.STATES:
            for i in range(3):
                rows.append(dict(accession='GSE193258', gsm=f'{cell}-{state}-{i}', cell_line=cell,
                                 state=state, evaluation_role='source_candidate', label_review=M.REVIEW_STATUS))
    for state in (*M.STATES, 'Resistant'):
        for i in range(3):
            rows.append(dict(accession='GSE255958', gsm=f'T-{state}-{i}', cell_line='PC9',
                             state=state, evaluation_role=('exploratory_only' if state == 'Resistant' else 'target_candidate'),
                             label_review=M.REVIEW_STATUS))
    return rows


def test_folds_have_no_outer_inner_leakage():
    rows = demo_rows()
    M.validate_cohort(rows)
    folds = M.make_folds(rows)
    assert len(folds) == 4
    for outer in folds:
        tr = set(outer['outer_train_gsm']); te = set(outer['outer_test_gsm'])
        assert (len(tr), len(te)) == (27, 9)
        assert tr.isdisjoint(te)
        assert len(outer['inner_folds']) == 3
        assert {i['validation_cell_line'] for i in outer['inner_folds']} == set(outer['outer_training_cell_lines'])
        for inner in outer['inner_folds']:
            itr = set(inner['train_gsm']); iv = set(inner['validation_gsm'])
            assert (len(itr), len(iv)) == (18, 9)
            assert itr.isdisjoint(iv)
            assert itr.union(iv) == tr
            assert itr.isdisjoint(te) and iv.isdisjoint(te)


def test_known_group_counts():
    rows = demo_rows()
    assert len(rows) == 48
    report = M.build_report(rows, {'samples.csv':'abc'})
    assert report['n_source'] == 36
    assert report['implementation_status'] == 'SPLITS_AND_POLICY_ONLY_NOT_NESTED_MODEL_RESULTS'
    assert report['code_does_not_fit_or_select_models']
    assert report['evaluation_governance']['GSE255958'].startswith('previously inspected')


@pytest.mark.parametrize('mutator', [
    lambda rows: rows.pop(),
    lambda rows: rows.__setitem__(1, dict(rows[1], gsm=rows[0]['gsm'])),
    lambda rows: rows.__setitem__(0, dict(rows[0], cell_line='PC9')),
    lambda rows: rows.__setitem__(0, dict(rows[0], label_review='APPROVED')),
    lambda rows: rows.__setitem__(40, dict(rows[40], evaluation_role='exploratory_only')),
])
def test_invalid_cohort_fails_closed(mutator):
    rows = demo_rows()
    mutator(rows)
    with pytest.raises(ValueError):
        M.validate_cohort(rows)


def test_reproducibility_under_reordered_metadata():
    rows = demo_rows()
    assert M.make_folds(rows) == M.make_folds(list(reversed(rows)))


def test_document_disclaims_independence_and_no_training():
    d = M.build_report(demo_rows(), {'samples.csv':'x'})
    md = M.make_markdown(d)
    assert 'not an untouched independent test' in md
    assert 'no model fitted' in md
    assert '18 samples' in md
    assert 'PC9' in md


def test_input_integrity_and_target_not_required(tmp_path):
    root = tmp_path
    src = root / M.INPUT_SOURCE
    meta = root / M.INPUT_META
    src.parent.mkdir(parents=True)
    src.write_bytes(b'fixture-source-file')
    rows = demo_rows()
    with meta.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    source_hash = M.digest(src)
    meta_hash = M.digest(meta)
    baseline = root / M.STAGE09
    baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'source':'GSE193258','target':'GSE255958',
                       'input_sha256': {meta.name:meta_hash, src.name:source_hash}}))
    stage30 = root / M.STAGE30
    stage30.write_text(json.dumps({'analysis':'stage30_offline_geo_provenance_documentation_review',
                       'n_checked':48, 'n_geo_title_label_matches':48, 'input_samples_sha256':meta_hash,
                       'cross_study_status':{'independent_external_validation_established':False,
                           'target_previously_examined_not_pristine_test':True,
                           'all_labels_keep_pending_review_status':True}}))
    report = M.run(root)  # target expression file never exists in this fixture
    assert len(report['nested_loco_protocol']['folds']) == 4
    assert (root / M.OUTPUT_JSON).exists()
    assert (root / M.OUTPUT_MD).exists()
    # Same-stage verification fails if source metadata changes.
    meta.write_text(meta.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        M.load_inputs(root)


def test_stage30_cannot_be_reinterpreted_as_independent(tmp_path):
    rows = demo_rows()
    assert all(r['label_review'] == M.REVIEW_STATUS for r in rows)
    d = M.build_report(rows, {})
    assert d['historic_results_were_seen_before_this_plan'] is True
    assert 'not a preregistered' in d['protocol_type']
    assert 'future_untouched_study' in d['evaluation_governance']
