"""Stage 29: input integrity, provenance, and descriptive-QC tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/29_provenance_scale_audit.py"
spec = importlib.util.spec_from_file_location("stage29_audit", SCRIPT)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def example_data():
    # 4 source cell lines, 3 states, 3 replicates = 36; 12 target PC9 samples.
    src_names = [f"S{i:02d}" for i in range(36)]
    tgt_names = [f"T{i:02d}" for i in range(12)]
    src = pd.DataFrame(np.zeros((3, 36)), index=["g1", "g2", "g3"], columns=src_names)
    tgt = pd.DataFrame(np.zeros((3, 12)), index=["g1", "g2", "g3"], columns=tgt_names)
    rows = []
    j = 0
    for line in ("H1975", "HCC2935", "HCC827", "PC9"):
        for state in module.STATES:
            for _ in range(3):
                rows.append(dict(gsm=src_names[j], accession="GSE193258", cell_line=line,
                                 state=state, label_review=module.KNOWN_STATUS,
                                 treatment="x", title="x", evaluation_role="source_candidate"))
                j += 1
    j = 0
    for state in (*module.STATES, "Resistant"):
        for _ in range(3):
            rows.append(dict(gsm=tgt_names[j], accession="GSE255958", cell_line="PC9",
                             state=state, label_review=module.KNOWN_STATUS,
                             treatment="x", title="x", evaluation_role=("exploratory_only" if state == "Resistant" else "target_candidate")))
            j += 1
    return src, tgt, pd.DataFrame(rows)


def test_metadata_validation_requires_proper_sample_roles():
    source, target, meta = example_data()
    assert module.check_metadata(meta, source, target)["every_label_still_pending_review"]
    meta.loc[meta.state == "Resistant", "evaluation_role"] = "target_candidate"
    with pytest.raises(ValueError):
        module.check_metadata(meta, source, target)


def test_label_status_is_not_certification():
    source, target, meta = example_data()
    meta.loc[0, "label_review"] = "REVIEWED"
    assert not module.check_metadata(meta, source, target)["every_label_still_pending_review"]


def test_gene_order_cannot_silently_change():
    source, target, meta = example_data()
    target = target.iloc[::-1]
    with pytest.raises(ValueError, match="Common genes"):
        module.compare_state_centroids(source, target, meta[meta.accession == "GSE193258"],
                                       meta[meta.accession == "GSE255958"])


def test_pc9_comparisons_are_descriptive_unpaired():
    source, target, meta = example_data()
    # Distinguish genes for Spearman, as real expression profiles do.
    for gene, val in zip(source.index, [1., 2., 3.]):
        source.loc[gene] = val
        target.loc[gene] = val + .5
    result = module.compare_state_centroids(source, target, meta[meta.accession == "GSE193258"],
                                            meta[meta.accession == "GSE255958"])
    for state in module.STATES:
        assert result[state]["source_n"] == 3
        assert result[state]["target_n"] == 3
        assert result[state]["median_abs_gene_mean_difference_in_log2_units"] == pytest.approx(.5)
        assert result[state]["gene_mean_spearman_rho"] == pytest.approx(1.)


def test_floor_uses_candidate_value_and_no_imputation():
    source, _, meta = example_data()
    source.iloc[:, :] = np.arange(3)[:, None]
    source.iloc[0, 0] = module.FLOOR
    summary = module.sample_summary(source, meta[meta.accession == "GSE193258"])
    assert summary[0]["fraction_at_candidate_0_01_floor"] == pytest.approx(1/3)
    assert summary[1]["fraction_at_candidate_0_01_floor"] == pytest.approx(0)


def test_reject_nonfinite_expression(tmp_path):
    file = tmp_path / "values.tsv.gz"
    pd.DataFrame({"gene_id": ["g1", "g2"], "S": [1, np.inf]}).to_csv(file, index=False, sep="\t", compression="gzip")
    with pytest.raises(ValueError, match="non-finite"):
        module.read_expression(file)


def test_hashes_guard_against_unrecorded_input_change(tmp_path):
    data_dir = tmp_path / module.INPUT_DIR
    data_dir.mkdir(parents=True)
    hashes = {}
    for filename in module.INPUT_FILES:
        path = data_dir / filename
        path.write_bytes(filename.encode())
        hashes[filename] = module.digest(path)
    report_dir = tmp_path / "results"
    report_dir.mkdir(parents=True)
    (tmp_path / module.BASELINE_REPORT).write_text(json.dumps({"input_sha256": hashes}), encoding="utf-8")
    assert module.check_input_hashes(tmp_path) == hashes
    (data_dir / module.INPUT_FILES[1]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.check_input_hashes(tmp_path)
