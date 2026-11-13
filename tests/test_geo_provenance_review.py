"""Stage-30 offline source mapping, fail-closed and reporting tests."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/30_geo_provenance_review.py"
SPEC = importlib.util.spec_from_file_location("stage30", SCRIPT)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fake_rows():
    rows = []
    for sample in M.expected_manifest().values():
        rows.append({"accession": sample["accession"], "gsm": sample["gsm"],
                     "cell_line": sample["cell_line"], "state": sample["state"],
                     "title": sample["geo_title"],
                     "evaluation_role": sample["expected_evaluation_role"],
                     "label_review": M.PENDING, "source_file": "test"})
    return rows


def test_manifest_is_complete():
    manifest = M.expected_manifest()
    assert len(manifest) == 48
    assert len([x for x in manifest.values() if x["accession"] == "GSE193258"]) == 36
    assert len([x for x in manifest.values() if x["state"] == "Resistant"]) == 3


def test_known_geo_sample_titles_and_replicates():
    manifest = M.expected_manifest()
    assert manifest["GSM5778052"]["geo_title"] == "PC9_osi_DTP_3 [RNA-seq]"
    assert manifest["GSM8083380"]["geo_title"] == "PC9 Osim 9d rep 1"
    assert manifest["GSM8083383"]["expected_evaluation_role"] == "exploratory_only"


def test_clean_rows_match_without_auto_approval():
    results = M.verify_mapping(fake_rows(), M.expected_manifest())
    assert len(results) == 48
    assert all(not row["scientifically_certified_for_cross_study_pooling"] for row in results)


@pytest.mark.parametrize("field,value", [("state", "DTP"), ("title", "fake sample"),
                                          ("cell_line", "HCC827"), ("evaluation_role", "source_candidate")])
def test_mismatched_target_fails_closed(field, value):
    rows = fake_rows()
    row = next(r for r in rows if r["gsm"] == "GSM8083374")
    row[field] = value
    with pytest.raises(ValueError, match="GEO metadata mismatch"):
        M.verify_mapping(rows, M.expected_manifest())


def test_modified_review_status_rejected():
    rows = fake_rows()
    rows[0]["label_review"] = "APPROVED"
    with pytest.raises(ValueError, match="review status"):
        M.verify_mapping(rows, M.expected_manifest())


def test_missing_sample_rejected():
    with pytest.raises(ValueError, match="GSM set differs"):
        M.verify_mapping(fake_rows()[:-1], M.expected_manifest())


def test_markdown_does_not_claim_validation():
    rows = fake_rows()
    report = M.make_report(rows, M.verify_mapping(rows, M.expected_manifest()), "x" * 64)
    text = M.make_markdown(report)
    assert "cannot" in text
    assert "log2(TPM + 0.01)" in text
    assert report["cross_study_status"]["independent_external_validation_established"] is False


def test_local_integration_offline(tmp_path):
    rows = fake_rows()
    base = tmp_path
    p = base / M.INPUT_CSV
    p.parent.mkdir(parents=True)
    with p.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    old = {"analysis": "stage29_provenance_and_expression_scale_audit", "source_n": 36,
           "target_n": 12, "genes": 18013, "input_sha256": {"samples.csv": sha},
           "all_48_labels_pending_manual_review": True,
           "sample_scale_summaries": [dict(accession=r["accession"], gsm=r["gsm"],
                                            cell_line=r["cell_line"], state=r["state"]) for r in rows]}
    q = base / M.INPUT_AUDIT
    q.parent.mkdir(parents=True)
    q.write_text(json.dumps(old), encoding="utf-8")
    for r in rows:
        if r["accession"] == "GSE255958":
            r["source_file"] = f"{r['gsm']}_fixture.results.gz"
    # Fixture source file map must agree with local CSV for this technical check.
    with p.open("w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    sha2 = hashlib.sha256(p.read_bytes()).hexdigest()
    old["input_sha256"]["samples.csv"] = sha2
    q.write_text(json.dumps(old), encoding="utf-8")
    comp = {"source_study": "GSE193258", "target_study": "GSE255958",
            "status": "gene_names_overlap_found", "samples": [
        {"gsm": r["gsm"], "file": r["source_file"]} for r in rows
        if r["accession"] == "GSE255958"]}
    c = base / M.INPUT_COMPAT; c.write_text(json.dumps(comp), encoding="utf-8")
    result = M.run(base)
    assert result["n_geo_title_label_matches"] == 48
    assert (base / M.OUT_JSON).is_file()
    assert (base / M.OUT_MD).is_file()
