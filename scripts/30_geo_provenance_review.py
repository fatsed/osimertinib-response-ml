"""Stage 30: offline GEO provenance cross-check for the 48 previously reviewed samples.

This code stores *documented* GEO series/sample evidence; it does not download
raw data, certify experimental independence, harmonize expression, or train models.
The evidence table was transcribed from the NCBI GEO series pages (see SOURCES).
Run at the repository root: python scripts/30_geo_provenance_review.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SOURCES = {
    "GSE193258_series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258",
    "GSE193258_sample_processing": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5777996",
    "GSE255958_series": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255958",
    "GSE255958_sample_processing": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8083374",
    "GSE255958_DTP_9d": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8083380",
}
PENDING = "PENDING_MANUAL_AND_MATRIX_CHECK"
INPUT_CSV = Path("data/processed/cross_study/samples.csv")
INPUT_AUDIT = Path("results/dcsl_stage29_provenance_scale_audit.json")
INPUT_COMPAT = Path("results/gse255958_compatibility.json")
OUT_JSON = Path("results/dcsl_stage30_geo_provenance.json")
OUT_MD = Path("docs/stage30_geo_provenance_review.md")

# Explicit, series-derived contiguous GSM blocks. Titles encode replicates 1..3.
SOURCE_BLOCKS = (
    ("H1975", "Control", 5777996, "DMSO"),
    ("H1975", "Acute", 5778002, "osi_acute"),
    ("H1975", "DTP", 5778005, "osi_DTP"),
    ("HCC2935", "Control", 5778011, "DMSO"),
    ("HCC2935", "Acute", 5778017, "osi_acute"),
    ("HCC2935", "DTP", 5778020, "osi_DTP"),
    ("HCC827", "Control", 5778026, "DMSO"),
    ("HCC827", "Acute", 5778032, "osi_acute"),
    ("HCC827", "DTP", 5778035, "osi_DTP"),
    ("PC9", "Control", 5778041, "DMSO"),
    ("PC9", "Acute", 5778047, "osi_acute"),
    ("PC9", "DTP", 5778050, "osi_DTP"),
)
TARGET_BLOCKS = (
    ("Control", 8083374, "PC9 ctrl 48h rep {}"),
    ("Acute", 8083377, "PC9 Osim 48h rep {}"),
    ("DTP", 8083380, "PC9 Osim 9d rep {}"),
    ("Resistant", 8083383, "PC9-R Osim 48h rep {}"),
)


def expected_manifest() -> dict[str, dict[str, str]]:
    """Record explicitly identified GEO titles and selected study groups."""
    expected = {}
    for cell_line, state, first, token in SOURCE_BLOCKS:
        for replicate in range(1, 4):
            gsm = f"GSM{first + replicate - 1}"
            expected[gsm] = {
                "accession": "GSE193258", "gsm": gsm, "cell_line": cell_line,
                "state": state, "replicate_label": str(replicate),
                "geo_title": f"{cell_line}_{token}_{replicate} [RNA-seq]",
                "expected_evaluation_role": "source_candidate",
                "geo_series_url": SOURCES["GSE193258_series"],
            }
    for state, first, template in TARGET_BLOCKS:
        for replicate in range(1, 4):
            gsm = f"GSM{first + replicate - 1}"
            expected[gsm] = {
                "accession": "GSE255958", "gsm": gsm, "cell_line": "PC9",
                "state": state, "replicate_label": str(replicate),
                "geo_title": template.format(replicate),
                "expected_evaluation_role": "exploratory_only" if state == "Resistant" else "target_candidate",
                "geo_series_url": SOURCES["GSE255958_series"],
            }
    if len(expected) != 48:
        raise ValueError("Expected exactly 48 unique GEO sample identifiers")
    return expected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_local_samples(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"accession", "gsm", "cell_line", "state", "title",
                    "evaluation_role", "label_review", "source_file"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Stage-08 samples.csv lacks required columns")
        rows = list(reader)
    if len(rows) != 48 or len({r["gsm"] for r in rows}) != 48:
        raise ValueError("Expected exactly 48 unique samples in stage-08 metadata")
    return rows


def verify_mapping(rows: list[dict[str, str]], manifest: dict[str, dict[str, str]]) -> list[dict]:
    """Fail closed on mismatches; no automatic label promotion."""
    if {r["gsm"] for r in rows} != set(manifest):
        raise ValueError("GSM set differs from the 48 accession-level GEO records")
    findings = []
    for row in rows:
        known = manifest[row["gsm"]]
        fields = ("accession", "cell_line", "state", "title", "evaluation_role")
        expected_values = (known["accession"], known["cell_line"], known["state"],
                           known["geo_title"], known["expected_evaluation_role"])
        matched = all(row[field] == expected for field, expected in zip(fields, expected_values))
        if not matched:
            detail = {field: (row[field], expected) for field, expected in zip(fields, expected_values)
                      if row[field] != expected}
            raise ValueError(f"GEO metadata mismatch for {row['gsm']}: {detail}")
        if row["label_review"] != PENDING:
            raise ValueError(f"Unexpected promoted/modified review status for {row['gsm']}")
        findings.append({
            **known, "geo_title_and_candidate_label_match": True,
            "label_review_unchanged": row["label_review"],
            "biological_replicate_independently_verified": False,
            "scientifically_certified_for_cross_study_pooling": False,
        })
    return findings


def verify_old_audit(previous: dict, rows: list[dict], input_hash: str) -> None:
    if previous.get("analysis") != "stage29_provenance_and_expression_scale_audit":
        raise ValueError("Unexpected stage-29 report type")
    if (previous.get("source_n"), previous.get("target_n"), previous.get("genes")) != (36, 12, 18013):
        raise ValueError("Previous cohort dimensions changed")
    if previous.get("input_sha256", {}).get("samples.csv") != input_hash:
        raise ValueError("Stage-08 samples.csv hash changed since stage 29")
    recorded = {(r["accession"], r["gsm"], r["cell_line"], r["state"])
                for r in previous.get("sample_scale_summaries", [])}
    actual = {(r["accession"], r["gsm"], r["cell_line"], r["state"]) for r in rows}
    if len(recorded) != 48 or actual != recorded:
        raise ValueError("Local metadata no longer matches stage-29 sample summaries")
    if previous.get("all_48_labels_pending_manual_review") is not True:
        raise ValueError("Stage-29 pending review status was modified")


def check_target_files(rows: list[dict], compatibility: dict) -> None:
    if compatibility.get("source_study") != "GSE193258" or compatibility.get("target_study") != "GSE255958":
        raise ValueError("Unexpected stage-07 study identifiers")
    if compatibility.get("status") not in (None, "gene_names_overlap_found"):
        raise ValueError("Unexpected stage-07 compatibility status")
    sample_files = {r["gsm"]: r["file"] for r in compatibility.get("samples", [])}
    if len(sample_files) != 12:
        raise ValueError("Stage-07 target file list is incomplete")
    for row in rows:
        if row["accession"] == "GSE255958" and sample_files.get(row["gsm"]) != row["source_file"]:
            raise ValueError(f"Target file mapping mismatch: {row['gsm']}")


def make_report(rows: list[dict], findings: list[dict], digest: str) -> dict:
    counts = dict(sorted(Counter((r["accession"], r["state"]) for r in rows).items()))
    return {
        "analysis": "stage30_offline_geo_provenance_documentation_review",
        "status": "GEO_TITLES_MATCH_LOCAL_METADATA__BIOLOGICAL_AND_CROSS_STUDY_VALIDATION_OPEN",
        "source_evidence_urls": SOURCES,
        "n_checked": len(findings),
        "n_geo_title_label_matches": sum(int(r["geo_title_and_candidate_label_match"]) for r in findings),
        "input_samples_sha256": digest,
        "counts_by_accession_and_state": {f"{a}:{state}": count for (a, state), count in counts.items()},
        "source_assay": {
            "study": "GSE193258", "reference_assembly_documented": "hg38",
            "quantification_documented": "HISAT2 v2.1.0; Salmon v0.8.2; Ensembl v79; tximport lengthScaledTPM",
            "exact_GEO_documented_expression_transform": "log2(TPM + 0.01)",
            "source_transform_formula_recorded_in_GEO": True,
            "selected_states": {"Control": "DMSO 24h", "Acute": "500 nM osimertinib 24h",
                                "DTP": "500 nM osimertinib 21d"},
            "biological_triplicates_claimed_in_series": True,
        },
        "target_assay": {
            "study": "GSE255958", "reference_assembly_documented": "hg19",
            "quantification_documented": "STAR v2.4.2a; RSEM v1.2.29; TPM supplied",
            "project_transform": "stage 08 applied log2(RSEM TPM + 0.01)",
            "selected_states": {"Control": "0.1% DMSO 48h", "Acute": "2 uM osimertinib 48h in selected PC9 metadata",
                                "DTP": "2 uM osimertinib 9d in selected PC9 metadata",
                                "Resistant": "resistant PC9 derivatives with 48h osimertinib; excluded from 3-class task"},
            "replicate_independence_not_proven_by_rep_labels": True,
        },
        "cross_study_status": {
            "equal_transform_formula_alone_proves_scale_equivalence": False,
            "gene_annotations_and_reference_genomes_differ": True,
            "treatment_durations_and_doses_differ": True,
            "target_previously_examined_not_pristine_test": True,
            "all_labels_keep_pending_review_status": True,
            "independent_external_validation_established": False,
            "suitable_roles_now": {
                "GSE193258": "exploratory source LOCO modeling; GEO title documented, not independently validated",
                "GSE255958": "previously inspected cross-study descriptive and exploratory sensitivity only",
                "GSE255958_resistant": "exploratory only; not eligible for 3-state DTP labels",
            },
        },
        "samples": findings,
        "auto_changed_any_metadata": False,
        "model_training_or_tuning_performed": False,
    }


def make_markdown(report: dict) -> str:
    src = report["source_assay"]
    tgt = report["target_assay"]
    return f"""# Stage 30 — GEO provenance and cohort decision record

**Status:** GEO title/sample-label alignment documented (n={report['n_geo_title_label_matches']}/48); independent biological validation and cross-study comparability **remain unresolved**.

## Reproducibility
- `samples.csv` SHA-256: `{report['input_samples_sha256']}`
- All 48 sample GSMs, titles, cell lines, provisional classes and evaluation roles exactly matched the NCBI GEO series sample lists as transcribed into the stage-30 reference manifest.
- The GSE255958 target member filenames match the previously recorded stage-07 compatibility report.
- The stage-29 sample list and hash have been checked; **no labels or matrices have been changed**.

## Protocol/assay evidence
| Attribute | GSE193258 | GSE255958 (PC9 subset) |
|---|---|---|
| Control | DMSO 24h | 0.1% DMSO 48h |
| Acute | 500 nM osimertinib 24h | selected metadata: 2 µM osimertinib 48h |
| DTP | 500 nM osimertinib 21d | selected metadata: 2 µM osimertinib 9d |
| Quantification | HISAT2 / Salmon / tximport | STAR / RSEM |
| Genome reference | {src['reference_assembly_documented']} | {tgt['reference_assembly_documented']} |
| Expression | GEO explicitly reports `log2(TPM + 0.01)` | GEO provides RSEM TPM; project stage-08 applies `log2(TPM + 0.01)` |

## Scientific interpretation
1. **Resolved:** Exact transformation formula is *documented on the GSE193258 GEO sample record*. Stage 29's uncertainty about the published formula can be amended; there is no need to infer it from the floor alone.
2. **Not resolved:** Equal mathematical expression transforms do not remove different quantification pipelines, annotation versions, genome assemblies, exposure regimes or batch effects. Matching gene symbols alone does not guarantee identical feature definitions.
3. The 36 GSE193258 samples are suitable for *exploratory* within-study LOCO comparisons using the documented source experiment, subject to experimental-replicate and sample-to-matrix review.
4. The nine common-state PC9 samples from GSE255958 have been repeatedly inspected during development. They **cannot** be described as an untouched independent test set. The three resistant samples remain exploratory-only.
5. Source GSE193258 states biological triplicates in the series design; a `rep` title suffix in the target is not by itself independent proof of biological replication.
6. Do not promote `PENDING_MANUAL_AND_MATRIX_CHECK` to a certified status automatically. A domain expert should review primary methods, provenance and per-sample lab metadata.

## References (GEO primary evidence)
- [GSE193258 series]({SOURCES['GSE193258_series']})
- [GSM5777996: processing pipeline and log2(TPM + 0.01)]({SOURCES['GSE193258_sample_processing']})
- [GSE255958 series]({SOURCES['GSE255958_series']})
- [GSM8083374: STAR/RSEM, hg19]({SOURCES['GSE255958_sample_processing']})
- [GSM8083380: selected PC9 DTP, 2 µM 9d]({SOURCES['GSE255958_DTP_9d']})

## Next decision before claims of new generalization
Retain source-only LOCO as descriptive development results; make a prospective validation plan with independent, compatible data and separate pre-specified model development from final testing. Do **not** change existing labels, expression matrices, or old metrics based on this audit.
"""


def run(root: Path) -> dict:
    samples_path = root / INPUT_CSV
    rows = read_local_samples(samples_path)
    stage29 = json.loads((root / INPUT_AUDIT).read_text(encoding="utf-8"))
    compat = json.loads((root / INPUT_COMPAT).read_text(encoding="utf-8"))
    file_hash = sha256(samples_path)
    verify_old_audit(stage29, rows, file_hash)
    check_target_files(rows, compat)
    findings = verify_mapping(rows, expected_manifest())
    report = make_report(rows, findings, file_hash)
    jpath, mpath = root / OUT_JSON, root / OUT_MD
    jpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    jpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mpath.write_text(make_markdown(report), encoding="utf-8")
    print(f"GEO title/GSM/state/role matches: {report['n_geo_title_label_matches']}/48")
    print("Source expression formula documented: log2(TPM + 0.01)")
    print("Different reference assemblies: source hg38; target hg19")
    print("Review still pending; no independent validation or new modeling performed")
    print(f"Saved report: {jpath}")
    print(f"Saved evidence memo: {mpath}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
