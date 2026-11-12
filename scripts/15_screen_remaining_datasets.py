from __future__ import annotations

import argparse
import csv
import gzip
import json
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("results/gse_remaining_screening.json")

STUDIES = {
    "GSE254485": {
        "order": 1,
        "purpose": "Time-course and acquired-resistance analysis",
        "assay": "bulk RNA-seq",
        "expected_gsm": 22,
        "candidate_role": "timecourse_candidate",
        "files": ["GSE254485_RNAseq_featureCounts_final.txt.gz"],
        "cautions": [
            "Day 28/35/56 labels do not automatically mean DTP.",
            "Time-course and resistance states must not be treated as interchangeable.",
            "Expression-column aliases need scientific verification.",
        ],
    },
    "GSE335846": {
        "order": 2,
        "purpose": "Cell-cycle phase control in treated PC9",
        "assay": "sorted-cell RNA-seq",
        "expected_gsm": 9,
        "candidate_role": "cell_cycle_candidate",
        "files": ["GSE335846_RNA-seq_for_GEO.txt.gz"],
        "cautions": [
            "Day 0/28 and G1/S/G2 form an unbalanced design.",
            "Different phase or clone counts must not be assumed independent or paired.",
            "Title matching does not establish biological replicate independence.",
        ],
    },
    "GSE202859": {
        "order": 3,
        "purpose": "Parental/resistant and SMARCA4 perturbation controls",
        "assay": "bulk RNA-seq",
        "expected_gsm": 52,
        "candidate_role": "resistance_candidate",
        "files": ["GSE202859_TPMs_allsamples.txt.gz"],
        "cautions": [
            "Resistant is not equivalent to DTP.",
            "Knockdown and drug-treatment comparisons require separate analysis.",
            "Expression-column aliases need scientific verification.",
        ],
    },
    "GSE282074": {
        "order": 4,
        "purpose": "Single-cell drug-exposure and technology comparisons",
        "assay": "single-cell RNA-seq with mixed experimental groups",
        "expected_gsm": 32,
        "candidate_role": "technology_benchmark_only",
        "files": ["GSE282074_RAW.tar"],
        "cautions": [
            "Multiple TAR members per GSM are not independent biological samples.",
            "Study contains technology controls and more than one drug.",
            "Individual cells must not be used as independent train/test samples.",
        ],
    },
    "GSE150949": {
        "order": 5,
        "purpose": "Cycling and non-cycling persister comparisons",
        "assay": "single-cell RNA-seq",
        "expected_gsm": 19,
        "candidate_role": "single_cell_candidate",
        "files": [
            "GSE150949_metaData_with_lineage.txt.gz",
            "GSE150949_pc9_count_matrix.csv.gz",
        ],
        "cautions": [
            "GSM experiments and thousands of cell barcodes are different units.",
            "Cell-level metadata and biological-replicate identity need joining.",
            "A cell-wise random split would overstate generalization.",
        ],
    },
}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required stage 03/04 report not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return list(reader)


def file_check(path: Path) -> dict:
    info = {"file": path.name, "exists": path.is_file()}
    if not path.is_file():
        info["status"] = "missing"
        return info

    info["size_bytes"] = path.stat().st_size
    try:
        if path.suffix.lower() == ".tar":
            with tarfile.open(path, "r:") as archive:
                members = archive.getmembers()
                info["archive_regular_files"] = sum(member.isfile() for member in members)
        elif path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as handle:
                if not handle.read(1):
                    raise ValueError("empty gzip content")
        else:
            with path.open("rb") as handle:
                if not handle.read(1):
                    raise ValueError("empty file")
        info["status"] = "header_readable"
    except (OSError, EOFError, tarfile.TarError, ValueError) as exc:
        info["status"] = "read_error"
        info["error"] = str(exc)
    return info


def summarize(root: Path, accession: str, study: dict,
              review: list[dict[str, str]], mapping: list[dict[str, str]]) -> dict:
    rows = [r for r in review if r["accession"] == accession]
    matches = [r for r in mapping if r["accession"] == accession]
    gsm = [r["gsm"] for r in rows if r["gsm"]]
    if len(gsm) != len(set(gsm)):
        raise ValueError(f"Duplicate GSM in {accession} review")

    role_rows = [r for r in rows if r["analysis_role"] == study["candidate_role"]]
    chosen = [] if accession == "GSE282074" else role_rows
    group_counts = Counter(r["candidate_group"] or "(not assigned)" for r in chosen)
    review_states = Counter(r["review_status"] or "(missing)" for r in chosen)
    flags = Counter(
        x.strip()
        for r in chosen
        for x in r["review_flags"].split("|")
        if x.strip()
    )

    mappings = Counter(r["mapping_status"] or "(missing)" for r in matches)
    proposed_gsm = {
        r["gsm"] for r in matches
        if r["gsm"] and r["mapping_status"] in {
            "exact_gsm", "exact_title", "alias_candidate"
        }
    }
    chosen_gsm = {r["gsm"] for r in chosen}
    file_checks = [
        file_check(root / "data/raw" / accession / filename)
        for filename in study["files"]
    ]

    issues = []
    if len(rows) != study["expected_gsm"]:
        issues.append("GEO sample count differs from the recorded study")
    if not chosen:
        issues.append("No samples approved as a modeling cohort by the prior review rule")
    if chosen_gsm - proposed_gsm:
        issues.append("Some candidate GSM records have no expression mapping candidate")
    if mappings.get("alias_candidate", 0):
        issues.append("Alias-based expression matches are not confirmed")
    if mappings.get("unresolved", 0) or mappings.get("ambiguous", 0):
        issues.append("Unresolved or ambiguous expression-column mappings remain")
    if any(f["status"] != "header_readable" for f in file_checks):
        issues.append("At least one local expression file is missing or unreadable")
    if len(review_states) != 1 or next(iter(review_states), "") != "VERIFIED":
        issues.append("Sample labels or independent biological replicates are not verified")
    if accession in {"GSE150949", "GSE282074"}:
        issues.append("Single-cell matrix requires replicate-aware, cell-level analysis")
    if accession == "GSE335846":
        issues.append("Cell-cycle phase groups are unequal; no balanced 2x3 design")

    return {
        "accession": accession,
        "geo_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
        "provisional_priority": study["order"],
        "potential_research_use": study["purpose"],
        "assay": study["assay"],
        "geo_gsm_records": len(rows),
        "review_role": study["candidate_role"],
        "records_in_review_role": len(role_rows),
        "candidate_gsm_records": len(chosen),
        "candidate_group_counts": dict(sorted(group_counts.items())),
        "candidate_review_status_counts": dict(sorted(review_states.items())),
        "candidate_review_flags": dict(sorted(flags.items())),
        "mapping_status_counts": dict(sorted(mappings.items())),
        "candidate_gsm_with_preliminary_mapping": len(chosen_gsm & proposed_gsm),
        "local_file_checks": file_checks,
        "blocking_review_issues": issues,
        "scientific_cautions": study["cautions"],
        "automatically_approved_for_training": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    review = read_csv(
        root / "data/metadata/review/sample_review_queue.csv",
        {"accession", "gsm", "analysis_role", "candidate_group",
         "review_flags", "review_status"},
    )
    mapping = read_csv(
        root / "data/metadata/mapping/sample_matrix_mapping.csv",
        {"accession", "gsm", "mapping_status", "expression_file"},
    )

    report = {
        "analysis": "stage_15_remaining_geo_screening",
        "objective": "Rank future experiments; no data relabeling or model fitting",
        "input_reports": [
            "data/metadata/review/sample_review_queue.csv",
            "data/metadata/mapping/sample_matrix_mapping.csv",
        ],
        "scientific_priority_is_provisional": True,
        "datasets": [
            summarize(root, accession, study, review, mapping)
            for accession, study in sorted(
                STUDIES.items(), key=lambda item: item[1]["order"]
            )
        ],
    }

    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for d in report["datasets"]:
        print(
            f"{d['accession']}: {d['candidate_gsm_records']}/"
            f"{d['geo_gsm_records']} candidate GSM; "
            f"mapping={d['mapping_status_counts']}"
        )
        for f in d["local_file_checks"]:
            print(f"  {f['file']}: {f['status']}")
        print(f"  Scientific review issues: {len(d['blocking_review_issues'])}")
    print(f"Saved: {destination}")
    print("No training eligibility was inferred from automated metadata checks.")


if __name__ == "__main__":
    main()
