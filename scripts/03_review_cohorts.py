from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "metadata" / "audit" / "geo_samples_unreviewed.csv"
OUTPUT = ROOT / "data" / "metadata" / "review"

CORE_PC9 = {f"GSM{i}" for i in range(8083374, 8083386)}
CORE_RESISTANCE = {f"GSM{i}" for i in range(6134696, 6134714)}

COLUMNS = [
    "accession", "gsm", "title", "cell_line", "genotype", "treatment",
    "time_point", "cell_cycle_marker", "candidate_cohort", "candidate_group",
    "analysis_role", "review_flags", "review_status", "characteristics_json",
]


def read_characteristics(raw: str) -> dict[str, list[str]]:
    values = json.loads(raw)
    if not isinstance(values, list):
        raise ValueError("Invalid GEO characteristics: expected a JSON list")
    data: dict[str, list[str]] = {}
    for entry in values:
        if not isinstance(entry, str) or ":" not in entry:
            continue
        name, value = entry.split(":", 1)
        name = re.sub(r"\s+", " ", name.strip().lower().replace("_", " ").replace("-", " "))
        value = value.strip()
        if value and value not in data.setdefault(name, []):
            data[name].append(value)
    return data


def field(data: dict[str, list[str]], name: str) -> str:
    return " | ".join(data.get(name, []))


def review_sample(row: dict[str, str]) -> dict[str, str]:
    accession = row["accession"]
    gsm = row["gsm"]
    attrs = read_characteristics(row["characteristics_json"])
    cell_line = field(attrs, "cell line")
    genotype = field(attrs, "genotype")
    treatment = field(attrs, "treatment")
    time_point = field(attrs, "time point") or field(attrs, "time")
    cycle = field(attrs, "mcherry expression") or field(attrs, "cherry expression")

    role = "other_samples"
    cohort = ""
    group = ""
    flags: list[str] = []
    cell_lower = cell_line.lower()
    genotype_lower = genotype.lower()
    treatment_lower = treatment.lower()
    time_lower = time_point.lower()

    if accession == "GSE193258":
        cohort = "GSE193258_3state"
        role = "three_state_candidate"
        if treatment_lower == "dmso":
            group = "Control"
        elif treatment_lower == "osimertinib acute 24 hrs.":
            group = "Acute"
        elif treatment_lower == "osimertinib dtps 3w.":
            group = "DTP"
        elif "washout" in treatment_lower:
            role = "washout_exploratory"
            group = "Washout"
        else:
            role = "needs_review"
            flags.append("unexpected_treatment")

    elif accession == "GSE255958":
        if gsm in CORE_PC9 and cell_lower == "pc9":
            cohort = "GSE255958_PC9_4state"
            role = "four_state_candidate"
            known = {
                ("parental", "untreated 0.1% dmso"): "Control",
                ("acute", "osimertinib 2 um 48h"): "Acute",
                ("drug-tolerant persister", "osimertinib 2 um 9d"): "DTP",
                ("resistant", "osimertinib 2 um 48h"): "Resistant",
            }
            group = known.get((genotype_lower, treatment_lower), "")
            if not group:
                role = "needs_review"
                flags.append("unexpected_core_conditions")
        elif cell_lower == "pc9" and genotype_lower == "drug-tolerant persister" and "osimertinib" in treatment_lower:
            role = "additional_pc9_batch"
            flags.append("different_experiment_from_core")
        if "th107 osim 9d" in row["title"].lower() and "11d" in treatment_lower:
            flags.append("title_9d_treatment_11d_conflict")

    elif accession == "GSE202859":
        if gsm in CORE_RESISTANCE and cell_lower in {"pc9", "h1975", "hcc827"}:
            cohort = "GSE202859_resistance"
            role = "resistance_candidate"
            if genotype_lower == "parental" and treatment_lower == "untreated":
                group = "Parental"
            elif genotype_lower == "osimertinib resistant" and "osimertinib" in treatment_lower:
                group = "Resistant"
            else:
                role = "needs_review"
                flags.append("unexpected_core_conditions")

    elif accession == "GSE254485":
        if cell_lower == "pc9" and time_lower in {"0d", "28d", "35d", "56d"}:
            cohort = "GSE254485_PC9_timecourse"
            group = f"Day_{time_lower.removesuffix('d')}"
            role = "timecourse_candidate"
        else:
            role = "other_resistance_samples"

    elif accession == "GSE335846":
        if cell_lower == "pc9" and re.fullmatch(r"day (0|28) (g1|s|g2)", time_lower):
            cohort = "GSE335846_cell_cycle"
            group = time_point.replace(" ", "_")
            role = "cell_cycle_candidate"
            flags.append("unequal_phase_sample_counts")

    elif accession == "GSE150949":
        if cell_lower == "pc9 lung cancer cell line":
            cohort = "GSE150949_PC9_single_cell"
            group = time_point
            role = "single_cell_candidate"
            flags.append("cells_not_independent_replicates")
        else:
            role = "mixed_cell_lines_or_other"

    elif accession == "GSE282074":
        role = "technology_benchmark_only"

    if not cell_line or cell_lower == "n/a":
        flags.append("cell_line_unavailable")
    if not treatment:
        flags.append("treatment_unavailable")
    if field(attrs, "cell lines") and not cell_line:
        flags.append("multiple_cell_lines")

    return {
        "accession": accession,
        "gsm": gsm,
        "title": row["title"],
        "cell_line": cell_line,
        "genotype": genotype,
        "treatment": treatment,
        "time_point": time_point,
        "cell_cycle_marker": cycle,
        "candidate_cohort": cohort,
        "candidate_group": group,
        "analysis_role": role,
        "review_flags": " | ".join(flags),
        "review_status": "PENDING_MANUAL_AND_MATRIX_CHECK",
        "characteristics_json": row["characteristics_json"],
    }


def write_csv(path: Path, data: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GEO sample cohorts for manual review")
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Audit CSV not found: {args.input}")

    with args.input.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "gsm", "title", "characteristics_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            parser.error(f"Missing input columns: {', '.join(sorted(missing))}")
        records = [review_sample(row) for row in reader]

    if not records:
        parser.error("No GEO samples were found")
    ids = [(row["accession"], row["gsm"]) for row in records]
    if len(ids) != len(set(ids)):
        parser.error("Duplicate GEO accession/GSM combinations found")

    summary = Counter(
        (r["accession"], r["candidate_cohort"], r["analysis_role"], r["candidate_group"])
        for r in records
    )
    counts = [
        {
            "accession": accession,
            "candidate_cohort": cohort,
            "analysis_role": role,
            "candidate_group": group,
            "gsm_records": count,
        }
        for (accession, cohort, role, group), count in sorted(summary.items())
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "sample_review_queue.csv", records, COLUMNS)
    write_csv(args.output / "candidate_counts.csv", counts, list(counts[0]))

    print(f"GEO records reviewed: {len(records)}")
    print(f"Rows requiring additional attention: {sum(bool(r['review_flags']) for r in records)}")
    print(f"Saved: {args.output / 'sample_review_queue.csv'}")
    print(f"Saved: {args.output / 'candidate_counts.csv'}")
    print("All candidate cohorts remain unverified until source and expression-column checks.")


if __name__ == "__main__":
    main()
