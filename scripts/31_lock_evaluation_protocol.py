"""Stage 31: deterministic, *plan-only* nested-LOCO split and evaluation protocol.

This program DOES NOT fit models, score outcomes, inspect target expression,
or convert historically explored source/target data into independent validation.
It fails closed when stage-09 input hashes or stage-30 status change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_META = Path("data/processed/cross_study/samples.csv")
INPUT_SOURCE = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
STAGE09 = Path("results/baseline_gse193258_loco.json")
STAGE30 = Path("results/dcsl_stage30_geo_provenance.json")
OUTPUT_JSON = Path("results/dcsl_stage31_evaluation_protocol.json")
OUTPUT_MD = Path("docs/stage31_evaluation_protocol.md")
STATES = ("Control", "Acute", "DTP")
CELL_LINES = ("H1975", "HCC2935", "HCC827", "PC9")
REVIEW_STATUS = "PENDING_MANUAL_AND_MATRIX_CHECK"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_inputs(root: Path) -> tuple[list[dict], dict]:
    baseline = json.loads((root / STAGE09).read_text(encoding="utf-8"))
    stage30 = json.loads((root / STAGE30).read_text(encoding="utf-8"))
    expected = baseline.get("input_sha256", {})
    for p in (INPUT_META, INPUT_SOURCE):
        found = digest(root / p)
        if found != expected.get(p.name):
            raise ValueError(f"Stage-09 input checksum mismatch: {p.name}")
    if baseline.get("source") != "GSE193258" or baseline.get("target") != "GSE255958":
        raise ValueError("Stage-09 study IDs changed")
    if (stage30.get("analysis") != "stage30_offline_geo_provenance_documentation_review"
            or stage30.get("input_samples_sha256") != expected.get("samples.csv")
            or stage30.get("n_checked") != 48
            or stage30.get("n_geo_title_label_matches") != 48
            or stage30.get("cross_study_status", {}).get("independent_external_validation_established") is not False
            or stage30.get("cross_study_status", {}).get("target_previously_examined_not_pristine_test") is not True
            or stage30.get("cross_study_status", {}).get("all_labels_keep_pending_review_status") is not True):
        raise ValueError("Stage-30 provenance or non-independent status changed")
    with (root / INPUT_META).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    validate_cohort(rows)
    return rows, {"samples.csv": expected["samples.csv"],
                  INPUT_SOURCE.name: expected[INPUT_SOURCE.name],
                  "stage30_report_sha256": digest(root / STAGE30)}


def validate_cohort(rows: list[dict]) -> None:
    required = {"accession", "gsm", "cell_line", "state", "evaluation_role", "label_review"}
    if len(rows) != 48 or not rows or not required.issubset(rows[0]):
        raise ValueError("Need exactly 48 samples and required metadata fields")
    if len({r["gsm"] for r in rows}) != 48 or any(not r["gsm"] for r in rows):
        raise ValueError("GSM IDs must be nonempty and unique")
    if any(r["label_review"] != REVIEW_STATUS for r in rows):
        raise ValueError("Unexpected label approval/review status")
    src = [r for r in rows if r["accession"] == "GSE193258"]
    tgt = [r for r in rows if r["accession"] == "GSE255958"]
    if len(src) != 36 or len(tgt) != 12 or any(r["evaluation_role"] != "source_candidate" for r in src):
        raise ValueError("Source and target roles/counts incorrect")
    count = Counter((r["cell_line"], r["state"]) for r in src)
    if count != Counter({(line, state): 3 for line in CELL_LINES for state in STATES}):
        raise ValueError("Source cell-line/state triplicates incomplete")
    if any(r["cell_line"] != "PC9" for r in tgt):
        raise ValueError("Unexpected target cell line")
    target_count = Counter((r["state"], r["evaluation_role"]) for r in tgt)
    expected = Counter({(s, "target_candidate"): 3 for s in STATES})
    expected[("Resistant", "exploratory_only")] = 3
    if target_count != expected:
        raise ValueError("Target labels/roles incorrect; resistant must stay exploratory")


def make_folds(rows: list[dict]) -> list[dict]:
    """Outer 3-lines->1 line; inner 2-lines->1 line, with GSM-level proof."""
    source = [r for r in rows if r["accession"] == "GSE193258"]
    by_line = {line: sorted((r["gsm"] for r in source if r["cell_line"] == line))
               for line in CELL_LINES}
    all_source = set(r["gsm"] for r in source)
    folds = []
    for held_out in CELL_LINES:
        outer_test = set(by_line[held_out])
        training_lines = [x for x in CELL_LINES if x != held_out]
        outer_train = set().union(*(set(by_line[c]) for c in training_lines))
        if len(outer_train) != 27 or len(outer_test) != 9 or outer_train & outer_test or outer_train | outer_test != all_source:
            raise AssertionError("Outer split overlap/incomplete")
        inner = []
        for inner_hold in training_lines:
            inner_val = set(by_line[inner_hold])
            inner_train = outer_train - inner_val
            if (len(inner_train), len(inner_val)) != (18, 9):
                raise AssertionError("Unexpected inner fold size")
            if inner_val & inner_train or (inner_train | inner_val) != outer_train or (inner_train | inner_val) & outer_test:
                raise AssertionError("Leakage: outer test entered inner split")
            inner.append({
                "validation_cell_line": inner_hold,
                "training_cell_lines": [c for c in training_lines if c != inner_hold],
                "train_gsm": sorted(inner_train), "validation_gsm": sorted(inner_val),
                "n_train": 18, "n_validation": 9,
            })
        if {x["validation_cell_line"] for x in inner} != set(training_lines):
            raise AssertionError("Inner folds do not cover all training groups")
        folds.append({"outer_test_cell_line": held_out,
                      "outer_training_cell_lines": training_lines,
                      "outer_train_gsm": sorted(outer_train),
                      "outer_test_gsm": sorted(outer_test),
                      "n_outer_train": 27, "n_outer_test": 9,
                      "inner_folds": inner})
    return folds


def candidate_specs() -> list[dict]:
    """Fixed illustrative candidate configurations; none is evaluated in this stage."""
    return [
        {"id": "dummy_prior", "family": "Dummy", "feature_selection": "none", "parameters": {"strategy": "prior"}},
        *[{"id": f"logistic_anova_500_C{c}", "family": "LogisticRegression",
           "feature_selection": "ANOVA", "parameters": {"k": 500, "C": c, "solver": "lbfgs", "max_iter": 3000}}
          for c in (0.01, 0.1, 1.0)],
        *[{"id": f"svm_anova_500_C{c}", "family": "LinearSVM",
           "feature_selection": "ANOVA", "parameters": {"k": 500, "C": c, "max_iter": 20000}}
          for c in (0.001, 0.01, 0.1)],
        *[{"id": f"dcsl_stable_500_gamma{g}", "family": "DCSL",
           "feature_selection": "Stable", "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": g}}
          for g in (0.0, 0.05)],
        {"id": "group_robust_stable_fixed", "family": "GroupRobust",
         "feature_selection": "Stable", "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": 0.0, "eta": 0.2, "temperature": 0.5}},
        {"id": "vrex_stable_fixed", "family": "VREx",
         "feature_selection": "Stable", "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": 0.0, "beta": 0.2}},
    ]


def build_report(rows: list[dict], hashes: dict) -> dict:
    return {
        "analysis": "stage31_prospective_style_protocol_for_historically_explored_data",
        "protocol_type": "versioned planning document; not a preregistered or independent test",
        "source_study": "GSE193258", "n_source": 36,
        "previously_inspected_target_study": "GSE255958", "n_target": 12,
        "data_sha256": hashes,
        "source_label_status": REVIEW_STATUS,
        "code_does_not_fit_or_select_models": True,
        "code_does_not_read_target_expression": True,
        "historic_results_were_seen_before_this_plan": True,
        "nested_loco_protocol": {
            "outer": "hold out one entire cell line; 4 outer folds x 9 samples",
            "inner": "leave one of three remaining cell lines out; 3 inner folds per outer fold",
            "selection": {
                "primary": "mean inner-fold balanced_accuracy (equally weighted by cell line)",
                "tie_break_1": "mean inner-fold DTP recall",
                "tie_break_2": "mean inner-fold macro-F1",
                "tie_break_3": "lexicographically smallest candidate ID",
                "scope": "select hyperparameters solely on outer training groups",
            },
            "outer_reporting": ["balanced_accuracy", "macro_f1", "DTP_recall", "per_class_recall",
                                "confusion_matrix_Control_Acute_DTP", "per_cell_line_predictions", "n_by_class"],
            "aggregation": "mean across four outer cell lines; also show per-cell-line metrics and pooled confusion matrix",
            "fixed_class_order": list(STATES),
            "seed": 42,
            "inference_threshold": "argmax/method-native decision; no post-hoc threshold tuning",
            "preprocessing": "Fit variance filtering, feature selection, standardization, imputation (if ever needed), and any batch adjustment strictly within the innermost training split; refit on outer-training split after inner selection. Never use outer test in fitting or selection.",
            "folds": make_folds(rows),
        },
        "candidate_configurations_for_future_implementation": candidate_specs(),
        "evaluation_governance": {
            "existing_four_source_cell_lines": "exploratory development only; previously inspected, including PC9",
            "GSE255958": "previously inspected external-looking cohort; exploratory sensitivity only; not an untouched independent test",
            "GSE255958_resistant": "exclude from three-state evaluation; exploratory only",
            "future_untouched_study": "needed for genuine independent validation; assess treatment/protocol compatibility and sample labels before locking evaluation",
            "future_study_gate": [
                "Confirm provenance, protocol, label mapping and biological replicate independence via primary evidence",
                "Freeze code commit, features, parameters, transforms, metrics, decision rule, eligible samples and exclusions before opening future outcomes",
                "Ensure future cohort did not influence the algorithm, preprocessing, candidate selection, or exploratory decisions",
                "Use the future cohort once as a final locked evaluation; report uncertainty and all failures",
            ],
            "interpretation": "No independence, causality, novelty, clinical accuracy or generalized performance is established by creating this split plan.",
        },
        "implementation_status": "SPLITS_AND_POLICY_ONLY_NOT_NESTED_MODEL_RESULTS",
    }


def make_markdown(report: dict) -> str:
    fold_lines = [
        f"| {f['outer_test_cell_line']} | {', '.join(f['outer_training_cell_lines'])} | "
        f"{f['n_outer_train']} | {f['n_outer_test']} | 3 x 18/9 |"
        for f in report["nested_loco_protocol"]["folds"]
    ]
    return "\n".join([
        "# Stage 31 — Evaluation protocol and independence boundaries",
        "",
        "**Status:** plan only; **no model fitted**. This is **not** prospective preregistration: all four source cell lines and the GSE255958 target have already been explored.",
        "",
        "## Inputs and roles",
        "- GSE193258 (36 samples; four cell lines x three states x three samples) is a historically examined **development** dataset.",
        "- GSE255958 (nine common-state PC9 samples, three resistant) has been previously inspected and is **not an untouched independent test**. Resistant is exploratory-only.",
        "- All 48 sample label reviews remain pending; stage 30 documented GEO titles, not independent biological certification.",
        "- The existing 18,013-gene common matrix was defined with both studies historically; for future genuinely independent testing, fix a feature-universe policy **before** inspecting new target data.",
        "",
        "## Planned nested leave-one-cell-line-out (LOCO)",
        "| Outer held-out line | Outer training lines | Train | Test | Inner train/validation |",
        "|---|---|---:|---:|---|",
        *fold_lines,
        "",
        "Each inner validation line is selected from the three outer-training lines, leaving two lines (18 samples) for inner training and one (9 samples) for validation. The outer test line is excluded from every inner step.",
        "",
        "## Frozen evaluation rules for future implementations",
        "- Primary inner selection: mean per-line balanced accuracy. Ties: mean DTP recall, then mean macro-F1, then candidate ID (lexicographic).",
        "- Exact candidate configurations are recorded in the JSON. Selecting any candidate after seeing outer-test results invalidates the intended separation.",
        "- For every inner split, fit **all** learned preprocessing (feature filters, selection, scaling, batch corrections if any) and the classifier using inner training only. Refit selected configuration on all outer-training samples, evaluate outer test once.",
        "- Report DTP recall, balanced accuracy, macro-F1, per-class recalls, confusion matrices, and predictions by held-out cell line; don't report only a pooled accuracy.",
        "- Aggregate across the four cell lines with equal weight, reporting the per-line results. Four groups and only three DTP samples per group imply substantial uncertainty.",
        "- Fix class order Control, Acute, DTP; seed 42; no tuned decision threshold after seeing held-out data.",
        "",
        "## Limits and independent validation gate",
        "1. Nested source-only LOCO is useful for an **exploratory** comparison, but it cannot retroactively make already inspected PC9 outcomes unseen.",
        "2. Do not use GSE255958 to choose methods, thresholds, feature selection, harmonization, or claims of independent transfer performance.",
        "3. A future independently acquired, eligible cohort must be kept untouched and have documented biological labels and compatible protocols. Freeze a code commit and analysis specification **before** opening its outcome labels.",
        "4. Do not equate common gene symbols, a shared log2(TPM+0.01) formula, or GEO title matches with full biological and processing compatibility.",
        "5. This stage makes **no model performance, novelty or clinical validity claim**.",
        "",
        "## Output integrity",
        "- JSON includes exact GSM split membership, hashes of original inputs, and stage-30 evidence report hash.",
        "- Only metadata, source-file checksum and existing reports are used. The target expression matrix is not read or required.",
        "",
    ])


def run(root: Path) -> dict:
    rows, hashes = load_inputs(root)
    report = build_report(rows, hashes)
    out = root / OUTPUT_JSON
    doc = root / OUTPUT_MD
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    doc.write_text(make_markdown(report), encoding="utf-8")
    print(f"Stage 31: 4 outer LOCO folds, 12 inner folds; 36 source samples")
    print("Outer train/test: 27/9; inner train/validation: 18/9")
    print("Input and stage-30 provenance hashes checked; GSE255958 expression NOT read")
    print("Plan only: no training, parameter tuning or independent validation")
    print(f"Saved: {out}")
    print(f"Saved: {doc}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
