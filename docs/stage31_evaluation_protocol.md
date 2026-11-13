# Stage 31 — Evaluation protocol and independence boundaries

**Status:** plan only; **no model fitted**. This is **not** prospective preregistration: all four source cell lines and the GSE255958 target have already been explored.

## Inputs and roles
- GSE193258 (36 samples; four cell lines x three states x three samples) is a historically examined **development** dataset.
- GSE255958 (nine common-state PC9 samples, three resistant) has been previously inspected and is **not an untouched independent test**. Resistant is exploratory-only.
- All 48 sample label reviews remain pending; stage 30 documented GEO titles, not independent biological certification.
- The existing 18,013-gene common matrix was defined with both studies historically; for future genuinely independent testing, fix a feature-universe policy **before** inspecting new target data.

## Planned nested leave-one-cell-line-out (LOCO)
| Outer held-out line | Outer training lines | Train | Test | Inner train/validation |
|---|---|---:|---:|---|
| H1975 | HCC2935, HCC827, PC9 | 27 | 9 | 3 x 18/9 |
| HCC2935 | H1975, HCC827, PC9 | 27 | 9 | 3 x 18/9 |
| HCC827 | H1975, HCC2935, PC9 | 27 | 9 | 3 x 18/9 |
| PC9 | H1975, HCC2935, HCC827 | 27 | 9 | 3 x 18/9 |

Each inner validation line is selected from the three outer-training lines, leaving two lines (18 samples) for inner training and one (9 samples) for validation. The outer test line is excluded from every inner step.

## Frozen evaluation rules for future implementations
- Primary inner selection: mean per-line balanced accuracy. Ties: mean DTP recall, then mean macro-F1, then candidate ID (lexicographic).
- Exact candidate configurations are recorded in the JSON. Selecting any candidate after seeing outer-test results invalidates the intended separation.
- For every inner split, fit **all** learned preprocessing (feature filters, selection, scaling, batch corrections if any) and the classifier using inner training only. Refit selected configuration on all outer-training samples, evaluate outer test once.
- Report DTP recall, balanced accuracy, macro-F1, per-class recalls, confusion matrices, and predictions by held-out cell line; don't report only a pooled accuracy.
- Aggregate across the four cell lines with equal weight, reporting the per-line results. Four groups and only three DTP samples per group imply substantial uncertainty.
- Fix class order Control, Acute, DTP; seed 42; no tuned decision threshold after seeing held-out data.

## Limits and independent validation gate
1. Nested source-only LOCO is useful for an **exploratory** comparison, but it cannot retroactively make already inspected PC9 outcomes unseen.
2. Do not use GSE255958 to choose methods, thresholds, feature selection, harmonization, or claims of independent transfer performance.
3. A future independently acquired, eligible cohort must be kept untouched and have documented biological labels and compatible protocols. Freeze a code commit and analysis specification **before** opening its outcome labels.
4. Do not equate common gene symbols, a shared log2(TPM+0.01) formula, or GEO title matches with full biological and processing compatibility.
5. This stage makes **no model performance, novelty or clinical validity claim**.

## Output integrity
- JSON includes exact GSM split membership, hashes of original inputs, and stage-30 evidence report hash.
- Only metadata, source-file checksum and existing reports are used. The target expression matrix is not read or required.
