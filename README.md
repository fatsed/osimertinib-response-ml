# Osimertinib Response ML Research

Reproducible computational research on **EGFR-mutant NSCLC** response to **osimertinib**, using public transcriptomic datasets and machine learning.

> **Research status:** feasibility and dataset audit. The final research question, base paper, model, and results have **not** been selected or validated yet.

## Project goals

1. Verify public datasets, sample labels, treatment durations, expression scale, and biological replicates.
2. Define a narrow, testable research question and select a reproducible computational baseline.
3. Build leakage-aware baseline experiments (e.g., leave-one-cell-line-out and held-out-study tests where valid).
4. Explore robustness and possible confounding, including cell-cycle-associated gene expression **only when supported by the data**.
5. Report honest evaluation metrics and limitations before drafting the paper.

## Repository layout

```text
configs/                 Dataset inventory and experiment configs
data/metadata/            Curated sample-level metadata (small, reviewable)
 data/raw/               Downloaded GEO files (not tracked by Git)
 data/processed/         Generated matrices (not tracked by Git)
docs/                    Research decisions, literature tracking, data dictionary
src/osimertinib_ml/      Reusable data, feature, model, and evaluation code
scripts/                 Executable data checks and experiments
experiments/             Experiment logs and settings
results/                 Metrics and summaries (no results yet)
figures/                 Figures generated from verified experiments
tests/                   Basic automated checks
```

## Quick start (Windows / VS Code)

In VS Code: **Terminal > New Terminal**. From the repository root:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python scripts/check_environment.py
python -m pytest
```

If PowerShell blocks activation, use `cmd` instead, or call the venv Python directly: `.venv\Scripts\python.exe scripts/check_environment.py`.

## Dataset candidates

Public GEO accessions currently under review: GSE193258, GSE255958, GSE254485, GSE202859, GSE335846, GSE282074, and GSE150949. See `configs/dataset_candidates.csv`.

**Important:** a GEO accession alone does not establish that samples have compatible biological labels or expression measurements. Every used sample must be verified against its source before analysis.

## Research integrity

- Keep raw downloads and large processed matrices **out of Git**; record accession, exact file/URL, date and checksum if used.
- Never invent model metrics. Store code/configuration/seed/splits with all results.
- Split by the appropriate independent unit (study, cell line, or biological sample), not individual single cells when cells share a biological replicate.
- Fit preprocessing, feature selection, calibration, and tuning **within training folds only**.
- Use held-out sets honestly; datasets inspected during method development are not untouched external validation.
- Research papers and manuscript drafts are kept in the parent `Osimertinib_Research` folder and are not uploaded to this code repository by default.

No model results have been produced in this repository yet.
