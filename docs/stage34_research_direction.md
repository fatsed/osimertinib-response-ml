# Stage 34 — Research direction and source-only feature-universe repair

**Decision date:** 2026-11-13  
**Status:** method-development / exploratory. **No independent validation or proven methodological novelty.**

## Research objective

**Working question:** Can an ML method distinguish *drug-tolerant persister (DTP)* transcriptomic profiles from the **acute osimertinib-treated** state, while limiting dependence on cell-line identity and study/protocol confounding?

This is **post-treatment state classification**, NOT prediction of a patient's future drug response, NOT proof of a DTP-specific causal mechanism, and NOT prediction of acquired resistance. Duration of drug exposure is inherently entangled with DTP-vs-Acute in the currently used primary study; computational adjustments alone cannot identify the biological effect of DTP independently of exposure duration.

## Decision on model

- **Retain Logistic Regression (ANOVA + 500 genes) as the principal *development reference***. In Stage 32, exploratory nested LOCO mean balanced accuracy was 0.9167 versus 0.8611 for DCSL; PC9 DTP recall was 1/3 versus 0/3. These results do not establish a method's population-level superiority (four cell lines and three DTP samples per line).
- **Retain DCSL as an exploratory research prototype and negative result**, rather than asserting superiority. Its variance penalty does not necessarily enforce a positive DTP-vs-Acute margin in an unseen cell line. In two of four Stage-32 folds, the selected DCSL gamma was 0.
- **Do not tune any method based on historically inspected PC9 or GSE255958 outcomes**. A distinct method contribution, if pursued, needs a separate mathematical formulation, training-only selection rules, ablation, and an untouched eligible cohort or appropriately identified confirmatory experiment.
- Do not present new methods for biological validation until there is credible dose/time matched data or clear evidence explaining what labels can establish.

## Immediate code correction: historical target-dependent gene universe

Stage 08 formed **18,013** source features by intersecting source GSE193258 genes with GSE255958 genes. There is no evidence that target *labels* entered training, but the target-defined feature universe compromises a strict target-independent feature-selection claim.

Stage 05 already created the **source-only 19,712-gene** prepared matrix. Stage 34 replays the **same Stage-31 candidate grid and exact nested LOCO protocol** on that matrix. Existing Stage-31/32/33 reports are preserved; comparison is a sensitivity analysis, not independent validation or a new model selection round.

Required invariants of `scripts/34_source_only_rebaseline.py`:

1. Verify unchanged Stage-31 protocol, Stage-32 identity and Stage-05 source provenance hashes. No access to GSE255958 expression or outcome files.
2. Confirm exactly 36 source GSMs, four cell lines, three candidate states and three replicates per state. Retain **PENDING_MANUAL_AND_MATRIX_CHECK** label status.
3. Confirm Stage-08 18,013-gene source file is an exact subset of the original Stage-05 source expression values and keep its reference role separate.
4. Run unchanged candidate settings and unchanged *training-only* nested LOCO; generate per-cell-line metrics and compare descriptively with Stage 32.
5. Do not choose future methods, feature counts or thresholds from Stage-34 test-fold performance.

## Evidence and dataset eligibility

| Data | Source-supported design | Eligibility decision for *new untouched* 3-state validation |
|---|---|---|
| GSE193258 | Four EGFR-mutant lines; Control, 24h Acute, 21d DTP, plus washouts. 60 total RNA-seq samples, 36 in main 3-state development set. | **Development only**; historically examined. Acute vs DTP confounded by duration. |
| GSE255958 | PC9 Control, Acute 48h 2uM, DTP day 9 2uM, resistant derivatives; divergent pipeline/reference from source. | **Not untouched**; already examined in stages 09/14/22/29. Exploratory only. |
| GSE153183 | PC9 parental vs 9d DTP, two biological replicates per class. | No Acute class; cannot confirm the full 3-state question. Prior published use also limits novelty claims. |
| GSE275199 | scRNA-seq, PC9 osimertinib 4d/11d and untreated controls; mixed H358 sotorasib conditions, pooled hashing experiment. | Not a verified independent 3-state bulk cohort; no acute-treatment group shown in GEO design; pseudoreplication risk. |
| GSE341996 | Recently indexed DTP RNA-seq; TPM file listed by OmicsDI. | **Unverified candidate**: exact GEO sample count, cell lines, replicates, Acute/control groups, and protocol not yet confirmed. No approval for fitting/testing. |
| GSE254485 / GSE335846 / GSE202859 / GSE150949 | Time-course, phase-sorted, acquired-resistant or scRNA-seq data previously screened. | Potential secondary biology or sensitivity studies; cannot silently relabel long exposure as DTP or resistant as acute/DTP; replicates and protocols need review. |

**Eligibility gate for a future independent 3-state test**: (i) independently defined control, acute and biologically supported DTP groups, (ii) replicate-level metadata and independent experimental units, (iii) known dose/time/cell line, (iv) source-only gene mapping with frozen feature policy, (v) new cohort kept uninspected until protocol freeze. If no dataset meets this gate, narrow the claim to a *descriptive, time-confounded development benchmark* rather than claiming validation.

## Literature alignment (selected high-priority sources)

- Criscione et al., 2022, source biology and 60-sample RNA-seq design: https://pmc.ncbi.nlm.nih.gov/articles/PMC9794691/
- PanDTP-12, 2026, recurring DTP signatures and GSE255958: https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2026.1882238/full
- Four-gene EGFR-TKI persister signature, 2026: https://pmc.ncbi.nlm.nih.gov/articles/PMC13613113/
- Tissue-aware confounder-controlled co-expression, 2026: https://pubmed.ncbi.nlm.nih.gov/42442052/
- DTP methods and biology systematic overview, 2026: https://www.nature.com/articles/s41698-026-01717-2
- GSE153183 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153183
- GSE275199 GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275199
- GSE341996 OmicsDI record: https://www.omicsdi.org/dataset/geo/GSE341996

## Decision gate after Stage 34 code run

If the source-only replay substantially changes the relative model behavior, report a **feature-universe sensitivity finding**, not an improvement; investigate the change without post-hoc retuning. If results are similar, record robustness to that particular gene-universe choice, still **not** proof of independence or clinical validity. Next pursue independent metadata verification of GSE341996 and a search for time/dose-comparable Acute and DTP cohorts. No new method is accepted merely because it wins on an already inspected fold.
