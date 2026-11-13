# Stage 30 — GEO provenance and cohort decision record

**Status:** GEO title/sample-label alignment documented (n=48/48); independent biological validation and cross-study comparability **remain unresolved**.

## Reproducibility
- `samples.csv` SHA-256: `ccc55e01786e70c078bc32b45733a4f5a5c5bef95216eb962da7cae9028e9e33`
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
| Genome reference | hg38 | hg19 |
| Expression | GEO explicitly reports `log2(TPM + 0.01)` | GEO provides RSEM TPM; project stage-08 applies `log2(TPM + 0.01)` |

## Scientific interpretation
1. **Resolved:** Exact transformation formula is *documented on the GSE193258 GEO sample record*. Stage 29's uncertainty about the published formula can be amended; there is no need to infer it from the floor alone.
2. **Not resolved:** Equal mathematical expression transforms do not remove different quantification pipelines, annotation versions, genome assemblies, exposure regimes or batch effects. Matching gene symbols alone does not guarantee identical feature definitions.
3. The 36 GSE193258 samples are suitable for *exploratory* within-study LOCO comparisons using the documented source experiment, subject to experimental-replicate and sample-to-matrix review.
4. The nine common-state PC9 samples from GSE255958 have been repeatedly inspected during development. They **cannot** be described as an untouched independent test set. The three resistant samples remain exploratory-only.
5. Source GSE193258 states biological triplicates in the series design; a `rep` title suffix in the target is not by itself independent proof of biological replication.
6. Do not promote `PENDING_MANUAL_AND_MATRIX_CHECK` to a certified status automatically. A domain expert should review primary methods, provenance and per-sample lab metadata.

## References (GEO primary evidence)
- [GSE193258 series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193258)
- [GSM5777996: processing pipeline and log2(TPM + 0.01)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5777996)
- [GSE255958 series](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE255958)
- [GSM8083374: STAR/RSEM, hg19](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8083374)
- [GSM8083380: selected PC9 DTP, 2 µM 9d](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM8083380)

## Next decision before claims of new generalization
Retain source-only LOCO as descriptive development results; make a prospective validation plan with independent, compatible data and separate pre-specified model development from final testing. Do **not** change existing labels, expression matrices, or old metrics based on this audit.
