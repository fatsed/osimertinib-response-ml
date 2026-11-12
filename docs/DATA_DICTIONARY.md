# Metadata schema (proposed; adjust after GEO audit)

One row per **biologically independent sample**, unless a separate cell-level table is explicitly defined.

| Field | Meaning |
| --- | --- |
| `geo_accession` | GEO series (GSE...) |
| `gsm_id` | GEO sample (GSM...), where available |
| `source_file` | Original downloaded data filename |
| `study_id` | Study/source identifier |
| `cell_line` | Cell line name exactly as reported |
| `condition_raw` | Original author-provided condition |
| `condition_harmonized` | Carefully reviewed analysis label |
| `drug` | Treatment name |
| `dose` | Original dose including units |
| `exposure_hours` | Drug exposure duration if known |
| `replicate_id` | Independent biological replicate identifier |
| `assay` | Bulk RNA-seq, scRNA-seq, etc. |
| `expression_unit` | Counts, TPM, log2(TPM+offset), etc. |
| `included` | Verified inclusion flag |
| `exclusion_reason` | Reason for any exclusion |

Keep provenance and ambiguity explicit; do not guess missing metadata.
