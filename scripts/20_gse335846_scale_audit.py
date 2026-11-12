from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = Path("results/gse335846_phase_preparation.json")
REPORT = Path("results/gse335846_scale_audit.json")
FIGURE = Path("figures/gse335846_expression_distributions.png")
SAMPLE_COLUMNS = {"gsm", "day", "phase", "clone", "treatment"}
IDENTIFIER_COLUMNS = {"feature_id", "probe_id", "source_gene_id"}
PERCENTILES = [0, 1, 5, 25, 50, 75, 95, 99, 100]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs(root: Path):
    previous = json.loads((root / PREPARATION).read_text(encoding="utf-8"))
    if previous.get("source") != "GSE335846":
        raise ValueError("Wrong stage-19 report")
    if previous.get("labels_verified") is not False:
        raise ValueError("Unexpected label verification status; review stage 19")

    matrix_path = root / previous["processed_expression"]
    sample_path = root / previous["candidate_metadata"]
    for path, key in [
        (matrix_path, "processed_expression_sha256"),
        (sample_path, "candidate_metadata_sha256"),
    ]:
        if not path.is_file() or sha256(path) != previous.get(key):
            raise ValueError(f"Missing or changed stage-19 input: {path}")

    samples = pd.read_csv(sample_path, dtype=str, keep_default_na=False)
    if not SAMPLE_COLUMNS.issubset(samples.columns):
        raise ValueError(f"Missing sample columns: {sorted(SAMPLE_COLUMNS - set(samples.columns))}")
    if len(samples) != 9 or samples["gsm"].duplicated().any():
        raise ValueError("Expected exactly nine unique GEO samples")
    samples["day"] = pd.to_numeric(samples["day"], errors="raise").astype(int)
    samples["clone"] = pd.to_numeric(samples["clone"], errors="raise").astype(int)
    if set(samples["day"]) != {0, 28} or set(samples["phase"]) != {"G1", "S", "G2"}:
        raise ValueError("Unexpected time or phase groups")
    groups = Counter(f"Day_{s.day}_{s.phase}" for s in samples.itertuples())
    if dict(groups) != previous.get("group_sizes"):
        raise ValueError("Sample groups changed since stage 19")
    if not samples.loc[samples["day"] == 0, "treatment"].str.lower().eq("untreated").all():
        raise ValueError("Day 0 treatment metadata changed")
    if not samples.loc[samples["day"] == 28, "treatment"].str.lower().str.contains("osimertinib", regex=False).all():
        raise ValueError("Day 28 treatment metadata changed")

    matrix = pd.read_csv(matrix_path, sep="\t", compression="gzip", dtype=str,
                         keep_default_na=False, low_memory=False)
    if matrix.columns.duplicated().any() or not IDENTIFIER_COLUMNS.issubset(matrix.columns):
        raise ValueError("Invalid feature identifiers in stage-19 matrix")
    if matrix["feature_id"].duplicated().any() or matrix["feature_id"].eq("").any():
        raise ValueError("Missing or repeated source-row feature ID")
    gsm = samples["gsm"].tolist()
    if not set(gsm).issubset(matrix.columns):
        raise ValueError("Expression matrix does not contain all nine sample columns")
    if len(matrix) != previous.get("features") or previous.get("samples") != 9:
        raise ValueError("Expression dimensions changed since stage 19")

    values = matrix[gsm].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Source data must be finite; no imputation is allowed")
    if int((values < 0).sum()) != previous["numeric_qc"]["negative_values_total"]:
        raise ValueError("Negative value count disagrees with stage 19")

    probe = matrix["probe_id"].str.strip()
    gene = matrix["source_gene_id"].str.strip()
    if (probe.ne("").sum() != previous["identifier_quality"]["nonempty_probe_ids"]
            or probe[probe.ne("")].nunique() != previous["identifier_quality"]["unique_nonempty_probe_ids"]
            or gene[gene.ne("")].nunique() != previous["identifier_quality"]["distinct_annotated_gene_ids"]):
        raise ValueError("Probe or gene annotation counts differ from stage 19")
    return previous, matrix, samples, values, matrix_path, sample_path


def distribution(values: np.ndarray, samples: pd.DataFrame) -> list[dict]:
    results = []
    for i, sample in samples.iterrows():
        col = values[:, i]
        qs = np.percentile(col, PERCENTILES)
        low = col[col < 0]
        unique, counts = np.unique(col, return_counts=True)
        common_index = int(np.argmax(counts))
        results.append({
            "gsm": sample["gsm"],
            "day": int(sample["day"]),
            "phase": sample["phase"],
            "clone": int(sample["clone"]),
            "treatment": sample["treatment"],
            "nonfinite_values": 0,
            "negative_values": int(low.size),
            "negative_fraction": float(low.size / col.size),
            "zero_values": int(np.count_nonzero(col == 0)),
            "distinct_numeric_values": int(unique.size),
            "most_frequent_value": float(unique[common_index]),
            "most_frequent_value_count": int(counts[common_index]),
            "quantiles": {str(p): float(q) for p, q in zip(PERCENTILES, qs)},
            "mean": float(col.mean()),
            "sd": float(col.std(ddof=1)),
        })
    return results


def duplicates(matrix: pd.DataFrame, values: np.ndarray) -> dict:
    probe = matrix["probe_id"].str.strip()
    gene = matrix["source_gene_id"].str.strip()
    dup_mask = probe.ne("") & probe.duplicated(keep=False)
    duplicated = matrix.loc[dup_mask, ["probe_id", "source_gene_id"]].copy()
    duplicated["source_gene_id"] = duplicated["source_gene_id"].str.strip()
    duplicated["probe_id"] = duplicated["probe_id"].str.strip()

    groups_with_conflicting_genes = 0
    groups_with_differing_values = 0
    duplicated_groups = 0
    for _, group in duplicated.groupby("probe_id", sort=False):
        duplicated_groups += 1
        nonempty_gene = group["source_gene_id"].loc[group["source_gene_id"].ne("")]
        if nonempty_gene.nunique() > 1:
            groups_with_conflicting_genes += 1
        subset = values[group.index.to_numpy()]
        if np.unique(subset, axis=0).shape[0] > 1:
            groups_with_differing_values += 1

    annotated = gene[gene.ne("")]
    gene_multiplicity = annotated.value_counts()
    result = {
        "total_source_probe_rows": int(len(matrix)),
        "unique_nonempty_probe_ids": int(probe[probe.ne("")].nunique()),
        "rows_with_duplicate_probe_id": int(dup_mask.sum()),
        "duplicate_probe_id_groups": duplicated_groups,
        "duplicate_probe_groups_with_conflicting_gene_annotations": groups_with_conflicting_genes,
        "duplicate_probe_groups_with_differing_numeric_profiles": groups_with_differing_values,
        "distinct_annotated_genes": int(annotated.nunique()),
        "genes_annotated_by_multiple_rows": int((gene_multiplicity > 1).sum()),
        "maximum_rows_per_gene": int(gene_multiplicity.max()),
        "rows_missing_gene_annotation": int(gene.eq("").sum()),
        "aggregated_probe_rows": False,
    }
    return result


def figure(distributions: list[dict], path: Path) -> None:
    labels = [f"D{d['day']}-{d['phase']}-C{d['clone']}" for d in distributions]
    boxes = []
    for d in distributions:
        q = d["quantiles"]
        boxes.append({
            "label": f"D{d['day']}-{d['phase']}-C{d['clone']}",
            "whislo": q["1"], "q1": q["25"], "med": q["50"],
            "q3": q["75"], "whishi": q["99"], "fliers": [],
        })
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bxp(boxes, showfliers=False, patch_artist=False)
    ax.axhline(0, linestyle="--", linewidth=0.9, alpha=0.55)
    ax.set_ylabel("Source-exported expression values (unverified scale)")
    ax.set_xlabel("Day – cell-cycle phase – clone")
    ax.set_title("GSE335846: untransformed sample distributions")
    ax.set_xticks(range(1, len(labels) + 1), labels, rotation=35, ha="right")
    ax.text(0.01, 0.02, "Whiskers: 1st–99th percentile; no log transform, clipping or normalization",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    previous, matrix, samples, values, matrix_path, sample_path = load_inputs(root)
    summaries = distribution(values, samples)
    identifier_audit = duplicates(matrix, values)
    medians = [d["quantiles"]["50"] for d in summaries]
    report = {
        "analysis": "GSE335846_stage20_scale_and_probe_audit",
        "source_study": "GSE335846",
        "n_samples": len(samples),
        "n_probe_rows": len(matrix),
        "stage19_input_sha256": {
            previous["processed_expression"]: sha256(matrix_path),
            previous["candidate_metadata"]: sha256(sample_path),
        },
        "source_file_sha256": previous["source_sha256"],
        "input_numeric_values_changed": False,
        "input_expression_scale": previous["expression_units"],
        "scale_classification": "processed_values; exact transformation and cross-sample comparability unverified",
        "sample_distributions": summaries,
        "between_sample_median_range": [float(min(medians)), float(max(medians))],
        "identifier_audit": identifier_audit,
        "scientific_label_review_complete": False,
        "approved_for_gene_level_analysis": False,
        "approved_for_external_validation": False,
        "training_or_statistical_testing_performed": False,
        "limitations": [
            "GEO reports size-factor normalisation, but exact exported numeric transformation is not established.",
            "Negative values are restricted to one source sample; their origin requires verification.",
            "Multiple probe rows annotate the same Ensembl gene; probe identity is not gene identity.",
            "Repeated probe IDs and possibly conflicting annotations must be reviewed before collapse.",
            "Nine samples are unevenly distributed across phase, day and clone; these factors are confounded.",
            "No extra normalization, log transform, clipping, probe collapsing or PCA was performed.",
            "Descriptive distribution differences do not establish batch effects or DTP biology.",
        ],
        "figure": FIGURE.as_posix(),
    }

    (root / FIGURE).parent.mkdir(parents=True, exist_ok=True)
    figure(summaries, root / FIGURE)
    (root / REPORT).parent.mkdir(parents=True, exist_ok=True)
    (root / REPORT).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Samples: {len(samples)} | probe rows: {len(matrix)} | annotated genes: {identifier_audit['distinct_annotated_genes']}")
    print(f"Duplicate probe groups: {identifier_audit['duplicate_probe_id_groups']} "
          f"(conflicting genes: {identifier_audit['duplicate_probe_groups_with_conflicting_gene_annotations']}, "
          f"differing values: {identifier_audit['duplicate_probe_groups_with_differing_numeric_profiles']})")
    for d in summaries:
        print(f"{d['gsm']} D{d['day']} {d['phase']} C{d['clone']}: "
              f"median={d['quantiles']['50']:.3f}, min={d['quantiles']['0']:.3f}, "
              f"negatives={d['negative_values']}, mode_n={d['most_frequent_value_count']}")
    print(f"Saved: {root / REPORT}")
    print(f"Saved: {root / FIGURE}")
    print("Exploratory audit only; no PCA, normalization, model fitting, or gene collapsing.")


if __name__ == "__main__":
    main()
