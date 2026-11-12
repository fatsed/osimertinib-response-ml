from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE_MATRIX = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
SAMPLE_TABLE = Path("data/processed/cross_study/samples.csv")
RAW_MATRIX = Path("data/raw/GSE193258/GSE193258_RNAseq_log2TPM_abundance.tsv.gz")
REVIEW = Path("data/metadata/review/sample_review_queue.csv")
MAPPING = Path("data/metadata/mapping/sample_matrix_mapping.csv")
BASELINE = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/gse193258_washout_challenge.json")
STATES = ("Control", "Acute", "DTP")
LONG_WASHOUT = {
    "H1975": ("72h", "72 hrs."),
    "HCC2935": ("10d", "10 days"),
    "HCC827": ("96h", "96 hrs."),
    "PC9": ("7d", "7 days"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training(root: Path):
    matrix_path = root / SOURCE_MATRIX
    metadata_path = root / SAMPLE_TABLE
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    expected = baseline["input_sha256"]
    for path in (matrix_path, metadata_path):
        if sha256(path) != expected[path.name]:
            raise ValueError(f"Input changed since stage 09: {path.name}")

    table = pd.read_csv(matrix_path, sep="\t", compression="gzip")
    if "gene_id" not in table:
        raise ValueError("No gene_id column in stage 08 source matrix")
    ids = table.pop("gene_id").astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Invalid or repeated source genes")
    samples = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    samples = samples.loc[samples["accession"] == "GSE193258"].copy()
    if len(samples) != 36 or samples["gsm"].nunique() != 36:
        raise ValueError("Expected 36 unique training GSMs")
    if set(table.columns) != set(samples["gsm"]):
        raise ValueError("Source sample names do not match stage 08 metadata")
    samples = samples.set_index("gsm").loc[list(table.columns)].rename_axis("gsm").reset_index()
    if samples.groupby(["cell_line", "state"]).size().to_dict() != {
        (line, state): 3 for line in LONG_WASHOUT for state in STATES
    }:
        raise ValueError("Unexpected source cell line/state design")
    values = table.to_numpy(dtype=float).T
    if not np.isfinite(values).all():
        raise ValueError("Source expression has invalid values")
    return ids.tolist(), values, samples, baseline


def load_washout_metadata(root: Path):
    metadata = pd.read_csv(root / REVIEW, dtype=str, keep_default_na=False)
    links = pd.read_csv(root / MAPPING, dtype=str, keep_default_na=False)
    wash = metadata.loc[
        (metadata["accession"] == "GSE193258")
        & (metadata["candidate_group"] == "Washout")
        & (metadata["analysis_role"] == "washout_exploratory")
    ].copy()
    links = links.loc[
        (links["accession"] == "GSE193258")
        & (links["expression_file"] == RAW_MATRIX.name)
        & (links["source_type"] == "matrix_column")
        & (links["mapping_status"] == "alias_candidate")
        & (links["gsm"].isin(wash["gsm"]))
    ].copy()
    if len(wash) != 24 or wash["gsm"].nunique() != 24:
        raise ValueError("Expected 24 distinct washout GSMs")
    if len(links) != 24 or links["gsm"].nunique() != 24 or links["column_or_member"].nunique() != 24:
        raise ValueError("Washout aliases are missing or duplicated")
    wash = wash.merge(
        links[["gsm", "column_or_member"]], on="gsm", validate="one_to_one"
    )
    stages = []
    for row in wash.itertuples(index=False):
        match = re.fullmatch(r"(H1975|HCC2935|HCC827|PC9)_(short|long)_wash_([1-3])", row.column_or_member)
        if match is None or row.cell_line != match.group(1):
            raise ValueError(f"Unexpected washout mapping for {row.gsm}")
        cell, kind, replicate = match.groups()
        title = row.title.removesuffix(" [RNA-seq]")
        duration = LONG_WASHOUT[cell][0] if kind == "long" else None
        expected_title = (f"{cell}_{kind}_washout_{duration}_{replicate}"
                          if duration else f"{cell}_short_washout_{replicate}")
        expected_treatment = (f"osimertinib DTPs long washout {LONG_WASHOUT[cell][1]}"
                              if kind == "long" else "osimertinib DTPs short washout 24 hrs.")
        if title != expected_title or row.treatment.strip().lower() != expected_treatment.lower():
            raise ValueError(f"Washout alias disagrees with GEO title/treatment: {row.gsm}")
        stages.append("ShortWashout" if kind == "short" else "LongWashout")
    wash["challenge_group"] = stages
    counts = wash.groupby(["cell_line", "challenge_group"]).size()
    if len(counts) != 8 or not counts.eq(3).all():
        raise ValueError("Expected three samples in each cell line/washout group")
    return wash.sort_values(["cell_line", "challenge_group", "gsm"]).reset_index(drop=True)


def load_washout_values(root: Path, genes: list[str], source: np.ndarray, samples: pd.DataFrame,
                        wash: pd.DataFrame) -> np.ndarray:
    raw_path = root / RAW_MATRIX
    header = pd.read_csv(raw_path, sep="\t", compression="gzip", nrows=0).columns.tolist()
    links = pd.read_csv(root / MAPPING, dtype=str, keep_default_na=False)
    known = links.loc[
        (links["accession"] == "GSE193258")
        & (links["expression_file"] == raw_path.name)
        & (links["mapping_status"] == "exact_title")
        & (links["gsm"].isin(samples["gsm"]))
    ]
    if len(known) != 36 or known["gsm"].nunique() != 36 or known["column_or_member"].nunique() != 36:
        raise ValueError("Missing original source sample mappings")
    source_columns = known.set_index("gsm").loc[samples["gsm"], "column_or_member"].tolist()
    wash_columns = wash["column_or_member"].tolist()
    requested = [header[0], *source_columns, *wash_columns]
    if len(set(requested)) != 61 or not set(requested).issubset(header):
        raise ValueError("Unexpected raw matrix columns; check mapping")
    frame = pd.read_csv(raw_path, sep="\t", compression="gzip", usecols=requested)
    frame = frame.loc[:, requested]
    ids = frame.iloc[:, 0].astype(str).str.strip()
    if ids.eq("").any() or ids.duplicated().any():
        raise ValueError("Invalid raw gene identifiers")
    frame.index = ids
    if not set(genes).issubset(set(frame.index)):
        raise ValueError("Processed genes are missing from the raw matrix")
    same = frame.loc[genes, source_columns].to_numpy(dtype=float).T
    if not np.isfinite(same).all() or not np.allclose(same, source, rtol=1e-6, atol=1e-6):
        raise ValueError("Raw reference and stage 08 source expression disagree")
    values = frame.loc[genes, wash_columns].to_numpy(dtype=float).T
    if not np.isfinite(values).all():
        raise ValueError("Washout expression contains invalid values")
    return values


def classifier():
    return Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=500)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)),
    ])


def summarize(records):
    scores = [row["p_dtp"] for row in records]
    return {
        "n": len(records),
        "mean_p_dtp": float(np.mean(scores)),
        "median_p_dtp": float(np.median(scores)),
        "predicted_dtp_count": sum(row["predicted_state"] == "DTP" for row in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    genes, x, samples, baseline = load_training(root)
    wash = load_washout_metadata(root)
    x_wash = load_washout_values(root, genes, x, samples, wash)
    y, lines = samples["state"].to_numpy(), samples["cell_line"].to_numpy()
    results = []
    for heldout in sorted(LONG_WASHOUT):
        test, train = (lines == heldout), (lines != heldout)
        model = classifier()
        model.fit(x[train], y[train])
        predicted = model.predict(x[test])
        expected = {
            row["gsm"]: row["predicted_state"]
            for fold in baseline["models"]["LogisticRegression"]["loco_folds"]
            if fold["held_out_cell_line"] == heldout
            for row in fold["predictions"]
        }
        actual = dict(zip(samples.loc[test, "gsm"], predicted))
        if actual != expected:
            raise ValueError(f"Stage 09 predictions not reproduced for {heldout}")

        holdwash = (wash["cell_line"].to_numpy() == heldout)
        if holdwash.sum() != 6:
            raise ValueError(f"Expected six held-out washout samples for {heldout}")
        indices = np.flatnonzero(holdwash)
        p_wash = model.predict_proba(x_wash[indices])
        pred_wash = model.predict(x_wash[indices])
        dtp_index = list(model.classes_).index("DTP")
        records = []
        for i, probabilities, predicted_state in zip(indices, p_wash, pred_wash):
            row = wash.iloc[int(i)]
            records.append({
                "gsm": row["gsm"],
                "challenge_group": row["challenge_group"],
                "treatment": row["treatment"],
                "predicted_state": str(predicted_state),
                "p_dtp": float(probabilities[dtp_index]),
                "alias_status": "automatic_title_and_treatment_check_only",
            })
        dtp_prob = model.predict_proba(x[test])[:, dtp_index]
        reference = [
            {"gsm": gsm, "predicted_state": str(state), "p_dtp": float(p)}
            for gsm, state, p, actual_state in zip(
                samples.loc[test, "gsm"], predicted, dtp_prob, y[test]
            ) if actual_state == "DTP"
        ]
        groups = {
            group: summarize([row for row in records if row["challenge_group"] == group])
            for group in ("ShortWashout", "LongWashout")
        }
        groups["OnDrugDTP"] = summarize(reference)
        results.append({"held_out_cell_line": heldout, "groups": groups, "washout_predictions": records})
        print(f"{heldout}: on-drug DTP {groups['OnDrugDTP']['mean_p_dtp']:.3f}; "
              f"short {groups['ShortWashout']['mean_p_dtp']:.3f}; "
              f"long {groups['LongWashout']['mean_p_dtp']:.3f}")

    report = {
        "analysis": "exploratory_washout_challenge",
        "design": "LOCO: train exclusively on Control/Acute/DTP; inspect unseen washout cells from the held-out cell line",
        "source_dataset": "GSE193258",
        "training_samples": 36,
        "washout_samples": 24,
        "shared_genes": len(genes),
        "stage09_predictions_reproduced": True,
        "washout_original_matrix_sha256": sha256(root / RAW_MATRIX),
        "folds": results,
        "limitations": [
            "Washout samples are not paired longitudinal samples and are not ground-truth negative controls for DTP.",
            "Long washout duration differs by cell line and is tied to regrowth.",
            "All DTP samples had long drug exposure; state and initial exposure duration remain confounded.",
            "Model confidence does not measure biological persistence or causality.",
            "Alias mappings were checked against names and treatment labels, not manually verified against the article.",
            "These results are exploratory and were designed after inspecting earlier LOCO outcomes.",
        ],
    }
    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {destination}")
    print("Washout challenge completed; DTP specificity and exposure-time effects are not identified.")


if __name__ == "__main__":
    main()
