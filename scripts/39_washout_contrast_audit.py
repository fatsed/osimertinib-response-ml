"""Exploratory, training-only-feature analysis of Acute/DTP contrast after washout.

Uses the GSE193258 source-only 19,712-gene matrix and its already downloaded
60-sample raw log2TPM matrix. In each of four LOCO folds, feature selection and
scaling use only Acute/DTP samples from the THREE other cell lines. Washout
samples are NEVER fit, used to select features, or treated as DTP labels.

The within-heldout contrast retention index is purely descriptive; the
held-out Acute and on-drug DTP centroids are used as *references*, not as a
predictive model or independent validation. Unpaired washout, exposure-time
confounding and different long-washout durations preclude causal inference.

Run: python scripts/39_washout_contrast_audit.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
STAGE31 = Path("results/dcsl_stage31_evaluation_protocol.json")
STAGE35 = Path("results/dcsl_stage35_binary_dtp_acute_loco.json")
STAGE11 = Path("results/gse193258_washout_challenge.json")
SOURCE_MATRIX = Path("data/processed/GSE193258/log2tpm_36samples.tsv.gz")
SOURCE_METADATA = Path("data/processed/GSE193258/samples_36.csv")
WASHOUT_RAW = Path("data/raw/GSE193258/GSE193258_RNAseq_log2TPM_abundance.tsv.gz")
OUTPUT = Path("results/washout_contrast_audit.json")
N_FEATURES = 500  # Previously fixed in Stage 35, not tuned on washout.
LINES = ("H1975", "HCC2935", "HCC827", "PC9")


def load_script(root: Path, filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(root / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fit_training_features(x: np.ndarray, labels: np.ndarray, *, k: int = N_FEATURES):
    """Return train-only ANOVA-selected and standardized features (Stage 35 order)."""
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=str)
    if x.ndim != 2 or x.shape[0] != len(labels) or not np.isfinite(x).all():
        raise ValueError("Invalid training matrix")
    if set(labels) != {"Acute", "DTP"}:
        raise ValueError("Exactly Acute/DTP labels required in training")
    if k < 1 or k > x.shape[1]:
        raise ValueError("Invalid k")
    pipe = Pipeline([
        ("variance", VarianceThreshold(threshold=1e-8)),
        ("anova", SelectKBest(score_func=f_classif, k=k)),
        ("scale", StandardScaler()),
    ])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        train_z = pipe.fit_transform(x, labels)
    selected = np.flatnonzero(pipe.named_steps["variance"].get_support())[
        pipe.named_steps["anova"].get_support()
    ]
    if len(selected) != k or not np.isfinite(train_z).all():
        raise ValueError("Invalid selected expression features")
    return pipe, np.asarray(selected, dtype=int), train_z


def normalized_direction(z_train: np.ndarray, train_labels: np.ndarray, groups: np.ndarray):
    """Unweighted mean of DTP-minus-Acute contrasts across training cell lines."""
    labels, groups = np.asarray(train_labels, dtype=str), np.asarray(groups, dtype=str)
    if len(z_train) != len(labels) or len(groups) != len(labels) or not np.isfinite(z_train).all():
        raise ValueError("Mismatched training labels")
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two training cell lines needed")
    deltas = []
    for line in unique:
        acute = z_train[(groups == line) & (labels == "Acute")]
        dtp = z_train[(groups == line) & (labels == "DTP")]
        if len(acute) != 3 or len(dtp) != 3:
            raise ValueError(f"Expected 3 Acute + 3 DTP replicates for {line}")
        deltas.append(dtp.mean(axis=0) - acute.mean(axis=0))
    avg = np.mean(deltas, axis=0)
    n = np.linalg.norm(avg)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError("Training contrast too small")
    return avg / n


def contrast_summary(acute: np.ndarray, dtp: np.ndarray,
                     short: np.ndarray, long: np.ndarray,
                     source_direction: np.ndarray) -> dict:
    """Descriptive means in training-selected space; no held-out model fitting.

    Retention = dot(Washout-Acute, DTP-Acute) / ||DTP-Acute||^2.
    It is 0 at the Acute centroid, 1 at the on-drug DTP centroid; can be
    negative or >1. Not a probability or a biological persistence estimate.
    """
    arrays = [np.asarray(v, dtype=float) for v in (acute, dtp, short, long)]
    if any(v.ndim != 2 or v.shape[0] != 3 for v in arrays):
        raise ValueError("Expected three samples in every held-out condition")
    p = arrays[0].shape[1]
    u = np.asarray(source_direction, dtype=float)
    if not p or any(v.shape[1] != p or not np.isfinite(v).all() for v in arrays) or u.shape != (p,):
        raise ValueError("Invalid held-out feature dimensions or values")
    if not np.isfinite(u).all() or not np.isclose(np.linalg.norm(u), 1.0, atol=1e-6):
        raise ValueError("Source direction must be finite and unit-length")
    a, d, s, l = (v.mean(axis=0) for v in arrays)
    ref = d - a
    energy = float(ref @ ref)
    if energy <= 1e-12:
        raise ValueError("Held-out DTP-Acute contrast is essentially zero")
    ref_norm = np.sqrt(energy)

    def summarize(v):
        delta = v - a
        return {
            "retention_along_heldout_DTP_Acute": float((delta @ ref) / energy),
            "cosine_to_heldout_DTP_Acute": float((delta @ ref) / (np.linalg.norm(delta) * ref_norm)) if np.linalg.norm(delta) > 1e-12 else None,
            "shift_along_training_DTP_direction": float(delta @ u),
            "distance_to_heldout_Acute_centroid": float(np.linalg.norm(delta)),
            "distance_to_heldout_DTP_centroid": float(np.linalg.norm(v - d)),
        }

    return {
        "n_per_condition": 3,
        "heldout_DTP_Acute_alignment_to_training_direction": float((ref @ u) / ref_norm),
        "on_drug_DTP_reference": summarize(d),
        "short_washout": summarize(s),
        "long_washout": summarize(l),
    }


def analyze(root: Path) -> dict:
    stage11 = load_script(root, "11_washout_challenge.py", "washout_stage11_for_audit")
    stage34 = load_script(root, "34_source_only_rebaseline.py", "source_stage34_for_audit")
    stage35 = load_script(root, "35_dtp_vs_acute_benchmark.py", "binary_stage35_for_audit")
    old_washout = json.loads((root / STAGE11).read_text(encoding="utf-8"))
    old_binary = json.loads((root / STAGE35).read_text(encoding="utf-8"))
    protocol = json.loads((root / STAGE31).read_text(encoding="utf-8"))
    provenance = old_binary.get("source_provenance", {})
    if old_binary.get("n_binary_samples") != 24 or old_binary.get("features_source_only") != 19712:
        raise ValueError("Stage 35 expected cohort/feature universe changed")
    if provenance.get("prepared_matrix_sha256") != stage11.sha256(root / SOURCE_MATRIX):
        raise ValueError("Stage 35 source expression checksum mismatch")
    if provenance.get("prepared_metadata_sha256") != stage11.sha256(root / SOURCE_METADATA):
        raise ValueError("Stage 35 metadata checksum mismatch")
    if (provenance.get("source_raw_sha256") != stage11.sha256(root / WASHOUT_RAW) or
            old_washout.get("washout_original_matrix_sha256") != stage11.sha256(root / WASHOUT_RAW)):
        raise ValueError("GSE193258 original 60-sample expression file changed")
    if old_binary.get("frozen_stage31_sha256") != stage11.sha256(root / STAGE31):
        raise ValueError("Frozen Stage 31 protocol changed")

    metadata = pd.read_csv(root / SOURCE_METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(metadata, protocol)
    metadata = metadata.rename(columns={"candidate_group": "state"})
    stage35.validate_binary_cohort(metadata, protocol["nested_loco_protocol"]["folds"])
    frame = pd.read_csv(root / SOURCE_MATRIX, sep="\t", compression="gzip")
    genes, source_x = stage34.validate_matrix(frame, metadata.gsm.tolist())
    wash = stage11.load_washout_metadata(root)
    wash_x = stage11.load_washout_values(root, list(map(str, genes)), source_x, metadata, wash)
    if set(wash.gsm) & set(metadata.gsm):
        raise ValueError("Washout/sample overlap")

    states, lines = metadata.state.to_numpy(), metadata.cell_line.to_numpy()
    out = {}
    for heldout in LINES:
        train_mask = (lines != heldout) & np.isin(states, ["Acute", "DTP"])
        held_mask = lines == heldout
        if train_mask.sum() != 18 or held_mask.sum() != 9:
            raise ValueError("Unexpected train and held-out cell-line sample counts")
        transform, selected, z_train = fit_training_features(source_x[train_mask], states[train_mask])
        train_direction = normalized_direction(z_train, states[train_mask], lines[train_mask])
        held_z = transform.transform(source_x[held_mask])
        held_labels = states[held_mask]
        line_wash = wash.cell_line.eq(heldout).to_numpy()
        z_wash = transform.transform(wash_x[line_wash])
        wash_groups = wash.loc[line_wash, "challenge_group"].to_numpy()
        if sum(held_labels == "Acute") != 3 or sum(held_labels == "DTP") != 3:
            raise ValueError(f"Held-out states missing: {heldout}")
        if sum(wash_groups == "ShortWashout") != 3 or sum(wash_groups == "LongWashout") != 3:
            raise ValueError(f"Washout groups missing: {heldout}")
        out[heldout] = {
            "selected_training_feature_count": len(selected),
            "selected_feature_indices_zero_based": selected.tolist(),
            "selected_feature_gene_symbols": genes[selected].tolist(),
            "training_cell_lines": sorted(set(lines[train_mask])),
            "description": "All comparisons after transforming with the other three cell lines only",
            **contrast_summary(
                held_z[held_labels == "Acute"], held_z[held_labels == "DTP"],
                z_wash[wash_groups == "ShortWashout"], z_wash[wash_groups == "LongWashout"],
                train_direction,
            ),
        }
        print(f"{heldout}: short retention={out[heldout]['short_washout']['retention_along_heldout_DTP_Acute']:.3f}, "
              f"long retention={out[heldout]['long_washout']['retention_along_heldout_DTP_Acute']:.3f}, "
              f"source/heldout contrast cosine={out[heldout]['heldout_DTP_Acute_alignment_to_training_direction']:.3f}")

    return {
        "analysis": "development_only_washout_gene_contrast_retention",
        "study": "GSE193258", "training_source_only_genes": len(genes),
        "new_model_fitted": False, "external_validation": False,
        "selected_features_only_from_other_cell_lines": True,
        "no_feature_selection_from_washout": True,
        "source_matrix_sha256": provenance["prepared_matrix_sha256"],
        "source_raw_sha256": provenance["source_raw_sha256"],
        "retention_definition": "(mean_washout - mean_acute) dot (mean_DTP - mean_acute) / ||mean_DTP - mean_acute||^2, on features selected/scaled by other cell lines",
        "limitations": [
            "Retentions are centroid-relative descriptive projections, not probabilities, resistance, biological persistence or causal effects.",
            "The observed held-out DTP and Acute centroids are used as reference labels; these estimates are NOT out-of-sample prediction accuracy.",
            "Only three unpaired sample replicates per condition; washout samples have no ground-truth DTP/reversion labels.",
            "The 24h Acute and 21d DTP groups differ in exposure duration, confounded with state.",
            "Long washout periods differ across cell lines (72 h, 10 d, 96 h, 7 d); never pool raw retention across them as one time point.",
            "All four cell lines and the Stage-11 washout patterns have been inspected before; results are exploratory.",
            "Washout sample aliases were mechanically title-matched; manually checking publication methods is still necessary.",
        ],
        "per_cell_line": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = analyze(root)
    dest = root / OUTPUT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Saved:", dest)
    print("Exploratory analysis only; no claim of washout persistence, resistance or domain-invariant signature.")


if __name__ == "__main__":
    main()
