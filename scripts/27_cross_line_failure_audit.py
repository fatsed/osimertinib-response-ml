"""Stage 27: post-hoc LOCO feature/shift/cell-cycle diagnostics.

NO new model training, parameter tuning, or independent external validation.
Uses stage-26 fold-specific selections, stage-23/09 verified source matrix,
and the stage-13 audited Hallmark gene sets. Held-out labels are accessed
ONLY for diagnostic statistics after training-fold feature selection.

Run: python scripts/27_cross_line_failure_audit.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MATRIX = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
STAGE23 = Path("results/dcsl_stage23_exploratory.json")
STAGE26 = Path("results/dcsl_stage26_feature_stability.json")
HALLMARK = Path("results/gse193258_full_cell_cycle_sets.json")
REPORT = Path("results/dcsl_stage27_failure_audit.json")
FIGURE = Path("figures/dcsl_stage27_failure_audit.png")
SELECTORS = ("anova", "stable")
STATES = ("Control", "Acute", "DTP")


def load_stage23(root: Path):
    path = root / "scripts/23_run_dcsl_experiments.py"
    spec = importlib.util.spec_from_file_location("stage23_for_stage27_audit", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load stage 23")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def validate_selections(stage26: dict, prior23: dict, gene_ids: np.ndarray,
                        groups: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Only reuse fold-selections whose training-only provenance is confirmed."""
    if not stage26.get("stage23_feature_hashes_reproduced"):
        raise ValueError("Stage 26 feature hashes were not verified")
    if stage26.get("input_sha256") != prior23.get("input_sha256"):
        raise ValueError("Stage-23/26 data hashes differ")
    if stage26.get("source_genes") != len(gene_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("Unexpected gene identifiers")
    expected = set(np.unique(groups))
    idx = {gene: i for i, gene in enumerate(gene_ids)}
    out = {}
    for method in SELECTORS:
        selections = {}
        folds = stage26["selectors"][method]["folds"]
        for fold in folds:
            line = fold["held_out_cell_line"]
            ids = fold["selected_gene_ids"]
            if line in selections or len(ids) != 500 or len(set(ids)) != 500:
                raise ValueError(f"Invalid selection: {method}/{line}")
            try:
                indices = np.sort([idx[g] for g in ids]).astype(int)
            except KeyError as exc:
                raise ValueError(f"Missing gene in source matrix: {exc}") from exc
            import hashlib
            digest = hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest()
            if digest != fold["selected_indices_sha256"]:
                raise ValueError(f"Selection/hash mismatch: {method}/{line}")
            selections[line] = indices
        if set(selections) != expected:
            raise ValueError(f"Missing fold for {method}")
        out[method] = selections
    return out


def load_hallmark_sets(report: dict, universe: set[str]) -> dict[str, set[str]]:
    """Use the two *previously audited* stage-13 Hallmark sets; no download."""
    if report.get("dataset") != "GSE193258" or report.get("genes") != len(universe):
        raise ValueError("Unexpected stage-13 reference")
    names = ("HALLMARK_G2M_CHECKPOINT", "HALLMARK_E2F_TARGETS")
    if not all(name in report.get("gene_sets", {}) for name in names):
        raise ValueError("Stage-13 Hallmark sets are missing")
    sets = {}
    for name in names:
        data = report["gene_sets"][name]
        available = set(data["available_genes"])
        if not available <= universe or len(available) != data["available_count"]:
            raise ValueError(f"Invalid gene-set annotations: {name}")
        sets[name] = available
    return sets


def sign_agreement(train_mean_contrast: np.ndarray, heldout_contrast: np.ndarray) -> float | None:
    """Post-hoc DTP-Acute direction concordance; held-out labels for diagnosis only."""
    a, b = np.asarray(train_mean_contrast), np.asarray(heldout_contrast)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("Contrast dimensions mismatch")
    usable = (a != 0) & (b != 0)
    return float(np.mean(np.sign(a[usable]) == np.sign(b[usable]))) if np.any(usable) else None


def standardized_centroid_shift(train_z: np.ndarray, train_y: np.ndarray,
                                train_groups: np.ndarray, heldout_z: np.ndarray,
                                heldout_y: np.ndarray) -> dict:
    """RMS centroid shift; training within-state contrasts provide a scale reference.

    Held-out labels are used for descriptive comparison only, after training-only
    feature selection and standardization. No p values are inferred from n=3.
    """
    if train_z.ndim != 2 or heldout_z.ndim != 2 or train_z.shape[1] != heldout_z.shape[1]:
        raise ValueError("Invalid shape")
    lines = sorted(set(train_groups))
    if len(lines) < 3 or not np.isfinite(train_z).all() or not np.isfinite(heldout_z).all():
        raise ValueError("Invalid training groups or expression")
    k = train_z.shape[1]
    result = {}
    for state in STATES:
        train_sub = train_z[train_y == state]
        test_sub = heldout_z[heldout_y == state]
        if not len(train_sub) or not len(test_sub):
            raise ValueError(f"State absent from diagnostic: {state}")
        heldout_dist = float(np.linalg.norm(test_sub.mean(axis=0) - train_sub.mean(axis=0)) / np.sqrt(k))
        training_leave_group_dists = []
        for line in lines:
            one = train_z[(train_groups == line) & (train_y == state)]
            other = train_z[(train_groups != line) & (train_y == state)]
            if not len(one) or not len(other):
                raise ValueError("Incomplete training group/state")
            training_leave_group_dists.append(float(np.linalg.norm(one.mean(axis=0) - other.mean(axis=0)) / np.sqrt(k)))
        reference = float(np.mean(training_leave_group_dists))
        result[state] = {
            "rms_heldout_vs_training_state_centroid": heldout_dist,
            "mean_rms_train_group_vs_other_train_centroids": reference,
            "heldout_to_training_reference_ratio": heldout_dist / reference if reference > 0 else None,
            "training_reference_per_group": dict(zip(lines, training_leave_group_dists)),
            "heldout_sample_count": int(len(test_sub)),
        }
    return result


def cycle_overlap(selected: set[str], sets: dict[str, set[str]], universe: set[str]) -> dict:
    if not selected or not selected <= universe:
        raise ValueError("Invalid selection")
    combined = set().union(*sets.values())
    result = {}
    for name, members in {**sets, "G2M_OR_E2F_UNION": combined}.items():
        overlap = len(selected & members)
        expected = len(selected) * len(members) / len(universe)
        result[name] = {
            "overlap_gene_count": overlap,
            "fraction_of_selected": float(overlap / len(selected)),
            "expected_count_if_random": float(expected),
            "observed_to_expected": float(overlap / expected) if expected else None,
            "background_fraction": float(len(members) / len(universe)),
        }
    return result


def save_figure(report: dict, dest: Path) -> None:
    folds = report["heldout_cell_lines"]
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.5))
    x = np.arange(len(folds))
    for method, offset in (("anova", -.18), ("stable", .18)):
        vals = [report["selectors"][method][c]["DTP_Acute_heldout_direction_agreement"] for c in folds]
        axs[0].bar(x + offset, vals, width=.36, label=method)
        cycle = [report["selectors"][method][c]["cell_cycle_annotation"]["G2M_OR_E2F_UNION"]["overlap_gene_count"] for c in folds]
        axs[1].bar(x + offset, cycle, width=.36, label=method)
        shifts = [report["selectors"][method][c]["state_conditional_shift"]["DTP"]["heldout_to_training_reference_ratio"] for c in folds]
        axs[2].bar(x + offset, shifts, width=.36, label=method)
    for ax in axs:
        ax.set_xticks(x, folds, rotation=22)
        ax.legend(fontsize=8)
    axs[0].set_title("Held-out DTP-Acute direction agreement")
    axs[0].set_ylabel("Fraction (post-hoc labels)")
    axs[0].set_ylim(0, 1)
    axs[1].set_title("Selected Hallmark cycle genes")
    axs[1].set_ylabel("Count in 500 selected genes")
    axs[2].set_title("DTP state-conditional centroid shift")
    axs[2].set_ylabel("Held-out / training reference")
    axs[2].axhline(1, linestyle="--", color="gray", lw=1)
    fig.suptitle("Stage 27: exploratory cross-cell-line diagnostics", fontsize=12)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=165)
    plt.close(fig)


def run(root: Path) -> dict:
    stage23 = load_stage23(root)
    _, x, meta, _, _ = stage23.load_stage09(root)  # verifies stage-09 data checksums
    gene_ids = pd.read_csv(root / MATRIX, sep="\t", compression="gzip", usecols=["gene_id"],
                           dtype=str, keep_default_na=False)["gene_id"].to_numpy()
    if x.shape != (36, 18013) or len(gene_ids) != x.shape[1]:
        raise ValueError("Unexpected dataset shape/order")
    y = meta["state"].to_numpy()
    groups = meta["cell_line"].to_numpy()
    prior23 = json.loads((root / STAGE23).read_text(encoding="utf-8"))
    prior26 = json.loads((root / STAGE26).read_text(encoding="utf-8"))
    if prior23.get("fixed_parameters_no_heldout_tuning", {}).get("features") != 500:
        raise ValueError("Unexpected stage-23 feature settings")
    fold_selections = validate_selections(prior26, prior23, gene_ids, groups)
    sets = load_hallmark_sets(json.loads((root / HALLMARK).read_text(encoding="utf-8")), set(gene_ids))
    folds = sorted(set(groups))
    report = {
        "analysis": "stage27_posthoc_cross_cell_line_failure_audit",
        "source": "GSE193258", "samples": len(y), "genes": len(gene_ids),
        "input_sha256": prior23["input_sha256"],
        "stage26_selection_hashes_verified": True,
        "number_of_genes_selected_per_fold": 500,
        "hallmark_reference": str(HALLMARK),
        "heldout_cell_lines": folds,
        "no_model_training_or_parameter_tuning": True,
        "heldout_labels_used_only_for_posthoc_diagnostics": True,
        "not_independent_validation": True,
        "limitations": [
            "The held-out data and PC9 have been repeatedly inspected; results are exploratory.",
            "Labels were used only to compute post-hoc contrasts and per-state centroid shifts, never to select genes.",
            "Each cell line has only three replicates per state; no formal uncertainty or causal claim.",
            "Cycle annotation overlap is descriptive, not evidence that cell cycle causes prediction errors.",
            "Standardization and feature selection come only from each training fold.",
            "Cross-study expression-scale and biological label validation remain provisional.",
        ],
        "stage26_overlap_reference": {m: prior26["selectors"][m]["average_pairwise_jaccard"] for m in SELECTORS},
        "selectors": {},
    }
    for method in SELECTORS:
        report["selectors"][method] = {}
        for heldout in folds:
            train = groups != heldout
            test = ~train
            selected = fold_selections[method][heldout]
            # Same fixed gene panel as stage 26; fit scaler on training fold ONLY.
            scaler = StandardScaler().fit(x[train][:, selected])
            ztr = scaler.transform(x[train][:, selected])
            zte = scaler.transform(x[test][:, selected])
            train_delta = []
            for line in sorted(set(groups[train])):
                dtp = ztr[(groups[train] == line) & (y[train] == "DTP")]
                acute = ztr[(groups[train] == line) & (y[train] == "Acute")]
                if len(dtp) != 3 or len(acute) != 3:
                    raise ValueError("Unexpected training group/state sizes")
                train_delta.append(dtp.mean(axis=0) - acute.mean(axis=0))
            reference = np.mean(train_delta, axis=0)
            heldout_delta = zte[y[test] == "DTP"].mean(axis=0) - zte[y[test] == "Acute"].mean(axis=0)
            selected_ids = set(gene_ids[selected])
            agreement = sign_agreement(reference, heldout_delta)
            shifts = standardized_centroid_shift(ztr, y[train], groups[train], zte, y[test])
            report["selectors"][method][heldout] = {
                "selection_sha256": next(f["selected_indices_sha256"] for f in prior26["selectors"][method]["folds"] if f["held_out_cell_line"] == heldout),
                "train_samples": int(train.sum()), "heldout_samples": int(test.sum()),
                "DTP_Acute_heldout_direction_agreement": agreement,
                "state_conditional_shift": shifts,
                "cell_cycle_annotation": cycle_overlap(selected_ids, sets, set(gene_ids)),
            }
        ratios = [report["selectors"][method][c]["state_conditional_shift"]["DTP"]["heldout_to_training_reference_ratio"] for c in folds]
        directions = [report["selectors"][method][c]["DTP_Acute_heldout_direction_agreement"] for c in folds]
        counts = [report["selectors"][method][c]["cell_cycle_annotation"]["G2M_OR_E2F_UNION"]["overlap_gene_count"] for c in folds]
        print(f"{method}: direction agreement=" + ", ".join(f"{c}={a:.3f}" for c, a in zip(folds, directions)))
        print(f"{method}: DTP shift ratio=" + ", ".join(f"{c}={a:.3f}" for c, a in zip(folds, ratios)))
        print(f"{method}: Hallmark G2M/E2F union genes=" + ", ".join(f"{c}={v}" for c, v in zip(folds, counts)))
    report_path = root / REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    save_figure(report, root / FIGURE)
    print("Saved report:", report_path)
    print("Saved figure:", root / FIGURE)
    print("Post-hoc diagnostics only: do NOT tune DCSL against these held-out groups.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
