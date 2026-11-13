
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier, STATES, cell_line_contrasts  # noqa: E402

PREVIOUS = Path("results/dcsl_stage23_exploratory.json")
REPORT = Path("results/dcsl_stage24_pc9_diagnostics.json")
FIGURE = Path("figures/dcsl_stage24_pc9_margins.png")
SOURCE_MATRIX = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
# Do not change these parameters or configuration definitions after seeing PC9 outcomes.
L2, GAMMA, TAU, K = 0.1, 0.05, 0.5, 500
CONFIGS = (
    ("ANOVA_CE", "anova", 0.0),
    ("Stable_CE", "stable", 0.0),
    ("ANOVA_Stability", "anova", GAMMA),
    ("DCSL", "stable", GAMMA),
)


def import_stage23(root: Path):
    path = root / "scripts/23_run_dcsl_experiments.py"
    spec = importlib.util.spec_from_file_location("dcsl_stage23_diagnostics_loader", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot import stage-23 loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_json(x):
    """Keep report strict JSON (no NaN or numpy scalars)."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(f"Not JSON serializable: {type(x)}")


def sample_diagnostics(model: DCSLClassifier, x: np.ndarray, meta: pd.DataFrame) -> list[dict]:
    logits = model.decision_function(x)
    prob = softmax(logits, axis=1)
    pred = np.asarray(STATES)[np.argmax(logits, axis=1)]
    return [{
        "gsm": str(gsm), "true_state": str(actual), "predicted_state": str(guess),
        "DTP_minus_Acute_logit": float(logit[2] - logit[1]),
        "DTP_probability": float(p[2]),
        "DTP_minus_Control_logit": float(logit[2] - logit[0]),
    } for gsm, actual, guess, logit, p in zip(
        meta["gsm"], meta["state"], pred, logits, prob
    )]


def top_contributions(gene_ids: np.ndarray, weights: np.ndarray,
                      observed_contrast: np.ndarray, limit: int = 10) -> dict:
    contribution = observed_contrast * weights
    indices = np.argsort(contribution, kind="stable")
    def entry(i):
        return {"gene_id": str(gene_ids[i]),
                "feature_contrast_PC9_DTP_minus_Acute": float(observed_contrast[i]),
                "DTP_minus_Acute_weight": float(weights[i]),
                "contrast_contribution": float(contribution[i])}
    return {"most_negative": [entry(i) for i in indices[:limit]],
            "most_positive": [entry(i) for i in indices[-limit:][::-1]]}


def train_to_test_comparison(model: DCSLClassifier, x_train: np.ndarray, y_train: np.ndarray,
                             g_train: np.ndarray, x_pc9: np.ndarray, y_pc9: np.ndarray,
                             all_gene_ids: np.ndarray) -> dict:
    # Using training-fit scaler only. DO NOT fit any transform on PC9.
    selected = model.selected_indices_
    z_tr = model.scaler_.transform(x_train[:, model.nonconstant_indices_])[:, model.selected_local_]
    z_te = model.scaler_.transform(x_pc9[:, model.nonconstant_indices_])[:, model.selected_local_]
    train_delta = cell_line_contrasts(z_tr, y_train, g_train)  # one vector / training line
    pc9_delta = z_te[y_pc9 == "DTP"].mean(axis=0) - z_te[y_pc9 == "Acute"].mean(axis=0)
    weight_dtp_minus_acute = model.coef_[:, 2] - model.coef_[:, 1]
    train_group_margins = train_delta @ weight_dtp_minus_acute
    pc9_group_margin = float(pc9_delta @ weight_dtp_minus_acute)
    # A diagnostic look at withheld labels, NEVER used to select genes or fit model.
    train_mean_delta = train_delta.mean(axis=0)
    sign_ok = (pc9_delta * train_mean_delta > 0)
    nonzero = (pc9_delta != 0) & (train_mean_delta != 0)
    sign_concordance = float(sign_ok[nonzero].mean()) if nonzero.any() else None
    return {
        "training_cell_lines": model.train_groups_,
        "train_DTP_minus_Acute_group_margins": [float(t) for t in train_group_margins],
        "train_group_margin_mean": float(np.mean(train_group_margins)),
        "train_group_margin_variance": float(np.var(train_group_margins)),
        "heldout_PC9_mean_DTP_minus_Acute_margin_difference": pc9_group_margin,
        "selected_genes_same_direction_in_PC9_fraction": sign_concordance,
        "sign_comparable_feature_count": int(nonzero.sum()),
        "posthoc_feature_contributions": top_contributions(
            all_gene_ids[selected], weight_dtp_minus_acute, pc9_delta),
        "contribution_sum_matches_heldout_margin_difference": bool(np.isclose(
            (weight_dtp_minus_acute * pc9_delta).sum(), pc9_group_margin, atol=1e-9)),
    }


def compare_penalty_effect(no_penalty: dict, with_penalty: dict) -> dict:
    a = {r["gsm"]: r for r in no_penalty["heldout_PC9_samples"]}
    b = {r["gsm"]: r for r in with_penalty["heldout_PC9_samples"]}
    if set(a) != set(b):
        raise ValueError("Different PC9 GSMs between ablations")
    return {
        "selected_features_identical": bool(no_penalty["feature_hash"] == with_penalty["feature_hash"]),
        "train_margin_variance_without_penalty": no_penalty["contrast_diagnostics"]["train_group_margin_variance"],
        "train_margin_variance_with_penalty": with_penalty["contrast_diagnostics"]["train_group_margin_variance"],
        "heldout_PC9_margin_shift_with_minus_without": [
            {"gsm": k, "change": float(b[k]["DTP_minus_Acute_logit"] - a[k]["DTP_minus_Acute_logit"])}
            for k in a
        ],
        "heldout_PC9_DTP_recall_without_penalty": no_penalty["heldout_PC9_DTP_recall"],
        "heldout_PC9_DTP_recall_with_penalty": with_penalty["heldout_PC9_DTP_recall"],
    }


def plot_margins(models: dict, info: pd.DataFrame, path: Path):
    # Order by true state (for visibility only), not training or tuning.
    states = {s: i for i, s in enumerate(STATES)}
    ordered_gsms = info.sort_values("state", key=lambda s: s.map(states))["gsm"].tolist()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, (model_name, result) in enumerate(models.items()):
        mapped = {r["gsm"]: r for r in result["heldout_PC9_samples"]}
        y = [mapped[g]["DTP_minus_Acute_logit"] for g in ordered_gsms]
        ax.scatter(np.arange(len(y)) + (i - 1.5) * .13, y, label=model_name, s=55, alpha=.85)
    ax.axhline(0, linestyle="--", linewidth=1, color="gray", label="DTP = Acute logit")
    ax.axvline(2.5, linewidth=.7, color="gray")
    ax.axvline(5.5, linewidth=.7, color="gray")
    ax.set_xticks(range(9), [f"{j+1}: {states_}" for j, states_ in enumerate(
        info.set_index("gsm").loc[ordered_gsms, "state"])], rotation=35, ha="right")
    ax.set_ylabel("DTP minus Acute logit (model score; not a probability)")
    ax.set_title("GSE193258 held-out PC9: DTP–Acute score margins")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(root: Path) -> dict:
    stage = import_stage23(root)
    prior = json.loads((root / PREVIOUS).read_text(encoding="utf-8"))
    if prior.get("analysis") != "DCSL_stage23_fixed_ablation_prototype":
        raise ValueError("Expected an unmodified stage-23 report")
    params = prior.get("fixed_parameters_no_heldout_tuning")
    if params != {"features": K, "l2": L2, "gamma": GAMMA, "tau": TAU}:
        raise ValueError("Stage-23 hyperparameters do not match")
    _, x, meta, _, _ = stage.load_stage09(root)  # built-in SHA checks: original data
    y = meta["state"].to_numpy()
    groups = meta["cell_line"].to_numpy()
    train = groups != "PC9"
    test = groups == "PC9"
    if int(train.sum()) != 27 or int(test.sum()) != 9:
        raise ValueError("Unexpected held-out PC9 counts")
    test_meta = meta.loc[test].copy().reset_index(drop=True)
    if test_meta["state"].value_counts().to_dict() != {k: 3 for k in STATES}:
        raise ValueError("PC9 testing class count changed")
    source = pd.read_csv(root / SOURCE_MATRIX, sep="\t", compression="gzip", usecols=["gene_id"], dtype=str)
    gene_ids = source["gene_id"].to_numpy()
    if len(gene_ids) != x.shape[1] or len(np.unique(gene_ids)) != len(gene_ids):
        raise ValueError("Gene IDs do not align with source matrix")
    report = {
        "analysis": "stage24_posthoc_PC9_failure_diagnostics",
        "source": "GSE193258", "held_out_cell_line": "PC9",
        "train_samples": int(train.sum()), "heldout_PC9_samples": int(test.sum()),
        "input_sha256": prior["input_sha256"],
        "configs": [q[0] for q in CONFIGS],
        "parameters_unchanged_since_stage23": params,
        "heldout_labels_used_only_for_posthoc_diagnostics": True,
        "no_tuning_performed": True,
        "PC9_data_never_enter_fit_feature_ranking_or_scaler": True,
        "notes": [
            "Held-out labels were previously inspected; exploratory diagnosis, not new validation.",
            "DTP–Acute scores are logit differences, not calibrated probabilities or causal effects.",
            "Feature contributions depend on training-fold scaling and are not stable biomarker estimates.",
            "The contrast on held-out PC9 uses labels ONLY after model fitting, never for selection or tuning.",
            "Original stage-23 predictions and selected-feature hashes are verified exactly.",
            "The original data labels and expression-scale harmonization remain provisional.",
        ],
        "models": {},
    }
    for label, selector, gamma in CONFIGS:
        model = DCSLClassifier(selector=selector, n_features=K, l2=L2, gamma=gamma, tau=TAU).fit(
            x[train], y[train], groups[train])
        samples = sample_diagnostics(model, x[test], test_meta)
        previous = next(f for f in prior["models"][label]["loco_folds"] if f["held_out_cell_line"] == "PC9")
        old = {r["gsm"]: (r["actual"], r["predicted"]) for r in previous["predictions"]}
        current = {r["gsm"]: (r["true_state"], r["predicted_state"]) for r in samples}
        selected_digest = hashlib.sha256(model.selected_indices_.astype("<i8").tobytes()).hexdigest()
        if old != current or selected_digest != previous["selected_feature_indices_sha256"]:
            raise ValueError(f"Stage-23 fold was not reproduced for {label}")
        if not np.isclose(model.train_margin_variance_, previous["train_margin_variance"], atol=1e-7, rtol=1e-7):
            raise ValueError(f"Training margin variance changed since stage 23 for {label}")
        tp = sum(r["true_state"] == "DTP" and r["predicted_state"] == "DTP" for r in samples)
        diag = train_to_test_comparison(model, x[train], y[train], groups[train], x[test], y[test], gene_ids)
        report["models"][label] = {
            "reproduces_stage23_PC9_predictions_and_features": True,
            "feature_hash": selected_digest,
            "heldout_PC9_DTP_recall": float(tp / 3),
            "heldout_PC9_samples": samples,
            "contrast_diagnostics": diag,
        }
        dtp_margin = [r["DTP_minus_Acute_logit"] for r in samples if r["true_state"] == "DTP"]
        print(f"{label}: PC9 DTP recall {tp}/3 | DTP logit-margin range: {min(dtp_margin):+.3f} to {max(dtp_margin):+.3f} "
              f"| training group margin var {model.train_margin_variance_:.5g}")
    report["penalty_ablation_diagnostics"] = {
        "anova": compare_penalty_effect(report["models"]["ANOVA_CE"], report["models"]["ANOVA_Stability"]),
        "stable": compare_penalty_effect(report["models"]["Stable_CE"], report["models"]["DCSL"]),
    }
    plot_margins(report["models"], test_meta, root / FIGURE)
    dest = root / REPORT
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=safe_json, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved report: {dest}")
    print(f"Saved figure: {root / FIGURE}")
    print("Post-hoc diagnostics only. Do NOT use held-out PC9 to tune DCSL or claim external validation.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
