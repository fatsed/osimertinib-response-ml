"""Stage 33: frozen nested-LOCO Logistic vs DCSL error decomposition.

Exploratory *post-hoc* analysis of stage 32. Refit exactly the selected models
on each outer TRAIN partition and verify their features and predictions against
the locked stage-32 report. Separately fit TWO pre-declared, matched-objective
DCSL ablations (ANOVA/Stable selector; gamma=0) to isolate the selector from
the contrast-variance penalty. They are diagnostics, NOT candidate selection.

No held-out label is used to fit, select, normalize or tune a model. Both PC9
and all other source groups have already been inspected in earlier stages.
No independent biological validation or method novelty is claimed here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier, cell_line_contrasts  # noqa: E402

STAGE32_SCRIPT = Path("scripts/32_run_nested_loco.py")
STAGE32_JSON = Path("results/dcsl_stage32_nested_loco.json")
STAGE31_JSON = Path("results/dcsl_stage31_evaluation_protocol.json")
BASELINE_JSON = Path("results/baseline_gse193258_loco.json")
OUT = Path("results/dcsl_stage33_dcsl_vs_logistic_diagnostics.json")
FIG = Path("figures/dcsl_stage33_pc9_dtp_logit_margins.png")
MATCHED_ABLATIONS = (
    "matched_anova_gamma0", "matched_stable_gamma0",
    "matched_anova_logistic_l2", "matched_stable_logistic_l2",
)


def import_stage32(root: Path):
    file = root / STAGE32_SCRIPT
    spec = importlib.util.spec_from_file_location("stage32_frozen_for_stage33", file)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_stage32_unchanged(prior: dict, protocol: dict, stage32, root: Path) -> None:
    """Fail closed on changes in prior data, protocol, evaluation or labels."""
    if (prior.get("analysis") != "stage32_nested_source_only_loco_exploratory"
            or prior.get("source") != "GSE193258"
            or prior.get("n_samples") != 36 or prior.get("n_genes") != 18013
            or prior.get("candidates_from_stage31_unchanged") is not True
            or prior.get("held_out_or_target_labels_used_for_fitting_preprocessing_or_selection") is not False
            or prior.get("target_expression_read") is not False
            or prior.get("independent_validation") is not False
            or prior.get("source_labels_still_provisional") is not True
            or prior.get("source_feature_universe_historically_built_with_target_genes") is not True):
        raise ValueError("Unexpected stage-32 provenance or independence status")
    if prior.get("protocol_sha256") != stage32.sha256(root / STAGE31_JSON):
        raise ValueError("Stage-31 protocol has changed since stage 32")
    baseline = json.loads((root / BASELINE_JSON).read_text(encoding="utf-8"))
    expected_inputs = stage32.verify_manifest(protocol, baseline, root)
    if expected_inputs != prior.get("input_sha256"):
        raise ValueError("Stage-32 input hashes no longer match the source/protocol")
    stage32.validate_candidate_specs(protocol["candidate_configurations_for_future_implementation"])
    folds = prior.get("outer_folds")
    if not isinstance(folds, list) or [f.get("outer_test_cell_line") for f in folds] != list(stage32.LINES):
        raise ValueError("Stage-32 outer fold membership or order changed")
    for fold in folds:
        for family in ("LogisticRegression", "DCSL"):
            details = fold.get("family_outer_results", {}).get(family, {})
            chosen = fold.get("selected_by_family_inner_only", {}).get(family)
            if not chosen or details.get("selected_candidate_id") != chosen:
                raise ValueError(f"Stage-32 selected candidate mismatch: {family}")


def feature_indices(model, family: str) -> np.ndarray:
    if family == "LogisticRegression":
        idx = np.flatnonzero(model.named_steps["filter"].get_support())
        return idx[model.named_steps["select"].get_support()].astype(int)
    if family == "DCSL":
        return np.asarray(model.selected_indices_, dtype=int)
    raise ValueError("Unexpected family")


def compare_sets(a: np.ndarray, b: np.ndarray) -> dict:
    aa, bb = set(map(int, a)), set(map(int, b))
    if len(aa) != len(a) or len(bb) != len(b) or not aa or not bb:
        raise ValueError("Missing or duplicated selected gene indices")
    shared = len(aa & bb)
    return {"n_first": len(aa), "n_second": len(bb), "shared_genes": shared,
            "jaccard": float(shared / len(aa | bb))}


def sample_scores(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-order-aware DTP minus Acute score (logit, not probability)."""
    classes = np.asarray(model.classes_, dtype=str).tolist()
    if set(classes) != {"Control", "Acute", "DTP"} or len(classes) != 3:
        raise ValueError("Unexpected model classes")
    raw = np.asarray(model.decision_function(x), dtype=np.float64)
    if raw.shape != (len(x), 3) or not np.isfinite(raw).all():
        raise ValueError("Expected finite multiclass logits")
    pred = np.asarray(model.predict(x), dtype=str)
    if not np.array_equal(pred, np.asarray(classes)[raw.argmax(axis=1)]):
        raise ValueError("Predictions contradict decision scores")
    margin = raw[:, classes.index("DTP")] - raw[:, classes.index("Acute")]
    return pred, margin


def verify_prior_predictions(prior_fold: dict, family: str, gsms: list[str], actual: list[str],
                             preds: np.ndarray, digest: str) -> None:
    ref = prior_fold["family_outer_results"][family]
    if digest != ref.get("selected_feature_indices_sha256"):
        raise ValueError(f"{family}: selected-feature hash differs from stage 32")
    expected = ref.get("predictions", [])
    present = [{"gsm": str(g), "actual": str(y), "predicted": str(p)}
               for g, y, p in zip(gsms, actual, preds)]
    if expected != present:
        raise ValueError(f"{family}: refitted predictions differ from stage 32")


def objective_decomposition(model: DCSLClassifier, x_train: np.ndarray,
                            y_train: np.ndarray, groups: np.ndarray) -> dict:
    """Recalculate three components of fitted training objective; no test data."""
    z = model.scaler_.transform(x_train[:, model.nonconstant_indices_])[:, model.selected_local_]
    logits = z @ model.coef_ + model.intercept_
    class_ix = np.asarray([model.classes_.tolist().index(y) for y in y_train], dtype=int)
    ce = float(-np.mean((logits - logsumexp(logits, axis=1, keepdims=True))[
        np.arange(len(y_train)), class_ix]))
    ridge = float(0.5 * model.l2 * np.sum(model.coef_**2))
    contrasts = cell_line_contrasts(z, y_train, groups)
    group_margins = contrasts @ (model.coef_[:, 2] - model.coef_[:, 1])
    variance = float(np.var(group_margins))
    penalty = float(model.gamma * variance)
    total = ce + ridge + penalty
    if not np.isclose(total, model.train_loss_, atol=1e-7, rtol=1e-7):
        raise ValueError("DCSL objective decomposition does not match its optimizer")
    return {"training_cross_entropy": ce, "training_l2_term": ridge,
            "training_margin_variance": variance, "gamma": model.gamma,
            "training_stability_term": penalty, "training_total_objective": total}


def compare_predictions(gsms: list[str], actual: list[str], predictions: dict,
                        margins: dict) -> tuple[list[dict], dict]:
    required = ("LogisticRegression", "DCSL", *MATCHED_ABLATIONS)
    if set(predictions) != set(required) or set(margins) != set(required):
        raise ValueError("Expected two frozen models and two fixed ablations")
    if len(gsms) != len(set(gsms)) or any(len(predictions[k]) != len(gsms) or len(margins[k]) != len(gsms)
                                         for k in required):
        raise ValueError("Sample score length or GSM integrity mismatch")
    records = []
    counts = {"both_correct": 0, "logistic_only_correct": 0,
              "dcsl_only_correct": 0, "both_wrong": 0, "prediction_disagreements": 0}
    for i, (gsm, truth) in enumerate(zip(gsms, actual)):
        lp, dp = str(predictions["LogisticRegression"][i]), str(predictions["DCSL"][i])
        a, b = lp == truth, dp == truth
        key = "both_correct" if a and b else ("logistic_only_correct" if a else
              "dcsl_only_correct" if b else "both_wrong")
        counts[key] += 1
        counts["prediction_disagreements"] += int(lp != dp)
        records.append({"gsm": str(gsm), "actual": truth,
                        "predicted": {k: str(predictions[k][i]) for k in required},
                        "DTP_minus_Acute_logit": {k: float(margins[k][i]) for k in required},
                        "logistic_correct": bool(a), "dcsl_correct": bool(b)})
    return records, counts


def evaluate_fold(x: np.ndarray, meta, protocol_fold: dict, prior_fold: dict,
                  by_id: dict, stage32) -> dict:
    index = {str(g): i for i, g in enumerate(meta.gsm)}
    tr = stage32.indices_for(protocol_fold["outer_train_gsm"], index)
    te = stage32.indices_for(protocol_fold["outer_test_gsm"], index)
    if set(tr).intersection(te) or len(tr) != 27 or len(te) != 9:
        raise ValueError("Training/test leakage or unexpected fold membership")
    y = meta.state.to_numpy(dtype=str)
    groups = meta.cell_line.to_numpy(dtype=str)
    gsms = meta.iloc[te].gsm.astype(str).tolist()
    truth = y[te].tolist()
    pred, margins, index_sets, chosen = {}, {}, {}, {}
    originals = {}
    for family in ("LogisticRegression", "DCSL"):
        name = prior_fold["selected_by_family_inner_only"][family]
        if name not in by_id or by_id[name]["family"] != family:
            raise ValueError("Selected candidate not found in locked configurations")
        model = stage32.fit_model(stage32.make_model(by_id[name]), family,
                                  x[tr], y[tr], groups[tr])
        pp, mm = sample_scores(model, x[te])
        digest = stage32.selected_sha256(model, family)
        verify_prior_predictions(prior_fold, family, gsms, truth, pp, digest)
        pred[family], margins[family] = pp, mm
        index_sets[family] = feature_indices(model, family)
        chosen[family] = {"candidate_id": name, "selected_feature_indices_sha256": digest}
        originals[family] = model
    # Fixed matched-objective contrasts, *not* a grid search or replacement for nested LOCO.
    for name, selector in (("matched_anova_gamma0", "anova"),
                           ("matched_stable_gamma0", "stable")):
        model = DCSLClassifier(selector=selector, n_features=500, l2=0.1,
                               gamma=0.0, tau=0.5).fit(x[tr], y[tr], groups[tr])
        pp, mm = sample_scores(model, x[te])
        pred[name], margins[name] = pp, mm
        index_sets[name] = feature_indices(model, "DCSL")
        chosen[name] = {"settings_fixed_for_diagnostic": {
            "selector": selector, "n_features": 500, "l2": 0.1, "gamma": 0.0, "tau": 0.5},
            "selected_feature_indices_sha256": stage32.selected_sha256(model, "DCSL")}
    # In sklearn, LogisticRegression(C) is CE_mean + ||W||²/(2*n_train*C).
    # Match its effective L2 *algebraically*, NOT by inspecting test outcomes.
    logistic_c = float(by_id[chosen["LogisticRegression"]["candidate_id"]]["parameters"]["C"])
    matching_l2 = 1.0 / (len(tr) * logistic_c)
    matched_models = {}
    for name, selector in (("matched_anova_logistic_l2", "anova"),
                           ("matched_stable_logistic_l2", "stable")):
        model = DCSLClassifier(selector=selector, n_features=500, l2=matching_l2,
                               gamma=0.0, tau=0.5).fit(x[tr], y[tr], groups[tr])
        pp, mm = sample_scores(model, x[te])
        matched_models[name] = model
        pred[name], margins[name] = pp, mm
        index_sets[name] = feature_indices(model, "DCSL")
        chosen[name] = {"settings_fixed_for_diagnostic": {
            "selector": selector, "n_features": 500, "l2": matching_l2,
            "l2_derivation": "1 / (n_outer_training_samples * frozen_logistic_C)",
            "gamma": 0.0, "tau": 0.5},
            "selected_feature_indices_sha256": stage32.selected_sha256(model, "DCSL")}
    if not np.array_equal(index_sets["matched_anova_logistic_l2"], index_sets["LogisticRegression"]):
        raise ValueError("ANOVA gene selection does not match frozen Logistic")
    if not np.array_equal(index_sets["matched_stable_logistic_l2"], index_sets["DCSL"]):
        raise ValueError("L2 change unexpectedly altered Stable feature selection")
    # Check that the reimplementation of the frozen Logistic objective is equivalent.
    if not np.array_equal(pred["matched_anova_logistic_l2"], pred["LogisticRegression"]):
        raise ValueError("Matched ANOVA/L2 predictions differ from frozen Logistic")
    logit_logistic = np.asarray(originals["LogisticRegression"].decision_function(x[te]))
    # Models return logits in different class order and are uncalibrated.
    # Center scores to remove per-sample softmax gauge before numerical comparison.
    class_order = originals["LogisticRegression"].classes_.tolist()
    matched_model = matched_models["matched_anova_logistic_l2"]
    converted = np.asarray(matched_model.decision_function(x[te]))[:,
                           [matched_model.classes_.tolist().index(c) for c in class_order]]
    centered_delta = (logit_logistic - logit_logistic.mean(axis=1, keepdims=True)) - \
                     (converted - converted.mean(axis=1, keepdims=True))
    max_centered_logit_diff = float(np.max(np.abs(centered_delta)))
    if max_centered_logit_diff > 0.02:
        raise ValueError("Algebraically matched Logistic objective produced divergent logits")
    if not np.array_equal(index_sets["matched_stable_gamma0"], index_sets["DCSL"]):
        raise ValueError("Gamma ablation changed feature selection (should be training-only and identical)")
    if np.array_equal(index_sets["matched_anova_gamma0"], index_sets["matched_stable_gamma0"]):
        raise ValueError("ANOVA and Stable selections are unexpectedly identical")
    # When gamma=0 is the frozen DCSL choice, matched-CE must exactly reproduce it.
    if by_id[chosen["DCSL"]["candidate_id"]]["parameters"]["gamma"] == 0:
        if not np.array_equal(pred["DCSL"], pred["matched_stable_gamma0"]) or not np.allclose(
                margins["DCSL"], margins["matched_stable_gamma0"], atol=1e-8):
            raise ValueError("Gamma-zero matched DCSL refit changed output")
    records, error_counts = compare_predictions(gsms, truth, pred, margins)
    objectives = {"frozen_dcsl": objective_decomposition(originals["DCSL"],
                                                          x[tr], y[tr], groups[tr])}
    selector_overlap = compare_sets(index_sets["matched_anova_gamma0"],
                                    index_sets["matched_stable_gamma0"])
    logistic_stable_overlap = compare_sets(index_sets["LogisticRegression"],
                                           index_sets["DCSL"])
    return {"outer_test_cell_line": protocol_fold["outer_test_cell_line"],
            "train_n": len(tr), "heldout_n": len(te),
            "frozen_configurations": chosen,
            "feature_overlap": {"logistic_ANOVA_vs_DCSL_Stable": logistic_stable_overlap,
                                "matched_ANOVA_vs_Stable": selector_overlap},
            "dcsl_training_objective": objectives,
            "per_model_heldout_DTP_recall_posthoc": {
                name: float(np.mean(pred[name][y[te] == "DTP"] == "DTP")) for name in pred},
            "algebraic_regularization_comparison": {
                "frozen_logistic_C": logistic_c, "outer_train_n": len(tr),
                "matched_mean_CE_l2": matching_l2,
                "original_dcsl_l2": float(by_id[chosen["DCSL"]["candidate_id"]]["parameters"]["l2"]),
                "max_centered_logit_difference_logistic_vs_matched_ANOVA_l2": max_centered_logit_diff,
                "matched_ANOVA_l2_predictions_equal_frozen_logistic": True,
            },
            "paired_error_counts": error_counts,
            "samples_posthoc": records}


def draw_pc9(report: dict, path: Path) -> None:
    pc9 = next(f for f in report["outer_folds"] if f["outer_test_cell_line"] == "PC9")
    samples = [s for s in pc9["samples_posthoc"] if s["actual"] == "DTP"]
    if len(samples) != 3:
        raise ValueError("Expected three PC9 DTP replicates")
    names = ("LogisticRegression", "matched_anova_gamma0", "matched_stable_gamma0",
             "matched_anova_logistic_l2", "matched_stable_logistic_l2", "DCSL")
    names_short = ("Logistic", "ANOVA / L2=0.1", "Stable / L2=0.1",
                   "ANOVA / matched L2", "Stable / matched L2", "Nested DCSL")
    x = np.arange(len(samples), dtype=float)
    width = 0.125
    fig, ax = plt.subplots(figsize=(11.8, 5.1))
    for j, (name, label) in enumerate(zip(names, names_short)):
        vals = [s["DTP_minus_Acute_logit"][name] for s in samples]
        ax.bar(x + (j-(len(names)-1)/2) * width, vals, width=width, label=label)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(x, [s["gsm"] for s in samples])
    ax.set_xlabel("Held-out PC9 DTP GSM (post-hoc diagnostic)")
    ax.set_ylabel("DTP − Acute logit score (NOT a probability)")
    ax.set_title("Stage 33: PC9 score margins — descriptive, not for model tuning")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(root: Path) -> dict:
    stage32 = import_stage32(root)
    stage31 = json.loads((root / STAGE31_JSON).read_text(encoding="utf-8"))
    prior = json.loads((root / STAGE32_JSON).read_text(encoding="utf-8"))
    assert_stage32_unchanged(prior, stage31, stage32, root)
    x, meta = stage32.load_source(root, stage31)
    specifications = stage31["candidate_configurations_for_future_implementation"]
    by_id = {s["id"]: s for s in specifications}
    folds = []
    for proto, old in zip(stage31["nested_loco_protocol"]["folds"], prior["outer_folds"]):
        if proto["outer_test_cell_line"] != old["outer_test_cell_line"]:
            raise ValueError("Stage-31/32 outer fold ordering differs")
        out = evaluate_fold(x, meta, proto, old, by_id, stage32)
        folds.append(out)
        print(f"{out['outer_test_cell_line']}: feature overlap={out['feature_overlap']['logistic_ANOVA_vs_DCSL_Stable']['shared_genes']}/500; "
              f"frozen DTP recalls Logistic={out['per_model_heldout_DTP_recall_posthoc']['LogisticRegression']:.3f} "
              f"DCSL={out['per_model_heldout_DTP_recall_posthoc']['DCSL']:.3f}")
    report = {
        "analysis": "stage33_posthoc_dcsl_vs_logistic_training_only_replay_and_ablations",
        "source": "GSE193258", "n_samples": 36,
        "stage31_sha256": stage32.sha256(root / STAGE31_JSON),
        "stage32_sha256": stage32.sha256(root / STAGE32_JSON),
        "input_sha256": prior["input_sha256"],
        "stage32_frozen_predictions_and_feature_hashes_reproduced": True,
        "candidate_selection_reused_without_changes": True,
        "ablation_fits_on_outer_training_only": True,
        "heldout_labels_used_to_train_select_or_tune": False,
        "target_expression_read": False,
        "independent_validation": False,
        "source_labels_provisional": True,
        "historically_seen_PC9": True,
        "ablation_status": "POSTHOC_DIAGNOSTICS_NOT_NESTED_CANDIDATES_OR_METHOD_IMPROVEMENT",
        "outer_folds": folds,
        "limitations": [
            "Four source cell lines, including PC9, have already influenced prior investigations.",
            "Post-hoc selector-by-regularization factorial ablations are NOT validated models or a new candidate search.",
            "Sklearn Logistic regularization equivalence uses lambda = 1/(n_train*C); this is not a model chosen by held-out results.",
            "Selected 18,013-gene universe was historically intersected with the target study.",
            "Raw margins are different models' uncalibrated logits and cannot be compared as probabilities.",
            "All source GEO state labels remain provisional pending independent biological verification.",
            "Correlated training folds and 3 DTP replicates per line preclude causal attribution.",
            "No target-study expression/outcomes were accessed, and no independent validation is claimed.",
        ],
    }
    target = root / OUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    draw_pc9(report, root / FIG)
    pc9 = next(f for f in folds if f["outer_test_cell_line"] == "PC9")
    print("PC9 post-hoc fixed ablation DTP recalls:", pc9["per_model_heldout_DTP_recall_posthoc"])
    print("Saved report:", target)
    print("Saved figure:", root / FIG)
    print("Exploratory diagnostics only. Do not tune to PC9, claim causality or independent validation.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
