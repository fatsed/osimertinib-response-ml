"""Stage 32: execute the *historically exploratory* stage-31 nested LOCO protocol.

Uses the frozen GSE193258 source matrix only.  Never reads the GSE255958
expression matrix. Candidate selection occurs within outer-training data;
all preprocessing/feature selection is fitted from scratch in each inner fold.
The four source cell lines (including PC9) were inspected before stage 31,
so these results are NOT untouched independent or preregistered validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osimertinib_ml.contrast_stability import DCSLClassifier  # noqa: E402
from osimertinib_ml.group_robust_contrast import GroupRobustDCSLClassifier  # noqa: E402
from osimertinib_ml.vrex_domain_generalization import VRExDCSLClassifier  # noqa: E402

PROTOCOL = Path("results/dcsl_stage31_evaluation_protocol.json")
BASELINE = Path("results/baseline_gse193258_loco.json")
STAGE30 = Path("results/dcsl_stage30_geo_provenance.json")
SOURCE_FILE = Path("data/processed/cross_study/gse193258_common_log2tpm.tsv.gz")
META_FILE = Path("data/processed/cross_study/samples.csv")
OUT = Path("results/dcsl_stage32_nested_loco.json")
FIG = Path("figures/dcsl_stage32_nested_loco_dtp_recall.png")
STATES = ("Control", "Acute", "DTP")
LINES = ("H1975", "HCC2935", "HCC827", "PC9")
FAMILIES = ("Dummy", "LogisticRegression", "LinearSVM", "DCSL", "GroupRobust", "VREx")
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(protocol: dict, baseline: dict, root: Path) -> dict:
    """Fail closed on missing, edited or contradictory data/protocol provenance."""
    if protocol.get("analysis") != "stage31_prospective_style_protocol_for_historically_explored_data":
        raise ValueError("Unexpected stage-31 manifest")
    if protocol.get("implementation_status") != "SPLITS_AND_POLICY_ONLY_NOT_NESTED_MODEL_RESULTS":
        raise ValueError("Stage-31 status changed")
    if (protocol.get("source_study") != "GSE193258" or protocol.get("n_source") != 36
            or protocol.get("source_label_status") != "PENDING_MANUAL_AND_MATRIX_CHECK"
            or protocol.get("historic_results_were_seen_before_this_plan") is not True
            or protocol.get("code_does_not_read_target_expression") is not True):
        raise ValueError("Source cohort/independence status changed")
    if baseline.get("source") != "GSE193258" or baseline.get("target") != "GSE255958":
        raise ValueError("Stage-09 source or target IDs changed")
    expected = protocol.get("data_sha256", {})
    actual = {p.name: sha256(root / p) for p in (SOURCE_FILE, META_FILE)}
    actual["stage30_report_sha256"] = sha256(root / STAGE30)
    if actual != expected:
        raise ValueError(f"Input/stage-30 hashes differ from locked stage 31: {actual}")
    for p in (SOURCE_FILE, META_FILE):
        if actual[p.name] != baseline.get("input_sha256", {}).get(p.name):
            raise ValueError(f"Source file differs from baseline: {p.name}")
    settings = protocol.get("nested_loco_protocol", {})
    if (settings.get("fixed_class_order") != list(STATES) or settings.get("seed") != SEED
            or settings.get("selection", {}).get("primary") != "mean inner-fold balanced_accuracy (equally weighted by cell line)"
            or settings.get("selection", {}).get("tie_break_1") != "mean inner-fold DTP recall"
            or settings.get("selection", {}).get("tie_break_2") != "mean inner-fold macro-F1"
            or settings.get("selection", {}).get("tie_break_3") != "lexicographically smallest candidate ID"):
        raise ValueError("Stage-31 evaluation rule changed")
    return actual


def validate_candidate_specs(specs: list[dict]) -> None:
    if not isinstance(specs, list) or len(specs) != 11:
        raise ValueError("Expected exactly 11 stage-31 candidates")
    if len({s["id"] for s in specs}) != 11 or Counter(s["family"] for s in specs) != Counter(
        {"Dummy": 1, "LogisticRegression": 3, "LinearSVM": 3, "DCSL": 2, "GroupRobust": 1, "VREx": 1}
    ):
        raise ValueError("Candidate families/IDs changed")
    # Check the *exact* frozen candidates, preventing silent post-hoc grid expansion.
    expected = [
        {"id": "dummy_prior", "family": "Dummy", "feature_selection": "none", "parameters": {"strategy": "prior"}},
        *[{"id": f"logistic_anova_500_C{c}", "family": "LogisticRegression", "feature_selection": "ANOVA",
            "parameters": {"k": 500, "C": c, "solver": "lbfgs", "max_iter": 3000}}
           for c in (0.01, 0.1, 1.0)],
        *[{"id": f"svm_anova_500_C{c}", "family": "LinearSVM", "feature_selection": "ANOVA",
            "parameters": {"k": 500, "C": c, "max_iter": 20000}}
           for c in (0.001, 0.01, 0.1)],
        *[{"id": f"dcsl_stable_500_gamma{g}", "family": "DCSL", "feature_selection": "Stable",
            "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": g}}
           for g in (0.0, 0.05)],
        {"id": "group_robust_stable_fixed", "family": "GroupRobust", "feature_selection": "Stable",
         "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": 0.0, "eta": 0.2, "temperature": 0.5}},
        {"id": "vrex_stable_fixed", "family": "VREx", "feature_selection": "Stable",
         "parameters": {"k": 500, "l2": 0.1, "tau": 0.5, "gamma": 0.0, "beta": 0.2}},
    ]
    if specs != expected:
        raise ValueError("Candidate ID, ordering or hyperparameters changed since stage 31")


def load_source(root: Path, protocol: dict) -> tuple[np.ndarray, pd.DataFrame]:
    """Reads only source expression and sample metadata; no target matrix."""
    source = pd.read_csv(root / SOURCE_FILE, sep="\t", compression="gzip", keep_default_na=False)
    if source.columns[0] != "gene_id" or source.columns.duplicated().any() or len(source) != 18013:
        raise ValueError("Source matrix schema unexpected")
    gene_ids = source.pop("gene_id").astype(str).str.strip()
    if gene_ids.eq("").any() or gene_ids.duplicated().any():
        raise ValueError("Invalid source gene IDs")
    x = source.to_numpy(dtype=np.float64).T
    if x.shape != (36, 18013) or not np.isfinite(x).all():
        raise ValueError("Invalid source matrix")
    meta = pd.read_csv(root / META_FILE, dtype=str, keep_default_na=False)
    wanted = {"accession", "gsm", "cell_line", "state", "label_review", "evaluation_role"}
    if (not wanted.issubset(meta.columns) or meta["gsm"].duplicated().any()
            or len(meta) != 48 or not meta["label_review"].eq("PENDING_MANUAL_AND_MATRIX_CHECK").all()):
        raise ValueError("Missing, duplicated or unexpectedly certified metadata")
    meta = meta.loc[meta["accession"] == "GSE193258"].copy()
    if (len(meta) != 36 or not meta["evaluation_role"].eq("source_candidate").all()
            or set(meta["gsm"]) != set(source.columns)
            or Counter(zip(meta["cell_line"], meta["state"])) != Counter(
                {(line, state): 3 for line in LINES for state in STATES}
            )):
        raise ValueError("Source GSM/state/line membership unexpected")
    meta = meta.set_index("gsm").loc[list(source.columns)].rename_axis("gsm").reset_index()
    validate_folds(protocol["nested_loco_protocol"]["folds"], meta)
    return x, meta


def indices_for(gsms: list[str], index: dict[str, int]) -> np.ndarray:
    if len(gsms) != len(set(gsms)) or any(g not in index for g in gsms):
        raise ValueError("Missing or duplicate GSM in fold")
    return np.asarray([index[g] for g in gsms], dtype=int)


def validate_folds(folds: list[dict], meta: pd.DataFrame) -> None:
    """Independently recomputes all expected group splits from source labels."""
    if len(folds) != 4 or [f["outer_test_cell_line"] for f in folds] != list(LINES):
        raise ValueError("Outer line order/count changed")
    by_line = {line: set(meta.loc[meta.cell_line == line, "gsm"]) for line in LINES}
    universe = set(meta.gsm)
    outer_test_union = set()
    for fold in folds:
        held = fold["outer_test_cell_line"]
        train, test = set(fold["outer_train_gsm"]), set(fold["outer_test_gsm"])
        if (len(fold["outer_train_gsm"]) != 27 or len(fold["outer_test_gsm"]) != 9
                or train != universe - by_line[held] or test != by_line[held]
                or fold["n_outer_train"] != 27 or fold["n_outer_test"] != 9
                or fold["outer_training_cell_lines"] != [c for c in LINES if c != held]):
            raise ValueError(f"Outer fold malformed: {held}")
        outer_test_union.update(test)
        inner = fold["inner_folds"]
        if len(inner) != 3 or [f["validation_cell_line"] for f in inner] != [c for c in LINES if c != held]:
            raise ValueError(f"Inner fold count/order malformed: {held}")
        for f in inner:
            val, tr = set(f["validation_gsm"]), set(f["train_gsm"])
            if (len(f["validation_gsm"]) != 9 or len(f["train_gsm"]) != 18
                    or val != by_line[f["validation_cell_line"]] or tr != train - val
                    or f["n_train"] != 18 or f["n_validation"] != 9
                    or f["training_cell_lines"] != [c for c in LINES if c not in (held, f["validation_cell_line"]) ]):
                raise ValueError(f"Inner split malformed: {held}/{f['validation_cell_line']}")
    if outer_test_union != universe:
        raise ValueError("Outer tests fail to cover source once")


def make_model(spec: dict):
    family, p = spec["family"], spec["parameters"]
    if family == "Dummy":
        return DummyClassifier(strategy=p["strategy"])
    if family in ("LogisticRegression", "LinearSVM"):
        clf = (LogisticRegression(C=p["C"], solver=p["solver"], max_iter=p["max_iter"])
               if family == "LogisticRegression" else
               LinearSVC(C=p["C"], max_iter=p["max_iter"], dual=True, random_state=SEED))
        return Pipeline([
            ("filter", VarianceThreshold(threshold=1e-8)),
            ("select", SelectKBest(f_classif, k=p["k"])),
            ("scale", StandardScaler()),
            ("classifier", clf),
        ])
    common = dict(selector="stable", n_features=p["k"], l2=p["l2"], tau=p["tau"], gamma=p["gamma"])
    if family == "DCSL":
        return DCSLClassifier(**common)
    if family == "GroupRobust":
        return GroupRobustDCSLClassifier(**common, eta=p["eta"], temperature=p["temperature"])
    if family == "VREx":
        return VRExDCSLClassifier(**common, beta=p["beta"])
    raise ValueError(f"Unsupported candidate family: {family}")


def fit_model(model, family: str, x: np.ndarray, y: np.ndarray, groups: np.ndarray):
    if family in ("DCSL", "GroupRobust", "VREx"):
        return model.fit(x, y, groups)
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        return model.fit(x, y)


def selected_sha256(model, family: str) -> str | None:
    if family == "Dummy":
        return None
    if family in ("DCSL", "GroupRobust", "VREx"):
        indices = np.asarray(model.selected_indices_, dtype="<i8")
    else:
        nonconstant = np.flatnonzero(model.named_steps["filter"].get_support())
        indices = np.asarray(nonconstant[model.named_steps["select"].get_support()], dtype="<i8")
    return hashlib.sha256(indices.tobytes()).hexdigest()


def scores(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y, dtype=str), np.asarray(pred, dtype=str)
    if len(y) != len(pred) or not len(y) or set(y) != set(STATES) or not set(pred).issubset(STATES):
        raise ValueError("Score requires all three states and valid predictions")
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=STATES, average="macro", zero_division=0)),
        "DTP_recall": float(recall_score(y, pred, labels=["DTP"], average="macro", zero_division=0)),
        "per_class_recall": {c: float(recall_score(y, pred, labels=[c], average="macro", zero_division=0)) for c in STATES},
        "confusion_matrix": confusion_matrix(y, pred, labels=STATES).tolist(),
        "n_by_class": {c: int(np.sum(y == c)) for c in STATES},
        "n": int(len(y)),
    }


def select_candidate(candidate_rows: list[dict]) -> dict:
    """Selection uses ONLY aggregate inner-validation scores, never outer metrics."""
    if not candidate_rows or len({r["id"] for r in candidate_rows}) != len(candidate_rows):
        raise ValueError("Candidate selection has no candidates or duplicate IDs")
    for row in candidate_rows:
        if len(row["inner_folds"]) != 3:
            raise ValueError("Each candidate must have three inner-validation folds")
    return min(candidate_rows, key=lambda r: (-r["mean_inner_balanced_accuracy"],
                                             -r["mean_inner_DTP_recall"],
                                             -r["mean_inner_macro_f1"], r["id"]))


def evaluate_outer_fold(x: np.ndarray, meta: pd.DataFrame, fold: dict, specs: list[dict], *,
                        make_model_fn=make_model) -> dict:
    index = {gsm: i for i, gsm in enumerate(meta.gsm)}
    y = meta.state.to_numpy()
    groups = meta.cell_line.to_numpy()
    tr = indices_for(fold["outer_train_gsm"], index)
    te = indices_for(fold["outer_test_gsm"], index)
    if set(tr) & set(te):
        raise ValueError("Outer train/test intersection")
    by_id = {s["id"]: s for s in specs}
    candidates = []
    for spec in specs:
        inner_scores = []
        for inner in fold["inner_folds"]:
            a = indices_for(inner["train_gsm"], index)
            b = indices_for(inner["validation_gsm"], index)
            if set(a) & set(b) or not set(a).issubset(tr) or not set(b).issubset(tr):
                raise ValueError("Inner indices overlap or include outer test")
            model = fit_model(make_model_fn(spec), spec["family"], x[a], y[a], groups[a])
            val_pred = model.predict(x[b])
            inner_scores.append({
                "validation_cell_line": inner["validation_cell_line"],
                "n_training": len(a), "n_validation": len(b),
                "metrics": scores(y[b], val_pred),
                "selected_feature_indices_sha256": selected_sha256(model, spec["family"]) if make_model_fn is make_model else None,
            })
        candidates.append({
            "id": spec["id"], "family": spec["family"],
            "mean_inner_balanced_accuracy": float(np.mean([r["metrics"]["balanced_accuracy"] for r in inner_scores])),
            "mean_inner_DTP_recall": float(np.mean([r["metrics"]["DTP_recall"] for r in inner_scores])),
            "mean_inner_macro_f1": float(np.mean([r["metrics"]["macro_f1"] for r in inner_scores])),
            "inner_folds": inner_scores,
        })
    winners = {family: select_candidate([row for row in candidates if row["family"] == family])["id"]
               for family in FAMILIES if any(row["family"] == family for row in candidates)}
    global_winner = select_candidate(candidates)["id"]
    family_results = {}
    for family, chosen_id in winners.items():
        spec = by_id[chosen_id]
        model = fit_model(make_model_fn(spec), family, x[tr], y[tr], groups[tr])
        predicted = np.asarray(model.predict(x[te]), dtype=str)
        family_results[family] = {
            "selected_candidate_id": chosen_id,
            "metrics": scores(y[te], predicted),
            "selected_feature_indices_sha256": selected_sha256(model, family) if make_model_fn is make_model else None,
            "predictions": [
                {"gsm": str(g), "actual": str(t), "predicted": str(p)}
                for g, t, p in zip(meta.iloc[te].gsm, y[te], predicted)
            ],
        }
    return {
        "outer_test_cell_line": fold["outer_test_cell_line"],
        "n_outer_train": len(tr), "n_outer_test": len(te),
        "candidates_inner_only": candidates,
        "selected_by_family_inner_only": winners,
        "global_candidate_selected_inner_only": global_winner,
        "family_outer_results": family_results,
    }


def summarize_families(outer: list[dict]) -> dict:
    out = {}
    for family in FAMILIES:
        folds = [f for f in outer if family in f["family_outer_results"]]
        if len(folds) != 4:
            raise ValueError(f"Four outer tests required for {family}")
        actual = [p["actual"] for f in folds for p in f["family_outer_results"][family]["predictions"]]
        predicted = [p["predicted"] for f in folds for p in f["family_outer_results"][family]["predictions"]]
        per_line = {f["outer_test_cell_line"]: f["family_outer_results"][family]["metrics"] for f in folds}
        out[family] = {
            "per_line": per_line,
            "selected_config_by_held_out_line": {
                f["outer_test_cell_line"]: f["family_outer_results"][family]["selected_candidate_id"] for f in folds
            },
            "mean_outer_balanced_accuracy": float(np.mean([per_line[c]["balanced_accuracy"] for c in LINES])),
            "mean_outer_DTP_recall": float(np.mean([per_line[c]["DTP_recall"] for c in LINES])),
            "mean_outer_macro_f1": float(np.mean([per_line[c]["macro_f1"] for c in LINES])),
            "pooled": scores(np.asarray(actual), np.asarray(predicted)),
        }
    return out


def plot_heatmap(families: dict, path: Path):
    matrix = np.asarray([[families[family]["per_line"][c]["DTP_recall"] for c in LINES] for family in FAMILIES])
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    img = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(LINES)), LINES)
    ax.set_yticks(range(len(FAMILIES)), FAMILIES)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, f"{int(round(3 * matrix[r,c]))}/3", ha="center", va="center",
                    color="white" if matrix[r,c] > 0.65 else "black")
    ax.set_title("Stage 32: nested LOCO DTP recall (exploratory; previously seen groups)")
    fig.colorbar(img, ax=ax, label="DTP recall")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(root: Path) -> dict:
    protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    input_hashes = verify_manifest(protocol, baseline, root)
    specs = protocol["candidate_configurations_for_future_implementation"]
    validate_candidate_specs(specs)
    x, meta = load_source(root, protocol)
    outer = []
    for f in protocol["nested_loco_protocol"]["folds"]:
        result = evaluate_outer_fold(x, meta, f, specs)
        outer.append(result)
        print(f"Outer {f['outer_test_cell_line']}: global inner winner={result['global_candidate_selected_inner_only']}; "
              f"family choices={result['selected_by_family_inner_only']}")
    family_summary = summarize_families(outer)
    report = {
        "analysis": "stage32_nested_source_only_loco_exploratory",
        "source": "GSE193258", "n_samples": 36, "n_genes": x.shape[1],
        "protocol_sha256": sha256(protocol_path),
        "input_sha256": input_hashes,
        "scikit_learn_version": sklearn.__version__,
        "evaluation": "4 outer groups, 3 inner groups per outer; training-only selection by family and global winner",
        "candidates_from_stage31_unchanged": True,
        "held_out_or_target_labels_used_for_fitting_preprocessing_or_selection": False,
        "target_expression_read": False,
        "source_feature_universe_historically_built_with_target_genes": True,
        "source_cell_lines_have_previously_been_examined": True,
        "source_labels_still_provisional": True,
        "independent_validation": False,
        "method_novelty_claim": False,
        "outer_folds": outer,
        "families": family_summary,
        "limitations": [
            "All four source cell lines, including PC9, were inspected before stage 31; exploratory only.",
            "Feature universe (18,013 common genes) was historically aligned using another study.",
            "Only four groups and three DTP replicates per group; performance uncertainty is large.",
            "GEO titles agree with candidate labels, but independent biological validation remains pending.",
            "No GSE255958 expression or outcomes were used in this run. No independent target benchmark is claimed.",
            "The global inner winner is recorded as a choice only; per-family outer results are exploratory method comparison.",
        ],
    }
    (root / OUT).parent.mkdir(parents=True, exist_ok=True)
    (root / OUT).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    plot_heatmap(family_summary, root / FIG)
    for family in FAMILIES:
        r = family_summary[family]
        print(f"{family:19s} mean outer BA={r['mean_outer_balanced_accuracy']:.3f} "
              f"mean DTP recall={r['mean_outer_DTP_recall']:.3f} "
              f"PC9 DTP={r['per_line']['PC9']['DTP_recall']:.3f}")
    print("Saved report:", root / OUT)
    print("Saved figure:", root / FIG)
    print("Exploratory nested development evaluation. NOT independent, prospective, or clinical validation.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
