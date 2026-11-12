from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import json
import warnings
from itertools import combinations
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path("data/processed/cross_study")
BASELINE = Path("results/baseline_gse193258_loco.json")
OUTPUT = Path("results/gse193258_full_cell_cycle_sets.json")
GMT_URL = (
    "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/"
    "h.all.v2025.1.Hs.symbols.gmt"
)
MIRROR_COMMIT = "5a12c046a905961323d6ddc49481dca65b18eea0"
MIRROR_BLOB_SHA = "8be0803e09bd411f8c0956294288ee4dd08d0b79"
MIRROR_PATH = "inst/extdata/h.all.v2025.1.Hs.symbols.gmt.gz"
MIRROR_RAW = f"https://raw.githubusercontent.com/pnnl/hpgsea/{MIRROR_COMMIT}/{MIRROR_PATH}"
MIRROR_API = (
    f"https://api.github.com/repos/pnnl/hpgsea/contents/{MIRROR_PATH}"
    f"?ref={MIRROR_COMMIT}"
)
CACHE = Path("data/raw/msigdb/h.all.v2025.1.Hs.symbols.gmt")
SETS = ("HALLMARK_G2M_CHECKPOINT", "HALLMARK_E2F_TARGETS")
EXPECTED_GENE_HASHES = {
    "HALLMARK_G2M_CHECKPOINT": "3f510640faed8d39a63cc77999f6c4334d42ad2dce07313215a647389b779260",
    "HALLMARK_E2F_TARGETS": "564b78438ff87fdfb0fd79653cc41163db2dab0829de7926f93204f7d8518967",
}
STATES = ("Control", "Acute", "DTP")
N_FEATURES = 500
N_CONTROLS = 5
SEED = 42


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(root: Path):
    matrix_file = root / DATA_DIR / "gse193258_common_log2tpm.tsv.gz"
    metadata_file = root / DATA_DIR / "samples.csv"
    previous = json.loads((root / BASELINE).read_text(encoding="utf-8"))
    for path in (matrix_file, metadata_file):
        if previous["input_sha256"].get(path.name) != checksum(path):
            raise ValueError(f"Input checksum differs from step 09: {path.name}")

    matrix = pd.read_csv(matrix_file, sep="\t", compression="gzip")
    if "gene_id" not in matrix:
        raise ValueError("Expression table has no gene_id column")
    genes = matrix.pop("gene_id").astype(str).str.strip().to_numpy()
    if not all(genes) or len(set(genes)) != len(genes):
        raise ValueError("Blank or duplicated gene IDs")
    if matrix.columns.duplicated().any():
        raise ValueError("Duplicated sample IDs")

    meta = pd.read_csv(metadata_file, dtype=str, keep_default_na=False)
    meta = meta.loc[meta["accession"] == "GSE193258"].copy()
    if len(meta) != 36 or meta["gsm"].nunique() != 36:
        raise ValueError("Expected 36 unique source GSM samples")
    if set(matrix.columns) != set(meta["gsm"]) or len(matrix.columns) != 36:
        raise ValueError("Sample names do not match expression columns")
    meta = meta.set_index("gsm").loc[list(matrix.columns)].reset_index()
    groups = meta.groupby(["cell_line", "state"]).size()
    if len(groups) != 12 or not groups.eq(3).all() or set(meta["state"]) != set(STATES):
        raise ValueError("Unexpected cell line and state groups")

    x = matrix.to_numpy(dtype=float).T
    if not np.isfinite(x).all():
        raise ValueError("Expression values contain NaN or infinity")
    return x, genes, meta, previous


def parse_gmt(payload: bytes):
    if not payload or len(payload) > 2_000_000:
        raise ValueError("GMT is empty or unexpectedly large")
    content = payload.decode("utf-8-sig")
    selected = {}
    names = set()
    for line in content.splitlines():
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3:
            raise ValueError("Invalid GMT row")
        name = fields[0]
        if name in names:
            raise ValueError(f"Duplicate gene set: {name}")
        names.add(name)
        if name in SETS:
            members = fields[2:]
            if len(members) != 200 or len(set(members)) != 200:
                raise ValueError(f"Unexpected gene count or duplicates: {name}")
            actual = hashlib.sha256("\n".join(sorted(members)).encode("utf-8")).hexdigest()
            if actual != EXPECTED_GENE_HASHES[name]:
                raise ValueError(f"Wrong MSigDB release or altered genes in {name}")
            selected[name] = set(members)
    if set(selected) != set(SETS):
        raise ValueError("Both required G2M and E2F Hallmark gene sets must be present")
    if len(names) not in (2, 50):
        raise ValueError("Expected the official 50-set Hallmark GMT or a two-set extract")
    return selected


def fetch_bytes(url: str, timeout: int):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; OsimertinibResearch/0.1)",
        "Accept": "application/octet-stream, text/plain, application/json",
    }
    with urlopen(Request(url, headers=headers), timeout=timeout) as response:
        payload = response.read(200_001)
    if len(payload) > 200_000:
        raise ValueError("Downloaded payload exceeds the expected file size")
    return payload


def check_mirror_blob(blob: bytes):
    header = f"blob {len(blob)}\0".encode("ascii")
    git_sha = hashlib.sha1(header + blob).hexdigest()
    if git_sha != MIRROR_BLOB_SHA:
        raise ValueError("GitHub mirror does not match the pinned file revision")
    return gzip.decompress(blob)


def download_gmt():
    attempts = [
        ("official MSigDB", GMT_URL, "plain", 10),
        ("pinned GitHub mirror", MIRROR_RAW, "gzip", 12),
        ("GitHub contents API", MIRROR_API, "api", 12),
    ]
    errors = []
    for label, url, kind, timeout in attempts:
        try:
            print(f"Trying {label}...")
            raw = fetch_bytes(url, timeout)
            if kind == "plain":
                payload = raw
            elif kind == "gzip":
                payload = check_mirror_blob(raw)
            else:
                data = json.loads(raw.decode("utf-8"))
                if data.get("encoding") != "base64":
                    raise ValueError("GitHub API did not return base64 content")
                payload = check_mirror_blob(base64.b64decode(data["content"], validate=False))
            selected = parse_gmt(payload)
            print(f"Gene sets verified from {label}.")
            return payload, selected, url
        except (OSError, TimeoutError, ValueError, UnicodeError, binascii.Error) as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Automatic download failed from the official server and both GitHub mirror routes. "
        "Check internet or proxy access, then retry; an existing GMT can also be passed with --gmt.\n"
        + "\n".join(errors)
    )


def read_gmt(gmt_path: Path | None, root: Path, refresh: bool = False):
    cache_path = root / CACHE
    if gmt_path is not None:
        local = gmt_path.resolve()
        if not local.is_file():
            raise FileNotFoundError(f"GMT path does not exist: {local}")
        payload = gzip.decompress(local.read_bytes()) if local.suffix == ".gz" else local.read_bytes()
        selected = parse_gmt(payload)
        url = str(local)
        retrieved_from = "user_supplied_file"
    elif cache_path.is_file() and not refresh:
        payload = cache_path.read_bytes()
        selected = parse_gmt(payload)
        url = str(cache_path)
        retrieved_from = "verified_local_cache"
        print(f"Using cached MSigDB Hallmark gene sets: {cache_path}")
    else:
        payload, selected, url = download_gmt()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".gmt.part")
        try:
            temporary.write_bytes(payload)
            temporary.replace(cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        retrieved_from = "auto_download"
        print(f"Saved verified reference to: {cache_path}")
    return selected, {
        "msigdb_release": "2025.1.Hs",
        "official_reference_url": GMT_URL,
        "retrieval_type": retrieved_from,
        "retrieval_url_or_path": url,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_set_sha256": EXPECTED_GENE_HASHES,
    }


def train_predict(x_train, y_train, x_test, genes):
    model = Pipeline([
        ("filter", VarianceThreshold(threshold=1e-8)),
        ("select", SelectKBest(score_func=f_classif, k=N_FEATURES)),
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(C=0.1, solver="lbfgs", max_iter=3000)),
    ])
    if x_train.shape[1] <= N_FEATURES:
        raise ValueError("Not enough genes for the 500-feature baseline")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        model.fit(x_train, y_train)
    pred = model.predict(x_test)
    remaining = genes[model.named_steps["filter"].get_support()]
    chosen = set(remaining[model.named_steps["select"].get_support()].tolist())
    return pred, chosen


def metrics(actual, predicted):
    return {
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=STATES, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=STATES).tolist(),
    }


def matched_random_indices(train_matrix, removal, excluded, random_seed):
    variation = np.log1p(train_matrix.var(axis=0))
    rng = np.random.default_rng(random_seed)
    available = np.ones(train_matrix.shape[1], dtype=bool)
    available[excluded] = False
    if available.sum() < len(removal):
        raise ValueError("Not enough non-cell-cycle genes for matched controls")
    picks = []
    for marker in rng.permutation(removal):
        candidates = np.flatnonzero(available)
        distance = np.abs(variation[candidates] - variation[marker])
        pool_size = min(20, len(candidates))
        near = candidates[np.argpartition(distance, pool_size - 1)[:pool_size]]
        chosen = int(rng.choice(near))
        picks.append(chosen)
        available[chosen] = False
    return np.array(picks, dtype=int)


def stability(selected_sets):
    comparisons = [len(left & right) / len(left | right)
                   for left, right in combinations(selected_sets, 2)]
    return float(np.mean(comparisons))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--gmt", type=Path, default=None)
    parser.add_argument("--refresh-gmt", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    x, genes, meta, previous = load_data(root)
    hallmark, source = read_gmt(args.gmt, root, refresh=args.refresh_gmt)
    available = set(genes)
    requested = {name: hallmark[name] & available for name in SETS}
    requested["G2M_E2F_UNION"] = requested[SETS[0]] | requested[SETS[1]]
    for label, members in requested.items():
        if len(members) < 75:
            raise ValueError(f"Too few aligned genes for {label}: {len(members)}")

    y = meta["state"].to_numpy()
    cell_lines = meta["cell_line"].to_numpy()
    combined_exclusion = np.flatnonzero(np.isin(genes, list(requested["G2M_E2F_UNION"])))
    previous_folds = {
        fold["held_out_cell_line"]: fold["predictions"]
        for fold in previous["models"]["LogisticRegression"]["loco_folds"]
    }
    result = {
        "analysis": "exploratory_full_hallmark_cell_cycle_ablation",
        "dataset": "GSE193258",
        "samples": len(meta),
        "genes": len(genes),
        "gmt": source,
        "gene_sets": {
            name: {
                "original_count": len(hallmark[name]),
                "available_count": len(requested[name]),
                "available_genes": sorted(requested[name]),
            } for name in SETS
        },
        "union_size": len(requested["G2M_E2F_UNION"]),
        "overlap_size": len(requested[SETS[0]] & requested[SETS[1]]),
        "evaluation": "Leave-one-cell-line-out; source dataset only; fixed 500-gene logistic regression",
        "control_iterations": N_CONTROLS,
        "baseline_reproduced": False,
        "comparisons": {},
    }

    group_names = (*SETS, "G2M_E2F_UNION")
    per_group = {
        group: {"removed_predictions": [], "random_predictions": [[] for _ in range(N_CONTROLS)],
                "folds": [], "selected_sets": []}
        for group in group_names
    }
    all_true, baseline_pred, baseline_features = [], [], []
    for fold_id, line in enumerate(sorted(set(cell_lines))):
        training = cell_lines != line
        testing = ~training
        truth = y[testing]
        base_pred, base_features = train_predict(x[training], y[training], x[testing], genes)
        record = [
            {"gsm": str(gsm), "true_state": str(t), "predicted_state": str(p)}
            for gsm, t, p in zip(meta.loc[testing, "gsm"], truth, base_pred)
        ]
        if record != previous_folds[line]:
            raise ValueError(f"Baseline predictions differ from step 09 for {line}")
        all_true.extend(truth.tolist())
        baseline_pred.extend(base_pred.tolist())
        baseline_features.append(base_features)

        for group_index, group in enumerate(group_names):
            members = requested[group]
            markers = np.flatnonzero(np.isin(genes, list(members)))
            keep = np.ones(len(genes), dtype=bool)
            keep[markers] = False
            pred, selected = train_predict(x[training][:, keep], y[training],
                                           x[testing][:, keep], genes[keep])
            bucket = per_group[group]
            bucket["removed_predictions"].extend(pred.tolist())
            bucket["selected_sets"].append(selected)
            fold_report = {
                "held_out_cell_line": line,
                "baseline": metrics(truth, base_pred),
                "without_set": metrics(truth, pred),
                "selected_markers_in_baseline": sorted(base_features & members),
                "predictions": [
                    {"gsm": str(gsm), "true_state": str(actual),
                     "baseline": str(before), "without_set": str(after)}
                    for gsm, actual, before, after in zip(
                        meta.loc[testing, "gsm"], truth, base_pred, pred
                    )
                ],
            }
            bucket["folds"].append(fold_report)
            for repeat in range(N_CONTROLS):
                controls = matched_random_indices(
                    x[training], markers, combined_exclusion,
                    SEED + fold_id * 10000 + group_index * 1000 + repeat,
                )
                control_keep = np.ones(len(genes), dtype=bool)
                control_keep[controls] = False
                control_pred, _ = train_predict(
                    x[training][:, control_keep], y[training],
                    x[testing][:, control_keep], genes[control_keep],
                )
                bucket["random_predictions"][repeat].extend(control_pred.tolist())
        print(f"Completed held-out cell line: {line}")

    baseline_metrics = metrics(all_true, baseline_pred)
    result["baseline_reproduced"] = True
    result["baseline"] = baseline_metrics
    result["baseline_selected_gene_mean_jaccard"] = stability(baseline_features)
    for group in group_names:
        bucket = per_group[group]
        ablated_metrics = metrics(all_true, bucket["removed_predictions"])
        random_scores = [metrics(all_true, predictions)
                         for predictions in bucket["random_predictions"]]
        result["comparisons"][group] = {
            "removed_genes": len(requested[group]),
            "without_set": ablated_metrics,
            "delta_balanced_accuracy": (
                ablated_metrics["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]
            ),
            "matched_control_balanced_accuracies": [
                row["balanced_accuracy"] for row in random_scores
            ],
            "selected_gene_mean_jaccard": stability(bucket["selected_sets"]),
            "folds": bucket["folds"],
        }
        print(f"{group}: removed {len(requested[group])} genes; "
              f"LOCO BA={ablated_metrics['balanced_accuracy']:.3f}; "
              f"random controls={[round(v, 3) for v in result['comparisons'][group]['matched_control_balanced_accuracies']]}")

    result["limitations"] = [
        "Post-hoc sensitivity analysis on the same folds already inspected in steps 09-12.",
        "Feature ablation tests model dependence, not causality or complete removal of cell-cycle information.",
        "Selected Hallmark G2M and E2F sets overlap; their effects cannot be interpreted independently.",
        "Random controls are matched approximately by training-fold expression variance.",
        "The held-out study is not used; no untouched external validation follows from these scores.",
        "No inference about DTP specificity is possible without additional condition- and time-matched controls.",
        "Sample annotations and replicate independence still require manual verification.",
    ]
    destination = root / OUTPUT
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {destination}")
    print("Exploratory results only; no causal claim about cell-cycle confounding")


if __name__ == "__main__":
    main()
