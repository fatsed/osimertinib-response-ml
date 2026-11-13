"""Exploratory Stage-39 geometry sensitivity audit (no model training).

Compares the centroid's displacement perpendicular to the held-out
DTP-minus-Acute axis, across training-only feature selection settings.
The existing Stage-39 report is reproduced at ANOVA/k=500 before producing
any new result. Washout values and held-out labels do NOT select features.

Run: python scripts/40_washout_geometry_sensitivity.py
This is descriptive, not an independent evaluation or a biomarker claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('data/processed/GSE193258/log2tpm_36samples.tsv.gz')
METADATA = Path('data/processed/GSE193258/samples_36.csv')
RAW = Path('data/raw/GSE193258/GSE193258_RNAseq_log2TPM_abundance.tsv.gz')
BINARY_REPORT = Path('results/dcsl_stage35_binary_dtp_acute_loco.json')
STAGE11_REPORT = Path('results/gse193258_washout_challenge.json')
STAGE31_REPORT = Path('results/dcsl_stage31_evaluation_protocol.json')
STAGE39_REPORT = Path('results/washout_contrast_audit.json')
OUTPUT = Path('results/washout_geometry_sensitivity.json')
COUNTS = (100, 250, 500, 1000)
METHODS = ('ANOVA', 'VarianceTopK')
LINES = ('H1975', 'HCC2935', 'HCC827', 'PC9')


def perpendicular_ratio(reference: dict, observation: dict) -> float:
    """Centroid-to-axis distance divided by held-out Acute--DTP gap."""
    length = float(reference['distance_to_heldout_Acute_centroid'])
    retention = float(observation['retention_along_heldout_DTP_Acute'])
    distance_dtp = float(observation['distance_to_heldout_DTP_centroid'])
    if not np.isfinite([length, retention, distance_dtp]).all() or length <= 1e-10:
        raise ValueError('Nonfinite or degenerate Acute--DTP contrast')
    radial_sq = distance_dtp ** 2 - ((retention - 1) * length) ** 2
    if radial_sq < -1e-7 * max(1, distance_dtp ** 2):
        raise ValueError('Inconsistent report geometry')
    return float(np.sqrt(max(radial_sq, 0.0)) / length)


def variance_select(x_train: np.ndarray, *, k: int):
    """Unsupervised training-only high-variance selector; stable tie-breaking."""
    variance = np.var(x_train, axis=0)
    eligible = np.flatnonzero(variance > 1e-8)
    if len(eligible) < k:
        raise ValueError('Too few eligible genes')
    order = np.lexsort((eligible, -variance[eligible]))[:k]
    indices = eligible[order]
    scaler = StandardScaler().fit(x_train[:, indices])
    return indices, scaler


def load_inputs(root: Path, s39):
    """Load and verify original data using existing Stage-11/34/35 guards."""
    stage11 = s39.load_script(root, '11_washout_challenge.py', 'stage11_for_geometry')
    stage34 = s39.load_script(root, '34_source_only_rebaseline.py', 'stage34_for_geometry')
    stage35 = s39.load_script(root, '35_dtp_vs_acute_benchmark.py', 'stage35_for_geometry')
    stage35_report = json.loads((root / BINARY_REPORT).read_text(encoding='utf-8'))
    stage11_report = json.loads((root / STAGE11_REPORT).read_text(encoding='utf-8'))
    protocol = json.loads((root / STAGE31_REPORT).read_text(encoding='utf-8'))
    previous = json.loads((root / STAGE39_REPORT).read_text(encoding='utf-8'))
    provenance = stage35_report.get('source_provenance', {})
    if stage35_report.get('n_binary_samples') != 24 or stage35_report.get('features_source_only') != 19712:
        raise ValueError('Stage 35 cohort/feature universe mismatch')
    if (provenance.get('prepared_matrix_sha256') != stage11.sha256(root / SOURCE) or
            provenance.get('prepared_metadata_sha256') != stage11.sha256(root / METADATA) or
            provenance.get('source_raw_sha256') != stage11.sha256(root / RAW) or
            stage11_report.get('washout_original_matrix_sha256') != stage11.sha256(root / RAW) or
            stage35_report.get('frozen_stage31_sha256') != stage11.sha256(root / STAGE31_REPORT)):
        raise ValueError('Stage 11/35 file integrity mismatch')
    if (previous.get('source_matrix_sha256') != provenance['prepared_matrix_sha256'] or
            previous.get('source_raw_sha256') != provenance['source_raw_sha256'] or
            previous.get('study') != 'GSE193258'):
        raise ValueError('Stage 39 provenance mismatch')
    meta = pd.read_csv(root / METADATA, dtype=str, keep_default_na=False)
    stage34.validate_cohort(meta, protocol)
    meta = meta.rename(columns={'candidate_group': 'state'})
    stage35.validate_binary_cohort(meta, protocol['nested_loco_protocol']['folds'])
    frame = pd.read_csv(root / SOURCE, sep='\t', compression='gzip')
    genes, x = stage34.validate_matrix(frame, meta.gsm.tolist())
    wash = stage11.load_washout_metadata(root)
    x_wash = stage11.load_washout_values(root, list(map(str, genes)), x, meta, wash)
    if set(wash.gsm) & set(meta.gsm):
        raise ValueError('Washout overlaps source samples')
    return meta, x, wash, x_wash, previous


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    s39 = __import__('importlib').util
    spec = s39.spec_from_file_location('stage39_for_sensitivity', root / 'scripts/39_washout_contrast_audit.py')
    if spec is None or spec.loader is None:
        raise FileNotFoundError('Stage 39 script not found')
    mod = s39.module_from_spec(spec)
    spec.loader.exec_module(mod)
    meta, x, wash, x_wash, original = load_inputs(root, mod)
    states, lines = meta.state.to_numpy(), meta.cell_line.to_numpy()
    records = []
    for line in LINES:
        tr = (lines != line) & np.isin(states, ['Acute', 'DTP'])
        held = lines == line
        hold_wash = wash.cell_line.eq(line).to_numpy()
        label = states[held]
        wash_label = wash.loc[hold_wash, 'challenge_group'].to_numpy()
        if (tr.sum() != 18 or held.sum() != 9 or hold_wash.sum() != 6 or
            sum(label == 'Acute') != 3 or sum(label == 'DTP') != 3 or
            sum(wash_label == 'ShortWashout') != 3 or sum(wash_label == 'LongWashout') != 3):
            raise ValueError(f'Bad sample allocation for {line}')
        for method in METHODS:
            for k in COUNTS:
                if method == 'ANOVA':
                    pipe, idx, ztrain = mod.fit_training_features(x[tr], states[tr], k=k)
                    zheld = pipe.transform(x[held])
                    zwash = pipe.transform(x_wash[hold_wash])
                else:
                    idx, scaler = variance_select(x[tr], k=k)
                    ztrain = scaler.transform(x[tr][:, idx])
                    zheld = scaler.transform(x[held][:, idx])
                    zwash = scaler.transform(x_wash[hold_wash][:, idx])
                direction = mod.normalized_direction(ztrain, states[tr], lines[tr])
                stats = mod.contrast_summary(
                    zheld[label == 'Acute'], zheld[label == 'DTP'],
                    zwash[wash_label == 'ShortWashout'],
                    zwash[wash_label == 'LongWashout'], direction,
                )
                ref = stats['on_drug_DTP_reference']
                short = stats['short_washout']
                long = stats['long_washout']
                short_orth = perpendicular_ratio(ref, short)
                long_orth = perpendicular_ratio(ref, long)
                if method == 'ANOVA' and k == 500:
                    baseline = original['per_cell_line'][line]
                    if not np.array_equal(np.asarray(idx), np.asarray(baseline['selected_feature_indices_zero_based'])):
                        raise ValueError(f'Stage39 feature selection not reproduced for {line}')
                    for key, value in [('short_washout', short), ('long_washout', long)]:
                        for metric in ('retention_along_heldout_DTP_Acute', 'distance_to_heldout_DTP_centroid'):
                            if not np.isclose(value[metric], baseline[key][metric], rtol=1e-8, atol=1e-8):
                                raise ValueError(f'Stage39 geometry mismatch: {line}/{key}/{metric}')
                records.append({
                    'cell_line': line, 'selector': method, 'k_features': k,
                    'short_orthogonal_ratio': short_orth,
                    'long_orthogonal_ratio': long_orth,
                    'long_minus_short_orthogonal_ratio': long_orth - short_orth,
                    'short_axis_retention': short['retention_along_heldout_DTP_Acute'],
                    'long_axis_retention': long['retention_along_heldout_DTP_Acute'],
                })
    report = {
        'analysis': 'exploratory_stage39_feature_selection_sensitivity',
        'source': 'GSE193258', 'n_variants': len(records),
        'feature_counts': list(COUNTS), 'feature_selectors': list(METHODS),
        'stage39_anova_500_reproduced': True,
        'external_validation': False, 'model_fitted': False,
        'heldout_and_washout_used_for_feature_selection': False,
        'metric': 'orthogonal distance of washout centroid from held-out Acute--DTP axis, normalized by held-out centroid gap',
        'records': records,
        'limitations': [
            'This is a descriptive sensitivity check, not model selection or prediction accuracy.',
            'Both the axis and feature space depend on the selection method and training fold.',
            'Held-out Acute/DTP labels define the reference axis; not an independent test.',
            'Only three unpaired replicates per condition; no ground-truth persistence labels.',
            'Long washout duration differs between cell lines; direct pooled time comparisons are invalid.',
            'Exposure duration is confounded with treatment state in the source.',
            'All groups have already been inspected; do not tune a method using these results.',
        ],
    }
    target = root / OUTPUT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print('Stage39 ANOVA/500 reproduced; sensitivity is DESCRIPTIVE only.')
    for method in METHODS:
        print(method)
        for k in COUNTS:
            for line in LINES:
                row = next(v for v in records if v['selector'] == method and v['k_features'] == k and v['cell_line'] == line)
                print(f'  {k:4} {line:8} short={row["short_orthogonal_ratio"]:.3f} long={row["long_orthogonal_ratio"]:.3f} delta={row["long_minus_short_orthogonal_ratio"]:+.3f}')
    print('Saved:', target)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
