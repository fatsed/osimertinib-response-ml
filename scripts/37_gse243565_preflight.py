
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("data/raw/GSE243565/GSE243565_normalized.counts.xls.gz")
SOURCE = Path("data/processed/GSE193258/log2tpm_36samples.tsv.gz")
ANNOTATION = Path("data/reference/gencode_v27/gencode.v27.annotation.gtf.gz")
REPORT = Path("results/stage37_gse243565_preflight.json")
TARGET_URL = ("https://ftp.ncbi.nlm.nih.gov/geo/series/GSE243nnn/"
              "GSE243565/suppl/GSE243565_normalized.counts.xls.gz")
ANNOTATION_URL = ("https://ftp.ebi.ac.uk/pub/databases/gencode/"
                  "Gencode_human/release_27/gencode.v27.annotation.gtf.gz")
COHORT = {
    "GSM7791042": {"description": "HCC827ASCL1_O1", "condition": "osimertinib_3d"},
    "GSM7791043": {"description": "HCC827ASCL1_O2", "condition": "osimertinib_3d"},
    "GSM7791044": {"description": "HCC827ASCL1_O3", "condition": "osimertinib_3d"},
    "GSM7791045": {"description": "HCC827ASCL1_P1", "condition": "persister_10d"},
    "GSM7791046": {"description": "HCC827ASCL1_P2", "condition": "persister_10d"},
    "GSM7791047": {"description": "HCC827ASCL1_P3", "condition": "persister_10d"},
}
ENSG = re.compile(r"^(ENSG\d+)(?:\.(\d+))?$")
GENE_ID_ATTR = re.compile(r'(?:^|;)\s*gene_id\s+"([^"]+)"')
GENE_NAME_ATTR = re.compile(r'(?:^|;)\s*gene_name\s+"([^"]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_gzip(url: str, path: Path) -> None:
    """Safely download a real gzip; never replace a valid file with HTML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Research-preflight/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as out:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        with temp.open("rb") as handle:
            if handle.read(2) != b"\x1f\x8b":
                raise ValueError("The server returned non-gzip data; no file was replaced")
        with gzip.open(temp, "rb") as handle:
            if not handle.read(1024):
                raise ValueError("Empty compressed file")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def read_matrix(path: Path) -> tuple[pd.DataFrame, str]:
    with gzip.open(path, "rb") as handle:
        data = handle.read()
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        try:
            return pd.read_excel(io.BytesIO(data), engine="xlrd"), "binary_xls"
        except ImportError as exc:
            raise RuntimeError("Binary XLS requires xlrd") from exc
    if data.startswith(b"PK\x03\x04"):
        try:
            return pd.read_excel(io.BytesIO(data), engine="openpyxl"), "binary_xlsx"
        except ImportError as exc:
            raise RuntimeError("Binary XLSX requires openpyxl") from exc
    if b"\x00" in data[:4096]:
        raise ValueError("Unrecognized binary matrix format")
    header = data[:16384].decode("utf-8-sig").splitlines()[0]
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    if sep not in header:
        raise ValueError("Cannot determine matrix delimiter")
    return pd.read_csv(io.BytesIO(data), sep=sep, low_memory=False), "delimited_text"


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def identify_columns(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    index = defaultdict(list)
    for name in columns:
        index[canonical(name)].append(name)
    selected = {}
    issues = []
    for gsm, metadata in COHORT.items():
        matches = sorted(set(index[canonical(gsm)] + index[canonical(metadata["description"]) ]))
        if len(matches) == 1:
            selected[gsm] = matches[0]
        else:
            issues.append(f"{gsm}: expected one matrix column, found {len(matches)}")
    if len(set(selected.values())) != len(selected):
        issues.append("Multiple GSMs matched the same matrix column")
    return selected, issues


def parse_gencode_genes(path: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]], int]:
    """Parse ONLY gene features, never transcript records, from GENCODE v27 GTF.

    Returns mapping for complete IDs, unversioned IDs, and symbol-to-base IDs.
    Multiple aliases are represented as sets; ambiguity is never silently resolved.
    """
    by_full = defaultdict(set)
    by_base = defaultdict(set)
    symbol_bases = defaultdict(set)
    n_genes = 0
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t", 8)
            if len(parts) != 9 or parts[2] != "gene":
                continue
            gene = GENE_ID_ATTR.search(parts[8])
            name = GENE_NAME_ATTR.search(parts[8])
            if not gene or not name:
                continue
            full_id, symbol = gene.group(1), name.group(1)
            match = ENSG.fullmatch(full_id)
            if not match:
                continue
            base_id = match.group(1)
            by_full[full_id].add(symbol)
            by_base[base_id].add(symbol)
            symbol_bases[symbol].add(base_id)
            n_genes += 1
    if n_genes < 25000 or len(by_base) < 25000:
        raise ValueError("Annotation does not resemble the full GENCODE v27 gene GTF")
    return dict(by_full), dict(by_base), dict(symbol_bases), n_genes


def audit_gene_mapping(target_ids: pd.Series, source_symbols: set[str],
                       by_full: dict[str, set[str]], by_base: dict[str, set[str]],
                       symbol_bases: dict[str, set[str]]) -> dict:
    """Version-sensitive, strictly one-to-one ID-to-symbol audit.

    Base-ID matches without an exact gene version are counted separately and
    are NOT silently accepted. No expression values or outcomes are consulted.
    """
    target = [str(x).strip() for x in target_ids]
    frequencies = Counter(target)
    strict = []
    fallback = []
    unmapped = 0
    ambiguous_annotation = 0
    duplicated_targets = 0
    examples = []
    for gene_id in target:
        match = ENSG.fullmatch(gene_id)
        if not match:
            unmapped += 1
            continue
        base = match.group(1)
        exact = by_full.get(gene_id, set())
        candidates = exact if exact else by_base.get(base, set())
        if not candidates:
            unmapped += 1
            continue
        if len(candidates) != 1:
            ambiguous_annotation += 1
            continue
        symbol = next(iter(candidates))
        if (frequencies[gene_id] != 1 or len(symbol_bases.get(symbol, set())) != 1):
            duplicated_targets += 1
            continue
        if symbol not in source_symbols:
            continue
        if exact:
            strict.append((gene_id, symbol))
        else:
            fallback.append((gene_id, symbol))
        if len(examples) < 5:
            examples.append({"target_ensembl_id": gene_id, "source_gene_symbol": symbol,
                             "version_exact": bool(exact)})
    strict_symbols = [symbol for _, symbol in strict]
    strict_unique = len(set(strict_symbols))
    return {
        "target_gene_rows": len(target),
        "target_gene_id_duplicates": sum(x - 1 for x in frequencies.values() if x > 1),
        "exact_version_unique_pairs_with_source": len(strict),
        "exact_version_unique_source_symbols": strict_unique,
        "base_id_only_candidate_pairs_not_approved": len(fallback),
        "unmapped_target_rows": unmapped,
        "ambiguous_gene_name_rows": ambiguous_annotation,
        "duplicate_or_non_one_to_one_rows": duplicated_targets,
        "mapping_examples_diagnostic": examples,
        "annotation_crosswalk_approved_for_predictions": False,
        "note": ("Exact-version pairs are diagnostics, not authorization to apply "
                 "the Stage 35 model; measurement units and a frozen preprocessing "
                 "protocol are unresolved. Base-ID fallbacks require manual review."),
    }


def inspect(root: Path, target_path: Path, annotation_path: Path) -> dict:
    source_path = root / SOURCE
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing Stage 05 source matrix: {source_path}")
    source = pd.read_csv(source_path, sep="\t", compression="gzip", usecols=["gene_id"], dtype=str)
    source_symbols = source["gene_id"].dropna().astype(str).str.strip()
    if len(source_symbols) != 19712 or source_symbols.eq("").any() or source_symbols.duplicated().any():
        raise ValueError("Unexpected source gene symbols; source data may have changed")
    matrix, fmt = read_matrix(target_path)
    matrix.columns = [str(c).strip() for c in matrix.columns]
    if matrix.empty or len(matrix.columns) < 7:
        raise ValueError("Target matrix is empty or incomplete")
    selected, issues = identify_columns(matrix.columns.tolist())
    gene_col = "GeneVer" if "GeneVer" in matrix.columns else matrix.columns[0]
    if gene_col in selected.values():
        issues.append("Gene ID column overlaps a GSM sample column")
    ids = matrix[gene_col]
    if ids.isna().any():
        issues.append("Missing target Ensembl identifiers")
    ids = ids.fillna("").astype(str).str.strip()
    ensembl_fraction = float(ids.str.fullmatch(r"ENSG\d+(?:\.\d+)?").mean())
    if ensembl_fraction < 0.9:
        issues.append("Target IDs do not look predominantly like Ensembl gene IDs")
    if ids.duplicated().any():
        issues.append("Target gene IDs have duplicate rows")
    numerical = {}
    for gsm, col in selected.items():
        values = pd.to_numeric(matrix[col], errors="coerce").to_numpy(dtype=float)
        numerical[gsm] = {
            "column": col,
            "missing_or_nonfinite": int((~np.isfinite(values)).sum()),
            "negative_values": int(np.sum(values < 0)),
            "zero_fraction": float(np.mean(values == 0)),
        }
        if not np.isfinite(values).all() or np.any(values < 0):
            issues.append(f"{gsm}: invalid or negative expression values")

    report = {
        "study": "GSE243565", "source": "GSE193258",
        "analysis": "identifier_mapping_and_measurement_preflight_no_model_run",
        "source_gene_count": len(source_symbols),
        "source_gene_id_type": "gene_symbol",
        "target_gene_id_type": "versioned_ensembl",
        "target_id_ensembl_fraction": ensembl_fraction,
        "target_file": str(target_path), "target_sha256": sha256(target_path),
        "target_file_format": fmt,
        "target_matrix_shape": [len(matrix), len(matrix.columns)],
        "target_id_column": gene_col,
        "sample_mapping": {gsm: {**item, "column": selected.get(gsm)} for gsm, item in COHORT.items()},
        "sample_quality": numerical,
        "technical_blockers": issues,
        "gencode_v27_annotation_url": ANNOTATION_URL,
        "annotation_present": annotation_path.is_file(),
    }
    if annotation_path.is_file():
        full, base, symbol_bases, n_genes = parse_gencode_genes(annotation_path)
        report["annotation_sha256"] = sha256(annotation_path)
        report["annotation_gene_features"] = n_genes
        report["gene_mapping_audit"] = audit_gene_mapping(ids, set(source_symbols), full, base, symbol_bases)
    else:
        report["technical_blockers"].append("GENCODE v27 annotation GTF missing: mapping not performed")
    report["external_model_evaluation_approved"] = False
    report["unresolved_scientific_issues"] = [
        "GEO target matrix is normalized gene counts, while the source is log2TPM. They are not interchangeable; no model predictions are authorized.",
        "A validated common quantification and transformation pipeline is needed before cross-study performance can be interpreted.",
        "Three-day osimertinib versus ten-day persister labels confound treatment duration with state.",
        "Samples have induced ASCL1; the cohort is not equivalent to unperturbed HCC827.",
        "HCC827 appeared in previous source development; this is not a new cell-line test.",
        "Biological replicate independence still requires original-methods verification.",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download target matrix only if absent")
    parser.add_argument("--download-annotation", action="store_true",
                        help="Download official GENCODE v27 GTF only if absent")
    parser.add_argument("--annotation-file", type=Path, default=None,
                        help="Existing local GENCODE v27 GTF (.gtf.gz), for offline use")
    args = parser.parse_args()
    target = ROOT / TARGET
    reference = args.annotation_file if args.annotation_file else ROOT / ANNOTATION
    if not target.is_file():
        if not args.download:
            print(f"Target file absent: {target}. Run with --download")
            return 2
        try:
            download_gzip(TARGET_URL, target)
        except Exception as exc:
            print(f"Target download failed: {exc}. See GEO GSE243565 supplementary files")
            return 2
    if not reference.is_file() and args.download_annotation:
        print("Downloading official GENCODE v27 annotation (this is a reference GTF, not sample data)...", flush=True)
        try:
            download_gzip(ANNOTATION_URL, reference)
        except Exception as exc:
            print(f"Annotation download failed: {exc}\nManual URL: {ANNOTATION_URL}\n"
                  f"Save to: {reference}")
            return 2
    try:
        report = inspect(ROOT, target, reference)
    except Exception as exc:
        print(f"Preflight stopped: {exc}")
        return 2
    output = ROOT / REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Target GSM samples identified: {sum(bool(x['column']) for x in report['sample_mapping'].values())}/6")
    if "gene_mapping_audit" in report:
        s = report["gene_mapping_audit"]
        print(f"GENCODE v27 exact-version 1:1 source-symbol matches: {s['exact_version_unique_source_symbols']}")
        print(f"Base-ID-only candidates needing review: {s['base_id_only_candidate_pairs_not_approved']}")
    else:
        print("Annotation absent; gene mapping NOT performed. Use --download-annotation.")
    print(f"Technical blockers: {len(report['technical_blockers'])}")
    for issue in report["technical_blockers"]:
        print(" - " + issue)
    print(f"Report: {output}")
    print("External evaluation: BLOCKED (normalized counts vs log2TPM; no model was run)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
