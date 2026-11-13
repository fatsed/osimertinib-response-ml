"""Frozen sample-integrity and split checks for the GSE193258 binary benchmark.

This is cohort-specific infrastructure; it does not infer biology or modify
sample labels. All samples remain provisional pending source review.
"""
from __future__ import annotations

import pandas as pd

STATES = ("Acute", "DTP")


def validate_binary_cohort(meta: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """Check that binary samples are exactly the source-only Stage-31 subset."""
    if not {"gsm", "cell_line", "state"}.issubset(meta) or meta.gsm.duplicated().any():
        raise ValueError("Invalid source metadata")
    if len(folds) != 4 or len({f["outer_test_cell_line"] for f in folds}) != 4:
        raise ValueError("Expected four distinct frozen outer folds")
    expected = {gsm for f in folds for gsm in f["outer_test_gsm"]}
    if len(expected) != 36 or set(meta.gsm) != expected:
        raise ValueError("Stage-31 source GSM identity mismatch")
    subset = meta.loc[meta.state.isin(STATES)].copy()
    if len(subset) != 24 or set(subset.state) != set(STATES):
        raise ValueError("Expected 24 Acute/DTP samples")
    if set(subset.cell_line) != {f["outer_test_cell_line"] for f in folds}:
        raise ValueError("Cell-line identities do not match frozen folds")
    count = subset.groupby(["cell_line", "state"]).size()
    if len(count) != 8 or not count.eq(3).all():
        raise ValueError("Need exactly 3 Acute and 3 DTP samples per cell line")
    return subset


def select_frozen_rows(fold: dict, metadata: pd.DataFrame) -> tuple[list[str], list[str], list[tuple[list[str], list[str]]]]:
    """Select Acute/DTP sample IDs using *only* the frozen nested LOCO splits."""
    lookup = metadata.set_index("gsm")["state"].to_dict()

    def keep(ids):
        if any(g not in lookup for g in ids):
            raise ValueError("GSM absent from source-only metadata")
        return [g for g in ids if lookup[g] in STATES]

    train, test = keep(fold["outer_train_gsm"]), keep(fold["outer_test_gsm"])
    if len(train) != 18 or len(test) != 6 or set(train) & set(test):
        raise ValueError("Invalid binary outer split")
    inners = [(keep(f["train_gsm"]), keep(f["validation_gsm"])) for f in fold["inner_folds"]]
    if len(inners) != 3:
        raise ValueError("Expected exactly three inner folds")
    for a, b in inners:
        if len(a) != 12 or len(b) != 6 or set(a) & set(b) or (set(a) | set(b)) != set(train):
            raise ValueError("Inner-fold overlap, missing rows or outer-test leakage")
    return train, test, inners
