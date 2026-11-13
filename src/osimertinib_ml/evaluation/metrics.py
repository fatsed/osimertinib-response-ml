"""Strict label-aware metrics for the binary Acute-versus-DTP benchmark."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

STATES = ("Acute", "DTP")


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """Reproduce Stage-35 metrics and explicit Acute/DTP confusion order."""
    y, pred = np.asarray(y, dtype=str), np.asarray(pred, dtype=str)
    if len(y) != len(pred) or not len(y) or set(y) != set(STATES) or not set(pred).issubset(STATES):
        raise ValueError("Binary scores require both true states and valid predictions")
    return {
        "n": len(y),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=STATES, average="macro", zero_division=0)),
        "dtp_recall": float(recall_score(y, pred, pos_label="DTP")),
        "acute_recall": float(recall_score(y, pred, pos_label="Acute")),
        "confusion_matrix_Acute_DTP": confusion_matrix(y, pred, labels=STATES).tolist(),
    }
