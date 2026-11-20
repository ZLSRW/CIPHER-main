from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)


def f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.01, 0.99, 199):
        prediction = probabilities >= threshold
        tp = np.sum((prediction == 1) & (labels == 1))
        fp = np.sum((prediction == 1) & (labels == 0))
        fn = np.sum((prediction == 0) & (labels == 1))
        denominator = 2 * tp + fp + fn
        score = 0.0 if denominator == 0 else 2 * tp / denominator
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def summarize(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    prediction = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "acc": float(accuracy_score(labels, prediction)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "sn": float(sensitivity),
        "sp": float(specificity),
        "threshold": float(threshold),
    }

