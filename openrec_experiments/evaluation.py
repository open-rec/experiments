"""Shared binary metrics and dataset metric dispatch."""
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .datasets import get_dataset

def binary_metrics(labels, scores):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("invalid labels/prediction length")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("predictions must be finite probabilities")
    return {"rows": len(y), "positives": int(y.sum()),
            "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            "logloss": float(log_loss(y, p, labels=[0, 1]))}


def evaluate(frame, scores, dataset):
    result = binary_metrics(frame.label, scores)
    scored = frame.assign(score=np.asarray(scores))
    return get_dataset(dataset).evaluate(scored, result)
