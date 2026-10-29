import time
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score)


def classification_metrics(y_true, y_pred, y_proba=None, labels=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    m = {"accuracy": accuracy_score(y_true, y_pred),
         "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
         "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
         "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
         "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
         "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    if y_proba is not None:
        try:
            m["roc_auc_ovr"] = roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)
        except ValueError:
            m["roc_auc_ovr"] = None      # a class is missing from this split
    return {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()}


def inference_time_ms(fn, X, repeat=5):
    """Mean latency per sample in milliseconds."""
    fn(X[:1])
    t = time.perf_counter()
    for _ in range(repeat):
        fn(X)
    return (time.perf_counter() - t) / repeat / len(X) * 1000.0
