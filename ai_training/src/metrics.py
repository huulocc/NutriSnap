from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, labels: list[str]) -> dict:
    y_pred = np.argmax(y_prob, axis=1)
    top3 = np.argsort(y_prob, axis=1)[:, -3:]
    return {
        "top1_accuracy": float(np.mean(y_pred == y_true)),
        "top3_accuracy": float(np.mean([truth in row for truth, row in zip(y_true, top3)])),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist(),
    }


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "label"})
