"""
Evaluation module for VoxTube.

Loads the hand-labeled Neplish dataset (data/neplish_dataset.csv),
runs both XLM-RoBERTa and VADER on every sample, and computes:
  - Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
  - 3×3 Confusion Matrix  [positive, neutral, negative]

Academic purpose: demonstrate that XLM-RoBERTa outperforms VADER on
Neplish (code-mixed Nepali-English) text — especially on Devanagari
comments where VADER assigns a neutral score of 0.000.
"""

from __future__ import annotations

import csv
import os

DATASET_PATH = os.getenv("DATASET_PATH", "data/neplish_dataset.csv")
LABELS       = ["positive", "neutral", "negative"]


def _load_dataset(path: str) -> tuple[list[str], list[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Labeled dataset not found at '{path}'. "
            "Expected CSV with columns: text, label"
        )
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text  = row.get("text", "").strip()
            label = row.get("label", "").strip().lower()
            if text and label in LABELS:
                texts.append(text)
                labels.append(label)
    return texts, labels


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Return accuracy, weighted precision/recall/F1, and confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix,
    )

    # Replace any prediction not in LABELS with 'neutral'
    y_pred_clean = [p if p in LABELS else "neutral" for p in y_pred]

    return {
        "accuracy":         round(float(accuracy_score(y_true, y_pred_clean)), 4),
        "precision":        round(float(precision_score(y_true, y_pred_clean,
                                          average="weighted", zero_division=0)), 4),
        "recall":           round(float(recall_score(y_true, y_pred_clean,
                                          average="weighted", zero_division=0)), 4),
        "f1":               round(float(f1_score(y_true, y_pred_clean,
                                          average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(
                                y_true, y_pred_clean, labels=LABELS
                            ).tolist(),
    }


def run_evaluation(dataset_path: str = DATASET_PATH) -> dict:
    """
    Run XLM-RoBERTa and VADER on the labeled dataset and return metrics.

    Returns:
        {
            "total_samples":      int
            "label_distribution": {"positive": int, "neutral": int, "negative": int}
            "xlm_roberta":        MetricsDict | None   (None if torch unavailable)
            "vader":              MetricsDict
            "note":               str | None
        }

    MetricsDict has keys: accuracy, precision, recall, f1, confusion_matrix
    """
    from .preprocessor import preprocess_batch
    from .sentiment    import analyze_batch

    texts, true_labels = _load_dataset(dataset_path)
    clean_texts = preprocess_batch(texts)
    results     = analyze_batch(clean_texts)

    xlm_preds   = [r["xlm_label"]   for r in results]
    vader_preds = [r["vader_label"]  for r in results]

    label_dist = {lbl: true_labels.count(lbl) for lbl in LABELS}

    # XLM-RoBERTa metrics — only if predictions were actually produced
    xlm_valid = [p for p in xlm_preds if p is not None]
    if len(xlm_valid) == len(texts):
        xlm_metrics = _compute_metrics(true_labels, xlm_preds)
        note = None
    else:
        xlm_metrics = None
        note = (
            "XLM-RoBERTa predictions unavailable — torch may not be installed "
            "or the model could not be loaded. VADER metrics are shown only."
        )

    return {
        "total_samples":      len(texts),
        "label_distribution": label_dist,
        "xlm_roberta":        xlm_metrics,
        "vader":              _compute_metrics(true_labels, vader_preds),
        "note":               note,
    }
