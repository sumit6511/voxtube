from __future__ import annotations
import csv
import os

DATASET_PATH = os.getenv("DATASET_PATH", "data/neplish_dataset.csv")
LABELS = ["positive", "neutral", "negative"]


def _load_dataset(path: str) -> tuple[list[str], list[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labeled dataset not found at '{path}'.")
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            text = row.get("text", "").strip()
            label = row.get("label", "").strip().lower()
            if text and label in LABELS:
                texts.append(text); labels.append(label)
    return texts, labels


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    y_pred_clean = [p if p in LABELS else "neutral" for p in y_pred]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred_clean)), 4),
        "precision": round(float(precision_score(y_true, y_pred_clean, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred_clean, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred_clean, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred_clean, labels=LABELS).tolist(),
    }


def run_evaluation(dataset_path: str = DATASET_PATH) -> dict:
    from .preprocessor import preprocess_batch
    from .sentiment import analyze_batch, analyze_batch_nepali_model

    texts, true_labels = _load_dataset(dataset_path)
    clean_texts = preprocess_batch(texts)
    results = analyze_batch(clean_texts)

    xlm_preds = [r["xlm_label"] for r in results]
    vader_preds = [r["vader_label"] for r in results]
    label_dist = {lbl: true_labels.count(lbl) for lbl in LABELS}

    xlm_valid = [p for p in xlm_preds if p is not None]
    if len(xlm_valid) == len(texts):
        xlm_metrics = _compute_metrics(true_labels, xlm_preds)
        note = None
    else:
        xlm_metrics = None
        note = "XLM-RoBERTa predictions unavailable - torch may not be installed. VADER metrics shown only."

    # Third comparison point: the Nepali-specific alternative model.
    # This is a genuine empirical test, not an assumed improvement — the
    # model's own documentation flags real weaknesses (see sentiment.py).
    # Runs best-effort: if the model can't load or fails, we simply omit
    # it from the response rather than failing the whole evaluation.
    nepali_results = analyze_batch_nepali_model(clean_texts)
    nepali_preds = [r["label"] if r else None for r in nepali_results]
    nepali_valid = [p for p in nepali_preds if p is not None]
    if len(nepali_valid) == len(texts):
        nepali_metrics = _compute_metrics(true_labels, nepali_preds)
        nepali_note = None
    else:
        nepali_metrics = None
        nepali_note = ("Nepali-specific model unavailable or failed to load — this comparison "
                        "is optional and does not affect the main XLM-RoBERTa vs VADER result.")

    return {
        "total_samples": len(texts), "label_distribution": label_dist,
        "xlm_roberta": xlm_metrics, "vader": _compute_metrics(true_labels, vader_preds), "note": note,
        "nepali_model": nepali_metrics, "nepali_model_note": nepali_note,
    }
