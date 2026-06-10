"""
Toxicity detection module for VoxTube.

Model : unitary/toxic-bert
Task  : Multi-label binary classification
Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate

Key difference from sentiment analysis:
  - Multi-label: a comment can be BOTH toxic AND obscene simultaneously.
  - Each label gets an independent sigmoid score (0–1), not a softmax.
  - A comment is flagged as toxic (is_toxic=1) if ANY label exceeds THRESHOLD.

All 6 scores are stored as JSON in the DB so the dashboard can show
a breakdown (e.g., "obscene: 0.87, insult: 0.45").
"""

from __future__ import annotations

import json

MODEL_NAME = "unitary/toxic-bert"
THRESHOLD  = 0.5   # any score above this → is_toxic = 1

# The 6 labels the model classifies
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Module-level singleton
_tox_pipe: object | None = None


def _get_model():
    """Lazy-load ToxicBERT. Downloaded once (~420 MB), cached locally after."""
    global _tox_pipe
    if _tox_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        _tox_pipe = hf_pipeline(
            task="text-classification",
            model=MODEL_NAME,
            device=device,
            truncation=True,
            max_length=512,
            top_k=None,     # return ALL labels, not just the top one
        )
    return _tox_pipe


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_score_dict(label_score_list: list[dict]) -> dict[str, float]:
    """
    Convert the model output for one comment to a clean score dict.

    Input  (from pipeline):  [{"label": "toxic", "score": 0.95}, ...]
    Output:                  {"toxic": 0.95, "severe_toxic": 0.02, ...}

    All 6 labels are always present (defaulting to 0.0 if missing).
    """
    raw = {item["label"]: round(float(item["score"]), 4) for item in label_score_list}
    return {label: raw.get(label, 0.0) for label in LABELS}


def _default_scores() -> dict[str, float]:
    """Return all-zero scores (safe fallback when torch is unavailable)."""
    return {label: 0.0 for label in LABELS}


# ── Main function ─────────────────────────────────────────────────────────────

def detect_toxicity_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run ToxicBERT on a list of pre-cleaned comment texts.

    Args:
        texts:      List of clean strings from the preprocessor.
        batch_size: Number of texts per model forward pass.

    Returns a list of dicts — one per input comment:
        {
            "is_toxic": int               0 or 1
            "scores":   dict[str, float]  {toxic, severe_toxic, obscene,
                                           threat, insult, identity_hate}
        }

    Fallback: if torch is not installed, returns is_toxic=0 and all
    scores=0.0 so the pipeline can still complete without crashing.
    """
    try:
        model = _get_model()
    except (ImportError, OSError, Exception) as e:
        # ImportError  : torch not installed
        # OSError      : model can't be downloaded (no internet / blocked)
        # Other        : any unexpected loading failure
        # In all cases, return safe defaults so the pipeline keeps running.
        # On a real machine with internet access this branch is never hit.
        return [{"is_toxic": 0, "scores": _default_scores()} for _ in texts]

    # Guard against empty strings (model raises on empty input)
    safe_texts = [t.strip() if t and t.strip() else "." for t in texts]

    # pipeline returns a list-of-lists when top_k=None + batch input
    raw_results = model(safe_texts, batch_size=batch_size)

    output = []
    for label_score_list in raw_results:
        scores   = _to_score_dict(label_score_list)
        is_toxic = int(any(v >= THRESHOLD for v in scores.values()))
        output.append({"is_toxic": is_toxic, "scores": scores})

    return output


def scores_to_json(scores: dict[str, float]) -> str:
    """Serialize a scores dict to a JSON string for DB storage."""
    return json.dumps(scores)


def json_to_scores(json_str: str) -> dict[str, float]:
    """Deserialize a JSON string from the DB back to a scores dict."""
    return json.loads(json_str) if json_str else _default_scores()
