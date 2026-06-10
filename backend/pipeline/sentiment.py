"""
Sentiment analysis module for VoxTube.

Two models run on every comment:

  1. XLM-RoBERTa  (primary model)
     Model : cardiffnlp/twitter-xlm-roberta-base-sentiment
     Why   : Multilingual — handles Devanagari script, Romanized Nepali,
             English, and code-mixed Neplish natively.
     Output: label (positive / neutral / negative) + confidence score.

  2. VADER  (academic baseline)
     Why   : Rule-based, deterministic, widely cited in NLP literature.
             English-focused — works on the Latin portions of Neplish.
             Storing both lets us show agreement / disagreement between
             a modern neural model and a classical baseline.
     Output: compound score (−1 to 1) + derived 3-way label.

Both results are stored per comment in the DB so the dashboard and
the research report can compare them side by side.
"""

from __future__ import annotations

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# ── Module-level singletons ───────────────────────────────────────────────────
# Loaded once on first call to _get_models(), then reused for every batch.
# This avoids reloading ~280MB weights on each job.

_xlm_pipe: object | None = None
_vader: SentimentIntensityAnalyzer | None = None


def _get_models():
    """
    Lazy-load both models.
    XLM-RoBERTa is downloaded from HuggingFace Hub on the very first call
    (~280 MB, cached locally after that). Subsequent calls are instant.
    """
    global _xlm_pipe, _vader

    if _xlm_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1  # GPU if present, else CPU
        _xlm_pipe = hf_pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            device=device,
            truncation=True,
            max_length=512,           # XLM-RoBERTa hard limit
        )

    if _vader is None:
        _vader = SentimentIntensityAnalyzer()

    return _xlm_pipe, _vader


# ── VADER helpers ─────────────────────────────────────────────────────────────

def _vader_label(compound: float) -> str:
    """Map VADER compound score → 3-way label using standard thresholds."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _run_vader_only(texts: list[str]) -> list[dict]:
    """Run VADER only (used in tests / when torch is not installed)."""
    _, vader = None, SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        safe = text.strip() if text and text.strip() else "."
        compound = vader.polarity_scores(safe)["compound"]
        results.append({
            "xlm_label":      None,
            "xlm_score":      None,
            "vader_label":    _vader_label(compound),
            "vader_compound": round(compound, 4),
        })
    return results


# ── Main function ─────────────────────────────────────────────────────────────

def analyze_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """
    Run XLM-RoBERTa + VADER on a list of pre-cleaned comment texts.

    Args:
        texts:      List of clean strings from the preprocessor.
        batch_size: XLM-RoBERTa processes this many texts at once.
                    Increase for speed on GPU, decrease if you get OOM errors.

    Returns a list of dicts — one per input comment:
        {
            "xlm_label":      str    "positive" | "neutral" | "negative"
            "xlm_score":      float  confidence, 0–1
            "vader_label":    str    "positive" | "neutral" | "negative"
            "vader_compound": float  −1.0 to 1.0
        }

    Note: XLM-RoBERTa requires torch to be installed. If torch is missing
    (unlikely in production), the function falls back to VADER-only and
    sets xlm_label / xlm_score to None.
    """
    try:
        xlm, vader = _get_models()
    except (ImportError, OSError, Exception):
        # ImportError : torch not installed
        # OSError     : model download blocked / no internet
        return _run_vader_only(texts)

    # Replace empty strings — XLM-RoBERTa raises on empty input
    safe_texts = [t.strip() if t and t.strip() else "." for t in texts]

    xlm_results = xlm(safe_texts, batch_size=batch_size)

    output = []
    for safe, xlm_res in zip(safe_texts, xlm_results):
        compound = vader.polarity_scores(safe)["compound"]
        output.append({
            "xlm_label":      xlm_res["label"].lower(),   # model returns lowercase already
            "xlm_score":      round(xlm_res["score"], 4),
            "vader_label":    _vader_label(compound),
            "vader_compound": round(compound, 4),
        })

    return output
