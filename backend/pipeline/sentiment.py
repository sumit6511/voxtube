from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Nepali-specific alternative model, used ONLY for the evaluation-page
# comparison (see evaluate.py), never in the live analysis pipeline.
# Known caveats, per the model's own card: trained without a properly
# represented neutral class, and on formal/news-style Nepali text rather
# than informal code-mixed social comments — so this is a genuine
# empirical comparison, not an assumed improvement.
NEPALI_MODEL_NAME = "sibendra/nepali-sentiment-analysis"

_xlm_pipe = None
_vader = None
_nepali_pipe = None


def _get_models():
    global _xlm_pipe, _vader
    if _xlm_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _xlm_pipe = hf_pipeline(task="sentiment-analysis", model=MODEL_NAME,
                                 device=device, truncation=True, max_length=512)
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _xlm_pipe, _vader


def _get_nepali_model():
    global _nepali_pipe
    if _nepali_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _nepali_pipe = hf_pipeline(task="sentiment-analysis", model=NEPALI_MODEL_NAME,
                                    device=device, truncation=True, max_length=512)
    return _nepali_pipe


def _vader_label(compound: float) -> str:
    if compound >= 0.05: return "positive"
    if compound <= -0.05: return "negative"
    return "neutral"


def _run_vader_only(texts: list[str]) -> list[dict]:
    vader = SentimentIntensityAnalyzer()
    results = []
    for text in texts:
        safe = text.strip() if text and text.strip() else "."
        compound = vader.polarity_scores(safe)["compound"]
        results.append({
            "xlm_label": None, "xlm_score": None,
            "vader_label": _vader_label(compound), "vader_compound": round(compound, 4),
        })
    return results


def analyze_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    """Production sentiment path — XLM-RoBERTa + VADER. Unchanged from
    before; this is what the live analysis pipeline actually calls."""
    try:
        xlm, vader = _get_models()
    except (ImportError, OSError, Exception):
        return _run_vader_only(texts)

    safe_texts = [t.strip() if t and t.strip() else "." for t in texts]
    xlm_results = xlm(safe_texts, batch_size=batch_size)

    output = []
    for safe, xlm_res in zip(safe_texts, xlm_results):
        compound = vader.polarity_scores(safe)["compound"]
        output.append({
            "xlm_label": xlm_res["label"].lower(), "xlm_score": round(xlm_res["score"], 4),
            "vader_label": _vader_label(compound), "vader_compound": round(compound, 4),
        })
    return output


# Label mapping: this model's output labels vary by checkpoint config
# (some report as LABEL_0/1/2, others as actual words) — normalize both.
_NEPALI_LABEL_MAP = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
    "0": "negative", "1": "neutral", "2": "positive",
}


def analyze_batch_nepali_model(texts: list[str], batch_size: int = 32) -> list[dict | None]:
    """Evaluation-only path — runs the Nepali-specific alternative model.
    Returns None entries (not a crash) if the model can't be loaded, so
    the evaluate.py comparison degrades gracefully rather than failing
    the whole /evaluate request."""
    try:
        model = _get_nepali_model()
    except Exception:
        return [None] * len(texts)

    safe_texts = [t.strip() if t and t.strip() else "." for t in texts]
    try:
        raw_results = model(safe_texts, batch_size=batch_size)
    except Exception:
        return [None] * len(texts)

    output = []
    for res in raw_results:
        raw_label = str(res.get("label", "")).lower()
        label = _NEPALI_LABEL_MAP.get(raw_label)
        if label is None:
            output.append(None)
        else:
            output.append({"label": label, "score": round(res.get("score", 0.0), 4)})
    return output
