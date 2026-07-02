from __future__ import annotations
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
_xlm_pipe = None
_vader = None


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
