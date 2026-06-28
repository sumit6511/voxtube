"""
Named Entity Recognition (NER) module for VoxTube.

Model  : dslim/bert-base-NER (BERT fine-tuned on CoNLL-2003)
Labels : PER (person), ORG (organisation), LOC (location), MISC (other)

Runs on Latin-script comments only (English + romanised Neplish).
Devanagari-script comments are skipped — standard English NER models
cannot process Devanagari — and this is noted in the result.

Academic note: extracted entities show *what* (or *who*) audiences
discuss beyond the keyword-level insight BERTopic already provides.
e.g., discovering that "Shakira", "FIFA", and "Morocco" dominate
comments is directly actionable for content creators and marketers.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

MODEL_NAME = "dslim/bert-base-NER"

# Map CoNLL-2003 B/I- tags to clean category names
_LABEL_MAP = {
    "PER":  "Person",
    "ORG":  "Organization",
    "LOC":  "Location",
    "MISC": "Miscellaneous",
}

# Devanagari detection
_DEV_RE = re.compile(r"[\u0900-\u097F]")

_ner_pipe = None


def _get_model():
    global _ner_pipe
    if _ner_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _ner_pipe = hf_pipeline(
            task="ner",
            model=MODEL_NAME,
            aggregation_strategy="simple",   # merge B-/I- tokens into spans
            device=device,
        )
    return _ner_pipe


def _clean_entity(text: str) -> str:
    """Strip leading @ # punctuation and normalise whitespace."""
    return re.sub(r"^[@#\s]+", "", text).strip()


def extract_entities(comments: list, max_latin: int = 300) -> dict:
    """
    Run NER on the Latin-script subset of comments.

    Args:
        comments:  list of Comment ORM / mock objects with attributes:
                   original_text, clean_text, lang, sentiment_label
        max_latin: cap on how many Latin-script comments to process
                   (avoids very long runs on large jobs)

    Returns:
        {
            "entities": [
                {
                    "text":      str        e.g. "Shakira"
                    "category":  str        "Person" | "Organization" | ...
                    "count":     int        total mentions
                    "sentiment": str        dominant sentiment among comments
                                           that mention this entity
                }
            ],
            "total_processed":   int   Latin-script comments processed
            "total_skipped":     int   Devanagari comments skipped
            "model_available":   bool
        }
    """
    latin   = []
    skipped = 0

    for c in comments:
        text = c.original_text or ""
        if _DEV_RE.search(text):
            skipped += 1
        else:
            latin.append(c)

    latin = latin[:max_latin]

    if not latin:
        return {
            "entities":        [],
            "total_processed": 0,
            "total_skipped":   skipped,
            "model_available": False,
        }

    # Try to load the model
    try:
        model = _get_model()
    except (ImportError, OSError, Exception):
        return {
            "entities":        [],
            "total_processed": len(latin),
            "total_skipped":   skipped,
            "model_available": False,
        }

    # Run NER in batches
    texts = [c.original_text or "." for c in latin]

    try:
        raw_results = model(texts, batch_size=32)
    except Exception:
        return {
            "entities":        [],
            "total_processed": len(latin),
            "total_skipped":   skipped,
            "model_available": True,
        }

    # Aggregate: entity_text → {count, sentiments}
    entity_info: dict[str, dict] = defaultdict(lambda: {
        "count": 0, "category": "", "sentiments": []
    })

    for comment, spans in zip(latin, raw_results):
        sentiment = comment.sentiment_label or "neutral"
        for span in spans:
            raw_label = span.get("entity_group", "")
            category  = _LABEL_MAP.get(raw_label, raw_label)
            if not category or span.get("score", 0) < 0.80:
                continue    # skip low-confidence spans

            entity_text = _clean_entity(span.get("word", ""))
            if len(entity_text) < 2:
                continue

            key = entity_text.lower()
            entity_info[key]["count"] += 1
            entity_info[key]["category"] = category
            entity_info[key]["sentiments"].append(sentiment)
            # Keep the most-common capitalisation as the display name
            if "variants" not in entity_info[key]:
                entity_info[key]["variants"] = Counter()
            entity_info[key]["variants"][entity_text] += 1

    # Build sorted entity list (by mention count)
    entities = []
    for key, info in entity_info.items():
        if info["count"] < 2:        # require at least 2 mentions
            continue
        dominant_sent = Counter(info["sentiments"]).most_common(1)[0][0]
        display_name  = (
            info["variants"].most_common(1)[0][0]
            if "variants" in info else key.title()
        )
        entities.append({
            "text":      display_name,
            "category":  info["category"],
            "count":     info["count"],
            "sentiment": dominant_sent,
        })

    entities.sort(key=lambda x: x["count"], reverse=True)

    return {
        "entities":        entities[:40],   # return top-40
        "total_processed": len(latin),
        "total_skipped":   skipped,
        "model_available": True,
    }
