from __future__ import annotations
import re
from collections import Counter, defaultdict

MODEL_NAME = "dslim/bert-base-NER"
_LABEL_MAP = {"PER": "Person", "ORG": "Organization", "LOC": "Location", "MISC": "Miscellaneous"}
_DEV_RE = re.compile(r"[\u0900-\u097F]")
_ner_pipe = None


def _get_model():
    global _ner_pipe
    if _ner_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _ner_pipe = hf_pipeline(task="ner", model=MODEL_NAME, aggregation_strategy="simple", device=device)
    return _ner_pipe


def _clean_entity(text: str) -> str:
    return re.sub(r"^[@#\s]+", "", text).strip()


def extract_entities(comments: list, max_latin: int = 300) -> dict:
    latin, skipped = [], 0
    for c in comments:
        text = c.original_text or ""
        if _DEV_RE.search(text): skipped += 1
        else: latin.append(c)
    latin = latin[:max_latin]

    if not latin:
        return {"entities": [], "total_processed": 0, "total_skipped": skipped, "model_available": False}

    try:
        model = _get_model()
    except (ImportError, OSError, Exception):
        return {"entities": [], "total_processed": len(latin), "total_skipped": skipped, "model_available": False}

    texts = [c.original_text or "." for c in latin]
    try:
        raw_results = model(texts, batch_size=32)
    except Exception:
        return {"entities": [], "total_processed": len(latin), "total_skipped": skipped, "model_available": True}

    entity_info: dict[str, dict] = defaultdict(lambda: {"count": 0, "category": "", "sentiments": []})

    for comment, spans in zip(latin, raw_results):
        sentiment = comment.sentiment_label or "neutral"
        for span in spans:
            raw_label = span.get("entity_group", "")
            category = _LABEL_MAP.get(raw_label, raw_label)
            if not category or span.get("score", 0) < 0.80: continue
            entity_text = _clean_entity(span.get("word", ""))
            if len(entity_text) < 2: continue
            key = entity_text.lower()
            entity_info[key]["count"] += 1
            entity_info[key]["category"] = category
            entity_info[key]["sentiments"].append(sentiment)
            if "variants" not in entity_info[key]: entity_info[key]["variants"] = Counter()
            entity_info[key]["variants"][entity_text] += 1

    entities = []
    for key, info in entity_info.items():
        if info["count"] < 2: continue
        dominant_sent = Counter(info["sentiments"]).most_common(1)[0][0]
        display_name = info["variants"].most_common(1)[0][0] if "variants" in info else key.title()
        entities.append({"text": display_name, "category": info["category"], "count": info["count"], "sentiment": dominant_sent})

    entities.sort(key=lambda x: x["count"], reverse=True)
    return {"entities": entities[:40], "total_processed": len(latin), "total_skipped": skipped, "model_available": True}
