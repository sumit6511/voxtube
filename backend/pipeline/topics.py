from __future__ import annotations
from collections import defaultdict

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def run_topic_modeling(texts: list[str], min_topic_size: int = 10) -> dict:
    from bertopic import BERTopic
    embedding_model = _get_embedding_model()
    adjusted_min = max(3, min(min_topic_size, len(texts) // 10))

    topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=adjusted_min,
                            calculate_probabilities=False, verbose=False)
    safe_texts = [t if t and t.strip() else "." for t in texts]
    topics, _ = topic_model.fit_transform(safe_texts)

    result_topics = []
    for _, row in topic_model.get_topic_info().iterrows():
        tid = int(row["Topic"])
        if tid == -1: continue
        words_weights = topic_model.get_topic(tid) or []
        keywords = [word for word, _ in words_weights[:10]]
        label = " | ".join(keywords[:3]) if keywords else f"Topic {tid}"
        result_topics.append({"topic_id": tid, "label": label, "keywords": keywords, "count": int(row["Count"])})

    return {"topic_assignments": [int(t) for t in topics], "topics": result_topics}


def aggregate_topic_sentiments(topic_assignments: list[int], sentiment_labels: list[str]) -> dict[int, dict]:
    VALID = {"positive", "neutral", "negative"}
    summary: dict = defaultdict(lambda: {"positive": 0, "neutral": 0, "negative": 0, "count": 0})
    for tid, label in zip(topic_assignments, sentiment_labels):
        if tid == -1: continue
        summary[tid]["count"] += 1
        summary[tid][label if label in VALID else "neutral"] += 1
    return dict(summary)
