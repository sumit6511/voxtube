"""
Topic modeling module for VoxTube.

Stack:
  Embedder : paraphrase-multilingual-MiniLM-L12-v2  (sentence-transformers)
             Understands both Devanagari and Latin script, so Neplish
             comments about the same topic cluster together regardless
             of which script they use.
  Reducer  : UMAP   — compresses high-dim embeddings to 2-D/5-D space
  Clusterer: HDBSCAN — finds dense regions (topics) and marks sparse
             comments as outliers (topic_id = -1)
  Model    : BERTopic — ties the above together + extracts keywords

Novel contribution:
  After topic modeling, we aggregate the per-comment sentiment labels
  (from Step 3b) into per-topic sentiment distributions. This is the
  "per-topic sentiment aggregation" that makes VoxTube's analysis
  actionable: you can see not just *what* viewers talk about, but
  *how they feel* about each topic.
"""

from __future__ import annotations

from collections import defaultdict

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Module-level singleton — embedding model is heavy, load once
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


# ── Topic modeling ────────────────────────────────────────────────────────────

def run_topic_modeling(texts: list[str], min_topic_size: int = 10) -> dict:
    """
    Run BERTopic on a list of pre-cleaned comment texts.

    Args:
        texts:          Clean strings from the preprocessor.
        min_topic_size: Minimum comments needed to form a topic.
                        Auto-adjusted downward for small datasets.

    Returns:
        {
            "topic_assignments": list[int]   one per comment, −1 = outlier
            "topics": [
                {
                    "topic_id": int,
                    "label":    str,        top-3 keywords joined with " | "
                    "keywords": list[str],  top-10 keywords
                    "count":    int,        comments in this topic
                },
                ...
            ]
        }
    """
    from bertopic import BERTopic

    embedding_model = _get_embedding_model()

    # Scale min_topic_size to the dataset size — avoids "no topics found"
    # on smaller comment sets (e.g. 50 comments → min_size = 5)
    adjusted_min = max(3, min(min_topic_size, len(texts) // 10))

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=adjusted_min,
        calculate_probabilities=False,   # faster; we don't need per-doc probs
        verbose=False,
    )

    # Replace empty strings — BERTopic raises on empty input
    safe_texts = [t if t and t.strip() else "." for t in texts]

    topics, _ = topic_model.fit_transform(safe_texts)

    # Build topic summary list (skip the -1 outlier pseudo-topic)
    result_topics = []
    for _, row in topic_model.get_topic_info().iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue

        words_weights = topic_model.get_topic(tid) or []
        keywords = [word for word, _ in words_weights[:10]]
        label    = " | ".join(keywords[:3]) if keywords else f"Topic {tid}"

        result_topics.append({
            "topic_id": tid,
            "label":    label,
            "keywords": keywords,
            "count":    int(row["Count"]),
        })

    return {
        "topic_assignments": [int(t) for t in topics],
        "topics": result_topics,
    }


# ── Per-topic sentiment aggregation (the key novel contribution) ──────────────

def aggregate_topic_sentiments(
    topic_assignments: list[int],
    sentiment_labels:  list[str],
) -> dict[int, dict]:
    """
    Compute the sentiment distribution for every topic.

    This is the "per-topic sentiment aggregation" novel contribution:
    instead of one overall sentiment score, each discovered topic gets
    its own positive / neutral / negative breakdown.

    Args:
        topic_assignments: list of topic IDs, one per comment (−1 = outlier)
        sentiment_labels:  list of labels ("positive" / "neutral" / "negative"),
                           one per comment, in the same order

    Returns:
        {
            topic_id: {
                "positive": int,
                "neutral":  int,
                "negative": int,
                "count":    int,
            }
        }
    """
    VALID = {"positive", "neutral", "negative"}
    summary: dict = defaultdict(
        lambda: {"positive": 0, "neutral": 0, "negative": 0, "count": 0}
    )

    for tid, label in zip(topic_assignments, sentiment_labels):
        if tid == -1:
            continue                               # outlier — skip
        summary[tid]["count"] += 1
        summary[tid][label if label in VALID else "neutral"] += 1

    return dict(summary)
