"""
2D projection module for VoxTube comment scatter plot.

Primary:  UMAP  (umap-learn) — best semantic separation
Fallback: PCA   (sklearn)    — always available, less separation

Both operate on the same sentence-transformer embeddings saved
by the RAG build_index step (data/{job_id}/embeddings.npy).

The scatter plot is one of the most visually striking features
for an FYP defense: it shows comment clusters in 2D space,
coloured by sentiment, making the NLP pipeline's output
immediately intuitive to a non-technical audience.
"""

from __future__ import annotations

import json
import os

import numpy as np

DATA_DIR = os.getenv("DATA_DIR", "data")


def _load_embeddings(job_id: str) -> tuple[np.ndarray, list[dict]]:
    folder = os.path.join(DATA_DIR, job_id)
    emb_path = os.path.join(folder, "embeddings.npy")
    cmt_path = os.path.join(folder, "comments.json")

    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"No embeddings found for job '{job_id}'. "
            "Re-analyze the video to generate them."
        )

    embeddings = np.load(emb_path)

    with open(cmt_path, encoding="utf-8") as f:
        comments = json.load(f)

    return embeddings, comments


def compute_2d_projection(job_id: str, comments_db: list) -> dict:
    """
    Project comment embeddings to 2D and return scatter plot data.

    Args:
        job_id:      Job identifier for loading embeddings from disk.
        comments_db: List of Comment ORM instances (for sentiment/lang labels).

    Returns:
        {
            "points": [
                {
                    "id":        str,     Comment.id — lets the frontend look up
                                          the FULL comment (this "text" field is
                                          truncated to keep the payload small)
                    "x":        float,
                    "y":        float,
                    "sentiment": str,     "positive"|"neutral"|"negative"
                    "lang":      str,     "nepali"|"english"|"neplish"
                    "text":      str,     first 70 chars of original_text
                    "is_toxic":  int,     0 or 1
                }
            ],
            "method": str,   "umap" | "pca"
            "total":  int
        }
    """
    embeddings, cmt_metadata = _load_embeddings(job_id)

    # Align embeddings with DB comments (same insertion order)
    n = min(len(embeddings), len(comments_db))
    embeddings = embeddings[:n]
    db_slice   = comments_db[:n]

    # ── Dimensionality reduction ──────────────────────────────────────────────
    method = "umap"
    try:
        import umap
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            verbose=False,
        )
        coords_2d = reducer.fit_transform(embeddings)

    except ImportError:
        # Fallback to PCA (sklearn, already installed)
        method = "pca"
        from sklearn.decomposition import PCA
        pca       = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)

    # ── Normalise to [-1, 1] for consistent chart axes ───────────────────────
    for dim in range(2):
        col   = coords_2d[:, dim]
        lo, hi = col.min(), col.max()
        if hi > lo:
            coords_2d[:, dim] = 2 * (col - lo) / (hi - lo) - 1

    # ── Build point list ──────────────────────────────────────────────────────
    points = []
    for i, c in enumerate(db_slice):
        x, y = float(coords_2d[i, 0]), float(coords_2d[i, 1])
        points.append({
            "id":        c.id,
            "x":        round(x, 4),
            "y":        round(y, 4),
            "sentiment": c.sentiment_label or "neutral",
            "lang":      c.lang            or "neplish",
            "text":      (c.original_text or "")[:70],
            "is_toxic":  int(c.is_toxic   or 0),
        })

    return {"points": points, "method": method, "total": len(points)}
