"""
RAG (Retrieval-Augmented Generation) module — Hybrid BM25 + FAISS search.

Two phases:

  INDEXING  build_index()  — called once per job at the end of the pipeline.

    Dense index  (FAISS):
      - Encodes all clean comment texts with paraphrase-multilingual-MiniLM-L12-v2.
      - L2-normalised → IndexFlatIP == cosine similarity.
      - Persisted to  data/{job_id}/faiss.index

    Sparse index (BM25):
      - Tokenises comments with a lightweight regex tokeniser.
      - Builds a BM25Okapi index (rank_bm25 library).
      - Persisted to  data/{job_id}/bm25.pkl
      - Falls back to saving token lists only when rank_bm25 is unavailable;
        the query path handles both gracefully.

    Metadata (shared):
      - data/{job_id}/comments.json  — [{id, text}, ...]

  QUERYING  query_rag()  — called per user message from the /chat endpoint.

    Hybrid retrieval (Reciprocal Rank Fusion):
      1. Get top-N candidates from FAISS  (dense, semantic)
      2. Get top-N candidates from BM25   (sparse, keyword-exact)
      3. Fuse rankings with RRF:  score = Σ 1/(k + rank)  for k=60
      4. Take the unified top-k, pass to Ollama as context.

    Why hybrid is better than dense-only:
      - FAISS alone misses exact keyword hits (e.g., "Shakira", "messi", a
        specific username) when the query phrasing differs from the comment.
      - BM25 alone misses paraphrase matches (e.g., "people love the song"
        vs "yo song ramro cha").
      - RRF fusion captures both without manual score normalisation.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import re

import numpy as np

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_HOST     = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
DATA_DIR        = os.getenv("DATA_DIR", "data")
TOP_K_DEFAULT   = 5
CANDIDATE_N     = 20   # retrieve this many from each index before fusing

# ── Tokeniser (shared between build and query) ────────────────────────────────

# Keep Devanagari + Latin words, discard punctuation / emoji tokens
_TOK_RE = re.compile(r"[\u0900-\u097F\w]+")

def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())

# ── Embedding model (lazy singleton) ─────────────────────────────────────────

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

# ── Path helpers ──────────────────────────────────────────────────────────────

def _paths(job_id: str) -> tuple[str, str, str]:
    folder = os.path.join(DATA_DIR, job_id)
    return (
        os.path.join(folder, "faiss.index"),
        os.path.join(folder, "bm25.pkl"),
        os.path.join(folder, "comments.json"),
    )

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — INDEXING
# ══════════════════════════════════════════════════════════════════════════════

def build_index(job_id: str, comments: list[dict]) -> None:
    """
    Build and persist the hybrid FAISS + BM25 index for a job.

    Args:
        job_id:   Job identifier used as the subfolder name.
        comments: List of {"id": str, "text": str} dicts.
                  Use clean_text from the preprocessor as "text".

    Writes to data/{job_id}/:
        faiss.index   — dense cosine similarity index
        bm25.pkl      — sparse BM25Okapi index (+ token lists as fallback)
        comments.json — comment IDs + texts for citation
    """
    import faiss
    faiss_path, bm25_path, cmt_path = _paths(job_id)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    texts = [c.get("text") or "." for c in comments]

    # ── Dense index (FAISS) ───────────────────────────────────────────────────
    embedder   = _get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_path)

    # ── Sparse index (BM25) ───────────────────────────────────────────────────
    tokenized = [_tokenize(t) for t in texts]

    bm25_payload: dict = {"tokenized": tokenized}
    try:
        from rank_bm25 import BM25Okapi
        bm25_payload["bm25"] = BM25Okapi(tokenized)
        bm25_payload["available"] = True
    except ImportError:
        # rank_bm25 not installed — store token lists so the query path can
        # do a simple TF-based fallback, or wait for the package to be installed.
        bm25_payload["available"] = False

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_payload, f)

    # ── Metadata ──────────────────────────────────────────────────────────────
    with open(cmt_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — QUERYING
# ══════════════════════════════════════════════════════════════════════════════

def _rrf_fusion(
    dense_idxs:  list[int],
    sparse_idxs: list[int],
    k: int = 60,
) -> list[int]:
    """
    Reciprocal Rank Fusion.

    score(doc) = Σ  1 / (k + rank)   for each ranked list containing doc

    k=60 is the standard default from the original RRF paper (Cormack 2009).
    It de-emphasises the absolute magnitude of ranks and smooths over the
    difference in score distributions between dense and sparse retrievers.

    Returns a list of document indices sorted by descending RRF score.
    """
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idxs):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_idxs):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def _load_artifacts(job_id: str) -> tuple:
    """
    Load FAISS index, BM25 payload, and comment metadata from disk.

    Returns (faiss_index, bm25_payload, comments_list).
    Raises FileNotFoundError if the FAISS index doesn't exist.
    """
    faiss_path, bm25_path, cmt_path = _paths(job_id)

    import faiss
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"No FAISS index for job '{job_id}'. "
            "Has the pipeline completed successfully?"
        )

    faiss_index = faiss.read_index(faiss_path)

    bm25_payload: dict = {"available": False, "tokenized": []}
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_payload = pickle.load(f)

    with open(cmt_path, encoding="utf-8") as f:
        comments = json.load(f)

    return faiss_index, bm25_payload, comments


def _dense_retrieve(
    faiss_index,
    query_vec: np.ndarray,
    n: int,
) -> list[int]:
    """Return top-n document indices from the FAISS index."""
    import faiss
    actual_n = min(n, faiss_index.ntotal)
    _, idxs   = faiss_index.search(query_vec, actual_n)
    return [int(i) for i in idxs[0] if 0 <= i < faiss_index.ntotal]


def _sparse_retrieve(
    bm25_payload: dict,
    query_tokens: list[str],
    n: int,
) -> list[int]:
    """
    Return top-n document indices from the BM25 index.

    Falls back to a simple term-frequency overlap count when rank_bm25 is
    not installed (bm25_payload["available"] == False).
    """
    tokenized = bm25_payload.get("tokenized", [])
    if not tokenized:
        return []

    if bm25_payload.get("available") and "bm25" in bm25_payload:
        # Full BM25Okapi retrieval
        bm25   = bm25_payload["bm25"]
        scores = bm25.get_scores(query_tokens)
    else:
        # Fallback: count how many query tokens appear in each document
        query_set = set(query_tokens)
        scores    = np.array([
            sum(1 for t in doc_tokens if t in query_set)
            for doc_tokens in tokenized
        ], dtype=float)

    actual_n = min(n, len(scores))
    return np.argsort(scores)[::-1][:actual_n].tolist()


def _call_ollama(question: str, source_comments: list[dict]) -> str:
    """Send retrieved comments + question to a local Ollama model."""
    import requests

    context = "\n".join(f"  - {c['text']}" for c in source_comments)
    prompt  = (
        f"You are an analyst summarising a YouTube video's comment section.\n\n"
        f"User question: \"{question}\"\n\n"
        f"Most relevant comments retrieved for this question:\n{context}\n\n"
        f"Answer in 2-3 sentences based strictly on the comments above. "
        f"Be specific — reference what the comments actually say. "
        f"Do not invent or assume information not present in the comments."
    )

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return (
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: open a terminal and run 'ollama serve'."
        )
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return (
                f"Model '{OLLAMA_MODEL}' not found in Ollama. "
                f"Pull it first: ollama pull {OLLAMA_MODEL}"
            )
        return f"Ollama error: {e}"
    except Exception as e:
        return f"Ollama error: {e}"


def query_rag(job_id: str, question: str, top_k: int = TOP_K_DEFAULT) -> dict:
    """
    Hybrid retrieval: fuse FAISS (dense) + BM25 (sparse) via RRF, then
    generate a grounded answer with Ollama.

    Args:
        job_id:   The job whose indexes to search.
        question: The user's natural-language question.
        top_k:    How many fused results to pass as context (default 5).

    Returns:
        {
            "answer":  str   — Ollama's grounded answer
            "sources": [{"id": str, "text": str, "score": float}, ...]
                       score = RRF score (higher = more relevant)
        }

    Raises:
        FileNotFoundError: if the FAISS index doesn't exist for this job.
    """
    embedder                           = _get_embedder()
    faiss_index, bm25_payload, comments = _load_artifacts(job_id)

    # ── Encode query ──────────────────────────────────────────────────────────
    import faiss
    q_vec    = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    q_tokens = _tokenize(question)

    # ── Retrieve from each index ──────────────────────────────────────────────
    candidate_n = min(CANDIDATE_N, len(comments))
    dense_idxs  = _dense_retrieve(faiss_index, q_vec, candidate_n)
    sparse_idxs = _sparse_retrieve(bm25_payload, q_tokens, candidate_n)

    # ── Fuse with RRF ─────────────────────────────────────────────────────────
    fused_idxs = _rrf_fusion(dense_idxs, sparse_idxs)

    # ── Build source list for top_k ───────────────────────────────────────────
    # Compute a normalised RRF score for display (0–1 range for the top-k)
    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idxs):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(sparse_idxs):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)

    max_score = max(rrf_scores.values()) if rrf_scores else 1.0

    sources = []
    for idx in fused_idxs[:top_k]:
        if 0 <= idx < len(comments):
            raw_score = rrf_scores.get(idx, 0.0)
            sources.append({
                "id":    comments[idx]["id"],
                "text":  comments[idx]["text"],
                "score": round(raw_score / max_score, 4),   # 0–1 normalised
            })

    # ── Generate answer ───────────────────────────────────────────────────────
    answer = _call_ollama(question, sources)

    return {"answer": answer, "sources": sources}
