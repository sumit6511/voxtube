"""
RAG module — Hybrid BM25 + FAISS, with embeddings persisted for the
2D scatter plot feature (umap_plot.py reads embeddings.npy).
"""

from __future__ import annotations

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
CANDIDATE_N     = 20

_TOK_RE = re.compile(r"[\u0900-\u097F\w]+")

def _tokenize(text: str) -> list[str]:
    return _TOK_RE.findall((text or "").lower())

_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _paths(job_id: str) -> tuple[str, str, str, str]:
    folder = os.path.join(DATA_DIR, job_id)
    return (
        os.path.join(folder, "faiss.index"),
        os.path.join(folder, "bm25.pkl"),
        os.path.join(folder, "comments.json"),
        os.path.join(folder, "embeddings.npy"),
    )


def build_index(job_id: str, comments: list[dict]) -> None:
    """Build FAISS + BM25 indexes, and save raw embeddings for the scatter plot."""
    import faiss

    faiss_path, bm25_path, cmt_path, emb_path = _paths(job_id)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    texts = [c.get("text") or "." for c in comments]

    embedder   = _get_embedder()
    embeddings = embedder.encode(texts, show_progress_bar=False,
                                  convert_to_numpy=True).astype(np.float32)

    # Save RAW (un-normalised) embeddings for UMAP/PCA — cosine metric handles
    # normalisation internally, no need to pre-normalise for the scatter plot.
    np.save(emb_path, embeddings)

    # FAISS needs L2-normalised vectors for cosine similarity via inner product
    faiss_embeddings = embeddings.copy()
    faiss.normalize_L2(faiss_embeddings)
    index = faiss.IndexFlatIP(faiss_embeddings.shape[1])
    index.add(faiss_embeddings)
    faiss.write_index(index, faiss_path)

    tokenized    = [_tokenize(t) for t in texts]
    bm25_payload = {"tokenized": tokenized}
    try:
        from rank_bm25 import BM25Okapi
        bm25_payload["bm25"]      = BM25Okapi(tokenized)
        bm25_payload["available"] = True
    except ImportError:
        bm25_payload["available"] = False

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_payload, f)

    with open(cmt_path, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)


def _rrf_fusion(dense_idxs: list[int], sparse_idxs: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idxs):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(sparse_idxs):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def _load_artifacts(job_id: str) -> tuple:
    import faiss
    faiss_path, bm25_path, cmt_path, _ = _paths(job_id)

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"No FAISS index for job '{job_id}'. Has the pipeline completed?"
        )

    faiss_index = faiss.read_index(faiss_path)

    bm25_payload: dict = {"available": False, "tokenized": []}
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25_payload = pickle.load(f)

    with open(cmt_path, encoding="utf-8") as f:
        comments = json.load(f)

    return faiss_index, bm25_payload, comments


def _dense_retrieve(faiss_index, query_vec: np.ndarray, n: int) -> list[int]:
    actual_n = min(n, faiss_index.ntotal)
    _, idxs  = faiss_index.search(query_vec, actual_n)
    return [int(i) for i in idxs[0] if 0 <= i < faiss_index.ntotal]


def _sparse_retrieve(bm25_payload: dict, query_tokens: list[str], n: int) -> list[int]:
    tokenized = bm25_payload.get("tokenized", [])
    if not tokenized: return []

    if bm25_payload.get("available") and "bm25" in bm25_payload:
        scores = bm25_payload["bm25"].get_scores(query_tokens)
    else:
        query_set = set(query_tokens)
        scores = np.array([
            sum(1 for t in doc_tokens if t in query_set) for doc_tokens in tokenized
        ], dtype=float)

    actual_n = min(n, len(scores))
    return np.argsort(scores)[::-1][:actual_n].tolist()


def _call_ollama(question: str, source_comments: list[dict], model: str | None = None) -> str:
    import requests
    selected_model = model or OLLAMA_MODEL
    context = "\n".join(f"  - {c['text']}" for c in source_comments)
    prompt = (
        f'You are an analyst summarizing a YouTube video\'s comment section.\n\n'
        f'User question: "{question}"\n\n'
        f'Most relevant comments retrieved for this question:\n{context}\n\n'
        f'Answer in 2-3 sentences based strictly on the comments above. '
        f'Be specific - reference what the comments actually say. '
        f'Do not invent or assume information not present in the comments.'
    )
    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": selected_model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (f"Cannot reach Ollama at {OLLAMA_HOST}. "
                "Make sure Ollama is running: open a terminal and run 'ollama serve'.")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return (f"Model '{selected_model}' not found in Ollama. "
                    f"Pull it first: ollama pull {selected_model}")
        return f"Ollama error: {e}"
    except Exception as e:
        return f"Ollama error: {e}"


def query_rag(job_id: str, question: str, top_k: int = TOP_K_DEFAULT,
              model: str | None = None) -> dict:
    import faiss

    embedder                            = _get_embedder()
    faiss_index, bm25_payload, comments = _load_artifacts(job_id)

    q_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    q_tokens = _tokenize(question)

    candidate_n = min(CANDIDATE_N, len(comments))
    dense_idxs  = _dense_retrieve(faiss_index, q_vec, candidate_n)
    sparse_idxs = _sparse_retrieve(bm25_payload, q_tokens, candidate_n)
    fused_idxs  = _rrf_fusion(dense_idxs, sparse_idxs)

    rrf_scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_idxs):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
    for rank, idx in enumerate(sparse_idxs):
        rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank + 1)
    max_score = max(rrf_scores.values()) if rrf_scores else 1.0

    sources = []
    for idx in fused_idxs[:top_k]:
        if 0 <= idx < len(comments):
            sources.append({
                "id":    comments[idx]["id"],
                "text":  comments[idx]["text"],
                "score": round(rrf_scores.get(idx, 0.0) / max_score, 4),
            })

    answer = _call_ollama(question, sources, model=model)
    return {"answer": answer, "sources": sources}
