"""
RAG module — Hybrid BM25 + FAISS, with embeddings persisted for the
2D scatter plot feature (umap_plot.py reads embeddings.npy).
"""

from __future__ import annotations

import json
import os
import pickle
import re
from functools import lru_cache

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


@lru_cache(maxsize=8)
def _load_artifacts(job_id: str) -> tuple:
    # Each job_id is a fresh UUID whose index/comments are written once by
    # build_index() and never touched again, so caching by job_id alone
    # (no mtime/staleness check) is safe — this just saves re-reading the
    # FAISS index, BM25 pickle, and full comments JSON from disk on every
    # single chat message in a conversation.
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


MAX_HISTORY_TURNS = 6  # last N messages (user+assistant) kept for context


def _recent_history(history: list[dict] | None) -> list[dict]:
    return list(history or [])[-MAX_HISTORY_TURNS:]


def _contextualized_query(question: str, history: list[dict] | None) -> str:
    """Cheap follow-up handling without an extra LLM round-trip: fold the
    last user message into the retrieval query so "what about the audio?"
    after "what do people think of the visuals?" still retrieves relevant
    comments instead of just whatever loosely matches "audio" alone."""
    for turn in reversed(_recent_history(history)):
        if turn.get("role") == "user" and turn.get("text") != question:
            return f"{turn['text']} {question}"
    return question


def _build_prompt(question: str, source_comments: list[dict], history: list[dict] | None = None) -> str:
    context = "\n".join(f"  - {c['text']}" for c in source_comments)
    history_block = ""
    recent = _recent_history(history)
    if recent:
        turns = "\n".join(
            f'{"User" if t.get("role") == "user" else "You"}: {t.get("text", "")}'
            for t in recent
        )
        history_block = f'Conversation so far:\n{turns}\n\n'
    return (
        f'You are an analyst discussing a YouTube video\'s comment section with a user.\n\n'
        f'{history_block}'
        f'User question: "{question}"\n\n'
        f'Relevant comments for this question:\n{context}\n\n'
        f'Answer in 1-2 direct, concise sentences that synthesize the overall '
        f'takeaway, using the conversation so far for context if the question '
        f'is a follow-up — do not quote comments verbatim, list them one by '
        f'one, or refer to "comment 1" / "one comment" etc. The source '
        f'comments are already shown separately to the user, so just give the '
        f'answer itself. Base it strictly on the comments above; do not '
        f'invent or assume anything not present in them.'
    )


def _call_ollama(question: str, source_comments: list[dict], model: str | None = None,
                  history: list[dict] | None = None) -> str:
    import requests
    selected_model = model or OLLAMA_MODEL
    prompt = _build_prompt(question, source_comments, history)
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


def _retrieve_sources(job_id: str, question: str, top_k: int,
                       history: list[dict] | None = None) -> list[dict]:
    import faiss

    embedder                            = _get_embedder()
    faiss_index, bm25_payload, comments = _load_artifacts(job_id)

    retrieval_query = _contextualized_query(question, history)
    q_vec = embedder.encode([retrieval_query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)
    q_tokens = _tokenize(retrieval_query)

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
    return sources


def query_rag(job_id: str, question: str, top_k: int = TOP_K_DEFAULT,
              model: str | None = None, history: list[dict] | None = None) -> dict:
    sources = _retrieve_sources(job_id, question, top_k, history)
    answer  = _call_ollama(question, sources, model=model, history=history)
    return {"answer": answer, "sources": sources}


def query_rag_stream(job_id: str, question: str, top_k: int = TOP_K_DEFAULT,
                      model: str | None = None, history: list[dict] | None = None):
    """NDJSON generator consumed by the /chat endpoint. Emits one JSON object
    per line: {"type": "sources"|"token"|"error"|"done", ...}."""
    import requests

    def _emit(obj: dict) -> str:
        return json.dumps(obj) + "\n"

    try:
        sources = _retrieve_sources(job_id, question, top_k, history)
    except Exception as e:
        yield _emit({"type": "error", "message": f"Retrieval failed: {e}"})
        return

    yield _emit({"type": "sources", "sources": sources})

    selected_model = model or OLLAMA_MODEL
    prompt = _build_prompt(question, sources, history)

    try:
        resp = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": selected_model, "prompt": prompt, "stream": True},
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        yield _emit({"type": "error", "message":
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: open a terminal and run 'ollama serve'."})
        return
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            yield _emit({"type": "error", "message":
                f"Model '{selected_model}' not found in Ollama. Pull it first: ollama pull {selected_model}"})
        else:
            yield _emit({"type": "error", "message": f"Ollama error: {e}"})
        return
    except Exception as e:
        yield _emit({"type": "error", "message": f"Ollama error: {e}"})
        return

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield _emit({"type": "token", "text": chunk["response"]})
            if chunk.get("done"):
                break
    except Exception as e:
        yield _emit({"type": "error", "message": f"Streaming error: {e}"})
        return

    yield _emit({"type": "done"})
