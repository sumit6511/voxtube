"""
RAG (Retrieval-Augmented Generation) module for VoxTube.

Two phases:

  INDEXING  build_index()  — called once per job inside run_pipeline.
    · Embeds all clean comment texts with the same multilingual model
      used by BERTopic (paraphrase-multilingual-MiniLM-L12-v2).
    · Builds a FAISS IndexFlatIP index (cosine similarity after L2 norm).
    · Persists the index + comment metadata to  data/{job_id}/  on disk.

  QUERYING  query_rag()  — called per user message from the /chat endpoint.
    · Embeds the user's question with the same model.
    · Retrieves the top-k most similar comments from the FAISS index.
    · Sends those comments + the question to Gemini 1.5 Flash (002).
    · Returns the LLM answer and the exact source comments it used.

This lets users ask natural-language questions like
"What do people think about the music?" and get answers
grounded in real comments — not hallucinated.
"""

from __future__ import annotations

import json
import os

import faiss
import numpy as np

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_HOST     = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")
DATA_DIR        = os.getenv("DATA_DIR", "data")
TOP_K_DEFAULT   = 5

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _job_folder(job_id: str) -> str:
    return os.path.join(DATA_DIR, job_id)


# ── Phase 1: Indexing ─────────────────────────────────────────────────────────

def build_index(job_id: str, comments: list[dict]) -> None:
    """
    Embed all comments and save a FAISS index to disk.

    Args:
        job_id  : Job identifier — used as the subfolder name.
        comments: List of  {"id": str, "text": str}  dicts.
                  Use clean_text from the preprocessor as "text".

    Creates:
        data/{job_id}/faiss.index    — searchable FAISS index
        data/{job_id}/comments.json  — comment IDs + texts for citation
    """
    embedder = _get_embedder()

    texts      = [c["text"] if c.get("text") else "." for c in comments]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # Normalize → IndexFlatIP becomes cosine similarity search
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    folder = _job_folder(job_id)
    os.makedirs(folder, exist_ok=True)
    faiss.write_index(index, os.path.join(folder, "faiss.index"))

    with open(os.path.join(folder, "comments.json"), "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)


# ── Phase 2: Querying ─────────────────────────────────────────────────────────

def _load_index(job_id: str) -> tuple:
    folder = _job_folder(job_id)
    idx_path = os.path.join(folder, "faiss.index")
    cmt_path = os.path.join(folder, "comments.json")

    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"No FAISS index for job '{job_id}'. "
            "Has the pipeline completed successfully?"
        )

    index = faiss.read_index(idx_path)
    with open(cmt_path, encoding="utf-8") as f:
        comments = json.load(f)

    return index, comments


def _call_ollama(question: str, source_comments: list[dict]) -> str:
    """Send retrieved comments + question to a local Ollama model."""
    import requests

    context = "\n".join(f"  • {c['text']}" for c in source_comments)

    prompt = (
        f'You are an analyst summarizing a YouTube video\'s comment section.\n\n'
        f'User question: "{question}"\n\n'
        f'Most relevant comments retrieved for this question:\n{context}\n\n'
        f'Answer in 2–3 sentences based strictly on the comments above. '
        f'Be specific — reference what the comments actually say. '
        f'Do not invent or assume information not present in the comments.'
    )

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,   # local models can be slow on first token
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.ConnectionError:
        return (
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            "Make sure Ollama is running: open a terminal and run 'ollama serve'."
        )
    except requests.exceptions.HTTPError as e:
        # 404 usually means the model isn't pulled yet
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
    Retrieve the most relevant comments and generate an answer with Gemini.

    Args:
        job_id  : The job whose index to search.
        question: The user's natural-language question.
        top_k   : How many comments to retrieve as context (default 5).

    Returns:
        {
            "answer":  str   — Gemini's grounded answer
            "sources": [{"id": str, "text": str, "score": float}, ...]
        }

    Raises:
        FileNotFoundError: if the FAISS index doesn't exist for this job.
    """
    embedder        = _get_embedder()
    index, comments = _load_index(job_id)

    # Embed and normalize the question
    q_vec = embedder.encode([question], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    k       = min(top_k, len(comments))
    scores, idxs = index.search(q_vec, k)

    sources = [
        {
            "id":    comments[i]["id"],
            "text":  comments[i]["text"],
            "score": round(float(scores[0][j]), 4),
        }
        for j, i in enumerate(idxs[0])
        if 0 <= i < len(comments)
    ]

    answer = _call_ollama(question, sources)

    return {"answer": answer, "sources": sources}
