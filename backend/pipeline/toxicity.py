from __future__ import annotations
import json

MODEL_NAME = "unitary/toxic-bert"
THRESHOLD = 0.5
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
_tox_pipe = None


def _get_model():
    global _tox_pipe
    if _tox_pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline
        device = 0 if torch.cuda.is_available() else -1
        _tox_pipe = hf_pipeline(task="text-classification", model=MODEL_NAME,
                                 device=device, truncation=True, max_length=512, top_k=None)
    return _tox_pipe


def _to_score_dict(label_score_list: list[dict]) -> dict[str, float]:
    raw = {item["label"]: round(float(item["score"]), 4) for item in label_score_list}
    return {label: raw.get(label, 0.0) for label in LABELS}


def _default_scores() -> dict[str, float]:
    return {label: 0.0 for label in LABELS}


def detect_toxicity_batch(texts: list[str], batch_size: int = 32) -> list[dict]:
    try:
        model = _get_model()
    except (ImportError, OSError, Exception):
        return [{"is_toxic": 0, "scores": _default_scores()} for _ in texts]

    safe_texts = [t.strip() if t and t.strip() else "." for t in texts]
    raw_results = model(safe_texts, batch_size=batch_size)

    output = []
    for label_score_list in raw_results:
        scores = _to_score_dict(label_score_list)
        is_toxic = int(any(v >= THRESHOLD for v in scores.values()))
        output.append({"is_toxic": is_toxic, "scores": scores})
    return output


def scores_to_json(scores: dict[str, float]) -> str:
    return json.dumps(scores)


def json_to_scores(json_str: str) -> dict[str, float]:
    return json.loads(json_str) if json_str else _default_scores()
