import json
import os
import uuid
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .database import engine, get_db, Base, run_migrations, SessionLocal
from .models import Job, Comment, Topic
from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    JobStatusResponse, ResultsResponse,
    CommentOut, TopicOut, SentimentSummary,
    ChatRequest, ChatResponse, SourceComment,
    EvaluationResponse, MetricsResult,
    JobSummary, JobListResponse,
)

DATA_DIR = os.getenv("DATA_DIR", "data")

Base.metadata.create_all(bind=engine)
run_migrations(engine)

app = FastAPI(title="VoxTube API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pipeline helpers ──────────────────────────────────────────────────────────

STAGES = {
    "fetching":        (5,  20),
    "preprocessing":   (20, 35),
    "analyzing":       (35, 55),
    "toxicity":        (55, 70),
    "building_topics": (70, 85),
    "building_rag":    (85, 98),
    "done":            (100, 100),
}

def _set_job(db, job_id: str, status: str, progress: int, **kwargs):
    updates = {"status": status, "progress": progress, **kwargs}
    db.query(Job).filter(Job.id == job_id).update(updates)
    db.commit()

def _parse_ts(s):
    from datetime import datetime
    if not s: return None
    try: return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except: return None

# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(job_id: str, youtube_url: str, max_comments: int):
    from .youtube import fetch_comments

    db = SessionLocal()
    try:
        _set_job(db, job_id, "fetching", 5)
        result = fetch_comments(youtube_url, max_comments)

        comment_rows = [
            Comment(
                job_id=job_id,
                original_text=item["text"],
                published_at=_parse_ts(item.get("published_at")),
            )
            for item in result["comments"]
        ]
        db.bulk_save_objects(comment_rows)
        _set_job(
            db, job_id, "preprocessing", 20,
            video_id=result["video_id"],
            video_title=result["video_title"],
            channel_title=result.get("channel_title"),
            view_count=result.get("view_count"),
            like_count=result.get("like_count"),
            comment_count=len(result["comments"]),
        )

        from .pipeline.preprocessor import preprocess_batch, detect_languages

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        clean_texts    = preprocess_batch([c.original_text for c in comments_in_db])
        lang_labels    = detect_languages(clean_texts)

        for comment, clean, lang in zip(comments_in_db, clean_texts, lang_labels):
            comment.clean_text = clean
            comment.lang       = lang
        db.commit()
        _set_job(db, job_id, "analyzing", 35)

        from .pipeline.sentiment import analyze_batch as sentiment_batch

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        texts          = [c.clean_text or c.original_text for c in comments_in_db]
        sent_results   = sentiment_batch(texts)

        for comment, res in zip(comments_in_db, sent_results):
            comment.sentiment_label = res["xlm_label"]
            comment.sentiment_score = res["xlm_score"]
            comment.vader_label     = res["vader_label"]
            comment.vader_compound  = res["vader_compound"]
        db.commit()
        _set_job(db, job_id, "toxicity", 55)

        from .pipeline.toxicity import detect_toxicity_batch, scores_to_json

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        texts          = [c.clean_text or c.original_text for c in comments_in_db]
        tox_results    = detect_toxicity_batch(texts)

        for comment, res in zip(comments_in_db, tox_results):
            comment.is_toxic      = res["is_toxic"]
            comment.toxicity_json = scores_to_json(res["scores"])
        db.commit()
        _set_job(db, job_id, "building_topics", 70)

        from .pipeline.topics import run_topic_modeling, aggregate_topic_sentiments

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        clean_texts    = [c.clean_text or c.original_text for c in comments_in_db]
        sent_labels    = [c.sentiment_label or "neutral"   for c in comments_in_db]

        topic_result = run_topic_modeling(clean_texts)
        assignments  = topic_result["topic_assignments"]

        for comment, tid in zip(comments_in_db, assignments):
            comment.topic_id = tid
        db.commit()

        sentiment_per_topic = aggregate_topic_sentiments(assignments, sent_labels)

        for t in topic_result["topics"]:
            tid  = t["topic_id"]
            sent = sentiment_per_topic.get(
                tid, {"positive": 0, "neutral": 0, "negative": 0, "count": 0}
            )
            db.add(Topic(
                job_id=job_id,
                topic_id=tid,
                label=t["label"],
                keywords_json=json.dumps(t["keywords"]),
                comment_count=sent["count"],
                positive_count=sent["positive"],
                neutral_count=sent["neutral"],
                negative_count=sent["negative"],
            ))
        db.commit()
        _set_job(db, job_id, "building_rag", 85)

        from .pipeline.rag import build_index

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        comment_dicts  = [
            {"id": c.id, "text": c.clean_text or c.original_text}
            for c in comments_in_db
        ]
        build_index(job_id, comment_dicts)
        _set_job(db, job_id, "done", 100)

    except Exception as e:
        _set_job(db, job_id, "failed", 0, error_message=str(e))
    finally:
        db.close()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health(): return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks,
            db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    db.add(Job(id=job_id, youtube_url=request.url, status="pending", progress=0))
    db.commit()
    background_tasks.add_task(run_pipeline, job_id, request.url, request.max_comments)
    return AnalyzeResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    return JobStatusResponse(
        job_id=job.id, status=job.status, progress=job.progress,
        comment_count=job.comment_count, video_id=job.video_id,
        video_title=job.video_title, error_message=job.error_message,
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
def get_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")

    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    topics   = db.query(Topic).filter(Topic.job_id   == job_id).all()

    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for c in comments:
        if c.sentiment_label in counts: counts[c.sentiment_label] += 1

    return ResultsResponse(
        job_id=job_id,
        video_id=job.video_id,
        video_title=job.video_title,
        youtube_url=job.youtube_url,
        channel_title=getattr(job, 'channel_title', None),
        view_count=getattr(job, 'view_count', None),
        like_count=getattr(job, 'like_count', None),
        total_comments=len(comments),
        sentiment_summary=SentimentSummary(**counts),
        topics=[TopicOut(
            topic_id=t.topic_id, label=t.label,
            keywords=json.loads(t.keywords_json) if t.keywords_json else [],
            comment_count=t.comment_count, positive_count=t.positive_count,
            neutral_count=t.neutral_count, negative_count=t.negative_count,
        ) for t in topics],
        comments=[CommentOut(
            id=c.id, original_text=c.original_text, clean_text=c.clean_text,
            sentiment_label=c.sentiment_label, sentiment_score=c.sentiment_score,
            vader_label=c.vader_label, vader_compound=c.vader_compound,
            is_toxic=c.is_toxic, toxicity_json=c.toxicity_json,
            topic_id=c.topic_id, lang=c.lang, published_at=c.published_at,
        ) for c in comments],
    )


# ── Job history ───────────────────────────────────────────────────────────────

@app.get("/jobs", response_model=JobListResponse)
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).all()
    return JobListResponse(
        jobs=[JobSummary(
            id=j.id, youtube_url=j.youtube_url, video_title=j.video_title,
            status=j.status, progress=j.progress,
            comment_count=j.comment_count or 0, created_at=j.created_at,
        ) for j in jobs],
        total=len(jobs),
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, db: Session = Depends(get_db)):
    import shutil
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    db.delete(job)
    db.commit()
    job_data_dir = os.path.join(DATA_DIR, job_id)
    if os.path.isdir(job_data_dir):
        shutil.rmtree(job_data_dir, ignore_errors=True)
    return {"deleted": job_id}


# ── Ollama model selector ─────────────────────────────────────────────────────

@app.get("/ollama/models")
def list_ollama_models():
    import requests as _req
    from .pipeline.rag import OLLAMA_HOST, OLLAMA_MODEL
    try:
        resp = _req.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return {"models": models, "default": OLLAMA_MODEL, "error": None}
    except _req.exceptions.ConnectionError:
        return {"models": [], "default": OLLAMA_MODEL,
                "error": "Ollama not running — start it with: ollama serve"}
    except Exception as e:
        return {"models": [], "default": OLLAMA_MODEL, "error": str(e)}


# ── Chat / RAG ────────────────────────────────────────────────────────────────

@app.post("/chat/{job_id}", response_model=ChatResponse)
def chat(job_id: str, request: ChatRequest, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")
    from .pipeline.rag import query_rag
    try:
        result = query_rag(job_id, request.question, model=request.model or None)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return ChatResponse(
        answer=result["answer"],
        sources=[SourceComment(**s) for s in result["sources"]],
    )


# ── NER ───────────────────────────────────────────────────────────────────────

@app.get("/ner/{job_id}")
def get_entities(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")
    from .pipeline.ner import extract_entities
    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    return extract_entities(comments)


# ── UMAP scatter plot ─────────────────────────────────────────────────────────

@app.get("/umap/{job_id}")
def get_umap(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")
    from .pipeline.umap_plot import compute_2d_projection
    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    try:
        return compute_2d_projection(job_id, comments)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── Export ────────────────────────────────────────────────────────────────────

@app.get("/export/{job_id}/excel")
def export_excel(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")
    from .pipeline.export import generate_excel_report
    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    topics   = db.query(Topic).filter(Topic.job_id   == job_id).all()
    buf = generate_excel_report(job, comments, topics)
    safe = "".join(c if c.isalnum() or c in " -_" else ""
                   for c in (job.video_title or "voxtube"))
    safe = safe.strip().replace(" ", "_")[:50] or "voxtube"
    filename = f"{safe}_{job_id[:8]}.xlsx"
    return StreamingResponse(buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/export/{job_id}/pdf")
def export_pdf(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Not complete. Status: {job.status}")
    from .pipeline.pdf_export import generate_pdf_report
    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    topics   = db.query(Topic).filter(Topic.job_id   == job_id).all()
    buf = generate_pdf_report(job, comments, topics)
    safe = "".join(c if c.isalnum() or c in " -_" else ""
                   for c in (job.video_title or "voxtube"))
    safe = safe.strip().replace(" ", "_")[:50] or "voxtube"
    filename = f"{safe}_{job_id[:8]}.pdf"
    return StreamingResponse(buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.get("/evaluate", response_model=EvaluationResponse)
def evaluate():
    from .pipeline.evaluate import run_evaluation
    try:
        result = run_evaluation()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    xlm = MetricsResult(**result["xlm_roberta"]) if result["xlm_roberta"] else None
    return EvaluationResponse(
        total_samples=result["total_samples"],
        label_distribution=result["label_distribution"],
        xlm_roberta=xlm,
        vader=MetricsResult(**result["vader"]),
        note=result["note"],
    )
