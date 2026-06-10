import json
import uuid
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .database import engine, get_db, Base
from .models import Job, Comment, Topic
from .schemas import (
    AnalyzeRequest, AnalyzeResponse,
    JobStatusResponse, ResultsResponse,
    CommentOut, TopicOut, SentimentSummary,
    ChatRequest, ChatResponse, SourceComment,
)

# Create all tables and run lightweight column migrations on startup
Base.metadata.create_all(bind=engine)
from .database import run_migrations
run_migrations(engine)

app = FastAPI(title="VoxTube API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pipeline helpers ──────────────────────────────────────────────────────────

# Progress milestones — frontend can show a labelled progress bar
STAGES = {
    "fetching":       (5,  20),
    "preprocessing":  (20, 35),
    "analyzing":      (35, 55),
    "toxicity":       (55, 70),
    "building_topics":(70, 85),
    "building_rag":   (85, 98),
    "done":           (100, 100),
}

def _set_job(db, job_id: str, status: str, progress: int, **kwargs):
    """Update job status + progress (and any extra fields) in one commit."""
    updates = {"status": status, "progress": progress, **kwargs}
    db.query(Job).filter(Job.id == job_id).update(updates)
    db.commit()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(job_id: str, youtube_url: str, max_comments: int):
    """
    Runs the full NLP pipeline in the background.
    Each step is implemented one by one — wired in as we go.
    """
    from .database import SessionLocal
    from .youtube import fetch_comments

    db = SessionLocal()
    try:

        # ── Step 2: Fetch YouTube comments ────────────────────────────────
        _set_job(db, job_id, "fetching", 5)

        result = fetch_comments(youtube_url, max_comments)

        # Bulk-insert comments with timestamps into DB
        from datetime import datetime as dt

        def _parse_ts(s):
            if not s: return None
            try: return dt.fromisoformat(s.replace('Z', '+00:00'))
            except Exception: return None

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
            comment_count=len(result["comments"]),
        )

        # ── Step 3a: Neplish preprocessing + language detection ──────────
        from .pipeline.preprocessor import preprocess_batch, detect_languages

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        original_texts = [c.original_text for c in comments_in_db]
        clean_texts    = preprocess_batch(original_texts)
        lang_labels    = detect_languages(clean_texts)

        for comment, clean, lang in zip(comments_in_db, clean_texts, lang_labels):
            comment.clean_text = clean
            comment.lang       = lang
        db.commit()

        _set_job(db, job_id, "analyzing", 35)

        # ── Step 3b: XLM-RoBERTa + VADER ────────────────────────────────
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

        # ── Step 3c: ToxicBERT ────────────────────────────────────────────
        from .pipeline.toxicity import detect_toxicity_batch, scores_to_json

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        texts          = [c.clean_text or c.original_text for c in comments_in_db]
        tox_results    = detect_toxicity_batch(texts)

        for comment, res in zip(comments_in_db, tox_results):
            comment.is_toxic      = res["is_toxic"]
            comment.toxicity_json = scores_to_json(res["scores"])
        db.commit()

        _set_job(db, job_id, "building_topics", 70)

        # ── Step 3d: BERTopic ────────────────────────────────────────────
        from .pipeline.topics import run_topic_modeling, aggregate_topic_sentiments

        comments_in_db = db.query(Comment).filter(Comment.job_id == job_id).all()
        clean_texts    = [c.clean_text or c.original_text for c in comments_in_db]
        sent_labels    = [c.sentiment_label or "neutral"   for c in comments_in_db]

        topic_result = run_topic_modeling(clean_texts)
        assignments  = topic_result["topic_assignments"]

        # Save topic_id on each comment
        for comment, tid in zip(comments_in_db, assignments):
            comment.topic_id = tid
        db.commit()

        # Per-topic sentiment distribution (novel contribution)
        sentiment_per_topic = aggregate_topic_sentiments(assignments, sent_labels)

        # Save Topic rows
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

        # ── Step 3e: RAG / FAISS ──────────────────────────────────────────
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
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        youtube_url=request.url,
        status="pending",
        progress=0,
    )
    db.add(job)
    db.commit()

    background_tasks.add_task(run_pipeline, job_id, request.url, request.max_comments)

    return AnalyzeResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def get_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        comment_count=job.comment_count,
        video_title=job.video_title,
        error_message=job.error_message,
    )


@app.get("/results/{job_id}", response_model=ResultsResponse)
def get_results(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(status_code=400, detail=f"Job not complete yet. Status: {job.status}")

    comments = db.query(Comment).filter(Comment.job_id == job_id).all()
    topics   = db.query(Topic).filter(Topic.job_id   == job_id).all()

    # Sentiment summary
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for c in comments:
        if c.sentiment_label in counts:
            counts[c.sentiment_label] += 1

    topics_out = [
        TopicOut(
            topic_id=t.topic_id,
            label=t.label,
            keywords=json.loads(t.keywords_json) if t.keywords_json else [],
            comment_count=t.comment_count,
            positive_count=t.positive_count,
            neutral_count=t.neutral_count,
            negative_count=t.negative_count,
        )
        for t in topics
    ]

    comments_out = [
        CommentOut(
            id=c.id,
            original_text=c.original_text,
            clean_text=c.clean_text,
            sentiment_label=c.sentiment_label,
            sentiment_score=c.sentiment_score,
            vader_label=c.vader_label,
            vader_compound=c.vader_compound,
            is_toxic=c.is_toxic,
            toxicity_json=c.toxicity_json,
            topic_id=c.topic_id,
            lang=c.lang,
            published_at=c.published_at,
        )
        for c in comments
    ]

    return ResultsResponse(
        job_id=job_id,
        video_title=job.video_title,
        total_comments=len(comments),
        sentiment_summary=SentimentSummary(**counts),
        topics=topics_out,
        comments=comments_out,
    )

# ── Chat / RAG endpoint ───────────────────────────────────────────────────────

@app.post("/chat/{job_id}", response_model=ChatResponse)
def chat(job_id: str, request: ChatRequest, db: Session = Depends(get_db)):
    """
    Ask a natural-language question about a video's comment section.
    Retrieves the most relevant comments via FAISS, then uses Gemini
    to generate a grounded answer with citations.
    """
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not complete yet. Current status: {job.status}"
        )

    from .pipeline.rag import query_rag

    try:
        result = query_rag(job_id, request.question)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceComment(**s) for s in result["sources"]],
    )
