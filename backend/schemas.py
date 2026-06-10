from pydantic import BaseModel
from typing import Optional, List


# ── Requests ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    url: str
    max_comments: int = 200   # how many comments to fetch


# ── Responses ─────────────────────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id:        str
    status:        str
    progress:      int
    comment_count: int
    video_title:   Optional[str] = None
    error_message: Optional[str] = None


class CommentOut(BaseModel):
    id:              str
    original_text:   str
    clean_text:      Optional[str]
    sentiment_label: Optional[str]
    sentiment_score: Optional[float]
    vader_label:     Optional[str]
    vader_compound:  Optional[float]
    is_toxic:        int
    toxicity_json:   Optional[str]
    topic_id:        Optional[int]
    lang:            Optional[str] = None  # 'nepali' | 'english' | 'neplish'

    model_config = {"from_attributes": True}


class TopicOut(BaseModel):
    topic_id:       int
    label:          Optional[str]
    keywords:       List[str]
    comment_count:  int
    positive_count: int
    neutral_count:  int
    negative_count: int


class SentimentSummary(BaseModel):
    positive: int
    neutral:  int
    negative: int


class ResultsResponse(BaseModel):
    job_id:            str
    video_title:       Optional[str]
    total_comments:    int
    sentiment_summary: SentimentSummary
    topics:            List[TopicOut]
    comments:          List[CommentOut]


# ── RAG chat ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


class SourceComment(BaseModel):
    id:    str
    text:  str
    score: float


class ChatResponse(BaseModel):
    answer:  str
    sources: List[SourceComment]
