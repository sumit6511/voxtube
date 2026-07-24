from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class AnalyzeRequest(BaseModel):
    url: str
    max_comments: int = 200


class AnalyzeResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    job_id:        str
    status:        str
    progress:      int
    comment_count: int
    video_id:      Optional[str] = None
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
    lang:            Optional[str]      = None
    published_at:    Optional[datetime] = None

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
    video_id:          Optional[str] = None
    video_title:       Optional[str] = None
    youtube_url:       Optional[str] = None
    channel_title:     Optional[str] = None
    view_count:        Optional[int] = None
    like_count:        Optional[int] = None
    total_comments:    int
    sentiment_summary: SentimentSummary
    topics:            List[TopicOut]
    comments:          List[CommentOut]


class ChatRequest(BaseModel):
    question: str
    model:    Optional[str] = None


class SourceComment(BaseModel):
    id:    str
    text:  str
    score: float


class ChatResponse(BaseModel):
    answer:  str
    sources: List[SourceComment]


class MetricsResult(BaseModel):
    accuracy:         float
    precision:        float
    recall:           float
    f1:               float
    confusion_matrix: List[List[int]]


class EvaluationResponse(BaseModel):
    total_samples:      int
    label_distribution: dict
    xlm_roberta:        Optional[MetricsResult] = None
    vader:              MetricsResult
    note:               Optional[str] = None
    # Third, optional comparison point — a Nepali-specific sentiment model.
    # Included for research comparison only; the main pipeline still uses
    # XLM-RoBERTa (see sentiment.py for why this isn't a blind swap).
    nepali_model:       Optional[MetricsResult] = None
    nepali_model_note:  Optional[str] = None


class JobSummary(BaseModel):
    id:            str
    youtube_url:   str
    video_title:   Optional[str] = None
    status:        str
    progress:      int
    comment_count: int
    created_at:    Optional[datetime] = None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs:  List[JobSummary]
    total: int
