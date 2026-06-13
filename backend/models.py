import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from .database import Base


def _uuid():
    return str(uuid.uuid4())


class Job(Base):
    __tablename__ = "jobs"

    id            = Column(String, primary_key=True, default=_uuid)
    youtube_url   = Column(String, nullable=False)
    video_id      = Column(String, nullable=True)
    video_title   = Column(String, nullable=True)
    # pending → fetching → preprocessing → analyzing → building_topics → building_rag → done | failed
    status        = Column(String, default="pending")
    progress      = Column(Integer, default=0)        # 0-100
    error_message = Column(Text, nullable=True)
    comment_count = Column(Integer, default=0)
    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    comments = relationship("Comment", back_populates="job", cascade="all, delete-orphan")
    topics   = relationship("Topic",   back_populates="job", cascade="all, delete-orphan")


class Comment(Base):
    __tablename__ = "comments"

    id            = Column(String, primary_key=True, default=_uuid)
    job_id        = Column(String, ForeignKey("jobs.id"), nullable=False)
    original_text = Column(Text, nullable=False)
    clean_text    = Column(Text, nullable=True)

    # XLM-RoBERTa
    sentiment_label = Column(String, nullable=True)   # positive / neutral / negative
    sentiment_score = Column(Float,  nullable=True)   # confidence

    # VADER baseline
    vader_label    = Column(String, nullable=True)
    vader_compound = Column(Float,  nullable=True)

    # ToxicBERT
    is_toxic      = Column(Integer, default=0)        # 0 or 1
    toxicity_json = Column(Text, nullable=True)       # JSON: {toxic, severe_toxic, ...}

    # BERTopic
    topic_id = Column(Integer, nullable=True)         # -1 = outlier

    # Language detection
    lang = Column(String, nullable=True)              # 'nepali' | 'english' | 'neplish'

    # Timestamp
    published_at = Column(DateTime, nullable=True)    # when the comment was posted

    parent_id = Column(String, nullable=True)   # None = top-level, str = reply

    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="comments")


class Topic(Base):
    __tablename__ = "topics"

    id       = Column(String,  primary_key=True, default=_uuid)
    job_id   = Column(String,  ForeignKey("jobs.id"), nullable=False)
    topic_id = Column(Integer, nullable=False)         # BERTopic topic number
    label    = Column(String,  nullable=True)
    keywords_json = Column(Text, nullable=True)        # JSON list of top keywords

    comment_count    = Column(Integer, default=0)
    positive_count   = Column(Integer, default=0)
    neutral_count    = Column(Integer, default=0)
    negative_count   = Column(Integer, default=0)

    job = relationship("Job", back_populates="topics")
