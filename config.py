"""
VoxTube Configuration
Centralized configuration for the application
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Model Configuration
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TOXICITY_MODEL = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Analysis Configuration
MAX_COMMENTS = 5000  # Maximum comments to fetch per video
BATCH_SIZE = 32  # Batch size for model inference
TOPIC_MIN_DOCS = 10  # Minimum documents per topic

# RAG Configuration
VECTOR_DB_PATH = "./vector_db"
TOP_K_RETRIEVAL = 5

# Dashboard Configuration
PAGE_TITLE = "VoxTube - YouTube Comment Analytics"
PAGE_ICON = "📊"

# Aspect Categories for ABSA
ASPECT_CATEGORIES = [
    "Content",
    "Audio Quality",
    "Video Quality",
    "Presentation",
    "Technical Issues",
    "Engagement"
]

# Sentiment Labels
SENTIMENT_LABELS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral"
}

# Toxicity Labels
TOXICITY_LABELS = {
    "hate": "Hate Speech",
    "offensive": "Offensive",
    "normal": "Normal"
}
