import os
import re
import socket
import time
from typing import Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

_URL_PATTERNS = [
    r"(?:v=)([a-zA-Z0-9_-]{11})",
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    r"(?:shorts/)([a-zA-Z0-9_-]{11})",
    r"(?:embed/)([a-zA-Z0-9_-]{11})",
]

def extract_video_id(url: str) -> Optional[str]:
    for pattern in _URL_PATTERNS:
        match = re.search(pattern, url)
        if match: return match.group(1)
    return None


def _build_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in your .env file.")
    return build("youtube", "v3", developerKey=api_key)


# ── Retry logic for transient failures ────────────────────────────────────────
# Retries server errors (500/502/503/504) and network blips with exponential
# backoff (1s, 2s, 4s). Does NOT retry permanent failures — quota exceeded,
# comments disabled, video not found — since retrying those can't help and
# would just burn quota/time before the caller sees the real error.

_RETRYABLE_STATUS = {500, 502, 503, 504}
_MAX_RETRIES       = 3
_BASE_DELAY        = 1.0   # seconds


def _execute_with_retry(request):
    """Execute a googleapiclient request, retrying transient failures only."""
    last_exception: Exception | None = None

    for attempt in range(_MAX_RETRIES):
        try:
            return request.execute()

        except HttpError as e:
            if e.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                last_exception = e
                time.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            raise   # permanent error, or retries exhausted — let the caller handle it

        except (socket.timeout, ConnectionError, TimeoutError) as e:
            if attempt < _MAX_RETRIES - 1:
                last_exception = e
                time.sleep(_BASE_DELAY * (2 ** attempt))
                continue
            raise ValueError(
                f"Could not reach the YouTube API after {_MAX_RETRIES} attempts "
                f"(network error: {e}). Check your connection and try again."
            )

    if last_exception:
        raise last_exception   # pragma: no cover — unreachable in practice


def _get_video_metadata(client, video_id: str) -> dict:
    request  = client.videos().list(part="snippet,statistics", id=video_id)
    response = _execute_with_retry(request)
    items = response.get("items", [])
    if not items:
        return {"title": "Unknown Video", "channel_title": None,
                "view_count": None, "like_count": None}

    snippet    = items[0].get("snippet", {})
    statistics = items[0].get("statistics", {})

    def _int(val):
        try: return int(val)
        except: return None

    return {
        "title":         snippet.get("title", "Unknown Video"),
        "channel_title": snippet.get("channelTitle"),
        "view_count":    _int(statistics.get("viewCount")),
        "like_count":    _int(statistics.get("likeCount")),
    }


def fetch_comments(youtube_url: str, max_comments: int = 200) -> dict:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Could not extract a video ID from: {youtube_url}")

    client = _build_client()
    meta   = _get_video_metadata(client, video_id)

    comments        = []
    next_page_token = None

    while len(comments) < max_comments:
        batch = min(100, max_comments - len(comments))
        try:
            request = client.commentThreads().list(
                part="snippet", videoId=video_id, maxResults=batch,
                pageToken=next_page_token, textFormat="plainText", order="relevance",
            )
            response = _execute_with_retry(request)
        except HttpError as e:
            reason = e.error_details[0].get("reason", "") if e.error_details else ""
            if e.status_code == 403:
                if reason == "commentsDisabled":
                    raise ValueError("Comments are disabled for this video.")
                if reason == "quotaExceeded":
                    raise ValueError("YouTube API quota exceeded. Try again tomorrow.")
                raise ValueError(f"YouTube API access denied: {e}")
            if e.status_code == 404:
                raise ValueError("Video not found. Check the URL.")
            raise

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text    = snippet["textDisplay"]
            pub     = snippet.get("publishedAt")
            if text.strip():
                comments.append({"text": text, "published_at": pub})

        next_page_token = response.get("nextPageToken")
        if not next_page_token: break

    return {
        "video_id":      video_id,
        "video_title":   meta["title"],
        "channel_title": meta["channel_title"],
        "view_count":    meta["view_count"],
        "like_count":    meta["like_count"],
        "comments":      comments,
    }
