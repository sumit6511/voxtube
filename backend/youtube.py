import os
import re
from typing import Optional
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


# ── Video ID extraction ───────────────────────────────────────────────────────

_URL_PATTERNS = [
    r"(?:v=)([a-zA-Z0-9_-]{11})",           # youtube.com/watch?v=ID
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",   # youtu.be/ID
    r"(?:shorts/)([a-zA-Z0-9_-]{11})",      # youtube.com/shorts/ID
    r"(?:embed/)([a-zA-Z0-9_-]{11})",       # youtube.com/embed/ID
]

def extract_video_id(url: str) -> Optional[str]:
    """Return the 11-char video ID from any common YouTube URL, or None."""
    for pattern in _URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── API helpers ───────────────────────────────────────────────────────────────

def _build_client():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is not set in your .env file.")
    return build("youtube", "v3", developerKey=api_key)


def _get_video_title(client, video_id: str) -> str:
    response = client.videos().list(
        part="snippet",
        id=video_id,
    ).execute()

    items = response.get("items", [])
    if not items:
        return "Unknown Video"
    return items[0]["snippet"]["title"]


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_comments(youtube_url: str, max_comments: int = 200) -> dict:
    """
    Fetch top-level comments from a YouTube video.

    Args:
        youtube_url:  Any valid YouTube URL format.
        max_comments: Maximum number of comments to collect (default 200).

    Returns:
        {
            "video_id":    str,
            "video_title": str,
            "comments":    [str, ...]   # plain-text comment strings
        }

    Raises:
        ValueError: Bad URL, comments disabled, or missing API key.
        HttpError:  Unhandled YouTube API error.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError(f"Could not extract a video ID from: {youtube_url}")

    client = _build_client()
    video_title = _get_video_title(client, video_id)

    comments       = []
    next_page_token = None

    while len(comments) < max_comments:
        batch = min(100, max_comments - len(comments))  # API hard-cap is 100

        try:
            response = client.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=batch,
                pageToken=next_page_token,
                textFormat="plainText",
                order="relevance",
            ).execute()

        except HttpError as e:
            reason = ""
            if e.error_details:
                reason = e.error_details[0].get("reason", "")

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
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            if text.strip():            # skip empty/whitespace comments
                comments.append(text)

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break                       # no more pages available

    return {
        "video_id":    video_id,
        "video_title": video_title,
        "comments":    comments,
    }
