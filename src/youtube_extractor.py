"""
YouTube Data Extraction Module
Handles fetching comments and metadata from YouTube API
"""

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import time
import re
from typing import List, Dict, Optional
from config import YOUTUBE_API_KEY, MAX_COMMENTS


class YouTubeExtractor:
    """Extract comments and metadata from YouTube videos"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or YOUTUBE_API_KEY
        if not self.api_key:
            raise ValueError("YouTube API key is required")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not extract video ID from URL: {url}")
    
    def get_video_metadata(self, video_id: str) -> Dict:
        """Fetch video metadata (title, description, statistics)"""
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=video_id
            ).execute()
            
            if not response['items']:
                raise ValueError(f"Video not found: {video_id}")
            
            video = response['items'][0]
            snippet = video['snippet']
            stats = video['statistics']
            
            return {
                'video_id': video_id,
                'title': snippet['title'],
                'description': snippet.get('description', ''),
                'channel_title': snippet['channelTitle'],
                'published_at': snippet['publishedAt'],
                'view_count': int(stats.get('viewCount', 0)),
                'like_count': int(stats.get('likeCount', 0)),
                'comment_count': int(stats.get('commentCount', 0)),
                'thumbnail_url': snippet['thumbnails']['high']['url'] if 'high' in snippet['thumbnails'] else ''
            }
        except HttpError as e:
            raise Exception(f"YouTube API error: {str(e)}")
    
    def get_comments(self, video_id: str, max_results: int = MAX_COMMENTS) -> List[Dict]:
        """
        Fetch top-level comments for a video
        Implements pagination and exponential backoff for quota management
        """
        comments = []
        next_page_token = None
        retry_count = 0
        max_retries = 3
        
        while len(comments) < max_results:
            try:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_results - len(comments)),
                    pageToken=next_page_token,
                    order='relevance',  # Get most relevant comments first
                    textFormat='plainText'
                )
                
                response = request.execute()
                
                for item in response['items']:
                    snippet = item['snippet']['topLevelComment']['snippet']
                    comment_data = {
                        'comment_id': item['snippet']['topLevelComment']['id'],
                        'text': snippet['textDisplay'],
                        'author': snippet['authorDisplayName'],
                        'author_channel_id': snippet.get('authorChannelId', {}).get('value', ''),
                        'like_count': snippet.get('likeCount', 0),
                        'published_at': snippet['publishedAt'],
                        'updated_at': snippet.get('updatedAt', snippet['publishedAt']),
                        'total_reply_count': item['snippet'].get('totalReplyCount', 0),
                        'is_reply': False
                    }
                    comments.append(comment_data)
                    
                    # Fetch replies if available (limited to save quota)
                    if item['snippet'].get('totalReplyCount', 0) > 0 and 'replies' in item:
                        for reply in item['replies']['comments'][:3]:  # Limit replies
                            reply_snippet = reply['snippet']
                            comments.append({
                                'comment_id': reply['id'],
                                'text': reply_snippet['textDisplay'],
                                'author': reply_snippet['authorDisplayName'],
                                'author_channel_id': reply_snippet.get('authorChannelId', {}).get('value', ''),
                                'like_count': reply_snippet.get('likeCount', 0),
                                'published_at': reply_snippet['publishedAt'],
                                'updated_at': reply_snippet.get('updatedAt', reply_snippet['publishedAt']),
                                'total_reply_count': 0,
                                'is_reply': True,
                                'parent_id': item['snippet']['topLevelComment']['id']
                            })
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                retry_count = 0  # Reset retry count on success
                
            except HttpError as e:
                error_code = e.resp.status
                if error_code == 403 and 'quotaExceeded' in str(e):
                    if retry_count < max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        time.sleep(wait_time)
                        retry_count += 1
                        continue
                    else:
                        print(f"Quota exceeded. Fetched {len(comments)} comments.")
                        break
                elif error_code == 403 and 'commentsDisabled' in str(e):
                    raise Exception("Comments are disabled for this video")
                else:
                    raise Exception(f"YouTube API error: {str(e)}")
        
        return comments[:max_results]
    
    def analyze_video(self, url: str, max_comments: int = MAX_COMMENTS) -> Dict:
        """
        Complete analysis pipeline for a YouTube video
        Returns video metadata and comments
        """
        video_id = self.extract_video_id(url)
        metadata = self.get_video_metadata(video_id)
        comments = self.get_comments(video_id, max_comments)
        
        return {
            'metadata': metadata,
            'comments': comments,
            'total_fetched': len(comments)
        }
