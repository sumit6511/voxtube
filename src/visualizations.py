"""
Visualization Module
Creates interactive charts and graphs for the dashboard
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
from typing import List, Dict
from collections import Counter
import numpy as np


class Visualizer:
    """Create visualizations for YouTube comment analysis"""
    
    def __init__(self):
        self.color_scheme = {
            'positive': '#2ecc71',
            'neutral': '#95a5a6',
            'negative': '#e74c3c',
            'hate': '#c0392b',
            'offensive': '#e67e22',
            'normal': '#27ae60'
        }
    
    def create_sentiment_pie(self, sentiment_stats: Dict) -> go.Figure:
        """Create pie chart for sentiment distribution"""
        labels = ['Positive', 'Neutral', 'Negative']
        values = [
            sentiment_stats.get('positive', 0),
            sentiment_stats.get('neutral', 0),
            sentiment_stats.get('negative', 0)
        ]
        colors = [self.color_scheme['positive'], 
                  self.color_scheme['neutral'], 
                  self.color_scheme['negative']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=14
        )])
        
        fig.update_layout(
            title_text="Sentiment Distribution",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_sentiment_bar(self, sentiment_stats: Dict) -> go.Figure:
        """Create bar chart for sentiment counts"""
        sentiments = ['Positive', 'Neutral', 'Negative']
        counts = [
            sentiment_stats.get('positive', 0),
            sentiment_stats.get('neutral', 0),
            sentiment_stats.get('negative', 0)
        ]
        colors = [self.color_scheme['positive'], 
                  self.color_scheme['neutral'], 
                  self.color_scheme['negative']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sentiments,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title_text="Comment Count by Sentiment",
            xaxis_title="Sentiment",
            yaxis_title="Number of Comments",
            height=400
        )
        
        return fig
    
    def create_sentiment_timeline(self, time_data: Dict) -> go.Figure:
        """Create timeline chart for sentiment over time"""
        if not time_data:
            return go.Figure()
        
        df = pd.DataFrame([
            {'time': k, 'sentiment': s, 'count': v}
            for k, sentiments in time_data.items()
            for s, v in sentiments.items()
        ])
        
        if df.empty:
            return go.Figure()
        
        # Pivot for stacked area chart
        pivot_df = df.pivot(index='time', columns='sentiment', values='count').fillna(0)
        pivot_df = pivot_df.reindex(columns=['positive', 'neutral', 'negative'], fill_value=0)
        
        fig = go.Figure()
        
        for sentiment, color in [('positive', self.color_scheme['positive']),
                                  ('neutral', self.color_scheme['neutral']),
                                  ('negative', self.color_scheme['negative'])]:
            if sentiment in pivot_df.columns:
                fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=pivot_df[sentiment],
                    mode='lines',
                    name=sentiment.capitalize(),
                    stackgroup='one',
                    fillcolor=color,
                    line=dict(color=color)
                ))
        
        fig.update_layout(
            title_text="Sentiment Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Number of Comments",
            height=450,
            hovermode='x unified'
        )
        
        return fig
    
    def create_toxicity_chart(self, toxicity_stats: Dict) -> go.Figure:
        """Create chart for toxicity statistics"""
        categories = ['Normal', 'Offensive', 'Hate Speech']
        counts = [
            toxicity_stats.get('normal_count', 0),
            toxicity_stats.get('offensive_count', 0),
            toxicity_stats.get('hate_count', 0)
        ]
        colors = [self.color_scheme['normal'], 
                  self.color_scheme['offensive'], 
                  self.color_scheme['hate']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=colors,
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title_text="Toxicity Analysis",
            xaxis_title="Category",
            yaxis_title="Number of Comments",
            height=400
        )
        
        return fig
    
    def create_topic_chart(self, topic_stats: Dict) -> go.Figure:
        """Create chart for topic distribution"""
        if not topic_stats or 'topic_distribution' not in topic_stats:
            return go.Figure()
        
        # Filter out outliers for main chart
        topics = [(t, c) for t, c in topic_stats['topic_distribution'] if t != -1][:10]
        
        if not topics:
            return go.Figure()
        
        topic_ids = [f"Topic {t}" for t, _ in topics]
        counts = [c for _, c in topics]
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_ids,
                y=counts,
                marker_color='steelblue',
                text=counts,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title_text="Top Discussion Topics",
            xaxis_title="Topic",
            yaxis_title="Number of Comments",
            height=400
        )
        
        return fig
    
    def create_engagement_chart(self, comments: List[Dict]) -> go.Figure:
        """Create scatter plot of sentiment vs engagement"""
        df = pd.DataFrame([
            {
                'sentiment': c.get('sentiment', 'neutral'),
                'likes': c.get('like_count', 0),
                'replies': c.get('total_reply_count', 0),
                'text': c.get('text', '')[:50] + '...'
            }
            for c in comments
        ])
        
        if df.empty:
            return go.Figure()
        
        fig = px.scatter(
            df,
            x='likes',
            y='replies',
            color='sentiment',
            color_discrete_map=self.color_scheme,
            hover_data=['text'],
            title='Comment Engagement vs Sentiment',
            labels={'likes': 'Likes', 'replies': 'Replies'}
        )
        
        fig.update_layout(height=450)
        
        return fig
    
    def create_wordcloud(self, texts: List[str], title: str = "Word Cloud") -> str:
        """Generate word cloud image as base64 string"""
        if not texts:
            return ""
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        if len(combined_text.strip()) < 10:
            return ""
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=100,
            contour_width=1,
            contour_color='steelblue'
        ).generate(combined_text)
        
        # Convert to image
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title, fontsize=16, pad=20)
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.read()).decode()
        
        return image_base64
    
    def create_summary_stats(self, comments: List[Dict], metadata: Dict) -> Dict:
        """Create summary statistics cards"""
        total_comments = len(comments)
        
        # Sentiment breakdown
        sentiments = [c.get('sentiment', 'neutral') for c in comments]
        positive_pct = sentiments.count('positive') / total_comments * 100 if total_comments > 0 else 0
        
        # Toxicity
        toxic_count = sum(1 for c in comments if c.get('is_toxic', False))
        toxicity_rate = toxic_count / total_comments * 100 if total_comments > 0 else 0
        
        # Engagement
        total_likes = sum(c.get('like_count', 0) for c in comments)
        avg_likes = total_likes / total_comments if total_comments > 0 else 0
        
        # Top topics
        topics = [c.get('topic_name', 'Unknown') for c in comments if c.get('topic', -1) != -1]
        top_topic = Counter(topics).most_common(1)[0][0] if topics else "N/A"
        
        return {
            'total_comments': total_comments,
            'positive_sentiment_pct': round(positive_pct, 1),
            'toxicity_rate': round(toxicity_rate, 1),
            'total_likes': total_likes,
            'avg_likes': round(avg_likes, 1),
            'top_topic': top_topic,
            'video_title': metadata.get('title', 'Unknown'),
            'view_count': metadata.get('view_count', 0),
            'video_likes': metadata.get('like_count', 0)
        }
    
    def create_comparison_chart(self, videos_data: List[Dict]) -> go.Figure:
        """Create comparison chart for multiple videos"""
        if len(videos_data) < 2:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Comparison', 'Toxicity Comparison',
                          'Engagement Comparison', 'Comment Volume'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        video_titles = [v['metadata']['title'][:30] + '...' for v in videos_data]
        
        # Sentiment data
        positive_pcts = []
        for v in videos_data:
            comments = v['comments']
            if comments:
                pos = sum(1 for c in comments if c.get('sentiment') == 'positive')
                positive_pcts.append(pos / len(comments) * 100)
            else:
                positive_pcts.append(0)
        
        fig.add_trace(
            go.Bar(x=video_titles, y=positive_pcts, name='Positive %', marker_color='green'),
            row=1, col=1
        )
        
        # Toxicity data
        toxic_rates = []
        for v in videos_data:
            comments = v['comments']
            if comments:
                toxic = sum(1 for c in comments if c.get('is_toxic', False))
                toxic_rates.append(toxic / len(comments) * 100)
            else:
                toxic_rates.append(0)
        
        fig.add_trace(
            go.Bar(x=video_titles, y=toxic_rates, name='Toxicity %', marker_color='red'),
            row=1, col=2
        )
        
        # Engagement
        avg_likes = []
        for v in videos_data:
            comments = v['comments']
            if comments:
                avg = sum(c.get('like_count', 0) for c in comments) / len(comments)
                avg_likes.append(avg)
            else:
                avg_likes.append(0)
        
        fig.add_trace(
            go.Bar(x=video_titles, y=avg_likes, name='Avg Likes', marker_color='blue'),
            row=2, col=1
        )
        
        # Comment volume
        comment_counts = [len(v['comments']) for v in videos_data]
        fig.add_trace(
            go.Bar(x=video_titles, y=comment_counts, name='Comments', marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=700, showlegend=False, title_text="Video Comparison")
        
        return fig
