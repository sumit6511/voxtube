"""
VoxTube - Main Streamlit Application
Interactive dashboard for YouTube comment analytics
"""

import streamlit as ts
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.youtube_extractor import YouTubeExtractor
from src.preprocessor import TextPreprocessor
from src.sentiment_analyzer import SentimentAnalyzer, AspectBasedSentimentAnalyzer
from src.toxicity_detector import ToxicityDetector
from src.topic_modeler import CommentTopicAnalyzer
from src.rag_chat import RAGChatbot
from src.visualizations import Visualizer
from src.report_generator import ReportGenerator
from config import PAGE_TITLE, PAGE_ICON, MAX_COMMENTS

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    .positive { color: #2ecc71; }
    .neutral { color: #95a5a6; }
    .negative { color: #e74c3c; }
    .toxic { color: #c0392b; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'comments' not in st.session_state:
    st.session_state.comments = []
if 'metadata' not in st.session_state:
    st.session_state.metadata = {}
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.markdown('<p class="main-header">📊 VoxTube</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multidimensional Sentiment Analysis & Topic Modeling for YouTube Comments</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys
    with st.expander("API Keys", expanded=False):
        youtube_api_key = st.text_input("YouTube API Key", type="password", 
                                        value=os.getenv("YOUTUBE_API_KEY", ""))
        gemini_api_key = st.text_input("Gemini API Key (for Chat)", type="password",
                                       value=os.getenv("GEMINI_API_KEY", ""))
    
    # Analysis Settings
    st.subheader("Analysis Settings")
    max_comments = st.slider("Max Comments", 100, MAX_COMMENTS, 1000, 100)
    
    st.subheader("Features")
    enable_sentiment = st.checkbox("Sentiment Analysis", value=True)
    enable_toxicity = st.checkbox("Toxicity Detection", value=True)
    enable_topics = st.checkbox("Topic Modeling", value=True)
    enable_chat = st.checkbox("AI Chat (RAG)", value=True)
    
    st.markdown("---")
    st.markdown("**About VoxTube**")
    st.markdown("""
    VoxTube analyzes YouTube comments using:
    - 🧠 XLM-RoBERTa for sentiment analysis
    - 🛡️ RoBERTa for toxicity detection
    - 📊 BERTopic for topic modeling
    - 🤖 Gemini for AI chat
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📥 Data Input", "📈 Analytics", "💬 AI Chat", "📄 Report"])

# Tab 1: Data Input
with tab1:
    st.header("Enter YouTube Video URL")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_url = st.text_input("Video URL", placeholder="https://www.youtube.com/watch?v=...")
    
    with col2:
        st.write("")
        st.write("")
        analyze_button = st.button("🚀 Analyze Comments", type="primary", use_container_width=True)
    
    if analyze_button and video_url:
        if not youtube_api_key:
            st.error("Please enter your YouTube API Key in the sidebar")
        else:
            with st.spinner("Fetching and analyzing comments... This may take a few minutes."):
                try:
                    # Step 1: Extract comments
                    st.info("Step 1/5: Fetching comments from YouTube...")
                    extractor = YouTubeExtractor(api_key=youtube_api_key)
                    result = extractor.analyze_video(video_url, max_comments=max_comments)
                    
                    st.session_state.metadata = result['metadata']
                    comments = result['comments']
                    
                    st.success(f"Fetched {len(comments)} comments!")
                    
                    # Step 2: Preprocess
                    st.info("Step 2/5: Preprocessing comments...")
                    preprocessor = TextPreprocessor()
                    comments = preprocessor.preprocess_comments(comments)
                    
                    # Step 3: Sentiment Analysis
                    if enable_sentiment:
                        st.info("Step 3/5: Analyzing sentiment...")
                        sentiment_analyzer = SentimentAnalyzer()
                        comments = sentiment_analyzer.analyze_comments(comments)
                    
                    # Step 4: Toxicity Detection
                    if enable_toxicity:
                        st.info("Step 4/5: Detecting toxic content...")
                        toxicity_detector = ToxicityDetector()
                        comments = toxicity_detector.analyze_comments(comments)
                    
                    # Step 5: Topic Modeling
                    if enable_topics:
                        st.info("Step 5/5: Extracting topics...")
                        topic_analyzer = CommentTopicAnalyzer()
                        comments = topic_analyzer.analyze_comments(comments)
                    
                    # Store results
                    st.session_state.comments = comments
                    st.session_state.analysis_done = True
                    
                    # Initialize chatbot if enabled
                    if enable_chat and gemini_api_key:
                        st.info("Initializing AI chat...")
                        st.session_state.chatbot = RAGChatbot(api_key=gemini_api_key)
                        st.session_state.chatbot.ingest_comments(comments)
                    
                    st.success("Analysis complete! Check the Analytics tab.")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Show video info if available
    if st.session_state.metadata:
        st.subheader("Video Information")
        meta = st.session_state.metadata
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Title", meta.get('title', 'N/A')[:30] + "...")
        with col2:
            st.metric("Channel", meta.get('channel_title', 'N/A'))
        with col3:
            st.metric("Views", f"{meta.get('view_count', 0):,}")
        with col4:
            st.metric("Likes", f"{meta.get('like_count', 0):,}")

# Tab 2: Analytics
with tab2:
    if not st.session_state.analysis_done:
        st.info("Please analyze a video first in the Data Input tab.")
    else:
        comments = st.session_state.comments
        visualizer = Visualizer()
        
        # Summary Stats
        st.subheader("📊 Summary Statistics")
        
        # Calculate stats
        total = len(comments)
        sentiments = [c.get('sentiment', 'neutral') for c in comments]
        positive = sentiments.count('positive')
        neutral = sentiments.count('neutral')
        negative = sentiments.count('negative')
        toxic = sum(1 for c in comments if c.get('is_toxic', False))
        total_likes = sum(c.get('like_count', 0) for c in comments)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total:,}</div>
                <div class="stat-label">Comments Analyzed</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value positive">{positive/total*100:.1f}%</div>
                <div class="stat-label">Positive Sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value negative">{negative/total*100:.1f}%</div>
                <div class="stat-label">Negative Sentiment</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value toxic">{toxic/total*100:.1f}%</div>
                <div class="stat-label">Toxicity Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total_likes:,}</div>
                <div class="stat-label">Total Likes</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                visualizer.create_sentiment_pie({
                    'positive': positive,
                    'neutral': neutral,
                    'negative': negative
                }),
                use_container_width=True
            )
        
        with col2:
            st.plotly_chart(
                visualizer.create_sentiment_bar({
                    'positive': positive,
                    'neutral': neutral,
                    'negative': negative
                }),
                use_container_width=True
            )
        
        # Toxicity Chart
        if enable_toxicity:
            st.subheader("🛡️ Toxicity Analysis")
            hate = sum(1 for c in comments if c.get('is_hate', False))
            offensive = sum(1 for c in comments if c.get('is_offensive', False))
            normal = total - hate - offensive
            
            st.plotly_chart(
                visualizer.create_toxicity_chart({
                    'hate_count': hate,
                    'offensive_count': offensive,
                    'normal_count': normal
                }),
                use_container_width=True
            )
        
        # Topic Analysis
        if enable_topics:
            st.subheader("📚 Topic Analysis")
            
            # Get topic distribution
            topic_counts = {}
            for c in comments:
                topic = c.get('topic_name', 'Unknown')
                if topic != 'Outliers':
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            if topic_counts:
                topic_df = pd.DataFrame([
                    {'Topic': k, 'Comments': v}
                    for k, v in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                ])
                
                st.bar_chart(topic_df.set_index('Topic'))
            
            # Word Cloud
            st.subheader("☁️ Word Cloud")
            texts = [c.get('cleaned_text', '') for c in comments if c.get('cleaned_text')]
            if texts:
                wordcloud_b64 = visualizer.create_wordcloud(texts, "Most Common Words")
                if wordcloud_b64:
                    st.image(f"data:image/png;base64,{wordcloud_b64}")
        
        # Comment Explorer
        st.subheader("🔍 Comment Explorer")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment",
                ["All", "Positive", "Neutral", "Negative"]
            )
        
        with col2:
            toxicity_filter = st.selectbox(
                "Filter by Toxicity",
                ["All", "Normal", "Offensive", "Hate Speech"]
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                ["Likes (High to Low)", "Newest First", "Confidence"]
            )
        
        # Apply filters
        filtered = comments.copy()
        
        if sentiment_filter != "All":
            filtered = [c for c in filtered if c.get('sentiment') == sentiment_filter.lower()]
        
        if toxicity_filter == "Hate Speech":
            filtered = [c for c in filtered if c.get('is_hate', False)]
        elif toxicity_filter == "Offensive":
            filtered = [c for c in filtered if c.get('is_offensive', False)]
        elif toxicity_filter == "Normal":
            filtered = [c for c in filtered if not c.get('is_toxic', False)]
        
        # Sort
        if sort_by == "Likes (High to Low)":
            filtered = sorted(filtered, key=lambda x: x.get('like_count', 0), reverse=True)
        elif sort_by == "Newest First":
            filtered = sorted(filtered, key=lambda x: x.get('published_at', ''), reverse=True)
        elif sort_by == "Confidence":
            filtered = sorted(filtered, key=lambda x: x.get('sentiment_confidence', 0), reverse=True)
        
        # Display comments
        st.write(f"Showing {len(filtered)} comments")
        
        for comment in filtered[:20]:  # Show top 20
            sentiment = comment.get('sentiment', 'neutral')
            sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😠"
            
            with st.expander(f"{sentiment_emoji} {comment.get('author', 'Anonymous')} - {comment.get('like_count', 0)} likes"):
                st.write(comment.get('text', ''))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"Sentiment: {sentiment.capitalize()} ({comment.get('sentiment_confidence', 0):.2f})")
                with col2:
                    if comment.get('is_toxic', False):
                        st.caption(f"⚠️ Toxic: {comment.get('toxicity', 'Unknown')}")
                with col3:
                    if comment.get('topic_name'):
                        st.caption(f"Topic: {comment.get('topic_name')}")

# Tab 3: AI Chat
with tab3:
    if not st.session_state.analysis_done:
        st.info("Please analyze a video first in the Data Input tab.")
    elif not st.session_state.chatbot:
        st.info("AI Chat is disabled. Enable it in the sidebar and re-analyze.")
    else:
        st.header("💬 Ask Questions About Your Comments")
        
        # Suggested questions
        st.subheader("Suggested Questions")
        suggested = st.session_state.chatbot.get_suggested_questions()
        cols = st.columns(2)
        for i, q in enumerate(suggested[:4]):
            with cols[i % 2]:
                if st.button(q, key=f"suggested_{i}"):
                    st.session_state.current_query = q
        
        # Chat interface
        st.markdown("---")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"**You:** {msg['content']}")
            else:
                st.markdown(f"**VoxTube AI:** {msg['content']}")
        
        # Input
        query = st.text_input("Your question:", 
                             value=st.session_state.get('current_query', ''),
                             placeholder="e.g., What do viewers like most about this video?")
        
        if st.button("Send", type="primary"):
            if query:
                with st.spinner("Thinking..."):
                    # Add user message to history
                    st.session_state.chat_history.append({'role': 'user', 'content': query})
                    
                    # Get response
                    response = st.session_state.chatbot.chat(query)
                    
                    # Add assistant message to history
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response['answer']})
                    
                    # Clear current query
                    if 'current_query' in st.session_state:
                        del st.session_state.current_query
                    
                    st.rerun()
        
        # Show sources for last response
        if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'assistant':
            with st.expander("View Sources"):
                st.caption("Comments used to generate this response:")
                # We'd need to store sources in history for this to work properly

# Tab 4: Report
with tab4:
    if not st.session_state.analysis_done:
        st.info("Please analyze a video first in the Data Input tab.")
    else:
        st.header("📄 Generate Report")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    generator = ReportGenerator()
                    
                    # Calculate stats
                    comments = st.session_state.comments
                    sentiments = [c.get('sentiment', 'neutral') for c in comments]
                    sentiment_stats = {
                        'total': len(comments),
                        'positive': sentiments.count('positive'),
                        'neutral': sentiments.count('neutral'),
                        'negative': sentiments.count('negative'),
                        'positive_pct': sentiments.count('positive') / len(comments) * 100,
                        'neutral_pct': sentiments.count('neutral') / len(comments) * 100,
                        'negative_pct': sentiments.count('negative') / len(comments) * 100
                    }
                    
                    toxic = sum(1 for c in comments if c.get('is_toxic', False))
                    hate = sum(1 for c in comments if c.get('is_hate', False))
                    offensive = sum(1 for c in comments if c.get('is_offensive', False))
                    toxicity_stats = {
                        'total': len(comments),
                        'toxic_count': toxic,
                        'hate_count': hate,
                        'offensive_count': offensive,
                        'normal_count': len(comments) - toxic,
                        'toxicity_rate': toxic / len(comments) * 100
                    }
                    
                    topic_stats = {
                        'topic_distribution': [(c.get('topic', -1), 1) for c in comments if c.get('topic', -1) != -1]
                    }
                    
                    pdf_bytes = generator.generate_report(
                        st.session_state.metadata,
                        comments,
                        sentiment_stats,
                        toxicity_stats,
                        topic_stats
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"voxtube_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    
                    st.success("Report generated successfully!")
                    
                except Exception as e:
                    st.error(f"Error generating report: {str(e)}")

# Footer
st.markdown("---")
st.caption("VoxTube - Final Year Project | St. Xavier's College | Developed by Sumit Kumar Sah & Utsav Adhikari")
