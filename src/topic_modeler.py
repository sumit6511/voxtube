"""
Topic Modeling Module
Uses BERTopic for extracting themes from YouTube comments
"""

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from typing import List, Dict, Optional
import umap
import hdbscan
from config import EMBEDDING_MODEL, TOPIC_MIN_DOCS


class TopicModeler:
    """
    Topic modeling using BERTopic
    Extracts themes and discussion topics from comments
    """
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model_name = embedding_model or EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.topic_model = None
        self.topics = None
        self.probs = None
    
    def fit_transform(self, texts: List[str], 
                      min_topic_size: int = 10,
                      nr_topics: str = 'auto') -> Dict:
        """
        Fit topic model and transform texts
        
        Args:
            texts: List of preprocessed texts
            min_topic_size: Minimum number of documents per topic
            nr_topics: Number of topics ('auto' or specific number)
        
        Returns:
            Dictionary with topics, probabilities, and topic info
        """
        # Filter out empty texts
        valid_texts = [t for t in texts if t and len(t.strip()) > 5]
        
        if len(valid_texts) < min_topic_size * 2:
            return {
                'topics': [-1] * len(texts),
                'probs': None,
                'topic_info': pd.DataFrame(),
                'topic_names': {}
            }
        
        # Configure UMAP for dimensionality reduction
        umap_model = umap.UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # Configure HDBSCAN for clustering
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Configure vectorizer for topic representation
        vectorizer = CountVectorizer(
            stop_words='english',
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        # Create and fit BERTopic model
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            nr_topics=nr_topics,
            verbose=True
        )
        
        # Fit and transform
        self.topics, self.probs = self.topic_model.fit_transform(valid_texts)
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Create topic name mapping
        topic_names = {}
        for topic_id in topic_info['Topic']:
            if topic_id != -1:
                keywords = self.topic_model.get_topic(topic_id)
                if keywords:
                    # Create name from top 3 keywords
                    name = ' | '.join([word for word, _ in keywords[:3]])
                    topic_names[topic_id] = name
                else:
                    topic_names[topic_id] = f"Topic {topic_id}"
            else:
                topic_names[topic_id] = "Outliers"
        
        return {
            'topics': self.topics,
            'probs': self.probs,
            'topic_info': topic_info,
            'topic_names': topic_names,
            'valid_texts': valid_texts
        }
    
    def get_topic_keywords(self, topic_id: int, top_n: int = 10) -> List[tuple]:
        """Get top keywords for a specific topic"""
        if self.topic_model is None:
            return []
        
        keywords = self.topic_model.get_topic(topic_id)
        if keywords:
            return keywords[:top_n]
        return []
    
    def get_document_topics(self, text: str) -> Dict:
        """Get topic predictions for a new document"""
        if self.topic_model is None:
            return {'topic': -1, 'confidence': 0.0}
        
        topic, prob = self.topic_model.transform([text])
        return {
            'topic': topic[0],
            'confidence': float(prob[0]) if prob is not None else 0.0
        }
    
    def visualize_topics(self):
        """Generate topic visualization"""
        if self.topic_model is None:
            return None
        
        return self.topic_model.visualize_topics()
    
    def visualize_barchart(self, top_n_topics: int = 8):
        """Generate bar chart of top topics"""
        if self.topic_model is None:
            return None
        
        return self.topic_model.visualize_barchart(top_n_topics=top_n_topics)
    
    def visualize_heatmap(self):
        """Generate topic similarity heatmap"""
        if self.topic_model is None:
            return None
        
        return self.topic_model.visualize_heatmap()
    
    def get_topic_stats(self, comments: List[Dict]) -> Dict:
        """Get statistics about topics in comments"""
        if not comments or 'topic' not in comments[0]:
            return {}
        
        topic_counts = {}
        for comment in comments:
            topic = comment.get('topic', -1)
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by count
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_topics': len([t for t in topic_counts.keys() if t != -1]),
            'outliers': topic_counts.get(-1, 0),
            'topic_distribution': sorted_topics
        }


class CommentTopicAnalyzer:
    """
    High-level interface for topic analysis of comments
    """
    
    def __init__(self):
        self.modeler = TopicModeler()
    
    def analyze_comments(self, comments: List[Dict], 
                        text_key: str = 'cleaned_text') -> List[Dict]:
        """
        Analyze topics for a list of comments
        
        Args:
            comments: List of comment dictionaries
            text_key: Key containing the text to analyze
        
        Returns:
            List of comments with added topic fields
        """
        texts = [comment.get(text_key, '') for comment in comments]
        
        # Fit topic model
        result = self.modeler.fit_transform(texts)
        
        # Add topic info to comments
        analyzed_comments = []
        valid_idx = 0
        
        for comment in comments:
            comment_copy = comment.copy()
            text = comment.get(text_key, '')
            
            if text and len(text.strip()) > 5 and valid_idx < len(result['topics']):
                topic_id = result['topics'][valid_idx]
                comment_copy['topic'] = topic_id
                comment_copy['topic_name'] = result['topic_names'].get(topic_id, 'Unknown')
                
                if result['probs'] is not None:
                    comment_copy['topic_confidence'] = float(result['probs'][valid_idx])
                else:
                    comment_copy['topic_confidence'] = 0.0
                
                valid_idx += 1
            else:
                comment_copy['topic'] = -1
                comment_copy['topic_name'] = 'Outliers'
                comment_copy['topic_confidence'] = 0.0
            
            analyzed_comments.append(comment_copy)
        
        return analyzed_comments
    
    def get_topic_wordcloud_data(self, comments: List[Dict]) -> Dict:
        """Get data for word cloud visualization by topic"""
        topic_texts = {}
        
        for comment in comments:
            topic = comment.get('topic', -1)
            text = comment.get('cleaned_text', '')
            
            if topic not in topic_texts:
                topic_texts[topic] = []
            topic_texts[topic].append(text)
        
        # Combine texts per topic
        topic_combined = {}
        for topic, texts in topic_texts.items():
            topic_combined[topic] = ' '.join(texts)
        
        return topic_combined
