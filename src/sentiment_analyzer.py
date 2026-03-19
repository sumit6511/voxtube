"""
Sentiment Analysis Module
Uses XLM-RoBERTa for multilingual sentiment classification
Supports English and code-mixed languages (Hinglish, Neplish)
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
from config import SENTIMENT_MODEL, BATCH_SIZE


class SentimentAnalyzer:
    """
    Multilingual Sentiment Analyzer using XLM-RoBERTa
    Supports: English, Code-mixed (Hinglish, Nepali-English)
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or SENTIMENT_MODEL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading sentiment model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping for cardiffnlp/twitter-xlm-roberta-base-sentiment
        self.id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
    
    def predict(self, text: str) -> Dict:
        """
        Predict sentiment for a single text
        
        Returns:
            Dictionary with sentiment label, confidence score, and probabilities
        """
        if not text or not isinstance(text, str):
            return {
                'label': 'neutral',
                'confidence': 0.0,
                'probabilities': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get all probabilities
        probs = probabilities[0].cpu().numpy()
        
        return {
            'label': self.id2label[predicted_class],
            'confidence': round(confidence, 4),
            'probabilities': {
                'negative': round(probs[0], 4),
                'neutral': round(probs[1], 4),
                'positive': round(probs[2], 4)
            }
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[Dict]:
        """
        Predict sentiment for multiple texts in batches
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Filter out empty texts
            valid_indices = [j for j, text in enumerate(batch) if text and isinstance(text, str)]
            valid_texts = [batch[j] for j in valid_indices]
            
            if not valid_texts:
                results.extend([{'label': 'neutral', 'confidence': 0.0, 
                                'probabilities': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}}] * len(batch))
                continue
            
            # Tokenize
            inputs = self.tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
                confidences = probabilities.max(dim=1)[0].cpu().numpy()
                all_probs = probabilities.cpu().numpy()
            
            # Create results for valid texts
            batch_results = []
            valid_idx = 0
            for j in range(len(batch)):
                if j in valid_indices:
                    pred_class = predicted_classes[valid_idx]
                    batch_results.append({
                        'label': self.id2label[pred_class],
                        'confidence': round(float(confidences[valid_idx]), 4),
                        'probabilities': {
                            'negative': round(float(all_probs[valid_idx][0]), 4),
                            'neutral': round(float(all_probs[valid_idx][1]), 4),
                            'positive': round(float(all_probs[valid_idx][2]), 4)
                        }
                    })
                    valid_idx += 1
                else:
                    batch_results.append({
                        'label': 'neutral',
                        'confidence': 0.0,
                        'probabilities': {'negative': 0.0, 'neutral': 1.0, 'positive': 0.0}
                    })
            
            results.extend(batch_results)
        
        return results
    
    def analyze_comments(self, comments: List[Dict], 
                        text_key: str = 'cleaned_text') -> List[Dict]:
        """
        Analyze sentiment for a list of comments
        
        Args:
            comments: List of comment dictionaries
            text_key: Key containing the text to analyze
        
        Returns:
            List of comments with added sentiment fields
        """
        texts = [comment.get(text_key, '') for comment in comments]
        predictions = self.predict_batch(texts)
        
        analyzed_comments = []
        for comment, prediction in zip(comments, predictions):
            comment_copy = comment.copy()
            comment_copy['sentiment'] = prediction['label']
            comment_copy['sentiment_confidence'] = prediction['confidence']
            comment_copy['sentiment_scores'] = prediction['probabilities']
            analyzed_comments.append(comment_copy)
        
        return analyzed_comments
    
    def get_sentiment_distribution(self, comments: List[Dict]) -> Dict:
        """Get overall sentiment distribution statistics"""
        sentiments = [c.get('sentiment', 'neutral') for c in comments]
        total = len(sentiments)
        
        if total == 0:
            return {'positive': 0, 'neutral': 0, 'negative': 0, 'total': 0}
        
        return {
            'positive': sentiments.count('positive'),
            'neutral': sentiments.count('neutral'),
            'negative': sentiments.count('negative'),
            'positive_pct': round(sentiments.count('positive') / total * 100, 2),
            'neutral_pct': round(sentiments.count('neutral') / total * 100, 2),
            'negative_pct': round(sentiments.count('negative') / total * 100, 2),
            'total': total
        }
    
    def get_sentiment_by_time(self, comments: List[Dict], 
                             time_buckets: str = 'hour') -> Dict:
        """
        Get sentiment distribution grouped by time periods
        
        Args:
            comments: List of comments with 'published_at' and 'sentiment' fields
            time_buckets: 'hour' or 'day'
        
        Returns:
            Dictionary with time buckets and sentiment counts
        """
        from collections import defaultdict
        from datetime import datetime
        
        time_sentiments = defaultdict(lambda: {'positive': 0, 'neutral': 0, 'negative': 0})
        
        for comment in comments:
            try:
                published = datetime.fromisoformat(comment['published_at'].replace('Z', '+00:00'))
                
                if time_buckets == 'hour':
                    key = published.strftime('%Y-%m-%d %H:00')
                else:
                    key = published.strftime('%Y-%m-%d')
                
                sentiment = comment.get('sentiment', 'neutral')
                time_sentiments[key][sentiment] += 1
            except:
                continue
        
        return dict(sorted(time_sentiments.items()))


class AspectBasedSentimentAnalyzer:
    """
    Aspect-Based Sentiment Analysis (ABSA)
    Identifies sentiment toward specific aspects (content, audio, video, etc.)
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load zero-shot classification model
        from transformers import pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Aspect categories
        self.aspects = [
            "content quality",
            "audio quality", 
            "video quality",
            "presentation style",
            "technical issues",
            "host personality"
        ]
    
    def analyze_aspects(self, text: str) -> Dict:
        """
        Identify which aspects are mentioned and their sentiment
        
        Returns:
            Dictionary with aspect predictions
        """
        if not text or len(text) < 10:
            return {}
        
        result = self.classifier(text, self.aspects, multi_label=True)
        
        aspects_detected = {}
        for label, score in zip(result['labels'], result['scores']):
            if score > 0.5:  # Threshold for aspect detection
                aspects_detected[label] = round(score, 4)
        
        return aspects_detected
    
    def analyze_comments_aspects(self, comments: List[Dict], 
                                  text_key: str = 'cleaned_text') -> List[Dict]:
        """Analyze aspects for a list of comments"""
        analyzed = []
        for comment in comments:
            aspects = self.analyze_aspects(comment.get(text_key, ''))
            comment_copy = comment.copy()
            comment_copy['aspects'] = aspects
            analyzed.append(comment_copy)
        return analyzed
