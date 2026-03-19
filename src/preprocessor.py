"""
Text Preprocessing Module
Handles cleaning and normalization of YouTube comments
"""

import re
import emoji
import html
from typing import List, Dict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


class TextPreprocessor:
    """Preprocess YouTube comments for NLP analysis"""
    
    def __init__(self):
        # Load spaCy model for advanced preprocessing
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(['python', '-m', 'spacy', 'download', 'en_core_web_sm'])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Common patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.extra_spaces = re.compile(r'\s+')
        
        # Common slang and abbreviations
        self.slang_dict = {
            'lol': 'laughing out loud',
            'lmao': 'laughing my ass off',
            'rofl': 'rolling on the floor laughing',
            'omg': 'oh my god',
            'wtf': 'what the fuck',
            'tbh': 'to be honest',
            'imo': 'in my opinion',
            'imho': 'in my humble opinion',
            'fyi': 'for your information',
            'brb': 'be right back',
            'btw': 'by the way',
            'idk': 'i do not know',
            'ikr': 'i know right',
            'nvm': 'never mind',
            'ttyl': 'talk to you later',
            'fr': 'for real',
            'cap': 'lie',
            'no cap': 'no lie',
            'bet': 'okay',
            'slay': 'do excellently',
            'fire': 'excellent',
            'lit': 'exciting',
            'dope': 'cool',
            'sus': 'suspicious',
            'bussin': 'delicious',
            'mid': 'average',
            'rizz': 'charisma',
            'gyatt': 'wow',
            'skibidi': 'cool',
            'sigma': 'independent',
            'mewing': 'jaw exercise',
            'fanum tax': 'sharing food'
        }
    
    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities"""
        text = self.html_pattern.sub(' ', text)
        text = html.unescape(text)
        return text
    
    def handle_emojis(self, text: str, convert_to_text: bool = True) -> str:
        """Convert emojis to text descriptions or remove them"""
        if convert_to_text:
            return emoji.demojize(text, delimiters=(" ", " "))
        else:
            return emoji.replace_emoji(text, replace='')
    
    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def remove_mentions(self, text: str) -> str:
        """Remove @mentions from text"""
        return self.mention_pattern.sub('', text)
    
    def normalize_slang(self, text: str) -> str:
        """Normalize common slang and abbreviations"""
        text_lower = text.lower()
        for slang, meaning in self.slang_dict.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(slang) + r'\b'
            text_lower = re.sub(pattern, meaning, text_lower)
        return text_lower if text_lower != text.lower() else text
    
    def normalize_repeated_chars(self, text: str) -> str:
        """Normalize repeated characters (e.g., 'sooooo' -> 'so')"""
        return re.sub(r'(.)\1{3,}', r'\1\1', text)
    
    def remove_extra_spaces(self, text: str) -> str:
        """Remove extra whitespace"""
        return self.extra_spaces.sub(' ', text).strip()
    
    def preprocess(self, text: str, 
                   remove_urls: bool = True,
                   remove_mentions: bool = True,
                   convert_emojis: bool = True,
                   normalize_slang: bool = True,
                   lowercase: bool = False) -> str:
        """
        Full preprocessing pipeline for a single comment
        
        Args:
            text: Raw comment text
            remove_urls: Whether to remove URLs
            remove_mentions: Whether to remove @mentions
            convert_emojis: Whether to convert emojis to text
            normalize_slang: Whether to normalize slang
            lowercase: Whether to convert to lowercase
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Clean HTML
        text = self.clean_html(text)
        
        # Step 2: Handle emojis
        if convert_emojis:
            text = self.handle_emojis(text, convert_to_text=True)
        
        # Step 3: Remove URLs
        if remove_urls:
            text = self.remove_urls(text)
        
        # Step 4: Remove mentions
        if remove_mentions:
            text = self.remove_mentions(text)
        
        # Step 5: Normalize slang
        if normalize_slang:
            text = self.normalize_slang(text)
        
        # Step 6: Normalize repeated characters
        text = self.normalize_repeated_chars(text)
        
        # Step 7: Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"]', ' ', text)
        
        # Step 8: Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        # Step 9: Lowercase (optional)
        if lowercase:
            text = text.lower()
        
        return text.strip()
    
    def preprocess_comments(self, comments: List[Dict], 
                           text_key: str = 'text') -> List[Dict]:
        """
        Preprocess a list of comment dictionaries
        
        Args:
            comments: List of comment dictionaries
            text_key: Key containing the text to preprocess
        
        Returns:
            List of comments with added 'cleaned_text' field
        """
        processed = []
        for comment in comments:
            cleaned = self.preprocess(comment.get(text_key, ''))
            comment_copy = comment.copy()
            comment_copy['cleaned_text'] = cleaned
            comment_copy['original_length'] = len(comment.get(text_key, ''))
            comment_copy['cleaned_length'] = len(cleaned)
            processed.append(comment_copy)
        
        return processed
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract key phrases using spaCy"""
        doc = self.nlp(text)
        
        # Extract noun phrases and named entities
        keywords = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                keywords.append(chunk.text.lower())
        
        for ent in doc.ents:
            keywords.append(ent.text.lower())
        
        # Count and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [kw for kw, _ in keyword_counts.most_common(top_n)]
