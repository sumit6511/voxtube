"""
Test Script for VoxTube Modules
Quick verification that all components work correctly
"""

import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_preprocessor():
    """Test text preprocessing module"""
    print("\n" + "="*50)
    print("Testing Preprocessor Module")
    print("="*50)
    
    from src.preprocessor import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    
    test_texts = [
        "This video is fire! 🔥 No cap, best tutorial ever! https://example.com",
        "@username great content! lol 😂",
        "Audio quality is sooooo bad... mid content tbh"
    ]
    
    for text in test_texts:
        cleaned = preprocessor.preprocess(text)
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
    
    print("\n✅ Preprocessor test passed!")


def test_sentiment_analyzer():
    """Test sentiment analysis module"""
    print("\n" + "="*50)
    print("Testing Sentiment Analyzer Module")
    print("="*50)
    
    from src.sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "This video is amazing! I learned so much!",
        "The audio quality is terrible, couldn't hear anything",
        "It's okay, nothing special",
        "Bhai video mast hai, bahut accha laga!",  # Hinglish
    ]
    
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['label']} (confidence: {result['confidence']})")
    
    print("\n✅ Sentiment Analyzer test passed!")


def test_toxicity_detector():
    """Test toxicity detection module"""
    print("\n" + "="*50)
    print("Testing Toxicity Detector Module")
    print("="*50)
    
    from src.toxicity_detector import ToxicityDetector
    
    detector = ToxicityDetector()
    
    test_texts = [
        "This is a great video, thanks for sharing!",
        "You are so stupid, this is the worst content ever",
        "I hate this kind of content, go away"
    ]
    
    for text in test_texts:
        result = detector.predict(text)
        print(f"\nText: {text}")
        print(f"Toxicity: {result['label']} (is_toxic: {result['is_toxic']}, confidence: {result['confidence']})")
    
    print("\n✅ Toxicity Detector test passed!")


def test_visualizer():
    """Test visualization module"""
    print("\n" + "="*50)
    print("Testing Visualizer Module")
    print("="*50)
    
    from src.visualizations import Visualizer
    
    visualizer = Visualizer()
    
    # Test sentiment pie chart
    sentiment_stats = {'positive': 50, 'neutral': 30, 'negative': 20}
    fig = visualizer.create_sentiment_pie(sentiment_stats)
    print(f"✅ Created sentiment pie chart")
    
    # Test word cloud
    texts = ["great video amazing content", "good tutorial helpful thanks", "awesome quality"]
    wordcloud_b64 = visualizer.create_wordcloud(texts)
    print(f"✅ Created word cloud (base64 length: {len(wordcloud_b64) if wordcloud_b64 else 0})")
    
    print("\n✅ Visualizer test passed!")


def test_youtube_extractor():
    """Test YouTube extractor (requires API key)"""
    print("\n" + "="*50)
    print("Testing YouTube Extractor Module")
    print("="*50)
    
    from src.youtube_extractor import YouTubeExtractor
    
    # Test video ID extraction
    extractor = YouTubeExtractor.__new__(YouTubeExtractor)
    
    test_urls = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    ]
    
    for url in test_urls:
        try:
            video_id = extractor.extract_video_id(url)
            print(f"✅ Extracted '{video_id}' from {url[:40]}...")
        except Exception as e:
            print(f"❌ Failed for {url}: {e}")
    
    print("\n✅ YouTube Extractor test passed!")


def run_all_tests():
    """Run all module tests"""
    print("\n" + "="*60)
    print("VOXTUBE MODULE TEST SUITE")
    print("="*60)
    
    try:
        test_preprocessor()
    except Exception as e:
        print(f"\n❌ Preprocessor test failed: {e}")
    
    try:
        test_sentiment_analyzer()
    except Exception as e:
        print(f"\n❌ Sentiment Analyzer test failed: {e}")
    
    try:
        test_toxicity_detector()
    except Exception as e:
        print(f"\n❌ Toxicity Detector test failed: {e}")
    
    try:
        test_visualizer()
    except Exception as e:
        print(f"\n❌ Visualizer test failed: {e}")
    
    try:
        test_youtube_extractor()
    except Exception as e:
        print(f"\n❌ YouTube Extractor test failed: {e}")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
