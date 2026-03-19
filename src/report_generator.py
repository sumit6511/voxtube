"""
PDF Report Generator Module
Generates comprehensive PDF reports of the analysis
"""

from fpdf import FPDF
from datetime import datetime
from typing import List, Dict
import io


class PDFReport(FPDF):
    """Custom PDF class for VoxTube reports"""
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 12)
        self.set_text_color(41, 128, 185)
        self.cell(0, 10, 'VoxTube - YouTube Comment Analytics Report', 0, 0, 'L')
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 0, 'R')
        self.ln(15)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        """Add chapter title"""
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f'  {title}', 0, 1, 'L', True)
        self.set_text_color(0, 0, 0)
        self.ln(5)
    
    def chapter_body(self, body):
        """Add chapter body text"""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 6, body)
        self.ln()
    
    def add_stat_box(self, label, value, x, y, w=45, h=25):
        """Add a statistics box"""
        self.set_xy(x, y)
        self.set_fill_color(236, 240, 241)
        self.rect(x, y, w, h, 'F')
        
        self.set_xy(x, y + 3)
        self.set_font('Arial', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(w, 6, label, 0, 2, 'C')
        
        self.set_font('Arial', 'B', 14)
        self.set_text_color(41, 128, 185)
        self.cell(w, 10, str(value), 0, 2, 'C')


class ReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self):
        self.pdf = PDFReport()
    
    def generate_report(self, metadata: Dict, comments: List[Dict], 
                       sentiment_stats: Dict, toxicity_stats: Dict,
                       topic_stats: Dict) -> bytes:
        """
        Generate a comprehensive PDF report
        
        Args:
            metadata: Video metadata
            comments: Analyzed comments
            sentiment_stats: Sentiment statistics
            toxicity_stats: Toxicity statistics
            topic_stats: Topic statistics
        
        Returns:
            PDF as bytes
        """
        pdf = PDFReport()
        pdf.add_page()
        
        # Title Page
        pdf.set_font('Arial', 'B', 24)
        pdf.set_text_color(41, 128, 185)
        pdf.cell(0, 20, 'VoxTube Analytics Report', 0, 1, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(10)
        
        # Video Information
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Video Information', 0, 1)
        pdf.set_font('Arial', '', 12)
        
        video_info = [
            f"Title: {metadata.get('title', 'N/A')}",
            f"Channel: {metadata.get('channel_title', 'N/A')}",
            f"Published: {metadata.get('published_at', 'N/A')[:10]}",
            f"Views: {metadata.get('view_count', 0):,}",
            f"Likes: {metadata.get('like_count', 0):,}",
            f"Comments Analyzed: {len(comments):,}"
        ]
        
        for info in video_info:
            pdf.cell(0, 8, info, 0, 1)
        
        pdf.ln(10)
        
        # Executive Summary
        pdf.chapter_title('Executive Summary')
        
        total = len(comments)
        positive_pct = sentiment_stats.get('positive_pct', 0)
        negative_pct = sentiment_stats.get('negative_pct', 0)
        toxicity_rate = toxicity_stats.get('toxicity_rate', 0)
        
        summary = (
            f"This report analyzes {total:,} comments from the YouTube video. "
            f"The overall sentiment is {'positive' if positive_pct > negative_pct else 'mixed' if abs(positive_pct - negative_pct) < 10 else 'negative'} "
            f"with {positive_pct:.1f}% positive and {negative_pct:.1f}% negative comments. "
            f"The toxicity level is {'low' if toxicity_rate < 5 else 'moderate' if toxicity_rate < 15 else 'high'} "
            f"at {toxicity_rate:.1f}%. "
            f"Key discussion topics include {topic_stats.get('topic_distribution', [['N/A', 0]])[0][0] if topic_stats.get('topic_distribution') else 'various subjects'}."
        )
        
        pdf.chapter_body(summary)
        
        # Sentiment Analysis
        pdf.add_page()
        pdf.chapter_title('Sentiment Analysis')
        
        sentiment_text = (
            f"Total Comments Analyzed: {sentiment_stats.get('total', 0):,}\n\n"
            f"Positive Comments: {sentiment_stats.get('positive', 0):,} ({sentiment_stats.get('positive_pct', 0):.1f}%)\n"
            f"Neutral Comments: {sentiment_stats.get('neutral', 0):,} ({sentiment_stats.get('neutral_pct', 0):.1f}%)\n"
            f"Negative Comments: {sentiment_stats.get('negative', 0):,} ({sentiment_stats.get('negative_pct', 0):.1f}%)\n\n"
            f"Sentiment Ratio: {sentiment_stats.get('positive', 0) / max(sentiment_stats.get('negative', 1), 1):.2f} "
            f"(Positive:Negative)"
        )
        
        pdf.chapter_body(sentiment_text)
        
        # Add sample comments by sentiment
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Sample Positive Comments:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        positive_comments = [c for c in comments if c.get('sentiment') == 'positive'][:3]
        for i, comment in enumerate(positive_comments, 1):
            text = comment.get('text', '')[:100]
            if len(comment.get('text', '')) > 100:
                text += '...'
            pdf.multi_cell(0, 5, f"{i}. \"{text}\" - {comment.get('author', 'Anonymous')}")
            pdf.ln(2)
        
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Sample Negative Comments:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        negative_comments = [c for c in comments if c.get('sentiment') == 'negative'][:3]
        for i, comment in enumerate(negative_comments, 1):
            text = comment.get('text', '')[:100]
            if len(comment.get('text', '')) > 100:
                text += '...'
            pdf.multi_cell(0, 5, f"{i}. \"{text}\" - {comment.get('author', 'Anonymous')}")
            pdf.ln(2)
        
        # Toxicity Analysis
        pdf.add_page()
        pdf.chapter_title('Toxicity & Safety Analysis')
        
        toxicity_text = (
            f"Toxicity Rate: {toxicity_stats.get('toxicity_rate', 0):.1f}%\n\n"
            f"Normal Comments: {toxicity_stats.get('normal_count', 0):,}\n"
            f"Offensive Comments: {toxicity_stats.get('offensive_count', 0):,}\n"
            f"Hate Speech Comments: {toxicity_stats.get('hate_count', 0):,}\n\n"
            f"Safety Assessment: {'Safe' if toxicity_stats.get('toxicity_rate', 0) < 5 else 'Moderate concern' if toxicity_stats.get('toxicity_rate', 0) < 15 else 'High toxicity detected'}"
        )
        
        pdf.chapter_body(toxicity_text)
        
        # Topic Analysis
        pdf.add_page()
        pdf.chapter_title('Topic Analysis')
        
        if topic_stats and 'topic_distribution' in topic_stats:
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Top Discussion Topics:', 0, 1)
            pdf.set_font('Arial', '', 10)
            
            for topic_id, count in topic_stats['topic_distribution'][:5]:
                if topic_id != -1:
                    pdf.cell(0, 6, f"Topic {topic_id}: {count} comments", 0, 1)
        
        pdf.ln(10)
        
        # Key Insights
        pdf.chapter_title('Key Insights & Recommendations')
        
        insights = []
        
        # Generate insights based on data
        if positive_pct > 70:
            insights.append("The video has received overwhelmingly positive feedback. Consider creating similar content.")
        elif negative_pct > 40:
            insights.append("The video has significant negative sentiment. Review viewer feedback for improvement areas.")
        
        if toxicity_rate > 10:
            insights.append("High toxicity detected. Consider moderating comments more actively.")
        
        if sentiment_stats.get('positive', 0) > sentiment_stats.get('negative', 0) * 2:
            insights.append("Strong positive engagement. This content resonates well with your audience.")
        
        avg_likes = sum(c.get('like_count', 0) for c in comments) / max(len(comments), 1)
        if avg_likes > 10:
            insights.append(f"High engagement rate (avg {avg_likes:.1f} likes per comment).")
        
        if not insights:
            insights.append("The video shows typical engagement patterns. Continue monitoring audience feedback.")
        
        for i, insight in enumerate(insights, 1):
            pdf.chapter_body(f"{i}. {insight}")
        
        # Footer
        pdf.ln(20)
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, 'Generated by VoxTube - Multidimensional YouTube Comment Analytics', 0, 1, 'C')
        pdf.cell(0, 10, '© 2025 VoxTube Project', 0, 1, 'C')
        
        # Return PDF as bytes
        return pdf.output(dest='S').encode('latin-1')
    
    def save_report(self, pdf_bytes: bytes, filename: str):
        """Save PDF report to file"""
        with open(filename, 'wb') as f:
            f.write(pdf_bytes)
