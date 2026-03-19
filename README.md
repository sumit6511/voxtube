# 📊 VoxTube

**Multidimensional Sentiment Analysis and Topic Modeling of YouTube Comments**

[![Python 3.10-3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)

---

## 🎯 Overview

VoxTube is an intelligent, web-based analytics framework that transforms unstructured YouTube comments into actionable insights. Using state-of-the-art Natural Language Processing (NLP) techniques, VoxTube helps content creators understand their audience's sentiment, identify discussion topics, and detect toxic content.

### Key Features

- 🧠 **Multilingual Sentiment Analysis** - XLM-RoBERTa for English & code-mixed text (Hinglish/Neplish)
- 🛡️ **Toxicity Detection** - Identify hate speech and offensive content
- 📊 **Topic Modeling** - BERTopic for extracting discussion themes
- 🤖 **AI Chat (RAG)** - Ask questions about your audience feedback
- 📈 **Interactive Visualizations** - Rich charts and word clouds
- 📄 **PDF Reports** - Generate comprehensive analysis reports

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 to 3.12
- YouTube Data API v3 key ([Get one here](https://console.cloud.google.com/apis/credentials))
- Gemini API key (optional, for chat feature) ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voxtube.git
   cd voxtube
   ```

2. **Create virtual environment**
   ```bash
   # On Windows
   py -3.11 -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

---

## 📖 Usage Guide

### 1. Data Input
- Enter a YouTube video URL
- Set the maximum number of comments to analyze (100-5000)
- Click "Analyze Comments"

### 2. View Analytics
- **Sentiment Distribution** - Pie and bar charts showing positive/neutral/negative breakdown
- **Toxicity Analysis** - Identify harmful content
- **Topic Modeling** - Discover what viewers are discussing
- **Word Cloud** - Visualize most common terms

### 3. AI Chat (RAG)
- Ask natural language questions about your comments
- Get AI-powered insights based on actual viewer feedback
- Example questions:
  - "What do viewers like most about this video?"
  - "What are the main complaints?"
  - "How is the audio/video quality?"

### 4. Generate Report
- Download a comprehensive PDF report
- Share insights with your team

---

## 🏗️ Architecture

```
VoxTube/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
├── src/
│   ├── youtube_extractor.py   # YouTube Data API integration
│   ├── preprocessor.py        # Text cleaning & normalization
│   ├── sentiment_analyzer.py  # XLM-RoBERTa sentiment analysis
│   ├── toxicity_detector.py   # Toxic content detection
│   ├── topic_modeler.py       # BERTopic topic extraction
│   ├── rag_chat.py            # RAG-based Q&A system
│   ├── visualizations.py      # Chart generation
│   └── report_generator.py    # PDF report creation
├── models/                    # Downloaded model cache
├── utils/                     # Utility functions
└── tests/                     # Unit tests
```

---

## 🤖 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Sentiment Analysis | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual sentiment classification |
| Toxicity Detection | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | 3-class hate speech, offensive, and normal classification |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings for RAG & topic modeling |
| Topic Modeling | BERTopic + HDBSCAN | Extract discussion themes |
| LLM for RAG | Gemini 1.5 Flash | Generate chat responses |

---

## 📊 Performance Metrics

Based on research and benchmarking:

| Task | Expected Accuracy |
|------|-------------------|
| Sentiment (English) | 90-95% |
| Sentiment (Code-Mixed) | 85-88% |
| Toxicity Detection | 92-97% |
| Topic Coherence | Cv > 0.45 |

---

## 🔑 API Keys Setup

### YouTube Data API v3

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Copy the key to your `.env` file

**Quota Limits**: 10,000 units/day (free tier)
- `commentThreads.list` = 1 unit
- `comments.list` = 1 unit
- `videos.list` = 1 unit

### Gemini API

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with Google account
3. Create API key
4. Copy the key to your `.env` file

---

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/
flake8 src/
```

---

## 📝 Project Structure Details

### Data Flow

1. **Extraction** (`youtube_extractor.py`)
   - Fetch comments via YouTube Data API v3
   - Handle pagination and quota management
   - Extract metadata (title, views, likes)

2. **Preprocessing** (`preprocessor.py`)
   - Clean HTML tags and URLs
   - Handle emojis (convert to text)
   - Normalize slang and abbreviations
   - Remove special characters

3. **Analysis Pipeline**
   - **Sentiment** (`sentiment_analyzer.py`): XLM-RoBERTa classification
   - **Toxicity** (`toxicity_detector.py`): Hate speech detection
   - **Topics** (`topic_modeler.py`): BERTopic clustering

4. **RAG Chat** (`rag_chat.py`)
   - Embed comments with FAISS
   - Retrieve relevant comments
   - Generate responses with Gemini

5. **Visualization** (`visualizations.py`)
   - Plotly charts for interactive dashboards
   - Word clouds for keyword visualization

6. **Reporting** (`report_generator.py`)
   - PDF generation with FPDF
   - Executive summary and insights

---

## 🎓 Academic Context

This project is developed as a Final Year Project (FYP) for:
- **Institution**: St. Xavier's College, Kathmandu
- **Affiliation**: Tribhuvan University
- **Course**: CSC 422 - Bachelor's in Computer Science and Information Technology
- **Supervisor**: Er. Rajan Karmacharya

### Team Members
- Sumit Kumar Sah (022BSCIT045)
- Utsav Adhikari (022BSCIT048)

---

## 📚 References

Key research papers and resources:

1. Hutto & Gilbert (2014) - VADER Sentiment Analysis
2. Devlin et al. (2019) - BERT: Pre-training of Deep Bidirectional Transformers
3. Liu et al. (2019) - RoBERTa: A Robustly Optimized BERT Pretraining Approach
4. Grootendorst (2022) - BERTopic: Neural topic modeling
5. Conneau et al. (2020) - XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning

See full references in the project proposal document.

---

## 🔒 Privacy & Legal

- **Data Privacy**: Comments are processed in memory and not stored permanently
- **GDPR Compliant**: No PII (Personally Identifiable Information) is collected
- **Fair Use**: Educational and research purposes only
- **YouTube API Compliance**: Adheres to YouTube API Services Terms of Service

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Google for YouTube Data API and Gemini
- Streamlit for the amazing web framework
- BERTopic developers for topic modeling tools

---

## 📧 Contact

For questions or support:
- Email: [your-email@example.com]
- GitHub Issues: [Create an issue](https://github.com/yourusername/voxtube/issues)

---

**⭐ Star this repository if you find it helpful!**
