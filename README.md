# VoxTube

VoxTube is a comprehensive web application designed to analyze YouTube comments using advanced Natural Language Processing (NLP) pipelines. It extracts comments from a given YouTube video URL and runs them through a multi-stage machine learning pipeline to uncover insights such as sentiment, toxicity, underlying topics, and allows users to query the comments via a Retrieval-Augmented Generation (RAG) system.

This project was built with a Python (FastAPI) backend and a React (TypeScript + Vite) frontend.

## Features

- **Comment Fetching**: Automatically fetches comments from a provided YouTube video.
- **Preprocessing & Language Detection**: Cleans up text (handling emojis and special characters) and detects languages, including support for Nepalese text preprocessing.
- **Sentiment Analysis**: Uses XLM-RoBERTa and VADER to accurately classify the sentiment of comments as positive, neutral, or negative.
- **Toxicity Detection**: Leverages ToxicBERT to identify toxic and harmful comments.
- **Topic Modeling**: Groups comments into coherent topics using BERTopic, helping you understand what viewers are talking about.
- **AI Chat / RAG System**: Ask natural language questions about the video's comment section! Powered by FAISS (for vector search) and Google's Gemini LLM to generate grounded answers with source comment citations.

## Tech Stack

### Backend
- **Framework**: FastAPI (Python)
- **Database**: SQLite (managed with SQLAlchemy & Pydantic)
- **NLP & ML**: Hugging Face Transformers, PyTorch, SentencePiece, VADER Sentiment, BERTopic, FAISS
- **LLM Integration**: Google GenAI (Gemini)

### Frontend
- **Framework**: React 19 + TypeScript
- **Build Tool**: Vite
- **Styling**: TailwindCSS
- **State Management**: Zustand
- **Data Visualization**: Recharts, d3-cloud
- **Icons**: Lucide React

## Project Structure

```
voxtube/
├── backend/                  # FastAPI backend application
│   ├── pipeline/             # NLP pipeline modules (sentiment, toxicity, topics, rag)
│   ├── database.py           # SQLAlchemy setup and migrations
│   ├── main.py               # FastAPI routes and background tasks
│   ├── models.py             # Database models
│   ├── schemas.py            # Pydantic schemas
│   ├── youtube.py            # YouTube API integration
│   └── requirements.txt      # Python dependencies
├── frontend/                 # React frontend application
│   ├── public/               # Static assets
│   ├── src/                  # React components, pages, and store
│   ├── package.json          # Node dependencies
│   ├── tailwind.config.js    # Tailwind configuration
│   └── vite.config.ts        # Vite configuration
├── data/                     # Data directory (if applicable)
├── voxtube.db                # SQLite database file
├── .env                      # Environment variables
└── .env.example              # Example environment variables
```

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- YouTube Data API Key
- Google Gemini API Key

### Backend Setup

1. **Navigate to the root directory**:
   ```bash
   cd voxtube
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```
   *(Note: You may want to install a CPU-only version of PyTorch first depending on your system, e.g., `pip install torch --index-url https://download.pytorch.org/whl/cpu`)*

4. **Set up Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```env
   YOUTUBE_API_KEY=your_youtube_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the backend server**:
   ```bash
   uvicorn backend.main:app --reload
   ```
   The backend API will be available at `http://localhost:8000`.

### Frontend Setup

1. **Navigate to the frontend directory**:
   ```bash
   cd voxtube/frontend
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Run the development server**:
   ```bash
   npm run dev
   ```
   The frontend application will be available at `http://localhost:5173`.

## Usage
1. Open the frontend in your browser.
2. Paste a YouTube video URL into the application.
3. The system will start fetching and processing comments in the background.
4. Once processing is complete, you can view the sentiment summary, toxicity reports, identified topics, and interact with the AI Chat to ask specific questions about viewer feedback.

## License
[Add your license here]
