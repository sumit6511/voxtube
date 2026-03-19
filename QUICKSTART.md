# VoxTube Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
# Create virtual environment
# Windows (recommended)
py -3.11 -m venv venv

# macOS/Linux
python3.11 -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

Recommended Python version: 3.10 to 3.12. Python 3.14 is not supported by several pinned dependencies in this project.

### Step 2: Get API Keys

#### YouTube Data API Key (Required)
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create an API Key
5. Copy it

#### Gemini API Key (Optional - for AI Chat)
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API Key
3. Copy it

### Step 3: Configure Environment

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your keys
YOUTUBE_API_KEY=your_youtube_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Using VoxTube

### Analyze a Video

1. **Paste a YouTube URL** in the Data Input tab
   - Example: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`

2. **Set Analysis Options** in the sidebar
   - Max comments: 100-5000
   - Enable/disable features

3. **Click "Analyze Comments"**
   - Wait 1-5 minutes depending on comment count

4. **View Results** in the Analytics tab
   - Sentiment distribution
   - Toxicity analysis
   - Topic modeling
   - Word cloud

### Use AI Chat

1. Go to the **AI Chat** tab
2. Ask questions like:
   - "What do viewers like most?"
   - "What are the main complaints?"
   - "How is the video quality?"

### Generate Report

1. Go to the **Report** tab
2. Click "Generate PDF Report"
3. Download and share!

---

## 🧪 Test Your Setup

Run the test script to verify everything works:

```bash
python test_modules.py
```

You should see output like:
```
==================================================
Testing Preprocessor Module
==================================================

Original: This video is fire! 🔥 No cap!
Cleaned:  This video is fire fire no lie

✅ Preprocessor test passed!
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'src'"
**Solution**: Make sure you're running from the project root directory

### Issue: "YouTube API key is required"
**Solution**: Add your API key to the `.env` file

### Issue: "Quota exceeded" error
**Solution**: 
- Wait 24 hours for quota reset
- Or create a new API key

### Issue: Models downloading slowly
**Solution**: First run will download ~2GB of models. Be patient!

### Issue: Out of memory
**Solution**: 
- Reduce max comments in sidebar
- Close other applications
- Use a machine with more RAM

---

## 📈 Performance Tips

1. **Start Small**: Test with 100-500 comments first
2. **Use GPU**: If available, models will run faster
3. **Cache Results**: Same video won't re-fetch comments
4. **Disable Unused Features**: Turn off chat/topic modeling if not needed

---

## 🎯 Next Steps

- Explore different video types
- Compare multiple videos
- Export reports for your content strategy
- Share insights with your team

---

## 📚 Learn More

- [Full Documentation](README.md)
- [Project Proposal](docs/PROPOSAL.md)
- [API Documentation](docs/API.md)

---

**Happy Analyzing! 🎉**
