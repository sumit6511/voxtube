"""
Neplish-aware text preprocessor for VoxTube.

Neplish = code-mixed Nepali + English YouTube comments.
Examples of what this handles:
  "यो video धेरै राम्रो cha yaar 😂😂😂"
  "bro yo song soooooo goood!!!! https://youtu.be/xyz @user #trending"
  "hahahaha 🔥🔥 ramro thiyo bro &amp; sis"

Design decision: Devanagari script is PRESERVED throughout.
XLM-RoBERTa's multilingual tokenizer handles it natively — no
transliteration needed. Removing Devanagari would lose meaning.
"""

import re
import html

import ftfy
import emoji


# ── Compiled regex patterns (created once at import, reused every call) ───────

_URL_RE      = re.compile(r'https?://\S+|www\.\S+')
_MENTION_RE  = re.compile(r'@\w+')
_HASHTAG_RE  = re.compile(r'#(\w+)')     # drop the # but keep the word
_REPEAT_RE   = re.compile(r'(.)\1{2,}')  # haaaa → haa  (collapse to max 2)

# Allow: Devanagari (U+0900–U+097F), word chars [a-zA-Z0-9_],
#        whitespace, common punctuation, colon (for :emoji_name: tokens),
#        ampersand (& unescaped from &amp; should be kept as "and")
# Note: - must stay at the END of the class to be treated as a literal hyphen
_NOISE_RE    = re.compile(r"[^\u0900-\u097F\w\s.,!?'\"&:-]")
_SPACE_RE    = re.compile(r'\s+')


# ── Core cleaning function ────────────────────────────────────────────────────

def clean_comment(text: str) -> str:
    """
    Clean a single raw YouTube comment for NLP processing.

    Steps (in order):
      1. Fix encoding artifacts         ftfy  (e.g. â€™ → ')
      2. Unescape HTML entities               (&amp; → &, &#39; → ')
      3. Remove URLs
      4. Remove @mentions
      5. Drop # from hashtags                 (#viral → viral)
      6. Convert emojis → text tokens         (😂 → :face_with_tears_of_joy:)
      7. Collapse repeated characters         (haaaa → haa, max 2)
      8. Remove remaining noise characters    (keep Devanagari + Latin + punct)
      9. Normalize whitespace

    Returns empty string if input is empty or whitespace-only.
    """
    if not text or not text.strip():
        return ""

    text = ftfy.fix_text(text)                   # 1
    text = html.unescape(text)                   # 2
    text = _URL_RE.sub("", text)                 # 3
    text = _MENTION_RE.sub("", text)             # 4
    text = _HASHTAG_RE.sub(r"\1", text)          # 5
    text = emoji.demojize(text)                  # 6 — 😂 → :face_with_tears_of_joy:
    text = _REPEAT_RE.sub(r"\1\1", text)         # 7
    text = _NOISE_RE.sub(" ", text)              # 8
    text = _SPACE_RE.sub(" ", text).strip()      # 9

    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Clean a list of comment strings.
    Returns a list of the same length with clean text for each input.
    """
    return [clean_comment(t) for t in texts]


# Common romanized Nepali words — presence of any of these in a Latin-script
# comment is a strong signal that it's Neplish, not English.
_NEPALI_WORDS = {
    'ramro','sanchai','cha','xa','xha','huncha','hunchha','thyo','bhayo',
    'haina','garne','gareko','garnu','garnuhos','bhanne','lagyo','lagcha',
    'dherai','ali','aile','aaja','hijo','pani','ani','tara','ho','hola',
    'hajur','dai','didi','bhai','bahini','maile','timle','kasle','kasari',
    'kasto','ramrai','sab','sabai','sundar','mazza','yaar','sathi','saathi',
    'kina','kaha','kahile','malai','timilai','kei','kehi','afno','tapai',
    'hamro','timro','mero','pugyo','aayo','gayo','basyo','khayo','garey',
}


def _has_nepali_words(text: str) -> bool:
    """Return True if the text contains any known romanized Nepali words."""
    return bool(set(text.lower().split()) & _NEPALI_WORDS)

_DEV_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> str:
    """
    Classify a comment's language into one of three categories:

    - 'nepali'  : >= 30% of non-space characters are Devanagari script.
    - 'english' : No Devanagari; langdetect is confident it's English.
    - 'neplish' : Everything else — code-mixed, romanized Nepali in
                  Latin script, or ambiguous text.

    The 30% Devanagari threshold means a comment like
    "यो song ramro cha" (mixed) → 'neplish', while
    "यो video धेरै राम्रो थियो" (mostly Devanagari) → 'nepali'.
    """
    if not text or not text.strip():
        return 'neplish'

    non_space = text.replace(' ', '')
    if not non_space:
        return 'neplish'

    dev_count = len(_DEV_RE.findall(text))
    dev_ratio = dev_count / len(non_space)

    if dev_ratio >= 0.30:
        return 'nepali'
    if dev_count > 0:
        return 'neplish'          # some Devanagari but not dominant

    # No Devanagari — check for romanized Nepali words first
    if _has_nepali_words(text):
        return 'neplish'

    # Fall back to langdetect for everything else
    try:
        from langdetect import detect
        lang = detect(text)
        return 'english' if lang == 'en' else 'neplish'
    except Exception:
        return 'neplish'


def detect_languages(texts: list[str]) -> list[str]:
    """Detect language for a batch of texts."""
    return [detect_language(t) for t in texts]
