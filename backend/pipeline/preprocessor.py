import re
import html
import ftfy
import emoji

_URL_RE     = re.compile(r'https?://\S+|www\.\S+')
_MENTION_RE = re.compile(r'@\w+')
_HASHTAG_RE = re.compile(r'#(\w+)')
_REPEAT_RE  = re.compile(r'(.)\1{2,}')
_NOISE_RE   = re.compile(r"[^\u0900-\u097F\w\s.,!?'\"&:-]")
_SPACE_RE   = re.compile(r'\s+')


def clean_comment(text: str) -> str:
    if not text or not text.strip(): return ""
    text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = _URL_RE.sub("", text)
    text = _MENTION_RE.sub("", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    text = emoji.demojize(text)
    text = _REPEAT_RE.sub(r"\1\1", text)
    text = _NOISE_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    return [clean_comment(t) for t in texts]


_NEPALI_WORDS = {
    'ramro','sanchai','cha','xa','xha','huncha','hunchha','thyo','bhayo','haina','garne','gareko',
    'garnu','garnuhos','bhanne','lagyo','lagcha','dherai','ali','aile','aaja','hijo','pani','ani',
    'tara','ho','hola','hajur','dai','didi','bhai','bahini','maile','timle','kasle','kasari','kasto',
    'ramrai','sab','sabai','sundar','mazza','yaar','sathi','saathi','kina','kaha','kahile','malai',
    'timilai','kei','kehi','afno','tapai','hamro','timro','mero','pugyo','aayo','gayo','basyo','khayo','garey',
}

def _has_nepali_words(text: str) -> bool:
    return bool(set(text.lower().split()) & _NEPALI_WORDS)

_DEV_RE = re.compile(r'[\u0900-\u097F]')

def detect_language(text: str) -> str:
    if not text or not text.strip(): return 'neplish'
    non_space = text.replace(' ', '')
    if not non_space: return 'neplish'
    dev_count = len(_DEV_RE.findall(text))
    dev_ratio = dev_count / len(non_space)
    if dev_ratio >= 0.30: return 'nepali'
    if dev_count > 0: return 'neplish'
    if _has_nepali_words(text): return 'neplish'
    try:
        from langdetect import detect
        lang = detect(text)
        return 'english' if lang == 'en' else 'neplish'
    except Exception:
        return 'neplish'


def detect_languages(texts: list[str]) -> list[str]:
    return [detect_language(t) for t in texts]
