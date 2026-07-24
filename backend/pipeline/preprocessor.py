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


# ── Romanized Nepali / Neplish dictionary ────────────────────────────────────
# Expanded from an initial ~50-word list. Informal romanized Nepali has no
# standard spelling, so most core words appear here in 2-4 common spelling
# variants (e.g. cha/chha/xa/xha are all the same copula). Organized by
# grammatical role purely to make the list maintainable — matching itself
# is a flat set lookup, not order-dependent.
_NEPALI_WORDS = {
    # Copula / existential verb (cha family) — extremely high frequency,
    # appears in nearly every Nepali sentence
    "cha", "chha", "xa", "xha", "chan", "chhan", "xan", "xhan",
    "thiyo", "thyo", "thie", "thiye", "hunchha", "huncha", "hunxa",
    "vayo", "bhayo", "vo", "bho", "vaeko", "bhaeko",

    # Negation forms
    "haina", "chaina", "xaina", "vaena", "bhaena", "hunna", "hudaina", "huudaina",

    # Common verbs — garne/hunu/khane/etc. families
    "garne", "gareko", "gardai", "garcha", "garchan", "garnu", "garnus",
    "garnuhos", "garyo", "gare", "gareu",
    "khane", "khancha", "khayo", "khanu", "khaisake",
    "herne", "hernu", "herchu", "hera", "herda", "herera",
    "sunne", "sunna", "sunchu", "suneko", "sunyo",
    "bhannu", "bhanchu", "bhanyo", "bhane", "bhaneko", "bhanne", "bhandai", "vanya",
    "aaunu", "aayo", "ayo", "aaucha", "aauchan", "aaudai",
    "jaanu", "gayo", "jancha", "jannu", "jaane",
    "basne", "basyo", "bascha", "baschan", "baseko",
    "dine", "diyo", "dinu", "dincha", "diney",
    "linu", "liyo", "lincha", "liyeko",
    "paune", "payo", "pauchu", "paincha", "paayo",
    "sakne", "sakchu", "sakiyo", "sakincha",
    "parne", "paryo", "parcha", "parchan", "parxa",
    "pugyo", "pugcha", "pugne",

    # Pronouns & possessives
    "ma", "malai", "mero", "mera", "meri",
    "timi", "timro", "timile", "timilai", "timiharu",
    "hami", "haru", "hamro", "hamile", "hamiharu",
    "usle", "usko", "uslai", "yo", "tyo",
    "yesto", "testo", "yasto", "testo", "yastai", "testai",
    "yiniharu", "tiniharu",
    "afno", "aphno", "aafno",
    "tapai", "tapaiko", "tapailai", "tapaile",

    # Adjectives / descriptive words
    "ramro", "ramri", "ramailo", "naramro", "naramailo",
    "sundar", "sunder", "thulo", "sano", "saano",
    "dherai", "dherei", "thorai", "ali", "alik", "alikati",
    "khatra", "khatarnak", "mazedar", "mazza", "majale", "mast",
    "sahi", "thik", "thikai", "gajjab", "ekdam", "ekdum", "ekkai", "saccai",

    # Question words / conjunctions
    "kina", "kaha", "kahaan", "kahile", "kasari", "kasto", "kasto",
    "ke", "kun", "kunai", "kohi", "kehi",
    "ani", "tara", "yesari", "tesari", "jasto", "jastai", "jasari",

    # Time words
    "aaja", "aajai", "bholi", "hijo", "ahile", "ahiley",
    "sadhai", "sadhain", "kahilekahi", "pahile", "pachi", "pachhi",

    # Social / expressive / honorifics
    "dhanyabad", "namaste", "namaskar", "sathi", "saathi",
    "dai", "didi", "bhai", "bahini", "hajur", "daju",
    "hola", "ho", "hun", "hos", "yaar", "ni",
    "mildaina", "vaneu",
}


def _tokenize_latin(text: str) -> list[str]:
    """Extract lowercase alphabetic word tokens, stripping any attached
    punctuation. clean_comment() deliberately preserves basic punctuation
    (.,!?'"&:-), so a naive whitespace split would leave tokens like
    "cha!" or "ramro," that never match the dictionary above even though
    the word itself is present. This regex sidesteps that entirely."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def _has_nepali_words(text: str) -> bool:
    return bool(set(_tokenize_latin(text)) & _NEPALI_WORDS)


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
