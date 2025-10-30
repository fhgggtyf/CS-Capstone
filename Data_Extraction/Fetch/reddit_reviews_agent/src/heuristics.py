import re

REVIEW_RE = re.compile(r"(review|impressions|first impressions|my thoughts|finished|beat|postmortem|should i buy|worth it|mini review|returning to|after\s+\d+\s*hours)", re.I)
PLATFORM_RE = re.compile(r"\b(pc|steam|ps5|ps4|playstation|xbox(?: series [sx])?|series [sx]|switch|nintendo)\b", re.I)
PLAYTIME_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(hours|hrs|h)\b", re.I)

def is_english(s: str, min_ratio=0.85, min_words=80):
    if not s:
        return False
    ascii_ratio = sum(1 for ch in s if ord(ch) < 128) / max(1, len(s))
    return ascii_ratio >= min_ratio and len(s.split()) >= min_words

def looks_like_review(title: str, body: str, flair_text: str, game_match: bool, min_words=150):
    if REVIEW_RE.search(title or "") or REVIEW_RE.search(flair_text or ""):
        return True
    return game_match and len((body or "").split()) >= min_words

def extract_platform(s: str):
    m = PLATFORM_RE.search(s or "")
    return m.group(0).lower() if m else None

def extract_playtime(s: str):
    m = PLAYTIME_RE.search(s or "")
    try:
        return float(m.group(1)) if m else None
    except Exception:
        return None
