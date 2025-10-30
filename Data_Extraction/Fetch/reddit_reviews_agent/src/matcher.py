import re
from typing import Dict, List, Optional

def normalize(text: str) -> str:
    """
    Normalize a string for game matching.

    The matcher relies on simple substring checks and Levenshtein distances
    against a normalized version of the input and candidate aliases.  In
    addition to lower‑casing and stripping punctuation, we also replace
    underscores with spaces.  This is important because some game names or
    aliases may contain underscores (e.g. "Counter_Strike 2" or user
    shorthand from URLs), and without replacing underscores they would never
    match the canonical name or alias.  We also remove smart quotes,
    apostrophes, trademark symbols, commas and colons for consistency.
    """
    t = (text or "").lower()
    # Replace underscores with spaces so that underscore‑separated names match
    # their space‑separated equivalents (e.g. "counter_strike" -> "counter strike").
    t = t.replace("_", " ")
    for ch in ["’", "'", "™", ",", ":"]:
        t = t.replace(ch, "")
    return t

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if len(a) == 0: return len(b)
    if len(b) == 0: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost))
        prev = cur
    return prev[-1]

def match_game(text: str, canon2aliases: Dict[str, List[str]], order: List[str], fuzzy_max_distance: int, require_platform_near_fuzzy: bool, platform_words: Optional[List[str]] = None) -> Optional[str]:
    t = normalize(text)
    # exact
    if "exact" in order:
        for canon, aliases in canon2aliases.items():
            if normalize(canon) in t:
                return canon
    # alias
    if "alias" in order:
        for canon, aliases in canon2aliases.items():
            for a in aliases:
                if normalize(a) in t:
                    return canon
    # fuzzy
    if "fuzzy" in order:
        words = re.findall(r"[a-z0-9][a-z0-9\-']+", t)
        for canon, aliases in canon2aliases.items():
            tokens = [normalize(canon)] + [normalize(a) for a in aliases]
            for w in words:
                for tok in tokens:
                    if levenshtein(w, tok) <= fuzzy_max_distance:
                        if require_platform_near_fuzzy and platform_words:
                            # Simple proximity check: platform word in same line span
                            line = w
                            if any(p in t for p in platform_words):
                                return canon
                        else:
                            return canon
    return None

# add at top if not present
import re

def slug_table(game: str) -> str:
    """
    Build a SQLite-safe table name from a game title:
    - lowercase
    - remove trademark-like glyphs
    - replace any non [a-z0-9_] char with underscore
    - collapse runs of underscores
    - ensure it doesn't start with a digit
    """
    s = (game or "").lower()
    # normalize some common glyphs
    s = s.replace("’", "'").replace("™", "")
    # replace any non [a-z0-9_] with underscore
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    # collapse multiple underscores and trim ends
    s = re.sub(r"_+", "_", s).strip("_")
    # ensure does not start with a digit
    if re.match(r"^\d", s):
        s = "t_" + s
    return f"{s}_reddit_game_reviews"

