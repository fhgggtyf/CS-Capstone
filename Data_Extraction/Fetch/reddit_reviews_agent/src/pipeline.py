import logging, time, re
from typing import Dict, List, Tuple
from datetime import datetime, timezone
from html import unescape

from .db import DB
from .heuristics import is_english, looks_like_review, extract_platform, extract_playtime, REVIEW_RE, PLATFORM_RE
from .matcher import match_game, slug_table
from .util import now_epoch, within_window
from .reddit_client import make_reddit, fetch_listing, search_posts

log = logging.getLogger(__name__)

def collect(cfg: dict):
    reddit = make_reddit(cfg)
    db = DB(cfg["project"]["db_path"])

    # time window
    start = int(datetime.fromisoformat(cfg["collection"]["time_window"]["start_utc"].replace("Z","+00:00")).timestamp())
    end   = int(datetime.fromisoformat(cfg["collection"]["time_window"]["end_utc"].replace("Z","+00:00")).timestamp())

    games = cfg["games"]
    heur  = cfg["heuristics"]
    subcfg = cfg["subreddits"]
    pull_modes = cfg["collection"]["pull_modes"]
    limits = cfg["collection"]["per_subreddit_limits"]
    comment_depth = int(cfg["collection"]["comment_depth"])

    # precompute platform words for fuzzy proximity
    platform_words = [w.lower() for w in ["pc","steam","ps5","ps4","xbox","series s","series x","switch","nintendo"]]

    # Build subreddit set per game
    broad = set([s for s in subcfg.get("broad", [])])
    overrides = set([s for s in subcfg.get("overrides", [])])

    for canon_game, aliases in games.items():
        table = slug_table(canon_game)
        db.ensure_game_table(table)

        game_subs = set(subcfg.get("game_specific", {}).get(canon_game, []))
        subs = sorted(broad | game_subs | overrides)

        log.info(f"[{canon_game}] Subreddits: {subs} -> table {table}")

        # Build search queries per game
        title_re_q = cfg["heuristics"]["review_title_or_flair_regex"]
        search_q = f'title:"{canon_game}" (review OR impressions OR "my thoughts" OR finished OR beat OR postmortem OR "should I buy" OR "worth it" OR "mini review")'

        for sub_name in subs:
            sub = reddit.subreddit(sub_name)

            # 1) listing pulls
            for mode in pull_modes:
                if mode == "search":
                    continue
                limit = int(limits.get(mode, 100))
                try:
                    posts = fetch_listing(sub, mode, limit)
                except Exception as e:
                    log.warning(f"Fetch listing failed for r/{sub_name} mode={mode}: {e}")
                    continue
                _process_posts(cfg, db, table, canon_game, games, posts, start, end, comment_depth, platform_words)

            # 2) search pulls
            limit = int(limits.get("search", 200))
            try:
                posts = search_posts(sub, search_q, limit)
            except Exception as e:
                log.warning(f"Search failed for r/{sub_name}: {e}")
                continue
            _process_posts(cfg, db, table, canon_game, games, posts, start, end, comment_depth, platform_words)

        db.commit()

    db.close()


def _process_posts(cfg, db: DB, table_name: str, canon_game: str, canon2aliases: Dict[str, List[str]], posts, start, end, comment_depth, platform_words):
    heur = cfg["heuristics"]
    min_words = int(heur["min_words_for_body_review"])
    ascii_ratio = float(heur["english_ascii_ratio_min"])

    count_kept = 0
    for p in posts:
        try:
            created = int(p.created_utc)
        except Exception:
            continue
        if not within_window(created, start, end):
            continue

        title = unescape(getattr(p, "title", "") or "")
        body  = unescape(getattr(p, "selftext", "") or "")
        flair = (getattr(p, "link_flair_text", "") or "").lower()
        sub   = str(getattr(p, "subreddit", ""))
        author= str(p.author) if p.author else None
        url   = f"https://reddit.com{p.permalink}"
        score = int(getattr(p, "score", 0))
        num_comments = int(getattr(p, "num_comments", 0))

        # Combine title and body with a double newline separator.  Using an
        # explicit newline sequence avoids syntax errors in the f-string.
        text = f"{title}\n\n{body}"
        # game match
        game = match_game(text, canon2aliases, cfg["matching"]["order"], int(cfg["matching"]["fuzzy_max_distance"]), bool(cfg["matching"]["require_platform_near_fuzzy"]), platform_words)
        if game != canon_game:
            # allow exact title mention of canon_game in title only as fallback
            if canon_game.lower() not in title.lower():
                continue

        eng = is_english(text, min_ratio=ascii_ratio, min_words=80)
        looks = looks_like_review(title, body, flair, game_match=True, min_words=min_words)

        if not (eng and looks):
            continue

        platform = extract_platform(text)
        hrs = extract_playtime(text)

        row = (
            f"t3_{p.id}", "post", sub, author, created, title, body, url, score, num_comments, None,
            canon_game, platform, hrs, int(looks), int(eng), now_epoch(), cfg["project"]["src_version"]
        )
        db.upsert_row(table_name, row)
        count_kept += 1

        if comment_depth and num_comments:
            _process_comments(cfg, db, table_name, canon_game, p, start, end)

    log.info(f"[{canon_game}] kept posts in batch: {count_kept}")


def _process_comments(cfg, db: DB, table_name: str, canon_game: str, submission, start, end):
    heur = cfg["heuristics"]
    min_words = int(heur["min_words_for_body_review"])
    ascii_ratio = float(heur["english_ascii_ratio_min"])

    try:
        submission.comments.replace_more(limit=0)
    except Exception as e:
        log.warning(f"replace_more failed: {e}")
        return

    for c in submission.comments.list():
        try:
            created = int(c.created_utc)
        except Exception:
            continue
        if not within_window(created, start, end):
            continue

        body = unescape(getattr(c, "body", "") or "")
        if not body:
            continue
        eng = is_english(body, min_ratio=ascii_ratio, min_words=80)
        if not eng:
            continue

        looks = len(body.split()) >= min_words  # simple rule for comments
        if not looks:
            continue

        platform = extract_platform(body)
        hrs = extract_playtime(body)

        row = (
            f"t1_{c.id}", "comment", str(submission.subreddit), str(c.author) if c.author else None,
            int(c.created_utc), None, body, f"https://reddit.com{c.permalink}", int(getattr(c, "score", 0)), None,
            c.parent_id, canon_game, platform, hrs, int(looks), int(eng), now_epoch(), cfg["project"]["src_version"]
        )
        db.upsert_row(table_name, row)
    db.commit()
