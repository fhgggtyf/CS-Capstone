import time
import logging
from typing import Iterable, Dict, Any, Optional
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type

import praw
import prawcore

log = logging.getLogger(__name__)

def make_reddit(cfg: dict) -> praw.Reddit:
    api = cfg["reddit_api"]
    reddit = praw.Reddit(
        client_id=api["client_id"],
        client_secret=api["client_secret"],
        user_agent=api["user_agent"],
        ratelimit_seconds=int(60 / max(1, api.get("max_requests_per_min", 60)))
    )
    return reddit

def _rate_sleep(max_per_min: int):
    # naive throttle: sleep to roughly cap requests per minute
    delay = max(0.0, 60.0 / max(1, max_per_min) * 0.6)
    if delay:
        time.sleep(delay)

@retry(wait=wait_exponential_jitter(2, 10), stop=stop_after_attempt(5), reraise=True, retry=retry_if_exception_type((prawcore.exceptions.RequestException, prawcore.exceptions.ResponseException, prawcore.exceptions.ServerError, prawcore.exceptions.Forbidden)))
def fetch_listing(subreddit, mode: str, limit: int):
    if mode == "new":
        return list(subreddit.new(limit=limit))
    if mode == "top_day":
        return list(subreddit.top(time_filter="day", limit=limit))
    if mode == "top_week":
        return list(subreddit.top(time_filter="week", limit=limit))
    raise ValueError(f"Unknown mode {mode}")

@retry(wait=wait_exponential_jitter(2, 10), stop=stop_after_attempt(5), reraise=True, retry=retry_if_exception_type((prawcore.exceptions.RequestException, prawcore.exceptions.ResponseException, prawcore.exceptions.ServerError, prawcore.exceptions.Forbidden)))
def search_posts(subreddit, query: str, limit: int):
    return list(subreddit.search(query=query, limit=limit, sort="new", syntax="lucene"))
