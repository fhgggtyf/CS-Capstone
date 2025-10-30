"""
Forum Thread Fetcher
====================

This script crawls pages of the Blizzard Overwatch "General Discussion" forum
and extracts the first post from every thread listed on those pages.  It
prompts the user for the number of pages to crawl, visits each page in
sequence, collects thread URLs, fetches the corresponding thread page,
extracts the first post (ignoring replies) and stores relevant details in
an SQLite database.  The database path is defined by ``DB_PATH``.  If the
database or its directories do not exist they will be created automatically.

The script is self contained and makes no assumptions about previously
downloaded HTML.  It uses the Discourse ``data-preloaded`` payload to
reliably obtain post information without executing JavaScript.  Should
network access be unavailable or restricted, the script can also operate
on local HTML files by pointing ``BASE_FORUM_URL`` and ``BASE_THREAD_URL``
to file URLs instead of HTTP URLs.

Usage
-----
Run the script from a terminal::

    python fetcher.py

You will be prompted to enter the number of pages to scrape.  For each
page, the script will output progress information on the console.  At
completion it reports how many posts were stored.

Database Schema
---------------
The SQLite table ``overwatch_2_official_forum_posts`` (name can be changed freely) holds one
row per thread.  The columns are:

- ``thread_id`` – numerical identifier of the thread
- ``thread_title`` – title of the thread as shown on the forum index
- ``thread_url`` – absolute URL of the thread
- ``post_id`` – unique post identifier within Discourse
- ``username`` – forum username of the author
- ``created_at`` – UTC timestamp of when the post was created
- ``cooked_html`` – HTML-formatted post content
- ``post_text`` – plain‑text version of the post

If a thread already exists in the database (based on ``thread_id``) it
will be skipped to avoid duplicates.  This behaviour can easily be
modified by adjusting the unique constraint on the table.

Limitations
-----------
* The script only processes the first post of each thread.  Replies and
  subsequent pages within a thread are ignored by design.
* Network access is required to fetch live pages; if unreachable the
  script can operate on downloaded HTML files by adjusting the base URL
  variables.
* The script assumes the forum is powered by Discourse and that the
  ``data-preloaded`` payload contains the post stream.  If the forum
  software changes, this parser may need adjustments.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import re
import html
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

import glob
import urllib.parse


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Base URL of the forum category to scrape.  The ``{page}`` placeholder will
# be replaced with the page number.  For example, page 1 corresponds to
# ``?page=1``, page 2 to ``?page=2``, etc.  If you wish to test against a
# locally downloaded HTML file, set BASE_FORUM_URL to a file URI, e.g.:
# "file:///home/user/forum_page_{page}.html".
BASE_FORUM_URL = (
    "https://us.forums.blizzard.com/en/overwatch/c/general-discussion/6?page={page}"
)

# Optional directory containing locally downloaded HTML pages.  If network
# requests fail, the crawler will attempt to load thread pages from this
# directory based on their slug and ID.  For example, a thread URL like
# ``https://us.forums.blizzard.com/en/overwatch/t/example-thread/12345`` will
# look for a file matching ``*example-thread_12345__*.html`` within
# ``LOCAL_HTML_DIR``.  You can set this to ``None`` to disable the
# fallback mechanism.
LOCAL_HTML_DIR: Optional[str] = os.path.dirname(__file__)

DB_PATH = os.path.join("Data_Extraction", "Database", "Raw_Reviews.db")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PostRecord:
    """Container for first‑post information extracted from a thread."""

    thread_id: int
    thread_title: str
    thread_url: str
    post_id: int
    username: str
    created_at: str
    cooked_html: str
    post_text: str


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_db_initialized(conn: sqlite3.Connection) -> None:
    """Create the posts table if it doesn't already exist.

    The table uses ``thread_id`` as a unique key to prevent duplicate
    insertion when crawling multiple times.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the SQLite database.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS overwatch_2_official_forum_posts (
            thread_id INTEGER PRIMARY KEY,
            thread_title TEXT,
            thread_url TEXT,
            post_id INTEGER,
            username TEXT,
            created_at TEXT,
            cooked_html TEXT,
            post_text TEXT
        );
        """
    )
    conn.commit()

def _parse_iso8601(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    try:
        # handle Z suffix
        if s.endswith('Z'):
            return datetime.fromisoformat(s.replace('Z', '+00:00'))
        return datetime.fromisoformat(s)
    except Exception:
        return None

MAX_PAGE_GUARD = 6000  # safety to avoid infinite loops

def crawl_forum_after_cutoff(cutoff: datetime) -> None:
    """
    Crawl pages until all threads newer than the cutoff are processed.
    Stops when an entire page contains only threads older than the cutoff.
    """
    page = 0
    total_saved = 0

    while page <= MAX_PAGE_GUARD:
        forum_url = BASE_FORUM_URL.format(page=page)
        print(f"Fetching forum page {page}: {forum_url}")
        forum_html = fetch_url(forum_url)
        if not forum_html:
            print(f"  Unable to fetch/parse forum page {page}. Stopping.")
            break

        threads = extract_thread_links(forum_html)
        if not threads:
            print("  No threads found on this page. Stopping.")
            break

        print(f"Found {len(threads)} thread(s) on page {page}")
        page_had_newer = False

        for thread_url, thread_title, thread_id in threads:
            print(f"  Checking thread {thread_id}: {thread_title}...")
            thread_html = fetch_url(thread_url)
            if not thread_html and LOCAL_HTML_DIR:
                local_path = find_local_thread_file(thread_url)
                if local_path:
                    try:
                        with open(local_path, "r", encoding="utf-8") as f:
                            thread_html = f.read()
                    except OSError:
                        thread_html = None
            if not thread_html:
                print(f"    Failed to fetch thread {thread_id}")
                continue

            rec = parse_first_post(thread_html)
            if not rec:
                print(f"    Could not parse first post for thread {thread_id}")
                continue

            dt = _parse_iso8601(rec.created_at)
            if not dt:
                # If timestamp missing, keep it just in case
                is_recent_enough = True
            else:
                # ✅ Keep only posts created AFTER cutoff
                is_recent_enough = (dt > cutoff)

            if not is_recent_enough:
                print(f"    Skipping (on/before cutoff): {rec.created_at}")
                continue

            page_had_newer = True
            rec.thread_title = thread_title or rec.thread_title or ""
            rec.thread_url = thread_url or rec.thread_url or ""
            upsert_post_record(rec)
            total_saved += 1

        if not page_had_newer:
            print("  All threads on this page are on/before cutoff. Stopping.")
            break

        print(f"Completed page {page}\n")
        page += 1

    print(f"Done. Saved {total_saved} post(s) into the database at {DB_PATH}.")

def extract_thread_links(page_html: str) -> List[Tuple[str, str, int]]:
    """Parse a forum index page and return a list of thread links.

    Each returned tuple contains ``(thread_url, thread_title, thread_id)``.

    Parameters
    ----------
    page_html : str
        Raw HTML of the forum index page.

    Returns
    -------
    List[Tuple[str, str, int]]
        A list of triples containing the absolute URL, thread title and
        numerical thread ID.
    """
    soup = BeautifulSoup(page_html, "html.parser")
    threads: List[Tuple[str, str, int]] = []
    for link in soup.find_all("a", class_="title"):
        href: Optional[str] = link.get("href")
        title: str = link.get_text(strip=True)
        if not href:
            continue
        # Ensure absolute URL; Discourse returns absolute links in our case
        thread_url: str = href
        # Attempt to derive thread id from URL by splitting on the last '/'
        # and taking the final segment.  For example:
        # https://us.forums.blizzard.com/en/overwatch/t/foo-bar/12345 -> 12345
        thread_id = None
        try:
            thread_id = int(thread_url.rstrip("/").split("/")[-1])
        except (ValueError, IndexError):
            # If we cannot parse an ID, skip this link
            continue
        threads.append((thread_url, title, thread_id))
    return threads


def parse_first_post(thread_html: str) -> Optional[PostRecord]:
    """
    Extract the first post from a Discourse thread page.

    Supports both:
      A) App HTML: <div id="data-preloaded" data-preloaded="..."> (post_stream JSON)
      B) Crawler HTML: Schema.org microdata with div#post_1 .post[itemprop="text"]
    """
    soup = BeautifulSoup(thread_html, "html.parser")

    # ---------- Path A: app HTML via data-preloaded ----------
    preloaded_div = soup.find("div", id="data-preloaded")
    preloaded_attr = preloaded_div.get("data-preloaded") if preloaded_div else None
    if preloaded_attr:
        try:
            outer_json = json.loads(html.unescape(preloaded_attr))
            topic_key = next((k for k in outer_json if k.startswith("topic_")), None)
            if topic_key:
                topic_data = json.loads(outer_json[topic_key])
                posts = (topic_data.get("post_stream") or {}).get("posts") or []
                if posts:
                    first = posts[0]
                    cooked_html = html.unescape(first.get("cooked") or "")
                    cooked_soup = BeautifulSoup(cooked_html, "html.parser")
                    post_text = cooked_soup.get_text(separator=" ", strip=True)
                    # thread_id from "topic_<id>"
                    try:
                        thread_id = int(topic_key.split("_", 1)[1])
                    except Exception:
                        thread_id = 0
                    return PostRecord(
                        thread_id=thread_id,
                        thread_title="",
                        thread_url="",
                        post_id=first.get("id") or 1,
                        username=(first.get("username") or "").strip(),
                        created_at=(first.get("created_at") or "").strip(),
                        cooked_html=cooked_html,
                        post_text=post_text,
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            # fall through to crawler parsing
            pass

    # ---------- Path B: crawler HTML via Schema.org microdata ----------
    # Example structure:
    # <div itemscope itemtype='http://schema.org/DiscussionForumPosting'>
    #   <meta itemprop='headline' content='...'>
    #   <link itemprop='url' href='.../t/<slug>/<id>'>
    #   ...
    #   <div id='post_1' class='topic-body crawler-post'>
    #     <div class='post' itemprop='text'> ... first post html ... </div>
    #   </div>
    article = soup.find(
        "div",
        attrs={"itemscope": True, "itemtype": re.compile("DiscussionForumPosting", re.I)}
    )
    post_1 = soup.find("div", id="post_1")
    content_div = post_1.find("div", class_="post") if post_1 else None
    if not content_div and post_1:
        content_div = post_1.find("div", attrs={"itemprop": "text"})

    if article and content_div:
        meta_title = article.find("meta", attrs={"itemprop": "headline"})
        meta_url = article.find("link", attrs={"itemprop": "url"})
        thread_title = (meta_title.get("content").strip()
                        if meta_title and meta_title.has_attr("content") else "")
        thread_url = (meta_url.get("href").strip()
                      if meta_url and meta_url.has_attr("href") else "")

        # Extract author
        username = ""
        creator = post_1.find(attrs={"itemprop": "author"}) if post_1 else None
        if creator:
            nm = creator.find(attrs={"itemprop": "name"})
            if nm:
                username = nm.get_text(strip=True)

        # Timestamp
        created_at = ""
        t = post_1.find("time") if post_1 else None
        if t and t.has_attr("datetime"):
            created_at = t["datetime"].strip()

        cooked_html = str(content_div)
        post_text = content_div.get_text(separator=" ", strip=True)

        # Thread ID from URL tail digits
        thread_id = 0
        if thread_url:
            m = re.search(r"/(\d+)(?:$|[/?#])", thread_url)
            if m:
                thread_id = int(m.group(1))

        # Crawler HTML doesn't expose a numeric post id reliably; use 1
        return PostRecord(
            thread_id=thread_id,
            thread_title=thread_title,
            thread_url=thread_url,
            post_id=1,
            username=username,
            created_at=created_at,
            cooked_html=cooked_html,
            post_text=post_text,
        )

    # Neither path matched
    return None


def fetch_url(url: str) -> Optional[str]:
    """Retrieve the content at the given URL.

    Uses the ``requests`` library to perform an HTTP GET.  If the
    ``url`` points to a local file (scheme ``file://``), the file is
    read directly from disk.  Returns the response text on success or
    ``None`` on failure.

    Parameters
    ----------
    url : str
        URL or file URI to fetch.

    Returns
    -------
    Optional[str]
        The text content of the response, or ``None`` if retrieval fails.
    """
    if url.startswith("file://"):
        path = url[len("file://"):]
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def find_local_thread_file(thread_url: str) -> Optional[str]:
    """Search the local HTML directory for a file matching the thread slug.

    The thread URL is expected to follow the pattern
    ``.../t/<slug>/<id>``.  The search constructs a glob pattern of the
    form ``*_<slug>_<id>__*.html`` and searches ``LOCAL_HTML_DIR`` for
    matching files.  If multiple files match, the most recently modified
    one is returned.

    Parameters
    ----------
    thread_url : str
        The absolute URL of the thread on the forum.

    Returns
    -------
    Optional[str]
        The path to a matching local HTML file if found, otherwise ``None``.
    """
    if not LOCAL_HTML_DIR:
        return None
    # Parse the URL to extract the slug and id
    try:
        parts = urllib.parse.urlparse(thread_url)
        path_segments = parts.path.strip("/").split("/")
        # Expect structure: /en/overwatch/t/<slug>/<id>
        slug_index = path_segments.index("t") + 1
        slug = path_segments[slug_index]
        thread_id = path_segments[slug_index + 1]
    except Exception:
        return None
    pattern = f"*{slug}_{thread_id}__*.html"
    search_path = os.path.join(LOCAL_HTML_DIR, pattern)
    matches = glob.glob(search_path)
    if not matches:
        return None
    # Return the most recently modified file
    matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return matches[0]


def store_post(conn: sqlite3.Connection, record: PostRecord) -> None:
    """Insert a ``PostRecord`` into the database if it doesn't exist.

    Uses ``thread_id`` as the primary key, so duplicate threads will be
    skipped.  On unique constraint violation, the record is ignored.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the SQLite database.
    record : PostRecord
        The data to insert.
    """
    with conn:
        try:
            conn.execute(
                """
                INSERT INTO overwatch_2_official_forum_posts (
                    thread_id, thread_title, thread_url, post_id,
                    username, created_at, cooked_html, post_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.thread_id,
                    record.thread_title,
                    record.thread_url,
                    record.post_id,
                    record.username,
                    record.created_at,
                    record.cooked_html,
                    record.post_text,
                ),
            )
        except sqlite3.IntegrityError:
            # Duplicate entry, ignore
            pass

def upsert_post_record(record: PostRecord):
    """Backwards-compatibility wrapper for store_post without an open conn."""
    conn = sqlite3.connect(DB_PATH)
    ensure_db_initialized(conn)
    store_post(conn, record)
    conn.close()

def crawl_forum(num_pages: int) -> None:
    """Orchestrate crawling of the forum and saving posts to the database.

    Parameters
    ----------
    num_pages : int
        Number of forum pages to traverse starting from 0.
    """
    # Ensure DB directory exists
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    ensure_db_initialized(conn)
    total_saved = 0
    for page in range(0, num_pages):
        page_url = BASE_FORUM_URL.format(page=page)
        print(f"Fetching forum page {page}: {page_url}")
        page_html = fetch_url(page_url)
        if not page_html:
            print(f"Failed to retrieve page {page}")
            continue
        threads = extract_thread_links(page_html)
        if not threads:
            print(f"No threads found on page {page}")
            continue
        print(f"Found {len(threads)} thread(s) on page {page}")
        for thread_url, thread_title, thread_id in threads:
            print(f"  Processing thread {thread_id}: {thread_title}...")
            thread_html = fetch_url(thread_url)
            if not thread_html:
                # Attempt to load from local HTML directory
                local_path = find_local_thread_file(thread_url)
                if local_path:
                    try:
                        with open(local_path, "r", encoding="utf-8") as f:
                            thread_html = f.read()
                        print(
                            f"    Loaded thread {thread_id} from local file {os.path.basename(local_path)}"
                        )
                    except OSError:
                        thread_html = None
                if not thread_html:
                    print(f"    Failed to fetch thread {thread_id}")
                    continue
            record = parse_first_post(thread_html)
            if not record:
                print(f"    Could not parse first post in thread {thread_id}")
                continue
            # Attach title and URL from index page
            record.thread_title = thread_title
            record.thread_url = thread_url
            store_post(conn, record)
            total_saved += 1
        print(f"Completed page {page}\n")
    conn.close()
    print(f"Done. Saved {total_saved} post(s) into the database at {DB_PATH}.")

def main():
    raw = input("How many forum pages to fetch? (number or 'max' for after Aug 10, 2023): ").strip().lower()
    if raw == "max":
        cutoff = datetime(2023, 8, 10, tzinfo=timezone.utc)
        crawl_forum_after_cutoff(cutoff)
    else:
        try:
            num_pages = int(raw)
        except ValueError:
            print("Please enter a positive integer or 'max'.")
            raise SystemExit(1)
        if num_pages <= 0:
            print("Please enter a positive integer.")
            raise SystemExit(1)
        crawl_forum(num_pages)



if __name__ == "__main__":
    main()