#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pagination pull for EA Forums GraphQL (MessageViewsForWidget)
- Replays the browser's GraphQL request with your headers/cookies
- Paginates using pageInfo.endCursor until hasNextPage = False
- Extracts review HTML/text + metadata, stores to SQLite
"""

import json
import time
import html
import sqlite3
from datetime import datetime
import requests
from typing import Optional
from datetime import datetime, timezone, timedelta

# ---------------------- CONFIG: FILL THESE IN ----------------------

GRAPHQL_URL = "https://forums.ea.com/t5/s/api/2.1/graphql?opname=MessageViewsForWidget"
PERSISTED_QUERY_HASH = "9e08d498b0a03960a32846316b983adc71a637e2133124196f0073e35b521495"
BOARD_ID = "board:fc-25-general-discussion-en"

BASE_VARIABLES = {
    "useAvatar": True,
    "useAuthorRank": True,
    "useBody": True,
    "useTextBody": True,
    "useKudosCount": True,
    "useTimeToRead": True,
    "useMedia": True,
    "useReadOnlyIcon": True,
    "useRepliesCount": True,
    "useSearchSnippet": False,
    "useSolvedBadge": True,
    "useFullPageInfo": False,
    "useTags": True,
    "tagsFirst": 10,
    "tagsAfter": None,
    "truncateBodyLength": -1,
    "useSpoilerFreeBody": True,
    "removeTocMarkup": True,
    "usePreviewSubjectModal": False,
    "useOccasionData": False,
    "useMessageStatus": False,
    "removeProcessingText": True,
    "useUnreadCount": False,
    "first": 20,
    "constraints": {
        "boardId": {"eq": BOARD_ID},
        "depth": {"eq": 0},
        "conversationStyle": {"eq": "FORUM"}
    },
    "sorts": {"conversationLastPostingActivityTime": {"direction": "DESC"}},
    "after": None,   # set to endCursor to resume
    "before": None,
    "last": None
}

# Copy the "Authorization: Bearer ..." from your DevTools (keep it fresh)
AUTH_BEARER = "Bearer HoUrA5qejdagsXSimNeX8x4RC2kI1SyiWtjEwTQu/Wc="

# Copy the ENTIRE Cookie header from the successful request in DevTools (keep it fresh)
COOKIE_HEADER = "ealocale=en-us; _gcl_au=1.1.1832194846.1754835600; _ga=GA1.1.759818092.1754835600; _scid=lmEkvbKPQXpzevq_MSl3wKx24D1TF9jS; _fbp=fb.1.1754835608274.624281566436461516; _tt_enable_cookie=1; _ttp=01K2A5MSFXJMVB91YKPFFFCJAJ_.tt.1; notice_location=ae; notice_behavior=implied,us; _ScCbts=%5B%221%3Bchrome.2%3A2%3A5%22%2C%22183%3Bchrome.2%3A2%3A5%22%5D; _sctr=1%7C1756584000000; csrf-aurora=e6925f4a16edb648ba98162937d33cf0f319ff979af460d32545f9d27024aaa409ab7a22b7b6391ae4738fc709e3b6ea434d23b4b648f82153a55c030e1653ad%7Ca621610e27f04e2e065661702639bd4b0b1c8a368fc78c0efe5eb6bd0d7dbd15; notice_location=ae; notice_behavior=implied,us; notice_preferences=2:; notice_gdpr_prefs=0,1,2:; notice_poptime=1599001200000; cmapi_gtm_bl=; cmapi_cookie_privacy=permit 1,2,3; _scid_r=oeEkvbKPQXpzevq_MSl3wKx24D1TF9jS_BGEAQ; ttcsid=1756661620697::QXjTnCoOJpU8_Xrw9ReF.2.1756661877352; notice_preferences=2:; notice_gdpr_prefs=0,1,2:; notice_poptime=1599001200000; cmapi_gtm_bl=; cmapi_cookie_privacy=permit 1,2,3; ttcsid_D1NCOORC77U9OS2TO8P0=1756661620697::m0tXZiSydtgfT3_FBlFc.2.1756661909591; _ga_Q3MDF068TF=GS2.1.s1756661620$o3$g1$t1756665028$j54$l0$h0; _sg_b_n=1756680058921; close_eaforum_survey_disable=true; LithiumLocalePreferences=en-US; LithiumTimezonePreferences=US%2FPacific; LiSESSIONID=617D94F5E319AAB2F2A00AA23E1546C0; TAsessionID=95981110-e64c-4089-85a6-b01b8a2b12f4,EXISTING; _sg_b_v=11%3B84222%3B1756764346; AWSALB=XZcG6st+IMP6U+fDBPmNH5hpK8w7jqFUeJFdcehUIaaXXbxGYEfbhYx+spoyg/6+K6hjP2LrwRSYyEvVBfVNqHoftV+isPu4mehKLAzpMSraSNR9WycSae3D/E3d; AWSALBCORS=XZcG6st+IMP6U+fDBPmNH5hpK8w7jqFUeJFdcehUIaaXXbxGYEfbhYx+spoyg/6+K6hjP2LrwRSYyEvVBfVNqHoftV+isPu4mehKLAzpMSraSNR9WycSae3D/E3d; LithiumVisitor=~2wjQT9lEPK1yp8JuD~N7YM9hyUcmXdxT0PM1AXw1S7fkXdxl0UEIFvQSLs8wBNu5k0hfpLZ3Rmuu0srYrWjvAEgDdJCoAe9dPAvDUEjA..; _sg_b_p=%2Fcategory%2Ffc-25-en%2C%2Fcategory%2Fapex-legends-en%2C%2Fcategory%2Fapex-legends-en%2Fdiscussions%2Fapex-legends-general-discussion-en%2C%2Fcategory%2Fapex-legends-en%2C%2Fcategory%2Fapex-legends-en%2Fdiscussions%2Fapex-legends-feedback-en%2C%2Fcategory%2Fapex-legends-en%2C%2Fcategory%2Fapex-legends-en%2Fdiscussions%2Fapex-legends-general-discussion-en%2C%2Fcategory%2Fapex-legends-en%2Fdiscussions%2Fapex-legends-general-discussion-en%2C%2Fcategory%2Ffc-25-en%2Fdiscussions%2Ffc-25-general-discussion-en%2C%2Fcategory%2Ffc-25-en%2Fdiscussions%2Ffc-25-general-discussion-en; _ga_P5SW3QHCQJ=GS2.1.s1756764344$o8$g1$t1756765320$j56$l0$h0"

# DB target and table
DB_PATH = r"Data_Extraction/Database/CS_Capstone.db"
TABLE_NAME = "fc_25_general_discussion_en_official_EA_forum_posts"

# Throttle between requests (avoid hammering)
REQUEST_DELAY_SEC = 0.4

# -------------------------------------------------------------------

from datetime import datetime, timezone
try:
    from dateutil.relativedelta import relativedelta  # nicer than 365*3
except Exception:
    relativedelta = None

def parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # handle "Z" and timezone offsets
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def three_year_cutoff(now_utc: Optional[datetime] = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    if relativedelta:
        return now_utc - relativedelta(years=3)
    # fallback if dateutil not installed
    return now_utc - timedelta(days=365*3)

def ensure_table(db_path: str, table_name: str) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name}(
            id TEXT PRIMARY KEY,          -- e.g. "message:123456"
            uid INTEGER,
            title TEXT,
            author TEXT,
            author_rank TEXT,
            post_date TEXT,
            body_html TEXT,
            body_text TEXT,
            replies_count INTEGER,
            tags TEXT
        );
    """)
    conn.commit()
    conn.close()


def normalise_time(ts: str) -> str:
    """Normalise ISO-ish timestamps to 'YYYY-MM-DD HH:MM:SS' if possible."""
    if not ts:
        return ""
    try:
        # Example: "2025-07-22T08:25:00.425-07:00"
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ts


def to_rows_from_edges(edges):
    """Extract review rows from the common GraphQL edge list."""
    rows = []
    for e in edges:
        n = e.get("node", {})
        author = (n.get("author") or {}).get("login", "")
        author_rank = ((n.get("author") or {}).get("rank") or {}).get("name", "")
        body_html = n.get("body") or ""              # the review HTML
        body_text = html.unescape(body_html).strip() # simple decode; keep HTML too
        replies_count = (
            ((n.get("conversation") or {}).get("topic") or {}).get("repliesCount")
            or n.get("repliesCount") or 0
        )

        # tags may appear as connection with edges; flatten if present
        tag_edges = ((n.get("tags") or {}).get("edges") or [])
        tags = ",".join([ (te.get("node") or {}).get("text","") for te in tag_edges if (te.get("node") or {}).get("text") ])

        rows.append((
            n.get("id", ""),                 # id
            n.get("uid", None),              # uid
            n.get("subject", ""),            # title/subject
            author,                          # author
            author_rank,                     # author rank
            normalise_time(n.get("postTime") or n.get("lastPublishTime")),
            body_html,
            body_text,
            int(replies_count) if replies_count is not None else 0,
            tags
        ))
    return rows


def insert_rows(db_path: str, table_name: str, rows):
    if not rows:
        return
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executemany(
        f"""INSERT OR REPLACE INTO {table_name}
            (id, uid, title, author, author_rank, post_date, body_html, body_text, replies_count, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows
    )
    conn.commit()
    conn.close()


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "content-type": "application/json",
        "authorization": AUTH_BEARER,
        "cookie": COOKIE_HEADER,
        # Optional but mirrors browser
        "origin": "https://forums.ea.com",
        "referer": "https://forums.ea.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9",
    })
    return s


import json

def _parse_edges_page(data: dict):
    """Supports either data.messageViewsForWidget.messages or data.messages shapes."""
    dd = (data.get("data") or {})
    container = (
        ((dd.get("messageViewsForWidget") or {}).get("messages"))
        or dd.get("messages")
    )
    if not container:
        # helpful diagnostic
        raise RuntimeError(f"Unexpected response shape: {json.dumps(data)[:800]}")
    edges = container.get("edges", []) or []
    page_info = container.get("pageInfo", {}) or {}
    end_cursor = page_info.get("endCursor") or page_info.get("endcursor")
    has_next = bool(page_info.get("hasNextPage"))
    return edges, end_cursor, has_next

def fetch_page(session, after_cursor=None, page_size=None):
    """
    POST one GraphQL page using your persisted query + variables.
    - session: requests.Session with your live cookies/auth headers.
    - after_cursor: str|None (use None to start; then pass endCursor each time)
    - page_size: int|None (override 'first' if you want a different per-page size)
    """
    vars_copy = dict(BASE_VARIABLES)
    if page_size is not None:
        vars_copy["first"] = int(page_size)
    vars_copy["after"] = after_cursor

    payload = {
        "operationName": "MessageViewsForWidget",
        "variables": vars_copy,
        "extensions": {
            "persistedQuery": {
                "version": 1,
                "sha256Hash": PERSISTED_QUERY_HASH
            }
        }
        # no "query" field needed if the hash exists on the server;
        # if you ever see PersistedQueryNotFound, retry with the full 'query' text.
    }

    r = session.post(GRAPHQL_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Basic error surface
    if "errors" in data and data["errors"]:
        raise RuntimeError(f"GraphQL error: {data['errors']}")

    edges, end_cursor, has_next = _parse_edges_page(data)

    # Hard-guard: only keep posts from this board (extra safety)
    edges = [
        e for e in edges
        if (((e.get("node") or {}).get("board") or {}).get("id")) == BOARD_ID
    ]

    return edges, end_cursor, has_next

def collect_until_3y(session, page_size=50):
    """
    Fetch pages (DESC by postTime) and stop as soon as we
    encounter a post older than 3 years.
    Returns the rows kept (only within range).
    """
    rows_all = []
    cursor = None
    cutoff = three_year_cutoff()  # now - 3 years (UTC)

    while True:
        edges, cursor, has_next = fetch_page(session, after_cursor=cursor, page_size=page_size)
        if not edges:
            break

        keep_edges = []
        hit_cutoff = False

        # edges are newest→oldest (postTime DESC)
        for e in edges:
            n = e.get("node", {})
            ts = n.get("postTime") or n.get("lastPublishTime")
            dt = parse_iso(ts)

            if dt is None:
                continue

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            if dt < cutoff:
                hit_cutoff = True
                break

            keep_edges.append(e)

        batch_rows = to_rows_from_edges(keep_edges)
        insert_rows(DB_PATH, TABLE_NAME, batch_rows)
        rows_all.extend(batch_rows)

        print(f"Fetched {len(edges)} posts, kept {len(keep_edges)} within 3 years, total now {len(rows_all)}.")

        if hit_cutoff or not has_next:
            break

    return rows_all

def main():
    ensure_table(DB_PATH, TABLE_NAME)

    session = make_session()

    # (optional but recommended) move raw Cookie header into the session jar:
    for kv in COOKIE_HEADER.split(";"):
        if "=" in kv:
            name, value = kv.strip().split("=", 1)
            session.cookies.set(name, value, domain="forums.ea.com", path="/")
    session.headers.pop("cookie", None)

    # Fetch until crossing the 3-year cutoff
    rows = collect_until_3y(session, page_size=50)  # set your desired per-page size
    print(f"Inserted {len(rows)} reviews within 3 years.")

if __name__ == "__main__":
    main()
