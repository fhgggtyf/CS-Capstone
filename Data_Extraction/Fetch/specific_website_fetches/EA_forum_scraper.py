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
BOARD_ID = "board:battlefield-2042-general-discussion-en"

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
AUTH_BEARER = "Bearer qKmUFJcRcwvO3+jFyaumXNZfGxYepyMktIQJxfNCcww="

# Copy the ENTIRE Cookie header from the successful request in DevTools (keep it fresh)
COOKIE_HEADER = "ealocale=en-us; _gcl_au=1.1.1832194846.1754835600; _ga=GA1.1.759818092.1754835600; _scid=lmEkvbKPQXpzevq_MSl3wKx24D1TF9jS; _fbp=fb.1.1754835608274.624281566436461516; _tt_enable_cookie=1; _ttp=01K2A5MSFXJMVB91YKPFFFCJAJ_.tt.1; notice_preferences=2:; notice_gdpr_prefs=0,1,2:; notice_poptime=1599001200000; cmapi_gtm_bl=; cmapi_cookie_privacy=permit 1,2,3; notice_preferences=2:; notice_gdpr_prefs=0,1,2:; notice_poptime=1599001200000; cmapi_gtm_bl=; cmapi_cookie_privacy=permit 1,2,3; LithiumLocalePreferences=en-US; LithiumTimezonePreferences=US%2FPacific; ak_bmsc=E0466A283A955465CE2B8B4A5E047588~000000000000000000000000000000~YAAQZ+cVAigAt0qaAQAAKPTsTx0JNVsygQV22auYbiNvn+brmaSmAVnvwaoRaMMT9G6YUG6KAW0D8s2Zedi1QyE5v2JLS7YCw5jWZ/2F3f0EpzKChS77Py6bUcnrCAIs8UmDUGOWqhailfqCik3XJ4Qb0NMPm2/DFcn8OWynrCN5HhEB3ckuAVpGOA+8I7HGmKwOWM1s/vG2Gmp73VjgiOAgkyZ1pAhEfWzgsaTIcI+oUn2rKFp635wI3F2+CWpHiHrLWUd23I/V81SsSWrUGSuMlLHeoz3jMzezWptDIMpizOBTd2/m3ZVSgPn42q+yMAaf8pGEjxC6Yc5M5yuJwHDpuAJ8nXUogCWHN4ecssLlnDx5kKueyfZP0NWvalf/uSEjY/Ez; TAsessionID=173dbdaa-0385-4c90-a5e5-a3dc05c8fc87|EXISTING; notice_behavior=implied,us; notice_location=ae; bm_sv=2FE9D7ACD106D65190AD57F61118B1B2~YAAQZ+cVAoQAt0qaAQAAQvjsTx17tG3k3DVF+DC9NWeUUCOvq13B9uhQdTYXd05nF2/2bejIh4mzxjcIWSNdK2yOaa54NYKixwcvaKuQnxduNpm6HgkFw3MjW+lQXMiE3reOk+lyXDSSyW9Da6JtADGtzUUC1PwMy2lylPrGqj8GCbM2mW802w7crjti300XWXutRqPBL6MTxf7E8+OeGpWmXPagay+1kH2YwH2kJU2JV4SZAK8BkCNfXac=~1; _scid_r=ouEkvbKPQXpzevq_MSl3wKx24D1TF9jS_BGEAg; _rdt_uuid=1762277522633.7d74b99c-a61b-4c51-8833-3493fa8856b6; _ScCbts=%5B%5D; _sctr=1%7C1762200000000; ttcsid=1762277522570::iD5b7pBQGD5bdxVSf7iy.3.1762277528213.0; ttcsid_D1NCOORC77U9OS2TO8P0=1762277522570::lH1kewCyIt1zRIVwrnJ1.3.1762277528213.0; _ga_Q3MDF068TF=GS2.1.s1762277522$o5$g1$t1762277538$j44$l0$h0; LiSESSIONID=0A9C6CC3CA35A2F4FB64D1D54CA1AEE4; csrf-aurora=d68a27e43e82788de0dced1e3a4c34be23300a8fb1e93809c09fe5cdfc4d2548f470712e8775cabc95258ea57e72ee7ccc752991b33723318cfc48b382d53687%7C8c1e2f648357abf6de23892e02518d72d8576ab1d5ade0cdb2db3b871207fff6; notice_behavior=implied,us; notice_location=ae; _sg_b_v=17%3B563273%3B1762277560; _sg_b_p=%2Fcategory%2Fbattlefield-en%2Fdiscussions%2Fbattlefield-2042-general-discussion-en%2C%2Fcategory%2Fbattlefield-en%2Fdiscussions%2Fbattlefield-2042-general-discussion-en; _sg_b_n=1762277580245; AWSALB=0MXI90E4CppjFD4w7vf9+Uax21e/Wq4Gvvz141yrsGJbUqiH64aWoWRQzitCkUwn/Tm0nlRjRM21uaaCTF+TbeQY8AYdzKpyfWr9HrlAqnhtmAzWZwdeINCACdCj; AWSALBCORS=0MXI90E4CppjFD4w7vf9+Uax21e/Wq4Gvvz141yrsGJbUqiH64aWoWRQzitCkUwn/Tm0nlRjRM21uaaCTF+TbeQY8AYdzKpyfWr9HrlAqnhtmAzWZwdeINCACdCj; LithiumVisitor=~24lu445aTWqZN8g66~n3S92t4_9a8_bKX2WNr5WSHcU4AWBLVTTN7A3nz8W3uUP6My0WkAaKdjSMG0NObl2xhuitLn1Ed-AyyfdhY2FQ..; _ga_P5SW3QHCQJ=GS2.1.s1762277561$o14$g1$t1762277584$j37$l0$h0"

# DB target and table
DB_PATH = r"Data_Extraction/Database/Raw_Reviews.db"
TABLE_NAME = "battlefield_2042_general_discussion_en_official_EA_forum_posts"

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

def two_year_cutoff(now_utc: Optional[datetime] = None) -> datetime:
    now_utc = now_utc or datetime.now(timezone.utc)
    if relativedelta:
        return now_utc - relativedelta(years=2)
    # fallback if dateutil not installed
    return now_utc - timedelta(days=365*2)

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

def collect_until_2y(session, page_size=50):
    """
    Fetch pages (DESC by postTime) and stop as soon as we
    encounter a post older than 2 years.
    Returns the rows kept (only within range).
    """
    rows_all = []
    cursor = None
    cutoff = two_year_cutoff()  # now - 2 years (UTC)

    while True:
        edges, cursor, has_next = fetch_page(session, after_cursor=cursor, page_size=page_size)
        if not edges:
            break

        keep_edges = []
        hit_cutoff = False
        cutoff_counter = 0

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
                cutoff_counter += 1
                print(f"⏳ Hit post older than 2 years: {dt.isoformat()} < {cutoff.isoformat()}, {cutoff_counter} so far in batch.")
                if cutoff_counter >= 40:
                    hit_cutoff = True
                    print("🚫 Reached cutoff threshold for this batch, stopping further fetches.")
                    break

            keep_edges.append(e)

        batch_rows = to_rows_from_edges(keep_edges)
        insert_rows(DB_PATH, TABLE_NAME, batch_rows)
        rows_all.extend(batch_rows)

        print(f"Fetched {len(edges)} posts, kept {len(keep_edges)} within 2 years, total now {len(rows_all)}.")

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
    rows = collect_until_2y(session, page_size=50)  # set your desired per-page size
    print(f"Inserted {len(rows)} reviews within 2 years.")

if __name__ == "__main__":
    main()
