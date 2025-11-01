from __future__ import annotations

#!/usr/bin/env python3
"""
Steam Review Scraper
====================

This script downloads user reviews from the Steam storefront for a given
application (game) and saves the results into a SQLite database.  It is
designed to repeatedly fetch pages of the most recent reviews until the
oldest review collected falls outside of a user‑defined time window.  The
user is prompted for the game name, the numeric Steam application ID and
a date range.  The scraper will then query the official Steam review API
(`store.steampowered.com/appreviews`) and store relevant details from
each review into a table called ``reviews``.

The resulting database file is named after the game (spaces replaced
with underscores) for easy identification, e.g. ``My_Game_reviews.db``.

Usage:
    Run the script directly and follow the on–screen prompts.  Example:

        $ python3 steam_reviews_scraper.py

    The script will ask for the game name, the numeric app ID and the
    date range (YYYY‑MM‑DD format).  It will then begin downloading
    reviews and storing them into the SQLite database.

Notes:
    * This script relies on the ``store.steampowered.com/appreviews`` JSON
      endpoint, which is the same data source used by the Steam
      storefront.  Each response contains a ``cursor`` field used to
      request subsequent batches.  Reviews are returned in descending
      order by creation time when ``filter=recent`` is specified.
    * The API limits the number of reviews returned per request.  The
      ``num_per_page`` parameter controls this size; the default in this
      script is 100, which is the maximum allowed.
    * If you intend to scrape a large number of reviews, be mindful of
      Steam’s terms of service and rate limits.  It’s generally a good
      idea to insert a small delay between requests to avoid overloading
      their servers.

Author: OpenAI ChatGPT
"""

import json
import sqlite3
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from pathlib import Path

try:
    # ``zoneinfo`` is available in Python 3.9+.  It provides time zone
    # definitions from the IANA database.  If it is unavailable (e.g. in
    # older Python versions) the script will fall back to naive datetime
    # comparisons.
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:
    ZoneInfo = None  # type: ignore

import requests  # type: ignore


def prompt_user() -> tuple[str, str, int, int]:
    """
    Prompt the user for input values and return them.

    This helper asks the user for the game name, Steam App ID and an optional
    date range.  If the user enters ``'y'`` (case‑insensitive) for both the
    start and end date prompts, the script automatically computes a time
    window spanning the last three years up to the current date.  Otherwise,
    the provided strings are parsed as dates in ``YYYY‑MM‑DD`` format.

    Returns
    -------
    tuple[str, str, int, int]
        A 4‑tuple containing the game name, app ID, start timestamp and
        end timestamp.  The timestamps are Unix epoch seconds in UTC.
    """
    game_name = input("Enter the game name: ").strip()
    if not game_name:
        print("Game name cannot be empty.")
        sys.exit(1)

    appid_input = input("Enter the numeric Steam app ID: ").strip()
    try:
        app_id = int(appid_input)
    except ValueError:
        print("App ID must be a valid integer.")
        sys.exit(1)

    tz = None
    if ZoneInfo is not None:
        # Use the user's time zone (America/New_York) for interpreting dates.  If
        # zoneinfo is unavailable or the zone cannot be loaded, timestamps will
        # be naive.
        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = None

    # Ask the user for a start and end date.  The user may enter 'y' for
    # both prompts to automatically select a three‑year period ending today.
    start_date_str = input(
        "Enter the start date (YYYY‑MM‑DD) – reviews older than this will be ignored (or 'y' for automatic 2‑year range): "
    ).strip().lower()
    end_date_str = input(
        "Enter the end date (YYYY‑MM‑DD) – reviews newer than this will be skipped (or 'y' for automatic 2‑year range): "
    ).strip().lower()

    # If the user opted for automatic selection for both values, compute a
    # three‑year window ending now.  Otherwise, parse the provided dates.
    if start_date_str == "y" and end_date_str == "y":
        # Determine the current time in the chosen timezone (if available).
        if tz:
            now = datetime.now(tz)
        else:
            now = datetime.now()
        # Start two years prior to now.  Using 365 days per year for
        # simplicity.  This avoids dependency on external libraries like
        # dateutil.
        start_dt = now - timedelta(days=365 * 2)
        end_dt = now
        # Convert to timestamps.  We do not adjust to the end of the day
        # because ``now`` already includes the current time.
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
    else:
        # Normal date parsing path.
        try:
            # The user may have entered uppercase 'Y' for yes, but we only reach
            # this branch if at least one value isn't 'y'.  Attempt to parse
            # both strings as dates.  If either is 'y' here it will raise.
            start_dt_plain = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_dt_plain = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print("Dates must be in the format YYYY‑MM‑DD or 'y' for automatic range.")
            sys.exit(1)
        if tz:
            # Interpret the start at midnight local time and the end at the end
            # of the day (23:59:59 local time) to include the full day.
            start_dt = start_dt_plain.replace(tzinfo=tz)
            end_dt = end_dt_plain.replace(tzinfo=tz) + timedelta(days=1) - timedelta(seconds=1)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())
        else:
            # Naive interpretation; treat as midnight and 23:59:59 local time.
            start_ts = int(time.mktime(start_dt_plain.timetuple()))
            end_ts = int(
                time.mktime(
                    (end_dt_plain + timedelta(days=1) - timedelta(seconds=1)).timetuple()
                )
            )

    # Validate that the start timestamp does not exceed the end timestamp.
    if start_ts > end_ts:
        print("The start date must not be later than the end date.")
        sys.exit(1)

    return game_name, str(app_id), start_ts, end_ts


def create_database(db_path: Path, table_name: str) -> sqlite3.Connection:
    """
    Create (or open) the SQLite database and ensure the reviews table exists.

    Parameters
    ----------
    db_path : Path
        The filesystem path to the SQLite database file.  Parent directories
        will be created if they do not already exist.
    table_name : str
        The name of the table to create.  The name should be a safe string
        containing only alphanumeric characters and underscores.

    Returns
    -------
    sqlite3.Connection
        An open connection to the SQLite database.  The caller is
        responsible for closing this connection.
    """
    # Ensure parent directories exist.  ``mkdir`` with ``parents=True`` has no
    # effect if the path already exists.
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Define the table schema.  Primary key on recommendationid prevents duplicates.
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            recommendationid TEXT PRIMARY KEY,
            game_name TEXT,
            app_id INTEGER,
            steamid TEXT,
            num_games_owned INTEGER,
            num_reviews INTEGER,
            playtime_forever INTEGER,
            playtime_last_two_weeks INTEGER,
            playtime_at_review INTEGER,
            last_played INTEGER,
            language TEXT,
            review TEXT,
            timestamp_created INTEGER,
            timestamp_updated INTEGER,
            voted_up BOOLEAN,
            votes_up INTEGER,
            votes_funny INTEGER,
            weighted_vote_score REAL,
            comment_count INTEGER,
            steam_purchase BOOLEAN,
            received_for_free BOOLEAN,
            written_during_early_access BOOLEAN,
            primarily_steam_deck BOOLEAN
        )
        """
    )
    conn.commit()
    return conn


def fetch_reviews_page(app_id: str, cursor: str | None = None, num_per_page: int = 100) -> dict:
    """Fetch a single page of reviews from the Steam API.

    Parameters
    ----------
    app_id : str
        The numeric Steam app ID as a string.
    cursor : str | None, optional
        The pagination cursor returned from a previous request.  If ``None``,
        the first page is retrieved.  Default is ``None``.
    num_per_page : int
        The number of reviews to request per page (maximum of 100).

    Returns
    -------
    dict
        The parsed JSON response from the API.  If the request fails or the
        response cannot be decoded as JSON, an exception will be raised.
    """
    # Compose the URL.  We request reviews sorted by most recent to allow
    # short‑circuiting once the oldest review on the page is outside the
    # desired date range.
    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "num_per_page": num_per_page,
        "filter": "recent",
        "language": "english",
        "purchase_type": "all",
    }
    if cursor:
        params["cursor"] = cursor
    # Add a User‑Agent header to reduce the chance of being blocked.
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SteamReviewScraper/1.0)"}
    response = requests.get(base_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def process_reviews(
    data: dict,
    game_name: str,
    app_id: str,
    start_ts: int,
    end_ts: int,
    conn: sqlite3.Connection,
    table_name: str,
) -> tuple[bool, str | None]:
    """Insert reviews from a JSON page into the database.

    Parameters
    ----------
    data : dict
        The JSON response from a call to ``fetch_reviews_page``.
    game_name : str
        The human‑readable name of the game.
    app_id : str
        The numeric Steam app ID as a string.
    start_ts : int
        The earliest allowed review timestamp (Unix epoch, inclusive).
    end_ts : int
        The latest allowed review timestamp (Unix epoch, inclusive).
    conn : sqlite3.Connection
        An open connection to the SQLite database.

    Returns
    -------
    tuple[bool, str | None]
        A pair containing a boolean indicating whether iteration should
        continue (``True`` to fetch the next page) and the next cursor to
        use.  If no cursor is returned, the second element will be
        ``None``.
    """
    reviews = data.get("reviews", [])
    cursor = data.get("cursor")
    continue_iter = True

    cur = conn.cursor()
    for review in reviews:
        ts = review.get("timestamp_created", 0)
        # Skip reviews newer than the end date
        if ts > end_ts:
            continue
        # Stop processing altogether if we encounter a review older than the start date.
        if ts < start_ts:
            continue_iter = False
            break

        rec_id = review.get("recommendationid")
        author = review.get("author", {})
        # Prepare the record for insertion
        record = (
            rec_id,
            game_name,
            int(app_id),
            author.get("steamid"),
            author.get("num_games_owned"),
            author.get("num_reviews"),
            author.get("playtime_forever"),
            author.get("playtime_last_two_weeks"),
            author.get("playtime_at_review"),
            author.get("last_played"),
            review.get("language"),
            review.get("review"),
            ts,
            review.get("timestamp_updated"),
            bool(review.get("voted_up")),
            review.get("votes_up"),
            review.get("votes_funny"),
            review.get("weighted_vote_score"),
            review.get("comment_count"),
            bool(review.get("steam_purchase")),
            bool(review.get("received_for_free")),
            bool(review.get("written_during_early_access")),
            bool(review.get("primarily_steam_deck")),
        )
        try:
            # Insert into the user‑specified table.  The table name is
            # interpolated directly into the SQL statement because SQLite does not
            # support parameterized table names.  The values themselves are
            # parameterized to avoid SQL injection.
            cur.execute(
                f"""
                INSERT OR IGNORE INTO {table_name} (
                    recommendationid,
                    game_name,
                    app_id,
                    steamid,
                    num_games_owned,
                    num_reviews,
                    playtime_forever,
                    playtime_last_two_weeks,
                    playtime_at_review,
                    last_played,
                    language,
                    review,
                    timestamp_created,
                    timestamp_updated,
                    voted_up,
                    votes_up,
                    votes_funny,
                    weighted_vote_score,
                    comment_count,
                    steam_purchase,
                    received_for_free,
                    written_during_early_access,
                    primarily_steam_deck
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                record,
            )
        except sqlite3.Error as e:
            print(f"Failed to insert review {rec_id}: {e}")
    conn.commit()
    return continue_iter, cursor


def main() -> None:
    game_name, app_id, start_ts, end_ts = prompt_user()
    # Construct a safe table name using the game name and date range.  Spaces
    # are replaced with underscores and dates are formatted as YYYYMMDD to
    # avoid characters that SQLite would treat as delimiters.
    safe_game_name = game_name.replace(" ", "_")
    # Format the dates from the timestamps.  Use the America/New_York timezone
    # if zoneinfo is available; otherwise default to UTC.
    tz = None
    if ZoneInfo is not None:
        try:
            tz = ZoneInfo("America/New_York")
        except Exception:
            tz = None
    if tz:
        start_dt = datetime.fromtimestamp(start_ts, tz)
        end_dt = datetime.fromtimestamp(end_ts, tz)
    else:
        start_dt = datetime.fromtimestamp(start_ts)
        end_dt = datetime.fromtimestamp(end_ts)
    start_label = start_dt.strftime("%Y%m%d")
    end_label = end_dt.strftime("%Y%m%d")
    table_name = f"{safe_game_name}_{start_label}_{end_label}_steam"

    # Use the specified relative database path.  This location is expected to
    # reside under the project directory.  The directory will be created if
    # necessary.
    db_path = Path("Data_Extraction/Database/Raw_Reviews.db")
    conn = create_database(db_path, table_name)

    print(
        f"Fetching reviews for {game_name} (App ID {app_id}) between "
        f"{start_dt} and {end_dt}.  Data will be stored in table '{table_name}'."
    )

    cursor: str | None = None
    page_count = 0
    while True:
        # Attempt to fetch a page.  If a network or HTTP error occurs, wait
        # and retry rather than aborting.  This loop continues until a
        # successful response is obtained.
        while True:
            try:
                data = fetch_reviews_page(app_id, cursor)
                break
            except requests.RequestException as e:
                # Log the error and pause before retrying.
                print(
                    f"Failed to fetch reviews (cursor={cursor!r}): {e}. "
                    "Retrying in 15 seconds..."
                )
                time.sleep(15)
                # Continue the inner loop to retry
                continue

        page_count += 1
        continue_iter, cursor = process_reviews(
            data, game_name, app_id, start_ts, end_ts, conn, table_name
        )
        print(f"Processed page {page_count}, next cursor: {cursor if cursor else 'None'}")
        if not continue_iter or not cursor:
            break
        # Be a polite client: wait briefly between requests to avoid hitting rate limits.
        time.sleep(1)

    conn.close()
    print(f"Finished.  Reviews have been saved into {db_path.resolve()} (table {table_name}).")


if __name__ == "__main__":
    main()