import sqlite3
from datetime import datetime, timedelta
import os

# ---------- CONFIGURATION ----------
SOURCE_DB = "Data_Extraction/Database/CS_Capstone_Sentiment_trial.db"
OUTPUT_DB = "Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"

TIME_CANDIDATES = [
    "time", "time_str", "created_utc", "created_at", "published_at", "updated_at",
    "date", "datetime", "timestamp_created", "post_time", "post_date", "posted_at",
    "review_date", "reply_time", "review_time", "comment_time", "comment_date",
    "scraped_at", "fetched_at"
]

KNOWN_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M","%Y-%m-%d","%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S.%f%z","%Y-%m-%dT%H:%M",
    "%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M:%S.%f","%Y/%m/%d %H:%M","%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S","%m/%d/%Y %H:%M:%S.%f","%m/%d/%Y %H:%M","%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M:%S.%f","%d/%m/%Y %H:%M","%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S","%d-%m-%Y %H:%M:%S.%f","%d-%m-%Y %H:%M","%d-%m-%Y",
    "%m-%d-%Y %H:%M:%S","%m-%d-%Y %H:%M:%S.%f","%m-%d-%Y %H:%M","%m-%d-%Y",
    "%b %d, %Y","%d %b %Y","%a, %d %b %Y %H:%M:%S %Z","%a %b %d %H:%M:%S %Y",
    "%Y.%m.%d","%Y.%m.%d %H:%M:%S","%Y.%m.%d %H:%M:%S.%f",
    "%Y%m%d","%Y%m%d%H%M","%Y%m%d%H%M%S","%Y%m%d%H%M%S%f","%Y%m%d%H%M%S.%f",
    "%d/%m/%y", "%d-%m-%y", "%m/%d/%y", "%y/%m/%d", "%y-%m-%d",
    "%m/%d/%Y %I:%M %p", "%m/%d/%y %I:%M %p", "%d/%m/%Y %I:%M %p", "%d/%m/%y %I:%M %p",
    "%Y-%m-%d %I:%M %p"
]
# -----------------------------------

def parse_time(value):
    """Try to parse a time string or timestamp into a datetime object."""
    if value is None:
        return None
    if isinstance(value, (int, float)):  # Unix timestamp
        try:
            return datetime.fromtimestamp(value / 1000 if value > 1e12 else value)
        except Exception:
            return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for fmt in KNOWN_TIME_FORMATS:
            try:
                return datetime.strptime(value, fmt)
            except Exception:
                continue
    return None

def get_latest_time(cursor, table, time_col):
    """Find the latest valid datetime in a table column."""
    cursor.execute(f"SELECT {time_col} FROM {table} WHERE {time_col} IS NOT NULL")
    latest = None
    for (val,) in cursor.fetchall():
        dt = parse_time(val)
        if dt and (latest is None or dt > latest):
            latest = dt
    return latest

def copy_schema(src_cur, dst_con):
    """Copy the schema (tables, indexes, etc.) from the source DB."""
    src_cur.execute("SELECT sql FROM sqlite_master WHERE type IN ('table', 'index', 'trigger') AND sql IS NOT NULL;")
    for (sql,) in src_cur.fetchall():
        try:
            dst_con.execute(sql)
        except sqlite3.OperationalError:
            pass  # some duplicates (e.g. autoincrement sequences) can be ignored
    dst_con.commit()

def filter_table(src_con, dst_con, table):
    """Copy only recent rows of a table into the destination DB."""
    src_cur = src_con.cursor()
    dst_cur = dst_con.cursor()

    # find time column
    src_cur.execute(f"PRAGMA table_info({table})")
    cols = [c[1] for c in src_cur.fetchall()]
    time_col = next((c for c in cols if c.lower() in TIME_CANDIDATES), None)
    if not time_col:
        print(f"⚪ Skipping '{table}': no recognizable time column.")
        src_cur.execute(f"SELECT * FROM {table}")
        rows = src_cur.fetchall()
        if rows:
            dst_cur.executemany(f"INSERT INTO {table} VALUES ({','.join(['?']*len(cols))})", rows)
            dst_con.commit()
        return

    # find latest timestamp
    latest = get_latest_time(src_cur, table, time_col)
    if not latest:
        print(f"⚪ Skipping '{table}': no parsable timestamps.")
        return
    cutoff = latest - timedelta(days=365*2)

    print(f"🕒 Filtering '{table}' (latest={latest}, cutoff={cutoff})")

    src_cur.execute(f"SELECT * FROM {table}")
    rows = []
    for row in src_cur.fetchall():
        row_dict = dict(zip(cols, row))
        dt = parse_time(row_dict.get(time_col))
        if not dt or dt >= cutoff:
            rows.append(row)

    if rows:
        dst_cur.executemany(f"INSERT INTO {table} VALUES ({','.join(['?']*len(cols))})", rows)
        dst_con.commit()
        print(f"✅ Copied {len(rows)} rows to '{table}'")
    else:
        print(f"⚪ No recent rows found in '{table}'")

def main():
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)

    src_con = sqlite3.connect(SOURCE_DB)
    dst_con = sqlite3.connect(OUTPUT_DB)
    src_cur = src_con.cursor()

    # Copy schema first
    copy_schema(src_cur, dst_con)

    src_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in src_cur.fetchall()]

    for table in tables:
        filter_table(src_con, dst_con, table)

    src_con.close()
    dst_con.close()
    print("\n🎉 New filtered DB saved as:", OUTPUT_DB)

if __name__ == "__main__":
    main()
