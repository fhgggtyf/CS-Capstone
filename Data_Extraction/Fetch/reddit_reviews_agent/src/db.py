import os
import sqlite3

# New DDL uses explicit placeholders for table and index names
DDL_TEMPLATE = """
CREATE TABLE IF NOT EXISTS {tbl} (
  id TEXT PRIMARY KEY,
  kind TEXT CHECK (kind IN ('post','comment')),
  subreddit TEXT,
  author TEXT,
  created_utc INTEGER,
  title TEXT,
  body TEXT,
  url TEXT,
  score INTEGER,
  num_comments INTEGER,
  parent_id TEXT,
  game_title TEXT,
  platform_guess TEXT,
  playtime_hours REAL,
  is_review INTEGER,
  is_english INTEGER,
  seen_at_utc INTEGER,
  src_version TEXT
);

CREATE INDEX IF NOT EXISTS {idx_game} ON {tbl}(game_title, created_utc);
CREATE INDEX IF NOT EXISTS {idx_sub}  ON {tbl}(subreddit, created_utc);
"""

RUNS_DDL = """
CREATE TABLE IF NOT EXISTS reddit_ingest_runs (
  run_id TEXT PRIMARY KEY,
  started_utc INTEGER,
  ended_utc INTEGER,
  cfg_hash TEXT,
  total_posts INT,
  total_comments INT,
  kept_posts INT,
  kept_comments INT,
  error_count INT
);
"""

def _quote_ident(name: str) -> str:
    # Double-quote the identifier and escape embedded quotes
    return '"' + name.replace('"', '""') + '"'

class DB:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute(RUNS_DDL)
        self.conn.commit()

    def ensure_game_table(self, table_name: str):
        tbl = _quote_ident(table_name)
        idx_game = _quote_ident(f"{table_name}_idx_game_time")
        idx_sub  = _quote_ident(f"{table_name}_idx_sub_time")
        self.conn.executescript(DDL_TEMPLATE.format(tbl=tbl, idx_game=idx_game, idx_sub=idx_sub))
        self.conn.commit()

    def upsert_row(self, table_name: str, row):
        tbl = _quote_ident(table_name)
        sql = f"""
        INSERT OR REPLACE INTO {tbl}
        (id,kind,subreddit,author,created_utc,title,body,url,score,num_comments,parent_id,
         game_title,platform_guess,playtime_hours,is_review,is_english,seen_at_utc,src_version)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        self.conn.execute(sql, row)

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()
