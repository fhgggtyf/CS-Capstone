import praw
import sqlite3
from datetime import datetime
from time import sleep

# 1. Setup Reddit API (fill in with your credentials)
reddit = praw.Reddit(
    client_id="7u6YFjXKp5GplCa6qb_qMQ",
    client_secret="5_wKBoOWosQqbsKhJ2VoIUyyjf1qdw",
    user_agent="python:reddit_add_time:v0.1 (by u/JSNX_2467)"
)

# 2. Connect to SQLite DB
conn = sqlite3.connect("Data_Extraction/Database/CS_Capstone_Sentiment.db")
cursor = conn.cursor()

# 3. Assume your table is 'reddit_comments' with columns: id, comment_id, time_unix, time_str
cursor.execute("SELECT id, comment_id FROM frustrated_sentiment_pos_neg_v2_League_of_Legends_Reddit_comments WHERE time_unix IS NULL OR time_str IS NULL")
rows = cursor.fetchall()

for db_id, comment_id in rows:
    try:
        # 4. Fetch comment from Reddit
        comment = reddit.comment(id=comment_id)
        unix_time = comment.created_utc  # float
        iso_time = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%dT%H:%M:%SZ')

        # 5. Update database
        cursor.execute("""
            UPDATE frustrated_sentiment_pos_neg_v2_League_of_Legends_Reddit_comments
            SET time_unix = ?, time_str = ?
            WHERE id = ?
        """, (unix_time, iso_time, db_id))
        conn.commit()

        print(f"Updated comment {comment_id}: {iso_time}")

        sleep(0.5)  # Be kind to Reddit’s API

    except Exception as e:
        print(f"Error with comment {comment_id}: {e}")

conn.close()