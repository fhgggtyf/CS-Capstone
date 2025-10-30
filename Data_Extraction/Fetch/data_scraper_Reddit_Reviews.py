import time
import praw
import sqlite3
import os

# Extracted Additional Data from Reddit Posts and Comments
# Exclusively used on League of Legends Reddit posts

def create_tables(cursor, gamename):
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {gamename}_Reddit_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id TEXT UNIQUE,
            title TEXT,
            body TEXT,
            author TEXT
        )
    ''')

    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {gamename}_Reddit_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time DATETIME,
            comment_id TEXT UNIQUE,
            parent_id TEXT,
            post_id TEXT,
            body TEXT,
            author TEXT,
            upvotes INTEGER DEFAULT 0,
            depth INTEGER
        )
    ''')

def insert_post(cursor, post_id, game_name, title, body, author):
    cursor.execute(f'''
        INSERT OR IGNORE INTO {game_name}_Reddit_posts (post_id, title, body, author)
        VALUES (?, ?, ?, ?)
    ''', (post_id, title, body, str(author)))

def insert_comment(cursor, game_name, time, comment_id, parent_id, post_id, body, author, score, depth):
    cursor.execute(f'''
        INSERT OR IGNORE INTO {game_name}_Reddit_comments (time, comment_id, parent_id, post_id, body, author, upvotes, depth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (time, comment_id, parent_id, post_id, body, str(author), score, depth))

def save_to_database(submission, game_name):
    db_path = os.path.join('Data_Extraction', 'Database', 'Raw_Reviews.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    create_tables(cursor, game_name)

    # Insert post
    insert_post(cursor, submission.id, game_name, submission.title, submission.selftext, submission.author)

    # Recursively insert comments
    def process_comments(comments, parent_id=None, depth=0):
        for comment in comments:
            insert_comment(
                cursor,
                game_name,
                comment.created_utc,
                comment.id,
                parent_id,
                submission.id,
                comment.body,
                comment.author,
                comment.score,
                depth
            )
            if comment.replies:
                process_comments(comment.replies, parent_id=comment.id, depth=depth + 1)

    submission.comments.replace_more(limit=None)
    process_comments(submission.comments)

    conn.commit()
    conn.close()
    print(f"✅ All post and comment data saved to database at depth levels.")

# User inputs
game_name = 'League_of_Legends'
web_url = ["https://www.reddit.com/r/leagueoflegends/comments/1ockwag/patch_2521_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1o0mgvv/patch_2520_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1noorpz/patch_2519_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1ncqazi/patch_2518_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1n0ti8a/patch_2517_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1mogdfk/patch_2516_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1mcjhlw/patch_2515_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1m0p2sa/patch_2514_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1ljiosg/patch_2513_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1l85tny/patch_2512_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1kwu713/patch_2511_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1klsybx/patch_2510_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1kavjs0/patch_2509_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1jzzdje/patch_2508_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1jp2b3n/patch_2507_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1jebf4v/patch_2506_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1j3imdx/patch_2505_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1itdlkg/patch_2504_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1ihppc7/patch_25s13_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1i7j0rz/patch_25s12_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1hvza6v/patch_25s11_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1hb9f09/patch_1424_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1gv511q/patch_1423_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1gkehyv/patch_1422_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1g9one8/patch_1421_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1fz69ft/patch_1420_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1foiy04/patch_1419_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1fdoffz/patch_1418_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1f2nre7/patch_1417_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1erepaw/patch_1416_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1eg02qv/patch_1415_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1e4vd0a/patch_1414_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1dobkft/patch_1413_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1ddjmfj/patch_1412_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1d3ijy4/patch_1411_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1crz056/patch_1410_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1cgzl11/patch_149_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1c5mymo/patch_148_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1bu5lu2/patch_147_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1biqwzg/patch_146_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1b7dqlu/patch_145_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1awkgeo/patch_144_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/1akhtig/patch_143_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/19dw626/patch_142_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/192mnsu/patch_141_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/18bjkig/patch_1324_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/17zxg0i/patch_1323_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/17q0evr/patch_1322_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/17fj5fm/patch_1321_notes/",
           "https://www.reddit.com/r/leagueoflegends/comments/174rtd4/patch_1320_notes/"
           ]

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="7u6YFjXKp5GplCa6qb_qMQ",
    client_secret="5_wKBoOWosQqbsKhJ2VoIUyyjf1qdw",
    user_agent="python:reddit_scraper_test:v0.1 (by u/JSNX_2467)"
)

for url in web_url:
    # Fetch submission
    submission = reddit.submission(url=url)
    submission.comments.replace_more(limit=None)

    # Save to database with hierarchy
    save_to_database(submission, game_name)
    print(f"✅ Finished saving all comments for {url}")
    time.sleep(10)  # short pause between posts

