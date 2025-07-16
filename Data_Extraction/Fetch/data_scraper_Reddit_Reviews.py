import praw
import sqlite3
import os

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

def insert_comment(cursor, comment_id, parent_id, post_id, body, author, score, depth):
    cursor.execute(f'''
        INSERT OR IGNORE INTO {game_name}_Reddit_comments (comment_id, parent_id, post_id, body, author, upvotes, depth)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (comment_id, parent_id, post_id, body, str(author), score, depth))

def save_to_database(submission, game_name):
    db_path = os.path.join('Data_Extraction', 'Database', 'CS_Capstone.db')
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
game_name = input("Input the game name: ").strip()
web_url = input("Input the Reddit post URL: ").strip()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id="7u6YFjXKp5GplCa6qb_qMQ",
    client_secret="5_wKBoOWosQqbsKhJ2VoIUyyjf1qdw",
    user_agent="python:reddit_scraper_test:v0.1 (by u/JSNX_2467)"
)

# Fetch submission
submission = reddit.submission(url=web_url)
submission.comments.replace_more(limit=None)

# Save to database with hierarchy
save_to_database(submission, game_name)
