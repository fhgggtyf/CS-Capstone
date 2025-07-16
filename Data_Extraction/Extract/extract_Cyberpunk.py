import sqlite3
import json
import gzip

print("Starting Cyberpunk 2077 data extraction...")

with gzip.open("Data_Extraction/Database/1091500_CYBERPUNK_2077/1091500_CYBERPUNK_2077_english_reviews_19700101-000000_20250714-093127.jsonl.gz", "rt", encoding="utf-8") as f:
    reviews = [json.loads(line) for line in f]

print(f"Loaded {len(reviews)} reviews from the JSONL file.")

conn = sqlite3.connect("Data_Extraction/Database/CS_Capstone.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS cyberpunk_steam (
        recommendationid TEXT PRIMARY KEY,
        author_steamid TEXT,
        playtime_at_review_minutes INTEGER,
        playtime_forever_minutes INTEGER,
        playtime_last_two_weeks_minutes INTEGER,
        last_played INTEGER,
        review_text TEXT,
        timestamp_created INTEGER,
        timestamp_updated INTEGER,
        voted_up BOOLEAN,
        votes_up INTEGER,
        votes_funny INTEGER,
        weighted_vote_score REAL,
        steam_purchase BOOLEAN,
        received_for_free BOOLEAN,
        written_during_early_access BOOLEAN
    )
''')

print("Created table")

# Insert data
for review in reviews:
    # Make sure score is float if it's accidentally a string
    review["weighted_vote_score"] = float(review["weighted_vote_score"])

    cursor.execute('''
        INSERT OR REPLACE INTO cyberpunk_steam (
            recommendationid, author_steamid, playtime_at_review_minutes,
            playtime_forever_minutes, playtime_last_two_weeks_minutes, last_played,
            review_text, timestamp_created, timestamp_updated, voted_up,
            votes_up, votes_funny, weighted_vote_score, steam_purchase,
            received_for_free, written_during_early_access
        ) VALUES (
            :recommendationid, :author_steamid, :playtime_at_review_minutes,
            :playtime_forever_minutes, :playtime_last_two_weeks_minutes, :last_played,
            :review_text, :timestamp_created, :timestamp_updated, :voted_up,
            :votes_up, :votes_funny, :weighted_vote_score, :steam_purchase,
            :received_for_free, :written_during_early_access
        )
    ''', review)

    print(f"Inserted review {review[0]}")

# Commit and close
conn.commit()
conn.close()