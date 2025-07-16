import sqlite3
import json
import gzip
import os
from time import sleep

print("Starting Cyberpunk 2077 data extraction...")

# Connect to SQLite and optimize performance
conn = sqlite3.connect("Data_Extraction\Database\CS_Capstone.db")
cursor = conn.cursor()

# Optional: Disable synchronous for better speed (unsafe on power loss)
cursor.execute("PRAGMA synchronous = OFF;")
cursor.execute("PRAGMA journal_mode = WAL;")

# Create table if it doesn't exist
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
conn.commit()
print("Created table")

# Read and insert reviews in batches
gz_path = "Data_Extraction/Database/1091500_CYBERPUNK_2077/1091500_CYBERPUNK_2077_english_reviews_19700101-000000_20250714-093127.jsonl.gz"
batch_size = 1000
batch = []
inserted_total = 0

with gzip.open(gz_path, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        review = json.loads(line)
        review["weighted_vote_score"] = float(review["weighted_vote_score"])
        batch.append(review)

        if i % batch_size == 0:
            cursor.executemany('''
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
            ''', batch)
            conn.commit()
            inserted_total += len(batch)
            print(f"Committed {inserted_total} reviews...")
            batch.clear()

# Commit any remaining reviews
if batch:
    cursor.executemany('''
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
    ''', batch)
    conn.commit()
    inserted_total += len(batch)
    print(f"Committed final {len(batch)} reviews...")

# Close the database
cursor.execute("PRAGMA wal_checkpoint(FULL);")
conn.close()
print(f"✅ Done. Total inserted reviews: {inserted_total}")
