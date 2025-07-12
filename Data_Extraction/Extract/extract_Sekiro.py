import pickle
import sqlite3
import os
import sys
import gc

PKL_PATH = "814380_SEKIRO_SHADOWS_DIE_TWICE/814380_SEKIRO_SHADOWS_DIE_TWICE_english_reviews_19700101-000000_20250709-010234.pkl"
DB_PATH = "Data Extraction/Database/CS_Capstone.db"

# Sanity check: 64-bit Python
if sys.maxsize < 2**32:
    raise RuntimeError("32-bit Python detected. Please use 64-bit Python to load large .pkl files.")

# Load pkl safely
print(f"[INFO] Attempting to load: {PKL_PATH} ({os.path.getsize(PKL_PATH) / (1024**3):.2f} GB)")

try:
    with open(PKL_PATH, "rb") as f:
        retreived_dict = pickle.load(f)

except (pickle.UnpicklingError, EOFError) as e:
    raise RuntimeError(f"[ERROR] Failed to load pickle file: {e}")

except MemoryError:
    raise RuntimeError("[ERROR] Not enough memory to load the pickle file. Consider a machine with more RAM.")

print("[INFO] Pickle file loaded successfully.")
print(f"[INFO] Total reviews loaded: {len(retreived_dict)}")

# DB setup
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS sekiro_steam (
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

# Insert data with validation
print("[INFO] Inserting into database...")
inserted = 0
for i, review in enumerate(retreived_dict):
    try:
        review["weighted_vote_score"] = float(review["weighted_vote_score"])

        cursor.execute('''
            INSERT OR REPLACE INTO sekiro_steam (
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

        inserted += 1
        if i % 10000 == 0:
            print(f"[INFO] Inserted {i} reviews...")

    except Exception as e:
        print(f"[WARN] Skipping entry at index {i} due to error: {e}")
        continue

conn.commit()
conn.close()
print(f"[INFO] Done. Total reviews inserted: {inserted}")

# Cleanup
gc.collect()
