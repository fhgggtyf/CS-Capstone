import pickle
import sqlite3

with open("814380_SEKIRO_SHADOWS_DIE_TWICE/814380_SEKIRO_SHADOWS_DIE_TWICE_reviews_19700101-000000_20250701-190858.pkl", "rb") as f:

    retreived_dict = pickle.load(f)

conn = sqlite3.connect("Data Extraction/Database/CS_Capstone.db")
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

# Insert data
for review in retreived_dict:
    # Make sure score is float if it's accidentally a string
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

# Commit and close
conn.commit()
conn.close()