# This script is unused as of 2024-06-10 but kept for reference.
# It extracts rows with negative sentiment from a source table and inserts them into a new table.
import sqlite3

# Path to your SQLite database
DB_PATH = "Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"

# Table names
SOURCE_TABLE = "frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only"
OUTPUT_TABLE = "frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only_negative_only"

# Connect to the database
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1) Create the output table if it does not exist
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} AS
    SELECT * FROM {SOURCE_TABLE} WHERE 0;
""")

# 2) Optionally clear the output table before inserting
cur.execute(f"DELETE FROM {OUTPUT_TABLE};")

# 3) Build the filter condition
condition = "final_label = 'NEGATIVE'"

# 4) Insert rows with filtering
cur.execute(f"""
    INSERT INTO {OUTPUT_TABLE}
    SELECT *
    FROM {SOURCE_TABLE}
    WHERE {condition};
""")

# Save changes and close
conn.commit()
conn.close()

print(f"Rows copied from {SOURCE_TABLE} into {OUTPUT_TABLE} with condition: {condition}")
