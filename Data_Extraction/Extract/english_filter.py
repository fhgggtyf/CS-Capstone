import sqlite3
from langdetect import detect, DetectorFactory
from tqdm import tqdm

# Fix randomness in langdetect for reproducibility
DetectorFactory.seed = 42

# Parameters
DB_PATH = "Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"
SOURCE_TABLE = "frustrated_sentiment_pos_neg_v2_sentiment_combined_negative_only"
OUTPUT_TABLE = "frustrated_sentiment_pos_neg_v2_sentiment_combined_negative_only_english_only"

# Connect to database
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1) Create the output table if it does not exist (clone schema only)
cur.execute(f"""
    CREATE TABLE IF NOT EXISTS {OUTPUT_TABLE} AS
    SELECT * FROM {SOURCE_TABLE} WHERE 0;
""")

# 2) Clear previous contents to avoid duplicates (optional)
cur.execute(f"DELETE FROM {OUTPUT_TABLE};")

# 3) Read all rows from the source table
cur.execute(f"SELECT * FROM {SOURCE_TABLE};")
rows = cur.fetchall()

# Get column names so we can reconstruct INSERT
col_names = [description[0] for description in cur.description]
placeholders = ", ".join("?" * len(col_names))
col_list = ", ".join(col_names)

# 4) Detect English rows and insert with progress bar
for row in tqdm(rows, desc="Processing rows", unit="row"):
    row_dict = dict(zip(col_names, row))
    text = row_dict.get("main_text", "")
    try:
        if text and detect(text) == "en":   # keep only English
            cur.execute(
                f"INSERT INTO {OUTPUT_TABLE} ({col_list}) VALUES ({placeholders});",
                row
            )
    except Exception:
        # Skip rows that fail detection (empty / bad input)
        continue

# 5) Commit and close
conn.commit()
conn.close()

print(f"✅ Finished! English rows copied into '{OUTPUT_TABLE}'")
