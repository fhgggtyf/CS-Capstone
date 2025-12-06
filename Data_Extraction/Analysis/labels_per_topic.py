import pandas as pd
import sqlite3

# === CONFIG ===
csv_path = "Data_Extraction/Analysis/Results/runs_improved_20251109_050628/k2_etaNone_drop0_uni/doc_topic_weights.csv"         # path to your LDA CSV
db_path = "Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"         # path to your SQLite DB
table_name = "frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only"       # replace with the actual table name
output_path = "k2_etaNone_drop0_uni_negative_topic_sentiment_summary.csv"

# === LOAD DATA ===
lda_df = pd.read_csv(csv_path)

# Connect to database
conn = sqlite3.connect(db_path)

# Get only the id and final_label columns
db_df = pd.read_sql_query(f"SELECT id, final_label FROM {table_name}", conn)

# Close connection
conn.close()

# === MERGE DATA ===
merged = lda_df.merge(db_df, on="id", how="left")

# === STEP 1: Find all unique final_label values ===
unique_labels = merged["final_label"].dropna().unique()
print("Unique sentiment labels found:", unique_labels)

# === STEP 2: Compute per-topic counts for each label ===
topic_label_counts = (
    merged.groupby(["dominant_topic", "final_label"])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Save to CSV
topic_label_counts.to_csv(output_path, index=False)
print(f"\nSaved topic–sentiment summary to: {output_path}")
