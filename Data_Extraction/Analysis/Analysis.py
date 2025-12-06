import pandas as pd
import sqlite3

# === CONFIG ===
CONFIG = "k14_etaNone_drop0_uni"
DATASET = "original"
TOPIC_CSV = "Data_Extraction/Analysis/Results/runs_improved_20251107_192957/k14_etaNone_drop0_uni/doc_topic_weights.csv"           # Path to your CSV
DB_PATH = "/Users/jsnx/Documents/GitHub/CS-Capstone/Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"       # Path to your SQLite database
TABLE_NAME = "frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only"               # Name of the table containing id, game_name, platform
TOP_N = 5                       # Number of top items to extract per group

# === LOAD DATA ===
print("Loading data...")
topics_df = pd.read_csv(TOPIC_CSV)
conn = sqlite3.connect(DB_PATH)
meta_df = pd.read_sql_query(f"SELECT id, game_name, comment_platform FROM {TABLE_NAME}", conn)
conn.close()

# Merge on ID
merged = pd.merge(topics_df, meta_df, on="id", how="inner")
topic_cols = [c for c in merged.columns if c.startswith("topic_")]

# === 1️⃣ SIZE PER TOPIC ===
print("Calculating topic sizes...")
topic_size = (
    merged["dominant_topic"]
    .value_counts()
    .reset_index()
    .rename(columns={"index": "topic", "dominant_topic": "count"})
)
topic_size.to_csv(CONFIG + "_" + DATASET + "_" + "topic_size.csv", index=False)

# === 2️⃣ TOP 5 TOPICS PER GAME ===
print("Finding top topics per game...")
game_topic_strength = merged.groupby("game_name")[topic_cols].mean()

def extract_top_topics(row):
    return ", ".join([
        f"{topic}:{score:.3f}" for topic, score in row.sort_values(ascending=False).head(TOP_N).items()
    ])

top5_per_game = game_topic_strength.apply(extract_top_topics, axis=1).reset_index()
top5_per_game.columns = ["game_name", "top_topics"]
top5_per_game.to_csv(CONFIG + "_" + DATASET + "_" + "top5_topics_per_game.csv", index=False)

# === 3️⃣ TOP 5 TOPICS PER PLATFORM ===
print("Finding top topics per platform...")
platform_topic_strength = merged.groupby("comment_platform")[topic_cols].mean()

top5_per_platform = platform_topic_strength.apply(extract_top_topics, axis=1).reset_index()
top5_per_platform.columns = ["comment_platform", "top_topics"]
top5_per_platform.to_csv(CONFIG + "_" + DATASET + "_" + "top5_topics_per_platform.csv", index=False)

# === 4️⃣ TOP 5 GAMES PER TOPIC ===
print("Finding top games per topic...")
topic_to_games = []
for topic in topic_cols:
    top_games = (
        merged.groupby("game_name")[topic]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_N)
        .reset_index()
    )
    top_games["topic"] = topic
    topic_to_games.append(top_games)

top_games_df = pd.concat(topic_to_games, ignore_index=True)
top_games_df.to_csv(CONFIG + "_" + DATASET + "_" + "top5_games_per_topic.csv", index=False)

# === 5️⃣ TOP 5 PLATFORMS PER TOPIC ===
print("Finding top platforms per topic...")
topic_to_platforms = []
for topic in topic_cols:
    top_platforms = (
        merged.groupby("comment_platform")[topic]
        .mean()
        .sort_values(ascending=False)
        .head(TOP_N)
        .reset_index()
    )
    top_platforms["topic"] = topic
    topic_to_platforms.append(top_platforms)

top_platforms_df = pd.concat(topic_to_platforms, ignore_index=True)
top_platforms_df.to_csv(CONFIG + "_" + DATASET + "_" + "top5_platforms_per_topic.csv", index=False)

# === DONE ===
print("\n✅ Analysis complete!")
print("Generated files:")
print(" - topic_size.csv")
print(" - top5_topics_per_game.csv")
print(" - top5_topics_per_platform.csv")
print(" - top5_games_per_topic.csv")
print(" - top5_platforms_per_topic.csv")
