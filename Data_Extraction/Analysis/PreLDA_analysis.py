import pandas as pd
import sqlite3  # or any other DB connector

# Connect to DB and load table
conn = sqlite3.connect("Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db")
df = pd.read_sql("SELECT final_label, game_name, comment_platform FROM frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only", conn)

# (i) Summary per game
summary_per_game = df.groupby(['game_name', 'final_label']).size().unstack(fill_value=0)
print("Per Game:\n", summary_per_game)
summary_per_game.to_csv("summary_per_game.csv")  # header included by default

# (ii) Overall summary
summary_overall = df['final_label'].value_counts().reset_index()
summary_overall.columns = ['final_label', 'count']  # <-- add header row
print("\nOverall:\n", summary_overall)
summary_overall.to_csv("summary_overall.csv", index=False)

# (iii) Summary per platform
summary_per_platform = df.groupby(['comment_platform', 'final_label']).size().unstack(fill_value=0)
print("\nPer Platform:\n", summary_per_platform)
summary_per_platform.to_csv("summary_per_platform.csv")  # header included by default