import sqlite3
import pandas as pd
from transformers import pipeline
import torch
import time
import json
from functools import wraps

# Mapping of table names (or substrings) to the column that contains the main text
# according to the provided Prefiltered data documentation.  All keys and values are
# lower-cased for case-insensitive matching.  When processing a table, if the
# table name matches exactly (or contains) one of these keys, the corresponding
# value will be used as the column name for the primary text field.  If no
# match is found, the first text column will be used as a fallback.
MAIN_TEXT_FIELD_MAP = {
    "metacritic": "main text",
    "reddit": "body",
    "steam": "review",
    "ea forum posts": "body text",
    "baldur’s gate 3 official forum": "content",
    "cyberpunk 2077 official forum": "main text",
    "escape from tarkov official forum posts": "main text",
    "escape from tarkov official forum replies": "main text",
    "escape from tarkov official forum": "main text",
    "league of legends reddit": "body",
    "overwatch 2 official forum": "post text",
}

# ========= TUNABLES =========
# If the top emotion's confidence is below this, we call it NEUTRAL overall.
NEUTRAL_CONFIDENCE = 0.70

# Include any emotion with score >= this in the "top_emotions" multi-label set.
EMOTION_INCLUDE_THRESHOLD = 0.30

# Batch size for model inference
BATCH_SIZE = 32

# ========= DEVICE & MODEL =========
device = 0 if torch.cuda.is_available() else -1

# Emotion classifier (anger, disgust, fear, joy, neutral, sadness, surprise)
emotion_pipeline = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device,
    return_all_scores=True,  # IMPORTANT for multi-label extraction
)

# Set a maximum token length for the emotion model.  Anything longer will be
# skipped to prevent runtime errors (e.g., sequence length > 512 tokens).
MAX_EMOTION_TOKEN_LENGTH = 512

def is_valid_emotion_text(text: str) -> bool:
    """Return True if the given text is short enough to be processed by the
    emotion model.

    We use the emotion model's tokenizer to count tokens without truncation
    or padding.  If the sequence length exceeds MAX_EMOTION_TOKEN_LENGTH,
    the text will be skipped.
    """
    try:
        tokens = emotion_pipeline.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(tokens["input_ids"]) <= MAX_EMOTION_TOKEN_LENGTH
    except Exception:
        return False

# Map emotions to coarse polarity for an optional final label
EMOTION_TO_POLARITY = {
    "joy": "POSITIVE",
    "surprise": "POSITIVE",
    "neutral": "NEUTRAL",
    "anger": "NEGATIVE",
    "disgust": "NEGATIVE",
    "fear": "NEGATIVE",
    "sadness": "NEGATIVE",
}

NEG_SET = {"anger", "disgust", "fear", "sadness"}
POS_SET = {"joy", "surprise"}

def derive_final_label_from_emotions(scores_by_label):
    """
    Given a dict {label: score}, return a coarse final label.
    - If the dominant emotion (max score) < NEUTRAL_CONFIDENCE => NEUTRAL
    - Else compare summed positives vs negatives over all labels.
    """
    if not scores_by_label:
        return "NEUTRAL"

    # Dominant
    dom_label = max(scores_by_label, key=scores_by_label.get)
    dom_score = scores_by_label[dom_label]

    if dom_score < NEUTRAL_CONFIDENCE:
        return "NEUTRAL"

    pos_sum = sum(scores_by_label.get(e, 0.0) for e in POS_SET)
    neg_sum = sum(scores_by_label.get(e, 0.0) for e in NEG_SET)

    if neg_sum > pos_sum:
        return "NEGATIVE"
    elif pos_sum > neg_sum:
        return "POSITIVE"
    else:
        # tie -> lean neutral
        return "NEUTRAL"

def pack_emotion_outputs(batch_texts):
    """
    Run the emotion model on a list of texts (batch) and return structured outputs.
    Returns a list of dicts aligned with batch_texts:
    {
      'dominant_emotion': str,
      'dominant_score': float,
      'top_emotions': 'anger, sadness',
      'emotions_json': '[{"label":"anger","score":0.62}, ...]',
      'final_label': 'NEGATIVE'|'NEUTRAL'|'POSITIVE'
    }
    """
    outputs = []
    # Hugging Face returns, for each input text, a list of {label, score} dicts.
    batch_results = emotion_pipeline(batch_texts, batch_size=BATCH_SIZE)

    for res in batch_results:
        # Normalize labels to lowercase and keep all scores
        scores_by_label = {d["label"].lower(): float(d["score"]) for d in res}

        # Dominant emotion
        dominant = max(scores_by_label, key=scores_by_label.get)
        dominant_score = scores_by_label[dominant]

        # Multi-label selection
        selected = [lbl for lbl, sc in scores_by_label.items() if sc >= EMOTION_INCLUDE_THRESHOLD]
        selected_sorted = sorted(selected, key=lambda l: scores_by_label[l], reverse=True)
        top_emotions = ", ".join(selected_sorted)

        # Serialize full set for audit/debug
        emotions_json = json.dumps(
            [{"label": lbl, "score": scores_by_label[lbl]} for lbl in sorted(scores_by_label, key=scores_by_label.get, reverse=True)],
            ensure_ascii=False
        )

        # Optional coarse polarity from emotions
        final_label = derive_final_label_from_emotions(scores_by_label)

        outputs.append({
            "dominant_emotion": dominant,
            "dominant_score": dominant_score,
            "top_emotions": top_emotions,
            "emotions_json": emotions_json,
            "final_label": final_label,
        })

    return outputs

def retry(tries=3, delay=5, exceptions=(Exception,)):
    """Simple retry decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(tries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt + 1 == tries:
                        raise
                    print(f"Error: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(tries=3, delay=2)
def process_table(conn_in, conn_out, table_name):
    """
    Process a single table from the input SQLite database and write the
    sentiment/emotion analysis results to the output database.

    Parameters
    ----------
    conn_in : sqlite3.Connection
        Connection to the input database from which to read tables.
    conn_out : sqlite3.Connection
        Connection to the output database where processed tables are stored.
    table_name : str
        Name of the table to process.
    """
    print(f"\nProcessing table: {table_name}")
    cur = conn_in.cursor()

    # Find available TEXT/CHAR columns and choose the documented main text column if possible
    cur.execute(f"PRAGMA table_info('{table_name}')")
    cols = cur.fetchall()
    # Build a list of text-like columns (char/varchar/text)
    text_cols = [c[1] for c in cols if "CHAR" in c[2].upper() or "TEXT" in c[2].upper()]
    if not text_cols:
        print(f"No text columns found in {table_name}. Skipping.")
        return
    # Determine the main text column based on the documentation mapping
    lower_table_name = table_name.lower()
    main_text_col = None
    for key, col_name in MAIN_TEXT_FIELD_MAP.items():
        if key == lower_table_name or key in lower_table_name:
            # If the exact column exists, use it; otherwise try an underscore variant
            # Check for a case-insensitive match
            for c in cols:
                if c[1].lower() == col_name.lower():
                    main_text_col = c[1]
                    break
            if main_text_col is None:
                # Try replacing spaces with underscores
                alt = col_name.replace(' ', '_')
                for c in cols:
                    if c[1].lower() == alt.lower():
                        main_text_col = c[1]
                        break
            # If still not found, use the original candidate name (case-insensitive)
            if main_text_col is None:
                main_text_col = col_name
            break
    # Fallback: if no mapping or column not found, use the first text column
    if main_text_col is None or main_text_col not in [c[1] for c in cols]:
        main_text_col = text_cols[0]
    new_table = f"frustrated_sentiment_emotions_{table_name}"

    # Row count for progress
    cur.execute(f"SELECT COUNT(*) FROM '{table_name}'")
    total_rows = cur.fetchone()[0]
    if total_rows == 0:
        print(f"{table_name} is empty. Skipping.")
        return

    chunksize = 1000
    processed = 0
    first = True

    # Use the input connection for reading the table in chunks
    for df in pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn_in, chunksize=chunksize):
        texts = df[main_text_col].astype(str).tolist()
        # Determine which rows are short enough for emotion analysis
        valid_indices = [i for i, text in enumerate(texts) if is_valid_emotion_text(text)]

        if not valid_indices:
            # If no valid rows, update progress based on total rows processed and skip writing
            processed += len(df)
            print(f"Progress: {processed}/{total_rows} rows ({processed/total_rows*100:.2f}%) (all skipped)")
            continue

        # Filter to only valid rows
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        texts_valid = df_valid[main_text_col].astype(str).tolist()

        # Batch emotion inference on valid texts only
        packed = pack_emotion_outputs(texts_valid)

        # Assign outputs to new columns for the valid dataframe
        df_valid["dominant_emotion"] = [p["dominant_emotion"] for p in packed]
        df_valid["dominant_score"] = [p["dominant_score"] for p in packed]
        df_valid["top_emotions"] = [p["top_emotions"] for p in packed]
        df_valid["emotions_json"] = [p["emotions_json"] for p in packed]
        df_valid["final_label"] = [p["final_label"] for p in packed]

        # Write valid rows to the output database
        df_valid.to_sql(new_table, conn_out, index=False, if_exists="replace" if first else "append")
        first = False

        processed += len(df)
        print(f"Progress: {processed}/{total_rows} rows ({processed/total_rows*100:.2f}%)")

    print(f"Finished {table_name}. Results written to {new_table}")

def main():
    """
    Entry point for running sentiment/emotion analysis.

    This function reads all non-output tables from the original
    `CS_Capstone.db` database and writes the processed results into a
    separate database named `CS_Capstone_Sentiment.db`.  The output
    database is created in the same directory as the input database if
    it does not already exist.
    """
    # input database containing the original tables
    input_db_path = "Data_Extraction/Database/CS_Capstone.db"
    # output database where results will be stored
    output_db_path = "Data_Extraction/Database/CS_Capstone_Sentiment.db"

    # Open connections to both databases
    conn_in = sqlite3.connect(input_db_path)
    conn_out = sqlite3.connect(output_db_path)

    cur = conn_in.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        # Skip any previously processed tables
        if "frustrated" in t.lower():
            continue
        process_table(conn_in, conn_out, t)

    # Close connections after processing all tables
    conn_in.close()
    conn_out.close()

if __name__ == "__main__":
    main()
