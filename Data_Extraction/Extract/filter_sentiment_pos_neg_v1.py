import sqlite3
import pandas as pd
from transformers import pipeline
import torch
import time
from functools import wraps

# Mapping of table names or substrings to the column containing the main text
# based on the Prefiltered data documentation. Keys and values are lower-cased
# for case-insensitive matching.
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

# Configure device: use GPU if available
device = 0 if torch.cuda.is_available() else -1

# Load a pretrained transformer for sentiment analysis
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# Define maximum sequence length allowed for the model.  DistilBERT and
# other BERT-like models typically support up to 512 tokens.  We use
# this value to filter out texts that are too long and would cause
# runtime errors.  You can also query this from the model config via
# `sentiment_pipeline.model.config.max_position_embeddings`.
MAX_TOKEN_LENGTH = 512

def is_valid_text_length(text: str) -> bool:
    """Return True if the text is short enough (<= MAX_TOKEN_LENGTH tokens).

    We use the pipeline's tokenizer to count tokens without padding
    or truncation.  If the token length exceeds our threshold, the
    text is considered invalid and will be skipped.
    """
    try:
        tokens = sentiment_pipeline.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # The tokenizer returns a dict with 'input_ids'
        return len(tokens["input_ids"]) <= MAX_TOKEN_LENGTH
    except Exception:
        # In case of any unexpected error during tokenization, treat as invalid
        return False

def retry(tries=3, delay=5, exceptions=(Exception,)):
    """A decorator for retrying a function call if it raises an exception."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < tries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
                    if attempts >= tries:
                        raise
                    print(f"Error: {e}. Retrying ({attempts}/{tries}) in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(tries=3, delay=2, exceptions=(Exception,))
def process_table(conn_in, conn_out, table_name):
    """Process one table: detect main text column, run sentiment analysis, and write to the output database with progress updates."""
    print(f"\nProcessing table: {table_name}")
    cur = conn_in.cursor()

    # Find available TEXT/CHAR columns and choose the documented main text column if possible
    cur.execute(f"PRAGMA table_info('{table_name}')")
    columns_info = cur.fetchall()
    text_columns = [col[1] for col in columns_info if "CHAR" in col[2].upper() or "TEXT" in col[2].upper()]
    if not text_columns:
        print(f"No text columns found in {table_name}. Skipping.")
        return
    # Determine the main text column based on the documentation mapping
    lower_table_name = table_name.lower()
    main_text_col = None
    for key, col_name in MAIN_TEXT_FIELD_MAP.items():
        if key == lower_table_name or key in lower_table_name:
            # Check for exact match of column name
            for col in columns_info:
                if col[1].lower() == col_name.lower():
                    main_text_col = col[1]
                    break
            if main_text_col is None:
                # Try underscore variant
                alt = col_name.replace(' ', '_')
                for col in columns_info:
                    if col[1].lower() == alt.lower():
                        main_text_col = col[1]
                        break
            if main_text_col is None:
                # Fallback to the candidate name itself
                main_text_col = col_name
            break
    # Fallback to first text column if mapping not found or column not present
    if main_text_col is None or main_text_col not in [c[1] for c in columns_info]:
        main_text_col = text_columns[0]
    new_table = f"frustrated_sentiment_pos_neg_v1_{table_name}"

    # Count total rows for progress tracking
    cur.execute(f"SELECT COUNT(*) FROM '{table_name}'")
    total_rows = cur.fetchone()[0]
    if total_rows == 0:
        print(f"Table {table_name} is empty. Skipping.")
        return

    chunksize = 1000
    processed = 0
    first = True

    for chunk in pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn_in, chunksize=chunksize):
        texts = chunk[main_text_col].astype(str).tolist()
        # Determine which texts are within the allowable token length.
        valid_indices = [i for i, text in enumerate(texts) if is_valid_text_length(text)]

        if not valid_indices:
            # No valid rows in this chunk; skip writing.  Update progress based on total rows processed.
            processed += len(chunk)
            percent = (processed / total_rows) * 100
            print(f"Progress: {processed}/{total_rows} rows ({percent:.2f}%) (all skipped)")
            continue

        # Filter the DataFrame to only valid rows
        df_valid = chunk.iloc[valid_indices].reset_index(drop=True)
        texts_valid = df_valid[main_text_col].astype(str).tolist()

        # Run sentiment analysis on the valid texts only
        results = sentiment_pipeline(texts_valid, batch_size=32)
        df_valid["sentiment_label"] = [r["label"] for r in results]
        df_valid["sentiment_score"] = [r["score"] for r in results]

        # Write the valid rows to the output database
        df_valid.to_sql(new_table, conn_out, index=False, if_exists="replace" if first else "append")
        first = False

        # Update progress based on the entire chunk (processed rows) even if some rows were skipped
        processed += len(chunk)
        percent = (processed / total_rows) * 100
        print(f"Progress: {processed}/{total_rows} rows ({percent:.2f}%)")

    print(f"Finished {table_name}. Results written to {new_table}")

def main():
    """
    Run the sentiment analysis on all applicable tables in the input database and
    write results to a new output database.  The input database is
    `CS_Capstone.db` and the output database is `CS_Capstone_Sentiment.db`.
    """
    input_db_path = "Data_Extraction/Database/CS_Capstone.db"
    output_db_path = "Data_Extraction/Database/CS_Capstone_Sentiment.db"

    # open connections to the input and output databases
    conn_in = sqlite3.connect(input_db_path)
    conn_out = sqlite3.connect(output_db_path)

    cur = conn_in.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    for table in tables:
        if 'frustrated' in table.lower():
            continue
        process_table(conn_in, conn_out, table)

    conn_in.close()
    conn_out.close()

if __name__ == "__main__":
    main()
