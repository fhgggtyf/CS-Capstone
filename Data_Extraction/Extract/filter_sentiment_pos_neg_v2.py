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

# OPTIONAL: install vaderSentiment ahead of time in your env (login node)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
except ImportError:
    vader = None

# Configure device (GPU if available)
device = 0 if torch.cuda.is_available() else -1

# Load transformer model (pretrained sentiment classifier)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# Define a maximum token length and a helper to skip texts that exceed it.
MAX_TOKEN_LENGTH = 512

def is_valid_text_length(text: str) -> bool:
    """Return True if the text length (in tokens) does not exceed MAX_TOKEN_LENGTH."""
    try:
        tokens = sentiment_pipeline.tokenizer(
            text,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(tokens["input_ids"]) <= MAX_TOKEN_LENGTH
    except Exception:
        return False

# Custom lists for lexicon-based heuristics
NEGATIVE_KEYWORDS = [
    "not interesting",
    "no desire to spend money",
    "no desire to spend",
    "waste of money",
    "too expensive",
]
ASPECT_NEGATIVE_CUES = [
    "bad game",
    "boring game",
    "dead game",
    "lag",
    "issue",
    "problem",
    "glitch",
    "bug",
    "crash",
    "battle pass",
    "season pass",
]

def hybrid_sentiment(text):
    """
    Combine transformer prediction with VADER/custom keywords and simple ABSA heuristics.
    Returns a tuple: (final_label, transformer_label, transformer_score, vader_compound)
    """
    # Default values if vader not available
    vader_compound = 0.0
    # Lower-case for keyword detection
    lower_text = text.lower()

    # Transformer prediction
    # Use truncation to handle long texts that exceed the model's max sequence length
    t_res = sentiment_pipeline(text, batch_size=1, truncation=True, padding=True)[0]
    t_label = t_res['label']
    t_score = t_res['score']

    # Confidence threshold: low confidence becomes neutral
    if t_score < 0.7:
        final_label = 'NEUTRAL'
    else:
        final_label = t_label

    # VADER analysis (if installed)
    if vader:
        vader_scores = vader.polarity_scores(text)
        vader_compound = vader_scores['compound']

    # Lexicon detection
    has_custom_negative = any(kw in lower_text for kw in NEGATIVE_KEYWORDS)
    has_aspect_negative = any(cue in lower_text for cue in ASPECT_NEGATIVE_CUES)

    # If VADER or keywords indicate negativity, adjust final label
    if vader_compound <= -0.3 or has_custom_negative or has_aspect_negative:
        # If transformer label was POSITIVE but strong negativity detected, set to MIXED
        if final_label == 'POSITIVE':
            final_label = 'MIXED'
        # If neutral but negative cues present, set to NEGATIVE
        elif final_label == 'NEUTRAL':
            final_label = 'NEGATIVE'
        # If transformer was already negative, keep it negative

    return final_label, t_label, t_score, vader_compound


def retry(tries=3, delay=5, exceptions=(Exception,)):
    """Retry decorator for robustness."""
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


@retry(tries=3, delay=2, exceptions=(Exception,))
def process_table(conn_in, conn_out, table_name):
    print(f"\nProcessing table: {table_name}")
    # use conn_in to read the schema and data
    cur = conn_in.cursor()
    # Get text columns and select the documented main text column if defined
    cur.execute(f"PRAGMA table_info('{table_name}')")
    columns_info = cur.fetchall()
    text_columns = [col[1] for col in columns_info if 'CHAR' in col[2].upper() or 'TEXT' in col[2].upper()]
    if not text_columns:
        print(f"No text columns in {table_name}. Skipping.")
        return
    # Determine main text column from documentation mapping
    lower_table_name = table_name.lower()
    main_text_col = None
    for key, col_name in MAIN_TEXT_FIELD_MAP.items():
        if key == lower_table_name or key in lower_table_name:
            # Try to find the exact column name
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
                main_text_col = col_name
            break
    # Fallback: use first text column if mapping fails or column not present
    if main_text_col is None or main_text_col not in [c[1] for c in columns_info]:
        main_text_col = text_columns[0]
    new_table = f"frustrated_sentiment_pos_neg_v2_{table_name}"

    # Count rows for progress
    cur.execute(f"SELECT COUNT(*) FROM '{table_name}'")
    total_rows = cur.fetchone()[0]
    if total_rows == 0:
        print(f"{table_name} is empty. Skipping.")
        return

    chunksize = 1000
    processed = 0
    first = True

    for chunk in pd.read_sql_query(f"SELECT * FROM '{table_name}'", conn_in, chunksize=chunksize):
        # Determine which rows are short enough to analyze
        texts = chunk[main_text_col].astype(str).tolist()
        valid_indices = [i for i, text in enumerate(texts) if is_valid_text_length(text)]

        if not valid_indices:
            # No valid rows in this chunk; update progress and skip writing
            processed += len(chunk)
            percent = (processed / total_rows) * 100
            print(f"Progress: {processed}/{total_rows} rows ({percent:.2f}%) (all skipped)")
            continue

        # Filter to only valid rows
        df_valid = chunk.iloc[valid_indices].reset_index(drop=True)
        # Apply hybrid sentiment function row-wise on valid rows
        results = df_valid[main_text_col].astype(str).apply(hybrid_sentiment)
        # Unpack result tuples
        df_valid[['final_label', 'transformer_label', 'transformer_score', 'vader_compound']] = pd.DataFrame(results.tolist(), index=df_valid.index)
        # Write valid rows to the new table
        df_valid.to_sql(new_table, conn_out, index=False, if_exists='replace' if first else 'append')
        first = False

        processed += len(chunk)
        percent = (processed / total_rows) * 100
        print(f"Progress: {processed}/{total_rows} rows ({percent:.2f}%)")

    print(f"Finished {table_name}. Results in {new_table}")


def main():
    """
    Run the hybrid sentiment analysis (VADER and transformer) across all
    applicable tables.  Results are stored in a separate output database
    `CS_Capstone_Sentiment.db` located alongside the original
    `CS_Capstone.db`.
    """
    input_db_path = "Data_Extraction/Database/CS_Capstone.db"
    output_db_path = "Data_Extraction/Database/CS_Capstone_Sentiment.db"

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
