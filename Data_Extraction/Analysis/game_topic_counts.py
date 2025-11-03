"""
Script to aggregate dominant topic counts per game and save the result
to an Excel file.

The script reads a CSV containing LDA topic assignments (similar to
``doc_topic_weights.csv``), merges it with a games table from an
SQLite database on the primary key ``id``, and computes, for each
``game_name``, how many documents were dominated by each topic.  The
resulting pivot table is written to an Excel file.

Usage:

```
python game_topic_counts.py --csv_path doc_topic_weights.csv \
                            --db_path mydatabase.db \
                            --table_name games \
                            --output_excel game_topic_counts.xlsx
```

Requirements:
  * pandas (install via ``pip install pandas openpyxl``).  The
    ``openpyxl`` package is required for writing Excel files.
  * SQLite database containing a table with columns ``id`` and
    ``game_name``.
"""

import argparse
import os
import sqlite3
from typing import Optional

import pandas as pd


def load_topics(csv_path: str) -> pd.DataFrame:
    """Load the topic weights CSV and return a DataFrame with id and dominant_topic.

    If the CSV does not already include a ``dominant_topic`` column, it
    will be computed by finding the topic column with the highest
    probability for each document.  The one‑based ``id`` column is
    derived from the zero‑based ``doc_index``.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing topic weights.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``id`` and ``dominant_topic``.
    """
    df = pd.read_csv(csv_path)
    if 'doc_index' not in df.columns:
        raise KeyError("CSV must contain a 'doc_index' column.")
    df['id'] = df['doc_index'] + 1
    if 'dominant_topic' not in df.columns:
        topic_cols = [c for c in df.columns if c.startswith('topic_')]
        if not topic_cols:
            raise KeyError("CSV must contain columns beginning with 'topic_' or a 'dominant_topic' column.")
        df['dominant_topic'] = df[topic_cols].astype(float).idxmax(axis=1)
        df['dominant_topic'] = df['dominant_topic'].str.replace('topic_', '')
        df['dominant_topic'] = df['dominant_topic'].astype(int)
    return df[['id', 'dominant_topic']]


def load_game_names(db_path: str, table_name: str) -> pd.DataFrame:
    """Load id and game_name columns from the database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database.
    table_name : str
        Name of the table containing game data.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``id`` and ``game_name``.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file '{db_path}' does not exist.")
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT id, game_name FROM {table_name}", conn)
    finally:
        conn.close()
    return df


def compute_topic_counts(game_df: pd.DataFrame, topic_df: pd.DataFrame) -> pd.DataFrame:
    """Compute a pivot table of topic counts for each game.

    Parameters
    ----------
    game_df : pandas.DataFrame
        DataFrame containing ``id`` and ``game_name``.
    topic_df : pandas.DataFrame
        DataFrame containing ``id`` and ``dominant_topic``.

    Returns
    -------
    pandas.DataFrame
        A pivot table with index ``game_name``, columns corresponding
        to each topic number, and values representing the count of
        documents where that topic was dominant for that game.
    """
    merged = game_df.merge(topic_df, on='id', how='inner')
    pivot = pd.pivot_table(
        merged,
        index='game_name',
        columns='dominant_topic',
        values='id',
        aggfunc='count',
        fill_value=0
    )
    # Sort columns numerically if they are integers; otherwise rely on default
    pivot.columns = pivot.columns.astype(str)
    return pivot


def save_to_excel(df: pd.DataFrame, output_path: str) -> None:
    """Save the pivot table to an Excel file.

    Parameters
    ----------
    df : pandas.DataFrame
        The pivot table to write.
    output_path : str
        Path to the output Excel file.  The directory must already exist.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")
    print(f"Writing topic counts to Excel file '{output_path}'…")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='TopicCounts')
    print(f"Excel file saved: '{output_path}'.")


def main(csv_path: str, db_path: str, table_name: str, output_excel: str) -> None:
    topic_df = load_topics(csv_path)
    game_df = load_game_names(db_path, table_name)
    topic_counts = compute_topic_counts(game_df, topic_df)
    save_to_excel(topic_counts, output_excel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create an Excel summary of dominant topic counts per game.")
    parser.add_argument('--csv_path', required=True, help="Path to doc_topic_weights.csv")
    parser.add_argument('--db_path', required=True, help="Path to the SQLite database file")
    parser.add_argument('--table_name', required=True, help="Name of the table containing games")
    parser.add_argument('--output_excel', required=True, help="Path to the output Excel file (e.g., game_topic_counts.xlsx)")
    args = parser.parse_args()
    main(args.csv_path, args.db_path, args.table_name, args.output_excel)