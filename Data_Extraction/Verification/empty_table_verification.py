import sqlite3

def list_table_row_counts(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Get all user tables (exclude internal SQLite ones)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cur.fetchall()]

    print(f"\n📊 Row counts for all tables in '{db_path}':\n")

    empty_tables = []

    for table in tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM '{table}'")
            count = cur.fetchone()[0]
            status = "✅" if count > 0 else "⚠️ EMPTY"
            print(f"{table:<50} → {count:>8} rows {status}")
            if count == 0:
                empty_tables.append(table)
        except Exception as e:
            print(f"{table:<50} → ❌ ERROR: {e}")

    conn.close()

    print("\n" + "=" * 70)
    if empty_tables:
        print("📭 Empty tables:")
        for t in empty_tables:
            print(f" - {t}")
    else:
        print("✅ No empty tables found.")
    print("=" * 70)


# Example usage
list_table_row_counts("Data_Extraction/Database/CS_Capstone_Sentiment_trial.db")


