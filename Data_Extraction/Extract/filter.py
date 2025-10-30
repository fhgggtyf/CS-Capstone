import sqlite3
import pandas as pd
import re

# Path to the source DB
SOURCE_DB = "Data_Extraction/Database/CS_Capstone.db"

# Platform tag (optional; leave blank if not needed)
PLATFORM = ""

# Create a single connection for all operations
conn = sqlite3.connect(SOURCE_DB)

# Lookup all table names
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("Tables found:", tables)

# === STEP 2: Define contraction equivalents ===
CONTRACTION_MAP = {
    "can't": "cannot",
    "cant" : "cannot",
    "won't": "will not",
    "wont": "will not",
    "dont": "do not",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "mightn't": "might not",
    "needn't": "need not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "who's": "who is",
    "i've": "i have",
    "you've": "you have",
    "they've": "they have",
    "we've": "we have",
    "i'll": "i will",
    "you'll": "you will",
    "they'll": "they will",
    "we'll": "we will",
    "i'd": "i would",
    "you'd": "you would",
    "they'd": "they would",
    "we'd": "we would",
    "ain't": "is not",
    "y'all": "you all",
    "gonna": "going to",
    "wanna": "want to",
    "plz": "please"
}

def normalize_contractions(text):
    text = text.lower()
    for contraction, full in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", full, text)
    return text

# === STEP 3: Define improved regex patterns for each category ===
frustration_keywords = {
    "Blocked Goals / Interruption": [
        r"\bstuck\w*\b", r"dead end", r"\bblock\w*\b", r"\bfroze\w*\b", r"not loading", r"will not continue", r"cannot continue", r"\binterrupt\w*\b", r"\bstall\w*\b", r"impossible to finish"
    ],
    "Ineffectiveness / Incompetence": [
        r"cannot beat", r"too hard", r"unfair", r"\bbroken\w*\b", r"\brig\w*\b", r"lost again", r"not skilled", r"\bfail\w*\b", r"\boverwhelm\w*\b", r"not responsive"
    ],
    "External Control / Coercion": [
        r"forced to", r"cannot skip", r"forced cutscene", r"too many tutorials?", r"no choice", r"scripted", r"\bover[- ]?control\w*\b", r"no freedom", r"cannot change", r"no options"
    ],
    "Exclusion / Rejection": [
        r"no one plays", r"alone", r"\bignor\w*\b", r"toxic", r"grief", r"\bbull\w*\b", r"kicked out", r"\breject\w*\b", r"\bharass\w*\b"
    ],
    "Unfairness / Inequity": [
        r"pay to win", r"\bunbalanc\w*\b", r"unfair ai", r"rng suck", r"always losing", r"exploiters?", r"cheaters?", r"matchmaking sucks?", r"\bsmurf\w*\b", r"unfair advantage"
    ],
    "Bugs / Technical Failures": [
        r"\bglitch\w*\b", r"\bbug\w*\b", r"\bcrash\w*\b", r"\blag\w*\b", r"server down", r"\berror\w*\b", r"unplayable", r"desync", r"frame drops?", r"performance issues"
    ],
    "Repetitiveness / Grind": [
        r"repetitive", r"\brepeat\w*\b", r"\bgrind\w*\b", r"same thing", r"boring", r"bored", r"farming", r"chores?", r"nothing new", r"repetitive tasks?", r"no variety"
    ],
    "Lack of Reward / Feedback": [
        r"no point", r"unrewarding", r"nothing to gain", r"empty reward", r"pointless", r"no feedback", r"waste of time", r"no satisfaction", r"no progress", r"no achievement", r"no incentive"
    ],
    "Cognitive Load / UI Issues": [
        r"\bconfus\w*\b", r"hard to understand", r"too many buttons?", r"poor ux", r"bad design", r"unintuitive", r"hard to learn", r"overloaded screen", r"cluttered ui", r"hard to navigate", r"poor usability"
    ],
    "Anger & Emotional Terms": [
        r"pissed off", r"\bannoy\w*\b", r"\benrag\w*\b", r"\binfuriat\w*\b", r"\brage quit\w*\b", r"\bi hate\b", r"\bmad\b", r"\bangry\b", r"stressed out", r"losing it", r"\bwtf\b", r"\bfrustrat\w*\b"
    ],
    "Helplessness / Hopelessness": [
        r"what is the point", r"i give up", r"hopeless", r"cannot do it anymore", r"stuck forever", r"defeated", r"just uninstall", r"no help"
    ],
    "Suggestions / Improvements": [
        r"please", r"should fix", r"needs improvement", r"better if", r"could be better", r"should add", r"needs more content", r"should change", r"needs balancing", r"should optimize", r"needs polish", r"should improve"
    ]

}

def normalize_contractions(text):
    text = text.lower()
    for contraction, full in CONTRACTION_MAP.items():
        text = re.sub(r"\b" + re.escape(contraction) + r"\b", full, text)
    return text

def match_categories_regex(text):
    text = normalize_contractions(str(text))
    matches = []
    for category, patterns in frustration_keywords.items():
        for pattern in patterns:
            if re.search(pattern, text):
                matches.append(category)
                break
    return ", ".join(matches) if matches else None

def match_categories_any_column(row, text_cols):
    """Check all text columns in a row and return matched categories."""
    for col in text_cols:
        text = row.get(col, "")
        if pd.isnull(text):
            continue
        categories = match_categories_regex(text)
        if categories:
            return categories
    return None

# Iterate over tables
for table_name in tables:
    # Skip internal SQLite tables
    if table_name.startswith("sqlite_"):
        continue

    if "frustrated" in table_name.lower():
        # Skip tables that already contain "frustrated" in their name
        continue
    elif "patch_notes" in table_name.lower():
        # Skip tables that already contain "patch_notes" in their name
        continue

    # Read the table into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Identify text columns (pandas stores SQLite TEXT as dtype "object")
    text_cols = [col for col in df.columns if df[col].dtype == object]
    if not text_cols:
        continue

    # Apply your match across those columns
    df["matched_categories"] = df.apply(
        lambda row: match_categories_any_column(row, text_cols), axis=1
    )

    # Keep only rows with matched categories
    filtered_df = df[df["matched_categories"].notnull()].copy()
    if filtered_df.empty:
        continue

    # Add the platform tag
    filtered_df["comment_platform"] = PLATFORM

    # Write out to <table_name>_frustrated
    target_table = f"{table_name}_frustrated"
    filtered_df.to_sql(target_table, conn, if_exists="replace", index=False)
    print(f"Saved {len(filtered_df)} frustrated records from {table_name} to {target_table}.")

# Close the connection once done
conn.close()