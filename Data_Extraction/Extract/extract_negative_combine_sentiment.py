# extract_negative_combine_sentiment.py
# ------------------------------------
# Creates version-specific combined tables that include ONLY "negative" rows.
# For STEAM tables, additionally excludes rows where voted_up = 1.
#
# Outputs:
#   frustrated_sentiment_emotions_sentiment_combined_negative_only
#   frustrated_sentiment_pos_neg_v1_sentiment_combined_negative_only
#   frustrated_sentiment_pos_neg_v2_sentiment_combined_negative_only
#
# No external dependencies beyond pandas/numpy.

import sqlite3
import pandas as pd
import numpy as np
import re
import sys
from datetime import datetime, timezone

# ====== CONFIG ======
DB_PATH = "Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db"

# Columns to keep at the front (from your combine logic)
BASE_FINAL_COLS = ["main_text", "comment_platform", "time", "game_name"]

# Version -> sentiment columns to preserve (and used to determine "negative")
VERSION_EXTRA_COLS = {
    # "frustrated_sentiment_emotions": [
    #     "dominant_emotion",
    #     "dominant_score",
    #     "top_emotions",
    #     "emotions_json",
    #     "final_label",
    # ],
    # "frustrated_sentiment_pos_neg_v1": [
    #     "sentiment_label",
    #     "sentiment_score",
    # ],
    "frustrated_sentiment_pos_neg_v2": [
        "final_label",
        "transformer_label",
        "transformer_score",
        "vader_compound",
    ],
}

# Candidate text/time columns (same spirit as your combine script)
MAIN_TEXT_CANDIDATES = [
    "main_text", "body", "body_text", "content", "text", "post_title",
    "review_text", "review", "comment", "message", "post_text", "description"
]
TIME_CANDIDATES = [
    "time", "time_str", "created_utc", "created_at", "published_at", "updated_at",
    "date", "datetime", "timestamp_created", "post_time", "post_date", "posted_at",
    "review_date", "reply_time", "review_time", "comment_time", "comment_date",
    "scraped_at", "fetched_at"
]

# A reasonably broad set of known formats — copied/adapted from your combine script
KNOWN_TIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S","%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S%z","%Y-%m-%d %H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M","%Y-%m-%d","%Y-%m-%dT%H:%M:%S","%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%dT%H:%M:%S%z","%Y-%m-%dT%H:%M:%S.%f%z","%Y-%m-%dT%H:%M",
    "%Y/%m/%d %H:%M:%S","%Y/%m/%d %H:%M:%S.%f","%Y/%m/%d %H:%M","%Y/%m/%d",
    "%m/%d/%Y %H:%M:%S","%m/%d/%Y %H:%M:%S.%f","%m/%d/%Y %H:%M","%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S","%d/%m/%Y %H:%M:%S.%f","%d/%m/%Y %H:%M","%d/%m/%Y",
    "%d-%m-%Y %H:%M:%S","%d-%m-%Y %H:%M:%S.%f","%d-%m-%Y %H:%M","%d-%m-%Y",
    "%m-%d-%Y %H:%M:%S","%m-%d-%Y %H:%M:%S.%f","%m-%d-%Y %H:%M","%m-%d-%Y",
    "%b %d, %Y","%d %b %Y","%a, %d %b %Y %H:%M:%S %Z","%a %b %d %H:%M:%S %Y",
    "%Y.%m.%d","%Y.%m.%d %H:%M:%S","%Y.%m.%d %H:%M:%S.%f",
    "%Y%m%d","%Y%m%d%H%M","%Y%m%d%H%M%S","%Y%m%d%H%M%S%f","%Y%m%d%H%M%S.%f"
]

# --------- helpers (from your combine logic) ----------
def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

GAME_NAME_MAP = {
    "assasinscreedunity": "Assasin's Creed Unity",
    "assassinscreedunity": "Assasin's Creed Unity",
    "cyberpunk2077": "Cyberpunk 2077",
    "sekiroshadowsdietwice": "Sekiro: Shadows Die Twice",
    "leagueoflegends": "League of Legends",
    "escapefromtarkov": "Escape From Tarkov",
    "crashbandicoot4itsabouttime": "Crash Bandicoot 4: It’s About Time",
    "crash4": "Crash Bandicoot 4: It’s About Time",
    "darksoulsremastered": "Dark Souls (Remastered)",
    "cuphead": "Cuphead",
    "supermeatboy": "Super Meat Boy",
    "devilmaycry5": "Devil May Cry 5",
    "battlefield4": "Battlefield 4",
    "wwe2k20": "WWE 2K20",
    "halothemasterchiefcollection": "Halo: The Master Chief Collection",
    "halomcc": "Halo: The Master Chief Collection",
    "themasterchiefcollection": "Halo: The Master Chief Collection",
    "fallout76": "Fallout 76",
    "masseffectandromeda": "Mass Effect: Andromeda",
    "battlefield2042": "Battlefield 2042",
    "batmanarkhamknight": "Batman: Arkham Knight",
    "anthem": "Anthem",
    "nomanssky": "No Man's Sky",
    "starwarsbattlefrontii": "Star Wars Battlefront II",
    "simcity2013": "SimCity (2013)",
    "mightyno9": "Mighty No. 9",
    "marvelsavengers": "Marvel's Avengers",
    "tonyhawkride": "Tony Hawk: Ride",
    "fortnite": "Fortnite",
    "dota2": "DOTA 2",
    "rainbow6siege": "Rainbow 6: Siege",
    "rainbowsixsiege": "Rainbow 6: Siege",
    "tomclancysrainbow6siege": "Rainbow 6: Siege",
    "bindingofisaacrepentance": "The Binding of Isaac: Repentance",
    "gettingoveritwithbennettfoddy": "Getting Over It with Bennett Foddy",
    "furi": "Furi",
    "jumpking": "Jump King",
    "mortalshell": "Mortal Shell",
    "streetfighterv": "Street Fighter V",
    "counterstrike2": "Counter_Strike 2",
    "apexlegends": "APEX Legends",
    "genshinimpact": "Genshin Impact",
    "callofdutywarzone": "Call of Duty: Warzone",
    "valorant": "Valorant",
    "hogwartslegacy": "Hogwarts Legacy",
    "palworld": "PalWorld",
    "baldursgate3": "Baldur's Gate 3",
    "baldursgate": "Baldur's Gate 3",
    "thelordoftheringsgollum": "The Lord of the Rings: Gollum",
    "callofdutymodernwarfareiii": "Call of Duty: Modern Warfare III",
    "warcraftiiireforged": "Warcraft III: Reforged",
    "pubgbattlegrounds": "PUBG: Battlegrounds",
    "pubg": "PUBG: Battlegrounds",
    "warthunder": "War Thunder",
    "helldivers2": "Helldivers 2",
    "overwatch2": "Overwatch 2",
    "deltaforce": "Delta Force (2024)",
    "easportsfc25": "EA SPORTS FC™ 25",
    "fc25": "EA SPORTS FC™ 25",
    "easports25": "EA SPORTS FC™ 25",
}

def get_game_name(table_name: str) -> str:
    norm = normalize_name(table_name)
    for key, game in GAME_NAME_MAP.items():
        if key in norm:
            return game
    return table_name

def get_platform_name(table_name: str):
    name = table_name.lower()
    if "reddit" in name:
        return "reddit"
    if "metacritic" in name:
        return "metacritic"
    if "steam" in name:
        return "steam"
    if "official" in name:
        return "official"
    return None

def alias_first_present(table_cols: set, candidates: list, alias: str) -> str:
    for c in candidates:
        if c in table_cols:
            return f"{c} AS {alias}"
    return f"NULL AS {alias}"

def _from_epoch_auto_scale(x: float) -> datetime:
    if x > 1e18: x /= 1e9
    elif x > 1e15: x /= 1e6
    elif x > 1e12: x /= 1e3
    return datetime.fromtimestamp(x, tz=timezone.utc)

def _try_parse_known_formats(s: str):
    for fmt in KNOWN_TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return None

def parse_any_datetime(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        if isinstance(val, pd.Timestamp):
            return (val.tz_localize("UTC") if val.tzinfo is None else val.tz_convert("UTC")).to_pydatetime()
    except Exception:
        pass
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            return _from_epoch_auto_scale(float(val))
        except Exception:
            return np.nan
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return np.nan
        if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
            if '.' not in s:
                digits = s.lstrip('+-')
                try:
                    if len(digits) == 8:
                        return datetime.strptime(digits, "%Y%m%d").replace(tzinfo=timezone.utc)
                    elif len(digits) == 12:
                        return datetime.strptime(digits, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
                    elif len(digits) == 14:
                        return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                    elif len(digits) == 17:
                        date_part, frac_part = digits[:14], digits[14:]
                        frac_micro = (frac_part + '000000')[:6]
                        return datetime.strptime(date_part + frac_micro, "%Y%m%d%H%M%S%f").replace(tzinfo=timezone.utc)
                except Exception:
                    pass
            try:
                return _from_epoch_auto_scale(float(s))
            except Exception:
                pass
        dt = _try_parse_known_formats(s)
        if dt is not None:
            return dt
        try:
            p = pd.to_datetime(s, utc=True, errors="coerce")
            if not pd.isna(p):
                return p.to_pydatetime()
        except Exception:
            pass
        try:
            from dateutil import parser as du_parser
            dt = du_parser.parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            return np.nan
    return np.nan

def best_datetime_series(df: pd.DataFrame, candidates: list) -> pd.Series:
    parsed_cols = []
    for col in candidates:
        if col in df.columns:
            parsed_cols.append(df[col].apply(parse_any_datetime))
    if not parsed_cols:
        return pd.Series([pd.NaT] * len(df), index=df.index, dtype="datetime64[ns, UTC]")
    combined = parsed_cols[0]
    for s in parsed_cols[1:]:
        combined = combined.where(pd.notna(combined), s)
    return pd.to_datetime(combined, utc=True, errors="coerce")

# ----- tiny progress bar (no external deps) -----
def progress_bar(iteration, total, width=40, prefix=""):
    if total <= 0:
        sys.stdout.write(f"{prefix} 0/0\n")
        return
    frac = iteration / total
    filled = int(width * frac)
    bar = "█" * filled + "-" * (width - filled)
    sys.stdout.write(f"\r{prefix} |{bar}| {iteration}/{total}")
    if iteration >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()

# ================== MAIN ==================
def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    all_names = [r[0] for r in cur.fetchall() if r[0] not in ('frustrated_sentiment_pos_neg_v2_sentiment_combined', 'frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only')]
    # Count all input tables across all versions for the progress bar
    total_tables = 0
    version_to_tables = {}
    for version_prefix in VERSION_EXTRA_COLS.keys():
        tables = [n for n in all_names if n.lower().startswith(version_prefix.lower()) and not n.lower().startswith("sqlite_")]
        version_to_tables[version_prefix] = tables
        total_tables += len(tables)

    processed = 0
    progress_bar(processed, total_tables, prefix="Processing")

    for version_prefix, extra_cols in VERSION_EXTRA_COLS.items():
        target_name = f"{version_prefix}_sentiment_combined_negative_only"
        input_tables = version_to_tables[version_prefix]

        all_rows = []

        for table in input_tables:
            # Discover columns
            info_df = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
            table_cols = set(info_df["name"].tolist())

            # Build WHERE clause for negatives
            where_clauses = []

            # Version-aware negative filters:
            # - v2/emotions: final_label = 'NEGATIVE'
            # - v1: sentiment_label = 'negative' (case-insensitive)
            if "final_label" in table_cols:
                where_clauses.append("final_label = 'NEGATIVE' COLLATE NOCASE")
            elif "sentiment_label" in table_cols:
                where_clauses.append("sentiment_label = 'negative' COLLATE NOCASE")
            else:
                # If neither is present, we cannot detect negativity; skip table
                processed += 1
                progress_bar(processed, total_tables, prefix="Processing")
                continue

            # Steam-specific exclusion: voted_up != 1 (if column exists)
            is_steam = "steam" in table.lower()
            if is_steam and "voted_up" in table_cols:
                where_clauses.append("(voted_up IS NULL OR voted_up != 1)")

            where_sql = " AND ".join(where_clauses)

            # Build SELECT list (mirror your combine behavior)
            main_text_sel = alias_first_present(table_cols, MAIN_TEXT_CANDIDATES, "main_text")
            time_sel = [c for c in TIME_CANDIDATES if c in table_cols]
            sentiment_sel = [(col if col in table_cols else f"NULL AS {col}") for col in extra_cols]
            select_parts = [main_text_sel] + sentiment_sel + time_sel
            select_sql = ", ".join(select_parts)

            # Pull only NEGATIVE rows here
            df = pd.read_sql_query(f"SELECT {select_sql} FROM {table} WHERE {where_sql}", conn)

            # Normalize time
            parsed = best_datetime_series(df, TIME_CANDIDATES)
            if parsed.notna().any():
                df["time"] = parsed.dt.strftime("%Y-%m-%d").values
            else:
                df["time"] = None

            # Attach game/platform (same as combine)
            df["game_name"] = get_game_name(table)
            df["comment_platform"] = get_platform_name(table)

            if not df.empty:
                final_cols = BASE_FINAL_COLS + extra_cols
                df = df.reindex(columns=final_cols)
                all_rows.append(df)

            processed += 1
            progress_bar(processed, total_tables, prefix="Processing")

        # Write combined negative-only table for this version
        if all_rows:
            combined = pd.concat(all_rows, ignore_index=True)
            combined.to_sql(target_name, conn, if_exists="replace", index=False)
            print(f"\n✅ Saved {len(combined)} rows to '{target_name}'.")
        else:
            print(f"\n(no negative data to combine for version '{version_prefix}')")

    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
