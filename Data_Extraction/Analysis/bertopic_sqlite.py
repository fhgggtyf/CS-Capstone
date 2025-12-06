"""
Simple BERTopic implementation for categorising player frustration causes
from game review data stored in a SQLite database.

This script connects to an SQLite database, extracts a column of review text
from a user‑specified table, and then applies the BERTopic topic modelling
pipeline to cluster the reviews. It outputs two CSV files:

* reviews_with_topics.csv – original reviews with their assigned topic id
* topic_keywords.csv – summary of topics with cTF‑IDF keywords and counts

Before running this script you will need to install the following packages:

    pip install bertopic sentence-transformers pandas scikit-learn hdbscan

Refer to the BERTopic documentation for more details on optional
configuration parameters. At a high level, BERTopic leverages a
transformer embedding model, UMAP dimensionality reduction, HDBSCAN
clustering and cTF‑IDF for keyword extraction to discover coherent topics
in text【998928615357404†L165-L171】.
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

import re
import spacy
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from rapidfuzz import fuzz
from tqdm import tqdm

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os

# Stop parallel deadlocks in Huggingface tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prevent BLAS from spawning too many threads (helps stability)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# -----------------------------
# Preprocessing configuration
# -----------------------------

# Raw game names as given (for removal)
RAW_GAME_TITLES = [
    "Crash 4",
    "Dark Souls",
    "Sekiro",
    "Cuphead",
    "Super Meat Boy",
    "Devil May Cry 5",
    "Battlefield 4",
    "Assasin's Creed: Unity",
    "Cyberpunk 2077",
    "Halo: MCC",
    "Fallout 76",
    "Mass Effect: Andromeda",
    "Battlefield 2042",
    "Batman: Arkham Knight (PC)",
    "Anthem",
    "No Man's Sky",
    "Starwars Battlefront 2",
    "Mighty No.9",
    "Fortnite",
    "DOTA 2",
    "League of Legends",
    "Rainbow 6: Siege",
    "The Binding Of Isaac: Repentance",
    "Getting Over It with Bennett Foddy",
    "Furi",
    "Jump King",
    "Mortal Shell",
    "Street Fighter V",
    "CS2",
    "APEX",
    "GENSHIN",
    "Call of Duty: Warzone",
    "Valorant",
    "Hogwarts: Legacy",
    "PalWorld",
    "Baldur's Gate 3",
    "LOTR: Gollum",
    "COD: Modern Warfare III",
    "Warcraft III: Reforged",
    "PUBG: BATTLEGROUNDS",
    "War Thunder",
    "Helldivers 2",
    "Overwatch 2",
    "Delta Force (2024)",
    "EA SPORTS FC 25",
    "Escape From Tarkov",
]


def _normalize_text_basic(text: str) -> str:
    """Step ① normalize: lowercase, strip URLs, keep alnum + space."""
    text = text.lower()
    # remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # remove non-alphanumeric chars (keep spaces)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_game_tokens(titles: list[str]) -> set[str]:
    """Turn game titles into tokens to remove (e.g., 'dark souls' -> 'dark', 'souls')."""
    tokens: set[str] = set()
    for title in titles:
        t = title.lower()
        # remove punctuation
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        for tok in t.split():
            if tok:
                tokens.add(tok)
        # Generate abbreviations: e.g., "assassins creed unity" -> "acu", "ac", etc.
        abbr = "".join(tok[0] for tok in t.split() if tok)
        tokens.add(abbr)
        if len(abbr) >= 2:
            tokens.add(abbr.lower())
    # also add some common abbreviations people might use
    # extra = {
    #     "dota", "lol", "r6", "cs", "csgo", "eft", "mw3", "mwii", "fc",
    #     "warzone", "helldivers", "overwatch", "pubg", "tarkov",
    # }
    # tokens.update(extra)
    return tokens


# Build stopword set: English stopwords + game tokens
# Custom because library ones too aggressive
BASE_STOPWORDS = {

    # --- Core function words ---
    "the", "a", "an", "and", "or", "so", "but",
    "if", "while", "though", "although", "however", "therefore", "then",
    "this", "that", "these", "those",
    "here", "there", "where", "when",
    "on", "in", "at", "to", "for", "from", "of", "into", "onto",
    "over", "under", "with", "without", "within", "across", "about",
    "as", "by", "via", "per",

    # --- Soft junk / filler words ---
    "really", "very", "just", "also", "even", "still",
    "literally", "basically", "actually",
    "maybe", "somehow", "kinda", "sorta",
    "etc", "etc.", "etcetera",
    "thing", "things", "stuff",

    # --- Internet slang / noise ---
    "lol", "lmao", "lmfao", "rofl",
    "omg", "wtf", "wtff", "idk",
    "pls", "plz", "u", "ur", "im", "ive",
    "xd", "XD", "xD",
    "haha", "hahaha", "hehe", "lul",

    # --- Weak verbs that dilute meaning ---
    "take", "give", "make", "put",
    "got", "gets", "getting",
    "use", "used", "using",

    # --- Unuseful profanity / offensive language ---
    "fuck", "fucks", "fucking", "fucked", "fuckin",
    "shit", "shits", "shitty",
    "bitch", "bitches", "bitching",
    "ass", "asses", "asshole", "assholes", "arse",
    "bastard", "bastards",
    "damn", "dammit", "damned",
    "hell", "wtf", "wtaf",
    "crap", "crappy",
    "dick", "dicks", "dickhead", "dickheads",
    "piss", "pissed", "pissing",
    "cunt", "cunts",
    "slut", "sluts",
    "whore", "whores",

    # --- Found repeating unuseful words ---
    "game", "games", 
    "play", "plays", "playing", "played", 
    "snail", # Random meme spam word
    "democracy", "dive", "earth", "Planet" # Helldivers2 meme spam noise
}

CUSTOM_STOPWORDS = BASE_STOPWORDS
# GAME_TOKENS = _build_game_tokens(RAW_GAME_TITLES)
# CUSTOM_STOPWORDS = BASE_STOPWORDS.union(GAME_TOKENS)


# # Global spaCy model (load once)
# # Disable parser/NER for speed; we only need lemmatization + POS.
# NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
_NLP = None
def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _NLP



###############################
# TOKENIZATION + LEMMATIZATION
###############################

def _tokenize_lemmatize(text: str) -> list[str]:
    """
    Lemmatize tokens, normalize, keep game names, keep useful numbers.
    DO NOT remove stopwords here — that comes AFTER bigram/trigram detection.
    """
    nlp = get_nlp()
    doc = nlp(text)

    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue

        lemma = token.lemma_.lower().strip()
        if not lemma:
            continue

        # skip absurdly long junk tokens
        if len(lemma) > 40:
            continue

        # NUMBER HANDLING
        if lemma.isdigit():
            # reject insane numeric garbage (crashes otherwise)
            if len(lemma) > 6:
                continue
            try:
                num = int(lemma)
            except ValueError:
                continue

            # keep useful numbers like 60, 120, 144, 1, 2, 76
            if 0 < num < 2000:
                tokens.append(lemma)
            continue

        tokens.append(lemma)

    return tokens



def _detect_phrases(docs_tokens: list[list[str]]) -> list[list[str]]:
    """
    Step ④: detect bigrams and trigrams using Gensim Phrases.
    Returns list of token lists with phrases merged (e.g. 'frame_rate').
    """
    if not docs_tokens:
        return docs_tokens

    # Build bigram and trigram models
    bigram = Phrases(docs_tokens, min_count=20, threshold=10)
    trigram = Phrases(bigram[docs_tokens], min_count=15, threshold=10)

    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)

    docs_bi = [bigram_mod[doc] for doc in docs_tokens]
    docs_tri = [trigram_mod[doc] for doc in docs_bi]
    return docs_tri


###############################
# FULL PREPROCESSING
###############################

from tqdm import tqdm

def preprocess_docs(raw_docs: list[str], min_words: int = 5):
    """
    1. Normalize
    2. Lemmatize (keep game names)
    3. Detect bigrams/trigrams
    4. Remove stopwords (English only)
    5. Filter short docs
    """

    print("\nStep 1/5: Normalizing…")
    normalized = [_normalize_text_basic(d) for d in tqdm(raw_docs)]

    print("\nStep 2/5: Lemmatizing… (stopwords NOT removed here)")
    tokenized = [_tokenize_lemmatize(txt) for txt in tqdm(normalized)]

    print("\nStep 3/5: Detecting bigrams/trigrams… (step skipped)")
    # tokenized = _detect_phrases(tokenized)

    print("\nStep 4/5: Removing English stopwords…")
    filtered_tokens = []
    for toks in tqdm(tokenized):
        # REMOVE ONLY ENGLISH STOPWORDS — KEEP GAME NAMES
        cleaned = [t for t in toks if t not in BASE_STOPWORDS]
        filtered_tokens.append(cleaned)

    print("\nStep 5/5: Filtering short documents…")
    docs_final = []
    idx_map = []
    for i, toks in enumerate(filtered_tokens):
        if len(toks) >= min_words:
            docs_final.append(" ".join(toks))
            idx_map.append(i)

    print(f"\nPreprocessing complete! {len(docs_final)} documents retained.\n")
    return docs_final, idx_map



def load_reviews(db_path: Path, table_name: str, text_column: str) -> pd.Series:
    """Load review texts from a SQLite database.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.
    table_name : str
        Name of the table containing the reviews.
    text_column : str
        Name of the column with review text.

    Returns
    -------
    pd.Series
        A Pandas Series containing the review texts as strings.
    """
    if not db_path.is_file():
        raise FileNotFoundError(f"Database file '{db_path}' does not exist.")

    query = f"SELECT {text_column} FROM {table_name}"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in table '{table_name}'.")
    # Ensure the column is a string type
    reviews = df[text_column].astype(str)
    return reviews


def run_bertopic(
    docs: list[str],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    min_topic_size: int = 30,
    nr_topics: str | int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run BERTopic on a list of documents and return topic assignments and summaries.

    Parameters
    ----------
    docs : list[str]
        List of document strings (reviews).
    embedding_model_name : str, optional
        Name of the sentence transformer model to use for embeddings.
        Defaults to "all-MiniLM-L6-v2", which is fast and reasonably accurate.
    min_topic_size : int, optional
        Minimum size of topics; smaller clusters will be merged. Defaults to 30.
    nr_topics : str | int | None, optional
        If None, no topic reduction is performed. If "auto", BERTopic will
        automatically merge similar topics. Otherwise, you can specify the
        desired number of topics as an integer.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        * reviews_with_topics – a DataFrame of the original documents with an
          assigned topic id
        * topic_info – a DataFrame summarizing each topic (topic id, size,
          and cTF‑IDF keywords)
    """

    # Initialize BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic.representation import KeyBERTInspired

    # Use ONE embedding model for everything
    embed_model = SentenceTransformer(embedding_model_name)
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    # Representation model
    rep_model = KeyBERTInspired()

    # Initialize BERTopic using the SAME embedding model
    topic_model = BERTopic(
        embedding_model=embed_model,
        representation_model=rep_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=True,
    )

    # Fit and transform
    topics, _ = topic_model.fit_transform(docs, embeddings)

    # Assign topics to reviews
    reviews_with_topics = pd.DataFrame({"review": docs, "topic": topics})

    # Get topic information
    topic_info = topic_model.get_topic_info()

    return reviews_with_topics, topic_info


def save_outputs(reviews_with_topics: pd.DataFrame, topic_info: pd.DataFrame, output_dir: Path) -> None:
    """Save the reviews with topic assignments and topic summary to CSV files.

    Parameters
    ----------
    reviews_with_topics : pd.DataFrame
        DataFrame containing the original reviews and their assigned topic ids.
    topic_info : pd.DataFrame
        DataFrame summarising each topic (topic id, count, and keywords).
    output_dir : Path
        Directory in which to save the CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    reviews_path = output_dir / "reviews_with_topics.csv"
    topic_info_path = output_dir / "topic_keywords.csv"
    reviews_with_topics.to_csv(reviews_path, index=False)
    topic_info.to_csv(topic_info_path, index=False)
    print(f"Saved topic assignments to {reviews_path}")
    print(f"Saved topic summary to {topic_info_path}")


def parse_nr_topics(x):
    """Parse nr_topics to allow integers or 'auto' while rejecting bad input."""
    if x is None:
        return None
    x = x.strip().lower()
    if x == "auto":
        return "auto"
    try:
        return int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "nr-topics must be an integer or 'auto'."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BERTopic on game reviews stored in a SQLite database"
    )
    parser.add_argument("db_path", type=Path, help="Path to the SQLite database file")
    parser.add_argument("table", type=str, help="Name of the table containing reviews")
    parser.add_argument("column", type=str, help="Name of the column with review text")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to save CSV output",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=30,
        help="Minimum size of topics for BERTopic",
    )
    parser.add_argument(
        "--nr-topics",
        type=parse_nr_topics,
        default=None,
        help="Number of topics to reduce to (int) or 'auto'. Default: None",
    )

    args = parser.parse_args()

    # Load review data
    reviews = load_reviews(args.db_path, args.table, args.column)
    print(f"Loaded {len(reviews)} raw reviews from {args.db_path}")

    # Preprocess: normalize, stopwords, lemmatize, bigrams/trigrams, filter short
    preprocessed_docs, idx_map = preprocess_docs(reviews.tolist())
    print(f"{len(preprocessed_docs)} reviews remaining after preprocessing")

    # get original reviews back by index
    original_selected = reviews.iloc[idx_map].tolist()

    # Run BERTopic on preprocessed text
    reviews_with_topics, topic_info = run_bertopic(
        docs=preprocessed_docs,
        embedding_model_name=args.embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
    )

    # ADD original reviews to dataframe
    reviews_with_topics["original_review"] = original_selected
    reviews_with_topics["cleaned_review"] = preprocessed_docs

    # Save outputs
    save_outputs(reviews_with_topics, topic_info, args.output_dir)



if __name__ == "__main__":
    main()