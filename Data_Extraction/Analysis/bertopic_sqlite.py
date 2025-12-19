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
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, BERTopic]:
    """Run BERTopic on a list of documents and return topic assignments, summaries and the model.

    This helper wraps BERTopic to produce reproducible results via a controllable
    ``random_state`` parameter. When provided, the random state is used to seed
    the underlying UMAP dimensionality reduction. If ``random_state`` is ``None``,
    BERTopic will use its default randomness which may lead to different topic
    assignments across runs.

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
        If ``None``, no topic reduction is performed. If ``"auto"``, BERTopic will
        automatically merge similar topics. Otherwise, you can specify the
        desired number of topics as an integer.
    random_state : int | None, optional
        An optional integer to seed the underlying UMAP model for reproducibility.
        When ``None``, randomness is uncontrolled.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, BERTopic]
        A tuple containing:
        * reviews_with_topics – a DataFrame of the documents with an assigned topic id
        * topic_info – a DataFrame summarising each topic (topic id, size, and keywords)
        * topic_model – the fitted BERTopic instance for further inspection/visualisation
    """

    # Initialize BERTopic
    from sentence_transformers import SentenceTransformer
    from bertopic.representation import KeyBERTInspired

    # UMAP will be seeded for reproducibility when random_state is provided. UMAP
    # (from the umap‑learn package) must be imported here because
    # ``bertopic`` does not expose a top‑level ``random_state`` argument.
    try:
        from umap import UMAP  # type: ignore
    except ImportError:
        # Fallback: If UMAP isn't available, simply leave the model unseeded.
        UMAP = None  # type: ignore

    # Use ONE embedding model for everything
    embed_model = SentenceTransformer(embedding_model_name)
    embeddings = embed_model.encode(docs, show_progress_bar=True)

    # Representation model
    rep_model = KeyBERTInspired()

    # Construct a UMAP model seeded with the provided random state (if available).
    umap_model = None
    if random_state is not None and UMAP is not None:
        # Default UMAP settings from BERTopic; adjust as desired.
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
        )

    # Initialize BERTopic using the same embedding model. When ``umap_model`` is
    # provided, BERTopic will use it for dimensionality reduction; otherwise it
    # falls back to its own defaults.
    topic_model = BERTopic(
        embedding_model=embed_model,
        representation_model=rep_model,
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        # Enable low_memory to reduce memory usage during UMAP and cTF‑IDF【53561172673549†L299-L324】.
        low_memory=True,
        verbose=True,
    )

    # Fit and transform
    topics, _ = topic_model.fit_transform(docs, embeddings)

    # Assign topics to reviews
    reviews_with_topics = pd.DataFrame({"review": docs, "topic": topics})

    # Get topic information
    topic_info = topic_model.get_topic_info()

    return reviews_with_topics, topic_info, topic_model


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

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducibility. Overrides the DEFAULT_SEED constant. "
            "When not provided and DEFAULT_SEED is set, the constant will be used; "
            "otherwise the current time (seconds since epoch) will be used."
        ),
    )

    # Additional visualisation options. Datamap relies on datashader and is suitable for
    # extremely large corpora. When enabled, the regular visualize_documents scatter will
    # be skipped and a datashader-based plot will be generated instead.
    parser.add_argument(
        "--doc-datamap",
        action="store_true",
        help=(
            "Generate a DataMapPlot (datashader) visualisation of documents coloured by topic. "
            "Recommended for corpora with hundreds of thousands of documents."
        ),
    )
    parser.add_argument(
        "--doc-datamap-interactive",
        action="store_true",
        help=(
            "When using --doc-datamap, save an interactive HTML instead of a static PNG. "
            "If omitted, a static PNG will be written."
        ),
    )
    parser.add_argument(
        "--doc-datamap-subsample",
        type=int,
        default=None,
        help=(
            "Optional number of documents to sample when fitting the 2D UMAP for the DataMapPlot. "
            "Fitting UMAP on all documents may be slow and memory intensive. When provided, a random "
            "subset of this size will be used to fit UMAP, after which all documents will be transformed."
        ),
    )

    # Maximum number of documents to include in the DataMapPlot. If the corpus
    # contains more documents than this limit, a random subset of that size
    # will be used for the datashader visualisation. This provides a hard cap
    # on memory usage during the document‑level visualisation. A smaller
    # sample trades off some precision in the scatter layout for guaranteed
    # stability. When unset, all documents will be visualised (not
    # recommended for very large datasets).
    parser.add_argument(
        "--doc-datamap-max-docs",
        type=int,
        default=None,
        help=(
            "Maximum number of documents to visualise in the DataMapPlot. When the number of "
            "documents exceeds this limit, a random sample of this many documents will be used. "
            "Setting this option provides a deterministic upper bound on memory use at the cost "
            "of only showing a subset of documents. If omitted, all documents are used (not "
            "recommended on very large corpora)."
        ),
    )

    args = parser.parse_args()

    # Determine the random seed. Users can override via the --seed flag, or by
    # editing the DEFAULT_SEED constant defined below. If both are None, fall
    # back to using the current Unix timestamp.
    # You can edit DEFAULT_SEED to a fixed integer to always obtain the same
    # results without specifying --seed at the command line.
    DEFAULT_SEED: int | None = None  # set to an integer for deterministic runs
    import time, random as _py_random, numpy as _py_np

    if args.seed is not None:
        seed = args.seed
    elif DEFAULT_SEED is not None:
        seed = DEFAULT_SEED
    else:
        seed = int(time.time())

    # Seed Python's and NumPy's random number generators. BERTopic relies on
    # UMAP which uses NumPy's global random state when the random_state argument is None.
    _py_random.seed(seed)
    _py_np.random.seed(seed)
    print(f"Using random seed: {seed}")
    # Load review data
    reviews = load_reviews(args.db_path, args.table, args.column)
    print(f"Loaded {len(reviews)} raw reviews from {args.db_path}")

    # Preprocess: normalize, stopwords, lemmatize, bigrams/trigrams, filter short
    preprocessed_docs, idx_map = preprocess_docs(reviews.tolist())
    print(f"{len(preprocessed_docs)} reviews remaining after preprocessing")

    # get original reviews back by index
    original_selected = reviews.iloc[idx_map].tolist()

    # Run BERTopic on preprocessed text
    reviews_with_topics, topic_info, topic_model = run_bertopic(
        docs=preprocessed_docs,
        embedding_model_name=args.embedding_model,
        min_topic_size=args.min_topic_size,
        nr_topics=args.nr_topics,
        random_state=seed,
    )


    # ADD original reviews to dataframe
    reviews_with_topics["original_review"] = original_selected
    reviews_with_topics["cleaned_review"] = preprocessed_docs

    # Persist outputs
    save_outputs(reviews_with_topics, topic_info, args.output_dir)

    # Write the seed used for this run into the output directory so that
    # downstream users can reproduce the exact same topic assignments.
    try:
        seed_file = args.output_dir / "seed.txt"
        with open(seed_file, "w", encoding="utf-8") as f:
            f.write(str(seed))
        print(f"Saved random seed to {seed_file}")
    except Exception as _exc:
        print(f"Could not write seed file: {_exc}")

    # ---------------------------------------------------------------------
    # Visualisations
    # ---------------------------------------------------------------------
    # After fitting BERTopic, we can generate a suite of interactive
    # visualisations to better understand the discovered topics. These
    # visualisations are based on Plotly and will either display in a Jupyter
    # notebook or save to HTML files in the output directory.
    try:
        # Generate an overview of the topics using a reduced 2D embedding. This
        # is similar to the PyLDAvis intertopic distance map, showing topics as
        # bubbles where size corresponds to frequency and distance corresponds
        # roughly to similarity/dissimilarity.
        fig_overview = topic_model.visualize_topics()
        overview_path = args.output_dir / "topics_overview.html"
        fig_overview.write_html(str(overview_path))
        print(f"Saved topics overview visualisation to {overview_path}")

        # Bar chart of the top terms per topic for the most frequent topics.
        fig_barchart = topic_model.visualize_barchart(top_n_topics=10)
        barchart_path = args.output_dir / "topics_barchart.html"
        fig_barchart.write_html(str(barchart_path))
        print(f"Saved topics bar chart to {barchart_path}")

        # Hierarchical clustering of topics to inspect relationships between topics.
        fig_hier = topic_model.visualize_hierarchy()
        hier_path = args.output_dir / "topics_hierarchy.html"
        fig_hier.write_html(str(hier_path))
        print(f"Saved topics hierarchy to {hier_path}")

        # Heatmap showing the similarity between topics.
        fig_heatmap = topic_model.visualize_heatmap()
        heatmap_path = args.output_dir / "topics_heatmap.html"
        fig_heatmap.write_html(str(heatmap_path))
        print(f"Saved topics heatmap to {heatmap_path}")

        # Term rank distribution visualisation to explore term distributions within topics.
        fig_termrank = topic_model.visualize_term_rank()
        termrank_path = args.output_dir / "topics_term_rank.html"
        fig_termrank.write_html(str(termrank_path))
        print(f"Saved topics term rank visualisation to {termrank_path}")

        # Document‑level visualisation. If the --doc-datamap flag is set, we generate a
        # datashader‑based DataMapPlot instead of the interactive scatter, as the
        # latter will consume excessive memory on very large datasets. Otherwise,
        # fallback to the default Plotly scatter.
        if args.doc_datamap:
            print("Generating DataMapPlot for documents…")
            try:
                # Determine the set of documents and corresponding topic labels to include
                # in the DataMapPlot. When the corpus size exceeds the max limit, sample
                # that many documents uniformly at random. Sampling is deterministic
                # using the provided seed. We also sample the topic labels using the
                # same indices to ensure lengths match between docs, topics and
                # embeddings. If no sampling is requested, use the full set.
                docs_for_datamap = preprocessed_docs
                topics_for_datamap = topic_model.topics_  # full topics for all docs
                sample_indices = None
                import numpy as _np
                if (
                    args.doc_datamap_max_docs is not None
                    and args.doc_datamap_max_docs > 0
                    and len(preprocessed_docs) > args.doc_datamap_max_docs
                ):
                    rng = _np.random.default_rng(seed)
                    sample_indices = rng.choice(
                        len(preprocessed_docs), size=args.doc_datamap_max_docs, replace=False
                    )
                    # Sort indices to maintain approximate original order
                    sample_indices.sort()
                    docs_for_datamap = [preprocessed_docs[i] for i in sample_indices]
                    # Align the topic labels with the sampled documents. topics_ is a
                    # list/array of length equal to the number of documents. We select
                    # the same indices to get topics for the sampled docs. If topics_
                    # is a list, list comprehension will work; if it is a numpy array,
                    # indexing with an array also works.
                    topics_arr = topic_model.topics_
                    try:
                        topics_for_datamap = [topics_arr[i] for i in sample_indices]
                    except Exception:
                        topics_for_datamap = topics_arr[sample_indices]

                # Compute sentence embeddings for the selected documents using the same
                # embedding model as BERTopic. Computing embeddings here avoids
                # recomputation inside visualize_document_datamap. We encode only
                # the sampled docs (or all docs if no sampling).
                from sentence_transformers import SentenceTransformer
                embed_model_dm = SentenceTransformer(args.embedding_model)
                emb_dm = embed_model_dm.encode(docs_for_datamap, show_progress_bar=True)

                # Construct a 2D UMAP model seeded with the random seed for reproducibility.
                from umap import UMAP  # type: ignore
                umap_2d = UMAP(
                    n_neighbors=15,
                    n_components=2,
                    min_dist=0.0,
                    metric="cosine",
                    random_state=seed,
                )
                # Optionally sample a subset of the embeddings for fitting UMAP to
                # further reduce memory and time during UMAP learning. Only apply
                # subsampling if the requested subsample size is positive and
                # smaller than the number of selected documents.
                if (
                    args.doc_datamap_subsample is not None
                    and args.doc_datamap_subsample > 0
                    and args.doc_datamap_subsample < len(emb_dm)
                ):
                    rng = _np.random.default_rng(seed)
                    idx_umap = rng.choice(
                        len(emb_dm), size=args.doc_datamap_subsample, replace=False
                    )
                    umap_2d.fit(emb_dm[idx_umap])
                else:
                    umap_2d.fit(emb_dm)
                # Transform all selected embeddings to 2D.
                reduced_embeddings = umap_2d.transform(emb_dm)

                # Use BERTopic's datashader visualization. Passing both the reduced
                # embeddings and the sampled topics ensures that lengths match
                # across docs, topics and embeddings. When interactive=True, a
                # Plotly HTML file is saved; otherwise a static PNG is saved.
                fig_datamap = topic_model.visualize_document_datamap(
                    docs_for_datamap,
                    topics=topics_for_datamap,
                    reduced_embeddings=reduced_embeddings,
                    interactive=args.doc_datamap_interactive,
                )
                if args.doc_datamap_interactive:
                    dm_path = args.output_dir / "documents_datamap.html"
                    fig_datamap.write_html(str(dm_path))
                else:
                    dm_path = args.output_dir / "documents_datamap.png"
                    fig_datamap.savefig(str(dm_path), dpi=300)
                print(
                    f"Saved documents DataMapPlot to {dm_path} (visualising {len(docs_for_datamap)} documents)"
                )
            except Exception as e:
                print(f"Could not generate DataMapPlot: {e}")
        else:
            # Fallback: interactive Plotly scatter via visualize_documents. Be aware
            # that this will be memory intensive on large datasets【913976633007349†L134-L150】.
            fig_documents = topic_model.visualize_documents(preprocessed_docs)
            docs_path = args.output_dir / "documents_umap.html"
            fig_documents.write_html(str(docs_path))
            print(f"Saved documents UMAP projection to {docs_path}")

    except Exception as viz_e:
        print(f"One or more visualisations could not be created: {viz_e}")



if __name__ == "__main__":
    main()