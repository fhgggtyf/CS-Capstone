# lda_topic_modeling_improved.py
#
# An enhanced version of the original streaming LDA pipeline.  This script
# preserves the streaming design (two-pass over a SQLite database) but adds
# several features aimed at making interpretation easier and reducing
# overlap between topics.  Notable additions include:
#
# * Exposed hyperparameters for the Dirichlet priors (``alpha`` and ``eta``)
#   so you can encourage sparser document–topic and topic–word
#   distributions.  Lower values will often lead to more distinct topics.
#
# * Optional bigram detection using ``gensim.models.Phrases``.  When
#   enabled, the script learns common word pairs on a bounded sample
#   (reservoir) and applies these phrases to the entire corpus.  This can
#   disambiguate topics by capturing collocations like ``"climate_change"``
#   instead of treating ``"climate"`` and ``"change"`` separately.
#
# * Vocabulary pruning: beyond the existing ``min_df`` and ``max_df``
#   thresholds, you can drop the ``N`` most frequent tokens after
#   dictionary construction via ``--drop-top-n``.  This is useful for
#   removing ubiquitous words that slip past the stopword list.
#
# * Export of topic word lists and document–topic weights.  After
#   training, the script writes ``outputs/topics_top_words.json`` (the
#   top terms and their probabilities per topic) and
#   ``outputs/doc_topic_weights.csv`` (one row per document with the
#   per-topic probabilities and dominant topic).  These files make it
#   straightforward to inspect the model output in Excel or other tools.
#
# * Simple automatic labeling of topics.  You can pass
#   ``--assign-themes simple`` to generate a human-friendly label for each
#   topic by concatenating the top words.  More sophisticated labeling
#   methods (e.g., using ``keybert`` or sentence embeddings) could be
#   plugged in here, but they are not enabled by default to avoid
#   external dependencies.

import argparse
import csv
import os
import random
import re
import sqlite3
import time
import math
import json
from collections import Counter
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import entropy

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora
from gensim.corpora import MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import CoherenceModel

try:
    # Optional UMAP import – falls back to PCA if unavailable.
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    # Optional bigram support – will import when needed.
    from gensim.models.phrases import Phrases, Phraser
    HAS_PHRASES = True
except Exception:
    HAS_PHRASES = False

def ensure_nltk():
    """Ensure required NLTK corpora are present.  Downloads if missing."""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')


def simple_tokenize(text: str):
    """Tokenize text into lowercased alphabetic tokens longer than two characters."""
    return [t for t in re.findall(r"[a-zA-Z]+", str(text or "").lower()) if len(t) > 2]


def preprocess_tokens(tokens, sw, lemm=None):
    """Remove stopwords and optionally lemmatize tokens."""
    toks = [w for w in tokens if w not in sw]
    if lemm is not None:
        toks = [lemm.lemmatize(w) for w in toks]
    return toks


def normalize_rows(mat):
    """Normalize rows of a 2D array to sum to one, avoiding division by zero."""
    s = mat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return mat / s


def jensen_shannon_divergence_matrix(P):
    """Compute symmetric Jensen–Shannon distance between all pairs of rows."""
    n = P.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = jensenshannon(P[i], P[j], base=2.0)
            M[i, j] = d
            M[j, i] = d
    return M


def gini_coefficient(counts):
    """Compute the Gini coefficient of an array of counts."""
    x = np.array(sorted(counts))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def topic_diversity(top_words):
    """Compute diversity metrics for topic word lists.

    Returns the fraction of unique words across all topics and the mean
    Jaccard overlap between all pairs of topics (lower is better).
    """
    k = len(top_words)
    n = len(top_words[0]) if k else 0
    all_words = set(w for tw in top_words for w in tw)
    unique_ratio = len(all_words) / max(1, k * n)
    jacc = []
    for a, b in combinations(top_words, 2):
        A, B = set(a), set(b)
        u = len(A | B)
        jacc.append(len(A & B) / u if u else 0.0)
    return unique_ratio, (float(np.mean(jacc)) if jacc else 0.0)


def dominant_topic_per_doc(doc_topic_dist):
    """Return an array of dominant topics for each document distribution."""
    return np.array([(max(dist, key=lambda x: x[1])[0] if dist else -1) for dist in doc_topic_dist])


# ----------------- Streaming over SQLite -----------------
class SQLiteBatchTextStream:
    """Streams raw texts from a SQLite table in batches."""
    def __init__(self, db_path: str, table: str, text_col: str, batch_size: int = 5000):
        self.db_path = db_path
        self.table = table
        self.text_col = text_col
        self.batch_size = batch_size

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT {self.text_col} FROM {self.table}")
        while True:
            rows = cur.fetchmany(self.batch_size)
            if not rows:
                break
            for (txt,) in rows:
                yield txt
        conn.close()


class TokenStream:
    """Streams tokenized documents from a raw-text iterator.

    If a bigram phraser is provided, it is applied after basic tokenization
    and preprocessing (stopword removal and lemmatization).  This class
    encapsulates the logic for tokenization and optionally phrase detection.
    """
    def __init__(self, raw_iter, sw, lemm=None, bigram_phraser=None):
        self.raw_iter = raw_iter
        self.sw = sw
        self.lemm = lemm
        self.bigram_phraser = bigram_phraser

    def __iter__(self):
        for t in self.raw_iter:
            tokens = simple_tokenize(t)
            tokens = preprocess_tokens(tokens, self.sw, self.lemm)
            if self.bigram_phraser is not None:
                tokens = self.bigram_phraser[tokens]
            yield tokens


def build_bigram_phraser(sample_texts, min_count: int = 20, threshold: float = 10.0):
    """Train a bigram Phraser on a list of tokenized documents.

    If ``sample_texts`` is empty or bigram support is unavailable, returns
    ``None``.
    """
    if not sample_texts or not HAS_PHRASES:
        return None
    phrases = Phrases(sample_texts, min_count=min_count, threshold=threshold, progress_per=10000)
    return Phraser(phrases)


def assign_theme_simple(top_words, num_terms=3):
    """Create a simple descriptive label by joining the first ``num_terms`` words."""
    return "_".join(top_words[:num_terms]) if top_words else ""


def main():
    parser = argparse.ArgumentParser(description="Streaming LDA with improved interpretability and configurability.")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--table", required=True, help="Table name in SQLite DB")
    parser.add_argument("--text-col", default="main_text", help="Column containing the text")
    parser.add_argument("--k", type=int, default=20, help="Number of topics")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--passes", type=int, default=15)
    parser.add_argument("--iters", type=int, default=600)
    parser.add_argument("--chunksize", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=5000, help="DB fetch size")
    parser.add_argument("--min-df", type=int, default=20, help="Min document frequency for dictionary filtering")
    parser.add_argument("--max-df", type=float, default=0.4, help="Max document frequency (fraction)")
    parser.add_argument("--topn", type=int, default=15, help="Number of top words per topic to save")
    parser.add_argument("--coherence-sample", type=int, default=20000, help="Sample size for c_v/c_npmi coherence")
    parser.add_argument("--pyldavis-sample", type=int, default=20000, help="Sample size for pyLDAvis")
    parser.add_argument("--scatter-sample", type=int, default=25000, help="Sample size for 2D scatter plot")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--alpha", default="asymmetric", help="Dirichlet prior for doc-topic (float or 'asymmetric'/'symmetric')")
    parser.add_argument("--eta", type=float, default=None, help="Dirichlet prior for topic-word; smaller values make topics sparser")
    parser.add_argument("--drop-top-n", type=int, default=0, help="Drop the N most frequent tokens after filtering extremes")
    parser.add_argument("--use-bigrams", action="store_true", help="Enable bigram phrase detection using gensim.Phrases")
    parser.add_argument("--bigram-min-count", type=int, default=20, help="Min count for bigram detection (when enabled)")
    parser.add_argument("--bigram-threshold", type=float, default=10.0, help="Threshold for bigram detection (when enabled)")
    parser.add_argument("--assign-themes", default="none", choices=["none", "simple"], help="Automatically assign a label/theme to each topic")
    parser.add_argument("--topics-file", default="outputs/topics_top_words.json", help="Path to JSON file for topic words")
    parser.add_argument("--doc-topics-file", default="outputs/doc_topic_weights.csv", help="Path to CSV file for document-topic weights")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs("outputs", exist_ok=True)

    # NLTK preparation
    ensure_nltk()
    sw = set(stopwords.words('english'))
    lemm = WordNetLemmatizer()

    # ----------------- Pass 1: Gather sample texts and count docs -----------------
    print("Pass 1/3: Streaming documents to build sample and count docs…")
    raw_stream_pass1 = SQLiteBatchTextStream(args.db, args.table, args.text_col, args.batch_size)
    sample_texts = []  # reservoir sample for coherence and bigram training
    n_docs_seen = 0
    for text in tqdm(raw_stream_pass1, desc="Pass1 (docs)"):
        tokens = simple_tokenize(text)
        tokens = preprocess_tokens(tokens, sw, lemm)
        n_docs_seen += 1
        # reservoir sampling for coherence and bigram training
        if args.coherence_sample > 0:
            if len(sample_texts) < args.coherence_sample:
                sample_texts.append(tokens)
            else:
                j = random.randint(0, n_docs_seen)
                if j < args.coherence_sample:
                    sample_texts[j] = tokens

    print(f"Total documents seen: {n_docs_seen}")

    # ----------------- Optional: learn bigram phraser -----------------
    bigram_phraser = None
    if args.use_bigrams:
        if not HAS_PHRASES:
            print("gensim.models.Phrases not available; bigram detection disabled.")
        else:
            print("Training bigram model on reservoir sample…")
            bigram_phraser = build_bigram_phraser(sample_texts, min_count=args.bigram_min_count, threshold=args.bigram_threshold)
            if bigram_phraser is None:
                print("Bigram training yielded no model; continuing without bigrams.")
            else:
                print("Bigram model trained.")
    # Convert sample texts to bigram tokens if necessary for coherence
    if bigram_phraser is not None:
        sample_texts = [bigram_phraser[toks] for toks in sample_texts]

    # ----------------- Pass 2: Build dictionary -----------------
    print("Pass 2/3: Building dictionary…")
    dictionary = corpora.Dictionary()
    raw_stream_pass2 = SQLiteBatchTextStream(args.db, args.table, args.text_col, args.batch_size)
    token_stream_pass2 = TokenStream(raw_stream_pass2, sw, lemm, bigram_phraser)
    for toks in tqdm(token_stream_pass2, total=n_docs_seen, desc="Pass2 (dict)"):
        dictionary.add_documents([toks], prune_at=None)

    # Filter extremes
    dictionary.filter_extremes(no_below=args.min_df, no_above=args.max_df, keep_n=None)
    # Drop most frequent tokens if requested
    if args.drop_top_n > 0:
        dictionary.filter_n_most_frequent(args.drop_top_n)
    dictionary.compactify()
    print(f"Dictionary size after filtering: {len(dictionary)}")

    # ----------------- Pass 3: Stream BOWs into MmCorpus -----------------
    print("Pass 3/3: Serializing corpus to outputs/corpus.mm …")
    corpus_mm_path = "outputs/corpus.mm"
    raw_stream_pass3 = SQLiteBatchTextStream(args.db, args.table, args.text_col, args.batch_size)
    token_stream_pass3 = TokenStream(raw_stream_pass3, sw, lemm, bigram_phraser)

    def bow_stream_with_progress():
        for toks in tqdm(token_stream_pass3, total=n_docs_seen, desc="Pass3 (serialize)"):
            yield dictionary.doc2bow(toks)

    # Serialize to disk using the built dictionary
    MmCorpus.serialize(corpus_mm_path, bow_stream_with_progress(), id2word=dictionary)
    corpus = MmCorpus(corpus_mm_path)
    num_docs = corpus.num_docs
    print(f"Serialized corpus with {num_docs} documents.")

    # ----------------- Train/Test split -----------------
    print("Splitting train/test…")
    all_idx = np.arange(num_docs)
    np.random.shuffle(all_idx)
    split = int(num_docs * 0.9)
    train_idx, test_idx = all_idx[:split], all_idx[split:]
    corpus_test = [corpus[i] for i in tqdm(test_idx, desc="Build test", unit="docs")]
    corpus_train = [corpus[i] for i in tqdm(train_idx, desc="Build train", unit="docs")]

    # ----------------- Train LDA -----------------
    print("Training topic model (LdaMulticore)…")
    start = time.time()
    # Convert alpha argument: allow float, list or 'asymmetric'/'symmetric'
    alpha_param = args.alpha
    # If numeric string, convert to float
    try:
        alpha_param = float(alpha_param)
    except (ValueError, TypeError):
        pass
    lda = LdaMulticore(
        corpus=corpus_train,
        id2word=dictionary,
        num_topics=args.k,
        random_state=args.seed,
        passes=args.passes,
        iterations=args.iters,
        chunksize=args.chunksize,
        workers=args.workers,
        alpha=alpha_param,
        eta=args.eta,
        eval_every=None
    )
    print(f"Training done in {time.time() - start:.1f}s.")

    # ----------------- Metrics -----------------
    print("Scoring perplexity on held-out…")
    log_perp = lda.log_perplexity(corpus_test)
    perplexity = math.exp(-log_perp)

    print("Computing coherence (u_mass, c_v, c_npmi)…")
    cm_umass = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coh_umass = cm_umass.get_coherence()
    coh_cv = coh_cnpmi = None
    if sample_texts:
        cm_cv = CoherenceModel(model=lda, texts=sample_texts, dictionary=dictionary, coherence='c_v')
        cm_cnpmi = CoherenceModel(model=lda, texts=sample_texts, dictionary=dictionary, coherence='c_npmi')
        coh_cv, coh_cnpmi = cm_cv.get_coherence(), cm_cnpmi.get_coherence()

    topics = lda.show_topics(num_topics=args.k, num_words=args.topn, formatted=False)
    top_word_lists = [[w for (w, p) in tw] for (_, tw) in topics]
    unique_ratio, avg_jacc = topic_diversity(top_word_lists)

    topic_word = normalize_rows(lda.get_topics())
    js_mat = jensen_shannon_divergence_matrix(topic_word)
    cos_mat = cosine_similarity(topic_word)

    # Topic sizes / balance (using train set counts as proxy)
    print("Estimating topic sizes from train set…")
    counts_train = Counter()
    for bow in tqdm(corpus_train, desc="Doc→topic (train)", unit="docs"):
        dist_i = lda.get_document_topics(bow, minimum_probability=0.0)
        if dist_i:
            k_dom = max(dist_i, key=lambda x: x[1])[0]
            counts_train[k_dom] += 1
    sizes_train = np.array([counts_train.get(k, 0) for k in range(args.k)], dtype=float)
    sizes_est = sizes_train * (num_docs / max(1, len(corpus_train)))
    proportions = sizes_est / max(1, sizes_est.sum())
    dist_entropy = entropy(proportions + 1e-12, base=2)
    gini = gini_coefficient(sizes_est)

    # ----------------- Save metrics -----------------
    summary = {
        "num_docs": int(num_docs),
        "num_topics": args.k,
        "vocab_size": int(topic_word.shape[1]),
        "engine": "lda_multicore",
        "coherence": {"c_v": coh_cv, "c_npmi": coh_cnpmi, "u_mass": coh_umass},
        "perplexity": perplexity,
        "topic_diversity": {"unique_word_ratio": unique_ratio, "avg_pairwise_jaccard": avg_jacc},
        "topic_size_proxy": {
            "counts_estimated": [float(x) for x in sizes_est],
            "proportions": [float(x) for x in proportions],
            "entropy_bits": float(dist_entropy),
            "gini": float(gini)
        }
    }
    with open("outputs/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Metrics (summary) ===")
    print(json.dumps(summary, indent=2))

    # ----------------- Export topic words -----------------
    print("Exporting topic words…")
    themes = []
    for idx, (topic_id, word_probs) in enumerate(topics):
        words = [(w, float(p)) for (w, p) in word_probs]
        label = None
        if args.assign_themes == "simple":
            label = assign_theme_simple([w for (w, _) in words])
        themes.append({"topic": int(topic_id), "label": label, "words": [
            {"term": w, "weight": float(p)} for (w, p) in words
        ]})
    with open(args.topics_file, "w") as f:
        json.dump(themes, f, indent=2)
    print(f"Topic words saved to {args.topics_file}")

    # ----------------- Export document-topic weights -----------------
    print("Exporting document-topic weights…")
    # Prepare header
    topic_headers = [f"topic_{k}" for k in range(args.k)]
    header = ["doc_index"] + topic_headers + ["dominant_topic"]
    with open(args.doc_topics_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        # iterate over all docs in corpus
        for i, bow in enumerate(tqdm(corpus, desc="Doc-topic export", unit="docs", total=num_docs)):
            dist = lda.get_document_topics(bow, minimum_probability=0.0)
            # start row with zeros
            row = [0.0] * args.k
            dom = -1
            max_prob = -1.0
            for (t, p) in dist:
                row[t] = float(p)
                if p > max_prob:
                    max_prob = p
                    dom = t
            writer.writerow([i] + row + [dom])
    print(f"Document-topic weights saved to {args.doc_topics_file}")

    # ----------------- Visualizations -----------------
    print("Saving plots…")
    # Topic size bar
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(args.k), sizes_est)
    plt.xlabel("Topic")
    plt.ylabel("Estimated docs")
    plt.title("Topic Sizes (scaled from train)")
    plt.tight_layout()
    plt.savefig("outputs/topic_sizes.png", dpi=150)
    plt.close()

    # JSD heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(js_mat, interpolation='nearest')
    plt.title("Inter-topic Distance (Jensen–Shannon)")
    plt.xlabel("Topic")
    plt.ylabel("Topic")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("outputs/intertopic_js_heatmap.png", dpi=150)
    plt.close()

    # Dendrogram (1 - cosine)
    dist = 1 - cos_mat
    np.fill_diagonal(dist, 0.0)
    iu = np.triu_indices_from(dist, k=1)
    Z = linkage(dist[iu], method='average')
    plt.figure(figsize=(10, 4))
    dendrogram(Z, labels=[f"T{k}" for k in range(args.k)])
    plt.title("Topic Hierarchy (1 - cosine)")
    plt.tight_layout()
    plt.savefig("outputs/topic_dendrogram.png", dpi=150)
    plt.close()

    # 2D doc scatter
    scatter_n = min(args.scatter_sample, len(corpus_train))
    if scatter_n > 0:
        sel = np.random.choice(len(corpus_train), scatter_n, replace=False)
        DT = np.zeros((scatter_n, args.k))
        dom_colors = np.zeros(scatter_n, dtype=int)
        for i_sel, idx in enumerate(tqdm(sel, desc="Scatter sample", unit="docs")):
            dist_i = lda.get_document_topics(corpus_train[idx], minimum_probability=0.0)
            for t, p in dist_i:
                DT[i_sel, t] = p
            dom_colors[i_sel] = max(dist_i, key=lambda x: x[1])[0] if dist_i else -1
        if HAS_UMAP:
            emb = umap.UMAP(random_state=args.seed).fit_transform(DT)
            title, outp = "Document Embedding (UMAP of doc-topic, sample)", "outputs/doc_scatter_umap.png"
        else:
            emb = PCA(n_components=2, random_state=args.seed).fit_transform(DT)
            title, outp = "Document Embedding (PCA of doc-topic, sample)", "outputs/doc_scatter_pca.png"
        plt.figure(figsize=(8, 6))
        plt.scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.7, c=dom_colors)
        plt.title(title)
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.tight_layout()
        plt.savefig(outp, dpi=150)
        plt.close()

    # ----------------- pyLDAvis (sample) -----------------
    if args.pyldavis_sample > 0:
        try:
            import pyLDAvis
            import pyLDAvis.gensim_models as gensimvis
            print("Preparing pyLDAvis sample…")
            sel = np.random.choice(len(corpus_train), min(args.pyldavis_sample, len(corpus_train)), replace=False)
            corpus_py = [corpus_train[i] for i in tqdm(sel, desc="pyLDAvis sample", unit="docs")]
            vis = gensimvis.prepare(lda, corpus_py, dictionary)
            pyLDAvis.save_html(vis, "outputs/pyldavis_sample.html")
        except Exception as e:
            print(f"pyLDAvis failed to generate: {e}")

    print("Artifacts saved in outputs/")


if __name__ == "__main__":
    main()