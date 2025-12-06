"""
keybert_enrichment.py
=====================

This module enriches cTF‑IDF keywords with additional keyphrases extracted
using KeyBERT.  KeyBERT is a minimal and easy‑to‑use keyword extraction
technique that leverages BERT embeddings and cosine similarity to find
sub‑phrases in a document that are most similar to the document itself【512879448139413†L56-L73】.
It operates by computing a document embedding, embedding candidate n‑gram
phrases and selecting those phrases whose embeddings are closest to the
document embedding【512879448139413†L69-L73】.

If the ``keybert`` package is unavailable, this module falls back to a simple
frequency‑based bigram extractor.  The fallback counts word and bigram
occurrences within each cluster and selects the most frequent n‑grams as
keywords.  Confidence scores are computed by normalising frequencies.

The `extract_keyphrases` function takes a list of documents and cluster
labels and returns a mapping from cluster label to a list of (phrase,
score) tuples.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

try:
    from keybert import KeyBERT  # type: ignore
    _KEYBERT_AVAILABLE = True
except ImportError:
    _KEYBERT_AVAILABLE = False


def _extract_keybert(documents: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """Extract keyphrases from a list of documents using KeyBERT.

    Parameters
    ----------
    documents : list[str]
        Concatenated documents belonging to a single cluster.
    top_n : int
        Number of keyphrases to return.

    Returns
    -------
    list of (str, float)
        Keyphrase and associated similarity score.  Higher scores indicate
        greater relevance.
    """
    if not _KEYBERT_AVAILABLE:
        raise RuntimeError("KeyBERT is not installed.")
    kw_model = KeyBERT()
    # Join cluster texts into a single document
    joined_text = " ".join(documents)
    keywords = kw_model.extract_keywords(
        joined_text,
        keyphrase_ngram_range=(1, 2),
        top_n=top_n,
        stop_words="english",
    )
    return [(phrase, float(score)) for phrase, score in keywords]


def _extract_fallback(documents: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """Fallback extraction using n‑gram frequency counts.

    This function computes unigrams and bigrams across the provided documents
    using a ``CountVectorizer`` and then selects the most frequent n‑grams.
    Confidence scores are normalised frequencies.
    """
    vectorizer = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(documents)
    # Sum counts across documents
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    # Compute frequencies
    freq_dict = {phrase: count for phrase, count in zip(vocab, counts)}
    # Select top_n phrases
    top_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    total = sum(count for _, count in top_items) or 1.0
    # Normalise counts as simple scores
    return [(phrase, count / total) for phrase, count in top_items]


def extract_keyphrases(documents: List[str], labels, top_n: int = 5) -> Dict[int, List[Tuple[str, float]]]:
    """Extract keyphrases for each cluster of documents.

    Parameters
    ----------
    documents : list[str]
        Original review texts.
    labels : array-like
        Cluster labels for each document.  Noise points (label ``-1``) are
        ignored.
    top_n : int, optional
        Number of keyphrases to extract per cluster.  Defaults to 5.

    Returns
    -------
    dict
        Mapping from cluster label to list of (keyphrase, score) pairs.
    """
    cluster_docs: Dict[int, List[str]] = defaultdict(list)
    for doc, label in zip(documents, labels):
        if label == -1:
            continue
        cluster_docs[int(label)].append(doc)
    results: Dict[int, List[Tuple[str, float]]] = {}
    for cls, docs in cluster_docs.items():
        try:
            if _KEYBERT_AVAILABLE:
                phrases = _extract_keybert(docs, top_n=top_n)
            else:
                phrases = _extract_fallback(docs, top_n=top_n)
        except Exception as e:
            logger.warning(
                "Failed to extract keyphrases for cluster %s using KeyBERT: %s; falling back.",
                cls,
                str(e),
            )
            phrases = _extract_fallback(docs, top_n=top_n)
        results[cls] = phrases
    return results
