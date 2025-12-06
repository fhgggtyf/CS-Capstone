"""
ctfidf.py
==========

This module implements the class‑based TF‑IDF (cTF‑IDF) computation used for
topic modeling.  Unlike standard TF‑IDF where each document is treated as
independent, cTF‑IDF merges all documents belonging to a single cluster
(class) into one long string and then computes the inverse document
frequency over classes instead of individual documents.  This emphasises
words that are unique to a particular cluster and down‑weights words that
appear in many clusters.  The idea, described by Maarten Grootendorst, is to
“supply all documents within a single class with the same class vector” by
joining them together and using the number of classes in the IDF
computation【178967755416880†L97-L123】.

The `compute_ctfidf` function accepts raw documents and their cluster labels
and returns the top keywords per cluster according to cTF‑IDF scores.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


@dataclass
class CTfidfResult:
    """Container for cTF‑IDF results.

    Attributes
    ----------
    keywords_per_class : Dict[int, List[Tuple[str, float]]]
        Mapping of cluster label to a list of (keyword, weight) pairs sorted in
        descending order by weight.  Noise points (label ``-1``) are omitted.
    vectorizer : CountVectorizer
        The fitted count vectorizer used to build the vocabulary.
    ctfidf_matrix : sp.csr_matrix
        The computed cTF‑IDF matrix with one row per cluster and one
        column per vocabulary term.
    """

    keywords_per_class: Dict[int, List[Tuple[str, float]]]
    vectorizer: CountVectorizer
    ctfidf_matrix: sp.csr_matrix


def _ctfidf_transform(count: sp.csr_matrix, n_samples: int) -> sp.csr_matrix:
    """Transform a count matrix into cTF‑IDF.

    Based on the formula described by Grootendorst for cTF‑IDF, the IDF is
    computed as ``log(n_samples / df)`` where ``n_samples`` is the total
    number of original documents (not the number of classes) and ``df`` is
    the document frequency per term across classes【178967755416880†L97-L123】.
    After applying the IDF weighting the rows are L1‑normalised.
    """
    # Compute document frequencies across columns
    df = np.squeeze(np.asarray(count.sum(axis=0)))
    idf = np.log((n_samples + 1) / (df + 1)) + 1
    idf_diag = sp.diags(idf, offsets=0, shape=(len(idf), len(idf)), format="csr")
    # Apply IDF weighting
    tf = count * idf_diag
    # L1 normalisation over rows
    tf = normalize(tf, norm="l1", axis=1)
    return tf


def compute_ctfidf(documents: List[str], labels: np.ndarray, top_n: int = 20) -> CTfidfResult:
    """Compute cTF‑IDF for clustered documents.

    Parameters
    ----------
    documents : list[str]
        Original review texts.
    labels : np.ndarray
        Cluster labels for each document.  Noise points (``-1``) are ignored.
    top_n : int, optional
        Number of top keywords to extract per cluster.  Defaults to 20.

    Returns
    -------
    CTfidfResult
        Dataclass containing keywords per class, the fitted vectoriser and
        the cTF‑IDF matrix.
    """
    # Group documents by cluster label
    cluster_docs: Dict[int, List[str]] = defaultdict(list)
    for doc, label in zip(documents, labels):
        if label == -1:
            continue  # ignore noise
        cluster_docs[int(label)].append(doc)
    if not cluster_docs:
        return CTfidfResult(keywords_per_class={}, vectorizer=CountVectorizer(), ctfidf_matrix=sp.csr_matrix((0, 0)))
    # Concatenate documents in each cluster
    joined_docs: List[str] = []
    class_labels: List[int] = []
    for cls, docs in cluster_docs.items():
        joined_docs.append(" ".join(docs))
        class_labels.append(cls)
    # Vectorise joined documents into raw count matrix
    vectorizer = CountVectorizer(stop_words="english")
    count_matrix = vectorizer.fit_transform(joined_docs)
    # n_samples is the total number of original documents, not the number of classes
    n_samples = len(documents)
    ctfidf = _ctfidf_transform(count_matrix, n_samples)
    # Extract top keywords per class
    vocab = np.array(vectorizer.get_feature_names_out())
    keywords_per_class: Dict[int, List[Tuple[str, float]]] = {}
    for idx, cls in enumerate(class_labels):
        row = ctfidf.getrow(idx).toarray().flatten()
        if row.sum() == 0:
            continue
        top_indices = row.argsort()[-top_n:][::-1]
        keywords = [(vocab[i], float(row[i])) for i in top_indices]
        keywords_per_class[cls] = keywords
    return CTfidfResult(
        keywords_per_class=keywords_per_class,
        vectorizer=vectorizer,
        ctfidf_matrix=ctfidf,
    )
