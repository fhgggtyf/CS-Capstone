"""
validation.py
=============

This module implements evaluation and validation routines for the topic
clustering pipeline.  Robust evaluation is critical in academic settings
because it demonstrates that the discovered topics are meaningful and
reproducible.  The metrics implemented here include:

* **Silhouette score** – Measures how similar a document is to its own
  cluster compared to other clusters.  Scores range from −1 to 1; higher
  scores indicate more distinct clusters.
* **Davies–Bouldin index** – Ratio of within‑cluster scatter to
  between‑cluster separation; lower values indicate better clustering.
* **Outlier percentage** – Proportion of reviews labelled as noise.
* **Intruder word test** – For each topic, insert a random keyword from
  another topic into its keyword list.  Topics with high intruder
  detectability are more coherent.  Here we approximate intruder
  detectability by computing the difference between the average cTF‑IDF
  weights of genuine keywords and the intruder keyword.

This module does not implement a full topic stability analysis across
random seeds due to computational cost, but the API is designed so that
additional validation metrics can be added easily.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute basic clustering metrics.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.
    labels : np.ndarray
        Cluster labels (−1 for noise).

    Returns
    -------
    dict
        Dictionary containing silhouette score, Davies–Bouldin index and
        outlier percentage.  If there are fewer than two clusters or all
        points are in one cluster, silhouette and Davies–Bouldin scores are
        undefined (returned as ``nan``).
    """
    # Remove noise points for silhouette/DB index computation
    mask = labels != -1
    unique_clusters = set(labels[mask])
    metrics = {}
    if len(unique_clusters) <= 1:
        metrics["silhouette_score"] = float("nan")
        metrics["davies_bouldin"] = float("nan")
    else:
        metrics["silhouette_score"] = float(silhouette_score(embeddings[mask], labels[mask], metric="euclidean"))
        metrics["davies_bouldin"] = float(davies_bouldin_score(embeddings[mask], labels[mask]))
    # Outlier percentage
    outlier_pct = float(np.sum(labels == -1) / len(labels)) if len(labels) > 0 else float("nan")
    metrics["outlier_percentage"] = outlier_pct
    return metrics


def intruder_word_test(keywords: Dict[int, List[Tuple[str, float]]], num_trials: int = 5) -> Dict[int, float]:
    """Approximate topic coherence via intruder word tests.

    For each topic, we randomly insert a keyword from another topic into its
    top keywords list and compare the average cTF‑IDF weights of the genuine
    keywords against the intruder keyword weight.  A larger difference
    indicates that the intruder stands out and the topic is more coherent.

    Parameters
    ----------
    keywords : dict
        Mapping from topic ID to list of (keyword, weight) pairs.
    num_trials : int, optional
        Number of random intruders to test per topic.  Defaults to 5.

    Returns
    -------
    dict
        Mapping from topic ID to average intruder score difference.  Higher
        values imply more coherent topics.
    """
    intruder_scores: Dict[int, float] = {}
    topic_ids = list(keywords.keys())
    if len(topic_ids) < 2:
        return {tid: float("nan") for tid in topic_ids}
    for tid, kw_list in keywords.items():
        if not kw_list:
            intruder_scores[tid] = float("nan")
            continue
        genuine_weights = [weight for _, weight in kw_list]
        # Average weight of genuine keywords
        avg_genuine = float(np.mean(genuine_weights))
        diffs = []
        for _ in range(num_trials):
            # Pick a random other topic
            other_tid = random.choice([t for t in topic_ids if t != tid])
            other_keywords = keywords[other_tid]
            if not other_keywords:
                continue
            intruder_word, intruder_weight = random.choice(other_keywords)
            # Compute difference between avg genuine weight and intruder weight
            diffs.append(avg_genuine - intruder_weight)
        intruder_scores[tid] = float(np.mean(diffs)) if diffs else float("nan")
    return intruder_scores


def topic_distribution(labels: np.ndarray) -> Dict[int, int]:
    """Compute the size of each cluster (including noise).

    Returns a dictionary mapping cluster labels to the number of documents.
    Noise is represented with label ``-1``.
    """
    return dict(Counter(int(l) for l in labels))
