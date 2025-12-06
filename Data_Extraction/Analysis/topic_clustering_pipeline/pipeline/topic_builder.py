"""
topic_builder.py
=================

This module assembles the outputs of clustering, cTF‑IDF, and keyphrase
extraction into rich topic objects.  Each topic corresponds to a cluster of
reviews and contains metadata for interpretability and downstream analysis.

Key data stored per topic:

* **topic_id** – The integer cluster label.
* **keywords** – The top cTF‑IDF keywords with weights.
* **phrases** – The top keyphrases and confidence scores (from KeyBERT or fallback).
* **cluster_size** – Number of documents in the cluster.
* **examples** – A sample of representative reviews (up to 10) from the cluster.
* **embedding** – Mean embedding vector for the cluster.
* **similarity** – Cosine similarity scores between this topic and all others.
* **hierarchy** – Optional hierarchical grouping of topics (represented as a
  dendrogram using Agglomerative Clustering).

The `build_topics` function constructs these topic objects and computes a
pairwise similarity matrix for all topics.  The optional `build_hierarchy`
function groups similar topics into broader categories using
AgglomerativeClustering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """Dataclass representing a single topic.

    Attributes
    ----------
    topic_id : int
        Cluster label for this topic.
    keywords : list[tuple[str, float]]
        List of cTF‑IDF keywords and weights.
    phrases : list[tuple[str, float]]
        List of keyphrases and scores from KeyBERT or the fallback.
    cluster_size : int
        Number of documents in the cluster.
    examples : list[str]
        Example review texts from this cluster (truncated to 10 items).
    embedding : np.ndarray
        Mean embedding vector for this cluster.
    similarity : Dict[int, float]
        Mapping of other topic IDs to cosine similarity scores.
    hierarchy_label : Optional[int]
        Label of the group assigned by hierarchical clustering (if computed).
    """
    topic_id: int
    keywords: List[tuple[str, float]]
    phrases: List[tuple[str, float]]
    cluster_size: int
    examples: List[str] = field(default_factory=list)
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    similarity: Dict[int, float] = field(default_factory=dict)
    hierarchy_label: Optional[int] = None


def build_topics(documents: List[str], labels: np.ndarray, embeddings: np.ndarray,
                 keywords: Dict[int, List[tuple[str, float]]],
                 phrases: Dict[int, List[tuple[str, float]]],
                 max_examples: int = 10,
                 compute_similarity: bool = True,
                 compute_hierarchy: bool = False,
                 hierarchy_clusters: int = 5) -> List[Topic]:
    """Construct a list of Topic objects.

    Parameters
    ----------
    documents : list[str]
        Original review texts.
    labels : np.ndarray
        Cluster labels for each document.
    embeddings : np.ndarray
        Dense embedding matrix.
    keywords : dict
        Mapping from cluster label to cTF‑IDF keyword lists.
    phrases : dict
        Mapping from cluster label to keyphrase lists.
    max_examples : int, optional
        Maximum number of representative examples to store per topic.  Defaults
        to 10.
    compute_similarity : bool, optional
        Whether to compute pairwise cosine similarity between topic embeddings.
        Defaults to True.
    compute_hierarchy : bool, optional
        Whether to perform hierarchical clustering on topic embeddings to
        organise topics into larger groups.  Defaults to False.
    hierarchy_clusters : int, optional
        Number of clusters for hierarchical grouping if ``compute_hierarchy`` is
        True.  Ignored otherwise.  Defaults to 5.

    Returns
    -------
    list[Topic]
        List of topic objects.
    """
    # Identify unique cluster labels (excluding noise)
    unique_labels = [label for label in sorted(set(labels)) if label != -1]
    if not unique_labels:
        logger.warning("No clusters to build topics from.")
        return []
    topics: List[Topic] = []
    # Precompute cluster embeddings and examples
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if len(indices) == 0:
            continue
        cluster_size = len(indices)
        # Compute mean embedding for the cluster
        cluster_embedding = embeddings[indices].mean(axis=0)
        # Choose representative examples (first n texts for reproducibility)
        sample_indices = indices[:max_examples]
        sample_texts = [documents[i] for i in sample_indices]
        topic = Topic(
            topic_id=int(label),
            keywords=keywords.get(int(label), []),
            phrases=phrases.get(int(label), []),
            cluster_size=cluster_size,
            examples=sample_texts,
            embedding=cluster_embedding,
        )
        topics.append(topic)
    # Compute similarities between topic embeddings
    if compute_similarity:
        embeddings_matrix = np.vstack([topic.embedding for topic in topics])
        similarity_matrix = cosine_similarity(embeddings_matrix)
        for i, topic in enumerate(topics):
            # Build similarity dict mapping topic_id -> similarity score
            sim_dict: Dict[int, float] = {}
            for j, other_topic in enumerate(topics):
                if i == j:
                    continue
                sim_dict[int(other_topic.topic_id)] = float(similarity_matrix[i, j])
            topic.similarity = sim_dict
    # Optionally compute hierarchical grouping
    if compute_hierarchy and len(topics) >= hierarchy_clusters:
        embeddings_matrix = np.vstack([topic.embedding for topic in topics])
        # In scikit‑learn >=1.4 the 'affinity' parameter is deprecated in favour of 'metric'
        try:
            clustering = AgglomerativeClustering(n_clusters=hierarchy_clusters, linkage="average", metric="cosine")
        except TypeError:
            # Fallback for older versions of scikit‑learn
            clustering = AgglomerativeClustering(n_clusters=hierarchy_clusters, linkage="average", affinity="cosine")
        h_labels = clustering.fit_predict(embeddings_matrix)
        for topic, h_label in zip(topics, h_labels):
            topic.hierarchy_label = int(h_label)
    return topics
