"""
clustering.py
==============

This module performs unsupervised clustering on dense text embeddings.

Preferred algorithm
-------------------
The primary algorithm is HDBSCAN (Hierarchical Density-Based Spatial
Clustering of Applications with Noise). HDBSCAN is well suited for
semantic embeddings because it can:

* discover clusters of varying density
* mark ambiguous points as noise (label ``-1``)
* expose soft cluster membership via the ``probabilities_`` attribute.

When HDBSCAN is not available, the module falls back to
``sklearn.cluster.DBSCAN`` with a small grid-search over ``eps`` and
``min_samples``.

Dimensionality reduction
------------------------
For high-dimensional embeddings (e.g. MiniLM with 384 dimensions) this
module *always* reduces the data with PCA to at most 100 dimensions
before clustering. This dramatically speeds up density-based algorithms
without materially harming cluster quality for topic modelling use
cases.

Progress & logging
------------------
Clustering can be long-running on large datasets, so this module adds:

* a heartbeat logger that periodically reports that clustering is
  still in progress;
* a ``tqdm`` progress bar when auto-tuning over multiple hyper-parameter
  configurations.

The main entry point is :func:`cluster_embeddings`, which returns a
:class:`ClusteringResult` dataclass.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances  # kept for compatibility if needed elsewhere
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _HDBSCAN_AVAILABLE = False


@dataclass
class ClusteringResult:
    """Container for clustering results.

    Attributes
    ----------
    labels :
        Integer cluster labels for each sample. Noise points are labelled
        ``-1``.
    probabilities :
        Probability of membership for each sample. When HDBSCAN is not
        available, a uniform probability of 1.0 is returned for points
        assigned to a cluster and 0.0 for noise points.
    cluster_sizes :
        Mapping from cluster label to the number of samples in that
        cluster (excluding noise).
    """

    labels: np.ndarray
    probabilities: np.ndarray
    cluster_sizes: Dict[int, int]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _start_heartbeat(name: str, interval: float = 30.0) -> threading.Event:
    """Start a lightweight "heartbeat" thread that logs periodically.

    Parameters
    ----------
    name :
        Human-readable label for the long-running task.
    interval :
        Number of seconds between log messages.

    Returns
    -------
    threading.Event
        An event that can be set to stop the heartbeat.
    """
    stop_event = threading.Event()

    def _beat() -> None:
        while not stop_event.wait(interval):
            logger.info("%s still running…", name)

    thread = threading.Thread(target=_beat, name=f"heartbeat-{name}", daemon=True)
    thread.start()
    return stop_event


def _compute_cluster_sizes(labels: np.ndarray) -> Dict[int, int]:
    """Return a dictionary mapping cluster label → size, excluding noise."""
    sizes: Dict[int, int] = {}
    for label in np.unique(labels):
        if label == -1:
            continue
        sizes[int(label)] = int(np.sum(labels == label))
    return sizes


# ---------------------------------------------------------------------------
# HDBSCAN auto-tuning
# ---------------------------------------------------------------------------

def _auto_tune_hdbscan(
    embeddings: np.ndarray,
    min_cluster_sizes: Iterable[int],
    min_samples_values: Iterable[Optional[int]],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Try several HDBSCAN configurations and keep the best one.

    "Best" is defined as the configuration with the largest number of
    non-noise points (i.e. points with label != -1).
    """
    if not _HDBSCAN_AVAILABLE:
        raise RuntimeError("HDBSCAN is not available but _auto_tune_hdbscan was called.")

    min_cluster_sizes = list(min_cluster_sizes)
    min_samples_values = list(min_samples_values)
    configs = list(product(min_cluster_sizes, min_samples_values))

    logger.info(
        "Auto-tuning HDBSCAN over %d configurations "
        "(min_cluster_size ∈ %s, min_samples ∈ %s)",
        len(configs),
        min_cluster_sizes,
        min_samples_values,
    )

    best_labels: Optional[np.ndarray] = None
    best_probs: Optional[np.ndarray] = None
    best_cluster_sizes: Dict[int, int] = {}
    max_non_noise = -1

    for min_cluster_size, min_samples in tqdm(
        configs,
        desc="HDBSCAN configs",
        unit="config",
    ):
        logger.debug(
            "Running HDBSCAN with min_cluster_size=%d, min_samples=%s",
            min_cluster_size,
            str(min_samples),
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            prediction_data=False,
            core_dist_n_jobs=-1,
            approx_min_span_tree=True,
        )
        labels = clusterer.fit_predict(embeddings)
        non_noise = int(np.sum(labels != -1))
        logger.debug(
            "HDBSCAN config min_cluster_size=%d, min_samples=%s produced %d non-noise points",
            min_cluster_size,
            str(min_samples),
            non_noise,
        )
        if non_noise > max_non_noise:
            max_non_noise = non_noise
            best_labels = labels
            best_probs = getattr(
                clusterer,
                "probabilities_",
                np.ones_like(labels, dtype=float),
            )
            best_cluster_sizes = _compute_cluster_sizes(labels)

    assert best_labels is not None and best_probs is not None  # for type-checkers
    return best_labels, best_probs, best_cluster_sizes


# ---------------------------------------------------------------------------
# DBSCAN auto-tuning (fallback)
# ---------------------------------------------------------------------------

def _auto_tune_dbscan(
    embeddings: np.ndarray,
    eps_values: Iterable[float],
    min_samples_values: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Fallback tuning for DBSCAN when HDBSCAN is unavailable.

    DBSCAN is tuned over a range of ``eps`` (neighbourhood radius) and
    ``min_samples`` values. The configuration producing the maximum
    number of non-noise points is selected.
    """
    eps_values = list(eps_values)
    min_samples_values = list(min_samples_values)
    configs = list(product(eps_values, min_samples_values))

    logger.info(
        "Auto-tuning DBSCAN over %d configurations "
        "(eps ∈ %s, min_samples ∈ %s)",
        len(configs),
        eps_values,
        min_samples_values,
    )

    best_labels: Optional[np.ndarray] = None
    best_probs: Optional[np.ndarray] = None
    best_cluster_sizes: Dict[int, int] = {}
    max_non_noise = -1

    for eps, min_samples in tqdm(
        configs,
        desc="DBSCAN configs",
        unit="config",
    ):
        logger.debug(
            "Running DBSCAN with eps=%.3f, min_samples=%d",
            eps,
            min_samples,
        )
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = clusterer.fit_predict(embeddings)
        non_noise = int(np.sum(labels != -1))
        logger.debug(
            "DBSCAN config eps=%.3f, min_samples=%d produced %d non-noise points",
            eps,
            min_samples,
            non_noise,
        )
        if non_noise > max_non_noise:
            max_non_noise = non_noise
            best_labels = labels
            # DBSCAN has no soft probabilities; use 1.0 for clustered, 0.0 for noise.
            best_probs = np.where(labels != -1, 1.0, 0.0)
            best_cluster_sizes = _compute_cluster_sizes(labels)

    assert best_labels is not None and best_probs is not None
    return best_labels, best_probs, best_cluster_sizes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
) -> ClusteringResult:
    """Cluster dense embeddings using HDBSCAN (preferred) or DBSCAN.

    The function automatically tunes hyper-parameters over a reasonable
    grid when no explicit parameters are supplied. It always performs PCA
    dimensionality reduction to at most 100 components for efficiency on
    large, high-dimensional datasets.

    Parameters
    ----------
    embeddings :
        Embedding matrix of shape ``(n_samples, n_features)``.
    min_cluster_size :
        Optional explicit ``min_cluster_size`` for HDBSCAN. If provided
        (with or without ``min_samples``), a single HDBSCAN run is
        performed instead of auto-tuning. Ignored when HDBSCAN is not
        available.
    min_samples :
        Optional explicit ``min_samples`` for HDBSCAN. See above.

    Returns
    -------
    ClusteringResult
        Dataclass containing labels, probabilities, and a cluster size
        dictionary.
    """
    if embeddings.size == 0:
        raise ValueError("Empty embedding matrix passed to clustering.")

    n_samples, n_features = embeddings.shape
    algo_name = "HDBSCAN" if _HDBSCAN_AVAILABLE else "DBSCAN"
    logger.info(
        "Starting clustering using %s on %d samples with %d-dim embeddings",
        algo_name,
        n_samples,
        n_features,
    )

    # ------------------------------------------------------------------
    # PCA dimensionality reduction (always ON for efficiency)
    # ------------------------------------------------------------------
    target_dim = min(100, n_features)
    if n_features > target_dim:
        logger.info(
            "Reducing dimensionality with PCA from %d → %d components "
            "before clustering",
            n_features,
            target_dim,
        )
        t0 = time.time()
        pca = PCA(n_components=target_dim, random_state=42)
        reduced = pca.fit_transform(embeddings)
        logger.info("PCA completed in %.2f seconds", time.time() - t0)
    else:
        logger.info("Skipping PCA: embedding dimension (%d) ≤ target (%d)", n_features, target_dim)
        reduced = embeddings

    # ------------------------------------------------------------------
    # Perform clustering with heartbeat logging
    # ------------------------------------------------------------------
    labels: np.ndarray
    probabilities: np.ndarray
    cluster_sizes: Dict[int, int]

    if _HDBSCAN_AVAILABLE:
        heartbeat = _start_heartbeat("HDBSCAN clustering")
        try:
            if min_cluster_size is not None or min_samples is not None:
                # Single HDBSCAN configuration
                mcs = min_cluster_size or max(5, n_samples // 100)
                logger.info(
                    "Running single HDBSCAN configuration: "
                    "min_cluster_size=%d, min_samples=%s",
                    mcs,
                    str(min_samples),
                )
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=mcs,
                    min_samples=min_samples,
                    metric="euclidean",
                    prediction_data=False,
                    core_dist_n_jobs=-1,
                    approx_min_span_tree=True,
                )
                labels = clusterer.fit_predict(reduced)
                probabilities = getattr(
                    clusterer,
                    "probabilities_",
                    np.ones_like(labels, dtype=float),
                )
                cluster_sizes = _compute_cluster_sizes(labels)
            else:
                # Auto-tune across a small grid of reasonable values
                candidate_min_cluster_sizes = [
                    max(5, n_samples // 500),
                    max(5, n_samples // 200),
                    max(5, n_samples // 100),
                ]
                candidate_min_samples: Iterable[Optional[int]] = [None, 5, 10]
                labels, probabilities, cluster_sizes = _auto_tune_hdbscan(
                    reduced,
                    candidate_min_cluster_sizes,
                    candidate_min_samples,
                )
        finally:
            heartbeat.set()
    else:
        # DBSCAN fallback with heartbeat
        heartbeat = _start_heartbeat("DBSCAN clustering")
        try:
            eps_values = [0.5, 0.3, 0.8]
            min_samples_values = [5, 10, 3]
            labels, probabilities, cluster_sizes = _auto_tune_dbscan(
                reduced,
                eps_values,
                min_samples_values,
            )
        finally:
            heartbeat.set()

    n_clustered = int(np.sum(labels != -1))
    n_noise = int(np.sum(labels == -1))
    logger.info(
        "Clustering complete: %d samples assigned to %d clusters "
        "(%d noise points)",
        n_clustered,
        len(cluster_sizes),
        n_noise,
    )

    return ClusteringResult(
        labels=labels,
        probabilities=probabilities,
        cluster_sizes=cluster_sizes,
    )
