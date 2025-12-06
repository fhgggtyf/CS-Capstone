"""
embeddings.py
================

This module contains utilities to compute dense vector representations (embeddings)
for collections of short game reviews.  The goal of these embeddings is to
capture the semantic meaning of each review so that similar complaints are
positioned close together in vector space.  These embeddings form the basis
for clustering in later stages of the pipeline.

Two modes are supported:

1. **OpenAI API** – The `text‑embedding‑3‑large` model can be used to produce
   high‑quality 3072‑dimensional vectors.  According to OpenAI, this model
   creates embeddings with up to 3072 dimensions and is their best performing
   embedding model【274848708673263†L170-L176】.  Embeddings are generated via the
   OpenAI API and require an API key (set in the ``OPENAI_API_KEY`` environment
   variable).  Pricing is per token; as of January 2024 the cost is
   \$0.00013 per thousand tokens【274848708673263†L206-L209】.
2. **Local approximation** – When no API key is provided or a local-only run
   is preferred, this module can fall back to using a TF‑IDF + TruncatedSVD
   pipeline to approximate semantic embeddings.  While far less expressive than
   transformer-based models, this approach runs without internet access and
   requires only scikit‑learn.

The resulting embedding matrix is always shaped ``(n_reviews, embedding_dim)``.
Embeddings can optionally be saved to disk as a NumPy ``.npy`` file.

Usage:

>>> from pipeline.embeddings import compute_embeddings
>>> embeddings = compute_embeddings(texts, model_type="openai", batch_size=512)

If the OpenAI API is unavailable, set ``model_type`` to ``"minilm"`` to
compute an approximate embedding matrix using TF‑IDF and dimensionality
reduction.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# Try to import sentence-transformers for local MiniLM embeddings.
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _SENTENCE_TRANS_AVAILABLE = True
except ImportError:
    # If sentence_transformers isn't available, we'll fall back to TF–IDF + SVD.
    _SENTENCE_TRANS_AVAILABLE = False

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

import warnings


logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Simple container for embedding results.

    Attributes
    ----------
    embeddings : np.ndarray
        The dense embedding matrix with shape `(n_samples, embedding_dim)`.
    model_type : str
        Name of the embedding model used.
    dimensions : int
        Number of columns in the embedding matrix.
    """

    embeddings: np.ndarray
    model_type: str
    dimensions: int


def _batch(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Yield successive batches from an iterable of texts.

    Parameters
    ----------
    iterable : Iterable[str]
        A collection of texts to chunk into batches.
    batch_size : int
        Maximum number of items per batch.

    Yields
    ------
    list[str]
        Lists containing up to ``batch_size`` texts.
    """
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _compute_openai_embeddings(texts: List[str], batch_size: int = 512,
                               dimensions: Optional[int] = None,
                               max_tokens_per_minute: Optional[int] = None) -> np.ndarray:
    """Compute embeddings via the OpenAI API.

    This function calls the OpenAI ``embeddings`` endpoint and returns a numpy
    array of shape ``(n_texts, embedding_dim)``.  It supports batching to
    maximise throughput and gracefully handles rate limits by waiting between
    batches if a `max_tokens_per_minute` limit is provided.

    The OpenAI documentation notes that the new large text embedding model
    creates embeddings with up to 3072 dimensions and can be shortened by
    specifying the ``dimensions`` parameter【274848708673263†L170-L233】.  Passing
    a smaller ``dimensions`` will truncate the embeddings server‑side, reducing
    memory usage and disk storage at the cost of a small performance penalty.

    Parameters
    ----------
    texts : list[str]
        List of input strings to embed.
    batch_size : int, optional
        Number of texts to send per API call.  Larger batches result in fewer
        network round trips but should not exceed the model's maximum token
        limits.  Defaults to 512.
    dimensions : int or None, optional
        Desired dimensionality of the returned vectors.  If ``None`` then the
        full 3072‑dimensional embeddings will be returned.  Providing a value
        between 256 and 3072 will instruct OpenAI to shorten embeddings as
        described in the API guide【274848708673263†L220-L233】.  Defaults to None.
    max_tokens_per_minute : int or None, optional
        Rate limit to respect when making repeated API calls.  If provided,
        batches will be spaced such that at most this many tokens are sent per
        minute.  Defaults to ``None``, meaning no rate limiting is applied.

    Returns
    -------
    np.ndarray
        Embedding matrix.
    """
    try:
        import openai  # type: ignore
    except ImportError:
        raise ImportError(
            "openai package is required for API embeddings. Install with `pip install openai`."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. Please set your OpenAI API key."
        )
    openai.api_key = api_key

    all_embeddings: List[np.ndarray] = []
    for batch in _batch(texts, batch_size):
        # Respect optional rate limits
        if max_tokens_per_minute:
            # Rough estimate: average 4 tokens per word; multiply by number of words
            total_tokens = sum(len(t.split()) for t in batch) * 4
            seconds_per_minute = 60.0
            wait_time = total_tokens / max_tokens_per_minute * seconds_per_minute
            if wait_time > 0:
                logger.info(
                    "Rate limiting: sleeping for %.2f seconds before next API call", wait_time
                )
                import time
                time.sleep(wait_time)
        response = openai.Embedding.create(
            input=batch,
            model="text-embedding-3-large",
            dimensions=dimensions,
        )
        batch_embeddings = np.array([record["embedding"] for record in response["data"]], dtype=float)
        all_embeddings.append(batch_embeddings)
        logger.info("Processed batch of %d texts via OpenAI API.", len(batch))
    embeddings = np.vstack(all_embeddings)
    return embeddings


def _compute_minilm_embeddings(texts: List[str], n_components: int = 300,
                               max_features: int = 5000) -> np.ndarray:
    """Compute simple approximate embeddings using TF‑IDF and SVD.

    This function serves as a fallback when no external embedding model is
    available.  It vectorises texts using scikit‑learn's ``TfidfVectorizer`` and
    then reduces dimensionality with ``TruncatedSVD``.  While not as powerful
    as transformer-based models, it preserves important information about
    vocabulary and co‑occurrence patterns.

    Parameters
    ----------
    texts : list[str]
        Input documents.
    n_components : int, optional
        Number of SVD components to retain.  Defaults to 300.
    max_features : int, optional
        Maximum number of TF‑IDF features.  Defaults to 5000.

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(n_texts, n_components)``.
    """
    logger.info(
        "Computing local TF‑IDF + SVD embeddings (n_components=%d, max_features=%d)",
        n_components,
        max_features,
    )
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    pipeline: Pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("svd", svd),
    ])
    # Fit the vectoriser separately to determine vocabulary size for SVD
    X = vectorizer.fit_transform(texts)
    n_features = X.shape[1]
    if n_components >= n_features:
        # TruncatedSVD requires n_components < n_features; adjust if necessary
        adjusted_components = max(1, n_features - 1)
        logger.warning(
            "Requested %d SVD components, but only %d features are available; reducing n_components to %d.",
            n_components,
            n_features,
            adjusted_components,
        )
        svd = TruncatedSVD(n_components=adjusted_components, random_state=42)
        pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("svd", svd),
        ])
    embeddings = pipeline.fit_transform(texts)
    return embeddings


def _compute_sentence_transformer_embeddings(
    texts: List[str], model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128
) -> np.ndarray:
    """Compute embeddings using a pretrained sentence transformer model.

    This helper loads a Hugging Face ``sentence-transformers`` model to
    generate dense semantic embeddings for each text.  The default model
    ``all-MiniLM-L6-v2`` produces 384‑dimensional vectors and offers a
    compelling balance between speed and quality for local usage.  If
    ``sentence-transformers`` is not installed, this function will raise an
    ``ImportError`` and callers should fall back to the TF–IDF/SVD
    implementation.

    Parameters
    ----------
    texts : list[str]
        Input documents to embed.
    model_name : str, optional
        Name of the pretrained model to load.  Defaults to
        ``"all-MiniLM-L6-v2"``.
    batch_size : int, optional
        Number of texts to encode per forward pass.  Adjust based on
        available memory.  Defaults to 128.

    Returns
    -------
    np.ndarray
        A matrix of shape ``(n_texts, embedding_dim)`` containing the
        embeddings.

    Raises
    ------
    ImportError
        If ``sentence-transformers`` is unavailable.
    """
    if not _SENTENCE_TRANS_AVAILABLE:
        raise ImportError(
            "sentence-transformers is not installed. Install with `pip install sentence-transformers`"
        )
    logger.info("Computing embeddings using sentence-transformer model %s", model_name)
    model = SentenceTransformer(model_name)
    # Encode texts in batches. The model returns a list which we convert to numpy array.
    embeddings_list: List[np.ndarray] = []
    for batch in _batch(texts, batch_size):
        embeddings_batch = model.encode(batch, batch_size=len(batch), show_progress_bar=False)
        embeddings_list.append(np.asarray(embeddings_batch))
    embeddings = np.vstack(embeddings_list)
    return embeddings


def compute_embeddings(texts: List[str], model_type: str = "openai",
                       batch_size: int = 512, **kwargs) -> EmbeddingResult:
    """Compute dense vector representations for a list of texts.

    Parameters
    ----------
    texts : list[str]
        Input texts (reviews).  Each element should be a string containing a
        player review.
    model_type : {'openai', 'minilm'}, optional
        Type of embedding model to use.  ``'openai'`` will call the OpenAI
        `text‑embedding‑3‑large` endpoint, while ``'minilm'`` will compute a
        local TF‑IDF + SVD approximation.  Defaults to ``'openai'``.
    batch_size : int, optional
        Batch size for the OpenAI API.  Ignored when ``model_type='minilm'``.
    **kwargs
        Additional keyword arguments passed to the underlying embedding
        functions.  For OpenAI, see ``_compute_openai_embeddings``.  For
        local embeddings, you may override ``n_components`` or ``max_features``.

    Returns
    -------
    EmbeddingResult
        A dataclass containing the embedding matrix and metadata.

    Notes
    -----
    If you provide an ``embeddings_file`` keyword argument, the embeddings will
    be saved to that path as a binary ``.npy`` file after computation.
    """
    logger.info("Starting embedding computation using model: %s", model_type)
    if model_type == "openai":
        dimensions = kwargs.get("dimensions")
        embeddings = _compute_openai_embeddings(
            texts,
            batch_size=batch_size,
            dimensions=dimensions,
            max_tokens_per_minute=kwargs.get("max_tokens_per_minute"),
        )
    elif model_type == "minilm":
        # Attempt to use sentence-transformer MiniLM model if available.  If
        # ``sentence-transformers`` is not installed, fall back to the
        # TF–IDF/SVD approximation.  The caller may override the model
        # name via ``model_name`` and batch size via ``st_batch_size``.
        if _SENTENCE_TRANS_AVAILABLE:
            model_name = kwargs.get("model_name", "all-MiniLM-L6-v2")
            st_batch_size = kwargs.get("st_batch_size", 128)
            try:
                embeddings = _compute_sentence_transformer_embeddings(
                    texts, model_name=model_name, batch_size=st_batch_size
                )
                dimensions = embeddings.shape[1]
            except Exception as e:
                # If any unexpected error occurs during sentence-transformer
                # embedding computation, issue a warning and fall back.
                warnings.warn(
                    f"Falling back to TF–IDF/SVD embeddings due to error: {e}"
                )
                n_components = kwargs.get("n_components", 300)
                max_features = kwargs.get("max_features", 5000)
                embeddings = _compute_minilm_embeddings(
                    texts, n_components=n_components, max_features=max_features
                )
                dimensions = embeddings.shape[1]
        else:
            warnings.warn(
                "sentence-transformers not available; using TF–IDF/SVD embeddings instead."
            )
            n_components = kwargs.get("n_components", 300)
            max_features = kwargs.get("max_features", 5000)
            embeddings = _compute_minilm_embeddings(
                texts, n_components=n_components, max_features=max_features
            )
            dimensions = embeddings.shape[1]
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Expected 'openai' or 'minilm'."
        )
    if model_type == "openai":
        dimensions = embeddings.shape[1]

    embeddings_file: Optional[str] = kwargs.get("embeddings_file")
    if embeddings_file:
        np.save(embeddings_file, embeddings)
        logger.info("Saved embeddings to %s", embeddings_file)

    return EmbeddingResult(embeddings=embeddings, model_type=model_type, dimensions=dimensions)
