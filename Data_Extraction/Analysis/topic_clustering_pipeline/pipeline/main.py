"""
main.py
=======

Entry point for the topic clustering pipeline.  This script orchestrates
embedding computation, clustering, keyword extraction, topic construction,
optional GPT labelling, and validation.  It supports two modes of operation:

1. **step1-4** – Runs Steps 1 through 4 (embedding, clustering, cTF‑IDF,
   keyphrase extraction, and topic assembly) and then halts so that you can
   manually inspect the intermediate results.  This mode writes a JSON
   summary of the topics to disk and prints a brief overview of each topic.
2. **step5-6** – Continues from Step 4 by calling GPT to label topics and
   computing validation metrics.  This mode expects that Steps 1–4 have
   already been executed in the same session (or loads their outputs from
   disk).

The script can operate on a user‑supplied CSV containing reviews (one per
line) or generate a synthetic dataset for demonstration purposes.  See
``generate_synthetic_reviews`` for details on the synthetic data.  By
default, the script uses the local ``minilm`` embedding implementation to
avoid requiring external API keys.  You can pass ``--embedding openai``
and provide a valid ``OPENAI_API_KEY`` environment variable to use
OpenAI's ``text‑embedding‑3‑large`` model.

Example usage:

    python -m pipeline.main --mode step1-4 --dataset reviews.csv --embedding openai

The outputs will be stored in the ``outputs/`` directory created in the
current working directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .embeddings import compute_embeddings
from .clustering import cluster_embeddings
from .ctfidf import compute_ctfidf
from .keybert_enrichment import extract_keyphrases
from .topic_builder import build_topics, Topic
from .gpt_labeler import label_topics_with_gpt
from .validation import evaluate_clustering, intruder_word_test, topic_distribution

logger = logging.getLogger(__name__)


def generate_synthetic_reviews(n_per_class: int = 50) -> List[str]:
    """Generate a synthetic dataset of game reviews.

    Reviews are generated from a handful of frustration categories such as
    performance issues, difficulty, monetisation, progression, social
    interaction and narrative coherence.  Each review is a short sentence
    composed from a pool of complaint phrases.

    Parameters
    ----------
    n_per_class : int, optional
        Number of reviews to generate per category.  Defaults to 50.

    Returns
    -------
    list[str]
        List of synthetic review texts.
    """
    import random
    random.seed(42)
    categories: Dict[str, List[str]] = {
        "performance": [
            "Lag spikes make the game unplayable",
            "The frame rate drops constantly during fights",
            "Rubberbanding ruins every match",
            "Servers disconnect and cause progress loss",
            "Ping spikes mess up the experience",
        ],
        "difficulty": [
            "The boss is unfairly hard with random one‑shots",
            "Enemies have too much health and damage",
            "The difficulty curve is horribly unbalanced",
            "AI enemies cheat and read my inputs",
            "Levels are impossible without grinding",
        ],
        "monetisation": [
            "Pay‑to‑win mechanics destroy fairness",
            "Loot boxes are overpriced and predatory",
            "Microtransactions gate important content",
            "The game pushes you to buy currency packs",
            "Season passes lock essential features behind a paywall",
        ],
        "progression": [
            "Progression is too slow and grindy",
            "XP requirements are absurd for level ups",
            "Reward drops are stingy and repetitive",
            "The game forces replaying old missions for progress",
            "Upgrades take forever to unlock",
        ],
        "social": [
            "Toxic teammates ruin matches",
            "Matchmaking pairs new players with veterans",
            "Chat is full of harassment and slurs",
            "Players AFK and grief without punishment",
            "The community is hostile to newcomers",
        ],
        "narrative": [
            "The story makes no sense and is full of plot holes",
            "Characters lack depth and motivation",
            "Dialogue is cringeworthy and poorly written",
            "The ending is unsatisfying and rushed",
            "Questlines are repetitive and boring",
        ],
    }
    reviews: List[str] = []
    for cls, phrases in categories.items():
        for _ in range(n_per_class):
            review = random.choice(phrases)
            # Slightly vary wording by shuffling synonyms or adding minor noise
            noise_words = ["", "…", "!!!", "?!"]
            review = review + random.choice(noise_words)
            reviews.append(review)
    random.shuffle(reviews)
    return reviews


def save_topics_json(topics: List[Topic], output_path: Path) -> None:
    """Save topics to a JSON file for inspection.

    The JSON includes cluster ID, size, keywords, phrases, and examples.
    """
    data = []
    for topic in topics:
        data.append({
            "topic_id": topic.topic_id,
            "cluster_size": topic.cluster_size,
            "keywords": topic.keywords,
            "phrases": topic.phrases,
            "examples": topic.examples,
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d topics to %s", len(topics), output_path)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Topic clustering pipeline for game reviews")
    parser.add_argument("--mode", choices=["step1-4", "step5-6"], default="step1-4",
                        help="Execution mode: run steps 1–4 or continue with steps 5–6")
    parser.add_argument("--dataset", type=str, default="", help="Path to CSV file with a 'text' column")
    parser.add_argument(
        "--sqlite", type=str, default="", help="Path to SQLite database containing review table"
    )
    parser.add_argument(
        "--table", type=str, default="reviews", help="Name of the table in the SQLite database"
    )
    parser.add_argument(
        "--column", type=str, default="text", help="Name of the text column in the SQLite table"
    )
    parser.add_argument("--embedding", choices=["openai", "minilm"], default="minilm",
                        help="Embedding model to use")
    parser.add_argument("--output", type=str, default="outputs", help="Directory to write outputs")
    parser.add_argument("--no_gpt", action="store_true", help="Disable GPT labelling in step5-6")
    args = parser.parse_args(argv)

    # ------------------------------
    # Enable logging to both console and file
    # ------------------------------
    log_path = os.path.join(args.output, "pipeline.log")

    # Format used for both console + file handlers
    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Clear existing handlers (important if running multiple times)
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_format)
    root_logger.addHandler(file_handler)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Output file: {log_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: Load dataset or generate synthetic reviews
    if args.sqlite:
        # Load reviews from a SQLite database table
        import sqlite3

        if not os.path.isfile(args.sqlite):
            logger.error("SQLite file %s does not exist.", args.sqlite)
            sys.exit(1)
        try:
            conn = sqlite3.connect(args.sqlite)
            query = f"SELECT {args.column} FROM {args.table}"
            df_sql = pd.read_sql_query(query, conn)
            conn.close()
        except Exception as exc:
            logger.error("Failed to read from SQLite database: %s", exc)
            sys.exit(1)
        if args.column not in df_sql.columns:
            logger.error("Column '%s' not found in the table '%s'.", args.column, args.table)
            sys.exit(1)
        documents: List[str] = df_sql[args.column].astype(str).tolist()
    elif args.dataset:
        df = pd.read_csv(args.dataset)
        if "text" not in df.columns:
            logger.error("Dataset must contain a 'text' column.")
            sys.exit(1)
        documents: List[str] = df["text"].astype(str).tolist()
    else:
        logger.info("No dataset provided; generating synthetic reviews for demonstration.")
        documents = generate_synthetic_reviews(n_per_class=50)
    logger.info("Loaded %d reviews.", len(documents))

    # Step 1: Compute embeddings (if mode includes step1)
    embedding_result = compute_embeddings(documents, model_type=args.embedding)
    embeddings = embedding_result.embeddings
    logger.info("Embeddings computed: shape=%s, model=%s", embeddings.shape, embedding_result.model_type)

    # Step 2: Cluster embeddings
    clustering = cluster_embeddings(embeddings)
    labels = clustering.labels
    logger.info("Found %d clusters (excluding noise).", len(clustering.cluster_sizes))

    # Step 3: Compute cTF‑IDF keywords
    ctfidf_res = compute_ctfidf(documents, labels, top_n=20)
    keywords = ctfidf_res.keywords_per_class

    # Step 3b: Extract keyphrases
    phrases = extract_keyphrases(documents, labels, top_n=5)

    # Step 4: Build topic objects
    topics = build_topics(
        documents,
        labels,
        embeddings,
        keywords,
        phrases,
        max_examples=10,
        compute_similarity=True,
        compute_hierarchy=True,
        hierarchy_clusters=min(5, max(len(clustering.cluster_sizes), 1)),
    )
    logger.info("Constructed %d topic objects.", len(topics))

    # Save intermediate topics for manual inspection
    topics_json = output_dir / "topics_step4.json"
    save_topics_json(topics, topics_json)

    # Print a brief overview of topics
    for topic in topics:
        print(f"Topic {topic.topic_id} (size={topic.cluster_size}):")
        print("  Keywords:", ", ".join([kw for kw, _ in topic.keywords[:10]]))
        print("  Phrases:", ", ".join([ph for ph, _ in topic.phrases[:5]]))
        print("  Examples:", topic.examples[0][:80], "...\n")

    if args.mode == "step1-4":
        logger.info("Step 1–4 completed. Review the topics JSON and re‑run with --mode step5-6 for labelling and validation.")
        return

    # Step 5: GPT labelling
    if not args.no_gpt:
        label_topics_with_gpt(topics, enabled=True)
    else:
        logger.info("GPT labelling disabled via --no_gpt.")

    # Step 6: Validation
    metrics = evaluate_clustering(embeddings, labels)
    intruder_scores = intruder_word_test(keywords)
    distribution = topic_distribution(labels)

    # Write validation results to disk
    results_path = output_dir / "validation.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "intruder_scores": intruder_scores,
            "topic_distribution": distribution,
        }, f, indent=2)
    logger.info("Validation metrics written to %s", results_path)

    # Print summary
    print("\nValidation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nIntruder Scores (higher is better):")
    for tid, score in intruder_scores.items():
        print(f"  Topic {tid}: {score:.4f}")
    print("\nTopic Distribution:")
    for label, count in distribution.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
