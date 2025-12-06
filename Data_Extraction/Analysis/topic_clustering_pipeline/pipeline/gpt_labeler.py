"""
gpt_labeler.py
===============

This module provides optional functionality to call OpenAI's GPT models to
generate human‑readable names and summaries for each topic.  Given a list of
``Topic`` objects, the ``label_topics_with_gpt`` function sends a prompt
containing the topic's keywords, keyphrases and representative examples and
requests a concise, human‑friendly description.

The GPT call is optional and controlled via the ``enabled`` flag.  If no
OpenAI API key is available or the ``openai`` package is not installed, the
function gracefully returns without modifying the topics.  Users must set the
``OPENAI_API_KEY`` environment variable to call the API.

Prompts are designed to encourage the model to assign frustration category
labels and suggest merges or splits when appropriate.  You can adjust the
template or add further instructions as needed.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from .topic_builder import Topic

logger = logging.getLogger(__name__)

try:
    import openai  # type: ignore
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


def _build_prompt(topic: Topic) -> str:
    """Construct a prompt for GPT summarisation.

    The prompt includes the cluster's keywords, keyphrases and a few example
    reviews.  It asks GPT to provide a human‑readable topic name, a concise
    summary of the frustration being expressed, and optional suggestions to
    merge or split topics.
    """
    keywords_str = ", ".join([kw for kw, _ in topic.keywords])
    phrases_str = ", ".join([ph for ph, _ in topic.phrases])
    examples_str = "\n- ".join(topic.examples[:3])
    prompt = (
        f"You are helping to analyse player reviews of video games to identify causes of frustration.\n"
        f"Given the following cluster information, provide:\n"
        f"1. A short, human‑readable name (1–4 words) capturing the main frustration theme.\n"
        f"2. A concise summary (2–3 sentences) of the frustration cause.\n"
        f"3. Optional: If this topic overlaps strongly with others or should be split, mention that.\n"
        f"4. Optional: Assign a general frustration category (e.g., Technical Performance, Game Balance, Progression, Monetisation, Social Interaction, UI/UX, Narrative).\n"
        f"\nKeywords: {keywords_str}\n"
        f"Keyphrases: {phrases_str}\n"
        f"Example reviews:\n- {examples_str}\n"
    )
    return prompt


def label_topics_with_gpt(topics: List[Topic], model: str = "gpt-3.5-turbo", enabled: bool = True,
                          temperature: float = 0.2) -> None:
    """Annotate topics with human‑readable names and summaries using GPT.

    Parameters
    ----------
    topics : list[Topic]
        List of topic objects to label.  Each topic will be mutated in place to
        include ``name`` and ``summary`` attributes on success.
    model : str, optional
        Name of the GPT model to use.  Defaults to ``gpt-3.5-turbo``.  You may
        switch to ``gpt-4-turbo`` or another model with better performance.
    enabled : bool, optional
        Whether to perform labelling.  If ``False``, the function returns
        immediately without contacting the API.  Defaults to True.
    temperature : float, optional
        Sampling temperature for GPT.  Lower values make output more
        deterministic.  Defaults to 0.2.

    Notes
    -----
    This function requires that ``openai`` is installed and that the
    ``OPENAI_API_KEY`` environment variable is set.  If either of these
    conditions is not met, labelling is skipped with a warning.
    """
    if not enabled:
        logger.info("GPT labelling disabled by configuration.")
        return
    if not _OPENAI_AVAILABLE:
        logger.warning("openai package not installed; skipping GPT labelling.")
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping GPT labelling.")
        return
    openai.api_key = api_key
    for topic in topics:
        prompt = _build_prompt(topic)
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message["content"].strip()
            # Heuristically parse the response into name and summary by splitting at first newline
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if lines:
                topic.name = lines[0]
                topic.summary = " ".join(lines[1:]) if len(lines) > 1 else ""
            else:
                topic.name = ""
                topic.summary = ""
            logger.info("GPT labelled topic %s as '%s'", topic.topic_id, getattr(topic, 'name', ''))
        except Exception as e:
            logger.error("Failed to label topic %s with GPT: %s", topic.topic_id, str(e))
            topic.name = ""
            topic.summary = ""
