# Topic Clustering Pipeline for Player Frustration Reviews

This repository contains a fully‑featured Python pipeline to discover and
analyse topics in large collections of short video‑game reviews.  The goal of
the project is to **categorise causes of player frustration** across many
games and produce human‑interpretable output suitable for academic use.

The pipeline is modular and extensible.  It computes semantic embeddings for
each review (optionally via the OpenAI API), clusters the embeddings with
HDBSCAN (or DBSCAN as a fallback), extracts distinctive keywords using
class‑based TF‑IDF (cTF‑IDF), enriches topics with keyphrases, constructs
rich topic objects, optionally calls GPT to generate concise labels and
summaries, and evaluates the coherence of the resulting topics.

## Repository Structure

```
pipeline/
 ├── embeddings.py          # compute review embeddings (OpenAI or local TF‑IDF + SVD)
 ├── clustering.py          # run HDBSCAN/DBSCAN clustering and auto‑tune hyperparameters
 ├── ctfidf.py              # compute class‑based TF‑IDF and extract top keywords
 ├── keybert_enrichment.py  # enrich topics with KeyBERT keyphrases or a fallback extractor
 ├── topic_builder.py       # assemble Topic objects and compute similarity & hierarchy
 ├── gpt_labeler.py         # optional GPT‑based labelling of topics
 ├── validation.py          # compute evaluation metrics and an intruder test
 ├── main.py                # command line entry point orchestrating the pipeline
README.md
```

## Requirements

* Python 3.8 or later
* Packages: `numpy`, `pandas`, `scikit‑learn`.  If available, install
  `hdbscan` and `keybert` to enable density‑based clustering and advanced
  keyphrase extraction.  Install `openai` if you wish to use GPT or the
  OpenAI embedding API.

Because this environment cannot fetch external dependencies, the included
demonstration runs with the fallback implementations (DBSCAN instead of
HDBSCAN and frequency‑based keyphrases instead of KeyBERT).  To realise the
full potential of the pipeline, install the optional packages in your own
environment.

## Quick Start

1. **Generate synthetic data and inspect topics**

   ```bash
   python -m pipeline.main --mode step1-4
   ```

   This command synthesises a small dataset of reviews, computes local
   TF‑IDF+SVD embeddings, clusters them, extracts keywords and phrases,
   builds topics and writes them to `outputs/topics_step4.json`.  It then
   prints a summary of each topic so you can inspect the intermediate
   results.

2. **Continue to GPT labelling and validation**

   After reviewing `topics_step4.json`, re‑run the script with:

   ```bash
   python -m pipeline.main --mode step5-6
   ```

   This will optionally call GPT (if you have set the `OPENAI_API_KEY`)
   to generate names and summaries for each topic and compute evaluation
   metrics such as silhouette score, Davies–Bouldin index, outlier
   percentage and intruder scores.  Results are saved to
   `outputs/validation.json`.

3. **Analyse your own data**

   Place your reviews in a CSV file with a `text` column and run:

   ```bash
   python -m pipeline.main --mode step1-4 --dataset my_reviews.csv --embedding openai
   ```

   Provide your OpenAI API key via the `OPENAI_API_KEY` environment
   variable to use the powerful `text‑embedding‑3‑large` model.  According
   to OpenAI, this model produces vectors with up to 3072 dimensions and
   offers the strongest retrieval performance【274848708673263†L170-L176】.  Pricing is
   $0.00013 per thousand tokens【274848708673263†L206-L209】.  You can set the
   `dimensions` parameter in `compute_embeddings` to shrink vectors (e.g., to
   1024 dimensions) with minimal loss in quality【274848708673263†L220-L233】.

## How it Works

### Step 1 – Embedding

Each review is converted into a dense vector.  When an API key is
available, the pipeline uses OpenAI's `text‑embedding‑3‑large` model, which
supports up to 3,072 dimensions and delivers best‑in‑class performance【274848708673263†L170-L176】.  If no API
key is supplied, the pipeline falls back to a TF‑IDF + SVD representation
(`minilm` mode).

### Step 2 – Clustering

Embeddings are clustered with HDBSCAN, a hierarchical density‑based
algorithm.  HDBSCAN identifies dense regions of points and labels points
that do not belong to any cluster as noise (`-1`)【824455034344165†L331-L339】.  The
algorithm returns cluster labels and membership probabilities.  If
`hdbscan` is unavailable, the pipeline falls back to DBSCAN and tunes
`eps` and `min_samples` over a small grid.

### Step 3 – Keyword Extraction

The pipeline uses **class‑based TF‑IDF** (cTF‑IDF) to find terms that
distinguish each cluster.  cTF‑IDF merges all documents in a cluster into
one long document and computes inverse document frequency over classes
rather than individual documents.  This approach highlights words that are
unique to a cluster and down‑weights common words【178967755416880†L97-L123】.  The
pipeline returns the top keywords per cluster along with their weights.

For additional nuance, the pipeline attempts to use **KeyBERT** to
extract keyphrases.  KeyBERT leverages BERT embeddings and cosine
similarity to select words and phrases most similar to the document【512879448139413†L56-L73】.  When
KeyBERT is not installed, a simple bigram frequency extractor serves as a
fallback.

### Step 4 – Topic Construction

For each cluster, the pipeline builds a `Topic` object containing the
cluster ID, size, keywords, keyphrases, a sample of reviews, the mean
embedding and a similarity map to other topics.  Topics can optionally
be grouped into higher‑level categories via agglomerative clustering on
their embeddings.

### Step 5 – GPT Labelling (optional)

Topics can be labelled with concise, human‑readable names and summaries
using GPT.  The labeller sends the top keywords, keyphrases and a few
example reviews to the GPT model and asks for a name and summary.  This
step is optional and controlled by the `--no_gpt` flag.  You must set
`OPENAI_API_KEY` for this step to work.

### Step 6 – Validation

The final step computes several metrics:

* **Silhouette score** – how well points fit within their own clusters vs
  others.
* **Davies–Bouldin index** – ratio of within‑cluster scatter to
  separation; lower is better.
* **Outlier percentage** – fraction of reviews labelled as noise.
* **Intruder word test** – approximates topic coherence by inserting a
  random intruder keyword and measuring how much its weight deviates from
  the genuine keywords; higher scores indicate more coherent topics.

Results are saved to `outputs/validation.json`.

## Hardware and Runtime Recommendations

Processing 800,000 reviews is computationally intensive.  Here are
guidelines:

* **Embedding choice** – Using OpenAI's `text‑embedding‑3‑large` offloads
  heavy computation to the API, returning 3,072‑dimensional vectors.  The
  OpenAI guide notes that embeddings can be shortened to lower dimensions
  (e.g., 256 or 1,024) by specifying the `dimensions` parameter with
  little performance loss【274848708673263†L220-L233】.  For large datasets this can
  dramatically reduce memory consumption.
* **Local fallback** – The `minilm` fallback uses TF‑IDF and SVD with
  300 dimensions.  It is fast and requires modest memory but lacks the
  semantic richness of transformer embeddings.
* **Clustering** – HDBSCAN has complexity roughly O(n log n).  For
  hundreds of thousands of points a GPU or multi‑core CPU is highly
  recommended.  If HDBSCAN is unavailable, DBSCAN may struggle with very
  large datasets; consider sampling or using an approximate nearest
  neighbour index.
* **Memory** – 800k × 3,072 float32 embeddings require ~9.8 GB of RAM.
  Shortening embeddings to 1,024 dimensions brings this down to ~3.3 GB.
* **Runtime** – With a modern GPU (e.g., NVIDIA RTX 3060) local embedding
  via transformer models can process thousands of reviews per second.  On
  CPU‑only hardware, embedding large corpora using transformers may take
  hours.  Using OpenAI embeddings avoids this cost entirely.

## Citations

* **OpenAI embedding model** – OpenAI’s `text‑embedding‑3‑large` model
  produces embeddings up to 3,072 dimensions and is the best performing
  embedding model【274848708673263†L170-L176】.  Pricing is $0.00013 per
  thousand tokens【274848708673263†L206-L209】 and embeddings can be shortened to
  smaller sizes by specifying the `dimensions` parameter【274848708673263†L220-L233】.
* **HDBSCAN noise labelling** – Points not belonging to any cluster are
  assigned the label `-1` and cluster membership strengths are available
  via the `probabilities_` attribute【824455034344165†L331-L339】.
* **cTF‑IDF** – Class‑based TF‑IDF joins documents in each cluster and uses
  the number of classes instead of the number of documents in the IDF
  calculation to emphasise class‑specific words【178967755416880†L97-L123】.
* **KeyBERT** – KeyBERT uses BERT embeddings and cosine similarity to find
  sub‑phrases that are most similar to the document, providing intuitive
  keyphrases【512879448139413†L56-L73】.

---

For questions or improvements, please open an issue or submit a pull request.
