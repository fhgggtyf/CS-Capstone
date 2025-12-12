# Investigating the Causes of Player Frustration in Modern Games

This repository contains the complete codebase for “Investigating the Causes of Player Frustration in Modern Games: A Large-Scale Review Analysis,” a computer science capstone project at New York University Abu Dhabi (NYUAD). The project leverages a multi-stage natural language processing (NLP) pipeline to mine and analyse over two million user reviews from Steam, Reddit, Metacritic and several official game forums. By combining bespoke data scrapers, systematic filtering (time windows, language detection and sentiment analysis) and state-of-the-art topic modelling (LDA and BERTopic), the pipeline categorises player complaints and maps them onto software architectural qualities and non-functional requirements (NFRs). The methodology and findings are detailed in the accompanying capstone report. This repository provides a reproducible implementation of that work so that other researchers and practitioners can extend the analysis or apply it to their own data sets.

## Contents
<!-- The table of contents provides a quick way to navigate this document. -->
1. Overview  
2. Repository Structure  
3. Setup and Installation  
4. Data Acquisition  
5. Data Cleaning and Sentiment Filtering  
6. Topic Modelling  
7. Latent Dirichlet Allocation (LDA)  
8. BERTopic  
9. Post-Processing and Analysis  
10. Results and Interpretation  
11. Customization and Extensibility  
12. Dependencies and Environment  
13. Project Organization  
14. Citation  
15. Acknowledgements  

## Overview

The goal of this project is to systematically identify what frustrates modern video game players. While difficulty is often assumed to be the primary cause of frustration, the capstone research demonstrates that technical instability, unfair matchmaking, toxic community behaviour, restrictive access policies and grind-heavy progression systems are more pervasive sources of discontent. By automating the collection and analysis of user reviews at scale, the project offers a reproducible blueprint for incorporating user feedback into software requirements engineering.

The pipeline proceeds in four high-level phases:

**1. Data Acquisition.** Custom scrapers collect raw reviews and comments from official APIs (Steam and Reddit) and from HTML pages (Metacritic and official game forums). A two-year time window is applied to ensure consistency across platforms.

**2. Filtering and Sentiment Analysis.** Records are filtered to English text only, duplicates are removed, and a hybrid sentiment classifier combines a fine-tuned DistilBERT model, VADER lexicon and domain-specific heuristics to label each review as positive, negative, neutral or mixed. Only negative reviews (those most likely to contain frustration) are retained for topic modelling.

**3. Topic Modelling.** Two unsupervised techniques are implemented:  
- **LDA:** A streaming, memory-efficient pipeline processes documents stored in SQLite and trains a multi-core LDA model, computing coherence scores, topic diversity metrics and exporting top words and document–topic weights.  
- **BERTopic:** A state-of-the-art clustering method combining transformer embeddings, UMAP dimensionality reduction, HDBSCAN clustering and cTF-IDF keyword extraction. An extensive custom stopword list is used to improve topic coherence.

**4. Post-Processing and Analysis.** Topic assignments are merged with metadata (game name, platform, playtime) to produce per-game/per-platform topic distributions. Summary tables and visualisations allow interpretation of which software qualities map to each topic cluster.

## Repository Structure

```
src/
├── Fetch/                           # Data acquisition scripts
│   ├── Steam_fetch_wrapper.py       # Fetches Steam reviews via Steam API
│   ├── data_scraper_Reddit_Reviews.py
│   ├── reddit_reviews_agent/
│   │   ├── configs/
│   │   ├── docs/
│   │   ├── scripts/run_local.sh
│   │   ├── src/
│   │   └── requirements.txt
│   ├── specific_website_fetches/
│   │   ├── Baldurs_gate_official_forum_fetcher.py
│   │   ├── EA_forum_scraper.py
│   │   ├── Overwatch_official_forum_fetcher.py
│   │   ├── data_scraper_CDPR_Cyberpunk_Forum.py
│   │   ├── data_scraper_Escape_from_Tarkov_Official_Forum.py
│   │   ├── metacritic_scraper.py
│   │   └── metacritic_scarper_wrapper.py
│   ├── reddit_comment_time_fetch.py
│   ├── fetch_html.py
│   └── steam_review_fetch.py
│
├── Extract/
│   ├── Time_filter.py
│   ├── english_filter.py
│   ├── filter.py
│   ├── filter_sentiment_emotion.py
│   ├── filter_sentiment_pos_neg_v1.py
│   ├── filter_sentiment_pos_neg_v2.py
│   ├── Combine_sentiment.py
│   ├── extract_negative_sentiment.py
│   ├── extract_negative_combine_sentiment.py
│   └── Visualization/
│       ├── Number_per_step/
│       ├── Platform_num/
│       ├── Sentiment_persentage/
│       ├── platform_counts.png
│       ├── sentiment_pie.png
│       └── review_counts_summary.csv
│
├── Analysis/
│   ├── PreLDA_analysis.py
│   ├── LDA_topic_modeling_improved.py
│   ├── bertopic_sqlite.py
│   ├── Bertopic_visualization.py
│   ├── game_topic_counts.py
│   ├── labels_per_topic.py
│   ├── retro_label_topics.py
│   ├── run_improved_all.py
│   ├── run_improved_negative.py
│   ├── collate_improved_runs_to_excel.py
│   └── Analysis.py
│
└── README.md
```

## Setup and Installation

Clone the repository:

```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Create a Python virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install dependencies:

```
pip install pandas numpy tqdm matplotlib seaborn scikit-learn \
gensim nltk pyldavis bertopic sentence-transformers \
hdbscan spacy rapidfuzz transformers==4.* \
praw pyyaml pytz tenacity
```

Optional dependencies for scraping:

```
pip install beautifulsoup4 requests
```

Download NLTK resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Data Acquisition

Scripts live in `src/Fetch`.

Key principles:

- Respect API limits (Steam API, PRAW).  
- Parameterize by game + date range.  
- Store everything in SQLite with clean schema.

Example Steam fetch:

```
python Steam_fetch_wrapper.py \
  --app-id 1091500 \
  --db-path Data_Extraction/Database/CS_Capstone.db \
  --table-prefix cyberpunk \
  --start-date 2023-01-01 \
  --end-date 2025-12-01 \
  --language english
```

Example forum scraper:

```
python data_scraper_Escape_from_Tarkov_Official_Forum.py \
  --output-path Data_Extraction/Forums/eft_forum_posts.csv \
  --max-pages 100
```

## Data Cleaning and Sentiment Filtering

Includes:

- **Temporal filtering**  
- **Language detection**  
- **Sentiment analysis (v1, v2 hybrid)**  
- **Negative-only extraction**  
- **Visualization of pipeline steps**

Final output table example:

```
frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only
```

## Topic Modelling

### LDA Topic Modelling

Streaming, memory-efficient, SQLite-driven.  
Computes:

- coherence (u_mass, c_v, c_npmi)  
- perplexity  
- topic diversity  
- inter-topic similarity  

Outputs:

- `topics_top_words.json`  
- `doc_topic_weights.csv`  

Run example:

```
python src/Analysis/LDA_topic_modeling_improved.py \
  --db-path Data_Extraction/Database/CS_Capstone.db \
  --table frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only \
  --text-column main_text \
  --output-dir Results/run_$(date +%Y%m%d_%H%M%S) \
  --num-topics 10 \
  --alpha auto \
  --eta 0.1 \
  --passes 5 \
  --iterations 100
```

### BERTopic Modelling

Uses transformer embeddings, UMAP, HDBSCAN, cTF-IDF.  
Custom stopword list removes:

- English stopwords  
- game titles  
- profanity  
- memes (“snail”, “democracy”, “planet”, etc.)  

Run example:

```
python src/Analysis/bertopic_sqlite.py \
  --db-path Data_Extraction/Database/CS_Capstone.db \
  --table frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only \
  --text-column main_text \
  --output-dir Results/bertopic_$(date +%Y%m%d_%H%M%S) \
  --embedding-model all-mpnet-base-v2 \
  --min-topic-size 50
```

## Post-Processing and Analysis

Scripts produce:

- `<config>_topic_size.csv`  
- `<config>_top5_topics_per_game.csv`  
- `<config>_top5_topics_per_platform.csv`  
- `<config>_top5_games_per_topic.csv`  
- `<config>_top5_platforms_per_topic.csv`  

Visualization and labeling scripts:

- `labels_per_topic.py`  
- `retro_label_topics.py`  
- `Bertopic_visualization.py`  
- `collate_improved_runs_to_excel.py`  

## Results and Interpretation

Highlights:

- LDA collapses themes unless K is large.  
- BERTopic produces 30 interpretable clusters.  
- Major frustration sources:  
  - server instability  
  - cheating/hacking  
  - toxic behaviour  
  - unfair matchmaking  
  - monetisation complaints  
  - poor ports/performance  
  - grind/RNG systems  
- Difficulty rarely forms a coherent topic.  
- PC players complain about performance; console players about access; mobile players about monetisation.

## Customization and Extensibility

You can:

- Add new data sources under `src/Fetch`.  
- Change time windows.  
- Adjust sentiment heuristics.  
- Swap embedding models.  
- Incorporate supervised learning.

## Dependencies and Environment

Key versions used:

| Package | Version | Notes |
|---------|---------|-------|
| pandas | 2.2 | Data manipulation |
| numpy | 1.26 | Numerical computing |
| gensim | 4.3 | LDA modelling |
| scikit-learn | 1.4 | PCA, metrics |
| nltk | 3.8 | Tokenization |
| pyLDAvis | 3.4 | LDA visualization |
| bertopic | 0.17 | BERTopic |
| sentence-transformers | 2.3 | Embeddings |
| hdbscan | 0.8 | Clustering |
| transformers | 4.38 | DistilBERT |
| spacy | 3.6 | Optional |
| praw | 7.8 | Reddit API |
| beautifulsoup4 | 4.12 | HTML parsing |
| rapidfuzz | 3.6 | Matching |
| pytz | 2024.1 | Timezones |

Hardware recommendations: 16–32 GB RAM for BERTopic; GPU helpful but not required.

## Project Organization

Example workflow:

```
# 1. Data collection
python src/Fetch/Steam_fetch_wrapper.py ...
cd src/Fetch/reddit_reviews_agent && ./scripts/run_local.sh

# 2. Apply filters
python src/Extract/Time_filter.py ...
python src/Extract/english_filter.py ...
python src/Extract/filter_sentiment_pos_neg_v2.py ...

# 3. Run BERTopic
python src/Analysis/bertopic_sqlite.py ...

# 4. Summaries
python src/Analysis/Analysis.py ...

# 5. Inspect results manually
```

## Citation

If you use this work:

**Jiacheng Xia. 2026. *Investigating the Causes of Player Frustration in Modern Games: A Large-Scale Review Analysis.* NYUAD Capstone Seminar, Spring 2026, Abu Dhabi, UAE.**

```
@misc{xia2026playerfrustration,
  title={Investigating the Causes of Player Frustration in Modern Games: A Large-Scale Review Analysis},
  author={Xia, Jiacheng},
  year={2026},
  howpublished={GitHub},
  note={\url{https://github.com/<your-username>/<repo-name>}}
}
```

## Acknowledgements

This project was undertaken as part of the NYUAD Computer Science capstone programme and was supervised by Professor Mohamad Kassab. Special thanks go to the maintainers of the open-source libraries that make large-scale NLP research accessible, including Gensim, BERTopic, Transformers and PRAW. The study also benefited from countless hours of feedback from fellow students and play testers who shared their experiences across Steam, Reddit, Metacritic and official forums.

_Last updated: 12 December 2025._
