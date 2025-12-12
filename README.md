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


## Data Cleaning and Sentiment Filtering

Includes:

- **Temporal filtering**  
- **Language detection**  
- **Sentiment analysis (v1, v2 hybrid)**  
- **Negative-only extraction**  
- **Visualization of pipeline steps**


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

### BERTopic Modelling

Uses transformer embeddings, UMAP, HDBSCAN, cTF-IDF.  
Custom stopword list removes:

- English stopwords  
- game titles  
- profanity  
- memes (“snail”, “democracy”, “planet”, etc.)  

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

## Full Script Documentation

### Analysis scripts
These scripts live in src/Analysis  and perform topic modelling, result collation, and reporting.

#### Analysis.py
Purpose:  Reads a doc_topic_weights.csv  file generated by the LDA pipeline and merges it with game
and platform metadata from a SQLite database. It then produces several summary CSV files showing topic
sizes, the top topics per game, the top topics per platform, and the top games/platforms per topic.
Usage:  Edit the CONFIG, DATASET , TOPIC_CSV , DB_PATH  and TABLE_NAME  variables near the top
of the script to point to your own data. Run the script from a terminal:

python src/Analysis/Analysis.py
Expected Output:  The script prints progress messages and writes the following files into the current
working directory (names include the CONFIG and DATASET  prefixes):
topic_size.csv  – counts of documents per topic.
top5_topics_per_game.csv  – average topic strengths per game with the top N topics listed.
top5_topics_per_platform.csv  – average topic strengths per platform.
top5_games_per_topic.csv  – the five games with highest mean weight for each topic.
top5_platforms_per_topic.csv  – analogous to the above but for platforms.• 
• 
• 
• 
• 
1

#### Bertopic_visualization.py
Purpose:  Loads  BERTopic  results  from  two  CSVs  ( topic_keywords.csv  and
reviews_with_topics.csv ) and produces interactive visualizations using Plotly. The script cleans the
representation column, constructs bar charts of topic counts, a treemap, a per‑topic keyword bar chart, a
sunburst chart, and a placeholder scatter plot.
Usage:  Set topic_file  and review_file  at the top of the script to the actual paths of your BERTopic
output files. Ensure you have plotly installed. Run:

```
python src/Analysis/Bertopic_visualization.py
```
Expected  Output:  The  script  prints  the  first  few  rows  of  the  loaded  DataFrames  and  opens  multiple
interactive Plotly windows. It does not write new files; the visualizations must be exported manually from
the browser if desired.

#### LDA_topic_modeling_improved.py
Purpose:  Implements  an  improved,  streaming  Latent  Dirichlet  Allocation  (LDA)  pipeline  that  reads
documents from a SQLite database, builds a vocabulary, serializes a bag‑of‑words corpus, trains an LDA
model with multiple passes, and computes evaluation metrics (coherence, perplexity, topic diversity). It
optionally  detects  bigrams,  discards  the  top‑N  frequent  words,  assigns  theme  labels  and  uses
multithreading to speed up training.
Usage:  This script exposes many command‑line options. At minimum you must supply the database path,
table name, text column and number of topics. A typical invocation looks like:
python src/Analysis/LDA_topic_modeling_improved.py \
--dbData_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db \
--table frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only \
--text-col main_text \
--k10\
--output-dir Results/run_ $(date+%Y%m%d_%H%M%S )
Other  flags  control  the  number  of  passes,  iterations,  chunk  size,  dictionary  thresholds  ( --min-df , 
--max-df ), whether to detect bigrams ( --use-bigrams ), the Dirichlet priors ( --alpha ,  --eta),
random seed, and sample sizes for computing coherence and visualisations. See the argument parser
within the script for full details.
Expected Output:  The script creates a new subdirectory under the specified --output-dir  containing:
metrics_summary.json  – model statistics and evaluation metrics.
topics_top_words.json  – top words and optional labels for each topic.
doc_topic_weights.csv  – per‑document topic mixtures and the dominant topic.• 
• 
• 
2

Optional plots ( topic_sizes.png , doc_scatter_umap.png , intertopic_js_heatmap.png ,
etc.) depending on flags.
A PyLDAvis HTML file if --generate-html  is enabled.

#### PreLDA_analysis.py
Purpose:  Generates simple summaries of the sentiment labels (e.g. POSITIVE, NEGATIVE, NEUTRAL) before
running LDA. It loads a table from the sentiment database and produces counts per game, overall counts
and counts per platform.
Usage:  Edit the hard‑coded SQLite path and table name at the top of the script. Run:

```
python src/Analysis/PreLDA_analysis.py
```
Expected  Output:  The  script  prints  three  summaries  to  stdout  and  saves  them  to
summary_per_game.csv ,  summary_overall.csv  and  summary_per_platform.csv  in the current
directory.

#### bertopic_sqlite.py
Purpose:  Executes the BERTopic pipeline on reviews stored in a SQLite database. It reads documents from a
specified table, cleans and tokenises the text (using custom stopwords and optional bigram detection),
embeds  documents  with  a  transformer  model,  clusters  them  with  HDBSCAN  and  extracts  per‑topic
keywords via cTF‑IDF. It outputs the topic assignments and keywords to CSV files.
Usage:  Invoke the script with command‑line arguments specifying the database, table and text column,
e.g.:
python src/Analysis/bertopic_sqlite.py \
--dbData_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db \
--table frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only \
--text-col main_text \
--output-dir Results/bertopic_ $(date+%Y%m%d_%H%M%S )
Optional flags include --embedding-model  (default all-mpnet-base-v2 ), --min-topic-size , --
nr-docs-save  (for  sampling),  --remove-stopwords-file  and  --use-bigrams .  See  the  internal
argument parser for full details.
Expected Output:  Under the specified output directory, two key CSV files are produced:
reviews_with_topics.csv  – original documents with their assigned topic IDs and probabilities.
topic_keywords.csv  – per‑topic count, name (if provided), and the list of representative
keywords (with weights).
No visualisations are created by default; use Bertopic_visualization.py  to explore the results.

#### collate_improved_runs_to_excel.py
Purpose:  Given  a  directory  containing  multiple  LDA  run  subdirectories  (each  produced  by
LDA_topic_modeling_improved.py ), this script collates their metrics, topic words, document counts
and images into a single Excel workbook. It extracts configuration parameters from each run’s directory
name and summarises them in a combined sheet.
Usage:  Run with the directory of runs and specify an output file name:
```
python src/Analysis/collate_improved_runs_to_excel.py runs_dir -osummary.xlsx
```
Where  runs_dir  contains  subdirectories  like  k15_eta0.05_drop0_uni ,  each  with  a
metrics_summary.json . The script uses pandas and the xlsxwriter  engine (installed automatically
via pandas) to generate the Excel file.
Expected Output:  A multi‑sheet Excel workbook summarising each run’s overview metrics, topic sizes
(estimated and actual), diversity metrics, top words per topic and embedded plots. A Combined  sheet lists
one row per run for quick comparison.

#### game_topic_counts.py
Purpose:  Aggregates the dominant topic counts per game. Given a doc_topic_weights.csv  file and a
database table containing id and game_name , it produces a pivot table counting how many documents
per game were dominated by each topic.
Usage:
python src/Analysis/game_topic_counts.py \
--csv_path doc_topic_weights.csv \
--db_path Data_Extraction/Database/CS_Capstone_Sentiment_time_filtered.db \
--table_name frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only \
--output_excel game_topic_counts.xlsx
Expected Output:  An Excel file ( game_topic_counts.xlsx ) containing a sheet named TopicCounts
where rows are game names, columns are topic IDs and each cell contains the count of documents with
that dominant topic for that game.

#### labels_per_topic.py
Purpose:  Merges LDA topic assignments with sentiment labels from the database to produce a per‑topic
sentiment summary. It counts how many documents with each sentiment label fall under each dominant
topic.
4

Usage:  Adjust the csv_path , db_path , table_name  and output_path  variables at the top of the
script. Then run:

```
python src/Analysis/labels_per_topic.py
```
Expected Output:  A CSV file ( <config>_negative_topic_sentiment_summary.csv ) containing one
row per topic with columns for each sentiment label and counts in each.

#### retro_label_topics.py
Purpose:  Retroactively  assigns  human‑friendly  labels  to  LDA  topics  using  KeyBERT.  It  searches  for  all
topics_top_words.json  files under a given root directory, builds pseudo‑documents from the top
words of each topic, extracts candidate phrases via KeyBERT and writes the best phrase back into the JSON’s
label field. Options include MMR diversification, n‑gram ranges, candidate counts, backups and dry‑run
mode.
Usage:
python src/Analysis/retro_label_topics.py path/to/runs_dir --mmr--diversity 0.7--backup
Where  path/to/runs_dir  is the directory containing subdirectories with  topics_top_words.json .
Install  the  KeyBERT  library  beforehand  ( pip  install  keybert  and  optionally  sentence-
transformers ).
Expected Output:  For each JSON file found, the script updates the  label field for each topic. If  --
backup is supplied, it writes a .bak file alongside the original. Progress is printed to stdout; no new files
are created beyond the backups.

### run_improved_all.py  and run_improved_negative.py
Purpose:  These orchestrators automate sweeping across multiple hyperparameter configurations for the
improved LDA script. They construct command‑line arguments to LDA_topic_modeling_improved.py
based on pre‑defined parameter grids (topic numbers,  eta priors, number of top words to drop and
whether to use bigrams), invoke the script for each combination and organise outputs into timestamped
directories. The run_improved_negative.py  version points to a negative‑only table.
Usage:  Simply run the script:
```
python src/Analysis/run_improved_all.py
```
or for the negative‑only dataset:
```
python src/Analysis/run_improved_negative.py
```
Before  running,  adjust  the  parameter  lists  ( k_values ,  eta_values ,  drop_topn_values , 
bigram_options ) at the top of each script to control the sweep. The script will create a timestamped
directory under Data_Extraction/Analysis/Results  and within it a subdirectory for each parameter
combination.
Expected  Output:  A  tree  of  run  directories,  each  containing  the  outputs  of
LDA_topic_modeling_improved.py  (metrics  JSON,  topic  words,  doc-topic  weights  and  plots)  and  a
run.log  capturing the script’s output. The orchestrator prints progress information and a final message
noting where the results are stored.

### Extract scripts
Scripts in src/Extract  perform data cleaning, filtering and sentiment labelling. Many of them operate
on SQLite tables and are configured via constants at the top of the file.

#### Combine_sentiment.py
Purpose:  Defines  helper  functions  to  merge  multiple  sentiment‑filtered  tables  (from  different  analysis
versions) into unified tables. It normalises column names and time formats and maps normalised game
names to canonical names. The script is marked as unused in the current pipeline but provides reference
code for combining tables.
Usage:  If  you  wish  to  use  it,  edit  DB_PATH ,  BASE_FINAL_COLS ,  VERSION_EXTRA_COLS  and  the
candidate column lists. Then import the functions into your own script or call them interactively. There is no
__main__  section.
Expected Output:  When integrated into a pipeline, this code can create unified tables in the database with
combined sentiment columns. Alone, the file does not produce outputs.

#### Time_filter.py
Purpose:  Creates a new SQLite database containing only rows from the last two years from each table in a
source database. It identifies a likely time column in each table, parses the latest timestamp and computes
a cutoff (latest minus 2 years). Rows older than the cutoff are dropped; rows without a valid timestamp are
retained.
Usage:  Set SOURCE_DB  and OUTPUT_DB  at the top. Running the script will delete any existing output DB
and repopulate it:

```
python src/Extract/Time_filter.py
```

Expected Output:  The script prints messages about each table (what time column it found and how many
rows were copied) and writes a new database file at the path specified by  OUTPUT_DB  containing the
filtered tables.

### #english_filter.py
Purpose:  Filters  a  table  to  contain  only  English  text.  It  reads  from  a  source  table  and  uses  the
langdetect  library to detect the language of each document’s  main_text , then writes English rows
into a new table with _english_only  suffix.
Usage:  Set DB_PATH , SOURCE_TABLE  and OUTPUT_TABLE  at the top of the script. Then run:

```
python src/Extract/english_filter.py
```

Expected Output:  The script shows a progress bar via  tqdm and inserts only English rows into the
specified output table in the same database. It prints a completion message when finished.

#### extract_negative_combine_sentiment.py
Purpose:  Creates combined tables containing  only  rows labelled with negative sentiment. It reads from
version‑specific sentiment tables (currently configured to process only the v2 hybrid sentiment table) and
writes a new table with the suffix _negative_only  into the same database. It includes helper functions
for normalising game names and time parsing.
Usage:  Set DB_PATH , adjust VERSION_EXTRA_COLS  if necessary and run:

```
python src/Extract/extract_negative_combine_sentiment.py
```

Expected  Output:  A  new  table  (e.g.
frustrated_sentiment_pos_neg_v2_sentiment_combined_negative_only )  is  created  in  the
database containing only negative reviews. The script prints status messages but produces no external files.

#### extract_negative_sentiment.py
Purpose:  Copies  rows  with  negative  sentiment  from  a  source  table  into  a  new  table  with
_negative_only  suffix. Unlike the combined version, this script merely filters by the  final_label
column and does not merge multiple versions. It is marked as unused but can serve as a simple template.
Usage:  Set DB_PATH , SOURCE_TABLE  and OUTPUT_TABLE , then run:

```
python src/Extract/extract_negative_sentiment.py
```

Expected Output:  The script inserts rows where final_label  equals 'NEGATIVE'  into the new table
and prints a message summarising the operation.

#### filter.py
Purpose:  Searches every table in the database for text containing predefined frustration patterns and
writes the matching rows into new tables suffixed with _frustrated . It normalises contractions, defines
regex  patterns  for  categories  such  as  “Bugs  /  Technical  Failures”,  “Unfairness  /  Inequity”  and
“Repetitiveness / Grind”, scans all text columns and adds a matched_categories  column listing which
categories were found. An optional PLATFORM  tag can be added to label the comment platform.
Usage:  Set SOURCE_DB  to the database you want to scan and optionally set PLATFORM  to a string such
as 'steam' , 'reddit' , etc. Run:

```
python src/Extract/filter.py
```

Expected Output:  For each table containing text columns, a new table (original name plus _frustrated )
is created in the same database with the subset of rows that matched at least one category. Progress
messages list which tables were processed and how many records were saved.

#### filter_sentiment_emotion.py
Purpose:  Runs an emotion classifier (DistilRoBERTa) on the text in each table to assign a dominant emotion
and a coarse final sentiment label (NEGATIVE, NEUTRAL or POSITIVE). It processes texts in batches, skipping
those exceeding the model’s maximum token length, and writes the results into new tables with a prefix
identifying the analysis version.
Usage:  This script requires  transformers  and PyTorch. Ensure you have a GPU or CPU available and
install the model via Hugging Face. The script automatically detects input tables from an input database
(CS_Capstone.db ) and writes to an output database ( CS_Capstone_Sentiment.db ). Run:

```
python src/Extract/filter_sentiment_emotion.py
```
Expected  Output:  For  each  table  processed,  a  new  table  prefixed  with
frustrated_sentiment_emotions_  is  created  in  the  output  database.  Additional  columns  include
dominant_emotion , dominant_score , top_emotions , emotions_json  and final_label .

### filter_sentiment_pos_neg_v1.py
Purpose:  Labels reviews with binary sentiment using a DistilBERT model fine‑tuned on the SST‑2 dataset. It
determines the main text column per table, applies length filters, runs sentiment analysis in batches and
writes the results into new tables prefixed with frustrated_sentiment_pos_neg_v1_ .
8

Usage:  The  script  expects  an  input  database  ( CS_Capstone.db )  and  writes  to
CS_Capstone_Sentiment.db . Simply run:

```
python src/Extract/filter_sentiment_pos_neg_v1.py
```

You may adjust the MAIN_TEXT_FIELD_MAP  and the path constants at the top.
Expected Output:  A new table for each processed input table, containing the original columns plus two
new columns: sentiment_label  (POSITIVE or NEGATIVE) and sentiment_score  (model confidence).
Progress messages show how many rows were processed and skipped.

### filter_sentiment_pos_neg_v2.py
Purpose:  Performs  hybrid  sentiment  classification  combining  the  transformer  model  from  v1,  VADER
lexicon scores and custom negative keyword heuristics. It assigns a final_label  (POSITIVE, NEGATIVE,
NEUTRAL or MIXED), retains the transformer’s raw label and score, and records the VADER compound score.
The  script  processes  each  table  in  chunks  and  writes  the  output  into  tables  prefixed  with
frustrated_sentiment_pos_neg_v2_ .
Usage:  Run the script after installing transformers  and, optionally, vaderSentiment :

```
python src/Extract/filter_sentiment_pos_neg_v2.py
```
Edit the MAIN_TEXT_FIELD_MAP  at the top to map table names to the main text column. The script uses
the same input and output database conventions as v1.
Expected  Output:  For  each  processed  table,  a  new  table  with  the  prefix
frustrated_sentiment_pos_neg_v2_  is created, containing the original columns plus final_label ,
transformer_label ,  transformer_score  and  vader_compound . Progress and chunk processing
information are printed.

## Fetch scripts
This category covers scripts under src/Fetch  that scrape or retrieve raw data.

### Steam_fetch_wrapper.py
Purpose:  Automates  scraping  of  Steam  reviews  for  multiple  games  by  invoking  the  interactive
steam_review_fetch.py  script with simulated user input. You define a list of game names and App IDs;
the wrapper loops through them and passes the necessary inputs to the target script.
Usage:  Populate game_list  with [<display_name>, <app_id>]  entries. Then run:

```
python src/Fetch/Steam_fetch_wrapper.py
```
Make sure  steam_review_fetch.py  is available and functional. The wrapper uses  subprocess.run
with input to answer the interactive prompts ( y for default date ranges).
Expected Output:  For each game in the list, the underlying Steam scraper produces a new table in the
SQLite database ( Raw_Reviews.db  or another DB specified in the interactive script) containing review
data (review text, metadata, votes, etc.). The wrapper prints messages before and after each game.

### steam_review_fetch.py
Purpose:  Interactively fetches reviews from the Steam Web API. It prompts the user for a game name, a
Steam App ID, and whether to filter by date range. It then paginates through all available reviews (100 per
page), captures metadata (votes, posted date, voted up/down) and writes the results into a SQLite database.
Usage:  Run the script directly and follow the prompts:

```
python src/Fetch/steam_review_fetch.py
```
You will be asked for the game name (used to name the table), the App ID (integer), whether to specify a
date window, and whether to proceed. A default two‑year window is offered by typing y for both start and
end dates.
Expected  Output:  A  new  table  named  <game_name>_steam_reviews  is  created  in  the  specified
database  (by  default  Raw_Reviews.db )  with  columns  such  as  id,  review,  time,  voted_up , 
votes_up , votes_funny , etc. Progress messages are printed for each page of reviews.

### reddit_comment_time_fetch.py
Purpose:  Fills in missing timestamp fields for Reddit comments in a sentiment‑labelled table. It uses the
PRAW API to retrieve each comment by ID, reads its  created_utc  timestamp, formats it as ISO and
updates two columns ( time_unix , time_str ) in place.
Usage:  Ensure you have valid Reddit API credentials. Edit the script to set the correct table name and
database path. Then run:

```
python src/Fetch/reddit_comment_time_fetch.py
```
Expected Output:  For each comment with missing timestamps, the script updates the two columns in the
database and prints a message. If an error occurs for a comment (e.g. due to a removed comment), it logs
the exception.
10

### data_scraper_Reddit_Reviews.py
Purpose:  Collects posts and comments from specific League of Legends patch‑note threads on Reddit. It
iterates  over  a  hard‑coded  list  of  URLs,  fetches  each  submission  via  PRAW,  recursively  traverses  the
comment  tree  and  stores  posts  and  comments  into  two  tables  ( <game>_Reddit_posts  and
<game>_Reddit_comments ) within Raw_Reviews.db .
Usage:  Set game_name  and web_url  at the top of the script, provide your Reddit API credentials in the
reddit = praw.Reddit(...)  block and run:

```
python src/Fetch/data_scraper_Reddit_Reviews.py
```
Expected Output:  Two tables for the specified game are created in Raw_Reviews.db . The posts table
records post ID, title, body and author; the  comments  table includes comment ID, parent ID, post ID,
author ,  body,  timestamp  and  depth.  The  script  prints  confirmation  messages  after  saving  each  post’s
comments.

### fetch_html.py
Purpose:  Simple utility to download the raw HTML bytes of any URL and save it to a file. It supports HTTP(S)
URLs and automatically generates an output filename based on the host and path if none is provided.
Usage:
```
python src/Fetch/fetch_html.py "https://example.com" -odownloaded.html
```
You can also specify an output directory ( -o dir/ ), in which case the script generates a timestamped
filename. A --insecure  flag allows skipping TLS verification (not recommended).
Expected Output:  A single  .html file saved to the specified location containing the raw bytes of the
fetched page. The script prints the full file path upon completion or an error message if the request fails.
reddit_reviews_agent  (package)
The reddit_reviews_agent  package implements a configurable ingestion agent for Reddit. It is not a
single script but a set of modules orchestrated by the run_local.sh  script.
scripts/run_local.sh
Purpose:  Wrapper shell script to execute the ingestion pipeline with a local YAML configuration. It sets
environment variables and calls the reddit_ingest.py  Python script.
Usage:  Modify  configs/reddit.example.yaml  to specify games, subreddits, time window, heuristics
and Reddit API credentials. Then run:
11

cdsrc/Fetch/reddit_reviews_agent
./scripts/run_local.sh
Expected Output:  A SQLite database (path defined in the YAML under project.db_path ) containing one
table per game (name derived from the game title) with posts and comments labelled by heuristics. A
reddit_ingest_runs  table records the run metadata. Log messages and counts are printed to the
console.

### reddit_ingest.py
Purpose:  Entry point for the Reddit ingestion pipeline. It loads the YAML configuration, sets up logging,
calls pipeline.collect()  and records run statistics.
Usage:  Normally invoked via run_local.sh . If run directly, provide the path to the YAML configuration:
python reddit_ingest.py path/to/config.yaml
Expected Output:  The script orchestrates the pipeline described below and does not return a value. It
writes data into the configured database and prints run statistics.

## Internal modules
config_loader.py  – loads YAML configs and provides a helper class for dotted retrieval.
db.py – wraps SQLite access, creates tables and indexes, performs upserts and manages a runs
metadata table.
heuristics.py  – contains helper functions to detect English texts, approximate review‑likeness,
extract platform tokens and playtime hours.
matcher.py  – normalises strings, performs exact/alias/fuzzy matching between post text and
canonical game names, and constructs safe table names.
pipeline.py  – the main logic that collects posts and comments from multiple subreddits, filters
them by time window, language and review heuristics, extracts metadata, and writes rows into the
database. It iterates over configured games and subreddits, applies heuristics and calls 
db.upsert_row  for each post and comment.
reddit_client.py  – sets up a PRAW Reddit client from the config and defines retry wrappers for
listing and search queries.
util.py  – utility functions for computing a configuration hash, converting nested dicts to flat lists,
obtaining the current epoch time and checking if timestamps fall within a window.
These modules are imported by reddit_ingest.py  and are not intended to be run directly.
Specific website fetchers
Scripts  in  src/Fetch/specific_website_fetches  target  particular  forums  and  review  sites.  They
typically prompt for user input and save results into Raw_Reviews.db .• 
• 
• 
• 
• 
• 
• 
12

### Baldurs_gate_official_forum_fetcher.py
Purpose:  Scrapes  the  first  post  of  each  thread  from  UBB.threads‑based  forums  (originally  for  Larian
Studios’ Baldur’s Gate 3 forum). It iterates through index pages, collects thread IDs and fetches each thread
to capture the opening post’s title, author , date/time and content. Posts are inserted into an SQLite table.
Usage:  Run the script and follow the prompts:

```
python src/Fetch/specific_website_fetches/Baldurs_gate_official_forum_fetcher.py
```
You will be asked for the base forum index URL and how many pages to crawl. The script can work with live
HTTP URLs or local HTML files prefaced with file:// .
Expected Output:  The script writes entries into the  Baldurs_gate_official_forum_posts  table in
Raw_Reviews.db . Each row includes the thread number , thread URL, page URL, title, author , post date/
time and content. It prints progress information for each page and thread processed.

### EA_forum_scraper.py
Purpose:  Uses EA’s GraphQL API to download forum posts from a specified board (e.g. Battlefield 2042
general discussion). It replays a browser request with your bearer token and cookies, paginates through the
conversation list, filters posts to a two‑year window and writes each message’s metadata and content to a
table.
Usage:  Supply your own AUTH_BEARER  and COOKIE_HEADER  values at the top (obtained from browser
DevTools). Edit BOARD_ID  and TABLE_NAME  as needed. Run:

```
python src/Fetch/specific_website_fetches/EA_forum_scraper.py
```
Expected  Output:  A  table  named  as  specified  by  TABLE_NAME  (e.g.
battlefield_2042_general_discussion_en_official_EA_forum_posts )  is  created  in
Raw_Reviews.db  with one row per forum message. Columns include post ID, uid, title, author , rank, post
date, HTML content, plain text, reply count and tags. The script prints progress messages showing how
many posts were inserted.

### Overwatch_official_forum_fetcher.py
Purpose:  Crawls the Blizzard Overwatch general discussion forum and extracts the first post of each thread.
It supports fetching pages by HTTP and optionally falls back to local HTML files. The script keeps scraping
until it encounters pages where all threads are older than a specified cutoff (two years by default).
Usage:  Run the script and enter the number of pages to scrape when prompted:

```
python src/Fetch/specific_website_fetches/Overwatch_official_forum_fetcher.py
```
The  script  can  be  configured  via  constants  at  the  top  (e.g.  BASE_FORUM_URL ,  LOCAL_HTML_DIR , 
DB_PATH ).
Expected Output:  An  overwatch_2_official_forum_posts  table in  Raw_Reviews.db  containing
one row per thread with the thread ID, title, URL, post ID, author , creation time, HTML content and plain
text. Progress output includes pages visited and threads processed.

### data_scraper_CDPR_Cyberpunk_Forum.py
Purpose:  Scrapes discussion threads from the CD Projekt Red Cyberpunk 2077 forums. For each page, it
extracts the post number , time, the quoted text (if the reply includes a quote), the main reply text and the
number of reactions. The extracted records are then saved to a versioned table.
Usage:  Run the script and provide the forum URL, number of pages to scrape and a version number when
prompted:

```
python src/Fetch/specific_website_fetches/data_scraper_CDPR_Cyberpunk_Forum.py
```
The script uses requests  and BeautifulSoup . Ensure network access and adjust the database path
(Raw_Reviews.db ) within the save_to_database  function if necessary.
Expected  Output:  After  processing,  a  table  named
Cyberpunk_2077_Official_Forum_Reviews_<version>  is created in  Raw_Reviews.db  containing
columns for post number , time, quoted text, main text and upvote count. The script prints each extracted
record and confirms completion.

### data_scraper_Escape_from_Tarkov_Official_Forum.py
Purpose:  Scrapes the Escape from Tarkov official forum. It retrieves the thread page, extracts metadata for
the original post and all replies, including nested quotes, and writes the results into two tables: one for
posts and one for replies. It handles pagination and throttles requests to avoid HTTP 429 responses.
Usage:  This script is not fully parameterised. To use it, call extract_response_data(url)  with a forum
thread URL, then save the returned data via save_to_database(data) . You may wrap these calls in your
own loop over multiple URLs. Example:
from
src.Fetch.specific_website_fetches.data_scraper_Escape_from_Tarkov_Official_Forum
importextract_response_data ,save_to_database
data=extract_response_data ("https://forum.escapefromtarkov.com/topic/12345-
14

title/")
save_to_database (data)
Expected  Output:  Two  tables  ( escape_from_Tarkov_Official_Forum_Posts  and
escape_from_Tarkov_Official_Forum_Replies )  in  Raw_Reviews.db  with  columns  for  post  ID,
reply ID, title, times, author , main text and associated post links. The script prints counts of replies per page
and any errors.

### metacritic_scarper_wrapper.py
Purpose:  Automates scraping of multiple games on Metacritic by looping over a list of URLs and invoking
the interactive metacritic_scraper.py  script. It simulates user input to accept default dates.
Usage:  Populate the game_list  variable with Metacritic game URLs and run:

```
python src/Fetch/specific_website_fetches/metacritic_scarper_wrapper.py
```
Expected  Output:  For  each  URL,  the  underlying  scraper  creates  a  new  table  in  the  SQLite  database
(CS_Capstone.db ) with reviews filtered by date and language. The wrapper prints start and end markers
for each game.

### metacritic_scraper.py
Purpose:  Fetches all user reviews for a given game on Metacritic across all platforms. It uses Selenium to
drive a headless Chrome session, scrolls to load all reviews, applies date filtering (default three years if
start/end dates are y), filters out non‑English reviews via a heuristic and writes the results into a SQLite
table. A slugified version of the game name and date range is used as the table name.
Usage:  Ensure Selenium and a compatible ChromeDriver are installed and on your PATH. Run the script and
answer the prompts:

```
python src/Fetch/specific_website_fetches/metacritic_scraper.py
```
Enter the Metacritic game URL, the start date ( YYYY MM DD  or y), and the end date ( YYYY MM DD  or
y). For example, entering y for both uses the last three years of reviews ending today.
Expected  Output:  The  script  writes  a  new  table  into  CS_Capstone.db  with  a  name  like
<game_slug>_<start>_<end>_metacritic .  Columns  include  review  ID,  user  name,  platform,  date,
score, text and any helpfulness votes. The script prints progress as it scrolls through pages and inserts
records, and may output warnings if Selenium is not properly installed.
15

## Typical pipeline sequence
The scripts can be composed into a multi‑phase pipeline to reproduce the capstone analysis. A typical
end‑to‑end run might look like this:
Data collection:  Use the fetchers in src/Fetch  to scrape reviews from Steam
(steam_review_fetch.py  or Steam_fetch_wrapper.py ), Reddit ( reddit_reviews_agent
via run_local.sh ), Metacritic ( metacritic_scraper.py ), and official forums (scripts under 
specific_website_fetches ). The resulting tables reside in Raw_Reviews.db  or 
CS_Capstone.db .
Time filtering:  Run Time_filter.py  on the scraped database to produce a time‑filtered
database that keeps only the last two years of data.
Language and sentiment filtering:  Apply english_filter.py  to isolate English reviews, then
run filter_sentiment_pos_neg_v2.py  (or v1/emotions) to assign sentiment labels. Extract
negative‑only tables with extract_negative_combine_sentiment.py  if desired. Optionally run 
filter.py  to tag specific frustration categories.
Topic modelling:  Use LDA_topic_modeling_improved.py  or bertopic_sqlite.py  to model
the negative review corpus. Experiment with hyperparameter sweeps via run_improved_all.py .
Visualise and interpret the topics using Bertopic_visualization.py  or the various collation
scripts.
Post‑processing:  Run Analysis.py , game_topic_counts.py , labels_per_topic.py  and 
collate_improved_runs_to_excel.py  to summarise topic distributions across games and
platforms and to generate publication‑ready reports.
Final remarks
Because many of these scripts were developed for a research project, they often require manual editing of
constants (database paths, table names, API keys) before they can be run. Always review the configuration
variables near the top of each script and adjust them to fit your environment. In production use, consider
wrapping these scripts into functions or command‑line tools with proper argument parsing to reduce
manual intervention.


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
