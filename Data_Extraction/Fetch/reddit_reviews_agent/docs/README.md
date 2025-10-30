# Reddit Game Reviews Agent (API-only)

Ingest Reddit posts and comments that look like player reviews/impressions for a predefined game list and store them in SQLite (one table per game). **No scraping**; uses the official Reddit API via PRAW.

## Quick Start

1. Create and fill `configs/reddit.example.yaml` with your Reddit OAuth creds and settings.
2. (Optional) Copy to `configs/reddit.yaml` and edit games/subreddits/time window.
3. Run:

```bash
./scripts/run_local.sh
```

This will create a virtualenv, install deps, and run the pipeline.

## Database

- SQLite path is set in the config (`project.db_path`); default: `Data_Extraction/Database/CS_Capstone.db`.
- For each game, a table `{game_slug}_reddit_game_reviews` is created with indexes.
- Idempotent upserts ensure safe re-runs (primary key on `id`).

## Heuristics (no sentiment)

- A post is kept if:
  - Title or flair matches the `review_title_or_flair_regex`, **or**
  - Body has at least `min_words_for_body_review` words **and** the game is matched.
- Comments are kept if English-like and word count ≥ `min_words_for_body_review` (configurable).
- English heuristic: ASCII ratio + minimum words.
- Platform and playtime are extracted via regex and stored as optional fields.

## Notes

- Add/adjust game aliases and subreddit lists in the YAML config.
- Respect Reddit API rate limits; tuning and retries are built-in.
- For megathreads, the search queries (title:"Game" + review keywords) usually surface them; all comments under kept posts are ingested.
