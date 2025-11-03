#!/usr/bin/env python3
"""
Retro-label topics using KeyBERT, recursively, WITHOUT retraining LDA.

- Scans the given ROOT recursively for files named 'topics_top_words.json'
  (works with nested structures like dir/somefolder/topics_top_words.json).
- For each file, generates/updates the 'label' field for every topic.
- Preserves words/weights; only updates 'label'.
- Optional: MMR + diversity, backups, dry-run, and a progress bar.

Usage:
  python retro_label_topics.py /path/to/Results/runs_improved_YYYYMMDD_HHMMSS \
      --mmr --diversity 0.7 --backup

Install:
  pip install keybert
  # optional for better quality embeddings:
  pip install sentence-transformers
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

def _kw_model():
    try:
        from keybert import KeyBERT
    except Exception as e:
        raise SystemExit(
            "KeyBERT is not installed.\n"
            "Install with:\n  pip install keybert\n"
            "Optionally add better embeddings:\n  pip install sentence-transformers"
        ) from e
    return KeyBERT()

def load_topics(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))

def save_topics(path: Path, topics: List[Dict[str, Any]], backup: bool):
    if backup:
        bak = path.with_suffix(".json.bak")
        if not bak.exists():
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(topics, indent=2), encoding="utf-8")

def build_text_from_words(words: List[Dict[str, Any]], topn: int) -> str:
    return " ".join([w.get("term", "") for w in words[:topn] if isinstance(w, dict)])

def label_with_keybert(text: str, kw, ngram: Tuple[int,int], top_n: int, mmr: bool, diversity: float) -> str:
    try:
        res = kw.extract_keywords(
            text,
            keyphrase_ngram_range=ngram,
            stop_words="english",
            top_n=top_n,
            use_mmr=mmr,
            diversity=diversity if mmr else None
        )
        if not res:
            return ""
        # res: list of (phrase, score)
        best = max(res, key=lambda x: x[1])
        return best[0].replace(" ", "_")
    except Exception:
        return ""

def simple_label(words: List[Dict[str, Any]], num_terms: int = 3) -> str:
    return "_".join([w.get("term", "") for w in words[:num_terms] if isinstance(w, dict)]) if words else ""

def process_topics_file(path: Path, kw, args) -> bool:
    topics = load_topics(path)
    changed = False
    for t in topics:
        words = t.get("words", [])
        text = build_text_from_words(words, topn=args.topn)
        lbl = label_with_keybert(
            text=text,
            kw=kw,
            ngram=(args.ngram_min, args.ngram_max),
            top_n=args.candidates,
            mmr=args.mmr,
            diversity=args.diversity
        )
        if not lbl:  # fallback if KeyBERT fails for this topic
            lbl = simple_label(words, num_terms=min(3, args.topn))
        if t.get("label") != lbl:
            t["label"] = lbl
            changed = True

    if changed and not args.dry_run:
        save_topics(path, topics, backup=args.backup)
    return changed

def find_all_topics_json(root: Path, name: str = "topics_top_words.json"):
    # Recursively yield all matching files, ignore hidden dirs
    for p in root.rglob(name):
        # Skip hidden folders/files just in case
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p

def main():
    ap = argparse.ArgumentParser(description="Retro-label LDA topics (recursive) using KeyBERT.")
    ap.add_argument("root", help="Path to a batch folder (or any folder) that contains nested run folders")
    ap.add_argument("--topn", type=int, default=12, help="Top words to build the pseudo-document")
    ap.add_argument("--ngram-min", type=int, default=1, help="Min n-gram for KeyBERT")
    ap.add_argument("--ngram-max", type=int, default=3, help="Max n-gram for KeyBERT")
    ap.add_argument("--candidates", type=int, default=5, help="How many candidate phrases KeyBERT scores")
    ap.add_argument("--mmr", action="store_true", help="Use MMR diversification")
    ap.add_argument("--diversity", type=float, default=0.7, help="MMR diversity weight (only used if --mmr)")
    ap.add_argument("--backup", action="store_true", help="Write a .bak of the JSON before overwriting")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    files = list(find_all_topics_json(root))

    if not files:
        raise SystemExit(f"No topics_top_words.json found under: {root}")

    # Lazy import tqdm only if we need it
    try:
        from tqdm import tqdm
        pbar = tqdm(files, desc="Relabeling topics", unit="run")
        use_pbar = True
    except Exception:
        pbar = files
        use_pbar = False

    kw = None if args.dry_run else _kw_model()

    total = len(files)
    modified = 0
    for f in pbar:
        changed = process_topics_file(f, kw, args) if kw else False
        if changed:
            modified += 1
        if use_pbar:
            pbar.set_postfix_str(f"{modified}/{total} modified")

    print(f"\nDone. Modified {modified} of {total} run(s).")
    if args.dry_run:
        print("Dry-run was enabled: no files were written.")

if __name__ == "__main__":
    main()
