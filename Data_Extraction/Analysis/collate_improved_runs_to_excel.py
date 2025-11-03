#!/usr/bin/env python3
"""
Collate the results of improved LDA runs into an Excel workbook.

This script expects a directory containing multiple run subdirectories. Each
run directory must contain at least a `metrics_summary.json` file. It will
also use additional files produced by the improved LDA script when
available:

  - `topics_top_words.json`: defines labels and top words per topic.
  - `doc_topic_weights.csv`: per-document topic mixture and dominant topic.
  - PNG images (e.g. `topic_sizes.png`, `intertopic_js_heatmap.png`,
    `topic_dendrogram.png`, `doc_scatter_umap.png`).  These are embedded
    into the Excel workbook if present.

For each run, a sheet is created with the following sections:

  1. Overview: number of docs, vocabulary size, coherence metrics,
     perplexity, diversity metrics, entropy, Gini, and configuration
     parameters (k, eta, drop_top_n, bigrams) parsed from the run
     directory name.

  2. Topic sizes: estimated counts and proportions as reported in
     `metrics_summary.json`.  If a `doc_topic_weights.csv` file exists,
     counts of documents per dominant topic are computed and added.

  3. Diversity: unique word ratio and average pairwise Jaccard.

  4. Topic words: a table of topic ID, optional label, and a comma-
     separated list of the top N words for that topic.  Uses
     `topics_top_words.json` when available.

  5. Plots: embeds any PNGs present in the run directory (preferring
     standard names), using a simple grid.

The script also generates a `Combined` sheet that collates one row per
run, capturing the overview metrics along with the configuration
parameters.

Usage:

  python collate_improved_runs_to_excel.py /path/to/runs_dir -o summary.xlsx

The runs directory should contain subdirectories named like
``k15_eta0.05_drop0_uni`` or similar, where the prefix `k{K}` is used to
determine the number of topics.  Additional configuration parameters
(`eta`, `drop{N}`, and a suffix indicating bigrams vs unigrams) will be
parsed from the directory name and included in the summary.

Dependencies: pandas, xlsxwriter (via pandas.ExcelWriter).
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# The candidate plot filenames in priority order.  If none are found
# among these names, the first available *.png is used.
PLOT_CANDIDATES = [
    "topic_sizes.png",
    "intertopic_js_heatmap.png",
    "doc_scatter_umap.png",
    "doc_scatter_pca.png",
    "topic_dendrogram.png",
]


def parse_run_name(name: str) -> Dict[str, Any]:
    """Parse run directory name to extract configuration parameters.

    Expected patterns include a prefix like `k15`, optionally followed by
    `_eta{value}`, `_drop{N}`, and a suffix indicating bigram usage
    (`_bi` or `_uni`).  Any missing component will be returned as None.

    Returns a dict with keys: k (int), eta (float or None), drop (int or
    None), bigrams (bool or None), raw (original name).
    """
    result: Dict[str, Any] = {"k": None, "eta": None, "drop": None, "bigrams": None, "raw": name}
    # Extract k
    m_k = re.match(r"k(\d+)", name)
    if m_k:
        result["k"] = int(m_k.group(1))
    # Extract eta
    m_eta = re.search(r"_eta([0-9.]+)", name)
    if m_eta:
        try:
            result["eta"] = float(m_eta.group(1))
        except ValueError:
            result["eta"] = m_eta.group(1)
    # Extract drop
    m_drop = re.search(r"_drop(\d+)", name)
    if m_drop:
        result["drop"] = int(m_drop.group(1))
    # Extract bigram tag
    if name.endswith("_bi"):
        result["bigrams"] = True
    elif name.endswith("_uni"):
        result["bigrams"] = False
    return result


def find_metrics_json(run_dir: Path) -> Optional[Path]:
    cand = run_dir / "metrics_summary.json"
    if cand.exists():
        return cand
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "topics_top_words.json":
            continue
        if p.stat().st_size > 0:
            return p
    return None


def find_topics_json(run_dir: Path) -> Optional[Path]:
    cand = run_dir / "topics_top_words.json"
    return cand if cand.exists() else None


def find_doc_topics_csv(run_dir: Path) -> Optional[Path]:
    cand = run_dir / "doc_topic_weights.csv"
    return cand if cand.exists() else None


def find_plots(run_dir: Path) -> List[Path]:
    # Return list of PNGs to embed; prefer known names, else any *.png.
    found = [run_dir / name for name in PLOT_CANDIDATES if (run_dir / name).exists()]
    if not found:
        found = sorted(run_dir.glob("*.png"))
    return found


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_overview_df(payload: Dict[str, Any], cfg: Dict[str, Any], src: Path) -> pd.DataFrame:
    rows = [
        ("Source file", str(src)),
        ("Run directory", cfg["raw"]),
        ("K (num_topics)", payload.get("num_topics")),
        ("Num docs", payload.get("num_docs")),
        ("Vocab size", payload.get("vocab_size")),
        ("Engine", payload.get("engine")),
        ("Eta", cfg.get("eta")),
        ("Drop top-N", cfg.get("drop")),
        ("Bigrams", cfg.get("bigrams")),
        ("Coherence c_v", safe_get(payload, ["coherence", "c_v"])),
        ("Coherence c_npmi", safe_get(payload, ["coherence", "c_npmi"])),
        ("Coherence u_mass", safe_get(payload, ["coherence", "u_mass"])),
        ("Perplexity", payload.get("perplexity")),
        ("Unique word ratio", safe_get(payload, ["topic_diversity", "unique_word_ratio"])),
        ("Avg pairwise Jaccard", safe_get(payload, ["topic_diversity", "avg_pairwise_jaccard"])),
        ("Entropy (bits)", safe_get(payload, ["topic_size_proxy", "entropy_bits"])),
        ("Gini", safe_get(payload, ["topic_size_proxy", "gini"])),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def build_topic_sizes_df(payload: Dict[str, Any], doc_counts: Optional[Dict[int, int]] = None) -> pd.DataFrame:
    tsp = payload.get("topic_size_proxy", {}) or {}
    counts = tsp.get("counts_estimated", []) or []
    props = tsp.get("proportions", []) or []
    k = len(counts)
    df = pd.DataFrame({
        "Topic": list(range(k)),
        "Estimated_Count": counts,
        "Estimated_Prop": props
    })
    if doc_counts:
        actual_counts = [doc_counts.get(i, 0) for i in range(k)]
        total = sum(actual_counts) or 1
        actual_props = [c / total for c in actual_counts]
        df["Actual_Count"] = actual_counts
        df["Actual_Prop"] = actual_props
    if not df.empty:
        if "Estimated_Prop" in df:
            df["Estimated_%"] = (df["Estimated_Prop"] * 100).round(3)
        if "Actual_Prop" in df:
            df["Actual_%"] = (df["Actual_Prop"] * 100).round(3)
        df = df.sort_values("Estimated_Count", ascending=False).reset_index(drop=True)
    return df


def build_diversity_df(payload: Dict[str, Any]) -> pd.DataFrame:
    td = payload.get("topic_diversity", {}) or {}
    return pd.DataFrame([
        {"Metric": "unique_word_ratio", "Value": td.get("unique_word_ratio")},
        {"Metric": "avg_pairwise_jaccard", "Value": td.get("avg_pairwise_jaccard")}
    ])


def build_topics_df(topics_data: List[Dict[str, Any]], max_words: int = 10) -> pd.DataFrame:
    rows = []
    for entry in topics_data:
        topic_id = entry.get("topic")
        label = entry.get("label")
        # join up to max_words words into a comma-separated string
        words_list = entry.get("words", [])
        top_terms = ", ".join([w["term"] for w in words_list[:max_words]])
        rows.append({"Topic": topic_id, "Label": label, "Top_Terms": top_terms})
    df = pd.DataFrame(rows)
    df.sort_values("Topic", inplace=True)
    return df


def count_dominant_topics(doc_csv: Path) -> Dict[int, int]:
    """Read doc_topic_weights.csv and count how many docs are assigned each dominant topic.

    Only the 'dominant_topic' column is needed; reading with pandas is
    efficient.  Returns a dict mapping topic ID to count.
    """
    counts: Dict[int, int] = {}
    try:
        df = pd.read_csv(doc_csv, usecols=["dominant_topic"])
        counts = df["dominant_topic"].value_counts().to_dict()
    except Exception:
        # fallback if file or column missing
        return {}
    return {int(k): int(v) for k, v in counts.items()}


def auto_cols(writer: pd.ExcelWriter, sheet_name: str, max_cols: int = 50, base_width: int = 20):
    """Auto-size columns up to max_cols by setting a fixed width."""
    ws = writer.sheets[sheet_name]
    for c in range(max_cols):
        ws.set_column(c, c, base_width)


def insert_images_grid(writer: pd.ExcelWriter, sheet_name: str, images: List[Path], start_row: int,
                       start_col: int = 0, max_per_row: int = 2, scale: float = 0.6) -> int:
    """Insert images in a grid on the given sheet starting at (start_row, start_col).
    Returns the next free row after inserting the images.
    """
    ws = writer.sheets[sheet_name]
    row = start_row
    col = start_col
    count_in_row = 0
    for img in images:
        try:
            ws.insert_image(row, col, str(img), {"x_scale": scale, "y_scale": scale})
        except Exception as e:
            ws.write(row, col, f"Could not insert image {img.name}: {e}")
        col += 8  # shift columns by 8 (approx width) for next image
        count_in_row += 1
        if count_in_row >= max_per_row:
            row += int(22 * scale * 2)
            col = start_col
            count_in_row = 0
    if count_in_row != 0:
        row += int(22 * scale * 2)
    return row


def build_combined_row(payload: Dict[str, Any], cfg: Dict[str, Any], src: Path) -> Dict[str, Any]:
    return {
        "Run directory": cfg["raw"],
        "K": cfg.get("k", payload.get("num_topics")),
        "Eta": cfg.get("eta"),
        "Drop top-N": cfg.get("drop"),
        "Bigrams": cfg.get("bigrams"),
        "Num docs": payload.get("num_docs"),
        "Vocab size": payload.get("vocab_size"),
        "Engine": payload.get("engine"),
        "c_v": safe_get(payload, ["coherence", "c_v"]),
        "c_npmi": safe_get(payload, ["coherence", "c_npmi"]),
        "u_mass": safe_get(payload, ["coherence", "u_mass"]),
        "Perplexity": payload.get("perplexity"),
        "Unique word ratio": safe_get(payload, ["topic_diversity", "unique_word_ratio"]),
        "Avg Jaccard": safe_get(payload, ["topic_diversity", "avg_pairwise_jaccard"]),
        "Entropy (bits)": safe_get(payload, ["topic_size_proxy", "entropy_bits"]),
        "Gini": safe_get(payload, ["topic_size_proxy", "gini"]),
    }


def collate_runs(root: Path, out_xlsx: Path):
    rows_combined: List[Dict[str, Any]] = []
    # Collect all subdirectories that look like run directories (start with 'k' and contain metrics)
    run_dirs = []
    for p in root.iterdir():
        if p.is_dir() and re.match(r"k\d+", p.name):
            if find_metrics_json(p) is not None:
                run_dirs.append(p)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with metrics found under {root}")

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        # Process each run
        for run_dir in sorted(run_dirs, key=lambda d: d.name):
            run_name = run_dir.name
            cfg = parse_run_name(run_name)
            mjson_path = find_metrics_json(run_dir)
            if mjson_path is None:
                continue
            payload = json.loads(mjson_path.read_text(encoding="utf-8"))
            rows_combined.append(build_combined_row(payload, cfg, mjson_path))
            sheet_name = run_name[:31]  # Excel sheet names limited to 31 chars
            # Build sections
            overview_df = build_overview_df(payload, cfg, mjson_path)
            # Get doc-topic counts if file exists
            doc_csv = find_doc_topics_csv(run_dir)
            doc_counts = count_dominant_topics(doc_csv) if doc_csv else None
            topic_sizes_df = build_topic_sizes_df(payload, doc_counts)
            diversity_df = build_diversity_df(payload)
            # Load topics file
            topics_json_path = find_topics_json(run_dir)
            topics_df = None
            if topics_json_path and topics_json_path.exists():
                topics_data = json.loads(topics_json_path.read_text(encoding="utf-8"))
                topics_df = build_topics_df(topics_data, max_words=10)
            # Write to Excel
            row_cursor = 0
            overview_df.to_excel(writer, sheet_name=sheet_name, startrow=row_cursor, index=False)
            row_cursor += len(overview_df) + 2
            topic_sizes_df.to_excel(writer, sheet_name=sheet_name, startrow=row_cursor, index=False)
            row_cursor += len(topic_sizes_df) + 2
            diversity_df.to_excel(writer, sheet_name=sheet_name, startrow=row_cursor, index=False)
            row_cursor += len(diversity_df) + 2
            if topics_df is not None:
                topics_df.to_excel(writer, sheet_name=sheet_name, startrow=row_cursor, index=False)
                row_cursor += len(topics_df) + 2
            # Plots
            images = find_plots(run_dir)
            writer.sheets[sheet_name].write(row_cursor, 0, "Plots")
            row_cursor += 1
            if images:
                row_cursor = insert_images_grid(writer, sheet_name, images, start_row=row_cursor, start_col=0, max_per_row=2, scale=0.6)
            else:
                writer.sheets[sheet_name].write(row_cursor, 0, "No plot images found.")
                row_cursor += 2
            auto_cols(writer, sheet_name)
        # Combined sheet
        combined_df = pd.DataFrame(rows_combined)
        # Sort by K and other parameters for easier comparison
        combined_df.sort_values(["K", "Eta", "Drop top-N", "Bigrams"], inplace=True)
        combined_df.to_excel(writer, sheet_name="Combined", index=False)
        auto_cols(writer, "Combined")


def main():
    ap = argparse.ArgumentParser(description="Collate improved LDA runs into an Excel workbook.")
    ap.add_argument("root", type=str, help="Path to the directory containing improved run subfolders")
    ap.add_argument("-o", "--output", type=str, default="lda_improved_summary.xlsx", help="Output XLSX file path")
    args = ap.parse_args()
    root = Path(args.root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    collate_runs(root, out)
    print(f"✅ Wrote Excel: {out}")


if __name__ == "__main__":
    main()