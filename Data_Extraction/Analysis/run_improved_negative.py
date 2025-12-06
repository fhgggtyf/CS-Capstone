#!/usr/bin/env python3
"""
Run a sweep of the enhanced LDA topic modelling script over a grid of
hyperparameters.  This orchestrator is similar to the original
``run.py`` but targets ``LDA_topic_modeling_improved.py`` and exposes
additional knobs (``eta``, ``drop_top_n``, bigram usage, etc.) to play
with.  Each run writes its metrics, plots and exported files into a
unique subdirectory under ``Results`` so you can compare them later.

Adjust the parameter lists (``k_values``, ``eta_values``, etc.) at the
top of the file to control the sweep.  Any combination of values
produces one run.  Unused parameters (e.g. no bigrams) result in
modifiers being omitted from the command line.
"""

import sys
import subprocess
import time
from pathlib import Path

# Project root: 2 levels up from this file (Analysis → Data_Extraction → <root>)
ROOT = Path(__file__).resolve().parents[2]

# Paths
# If your script has a different name, update it here:
SCRIPT = ROOT / "Data_Extraction" / "Analysis" / "LDA_topic_modeling_improved.py"
DB = ROOT / "Data_Extraction" / "Database" / "CS_Capstone_Sentiment_time_filtered.db"

# Name of the table and text column in your database.  Modify as
# necessary for your dataset.
TABLE = "frustrated_sentiment_pos_neg_v2_sentiment_combined_english_only_negative_only"
TEXT_COL = "main_text"

# Time-stamped output root.  All run subdirectories will be created
# inside here.  Each subdirectory name encodes the parameter
# combination used for the run.
ts = time.strftime("%Y%m%d_%H%M%S")
title = "lda_neg"
OUTROOT = ROOT / "Data_Extraction" / "Analysis" / "Results" / f"runs_improved_{title}_{ts}"
OUTROOT.mkdir(parents=True, exist_ok=True)

# Common command-line arguments passed to every invocation of the LDA
# script.  Modify these to set the defaults for your experiment (e.g.
# passes, iters, batch size).  The values chosen here mirror those in
# the original run script.
common_args = [
    str(SCRIPT),
    "--db", str(DB),
    "--table", TABLE,
    "--text-col", TEXT_COL,
    "--workers", "8",
    "--passes", "15",
    "--iters", "600",
    "--chunksize", "4000",
    "--min-df", "20",
    "--max-df", "0.4",
    "--batch-size", "10000",
    "--coherence-sample", "20000",
    "--pyldavis-sample", "20000",
    "--scatter-sample", "25000",
    "--assign-themes", "simple",
    "--output-dir", "outputs_neg"
]

# Parameter grids.  Tweak these lists to explore different settings.
# Each combination of values produces a separate run.  For example,
# with three K values and two eta values, you'll get six runs.  If you
# only want to vary K, set the other lists to single-element lists.
k_values = list(range(10, 21))      # Number of topics to try (2 through 20)
eta_values = [0.1, 0.05, 0.02]    # Dirichlet prior for topic-word sparsity
drop_topn_values = [0, 50]         # How many of the most frequent tokens to drop
bigram_options = [False, True]     # Whether to enable bigram detection
alpha_value = "asymmetric"         # Use asymmetric prior for doc-topic

# Loop over all parameter combinations
for k in k_values:
    for eta in eta_values:
        for drop_top_n in drop_topn_values:
            for use_bigrams in bigram_options:
                # Compose a unique directory name for this run
                bigram_tag = "bi" if use_bigrams else "uni"
                dir_name = f"k{k}_eta{eta}_drop{drop_top_n}_{bigram_tag}"
                run_dir = OUTROOT / dir_name
                run_dir.mkdir(parents=True, exist_ok=True)
                log_file = run_dir / "run.log"

                # Construct the command line for this run.  We always
                # specify K, eta, alpha and drop-top-n.  Bigram
                # detection is toggled via the presence of
                # "--use-bigrams".
                cmd = [sys.executable] + common_args + [
                    "--k", str(k),
                    "--alpha", str(alpha_value),
                    "--eta", str(eta) if eta is not None else "auto",
                    "--drop-top-n", str(drop_top_n),
                    "--seed", str(1000 + k)  # ensure reproducibility across runs
                ]
                if use_bigrams:
                    cmd.append("--use-bigrams")

                # Informative message on stdout
                print(f">>> Running improved LDA: K={k}, eta={eta}, drop_top_n={drop_top_n}, bigrams={use_bigrams}")
                print(" ".join(cmd))

                # Launch the process and capture stdout/stderr to a log
                with open(log_file, "w") as log:
                    try:
                        subprocess.run(
                            cmd,
                            cwd=SCRIPT.parent,  # run relative to the script location
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True
                        )
                    except subprocess.CalledProcessError:
                        print(f"[ERROR] run {dir_name} failed (see {log_file})")
                        continue

                # Move contents of the outputs directory into this run's directory
                out_outputs = SCRIPT.parent / "outputs_neg"
                if out_outputs.exists():
                    for f in out_outputs.iterdir():
                        # Use rename to move the file; if destination exists, overwrite
                        target = run_dir / f.name
                        if target.exists():
                            target.unlink()
                        f.rename(target)

# Summary message
print(f"\nAll runs completed. Results stored in {OUTROOT}")