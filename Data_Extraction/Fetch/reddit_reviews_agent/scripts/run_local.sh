#!/usr/bin/env bash
set -euo pipefail

# --- LOGGING SETUP (captures ALL output from this point on) ---
mkdir -p logs
LOG="logs/run_$(date +%F_%H-%M-%S).log"
export PYTHONUNBUFFERED=1  # make Python unbuffered so logs flush immediately

# mirror ALL stdout + stderr to both terminal and log file
exec > >(tee -a "$LOG") 2>&1

echo "[INFO] Logging to $LOG"

# --- LOG ROTATION: keep only the most recent N logs (including current) ---
KEEP=10
if ls logs/run_*.log >/dev/null 2>&1; then
  # shellcheck disable=SC2207
  FILES=( $(ls -1t logs/run_*.log) )   # newest first
  COUNT=${#FILES[@]}
  if (( COUNT > KEEP )); then
    TO_DELETE=( "${FILES[@]:KEEP}" )
    echo "[INFO] Log rotation: deleting ${#TO_DELETE[@]} old logs"
    rm -f "${TO_DELETE[@]}"
  else
    echo "[INFO] Log rotation: nothing to delete (have $COUNT, keep $KEEP)"
  fi
else
  echo "[INFO] No previous logs to rotate"
fi
# --------------------------------------------------------------------------

# Create and activate virtual environment
echo "[INFO] Creating virtual environment"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
echo "[INFO] Virtual environment activated"

# If stdbuf exists, use it to force line-buffered output for pip (better live logs)
if command -v stdbuf >/dev/null 2>&1; then
  STDBUF="stdbuf -oL -eL"
  echo "[INFO] stdbuf found; enabling line-buffered output for pip"
else
  STDBUF=""
  echo "[INFO] stdbuf not found; continuing without line-buffering for pip"
fi

# Install dependencies (logged live)
echo "[INFO] Installing dependencies from requirements.txt"
$STDBUF pip install -r Data_Extraction/Fetch/reddit_reviews_agent/requirements.txt

# Run the Reddit ingestion
echo "[INFO] Starting reddit_ingest.py"
python Data_Extraction/Fetch/reddit_reviews_agent/reddit_ingest.py --config Data_Extraction/Fetch/reddit_reviews_agent/configs/reddit.example.yaml
EXIT_CODE=$?

echo "[INFO] Completed at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE

