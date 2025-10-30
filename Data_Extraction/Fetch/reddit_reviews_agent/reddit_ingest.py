#!/usr/bin/env python3
import argparse, logging, json, os, sys, hashlib
from src.config_loader import Config
from src.pipeline import collect

def main():
    ap = argparse.ArgumentParser(description="Reddit Reviews Ingestion (API-only)")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = Config.from_path(args.config).data

    level = cfg.get("runtime", {}).get("log_level", "INFO")
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s")

    logging.info("Starting collection run...")
    collect(cfg)
    logging.info("Done.")

if __name__ == "__main__":
    main()
