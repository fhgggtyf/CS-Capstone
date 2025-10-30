#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
from typing import Optional

import requests
# Optional: silence the LibreSSL warning you saw
try:
    from urllib3.exceptions import NotOpenSSLWarning
    import warnings
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    pass

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def default_name_from_url(url: str) -> str:
    p = urlparse(url)
    host = (p.netloc or "site").replace(":", "_")
    path = (p.path or "/").strip("/").replace("/", "_") or "index"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{host}__{path}__{ts}.html"

def derive_filename(url: str, out: Optional[str]) -> Path:
    if out:
        out_path = Path(out)
        return out_path / default_name_from_url(url) if out_path.is_dir() else out_path
    return Path(default_name_from_url(url))

def fetch_raw_html(url: str, timeout: int = 30, verify_tls: bool = True) -> bytes:
    headers = {
        "User-Agent": DEFAULT_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }
    with requests.Session() as s:
        s.headers.update(headers)
        r = s.get(url, timeout=timeout, allow_redirects=True, verify=verify_tls)
        r.raise_for_status()
        return r.content  # original bytes

def main():
    ap = argparse.ArgumentParser(description="Fetch a URL and save the original HTML bytes to a file.")
    ap.add_argument("url", help="Website URL to fetch")
    ap.add_argument("-o", "--out", help="Output file path (or directory). If a directory, a name is auto-generated.")
    ap.add_argument("--timeout", type=int, default=30, help="Request timeout (seconds)")
    ap.add_argument("--insecure", action="store_true", help="Skip TLS verification (not recommended)")
    args = ap.parse_args()

    out_path = derive_filename(args.url, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        html_bytes = fetch_raw_html(args.url, timeout=args.timeout, verify_tls=not args.insecure)
    except requests.RequestException as e:
        raise SystemExit(f"[error] Failed to fetch {args.url}: {e}")

    with open(out_path, "wb") as f:
        f.write(html_bytes)

    print(f"[ok] Saved: {out_path.resolve()}")

if __name__ == "__main__":
    main()
