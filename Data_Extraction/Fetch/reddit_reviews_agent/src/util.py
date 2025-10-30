import hashlib
import time
from datetime import datetime, timezone
from typing import Dict, Any

def now_epoch() -> int:
    return int(time.time())

def cfg_hash(d: Dict[str, Any]) -> str:
    s = repr(sorted(list(_flatten(d).items()))).encode("utf-8")
    return hashlib.sha1(s).hexdigest()

def _flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def within_window(epoch_ts: int, start_ts: int, end_ts: int) -> bool:
    return start_ts <= epoch_ts <= end_ts
