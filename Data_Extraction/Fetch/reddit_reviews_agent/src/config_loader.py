import yaml
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    data: Dict[str, Any]

    @classmethod
    def from_path(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)

    def get(self, dotted: str, default=None):
        cur = self.data
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
