"""Config loading. Everything tunable lives in config.yaml, not in code."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    raw: dict

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, dotted: str, default=None):
        """cfg.get('llm.model') — dotted lookup so callers stay readable."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default=None) -> Optional[Path]:
        """A config value that names a directory, resolved against the project root."""
        v = self.get(dotted, default)
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() else ROOT / p


def load(path: Optional[Path] = None) -> Config:
    p = Path(path) if path else ROOT / "config.yaml"
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p) as f:
        return Config(yaml.safe_load(f))
