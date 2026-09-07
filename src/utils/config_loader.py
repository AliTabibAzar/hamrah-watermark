"""Loads config/settings.yaml once and reuses it everywhere."""

from functools import lru_cache
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_settings(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else _DEFAULT_PATH
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)
