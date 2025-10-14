"""YAML config loading helper."""
from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

