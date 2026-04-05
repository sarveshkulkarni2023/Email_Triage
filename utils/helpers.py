"""
Utility helpers shared across the project.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)


def get_env_var(name: str, default: Optional[str] = None) -> str:
    """Read an environment variable, raising if missing and no default."""
    value = os.getenv(name, default)
    if value is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a float to [lo, hi]."""
    return max(lo, min(hi, value))
