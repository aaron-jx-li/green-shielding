"""Shared config loader for management_reasoning (models.yaml + env)."""

from __future__ import annotations

import os
from typing import Dict, Optional

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_YAML = os.path.join(_PACKAGE_DIR, "models.yaml")


def load_models_yaml(path: Optional[str] = None) -> Dict[str, str]:
    """Minimal YAML loader for the flat ``vertex:`` block (no PyYAML required)."""
    path = path or DEFAULT_MODELS_YAML
    out: Dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    in_vertex = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.strip().startswith("#") or not raw.strip():
                continue
            if raw.startswith("vertex:"):
                in_vertex = True
                continue
            if in_vertex and raw and not raw.startswith(" ") and not raw.startswith("\t"):
                in_vertex = False
            if in_vertex and ":" in raw:
                key, _, val = raw.strip().partition(":")
                out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def models_config() -> Dict[str, str]:
    return load_models_yaml(DEFAULT_MODELS_YAML)
