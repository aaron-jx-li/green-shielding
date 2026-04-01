"""Shared JSON I/O helpers."""

import json
import os
import tempfile
from typing import Any


def atomic_json_dump(obj: Any, path: str) -> None:
    """
    Atomic-ish save: write to a temp file on the same filesystem then replace.
    Prevents corrupt JSON if the job is interrupted mid-write.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
