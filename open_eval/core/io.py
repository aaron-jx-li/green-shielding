import json
from typing import Any, Dict, List, Optional


def load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return obj


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_existing_output(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "per_sample" in obj and isinstance(obj["per_sample"], list):
            return obj
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[resume] Failed to read existing output {path}: {e}")
    return None


def build_done_index_set(per_sample: List[Dict[str, Any]]) -> set:
    done = set()
    for r in per_sample:
        if not isinstance(r, dict):
            continue
        idx = r.get("index", None)
        if isinstance(idx, int):
            done.add(idx)
    return done
