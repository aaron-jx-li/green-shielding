from __future__ import annotations
import json
from typing import Any, Dict, List, Optional, Tuple
from datasets import Dataset

ALPHABET = [chr(65 + i) for i in range(26)]  # ["A".."Z"]


def letters_for(n: int) -> List[str]:
    if n < 1:
        raise ValueError("Need at least 1 choice.")
    if n > 26:
        raise ValueError("This helper supports up to 26 choices (A..Z).")
    return ALPHABET[:n]


def sanitize_options(
    opt_dict: Dict[str, Any], answer_letter: Optional[str]
) -> Tuple[Optional[List[str]], Optional[int]]:
    """Keep only non-empty options, return (choices, solution_index)."""
    ordered_keys = sorted(opt_dict.keys())
    clean_keys = [
        k for k in ordered_keys if isinstance(opt_dict[k], str) and opt_dict[k].strip() != ""
    ]
    if not clean_keys:
        return None, None
    choices = [opt_dict[k].strip() for k in clean_keys]
    al = (answer_letter or "").strip().upper()
    if al not in clean_keys:
        return None, None
    sol_idx = clean_keys.index(al)
    return choices, sol_idx


def _read_json_list(path: str) -> List[dict]:
    with open(path, "r") as f:
        return json.load(f)


def get_dataset(task: str) -> Dataset:
    if task == "medqa_diag":
        data = _read_json_list("./data/MedQA_ED_diagnosis.json")
        return Dataset.from_list(data)
    if task == "medxpertqa_diag":
        data = _read_json_list("./data/medxpertqa_diag.json")
        return Dataset.from_list(data)
    raise NotImplementedError(f"Data loading function not implemented for task: {task}")
