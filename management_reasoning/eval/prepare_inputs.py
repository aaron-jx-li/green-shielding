#!/usr/bin/env python3
"""Build open_eval-style input rows from management_reasoning response JSONL.

Joins by ``sample_id`` (batch JSONL is unordered). Uses ``parsed.diagnosis`` as
``model_response`` (falls back to JSON diagnosis in raw_response).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array: {path}")
    return data


def extract_diagnosis_text(record: Dict[str, Any]) -> Optional[str]:
    """Pull diagnosis string from management_reasoning target output."""
    parsed = record.get("parsed")
    if isinstance(parsed, dict):
        dx = parsed.get("diagnosis")
        if isinstance(dx, str) and dx.strip():
            return dx.strip()
        if parsed.get("refusal") is True:
            return None

    raw = record.get("raw_response")
    if isinstance(raw, str) and raw.strip():
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            # Fence / partial: try first object via simple brace scan
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    obj = None
            else:
                obj = None
        if isinstance(obj, dict):
            dx = obj.get("diagnosis")
            if isinstance(dx, str) and dx.strip():
                return dx.strip()
    return None


def extract_full_response_text(record: Dict[str, Any]) -> Optional[str]:
    """Use free-form model text (legacy diagnosis-only runs)."""
    for key in ("model_response", "raw_response"):
        text = record.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def build_eval_inputs(
    *,
    responses_jsonl: str,
    cohort_json: str,
    arm: str,
    require_parse_ok: bool = True,
    answer_mode: str = "diagnosis",
    question_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return a list aligned with cohort order (index == sample_id when cohort is
    prepared that way). Missing responses become empty model_response.

    ``answer_mode``:
      - ``diagnosis``: MR structured ``parsed.diagnosis`` (default)
      - ``full_response``: entire free-form answer text

    ``question_filter``: if set (e.g. ``\"dx\"``), only index independent JSONL
    rows whose ``question_id`` matches (avoids last-row overwrite).
    """
    if answer_mode not in ("diagnosis", "full_response"):
        raise ValueError(f"unknown answer_mode: {answer_mode}")

    cohort = _load_json_array(cohort_json)
    by_id: Dict[int, Dict[str, Any]] = {}
    for row in _load_jsonl(responses_jsonl):
        if question_filter is not None and row.get("question_id") != question_filter:
            continue
        sid = row.get("sample_id")
        if isinstance(sid, int):
            by_id[sid] = row

    out: List[Dict[str, Any]] = []
    for i, sample in enumerate(cohort):
        sid = int(sample.get("sample_id", i))
        raw_input = (sample.get("raw_input") or "").strip()
        # Always score against the original patient inquiry for P/H/C grounding,
        # matching open_eval's use of raw_input as QUESTION.
        rec = by_id.get(sid)
        model_response = ""
        parse_ok = False
        refusal = False
        if rec is not None:
            parse_ok = bool(rec.get("parse_ok"))
            refusal = bool(rec.get("refusal"))
            if (not require_parse_ok) or parse_ok:
                if answer_mode == "full_response":
                    model_response = extract_full_response_text(rec) or ""
                else:
                    dx = extract_diagnosis_text(rec)
                    model_response = dx or ""
        out.append(
            {
                "sample_id": sid,
                "index": i,
                "arm": arm,
                "raw_input": raw_input,
                "neutralized_prompt": sample.get("neutralized_prompt"),
                "reference_diagnosis": sample.get("reference_diagnosis"),
                "model_response": model_response,
                "parse_ok": parse_ok,
                "refusal": refusal,
                "has_target_response": bool(model_response),
                "answer_mode": answer_mode,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--responses_jsonl", required=True)
    ap.add_argument(
        "--cohort_json",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    ap.add_argument(
        "--arm",
        choices=("raw", "neutralized", "remove_all", "ct_old", "ct_new"),
        required=True,
    )
    ap.add_argument("--out_path", required=True)
    ap.add_argument(
        "--answer_mode",
        choices=("diagnosis", "full_response"),
        default="diagnosis",
    )
    ap.add_argument(
        "--allow_parse_fail",
        action="store_true",
        help="Still try to extract diagnosis when parse_ok=false",
    )
    args = ap.parse_args()

    rows = build_eval_inputs(
        responses_jsonl=args.responses_jsonl,
        cohort_json=args.cohort_json,
        arm=args.arm,
        require_parse_ok=not args.allow_parse_fail,
        answer_mode=args.answer_mode,
    )
    n_ok = sum(1 for r in rows if r["has_target_response"])
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    label = "answers" if args.answer_mode == "full_response" else "diagnosis"
    print(f"Wrote {len(rows)} rows ({n_ok} with {label}) → {args.out_path}")


if __name__ == "__main__":
    main()
