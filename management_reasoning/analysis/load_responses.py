"""Load management-reasoning response JSONL and pivot independent suites."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from management_reasoning.prompts import QUESTION_IDS, RESPONSE_FIELDS

RESPONSES_ROOT = "./results/management_reasoning/responses/vertex"


def responses_path(model_dir: str, arm: str, tag: str) -> str:
    return os.path.join(RESPONSES_ROOT, model_dir, f"{arm}_{tag}", "responses.jsonl")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _fields_from_parsed(parsed: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not parsed:
        return {}
    out: Dict[str, Any] = {}
    for field in RESPONSE_FIELDS.values():
        if field in parsed:
            out[field] = parsed[field]
    return out


def index_multiask(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """sample_id → {parse_ok, refusal, fields…} for multi-ask suites."""
    by_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sample_id"])
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else None
        by_id[sid] = {
            "sample_id": sid,
            "parse_ok": bool(row.get("parse_ok")),
            "refusal": bool(row.get("refusal")),
            "fields": _fields_from_parsed(parsed),
            "suite": row.get("suite"),
        }
    return by_id


def pivot_independent(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Merge 5 independent rows per sample into one field dict."""
    by_id: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        sid = int(row["sample_id"])
        qid = row.get("question_id")
        if qid not in RESPONSE_FIELDS:
            continue
        field = RESPONSE_FIELDS[qid]
        entry = by_id.setdefault(
            sid,
            {
                "sample_id": sid,
                "parse_ok": True,
                "refusal": False,
                "fields": {},
                "field_ok": {},
                "suite": "independent",
            },
        )
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else None
        ok = bool(row.get("parse_ok")) and parsed is not None and field in parsed
        entry["field_ok"][field] = ok
        if ok:
            entry["fields"][field] = parsed[field]
        if row.get("refusal"):
            entry["refusal"] = True
        # overall parse_ok: all five questions present and ok
    for entry in by_id.values():
        oks = [entry["field_ok"].get(RESPONSE_FIELDS[q], False) for q in QUESTION_IDS]
        entry["parse_ok"] = all(oks)
    return by_id


def load_suite(
    model_dir: str,
    *,
    arm: str,
    tag: str,
    independent: bool = False,
) -> Dict[int, Dict[str, Any]]:
    path = responses_path(model_dir, arm, tag)
    rows = load_jsonl(path)
    if independent:
        return pivot_independent(rows)
    return index_multiask(rows)


def load_claude_ablation_grid(model_dir: str = "claude-opus-4-5_20251101") -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Baseline + planned Claude ablation suites keyed by short label."""
    return {
        "raw": load_suite(model_dir, arm="raw", tag="primary_batch"),
        "neut": load_suite(model_dir, arm="neutralized", tag="primary_batch"),
        "indep": load_suite(model_dir, arm="raw", tag="independent_batch", independent=True),
        "ord1": load_suite(model_dir, arm="raw", tag="order_ord1_batch"),
        "ord2": load_suite(model_dir, arm="raw", tag="order_ord2_batch"),
        "ord3": load_suite(model_dir, arm="raw", tag="order_ord3_batch"),
    }


def load_independent_remove_all_grid(
    model_dir: str,
    *,
    reuse_raw_independent_tag: Optional[str] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Raw vs remove_all independent (MR single-question card).

    For Claude, pass ``reuse_raw_independent_tag="independent_batch"`` to use the
    prior Claude raw independent collect. Gemini uses ``independent_remove_all_batch``
    for both arms.
    """
    raw_tag = reuse_raw_independent_tag or "independent_remove_all_batch"
    return {
        "raw": load_suite(model_dir, arm="raw", tag=raw_tag, independent=True),
        "remove_all": load_suite(
            model_dir, arm="remove_all", tag="independent_remove_all_batch", independent=True
        ),
    }


def load_independent_new_neu_grid(
    model_dir: str,
    *,
    reuse_raw_independent_tag: Optional[str] = None,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Raw vs new_neu content+tone independent arms (ct_old, ct_new).

    Claude raw: ``reuse_raw_independent_tag="independent_batch"``.
    Gemini raw: ``reuse_raw_independent_tag="independent_remove_all_batch"``
    (same MR independent protocol). Default raw tag is independent_remove_all_batch.
    """
    raw_tag = reuse_raw_independent_tag or "independent_remove_all_batch"
    tag = "independent_new_neu_batch"
    return {
        "raw": load_suite(model_dir, arm="raw", tag=raw_tag, independent=True),
        "ct_old": load_suite(model_dir, arm="ct_old", tag=tag, independent=True),
        "ct_new": load_suite(model_dir, arm="ct_new", tag=tag, independent=True),
    }


def load_independent_factor_grid(
    model_dir: str,
    *,
    reuse_raw_independent_tag: Optional[str] = None,
    factor_tag: str = "independent_factor_batch",
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Raw vs paper factor-pair independent arms (format_tone, content_format).

    Claude raw: ``reuse_raw_independent_tag="independent_batch"``.
    Gemini raw: ``reuse_raw_independent_tag="independent_remove_all_batch"``.
    """
    raw_tag = reuse_raw_independent_tag or "independent_remove_all_batch"
    return {
        "raw": load_suite(model_dir, arm="raw", tag=raw_tag, independent=True),
        "format_tone": load_suite(
            model_dir, arm="format_tone", tag=factor_tag, independent=True
        ),
        "content_format": load_suite(
            model_dir, arm="content_format", tag=factor_tag, independent=True
        ),
    }


def load_independent_ra_new_grid(
    model_dir: str,
    *,
    reuse_raw_independent_tag: Optional[str] = None,
    ra_tag: str = "independent_ra_new_batch",
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Raw vs gpt-5.2 remove_all (``ra_new``) independent arms."""
    raw_tag = reuse_raw_independent_tag or "independent_remove_all_batch"
    return {
        "raw": load_suite(model_dir, arm="raw", tag=raw_tag, independent=True),
        "ra_new": load_suite(model_dir, arm="ra_new", tag=ra_tag, independent=True),
    }
