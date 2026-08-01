"""Build Vertex Batch request JSONL for Flash-Lite diagnosis judges."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from open_eval.eval.metrics import extract_P_H_C_from_truth_record

from management_reasoning.eval.batch.paths import (
    DEFAULT_COHORT,
    DEFAULT_TRUTH,
    answer_mode_for,
    local_pair_index,
    local_requests,
    make_custom_id,
    question_filter_for,
    response_path_for,
)
from management_reasoning.eval.prepare_inputs import build_eval_inputs
from management_reasoning.eval.prompts import (
    DX_EXTRACT_SYSTEM,
    GROUNDING_SYSTEM,
    SEM_MATCH_BATCH_SYSTEM,
    UNCERTAINTY_SYSTEM,
)
from management_reasoning.eval.text import fuzzy_match, normalize_text

EXTRACT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "extracted_diagnoses": {"type": "array", "items": {"type": "string"}},
        "top_k_diagnoses": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["extracted_diagnoses", "top_k_diagnoses"],
    "additionalProperties": False,
}

UNC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"uncertainty_flag": {"type": "boolean"}},
    "required": ["uncertainty_flag"],
    "additionalProperties": False,
}

SEM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"matches": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["matches"],
    "additionalProperties": False,
}

# Grounding schema is nested; omit strict schema — prompt + JSON mime type.
GROUND_SCHEMA: Optional[Dict[str, Any]] = None


def _gemini_judge_line(
    *,
    custom_id: str,
    system: str,
    user: str,
    thinking_level: str = "HIGH",
    json_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_level": thinking_level},
    }
    if json_schema is not None:
        generation_config["response_json_schema"] = json_schema
    return {
        "custom_id": custom_id,
        "request": {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "system_instruction": {"parts": [{"text": system}]},
            "generation_config": generation_config,
        },
    }


def _load_json_array(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"expected JSON array: {path}")
    return data


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _eval_rows(
    *,
    suite: str,
    target: str,
    arm: str,
    cohort_json: str,
    start_idx: int,
    end_idx: int,
) -> List[Dict[str, Any]]:
    responses = response_path_for(suite, target, arm)
    if not os.path.isfile(responses):
        raise FileNotFoundError(f"Missing target responses: {responses}")
    rows = build_eval_inputs(
        responses_jsonl=responses,
        cohort_json=cohort_json,
        arm=arm,
        require_parse_ok=True,
        answer_mode=answer_mode_for(suite),
        question_filter=question_filter_for(suite),
    )
    return rows[start_idx : end_idx + 1]


def prepare_extract(
    *,
    suite: str,
    target: str,
    arm: str,
    start_idx: int = 0,
    end_idx: int = 0,
    cohort_json: str = DEFAULT_COHORT,
    top_k: int = 8,
    thinking_level: str = "HIGH",
) -> str:
    rows = _eval_rows(
        suite=suite,
        target=target,
        arm=arm,
        cohort_json=cohort_json,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    out_rows: List[Dict[str, Any]] = []
    for sample in rows:
        sid = int(sample["sample_id"])
        q = (sample.get("raw_input") or "").strip()
        ans = (sample.get("model_response") or "").strip()
        if not q or not ans:
            continue
        user = (
            f"QUESTION:\n{q}\n\nMODEL_ANSWER:\n{ans}\n\nTOP_K:\n{top_k}\n\n"
            "Return STRICT JSON."
        )
        out_rows.append(
            _gemini_judge_line(
                custom_id=make_custom_id("extract", target, arm, sid),
                system=DX_EXTRACT_SYSTEM,
                user=user,
                thinking_level=thinking_level,
                json_schema=EXTRACT_SCHEMA,
            )
        )
    path = local_requests(suite, target, arm, "extract")
    _write_jsonl(path, out_rows)
    return path


def prepare_unc(
    *,
    suite: str,
    target: str,
    arm: str,
    start_idx: int = 0,
    end_idx: int = 0,
    cohort_json: str = DEFAULT_COHORT,
    thinking_level: str = "HIGH",
) -> str:
    rows = _eval_rows(
        suite=suite,
        target=target,
        arm=arm,
        cohort_json=cohort_json,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    out_rows: List[Dict[str, Any]] = []
    for sample in rows:
        sid = int(sample["sample_id"])
        q = (sample.get("raw_input") or "").strip()
        ans = (sample.get("model_response") or "").strip()
        if not q or not ans:
            continue
        user = f"QUESTION:\n{q}\n\nMODEL_ANSWER:\n{ans}\n\nReturn STRICT JSON."
        out_rows.append(
            _gemini_judge_line(
                custom_id=make_custom_id("unc", target, arm, sid),
                system=UNCERTAINTY_SYSTEM,
                user=user,
                thinking_level=thinking_level,
                json_schema=UNC_SCHEMA,
            )
        )
    path = local_requests(suite, target, arm, "unc")
    _write_jsonl(path, out_rows)
    return path


def _sem_key(a: str, b: str) -> str:
    a_n, b_n = normalize_text(a), normalize_text(b)
    return f"{a_n}|||{b_n}" if a_n <= b_n else f"{b_n}|||{a_n}"


def prepare_sem(
    *,
    suite: str,
    target: str,
    arm: str,
    extract_collected: str,
    truth_path: str = DEFAULT_TRUTH,
    max_pairs_per_call: int = 50,
    thinking_level: str = "HIGH",
) -> str:
    """Build sem-match batches from collected extract JSONL + truth P sets."""
    extracts = {r["sample_id"]: r for r in _load_jsonl(extract_collected) if "sample_id" in r}
    truth = _load_json_array(truth_path)

    out_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []

    for sid, ex in sorted(extracts.items()):
        if sid < 0 or sid >= len(truth):
            continue
        P, H, _C = extract_P_H_C_from_truth_record(truth[sid])
        if not isinstance(P, list) or not P:
            continue
        top_k = ex.get("top_k_diagnoses") or ex.get("extracted_diagnoses") or []
        if not isinstance(top_k, list):
            continue
        top_k = [str(x).strip() for x in top_k if str(x).strip()]

        pairs: List[Tuple[str, str]] = []
        for dx in top_k:
            for p in P:
                if fuzzy_match(dx, str(p)):
                    continue  # resolved locally; still record in index as auto
                pairs.append((dx, str(p)))

        # Always record auto-matches in index for aggregate
        auto: List[Dict[str, Any]] = []
        judged_pairs: List[Tuple[str, str]] = []
        for dx in top_k:
            for p in P:
                p = str(p)
                if fuzzy_match(dx, p):
                    auto.append({"dx_a": dx, "dx_b": p, "match": True, "note": "string_match"})
                else:
                    judged_pairs.append((dx, p))

        chunk_i = 0
        for i in range(0, len(judged_pairs), max_pairs_per_call):
            chunk = judged_pairs[i : i + max_pairs_per_call]
            if not chunk:
                break
            payload = [{"dx_a": a, "dx_b": b} for a, b in chunk]
            user = "PAIRS:\n" + json.dumps(payload, ensure_ascii=False) + "\nReturn STRICT JSON."
            cid = make_custom_id("sem", target, arm, sid, chunk=chunk_i)
            out_rows.append(
                _gemini_judge_line(
                    custom_id=cid,
                    system=SEM_MATCH_BATCH_SYSTEM,
                    user=user,
                    thinking_level=thinking_level,
                    json_schema=SEM_SCHEMA,
                )
            )
            index_rows.append(
                {
                    "custom_id": cid,
                    "sample_id": sid,
                    "chunk": chunk_i,
                    "pairs": [{"dx_a": a, "dx_b": b} for a, b in chunk],
                }
            )
            chunk_i += 1

        index_rows.append(
            {
                "custom_id": make_custom_id("sem", target, arm, sid, chunk=-1),
                "sample_id": sid,
                "chunk": -1,
                "auto_matches": auto,
                "top_k_diagnoses": top_k,
                "P": P,
                "H": H,
            }
        )

    path = local_requests(suite, target, arm, "sem")
    _write_jsonl(path, out_rows)
    _write_jsonl(local_pair_index(suite, target, arm), index_rows)
    return path


def prepare_ground(
    *,
    suite: str,
    target: str,
    arm: str,
    extract_collected: str,
    start_idx: int = 0,
    end_idx: int = 0,
    cohort_json: str = DEFAULT_COHORT,
    max_grounding_dx: int = 8,
    thinking_level: str = "HIGH",
) -> str:
    extracts = {r["sample_id"]: r for r in _load_jsonl(extract_collected)}
    rows = _eval_rows(
        suite=suite,
        target=target,
        arm=arm,
        cohort_json=cohort_json,
        start_idx=start_idx,
        end_idx=end_idx,
    )
    out_rows: List[Dict[str, Any]] = []
    for sample in rows:
        sid = int(sample["sample_id"])
        q = (sample.get("raw_input") or "").strip()
        ans = (sample.get("model_response") or "").strip()
        ex = extracts.get(sid) or {}
        dxs = ex.get("top_k_diagnoses") or ex.get("extracted_diagnoses") or []
        if not isinstance(dxs, list):
            dxs = []
        dxs = [str(x).strip() for x in dxs if str(x).strip()][:max_grounding_dx]
        if not q or not ans or not dxs:
            continue
        user = (
            f"QUESTION:\n{q}\n\nMODEL_ANSWER:\n{ans}\n\n"
            f"EXTRACTED_DIAGNOSES:\n{json.dumps(dxs, ensure_ascii=False)}\n\n"
            "Return STRICT JSON."
        )
        out_rows.append(
            _gemini_judge_line(
                custom_id=make_custom_id("ground", target, arm, sid),
                system=GROUNDING_SYSTEM,
                user=user,
                thinking_level=thinking_level,
                json_schema=GROUND_SCHEMA,
            )
        )
    path = local_requests(suite, target, arm, "ground")
    _write_jsonl(path, out_rows)
    return path
