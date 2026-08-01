"""Build Vertex Batch request JSONL for Gemini and Claude."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from management_reasoning.batch.paths import (
    LEGACY_DIAG_INSTRUCTION,
    LEGACY_DIAG_TEMPERATURE,
    VALID_ARMS,
    custom_id,
    local_request_jsonl,
    suite_mode,
    suite_order_id,
)
from management_reasoning.prompts import (
    CONSENSUS_OPTIONS,
    ORDER_VARIANTS,
    QUESTION_IDS,
    RESPONSE_FIELDS,
    TRIAGE_OPTIONS,
    build_single_question_instruction,
    build_task_instruction,
    build_user_message,
)

# Vertex Batch rejects ``null`` inside ``enum``. Keep nullable via type union only.
_BATCH_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": ["string", "null"]},
        "diagnostic_consensus": {
            "type": ["string", "null"],
            "enum": list(CONSENSUS_OPTIONS),
        },
        "care_seeking": {
            "type": ["string", "null"],
            "enum": list(TRIAGE_OPTIONS),
        },
        "cant_miss_ruling_out_question": {"type": ["string", "null"]},
        "next_steps_consensus": {
            "type": ["string", "null"],
            "enum": list(CONSENSUS_OPTIONS),
        },
        "refusal": {"type": "boolean"},
    },
    "required": [
        "diagnosis",
        "diagnostic_consensus",
        "care_seeking",
        "cant_miss_ruling_out_question",
        "next_steps_consensus",
        "refusal",
    ],
    "additionalProperties": False,
}


def _single_question_schema(qid: str) -> Dict[str, Any]:
    """JSON schema for one independent question.

    Vertex Gemini Batch rejects ``type: [\"string\", \"null\"]`` on properties
    (``must be a boolean or an object``). Use plain string types; refusal covers
    decline-to-answer (empty string / refusal=true).
    """
    field = RESPONSE_FIELDS[qid]
    if qid in ("dx", "c"):
        prop: Dict[str, Any] = {"type": "string"}
    elif qid in ("a", "d"):
        prop = {"type": "string", "enum": list(CONSENSUS_OPTIONS)}
    elif qid == "b":
        prop = {"type": "string", "enum": list(TRIAGE_OPTIONS)}
    else:
        raise ValueError(qid)
    return {
        "type": "object",
        "properties": {field: prop, "refusal": {"type": "boolean"}},
        "required": [field, "refusal"],
        "additionalProperties": False,
    }


def _inquiry_for_arm(sample: Dict[str, Any], arm: str) -> str:
    # Non-raw arms (neutralized, remove_all, ct_old, ct_new) store text under
    # neutralized_prompt in the suite input JSON.
    key = "raw_input" if arm == "raw" else "neutralized_prompt"
    text = sample.get(key) or ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"empty {key} for sample_id={sample.get('sample_id')}")
    return text


def gemini_request_line(
    *,
    provider: str,
    arm: str,
    sample_id: int,
    system: str,
    user: str,
    include_json_schema: bool = True,
    order_id: Optional[str] = None,
    question_id: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    freeform_text: bool = False,
) -> Dict[str, Any]:
    generation_config: Dict[str, Any] = {
        "thinking_config": {"thinking_level": "HIGH"},
    }
    if freeform_text:
        # Plain text (legacy diagnosis-only); do not force JSON mime.
        if temperature is not None:
            generation_config["temperature"] = temperature
    else:
        generation_config["response_mime_type"] = "application/json"
        if include_json_schema:
            generation_config["response_json_schema"] = (
                json_schema or _BATCH_RESPONSE_JSON_SCHEMA
            )
        if temperature is not None:
            generation_config["temperature"] = temperature

    return {
        "custom_id": custom_id(
            provider, arm, sample_id, order_id=order_id, question_id=question_id
        ),
        "request": {
            "contents": [
                {"role": "user", "parts": [{"text": user}]},
            ],
            "system_instruction": {"parts": [{"text": system}]},
            "generation_config": generation_config,
        },
    }


def claude_request_line(
    *,
    provider: str,
    arm: str,
    sample_id: int,
    system: str,
    user: str,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    order_id: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    req: Dict[str, Any] = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if temperature is not None:
        req["temperature"] = temperature
    return {
        "custom_id": custom_id(
            provider, arm, sample_id, order_id=order_id, question_id=question_id
        ),
        "request": req,
    }


def _append_provider_row(
    rows: List[Dict[str, Any]],
    *,
    provider: str,
    arm: str,
    sample_id: int,
    system: str,
    user: str,
    include_json_schema: bool,
    order_id: Optional[str] = None,
    question_id: Optional[str] = None,
    json_schema: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    freeform_text: bool = False,
) -> None:
    if provider == "gemini":
        rows.append(
            gemini_request_line(
                provider=provider,
                arm=arm,
                sample_id=sample_id,
                system=system,
                user=user,
                include_json_schema=include_json_schema,
                order_id=order_id,
                question_id=question_id,
                json_schema=json_schema,
                temperature=temperature,
                freeform_text=freeform_text,
            )
        )
    else:
        rows.append(
            claude_request_line(
                provider=provider,
                arm=arm,
                sample_id=sample_id,
                system=system,
                user=user,
                temperature=temperature,
                order_id=order_id,
                question_id=question_id,
            )
        )


def build_request_rows(
    samples: Sequence[Dict[str, Any]],
    *,
    provider: str,
    arm: str,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    include_json_schema: bool = True,
    suite: str = "primary",
    order: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    if provider not in ("gemini", "claude"):
        raise ValueError(f"unknown provider: {provider}")
    if arm not in VALID_ARMS:
        raise ValueError(f"unknown arm: {arm}")

    n = len(samples)
    if end_idx is None:
        end_idx = n - 1
    if start_idx < 0 or end_idx < start_idx or end_idx >= n:
        raise ValueError(f"Invalid index range [{start_idx}, {end_idx}] for n={n}")

    mode = suite_mode(suite)
    order_id = suite_order_id(suite)
    if order is None and order_id:
        order = ORDER_VARIANTS[order_id]
    rows: List[Dict[str, Any]] = []

    if mode == "legacy_freeform":
        # Paper diagnosis-only: old instruction, inquiry as-is, no JSON schema.
        system = LEGACY_DIAG_INSTRUCTION
        for i in range(start_idx, end_idx + 1):
            sample = samples[i]
            sample_id = int(sample.get("sample_id", i))
            user = _inquiry_for_arm(sample, arm)
            _append_provider_row(
                rows,
                provider=provider,
                arm=arm,
                sample_id=sample_id,
                system=system,
                user=user,
                include_json_schema=False,
                temperature=LEGACY_DIAG_TEMPERATURE,
                freeform_text=True,
            )
        return rows

    if mode == "independent":
        systems = {qid: build_single_question_instruction(qid) for qid in QUESTION_IDS}
        schemas = {qid: _single_question_schema(qid) for qid in QUESTION_IDS}
        for i in range(start_idx, end_idx + 1):
            sample = samples[i]
            sample_id = int(sample.get("sample_id", i))
            user = build_user_message(_inquiry_for_arm(sample, arm))
            for qid in QUESTION_IDS:
                # Gemini Batch unions response_json_schema across the JSONL and
                # injects null for missing properties → invalid schema. Skip
                # structured schema for Gemini independent (prompt + JSON mime).
                use_schema = include_json_schema and provider != "gemini"
                _append_provider_row(
                    rows,
                    provider=provider,
                    arm=arm,
                    sample_id=sample_id,
                    system=systems[qid],
                    user=user,
                    include_json_schema=use_schema,
                    question_id=qid,
                    json_schema=schemas[qid] if use_schema else None,
                )
        return rows

    # multi-ask (primary or order_*)
    system = build_task_instruction(order) if order else build_task_instruction()
    for i in range(start_idx, end_idx + 1):
        sample = samples[i]
        sample_id = int(sample.get("sample_id", i))
        user = build_user_message(_inquiry_for_arm(sample, arm))
        _append_provider_row(
            rows,
            provider=provider,
            arm=arm,
            sample_id=sample_id,
            system=system,
            user=user,
            include_json_schema=include_json_schema,
            order_id=order_id,
        )
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def prepare_local_jsonl(
    samples: Sequence[Dict[str, Any]],
    *,
    provider: str,
    arm: str,
    suite: str = "primary",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    include_json_schema: bool = True,
) -> str:
    rows = build_request_rows(
        samples,
        provider=provider,
        arm=arm,
        start_idx=start_idx,
        end_idx=end_idx,
        include_json_schema=include_json_schema,
        suite=suite,
    )
    out = local_request_jsonl(provider, arm, suite=suite)
    write_jsonl(out, rows)
    return out
