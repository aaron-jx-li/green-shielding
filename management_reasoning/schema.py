"""Parse and validate structured Dx + management-reasoning model responses."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from management_reasoning.prompts import CONSENSUS_OPTIONS, TRIAGE_OPTIONS

_TASK_FIELDS = (
    "diagnosis",
    "diagnostic_consensus",
    "care_seeking",
    "cant_miss_ruling_out_question",
    "next_steps_consensus",
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
# Models sometimes copy "2. Go to the emergency department now" from a numbered list.
_TRIAGE_INDEX_PREFIX_RE = re.compile(r"^\s*\d+\.\s*")

# JSON Schema for provider constrained decoding / documentation.
RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": ["string", "null"]},
        "diagnostic_consensus": {"type": ["string", "null"], "enum": list(CONSENSUS_OPTIONS) + [None]},
        "care_seeking": {"type": ["string", "null"], "enum": list(TRIAGE_OPTIONS) + [None]},
        "cant_miss_ruling_out_question": {"type": ["string", "null"]},
        "next_steps_consensus": {"type": ["string", "null"], "enum": list(CONSENSUS_OPTIONS) + [None]},
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


@dataclass
class ParsedResponse:
    parse_ok: bool
    parsed: Optional[Dict[str, Any]]
    error: Optional[str]
    refusal: bool = False


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t, count=1)
    return t.strip()


def _canonicalize_care_seeking(value: Any) -> Optional[str]:
    """
    Exact triage match, with one format-only repair: strip a leading ``N. `` index.

    Does not fuzzy-match paraphrases (e.g. ``go to ER`` still fails).
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if s in TRIAGE_OPTIONS:
        return s
    stripped = _TRIAGE_INDEX_PREFIX_RE.sub("", s, count=1).strip()
    if stripped in TRIAGE_OPTIONS:
        return stripped
    return None


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def parse_model_response(raw_text: str) -> ParsedResponse:
    """
    Parse model output into the management-reasoning schema.

    Consensus values are casefolded to canonical ``high`` / ``low``.
    Triage must match ``TRIAGE_OPTIONS`` exactly, except an optional leading
    ``N. `` list-index prefix is stripped (format-only; no paraphrase matching).
    """
    if raw_text is None or not str(raw_text).strip():
        return ParsedResponse(False, None, "empty response", False)

    cleaned = _strip_fences(str(raw_text))
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return ParsedResponse(False, None, f"invalid JSON: {e}", False)

    if not isinstance(obj, dict):
        return ParsedResponse(False, None, "JSON root must be an object", False)

    missing = [k for k in (*_TASK_FIELDS, "refusal") if k not in obj]
    if missing:
        return ParsedResponse(False, None, f"missing keys: {missing}", False)

    refusal = _as_bool(obj.get("refusal"))
    if refusal is None:
        return ParsedResponse(False, None, "refusal must be a boolean", False)

    if refusal:
        parsed = {k: obj.get(k) for k in _TASK_FIELDS}
        parsed["refusal"] = True
        return ParsedResponse(True, parsed, None, True)

    diagnosis = obj.get("diagnosis")
    cant_miss = obj.get("cant_miss_ruling_out_question")
    if not isinstance(diagnosis, str) or not diagnosis.strip():
        return ParsedResponse(False, None, "diagnosis must be a non-empty string when refusal=false", False)
    if not isinstance(cant_miss, str) or not cant_miss.strip():
        return ParsedResponse(
            False,
            None,
            "cant_miss_ruling_out_question must be a non-empty string when refusal=false",
            False,
        )

    def _consensus(field: str) -> Optional[str]:
        val = obj.get(field)
        if not isinstance(val, str):
            return None
        canon = val.strip().casefold()
        if canon not in CONSENSUS_OPTIONS:
            return None
        return canon

    diagnostic_consensus = _consensus("diagnostic_consensus")
    next_steps_consensus = _consensus("next_steps_consensus")
    if diagnostic_consensus is None:
        return ParsedResponse(
            False,
            None,
            "diagnostic_consensus must be exactly 'high' or 'low' (case-insensitive)",
            False,
        )
    if next_steps_consensus is None:
        return ParsedResponse(
            False,
            None,
            "next_steps_consensus must be exactly 'high' or 'low' (case-insensitive)",
            False,
        )

    care = _canonicalize_care_seeking(obj.get("care_seeking"))
    if care is None:
        return ParsedResponse(
            False,
            None,
            "care_seeking must exactly match one of TRIAGE_OPTIONS "
            "(optional leading 'N. ' index only; no paraphrase match)",
            False,
        )

    parsed = {
        "diagnosis": diagnosis.strip(),
        "diagnostic_consensus": diagnostic_consensus,
        "care_seeking": care,
        "cant_miss_ruling_out_question": cant_miss.strip(),
        "next_steps_consensus": next_steps_consensus,
        "refusal": False,
    }
    return ParsedResponse(True, parsed, None, False)


def assert_schema_self_check() -> None:
    """Lightweight offline checks (valid / refusal / bad triage)."""
    good = json.dumps(
        {
            "diagnosis": "GERD",
            "diagnostic_consensus": "High",
            "care_seeking": "Schedule non-urgent medical appointment",
            "cant_miss_ruling_out_question": "Any chest pain with exertion?",
            "next_steps_consensus": "low",
            "refusal": False,
        }
    )
    r = parse_model_response(good)
    assert r.parse_ok and r.parsed is not None
    assert r.parsed["diagnostic_consensus"] == "high"

    refused = json.dumps(
        {
            "diagnosis": None,
            "diagnostic_consensus": None,
            "care_seeking": None,
            "cant_miss_ruling_out_question": None,
            "next_steps_consensus": None,
            "refusal": True,
        }
    )
    r2 = parse_model_response(refused)
    assert r2.parse_ok and r2.refusal

    bad = json.dumps(
        {
            "diagnosis": "x",
            "diagnostic_consensus": "high",
            "care_seeking": "go to ER",
            "cant_miss_ruling_out_question": "q",
            "next_steps_consensus": "low",
            "refusal": False,
        }
    )
    r3 = parse_model_response(bad)
    assert not r3.parse_ok

    numbered = json.dumps(
        {
            "diagnosis": "x",
            "diagnostic_consensus": "high",
            "care_seeking": "2. Go to the emergency department now",
            "cant_miss_ruling_out_question": "q",
            "next_steps_consensus": "low",
            "refusal": False,
        }
    )
    r4 = parse_model_response(numbered)
    assert r4.parse_ok and r4.parsed is not None
    assert r4.parsed["care_seeking"] == "Go to the emergency department now"


if __name__ == "__main__":
    assert_schema_self_check()
    print("schema self-check OK")
