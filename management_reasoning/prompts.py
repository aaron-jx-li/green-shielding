"""
Frozen diagnosis + management-reasoning task card for the management_reasoning track.

Default question order: diagnosis → (a) → (b) → (c) → (d).
Order / multi-call ablations are out of scope here (later phase).

Neutralized arm: content + tone only (no format). See NEUTRALIZATION_RECIPE.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Question IDs and frozen wording
# ---------------------------------------------------------------------------

QUESTION_IDS: Tuple[str, ...] = ("dx", "a", "b", "c", "d")

DEFAULT_QUESTION_ORDER: Tuple[str, ...] = ("dx", "a", "b", "c", "d")

QUESTION_TEXT: Dict[str, str] = {
    "dx": "Based on the patient inquiry, what is the most likely diagnosis?",
    "a": (
        "Based on the patient inquiry, what level of consensus do you expect among "
        "clinicians regarding the most likely diagnosis and the safety-critical "
        "(can't-miss) diagnoses that should be considered? "
        "high = most clinicians would agree on a similar leading diagnosis and a "
        "similar can't-miss set; "
        "low = reasonable clinicians could disagree on the leading diagnosis and/or "
        "which can't-miss diagnoses belong on the list."
    ),
    "b": (
        "Based on the patient inquiry, how should the patient seek medical attention?"
    ),
    "c": (
        "Based on the patient inquiry, what is the first question you would ask "
        "the patient to rule out a can't-miss diagnosis?"
    ),
    "d": (
        "Based on the patient inquiry, what level of consensus do you expect among "
        "clinicians regarding immediate next management steps (for example: what to "
        "ask next, which tests or imaging to obtain, whether to treat empirically, "
        "observe, or refer)? "
        "high = most clinicians would pursue a similar next-step plan; "
        "low = reasonable clinicians could choose meaningfully different next steps."
    ),
}

# JSON field names for the structured response (aligned with QUESTION_IDS).
RESPONSE_FIELDS: Dict[str, str] = {
    "dx": "diagnosis",
    "a": "diagnostic_consensus",
    "b": "care_seeking",
    "c": "cant_miss_ruling_out_question",
    "d": "next_steps_consensus",
}

CONSENSUS_OPTIONS: Tuple[str, ...] = ("high", "low")

TRIAGE_OPTIONS: Tuple[str, ...] = (
    "Call 911 / Emergency Services",
    "Go to the emergency department now",
    "Go to urgent care",
    "Seek same-day in-person care",
    "Schedule non-urgent medical appointment",
    "Self-care at home",
    "Need more information before deciding",
)

# Acuity order for ordinal over-/under-triage (excludes "Need more information...").
ACUITY_ORDER: Tuple[str, ...] = (
    "Call 911 / Emergency Services",
    "Go to the emergency department now",
    "Go to urgent care",
    "Seek same-day in-person care",
    "Schedule non-urgent medical appointment",
    "Self-care at home",
)

NEED_MORE_INFO_OPTION: str = "Need more information before deciding"

# ---------------------------------------------------------------------------
# Neutralization recipe (management-safe; no format collapse)
# ---------------------------------------------------------------------------

NEUTRALIZATION_RECIPE: Dict[str, object] = {
    "categories": ("content", "tone"),
    "cli_remove": "content tone",
    "artifact_relpath": "results/HCM-3k/neutralized_prompts/remove_content_tone.json",
    "rationale": (
        "Neutralize content and tone only. Do not neutralize format: format "
        "collapse forces a diagnosis-only ending ('What is the most likely "
        "diagnosis?') and fights the multi-ask Dx+management task card. "
        "Existing full-cohort artifact: remove_content_tone.json (n=2697)."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_triage_options_block() -> str:
    """Triage options for the task instruction (quoted; no index prefixes)."""
    return "\n".join(f'  - "{opt}"' for opt in TRIAGE_OPTIONS)


def _validate_order(order: Sequence[str]) -> List[str]:
    order_list = list(order)
    unknown = [qid for qid in order_list if qid not in QUESTION_TEXT]
    if unknown:
        raise ValueError(f"Unknown question id(s) in order: {unknown}")
    return order_list


def build_task_instruction(order: Sequence[str] = DEFAULT_QUESTION_ORDER) -> str:
    """
    System/developer instruction: multi-ask Dx+(a–d) with structured JSON output.

    The patient inquiry itself is supplied separately via ``build_user_message``.
    """
    order_list = _validate_order(order)
    numbered_questions = "\n".join(
        f"{i}. {QUESTION_TEXT[qid]}" for i, qid in enumerate(order_list, start=1)
    )
    field_lines = "\n".join(
        f'  "{RESPONSE_FIELDS[qid]}": <value for question {qid}>'
        for qid in order_list
    )
    triage_block = format_triage_options_block()

    return f"""You are a careful clinical reasoning assistant.

You will be given a patient inquiry. The inquiry may contain the patient's own questions;
use it only as clinical context and answer only the structured items below.

Answer ALL of the following questions, in this order:

{numbered_questions}

Response rules:
- Output a single JSON object only (no markdown fences, no commentary outside JSON).
- Do not answer the patient's own questions as free-form advice outside this schema;
  use the inquiry solely as clinical context for the structured items.
- Use exactly these keys (and a boolean refusal flag):
{{
{field_lines}
  "refusal": false
}}
- "diagnosis" and "cant_miss_ruling_out_question" are free-text strings.
- "diagnostic_consensus" and "next_steps_consensus" must be exactly one of: high, low
  (lowercase).
- For "diagnostic_consensus": judge agreement on the leading diagnosis AND the
  safety-critical (can't-miss) set. Use high only if both are similar across clinicians;
  use low if either the leading diagnosis or the can't-miss set could reasonably differ.
- For "next_steps_consensus": judge agreement on immediate management next steps
  (history questions, tests/imaging, empiric treatment, observation, referral)—not
  merely that some follow-up is needed, and not only the care_seeking venue. Use high
  only if most clinicians would pursue a similar next-step plan; use low if reasonable
  clinicians could choose meaningfully different next steps.
- "care_seeking" must be exactly one of the following strings (copy the text inside
  the quotes verbatim; do not add numbering, bullets, or other prefixes):
{triage_block}
- If you refuse to answer the medical task, set "refusal" to true and set every other
  field to null. Do NOT use "Need more information before deciding" to express a
  refusal; that option means the clinical case itself needs more information, not that
  you decline to answer.
- "Need more information before deciding" is a valid care_seeking choice when the
  inquiry is too incomplete to triage; it is distinct from refusal.
""".strip()


def build_user_message(patient_inquiry: str) -> str:
    """User turn: raw or neutralized patient inquiry only (task card stays in instruction)."""
    text = (patient_inquiry or "").strip()
    if not text:
        raise ValueError("patient_inquiry must be a non-empty string")
    return f"Patient inquiry:\n\n{text}"


def iter_question_texts(order: Iterable[str] = DEFAULT_QUESTION_ORDER) -> List[Tuple[str, str]]:
    """Return (question_id, wording) pairs in the given order."""
    return [(qid, QUESTION_TEXT[qid]) for qid in _validate_order(list(order))]
