from __future__ import annotations
from typing import Tuple
from llm_utils import chat


def judge(question: str, response: str, truth: str, *, model: str = "gpt-4.1-mini") -> Tuple[bool, str]:
    """Strict grading: only returns CORRECT if model answer matches truth exactly/semantically."""
    msgs = [
        {
            "role": "system",
            "content": (
                "You are a strict grader. Decide if the MODEL_ANSWER matches the GROUND_TRUTH semantically. "
                "If ambiguous, partially correct, hedged, or wrong, output WRONG. "
                "If MODEL_ANSWER proposes a list but doesn't clarify that GROUND_TRUTH is the best answer, output WRONG. "
                "Output a single word only: CORRECT or WRONG."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{(question or '').strip()}\n\n"
                f"MODEL_ANSWER:\n{(response or '').strip()}\n\n"
                f"GROUND_TRUTH:\n{(truth or '').strip()}"
            ),
        },
    ]
    raw = chat(msgs, model=model, temperature=0.0, max_tokens=50).strip().upper()
    return ("CORRECT" in raw), raw
