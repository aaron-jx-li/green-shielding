from __future__ import annotations
import time
from typing import Dict, List, Optional
from openai import OpenAI

_client = OpenAI()


def chat(
    messages: List[Dict[str, str]],
    model: str,
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    retries: int = 5,
    backoff: float = 0.4,
) -> str:
    """Robust wrapper around OpenAI chat API with retries."""
    # print(messages)
    last_err = None
    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"OpenAI chat failed after {retries} attempts: {last_err}")
