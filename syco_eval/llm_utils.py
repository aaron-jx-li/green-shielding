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
   
    last_err = None
    for attempt in range(retries):
        try:
            # print(messages)
            if "gpt-5" in model: # newer models don't support temperature or max_output_tokens
                resp = _client.responses.create(
                    model=model,
                    input=messages,
                )
            else:
                resp = _client.responses.create(
                    model=model,
                    input=messages,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            return resp.output_text or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"OpenAI chat failed after {retries} attempts: {last_err}")
