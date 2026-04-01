from __future__ import annotations

from typing import Any, Dict, Optional

from openai import OpenAI

from open_eval.core.json_utils import robust_json_loads

_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def call_json_judge(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    retries: int = 4,
) -> Dict[str, Any]:
    client = get_openai_client()
    last_err = None
    last_text = None

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            last_text = resp.output_text
            return robust_json_loads(last_text)

        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if ("json" in msg) or ("expecting" in msg) or ("delimiter" in msg) or ("decode" in msg) or ("unbalanced" in msg):
                user_prompt = (
                    user_prompt
                    + "\n\nIMPORTANT: Your previous output was invalid. "
                    "Return ONLY a single valid JSON object, no markdown, no extra text."
                )

    if last_text:
        snippet = last_text[:400].replace("\n", "\\n")
        tail = last_text[-200:].replace("\n", "\\n") if len(last_text) > 600 else ""
        raise RuntimeError(f"Judge failed after retries: {last_err}\nFirst400={snippet}\nLast200={tail}")

    raise RuntimeError(f"Judge failed after retries: {last_err}")
