"""Vertex Gemini generate helper (sync + async)."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from management_reasoning.clients.vertex import make_vertex_client
from management_reasoning.schema import RESPONSE_JSON_SCHEMA

GenerateResult = Tuple[str, Dict[str, Any]]


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text is None and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            text = ""
    return (text or "").strip()


def _extract_usage(resp: Any) -> Dict[str, Any]:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    keys = (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
        "cached_content_token_count",
        "tool_use_prompt_token_count",
    )
    usage: Dict[str, Any] = {}
    for k in keys:
        v = getattr(um, k, None)
        if v is not None:
            usage[k] = int(v)
    # Billed output ≈ candidates + thoughts (Google prices thinking as output).
    cand = usage.get("candidates_token_count") or 0
    thoughts = usage.get("thoughts_token_count") or 0
    usage["billed_output_token_count"] = cand + thoughts
    return usage


def _generate_once(
    client: Any,
    model: str,
    system: str,
    user: str,
    *,
    use_json_schema: bool,
) -> GenerateResult:
    from google.genai import types

    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=user)]),
    ]
    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "response_mime_type": "application/json",
    }
    if use_json_schema:
        config_kwargs["response_json_schema"] = RESPONSE_JSON_SCHEMA

    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    except Exception as first_err:
        if use_json_schema:
            try:
                config_kwargs.pop("response_json_schema", None)
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as second_err:
                raise RuntimeError(
                    f"Vertex Gemini generate failed for model={model}: {second_err}"
                ) from second_err
        else:
            raise RuntimeError(
                f"Vertex Gemini generate failed for model={model}: {first_err}"
            ) from first_err

    return _extract_text(resp), _extract_usage(resp)


def generate(
    system: str,
    user: str,
    model: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    use_json_schema: bool = True,
) -> GenerateResult:
    """Call Vertex Gemini with system + user turns (sync). Returns (text, usage)."""
    client = client or make_vertex_client(project=project, location=location)
    return _generate_once(
        client, model, system, user, use_json_schema=use_json_schema
    )


async def generate_async(
    system: str,
    user: str,
    model: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    use_json_schema: bool = True,
) -> GenerateResult:
    """
    Async wrapper: runs sync Vertex generate in a worker thread so many
    calls can overlap under an asyncio semaphore. Returns (text, usage).
    """
    client = client or make_vertex_client(project=project, location=location)
    return await asyncio.to_thread(
        _generate_once,
        client,
        model,
        system,
        user,
        use_json_schema=use_json_schema,
    )
