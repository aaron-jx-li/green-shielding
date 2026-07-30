"""Vertex Gemini generate helper (sync + async)."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from management_reasoning.clients.vertex import make_vertex_client
from management_reasoning.schema import RESPONSE_JSON_SCHEMA


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text is None and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            text = ""
    return (text or "").strip()


def _generate_once(
    client: Any,
    model: str,
    system: str,
    user: str,
    *,
    use_json_schema: bool,
) -> str:
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

    return _extract_text(resp)


def generate(
    system: str,
    user: str,
    model: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    use_json_schema: bool = True,
) -> str:
    """Call Vertex Gemini with system + user turns (sync)."""
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
) -> str:
    """
    Async wrapper: runs sync Vertex generate in a worker thread so many
    calls can overlap under an asyncio semaphore.
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
