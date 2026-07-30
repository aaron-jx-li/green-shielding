"""Claude on Vertex AI via AnthropicVertex (rawPredict), not generateContent.

Claude Model Garden models are under ``publishers/anthropic`` and are not
supported by the Gemini ``generateContent`` API. Use Anthropic's Vertex client.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from management_reasoning.clients.vertex import get_location, get_project

_ACCESS_HINT = (
    "Claude-on-Vertex requires Anthropic models enabled in Vertex Model Garden "
    "for this project, and calls must go through AnthropicVertex (not "
    "google.genai generateContent). "
    "Pinned ID: management_reasoning/models.yaml → claude_target "
    "(e.g. claude-opus-4-5@20251101). Region: global or us-east5."
)


def _make_anthropic_client(
    *,
    project: Optional[str] = None,
    location: Optional[str] = None,
    async_mode: bool = False,
) -> Any:
    proj = get_project(project)
    region = get_location(location)
    try:
        if async_mode:
            from anthropic import AsyncAnthropicVertex

            return AsyncAnthropicVertex(project_id=proj, region=region)
        from anthropic import AnthropicVertex

        return AnthropicVertex(project_id=proj, region=region)
    except ImportError as e:
        raise RuntimeError(
            "anthropic package with Vertex support is required. "
            "pip install 'anthropic[vertex]'\n" + _ACCESS_HINT
        ) from e


def _extract_text(message: Any) -> str:
    parts = []
    for block in getattr(message, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts).strip()


def _raise_access_error(model: str, err: Exception) -> None:
    msg = str(err)
    if "404" in msg or "NOT_FOUND" in msg or "not_found" in msg.lower():
        raise RuntimeError(
            f"Claude model not found on Vertex (model={model}). {_ACCESS_HINT}\n"
            f"Underlying error: {err}"
        ) from err
    if "403" in msg or "PERMISSION" in msg.upper() or "permission" in msg.lower():
        raise RuntimeError(
            f"Permission denied for Claude on Vertex (model={model}). {_ACCESS_HINT}\n"
            f"Underlying error: {err}"
        ) from err
    raise RuntimeError(f"Vertex Claude generate failed for model={model}: {err}") from err


def generate(
    system: str,
    user: str,
    model: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """
    Call Claude via AnthropicVertex.

    ``client`` may be an ``AnthropicVertex`` instance; otherwise one is created.
    ``location`` maps to AnthropicVertex ``region`` (``global`` and ``us-east5`` OK).
    """
    client = client or _make_anthropic_client(project=project, location=location)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        _raise_access_error(model, e)
    return _extract_text(message)


async def generate_async(
    system: str,
    user: str,
    model: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    max_tokens: int = 4096,
) -> str:
    """Async Claude via AsyncAnthropicVertex (or thread-offload of a sync client)."""
    if client is not None and type(client).__name__ == "AnthropicVertex":
        return await asyncio.to_thread(
            generate,
            system,
            user,
            model,
            client=client,
            project=project,
            location=location,
            max_tokens=max_tokens,
        )

    client = client or _make_anthropic_client(
        project=project, location=location, async_mode=True
    )
    try:
        message = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        _raise_access_error(model, e)
    return _extract_text(message)
