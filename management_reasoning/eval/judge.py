"""Vertex Gemini JSON judge with optional thinking (Flash-Lite default)."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from management_reasoning.clients.vertex import get_location, get_project, make_vertex_client
from management_reasoning.eval.json_utils import robust_json_loads

UsageDict = Dict[str, Any]
JudgeResult = Tuple[Dict[str, Any], UsageDict]


def _extract_text(resp: Any) -> str:
    text = getattr(resp, "text", None)
    if text is None and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
            chunks = []
            for p in parts:
                if getattr(p, "thought", False):
                    continue
                chunks.append(getattr(p, "text", "") or "")
            text = "".join(chunks)
            if not text.strip():
                text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            text = ""
    return (text or "").strip()


def _extract_usage(resp: Any) -> UsageDict:
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return {}
    usage: UsageDict = {}
    for k in (
        "prompt_token_count",
        "candidates_token_count",
        "thoughts_token_count",
        "total_token_count",
        "cached_content_token_count",
    ):
        v = getattr(um, k, None)
        if v is not None:
            usage[k] = int(v)
    cand = usage.get("candidates_token_count") or 0
    thoughts = usage.get("thoughts_token_count") or 0
    usage["billed_output_token_count"] = cand + thoughts
    return usage


def _thinking_config(thinking_level: Optional[str]):
    if not thinking_level:
        return None
    from google.genai import types

    level = str(thinking_level).strip().upper()
    enum_map = {
        "MINIMAL": types.ThinkingLevel.MINIMAL,
        "LOW": types.ThinkingLevel.LOW,
        "MEDIUM": types.ThinkingLevel.MEDIUM,
        "HIGH": types.ThinkingLevel.HIGH,
    }
    tl = enum_map.get(level)
    if tl is None:
        return types.ThinkingConfig(thinking_level=level)
    return types.ThinkingConfig(thinking_level=tl)


def call_json_judge(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    temperature: float = 0.0,
    thinking_level: str = "HIGH",
    retries: int = 4,
) -> Dict[str, Any]:
    obj, _usage = call_json_judge_with_usage(
        model,
        system_prompt,
        user_prompt,
        client=client,
        project=project,
        location=location,
        temperature=temperature,
        thinking_level=thinking_level,
        retries=retries,
    )
    return obj


def call_json_judge_with_usage(
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    temperature: float = 0.0,
    thinking_level: str = "HIGH",
    retries: int = 4,
) -> JudgeResult:
    from google.genai import types

    client = client or make_vertex_client(
        project=get_project(project), location=get_location(location)
    )
    last_err: Optional[Exception] = None
    last_text: Optional[str] = None
    usage_acc: UsageDict = {}

    prompt = user_prompt
    for _attempt in range(retries):
        try:
            config_kwargs: Dict[str, Any] = {
                "system_instruction": system_prompt,
                "response_mime_type": "application/json",
                "temperature": temperature,
            }
            tc = _thinking_config(thinking_level)
            if tc is not None:
                config_kwargs["thinking_config"] = tc

            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
                ],
                config=types.GenerateContentConfig(**config_kwargs),
            )
            last_text = _extract_text(resp)
            usage_acc = _extract_usage(resp)
            return robust_json_loads(last_text), usage_acc
        except (json.JSONDecodeError, ValueError, RuntimeError) as e:
            last_err = e
            prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous output was invalid. "
                "Return ONLY a single valid JSON object, no markdown, no extra text."
            )
        except Exception as e:
            last_err = e
            # Retry once with a stricter reminder for parse-ish API errors.
            if last_text:
                try:
                    return robust_json_loads(last_text), usage_acc
                except Exception:
                    pass
            prompt = (
                user_prompt
                + "\n\nIMPORTANT: Your previous output was invalid. "
                "Return ONLY a single valid JSON object, no markdown, no extra text."
            )

    if last_text:
        snippet = last_text[:400].replace("\n", "\\n")
        raise RuntimeError(
            f"Judge failed after retries: {last_err}\nFirst400={snippet}"
        )
    raise RuntimeError(f"Judge failed after retries: {last_err}")
