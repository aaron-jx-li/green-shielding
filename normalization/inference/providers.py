from __future__ import annotations

import os
from typing import Any, List, Optional, Union

from anthropic import Anthropic
from openai import OpenAI


def detect_provider(model_name: str) -> str:
    if model_name.lower().startswith("claude"):
        return "anthropic"
    return "openai"


def query_openai_responses_system(
    client: OpenAI,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    num_runs: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    """OpenAI Responses API with system role (legacy ``inference.py`` behavior, incl. gpt-5 kwargs)."""
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]

    outs: List[str] = []
    for k in range(num_runs):
        kwargs: dict[str, Any] = {"model": model, "input": messages}
        if temperature is not None and "gpt-5" not in model:
            kwargs["temperature"] = temperature
        if max_tokens is not None and "gpt-5" not in model:
            kwargs["max_output_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed + k

        resp = client.responses.create(**kwargs)
        outs.append(resp.output_text or "")

    return outs


def query_openai_responses_developer(
    client: OpenAI,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    num_runs: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    """OpenAI Responses API with developer role (``inference_w_anthropic.py`` openai path)."""
    messages = [
        {"role": "developer", "content": instruction},
        {"role": "user", "content": input_text},
    ]

    outs: List[str] = []
    for k in range(num_runs):
        kwargs: dict[str, Any] = {"model": model, "input": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed + k

        resp = client.responses.create(**kwargs)
        outs.append(resp.output_text or "")

    return outs


def query_anthropic_messages(
    client: Anthropic,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    num_runs: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    if max_tokens is None:
        max_tokens = 4096

    outs: List[str] = []
    for _k in range(num_runs):
        kwargs: dict[str, Any] = {
            "model": model,
            "system": instruction,
            "messages": [{"role": "user", "content": input_text}],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature

        resp = client.messages.create(**kwargs)
        txt = ""
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                txt += blk.text
        outs.append(txt.strip())

    return outs


def query_multi_provider(
    client: Union[OpenAI, Anthropic],
    provider: str,
    input_text: str,
    model: str,
    instruction: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    num_runs: int = 1,
    seed: Optional[int] = None,
) -> List[str]:
    if provider == "anthropic":
        if not isinstance(client, Anthropic):
            raise TypeError("Client must be Anthropic instance for anthropic provider")
        return query_anthropic_messages(
            client, input_text, model, instruction, temperature, max_tokens, num_runs, seed
        )
    if provider == "openai":
        if not isinstance(client, OpenAI):
            raise TypeError("Client must be OpenAI instance for openai provider")
        return query_openai_responses_developer(
            client, input_text, model, instruction, temperature, max_tokens, num_runs, seed
        )
    raise ValueError(f"Unknown provider: {provider}")


def make_anthropic_client(api_key: Optional[str]) -> Anthropic:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Anthropic API key required. Set --anthropic_api_key or ANTHROPIC_API_KEY environment variable."
        )
    return Anthropic(api_key=key)
