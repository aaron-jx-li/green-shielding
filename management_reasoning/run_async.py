#!/usr/bin/env python3
"""Async Vertex inference for mid-size pilots (Gemini or Claude; bounded concurrency)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.clients import claude_vertex_client, gemini_client
from management_reasoning.clients.vertex import VertexConfigError, get_location, get_project, make_vertex_client
from management_reasoning.config import models_config
from management_reasoning.prompts import build_task_instruction, build_user_message
from management_reasoning.schema import parse_model_response


def _load_done_ids(jsonl_path: str) -> Set[int]:
    done: Set[int] = set()
    if not os.path.isfile(jsonl_path):
        return done
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get("sample_id")
            if isinstance(sid, int):
                done.add(sid)
    return done


async def _append_jsonl(path: str, record: Dict[str, Any], lock: asyncio.Lock) -> None:
    async with lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _is_transient(err: Exception) -> bool:
    msg = str(err).lower()
    needles = (
        "429",
        "rate",
        "quota",
        "timeout",
        "temporar",
        "unavailable",
        "503",
        "500",
        "connection",
        "reset",
        "overloaded",
    )
    return any(n in msg for n in needles)


async def _generate_with_retries(
    *,
    provider: str,
    system: str,
    user: str,
    model: str,
    client: Any,
    project: str,
    location: str,
    max_retries: int,
) -> str:
    last: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            if provider == "claude":
                return await claude_vertex_client.generate_async(
                    system,
                    user,
                    model,
                    client=client,
                    project=project,
                    location=location,
                )
            return await gemini_client.generate_async(
                system,
                user,
                model,
                client=client,
                project=project,
                location=location,
            )
        except Exception as e:
            last = e
            if attempt >= max_retries or not _is_transient(e):
                raise
            await asyncio.sleep(2 ** attempt)
    assert last is not None
    raise last


async def _run(args: argparse.Namespace) -> None:
    models_cfg = models_config()
    default_gemini = models_cfg.get("gemini_target", "gemini-3.1-pro-preview")
    default_claude = models_cfg.get("claude_target", "claude-opus-4-5@20251101")
    default_location = models_cfg.get("location_default", "global")
    if args.provider == "claude":
        default_location = models_cfg.get("claude_location_default") or default_location

    model = args.model or (default_claude if args.provider == "claude" else default_gemini)
    location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or default_location
    model_dir = model.replace("@", "_")
    out_dir = args.out_dir or (
        f"./results/management_reasoning/responses/vertex/{model_dir}/{args.arm}"
    )

    try:
        project = get_project(args.project)
        location = get_location(location)
    except VertexConfigError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    with open(args.input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("input_path must be a JSON list")

    n = len(data)
    start, end = args.start_idx, args.end_idx
    if start < 0 or end < start or end >= n:
        raise SystemExit(f"Invalid index range [{start}, {end}] for n={n}")

    out_jsonl = args.out_jsonl or os.path.join(out_dir, "responses.jsonl")
    done = _load_done_ids(out_jsonl) if args.skip_existing else set()
    instruction = build_task_instruction()

    if args.provider == "claude":
        client = claude_vertex_client._make_anthropic_client(
            project=project, location=location, async_mode=True
        )
    else:
        client = make_vertex_client(project=project, location=location)

    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)

    todo: List[Dict[str, Any]] = []
    for i in range(start, end + 1):
        sample = data[i]
        sample_id = int(sample.get("sample_id", i))
        if args.skip_existing and sample_id in done:
            continue
        todo.append(sample)

    print(
        f"Vertex async: provider={args.provider} model={model} project={project} "
        f"location={location} arm={args.arm} range=[{start},{end}] todo={len(todo)} "
        f"concurrency={args.concurrency} out={out_jsonl}"
    )

    stats = {"attempted": 0, "parse_ok": 0, "refusal": 0, "errors": 0}
    t0 = time.perf_counter()
    pbar = tqdm(total=len(todo), desc="async-inference")

    async def one(sample: Dict[str, Any]) -> None:
        sample_id = int(sample.get("sample_id"))
        inquiry_key = "raw_input" if args.arm == "raw" else "neutralized_prompt"
        inquiry = sample.get(inquiry_key) or ""
        record: Dict[str, Any] = {
            "sample_id": sample_id,
            "arm": args.arm,
            "provider": args.provider,
            "model": model,
            "project": project,
            "location": location,
            "raw_response": None,
            "parsed": None,
            "parse_ok": False,
            "error": None,
            "refusal": False,
        }
        async with sem:
            try:
                user_msg = build_user_message(inquiry)
                raw = await _generate_with_retries(
                    provider=args.provider,
                    system=instruction,
                    user=user_msg,
                    model=model,
                    client=client,
                    project=project,
                    location=location,
                    max_retries=args.max_retries,
                )
                record["raw_response"] = raw
                parsed = parse_model_response(raw)
                record["parse_ok"] = parsed.parse_ok
                record["parsed"] = parsed.parsed
                record["error"] = parsed.error
                record["refusal"] = parsed.refusal
            except Exception as e:
                record["error"] = f"generate failed: {e}"

        await _append_jsonl(out_jsonl, record, write_lock)
        stats["attempted"] += 1
        if record["parse_ok"]:
            stats["parse_ok"] += 1
        if record.get("refusal"):
            stats["refusal"] += 1
        if record.get("error") and not record["parse_ok"]:
            stats["errors"] += 1
        pbar.update(1)

    await asyncio.gather(*(one(s) for s in todo))
    pbar.close()
    elapsed = time.perf_counter() - t0
    attempted = stats["attempted"]
    parse_rate = (stats["parse_ok"] / attempted) if attempted else 0.0
    print(
        f"Done. attempted={attempted} parse_ok={stats['parse_ok']} "
        f"({parse_rate:.1%}) refusal={stats['refusal']} "
        f"errors={stats['errors']} wall_s={elapsed:.1f} out={out_jsonl}"
    )


def main() -> None:
    models_cfg = models_config()
    default_gemini = models_cfg.get("gemini_target", "gemini-3.1-pro-preview")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_path",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--provider", choices=("gemini", "claude"), default="gemini")
    ap.add_argument("--model", default=None, help=f"Default: from models.yaml (*_target)")
    ap.add_argument("--arm", choices=("raw", "neutralized"), default="raw")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=49, help="Inclusive; Pilot A default n=50")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    ap.add_argument("--project", default=None)
    ap.add_argument("--location", default=None)
    ap.add_argument("--out_jsonl", default=None)
    args = ap.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    # Silence unused default in help path
    _ = default_gemini
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
