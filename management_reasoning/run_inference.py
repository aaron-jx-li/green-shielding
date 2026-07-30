#!/usr/bin/env python3
"""Sync Vertex inference for management_reasoning (smoke / small runs)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Set

from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.clients import claude_vertex_client, gemini_client
from management_reasoning.clients.vertex import VertexConfigError, get_location, get_project
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


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    models_cfg = models_config()
    default_gemini = models_cfg.get("gemini_target", "gemini-3-pro-preview")
    default_claude = models_cfg.get("claude_target", "claude-opus-4-5")
    default_location = models_cfg.get("location_default", "global")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_path",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Default: results/management_reasoning/responses/vertex/{model}/{arm}",
    )
    ap.add_argument("--provider", choices=("gemini", "claude"), default="gemini")
    ap.add_argument("--model", default=None, help="Defaults from models.yaml (*_target)")
    ap.add_argument("--arm", choices=("raw", "neutralized"), default="raw")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=9, help="Inclusive end index (smoke default n=10)")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    ap.add_argument("--no_skip_existing", action="store_false", dest="skip_existing")
    ap.add_argument("--project", default=None)
    ap.add_argument("--location", default=None)
    ap.add_argument("--out_jsonl", default=None, help="Override output JSONL path")
    args = ap.parse_args()

    model = args.model or (default_claude if args.provider == "claude" else default_gemini)
    if args.provider == "claude":
        default_location = models_cfg.get("claude_location_default") or default_location
    location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or default_location
    # Avoid awkward @ in output directory names.
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
    generate_fn = gemini_client.generate if args.provider == "gemini" else claude_vertex_client.generate

    print(
        f"Vertex sync: provider={args.provider} model={model} project={project} "
        f"location={location} arm={args.arm} range=[{start},{end}] out={out_jsonl}"
    )

    for i in tqdm(range(start, end + 1), desc="inference"):
        sample = data[i]
        sample_id = int(sample.get("sample_id", i))
        if args.skip_existing and sample_id in done:
            continue

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
        try:
            user_msg = build_user_message(inquiry)
            raw = generate_fn(
                instruction,
                user_msg,
                model,
                project=project,
                location=location,
            )
            record["raw_response"] = raw
            parsed = parse_model_response(raw)
            record["parse_ok"] = parsed.parse_ok
            record["parsed"] = parsed.parsed
            record["error"] = parsed.error
            record["refusal"] = parsed.refusal
        except VertexConfigError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            record["error"] = f"generate failed: {e}"

        _append_jsonl(out_jsonl, record)
        done.add(sample_id)

    print(f"Done. Wrote results to {out_jsonl}")


if __name__ == "__main__":
    main()
