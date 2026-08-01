#!/usr/bin/env python3
"""
Cheap probe: split diagnostic consensus into leading vs can't-miss (n=30).

Runs Gemini with separate high/low fields so we can measure how often they disagree.
Also writes a clinician labeling CSV for the same 30 inquiries.
Does not change the frozen main task card in prompts.py.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from management_reasoning.clients import gemini_client
from management_reasoning.clients.vertex import get_location, get_project, make_vertex_client
from management_reasoning.config import models_config
from management_reasoning.prompts import (
    CONSENSUS_OPTIONS,
    TRIAGE_OPTIONS,
    format_triage_options_block,
    build_user_message,
)
from management_reasoning.schema import ParsedResponse, _strip_fences, _as_bool, _canonicalize_care_seeking

PROBE_N = 30


def build_split_a_instruction() -> str:
    triage_block = format_triage_options_block()
    return f"""You are a careful clinical reasoning assistant.

You will be given a patient inquiry. The inquiry may contain the patient's own questions;
use it only as clinical context and answer only the structured items below.

Answer ALL of the following questions, in this order:

1. Based on the patient inquiry, what is the most likely diagnosis?
2. Based on the patient inquiry, what level of consensus do you expect among clinicians
   regarding the most likely (leading) diagnosis?
   high = most clinicians would agree on a similar leading diagnosis;
   low = reasonable clinicians could favor meaningfully different leading diagnoses.
3. Based on the patient inquiry, what level of consensus do you expect among clinicians
   regarding the safety-critical (can't-miss) diagnoses that should be considered?
   high = most clinicians would agree on a similar can't-miss set;
   low = reasonable clinicians could disagree on which can't-miss diagnoses belong on the list.
4. Based on the patient inquiry, how should the patient seek medical attention?
5. Based on the patient inquiry, what is the first question you would ask the patient to
   rule out a can't-miss diagnosis?
6. Based on the patient inquiry, what level of consensus do you expect among clinicians
   regarding immediate next management steps (for example: what to ask next, which tests
   or imaging to obtain, whether to treat empirically, observe, or refer)?
   high = most clinicians would pursue a similar next-step plan;
   low = reasonable clinicians could choose meaningfully different next steps.

Response rules:
- Output a single JSON object only (no markdown fences, no commentary outside JSON).
- Use exactly these keys:
{{
  "diagnosis": <string>,
  "diagnostic_consensus_leading": "high" or "low",
  "diagnostic_consensus_cantmiss": "high" or "low",
  "care_seeking": <one of the triage strings below>,
  "cant_miss_ruling_out_question": <string>,
  "next_steps_consensus": "high" or "low",
  "refusal": false
}}
- Consensus fields must be exactly high or low (lowercase).
- "care_seeking" must be exactly one of:
{triage_block}
- If you refuse, set "refusal" to true and all other fields to null.
""".strip()


SPLIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": ["string", "null"]},
        "diagnostic_consensus_leading": {
            "type": ["string", "null"],
            "enum": list(CONSENSUS_OPTIONS) + [None],
        },
        "diagnostic_consensus_cantmiss": {
            "type": ["string", "null"],
            "enum": list(CONSENSUS_OPTIONS) + [None],
        },
        "care_seeking": {"type": ["string", "null"], "enum": list(TRIAGE_OPTIONS) + [None]},
        "cant_miss_ruling_out_question": {"type": ["string", "null"]},
        "next_steps_consensus": {
            "type": ["string", "null"],
            "enum": list(CONSENSUS_OPTIONS) + [None],
        },
        "refusal": {"type": "boolean"},
    },
    "required": [
        "diagnosis",
        "diagnostic_consensus_leading",
        "diagnostic_consensus_cantmiss",
        "care_seeking",
        "cant_miss_ruling_out_question",
        "next_steps_consensus",
        "refusal",
    ],
    "additionalProperties": False,
}


def parse_split_response(raw_text: str) -> ParsedResponse:
    if raw_text is None or not str(raw_text).strip():
        return ParsedResponse(False, None, "empty response", False)
    try:
        obj = json.loads(_strip_fences(str(raw_text)))
    except json.JSONDecodeError as e:
        return ParsedResponse(False, None, f"invalid JSON: {e}", False)
    if not isinstance(obj, dict):
        return ParsedResponse(False, None, "JSON root must be an object", False)

    needed = [
        "diagnosis",
        "diagnostic_consensus_leading",
        "diagnostic_consensus_cantmiss",
        "care_seeking",
        "cant_miss_ruling_out_question",
        "next_steps_consensus",
        "refusal",
    ]
    missing = [k for k in needed if k not in obj]
    if missing:
        return ParsedResponse(False, None, f"missing keys: {missing}", False)

    refusal = _as_bool(obj.get("refusal"))
    if refusal is None:
        return ParsedResponse(False, None, "refusal must be a boolean", False)
    if refusal:
        parsed = {k: obj.get(k) for k in needed}
        parsed["refusal"] = True
        return ParsedResponse(True, parsed, None, True)

    def _consensus(field: str) -> Optional[str]:
        val = obj.get(field)
        if not isinstance(val, str):
            return None
        canon = val.strip().casefold()
        return canon if canon in CONSENSUS_OPTIONS else None

    leading = _consensus("diagnostic_consensus_leading")
    cantmiss = _consensus("diagnostic_consensus_cantmiss")
    next_steps = _consensus("next_steps_consensus")
    if leading is None:
        return ParsedResponse(False, None, "diagnostic_consensus_leading must be high|low", False)
    if cantmiss is None:
        return ParsedResponse(False, None, "diagnostic_consensus_cantmiss must be high|low", False)
    if next_steps is None:
        return ParsedResponse(False, None, "next_steps_consensus must be high|low", False)

    dx = obj.get("diagnosis")
    q = obj.get("cant_miss_ruling_out_question")
    care = _canonicalize_care_seeking(obj.get("care_seeking"))
    if not isinstance(dx, str) or not dx.strip():
        return ParsedResponse(False, None, "diagnosis must be non-empty", False)
    if not isinstance(q, str) or not q.strip():
        return ParsedResponse(False, None, "cant_miss_ruling_out_question must be non-empty", False)
    if care is None:
        return ParsedResponse(False, None, "care_seeking must match TRIAGE_OPTIONS", False)

    parsed = {
        "diagnosis": dx.strip(),
        "diagnostic_consensus_leading": leading,
        "diagnostic_consensus_cantmiss": cantmiss,
        "care_seeking": care,
        "cant_miss_ruling_out_question": q.strip(),
        "next_steps_consensus": next_steps,
        "refusal": False,
    }
    return ParsedResponse(True, parsed, None, False)


async def _run_arm(
    *,
    data: List[Dict[str, Any]],
    arm: str,
    out_jsonl: str,
    model: str,
    project: str,
    location: str,
    end_idx: int,
    concurrency: int,
) -> None:
    # Temporarily patch constrained-decoding schema (imported into gemini_client).
    import management_reasoning.clients.gemini_client as gc
    import management_reasoning.schema as schema_mod

    old_gc_schema = gc.RESPONSE_JSON_SCHEMA
    old_schema = schema_mod.RESPONSE_JSON_SCHEMA
    gc.RESPONSE_JSON_SCHEMA = SPLIT_SCHEMA
    schema_mod.RESPONSE_JSON_SCHEMA = SPLIT_SCHEMA

    instruction = build_split_a_instruction()
    client = make_vertex_client(project=project, location=location)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    async def one(i: int) -> None:
        sample = data[i]
        inquiry = sample["raw_input" if arm == "raw" else "neutralized_prompt"]
        record: Dict[str, Any] = {
            "sample_id": int(sample["sample_id"]),
            "arm": arm,
            "provider": "gemini",
            "model": model,
            "probe": "split_diagnostic_consensus",
            "raw_response": None,
            "parsed": None,
            "parse_ok": False,
            "error": None,
            "refusal": False,
        }
        async with sem:
            try:
                raw, usage = await gemini_client.generate_async(
                    instruction,
                    build_user_message(inquiry),
                    model,
                    client=client,
                    project=project,
                    location=location,
                )
                record["raw_response"] = raw
                record["usage"] = usage or None
                parsed = parse_split_response(raw)
                record["parse_ok"] = parsed.parse_ok
                record["parsed"] = parsed.parsed
                record["error"] = parsed.error
                record["refusal"] = parsed.refusal
            except Exception as e:
                record["error"] = f"generate failed: {e}"
        async with lock:
            with open(out_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        await asyncio.gather(*(one(i) for i in range(0, end_idx + 1)))
    finally:
        gc.RESPONSE_JSON_SCHEMA = old_gc_schema
        schema_mod.RESPONSE_JSON_SCHEMA = old_schema


def write_clinician_csv(data: List[Dict[str, Any]], path: str, n: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "raw_input",
                "neutralized_prompt",
                "reference_diagnosis",
                "clinician_leading_consensus_high_low",
                "clinician_cantmiss_consensus_high_low",
                "notes",
            ],
        )
        w.writeheader()
        for i in range(n):
            s = data[i]
            w.writerow(
                {
                    "sample_id": s["sample_id"],
                    "raw_input": s["raw_input"],
                    "neutralized_prompt": s["neutralized_prompt"],
                    "reference_diagnosis": s.get("reference_diagnosis") or "",
                    "clinician_leading_consensus_high_low": "",
                    "clinician_cantmiss_consensus_high_low": "",
                    "notes": "",
                }
            )


def summarize(path: str) -> None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if r.get("parse_ok") and r.get("parsed")]
    print(f"\n=== {path} ===")
    print(f"n={len(rows)} parse_ok={len(ok)}")
    if not ok:
        return
    lead = Counter(r["parsed"]["diagnostic_consensus_leading"] for r in ok)
    miss = Counter(r["parsed"]["diagnostic_consensus_cantmiss"] for r in ok)
    print("leading", dict(lead))
    print("cantmiss", dict(miss))
    disagree = sum(
        1
        for r in ok
        if r["parsed"]["diagnostic_consensus_leading"]
        != r["parsed"]["diagnostic_consensus_cantmiss"]
    )
    print(
        f"leading != cantmiss: {disagree}/{len(ok)} ({100 * disagree / len(ok):.0f}%)"
    )
    both_high = sum(
        1
        for r in ok
        if r["parsed"]["diagnostic_consensus_leading"] == "high"
        and r["parsed"]["diagnostic_consensus_cantmiss"] == "high"
    )
    both_low = sum(
        1
        for r in ok
        if r["parsed"]["diagnostic_consensus_leading"] == "low"
        and r["parsed"]["diagnostic_consensus_cantmiss"] == "low"
    )
    print(f"both_high={both_high} both_low={both_low}")
    print("disagree examples:")
    n = 0
    for r in ok:
        a = r["parsed"]["diagnostic_consensus_leading"]
        b = r["parsed"]["diagnostic_consensus_cantmiss"]
        if a == b:
            continue
        n += 1
        if n <= 8:
            print(
                f"  id={r['sample_id']}: leading={a} cantmiss={b} | "
                f"dx={r['parsed']['diagnosis'][:80]}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input_path",
        default="./results/management_reasoning/data/hcm_full_inputs.json",
    )
    ap.add_argument("--n", type=int, default=PROBE_N)
    ap.add_argument("--arm", choices=("raw", "neutralized", "both"), default="both")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument(
        "--out_dir",
        default="./results/management_reasoning/responses/vertex/gemini-3.1-pro-preview/split_a_probe",
    )
    ap.add_argument(
        "--clinician_csv",
        default="./results/management_reasoning/data/clinician_split_a_labels_n30.csv",
    )
    args = ap.parse_args()

    with open(args.input_path, encoding="utf-8") as f:
        data = json.load(f)

    write_clinician_csv(data, args.clinician_csv, args.n)
    print(f"Wrote clinician labeling sheet: {args.clinician_csv}")

    cfg = models_config()
    model = cfg.get("gemini_target", "gemini-3.1-pro-preview")
    project = get_project()
    location = get_location(cfg.get("location_default", "global"))
    end_idx = args.n - 1

    arms = ["raw", "neutralized"] if args.arm == "both" else [args.arm]

    async def _all() -> None:
        for arm in arms:
            out = os.path.join(args.out_dir, arm, "responses.jsonl")
            print(f"Running split-A probe: arm={arm} n={args.n} model={model}")
            await _run_arm(
                data=data,
                arm=arm,
                out_jsonl=out,
                model=model,
                project=project,
                location=location,
                end_idx=end_idx,
                concurrency=args.concurrency,
            )
            summarize(out)

    asyncio.run(_all())


if __name__ == "__main__":
    main()
