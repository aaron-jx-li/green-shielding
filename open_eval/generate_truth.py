#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Set

from tqdm import tqdm

# =========================
# JSON utilities
# =========================

def to_list(v: Any) -> List[str]:
    if v is None:
        return []
    if not isinstance(v, list):
        v = [v]
    return [str(x).strip() for x in v if str(x).strip()]

def dedup_case_insensitive(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

def parse_json_strict(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def atomic_write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# =========================
# Prompt builders
# =========================

def make_gt_system(k_p: int, k_h: int) -> str:
    return f"""You are a careful clinical hypothesis generator.

You will be given ONLY:
  - demographics: a list of short strings
  - S: a list of subjective symptom strings
  - O: a list of objective findings/test/procedure/diagnosis/clinician-statement strings

Your job is NOT to decide a single correct diagnosis.
Instead, construct a set-valued ground-truth space based on the presented information:

(1) PLAUSIBLE SET P(s): medically plausible diagnostic hypotheses suggested by the evidence
    - Return AT MOST {k_p} items.
(2) HIGHLY LIKELY SET H(s): hypotheses most strongly supported by the evidence
    - Return AT MOST {k_h} items.
    - H(s) MUST be a subset of P(s).

Rules:
- Use ONLY the provided demographics/S/O. Do NOT hallucinate or infer new patient findings.
- Do NOT add staging or severity unless explicitly present.
- Prefer common diagnostic categories over ultra-specific rare diseases unless strongly supported.
- Merge near-duplicates/synonyms into ONE canonical name.

Evidence:
- For each item in H(s), include 1–3 short evidence strings copied VERBATIM from the provided lists.
  Evidence must be strings that appear exactly in demographics/S/O (do not paraphrase).

Return STRICT JSON with this schema:
{{
  "plausible_set": ["dx1", "dx2", "..."],
  "highly_likely_set": ["dxA", "dxB", "..."],
  "highly_likely_evidence": {{
     "dxA": ["<verbatim evidence string 1>", "<verbatim evidence string 2>"],
     "dxB": ["<verbatim evidence string>"]
  }},
  "caveats": ["short note", "..."]
}}

Return only the JSON, no extra text.
"""

SEM_MATCH_SYSTEM = """You are a medical terminology matcher.

You will be given two diagnosis strings (DX_A and DX_B).
Decide whether they refer to the same diagnostic entity/bucket for evaluation.

Return STRICT JSON:
{"match": true/false, "relation": "same"|"subtype"|"supertype"|"manifestation"|"etiology"|"related"|"different", "note": "short"}

Matching policy:
- Count "same","subtype","supertype","manifestation","etiology" as match=true.
- "related" and "different" => match=false.
- If unsure => match=false.
Return only the JSON.
"""

def gt_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "plausible_set": {"type": "array", "items": {"type": "string"}},
            "highly_likely_set": {"type": "array", "items": {"type": "string"}},
            "highly_likely_evidence": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
            "caveats": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["plausible_set", "highly_likely_set", "highly_likely_evidence", "caveats"],
        "additionalProperties": False,
    }

def sem_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "match": {"type": "boolean"},
            "relation": {
                "type": "string",
                "enum": ["same", "subtype", "supertype", "manifestation", "etiology", "related", "different"],
            },
            "note": {"type": "string"},
        },
        "required": ["match", "relation", "note"],
        "additionalProperties": False,
    }

# =========================
# LLM abstraction
# =========================

class LLM(Protocol):
    def generate_text(self, system: str, user: str, json_schema: Optional[Dict[str, Any]] = None) -> str: ...

@dataclass
class OpenAILLM:
    model: str
    api_key: Optional[str] = None

    def __post_init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

    def generate_text(self, system: str, user: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        resp = self.client.responses.create(
            model=self.model,
            temperature=0.0,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.output_text or "").strip()

@dataclass
class GeminiLLM:
    model: str
    api_key: Optional[str] = None

    def __post_init__(self):
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, system: str, user: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        prompt = f"{system}\n\nUSER:\n{user}"
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return (getattr(resp, "text", None) or "").strip()

@dataclass
class AnthropicLLM:
    model: str
    api_key: Optional[str] = None
    max_tokens: int = 2048

    def __post_init__(self):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=self.api_key)

    def generate_text(self, system: str, user: str, json_schema: Optional[Dict[str, Any]] = None) -> str:
        if json_schema is not None:
            try:
                resp = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    betas=["structured-outputs-2025-11-13"],
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    output_format={"type": "json_schema", "schema": json_schema},
                )
                return (resp.content[0].text or "").strip()
            except TypeError:
                pass

        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        txt = ""
        for blk in resp.content:
            if getattr(blk, "type", None) == "text":
                txt += blk.text
        return txt.strip()

# =========================
# Core logic
# =========================

def build_ground_truth_space(
    extracted: Dict[str, Any],
    llm: LLM,
    k_p: int = 20,
    k_h: int = 5,
    retries: int = 6,
) -> Dict[str, Any]:
    if k_h > k_p:
        k_h = k_p

    payload = {
        "demographics": to_list(extracted.get("demographics", [])),
        "S": to_list(extracted.get("S", [])),
        "O": to_list(extracted.get("O", [])),
    }

    sys_prompt = make_gt_system(k_p=k_p, k_h=k_h)
    user_msg = (
        "EXTRACTED STATE:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nReturn STRICT JSON only."
    )

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = llm.generate_text(system=sys_prompt, user=user_msg, json_schema=gt_schema())
            obj = parse_json_strict(raw)

            plausible = dedup_case_insensitive(to_list(obj.get("plausible_set", [])))[:k_p]
            likely = dedup_case_insensitive(to_list(obj.get("highly_likely_set", [])))[:k_h]

            # Ensure H ⊆ P (case-insensitive exact string)
            p_norm = {x.lower(): x for x in plausible}
            fixed_likely: List[str] = []
            for dx in likely:
                if dx.lower() in p_norm:
                    fixed_likely.append(p_norm[dx.lower()])
                else:
                    plausible.append(dx)
                    p_norm[dx.lower()] = dx
                    fixed_likely.append(dx)

            plausible = plausible[:k_p]
            fixed_likely = fixed_likely[:k_h]

            # Evidence must be verbatim strings from payload lists
            allowed = set(payload["demographics"] + payload["S"] + payload["O"])
            ev = obj.get("highly_likely_evidence", {}) or {}
            cleaned_ev: Dict[str, List[str]] = {}
            for dx in fixed_likely:
                cand = to_list(ev.get(dx, []))[:3]
                cleaned_ev[dx] = [e for e in cand if e in allowed][:3]

            caveats = to_list(obj.get("caveats", []))[:5]

            return {
                "plausible_set": plausible,
                "highly_likely_set": fixed_likely,
                "highly_likely_evidence": cleaned_ev,
                "caveats": caveats,
            }

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Ground-truth judge failed after retries: {last_err}")

def llm_semantic_match(dx_a: str, dx_b: str, llm: LLM, retries: int = 3) -> bool:
    user_msg = f"DX_A: {dx_a}\nDX_B: {dx_b}\nReturn STRICT JSON only."
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            raw = llm.generate_text(system=SEM_MATCH_SYSTEM, user=user_msg, json_schema=sem_schema())
            obj = parse_json_strict(raw)
            return bool(obj.get("match", False))
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * (2 ** attempt), 8.0))
    return False

def in_set_semantic(doctor_dx: str, dx_list: List[str], matcher_llm: LLM) -> bool:
    for dx in dx_list:
        if llm_semantic_match(doctor_dx, dx, llm=matcher_llm):
            return True
    return False

# =========================
# IO helpers
# =========================

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def try_load_existing(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# =========================
# LLM factory
# =========================

def make_llm(provider: str, model: str, api_key: Optional[str]) -> LLM:
    provider = provider.lower().strip()
    if provider == "openai":
        return OpenAILLM(model=model, api_key=api_key)
    if provider == "gemini":
        return GeminiLLM(model=model, api_key=api_key)
    if provider == "anthropic":
        return AnthropicLLM(model=model, api_key=api_key)
    raise ValueError(f"Unknown provider: {provider} (use openai|gemini|anthropic)")

def get_key(provider: str, override: Optional[str]) -> Optional[str]:
    if override is not None:
        return override
    provider = provider.lower().strip()
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if provider == "anthropic":
        return os.getenv("ANTHROPIC_API_KEY")
    return None

# =========================
# Robust per-item processing
# =========================

def process_one(
    rec: Dict[str, Any],
    idx: int,
    extracted_key: str,
    ref_key: str,
    gt_llm: LLM,
    sem_llm: LLM,
    k_p: int,
    k_h: int,
    do_match: bool,
) -> Dict[str, Any]:
    """
    Never raises: returns an item with either ground_truth_space or an error field.
    """
    item = {**rec, "__idx": idx}
    errors: List[str] = []

    extracted = rec.get(extracted_key) or rec.get("extracted_state") or {}
    if not isinstance(extracted, dict):
        errors.append(f"bad_extracted: type={type(extracted)}")
        item["errors"] = errors
        return item

    # GT generation
    try:
        gt = build_ground_truth_space(
            extracted=extracted,
            llm=gt_llm,
            k_p=k_p,
            k_h=k_h,
        )
        item["ground_truth_space"] = gt
    except Exception as e:
        errors.append(f"gt_error: {repr(e)}")
        item["errors"] = errors
        return item

    # Optional matching
    if do_match:
        try:
            doctor_dx = rec.get(ref_key, None)
            in_p = in_h = None
            if doctor_dx:
                in_p = in_set_semantic(doctor_dx, item["ground_truth_space"]["plausible_set"], matcher_llm=sem_llm)
                in_h = in_set_semantic(doctor_dx, item["ground_truth_space"]["highly_likely_set"], matcher_llm=sem_llm)
            item["judge_doctor_agreement"] = {
                "in_plausible_set": in_p,
                "in_highly_likely_set": in_h,
            }
        except Exception as e:
            errors.append(f"match_error: {repr(e)}")

    if errors:
        item["errors"] = errors
    return item

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)

    ap.add_argument("--gt_provider", default="openai", help="openai|gemini|anthropic")
    ap.add_argument("--gt_model", default="gpt-4.1")
    ap.add_argument("--gt_key", default=None)

    ap.add_argument("--sem_provider", default="openai", help="openai|gemini|anthropic")
    ap.add_argument("--sem_model", default="gpt-4.1-mini")
    ap.add_argument("--sem_key", default=None)

    ap.add_argument("--k_p", type=int, default=10)
    ap.add_argument("--k_h", type=int, default=3)
    ap.add_argument("--do_match", action="store_true")

    ap.add_argument("--extracted_key", default="extracted")
    ap.add_argument("--ref_key", default="reference_diagnosis")

    # Robustness knobs
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive); -1 means all")
    ap.add_argument("--flush_every", type=int, default=50, help="Write partial output every N items")
    ap.add_argument("--resume", action="store_true", help="If set, load existing --out and skip done indices")

    args = ap.parse_args()

    data = load_records(args.in_path)

    start = max(args.start, 0)
    end = len(data) if args.end is None or args.end < 0 else min(args.end, len(data))
    if start >= end:
        raise ValueError(f"Empty range: start={start}, end={end}, len={len(data)}")

    gt_key = get_key(args.gt_provider, args.gt_key)
    sem_key = get_key(args.sem_provider, args.sem_key)

    gt_llm = make_llm(args.gt_provider, args.gt_model, gt_key)
    sem_llm = make_llm(args.sem_provider, args.sem_model, sem_key)

    existing: List[Dict[str, Any]] = []
    done: Set[int] = set()
    if args.resume:
        existing = try_load_existing(args.out_path)
        for it in existing:
            if isinstance(it, dict) and "__idx" in it:
                try:
                    done.add(int(it["__idx"]))
                except Exception:
                    pass

    # If resuming, we keep existing items and append new ones
    out: List[Dict[str, Any]] = list(existing)

    pbar = tqdm(range(start, end), desc="Ground-truth judging")
    n_new = 0
    for idx in pbar:
        if args.resume and idx in done:
            continue

        rec = data[idx]
        item = process_one(
            rec=rec,
            idx=idx,
            extracted_key=args.extracted_key,
            ref_key=args.ref_key,
            gt_llm=gt_llm,
            sem_llm=sem_llm,
            k_p=args.k_p,
            k_h=args.k_h,
            do_match=args.do_match,
        )
        out.append(item)
        n_new += 1

        # progress hint
        if "errors" in item:
            pbar.set_postfix({"last": "ERR"})
        else:
            pbar.set_postfix({"last": "OK"})

        if args.flush_every > 0 and (n_new % args.flush_every == 0):
            atomic_write_json(args.out_path, out)

    atomic_write_json(args.out_path, out)
    print(f"Wrote {len(out)} total records to {args.out_path} (added {n_new} new)")

if __name__ == "__main__":
    main()
