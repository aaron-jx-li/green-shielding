#!/usr/bin/env python3
import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

# ============================================================
# 1) Ground-truth space judge: extracted state -> P(s), H(s)
# ============================================================

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
- Use ONLY the provided demographics/S/O. Do NOT use any other knowledge about the dataset.
- Do NOT hallucinate or infer new patient findings.
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

If the information is too vague, keep the sets smaller and include caveats.
Return only the JSON, no extra text.
"""

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
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))

def build_ground_truth_space(
    extracted: Dict[str, Any],
    model: str = "gpt-4.1",
    temperature: float = 0.0,
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
    user_msg = f"EXTRACTED STATE:\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n\nReturn STRICT JSON."

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            obj = parse_json_strict(resp.output_text)

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
            allowed_evidence = set(payload["demographics"] + payload["S"] + payload["O"])
            ev = obj.get("highly_likely_evidence", {}) or {}

            cleaned_ev: Dict[str, List[str]] = {}
            for dx in fixed_likely:
                cand = to_list(ev.get(dx, []))[:3]
                cleaned_ev[dx] = [e for e in cand if e in allowed_evidence][:3]

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

# ============================================================
# 2) Semantic diagnosis matching (LLM only; no fuzzy match)
# ============================================================

SEM_MATCH_SYSTEM = """You are a medical terminology matcher.

You will be given two diagnosis strings (DX_A and DX_B).
Decide whether they refer to:
  - the same clinical condition,
  - the same diagnostic category,
  - or one is a clinically standard subtype/supertype or manifestation/etiology of the other,
for the purposes of evaluation.

Return STRICT JSON:
{"match": true/false, "relation": "same"|"subtype"|"supertype"|"manifestation"|"etiology"|"related"|"different", "note": "short"}

Definitions:
- "same": synonyms, abbreviations, spelling variants, or equivalent terms (e.g., "H. pylori" vs "Helicobacter pylori infection").
- "subtype": one is a more specific named form of the other (e.g., "viral gastroenteritis" is a subtype of "gastroenteritis").
- "supertype": one is a broader diagnostic category than the other (reverse of subtype).
- "manifestation": one is the typical clinical manifestation/syndrome caused by the other.
- "etiology": one names the cause and the other names the resulting condition.

Matching policy:
- Count "same", "subtype", "supertype", "manifestation", and "etiology" as match = true.
- Count "related" as match = false by default.
- Count "different" as match = false.
- If unsure, return match = false.

Important rules:
- Do NOT mark as match if they are different common causes of the same symptom.
- Do NOT mark as match if they are merely associated or co-occurring.
- Only mark match = true if a typical clinician would reasonably treat them as the same diagnostic entity/bucket.

Return only the JSON, no explanation outside it.
"""

def llm_semantic_match(dx_a: str, dx_b: str, model: str, retries: int = 3) -> bool:
    user_msg = f"DX_A: {dx_a}\nDX_B: {dx_b}\nReturn STRICT JSON."
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0.0,
                input=[
                    {"role": "system", "content": SEM_MATCH_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            obj = parse_json_strict(resp.output_text)
            return bool(obj.get("match", False))
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * (2 ** attempt), 8.0))
    return False

def in_set_semantic(doctor_dx: str, dx_list: List[str], sem_model: str) -> bool:
    for dx in dx_list:
        if llm_semantic_match(doctor_dx, dx, model=sem_model):
            return True
    return False

# ============================================================
# 3) IO helpers
# ============================================================

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def save_json_array(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# ============================================================
# 4) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON array/JSONL that includes `extracted`")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON with ground truth space")
    ap.add_argument("--gt_model", default="gpt-4.1", help="Strong model for ground-truth generation")
    ap.add_argument("--k_p", type=int, default=10)
    ap.add_argument("--k_h", type=int, default=3)
    ap.add_argument("--sem_model", default="gpt-4.1-mini", help="Model for semantic dx matching")
    ap.add_argument("--do_match", action="store_true", help="If set, compare reference_diagnosis to P/H sets")
    ap.add_argument("--extracted_key", default="extracted", help="Key containing extracted dict (default: extracted)")
    ap.add_argument("--ref_key", default="reference_diagnosis", help="Key containing reference dx (default: reference_diagnosis)")
    args = ap.parse_args()

    data = load_records(args.in_path)

    out: List[Dict[str, Any]] = []
    for idx, rec in enumerate(tqdm(data, desc="Ground-truth judging")):
        extracted = rec.get(args.extracted_key) or rec.get("extracted_state") or {}
        if not isinstance(extracted, dict):
            continue

        gt = build_ground_truth_space(
            extracted=extracted,
            model=args.gt_model,
            k_p=args.k_p,
            k_h=args.k_h,
        )

        item = {**rec, "ground_truth_space": gt}

        if args.do_match:
            doctor_dx = rec.get(args.ref_key, None)
            in_p = in_h = None
            if doctor_dx:
                in_p = in_set_semantic(doctor_dx, gt["plausible_set"], sem_model=args.sem_model)
                in_h = in_set_semantic(doctor_dx, gt["highly_likely_set"], sem_model=args.sem_model)
            item["judge_doctor_agreement"] = {
                "in_plausible_set": in_p,
                "in_highly_likely_set": in_h,
            }

        out.append(item)

    save_json_array(args.out_path, out)
    print(f"Wrote {len(out)} records to {args.out_path}")

if __name__ == "__main__":
    main()
