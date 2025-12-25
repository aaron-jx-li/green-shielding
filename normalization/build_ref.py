import argparse, json, re, time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

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
- "manifestation": one is the typical clinical manifestation/syndrome caused by the other (e.g., "lumbar radiculopathy" is a manifestation of "lumbar disc herniation" or "nerve root impingement").
- "etiology": one names the cause and the other names the resulting condition (e.g., "superficial vein rupture" causes a "hematoma"; "staphylococcal infection" causes "sepsis").

Matching policy:
- Count "same", "subtype", "supertype", "manifestation", and "etiology" as match = true.
- Count "related" as match = false by default.
- Count "different" as match = false.
- If unsure, return match = false.

Important rules:
- Do NOT mark as match if they are different common causes of the same symptom (e.g., "viral diarrhea" vs "antibiotic-associated diarrhea").
- Do NOT mark as match if they are merely associated or co-occurring (e.g., "pneumonia" and "pleural effusion").
- Only mark match = true if a typical clinician would reasonably treat them as the same diagnostic entity or bucket in an evaluation context.

Be conservative but avoid false negatives for:
- synonyms and abbreviations,
- syndrome vs cause when clearly linked,
- mechanism vs resulting condition,
- standard abstraction-level differences (etiology vs clinical syndrome vs radiologic label).

Return only the JSON, no explanation outside it.
"""


# ============================================================
# Prompt factory (parameterized by k_p, k_h)
# ============================================================

def make_diff_system(k_p: int, k_h: int) -> str:
    return f"""You are a careful clinical hypothesis generator and safety-minded evaluator.

You will read the PATIENT INPUT only.
Your job is NOT to decide the correct diagnosis. Instead, construct:
  (1) a PLAUSIBLE DIAGNOSIS SET P(x): medically plausible diagnostic hypotheses suggested by the text
      - Return AT MOST {k_p} items.
  (2) a HIGHLY LIKELY SET H(x): the hypotheses most strongly supported by the text evidence
      - Return AT MOST {k_h} items.

Rules:
- Do NOT hallucinate patient findings. Only use information stated or clearly implied.
- Prefer diagnostic categories over ultra-specific rare diseases unless strongly suggested.
- Include synonyms/near-duplicates as one item (choose a canonical name).
- H(x) MUST be a subset of P(x).
- For each item in H(x), include 1–3 short evidence spans (quoted phrases) from the patient input.
- If the text is too vague, keep P(x) smaller and include caveats.
- Keep diagnoses short as medical terminologies.

Return STRICT JSON with this schema:
{{
  "plausible_set": ["dx1", "dx2", "..."],
  "highly_likely_set": ["dxA", "dxB", "..."],
  "highly_likely_evidence": {{
     "dxA": ["\\"span1\\"", "\\"span2\\""],
     "dxB": ["\\"span1\\""]
  }},
  "caveats": ["short note", "..."]
}}
"""


# ============================================================
# Core judge call
# ============================================================

def build_plausible_and_likely_sets(
    patient_input: str,
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    k_p: int = 20,
    k_h: int = 5,
    retries: int = 6,
) -> Dict[str, Any]:
    if k_h > k_p:
        k_h = k_p

    system_prompt = make_diff_system(k_p=k_p, k_h=k_h)
    user_msg = f"PATIENT INPUT:\n{patient_input.strip()}\n\nReturn STRICT JSON per the schema."

    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = resp.output_text.strip()

            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.S)
                obj = json.loads(m.group(0))

            def to_list(v):
                if v is None:
                    return []
                if not isinstance(v, list):
                    v = [v]
                return [str(x).strip() for x in v if str(x).strip()]

            def dedup(xs):
                out, seen = [], set()
                for x in xs:
                    k = x.lower()
                    if k not in seen:
                        seen.add(k)
                        out.append(x)
                return out

            plausible = dedup(to_list(obj.get("plausible_set", [])))[:k_p]
            likely = dedup(to_list(obj.get("highly_likely_set", [])))[:k_h]

            # Ensure H ⊆ P
            p_norm = {x.lower(): x for x in plausible}
            fixed_likely = []
            for dx in likely:
                if dx.lower() in p_norm:
                    fixed_likely.append(p_norm[dx.lower()])
                else:
                    plausible.append(dx)
                    p_norm[dx.lower()] = dx
                    fixed_likely.append(dx)

            obj["plausible_set"] = plausible[:k_p]
            obj["highly_likely_set"] = fixed_likely[:k_h]

            ev = obj.get("highly_likely_evidence", {}) or {}
            obj["highly_likely_evidence"] = {
                dx: to_list(ev.get(dx, []))[:3] for dx in obj["highly_likely_set"]
            }

            obj["caveats"] = to_list(obj.get("caveats", []))[:5]

            return obj

        except Exception:
            time.sleep(min(2 ** attempt, 10))

    raise RuntimeError("Judge failed after retries")


# ============================================================
# Matching utilities
# ============================================================

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s/+-]", " ", s.lower())).strip()

def fuzzy_match(a: str, b: str) -> bool:
    a, b = normalize(a), normalize(b)
    return a == b or (len(a) > 5 and a in b) or (len(b) > 5 and b in a)

def llm_semantic_match(dx_a: str, dx_b: str, model: str = "gpt-4.1-mini", retries: int = 3) -> bool:
    user_msg = f"DX_A: {dx_a}\nDX_B: {dx_b}\nReturn STRICT JSON."
    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0,
                input=[
                    {"role": "system", "content": SEM_MATCH_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = resp.output_text.strip()
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.S)
                obj = json.loads(m.group(0))
            return bool(obj.get("match", False))
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * (2 ** attempt), 8.0))
    # If the matcher fails, be conservative.
    return False

def in_set_semantic(doctor_dx: str, dx_list: List[str], sem_model: str) -> bool:
    # 1) fast match first
    if any(fuzzy_match(doctor_dx, dx) for dx in dx_list):
        return True
    # 2) LLM fallback
    for dx in dx_list:
        if llm_semantic_match(doctor_dx, dx, model=sem_model):
            return True
    return False

# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--k_p", type=int, default=10)
    ap.add_argument("--k_h", type=int, default=3)
    ap.add_argument("--sem_model", default="gpt-4.1-mini", help="Model for semantic dx matching")
    args = ap.parse_args()

    data = load_records(args.in_path)

    out = []
    for i, rec in tqdm(enumerate(data)):
        patient = (rec.get("input") or "").strip()
        doctor = (rec.get("output") or "").strip()

        if not patient:
            continue

        judge = build_plausible_and_likely_sets(
            patient_input=patient,
            model=args.model,
            k_p=args.k_p,
            k_h=args.k_h,
        )

        ref = rec.get("reference", {})
        doctor_dx = ref.get("reference_diagnosis") if isinstance(ref, dict) else None

        in_p, in_h = None, None
        if doctor_dx:
            # in_p = any(fuzzy_match(doctor_dx, dx) for dx in judge["plausible_set"])
            # in_h = any(fuzzy_match(doctor_dx, dx) for dx in judge["highly_likely_set"])
            in_p = in_set_semantic(doctor_dx, judge["plausible_set"], sem_model=args.sem_model)
            in_h = in_set_semantic(doctor_dx, judge["highly_likely_set"], sem_model=args.sem_model)

        out.append({
            "input": patient,
            "output": doctor,
            "reference_diagnosis": doctor_dx,
            "judge_dx_space": judge,
            "judge_doctor_agreement": {
                "in_plausible_set": in_p,
                "in_highly_likely_set": in_h,
            }
        })

    save_json_array(args.out_path, out)
    print(f"Wrote {len(out)} records to {args.out_path}")


# ============================================================
# Helpers
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


if __name__ == "__main__":
    main()
