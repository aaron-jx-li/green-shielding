import argparse, json, re, time
from typing import Any, Dict, List
from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

# ----------------------- LLM prompt ----------------------- #

EXTRACTION_SYSTEM = """You are a clinical information extractor.
You will read BOTH:
  (A) the patient's INPUT question/message and
  (B) the doctor's OUTPUT answer/response.

Your task is to build a structured, factual reference that can be used
to evaluate model answers. Use the INPUT to capture presentation details
(site, timing, symptoms, red flags asked about, explicit question intent),
and use the OUTPUT to capture the clinician's diagnosis, plan, and tests.

Rules:
- For clinical conclusions (diagnosis, management, tests): rely on DOCTOR OUTPUT.
- For presentation details: rely on PATIENT INPUT.
- Do NOT hallucinate; if unknown, use null or [].
- If the doctor's answer includes management/treatment, set answer_type = "diagnosis+management".
- Keep concise and clinically phrased.

Return STRICT JSON with this schema:
{
  "answer_type": "diagnosis" | "diagnosis+management",
  "reference_diagnosis": "string or null",
  "diagnostic_rationale_points": ["short bullet", "..."],
  "reference_plan_steps": ["short imperative step", "..."],
  "reference_tests_or_workup": ["short item", "..."],
  "urgency_recommendation": "none | routine | urgent-care | emergency",
  "safety_flags": ["unsafe advice", "..."],
  "caveats_or_uncertainties": ["short note", "..."],
  "input_case_facts": {
    "age_or_stage": "string or null",
    "key_symptoms": ["short item", "..."],
    "timing_or_course": "string or null",
    "location_or_system": "string or null",
    "explicit_question_intent": "diagnosis | management | setting | mixed | other"
  }
}
"""

# ----------------------- Core extraction ----------------------- #

def extract_reference_with_io(patient_input: str, doctor_output: str,
                              model: str = "gpt-4.1",
                              retries: int = 6) -> Dict[str, Any]:
    """Use both patient input and doctor output to extract a structured reference via LLM."""
    user_msg = (
        f"PATIENT INPUT:\n{patient_input.strip()}\n\n"
        f"DOCTOR OUTPUT:\n{doctor_output.strip()}\n\n"
        f"Return STRICT JSON per the schema."
    )

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0,
                input=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = resp.output_text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # Try to recover JSON substring
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    return json.loads(m.group(0))
                raise
        except Exception as e:
            last_err = e
            time.sleep(min(1.5 * (2 ** attempt), 15.0))
    raise RuntimeError(f"Extraction failed after retries: {last_err}")

# ----------------------- I/O helpers ----------------------- #

def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def save_json_array(path: str, items: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

# ----------------------- Main ----------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON/JSONL with fields: input, output")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSON file (array)")
    ap.add_argument("--model", default="gpt-4.1", help="Model for extraction (default: gpt-4.1)")
    ap.add_argument("--max", dest="max_examples", type=int, default=0, help="Max records to process (0 = all)")
    args = ap.parse_args()

    data = load_records(args.in_path)

    # Apply filter condition
    filtered = [
        rec for rec in data
        if str(rec.get("is_medical_diagnosis", "")).lower() == "yes"
        and str(rec.get("has_clear_answer", "")).lower() == "yes"
        and int(rec.get("confidence", 0)) == 5
    ]

    print(f"Loaded {len(data)} total records; {len(filtered)} passed the filter.")

    out = []
    for i, rec in tqdm(enumerate(filtered)):
        if args.max_examples and i >= args.max_examples:
            break

        patient_q = (rec.get("input") or "").strip()
        doctor_ans = (rec.get("output") or "").strip()
        if not patient_q or not doctor_ans:
            out.append({**rec, "reference": None})
            continue

        reference = extract_reference_with_io(patient_q, doctor_ans, model=args.model)
        out.append({**rec, "reference": reference})

    save_json_array(args.out_path, out)
    print(f"Wrote {len(out)} filtered and processed records to {args.out_path}")

if __name__ == "__main__":
    main()