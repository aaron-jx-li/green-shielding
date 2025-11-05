import argparse
import json
import os
import sys
import time
from typing import Dict, Any, Iterable, List, Tuple
from openai import OpenAI
from tqdm import tqdm

CLIENT = OpenAI()

SYSTEM_PROMPT = """You are a meticulous evaluator for a physician Q&A dataset.

Given:
- instruction: dataset meta-instruction shown to the model that produced the answer
- input: the patient's question
- output: the model's answer

Decide TWO THINGS:

A) IS_MEDICAL_DIAGNOSIS (yes/no):
   "yes" if the patient's INPUT is primarily asking for a medical diagnosis
   or confirmation/denial of a suspected diagnosis or cause (e.g., “Do these
   symptoms mean X?”, “Is it Y?”, “What is likely causing this?”).
   It is "no" for clearly non-diagnostic topics (insurance, paperwork, general
   wellness tips without a diagnostic request, administrative/medication-logistics only, etc.).

B) HAS_CLEAR_ANSWER (yes/no):
   "yes" if the OUTPUT clearly addresses the diagnostic question by:
     - stating a likely diagnosis or most likely cause, OR
     - clearly affirming/denying the patient's suspected diagnosis (e.g., “unlikely appendicitis”).
   "no" if the OUTPUT is vague, only gives generic safety disclaimers, or provides possibilities
   without a main conclusion, or sidesteps the diagnostic question.
   Disclaimers are fine, but there must be a clear diagnostic take (e.g., “most consistent with viral diarrhea”)
   or an explicit yes/no on the patient’s hypothesis.

Return STRICT JSON with:
{
  "is_medical_diagnosis": "yes" | "no",
  "has_clear_answer": "yes" | "no",
  "confidence": 1-5,   // 1=low, 5=high
  "rationale": "short, concrete reasoning (<= 3 sentences)"
}
"""

def load_records(path: str) -> Iterable[Dict[str, Any]]:
    """
    Load either a JSON array file or a JSONL file.
    Yields dicts with keys: instruction, input, output (if present).
    """
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]

def call_judge(model: str, record: Dict[str, Any], max_retries: int = 6, base_delay: float = 1.0) -> Dict[str, Any]:
    """
    Query gpt-4.1 judge with exponential backoff on rate limits/transients.
    Returns parsed JSON.
    """
    user_content = {
        "instruction": record.get("instruction", ""),
        "input": record.get("input", ""),
        "output": record.get("output", ""),
    }

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = CLIENT.responses.create(
                model=model,  # e.g., "gpt-4.1"
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Evaluate this example:\n{json.dumps(user_content, ensure_ascii=False)}"
                    },
                ],
                # You can raise temperature slightly if you want more decisive judging.
                temperature=0.0,
            )

            # The Responses API puts the model’s final text in resp.output_text
            text = resp.output_text.strip()
            # The judge is instructed to return strict JSON
            return json.loads(text)

        except Exception as e:
            last_err = e
            # Basic backoff on 429/5xx; immediate fail on JSON parse errors is handled above.
            sleep_s = base_delay * (2 ** attempt)
            time.sleep(sleep_s)

    # If we get here, all retries failed
    raise RuntimeError(f"Judge call failed after {max_retries} attempts: {last_err}")

def normalize_label(val: Any) -> str:
    s = str(val).strip().lower()
    return "yes" if s in {"y", "yes", "true", "1"} else "no"

def judge_file(in_path: str, out_path: str, model: str, max_examples: int = 0) -> None:
    """
    Run the judge over the dataset and write JSONL with added fields:
    is_medical_diagnosis, has_clear_answer, confidence, rationale
    """
    n = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        samples = list(load_records(in_path))
        # print(samples[0])
        results = []
        for rec in tqdm(samples):
            if max_examples and n >= max_examples:
                break
            result = call_judge(model, rec)
            rec_out = {
                **rec,
                "is_medical_diagnosis": normalize_label(result.get("is_medical_diagnosis", "no")),
                "has_clear_answer": normalize_label(result.get("has_clear_answer", "no")),
                "confidence": int(result.get("confidence", 3)),
                "rationale": result.get("rationale", "").strip(),
            }
            results.append(rec_out)
            out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n += 1
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {n} judged examples to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input data (JSON array or JSONL)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output JSONL with judgments")
    ap.add_argument("--model", default="gpt-4.1", help="Judge model (default: gpt-4.1)")
    ap.add_argument("--max", dest="max_examples", type=int, default=0,
                    help="Max examples to process (0 = all)")
    args = ap.parse_args()

    judge_file(args.in_path, args.out_path, args.model, args.max_examples)

if __name__ == "__main__":
    main()