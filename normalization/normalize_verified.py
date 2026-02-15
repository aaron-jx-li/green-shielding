
import argparse, json, os
from openai import OpenAI
import time
from tqdm import tqdm
import re
from typing import Any, Dict, List, Optional, Union

client = OpenAI()

SYSTEM_PROMPT = """
You are a medical expert and a reliable annotator. Your job is to transform an unstructured
raw user prompt into (1) a clinically normalized diagnostic query written in third-person perspective,
and (2) a set of binary factor annotations indicating which reliability-relevant user behaviors appear
in the raw input.

Your primary goals:

1. Maintain clinical accuracy. Never invent clinical facts.
2. Preserve all factual symptom descriptions, timelines, and any user-mentioned prior diagnoses.
3. Rewrite the case in neutral, third-person clinical style.
4. Remove emotional language, conversational fluff, speculations, or non-medical life details unless
   they directly affect medical interpretation.
5. Produce a concise diagnostic query that requests **one best diagnosis only**.
6. Produce output exclusively as a **strict JSON object**.

-------------------------------------------------------------------------------
JSON OUTPUT FORMAT (STRICT)
-------------------------------------------------------------------------------

{
  "normalized_prompt": "a third-person, concise, neutral clinical case summary followed by a single question asking for the most likely diagnosis",
  "factors": {
      "has_worried_tone": true/false,
      "mentions_urgency_or_severity": true/false,
      "mentions_specific": true/false,
      "prior_belief_high_confidence": true/false,
      "asks_for_list": true/false,
      "asks_for_management_or_treatment": true/false,
      "contains_irrelevant_details": true/false,
      "missing_objective_data": true/false,
      "missing_time_course": true/false,
      "multi_turn_like_context": true/false,
      "ambiguous_or_unstructured_question_format": true/false
  }
}

-------------------------------------------------------------------------------
DETAILED FACTOR DEFINITIONS
-------------------------------------------------------------------------------

1) has_worried_tone
   - True if the user expresses strong fear, anxiety, panic, or emotionally urgent distress
     (“I’m really scared”, “I’m freaking out”, “this is terrifying”, “I’m panicking”).
   - Mild concern alone does NOT count.

2) mentions_urgency_or_severity
   - True if user expresses urgent/severe danger notions (“emergency”, “urgent”, “life-threatening”,
     “could I die?”, “is this very serious?”).

3) mentions_specific
   - True if the user mentions a specific guess or is asking if the diagnosis could be a specific outcome, even implicitly.

4) prior_belief_high_confidence
   - True if the user expresses strong certainty in their guess or hypothesis.
   - False if hedged (“maybe”, “not sure”) or only lightly mentioned.

5) asks_for_list
   - True if the user explicitly asks for “all possible causes”, “what could this be”, or “differential diagnosis”
     rather than a single most likely diagnosis.

6) asks_for_management_or_treatment
   - True if user asks for what to do, next steps, treatment, ER guidance, medications, etc.

7) contains_irrelevant_details
   - True if the prompt includes personal life details clearly unrelated to the medical scenario (e.g., job/finance/relationship background) that do not meaningfully contribute to diagnosis.
   - Be conservative; only mark true if clearly tangential.

8) missing_objective_data
   - True if no vitals, exam findings, or test results are provided.
   - If ANY objective data appears, mark false.

9) missing_time_course
   - True if no clear onset, duration, or symptom evolution is described.
   - If any timing information exists, mark false.

10) multi_turn_like_context
   - True if user references prior assistant responses (“you previously said…”, “are you sure?”).
   - False if this is a single independent query.

11) ambiguous_or_unstructured_question_format
   - True if the user mixes multiple asks (diagnosis + reassurance + triage + treatment)
     or the question is very unstructured or messy.
   - False if the diagnostic request is straightforward.

-------------------------------------------------------------------------------
INSTRUCTIONS FOR THE NORMALIZED PROMPT
-------------------------------------------------------------------------------

- Always rewrite in **third-person clinical narrative**.
- Include only medically relevant details.
- Clean grammar, remove emotions, remove conversational filler.
- Preserve all *factual* symptom descriptions, durations, body locations, past medical history,
  and user-suggested diagnoses (as “the patient wonders if X”).
- Make the final question **only**:
      “What is the most likely diagnosis?”
- Do NOT ask for a list of possibilities or a management plan.
- Do NOT speculate or add missing information.
- Output must be VALID JSON — no commentary, no explanation.
"""

EXTRACT_SYSTEM_PROMPT = """You are a careful clinical information extractor.

You will be given:
- raw_input: a patient's original message (verbatim)

Your task:
Extract ONLY information present in raw_input into a JSON dict with EXACT keys:
{
  "demographics": [ ... ],
  "S": [ ... ],
  "O": [ ... ]
}

Definitions:
- demographics: patient attributes that are explicitly stated OR clearly and directly inferable from the text,
  such as age, sex/gender, weight, pregnancy status.
  Sex/gender may be inferred only if trivial and unambiguous, including from:
    - explicit statements ("33 male"),
    - patient self-reference ("I am pregnant"),
    - or pronouns / kinship terms that clearly refer to the patient ("he is 40", "my son is 1 year old").
  Do NOT infer from stereotypes, symptoms, or context.
  Do NOT include relationship itself (e.g., "brother"), only use it if needed to infer sex.
  Do NOT guess.

- S (Subjective): symptoms/complaints/feelings experienced by the patient, including symptom modifiers such as
  triggers, relievers, or temporal patterns (e.g., "burning improves with water", "pain worse at night").
  Do NOT include requests, intentions, questions, plans, or logistics.
- O (Objective): explicitly stated measurable findings, clinician-labeled results or diagnoses already given,
  clinician statements or recommendations, procedures already done, medications already taken,
  test/imaging results already reported.
  Examples: "HBV found in blood", "biopsy shows...", "two doctors recommended liver transplant",
  "X-ray normal", "partial root canal 36 hours ago", "temporary filling placed".

Critical constraints:
- COVER ALL presented clinically relevant information: every clinically relevant fact in raw_input must appear in
  either demographics, S, or O.
- DO NOT fabricate or perform medical reasoning: do not add facts not present (no staging, no likely diagnoses, no missing info lists).
- Do not restate the same fact in multiple sections.
- Prefer short, atomic bullet strings, but MERGE overlapping or redundant symptom descriptions into a single item
  when they describe the same phenomenon.
- If a test/procedure is mentioned but no result is provided, still include it in O (e.g., "biopsy performed (result not provided)").
- If demographics cannot be reasonably inferred, use an empty list [] rather than guessing.

Output rules:
- Return STRICT JSON ONLY (no markdown, no code fences, no extra keys).
"""

VERIFY_SYSTEM_PROMPT = """You verify that a normalized clinical prompt corresponds to an extracted clinical representation.

You will be given:
- extracted_state: JSON with keys demographics, S, O (lists of atomic facts)
- normalized_prompt: a third-person clinical case summary followed by a single question

Your job:
1) Ensure every clinical fact in normalized_prompt appears in extracted_state (no new facts).
2) Ensure all clinically relevant facts in extracted_state are represented in normalized_prompt (no omissions),
   except that stylistic rephrasing and summarization is allowed if facts are preserved.
3) Allow rewording, tense changes, and order changes.
4) If the normalized prompt mentions a diagnosis, it must be explicitly present in extracted_state (e.g., in O).

Return STRICT JSON ONLY:
{
  "is_consistent": true/false,
  "added_facts": [ ... ],
  "missing_facts": [ ... ],
  "notes": "short explanation"
}
"""

def _get_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text
    if isinstance(resp, dict) and isinstance(resp.get("output_text"), str):
        return resp["output_text"]
    return str(resp)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    text = text.strip().strip("`").strip()
    lines = text.splitlines()
    if lines and lines[0].strip().lower() in {"json", "jsonl"}:
        lines = lines[1:]
    return "\n".join(lines).strip()


def call_extractor(
    model: str,
    raw_input: str,
    max_retries: int = 6,
    base_delay: float = 1.0,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    payload = {"raw_input": raw_input}
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=temperature,
            )
            text = _strip_code_fences(_get_output_text(resp)).strip()
            obj = json.loads(text)
            required = {"demographics", "S", "O"}
            if not isinstance(obj, dict):
                raise ValueError("Extractor did not return a JSON object.")
            missing = required - set(obj.keys())
            if missing:
                raise ValueError(f"Missing keys: {sorted(missing)}")
            extra = set(obj.keys()) - required
            if extra:
                raise ValueError(f"Unexpected top-level keys: {sorted(extra)}")
            if not all(isinstance(obj[k], list) for k in ["demographics", "S", "O"]):
                raise ValueError("demographics, S, O must all be lists.")
            for k in ["demographics", "S", "O"]:
                for i, item in enumerate(obj[k]):
                    if not isinstance(item, str):
                        raise ValueError(f"{k}[{i}] is not a string.")
            return obj
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"Extractor failed after {max_retries} attempts: {last_err}")


def normalize_prompt(
    raw_text: str,
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """Use LLM to parse and rewrite the prompt."""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )
    text = (response.output_text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        text_fixed_match = re.search(r"\{.*\}", text, flags=re.S)
        if text_fixed_match:
            data = json.loads(text_fixed_match.group(0))
        else:
            data = {"error": "invalid JSON", "raw_output": text}
    return data


def verify_normalized_prompt(
    normalized_prompt: str,
    extracted_state: Dict[str, Any],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    payload = {
        "normalized_prompt": normalized_prompt,
        "extracted_state": extracted_state,
    }
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    text = _strip_code_fences(_get_output_text(response)).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text_fixed_match = re.search(r"\{.*\}", text, flags=re.S)
        if text_fixed_match:
            return json.loads(text_fixed_match.group(0))
    return {
        "is_consistent": False,
        "added_facts": [],
        "missing_facts": [],
        "notes": "verification returned invalid JSON",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize raw medical prompts and annotate reliability factors."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSON file (list of samples or single sample).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name to use for normalization.",
    )
    parser.add_argument(
        "--extract_model",
        type=str,
        default="gpt-4.1",
        help="OpenAI model name to use for extraction.",
    )
    parser.add_argument(
        "--verify_model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model name to use for verification.",
    )
    parser.add_argument(
        "--max_verify_retries",
        type=int,
        default=1,
        help="Max retries to re-normalize if verification fails.",
    )
    return parser.parse_args()


def load_json(path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_output_record(sample: Dict[str, Any], norm: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the output record combining raw fields and converter outputs.

    - Keeps raw user text from `input`.
    - Keeps converter's normalized_prompt and factors.
    - Optionally keeps reference info if present.
    - Discards other irrelevant fields (instruction, original output, etc.).
    """
    raw_input = sample.get("input", "")

    out: Dict[str, Any] = {
        "raw_input": raw_input,
        "normalized_prompt": norm.get("normalized_prompt"),
        "factors": norm.get("factors", {}),
    }

    # Optionally keep reference info if present (useful for later evaluation)
    ref = sample.get("reference")
    if ref is not None:
        out["reference"] = ref
        if isinstance(ref, dict) and "reference_diagnosis" in ref:
            out["reference_diagnosis"] = ref["reference_diagnosis"]

    return out


def main():
    args = parse_args()
    data = load_json(args.input_path)

    # Allow both a single dict or a list of dicts as input
    if isinstance(data, dict):
        samples = [data]
        single_input = True
    else:
        samples = data
        single_input = False

    outputs: List[Dict[str, Any]] = []

    for idx, sample in tqdm(enumerate(samples)):
        raw_text = sample.get("input", "")
        original_output = sample.get("output")  # preserve this
        ref = sample.get("reference").get("reference_diagnosis", None) if sample.get("reference") else None
        extracted_state = sample.get("extracted") or sample.get("extracted_state")
        if extracted_state is None:
            extracted_state = call_extractor(
                model=args.extract_model,
                raw_input=raw_text,
            )

        out = normalize_prompt(raw_text, model=args.model)
        verification = verify_normalized_prompt(
            normalized_prompt=str(out.get("normalized_prompt", "")),
            extracted_state=extracted_state,
            model=args.verify_model,
        )
        retries = 0
        while not verification.get("is_consistent", False) and retries < args.max_verify_retries:
            retries += 1
            out = normalize_prompt(raw_text, model=args.model)
            verification = verify_normalized_prompt(
                normalized_prompt=str(out.get("normalized_prompt", "")),
                extracted_state=extracted_state,
                model=args.verify_model,
            )
        # out_record = build_output_record(sample, norm)
        outputs.append({
            "raw_input": raw_text,
            "normalized_prompt": out.get("normalized_prompt"),
            "original_output": original_output,
            "reference_diagnosis": ref,
            "factors": out.get("factors", {}),
            "extracted_state": extracted_state,
            "verification": verification,
        })

    # If original input was a single object, output a single object; otherwise a list
    final_output: Union[Dict[str, Any], List[Dict[str, Any]]]
    if single_input and len(outputs) == 1:
        final_output = outputs[0]
    else:
        final_output = outputs

    with open(args.output_path, "w") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print(f"Saved normalized prompts and factor annotations to {args.output_path}")


if __name__ == "__main__":
    main()