#!/usr/bin/env python3
"""
Factor-selective prompt neutralization.

This script extends your original neutralizer (which removed all factors at once) so the user can
choose which *broad categories* to neutralize: {content, format, tone}. You can specify 1–3.

It uses the 7 factor list:

Content:
  F1 mentions_specific_guess - user mentions a specific diagnosis guess
  F2 contains_irrelevant_details - includes tangential personal life details
  F3 lack_of_objective_data - NO vitals, exam findings, or test results (mark false if ANY exist)
  F4 lack_of_symptom_history - NO onset/duration/evolution (mark false if ANY timing exists, even vague)
Format:
  F5 unstructured_question_format - MULTIPLE DISTINCT asks only, NOT informal phrasing
Tone:
  F6 emotional_or_urgent_tone - worry/anxiety/panic OR urgency/severity emphasis (merged)
  F7 first_person_perspective - raw input uses first-person ("I", "my")

Important notes:
- "lack_of_objective_data" (F3) and "lack_of_symptom_history" (F4) describe what the user did NOT provide.
  You cannot truly "remove" these without inventing facts. Neutralization never adds new tests/history.
  Selecting CONTENT for neutralization only removes *belief/speculation* and *irrelevant details*; it will
  NOT fabricate missing clinical info.
- Use fine-grained criteria: F4 should be FALSE if ANY timing info exists (even "recently" or "a while").
  F5 should be TRUE only for multiple distinct asks, not informal phrasing.
  F6 should be TRUE only for explicit emotional/urgent language, not mild concern.
- Tone controls perspective: when tone neutralization is True, output is third-person clinical narrative.
  When tone neutralization is False, output stays in first-person (patient's voice).
"""

import argparse, json, re, time
from typing import Any, Dict, List, Optional, Union, Set
from tqdm import tqdm
from openai import OpenAI

client = OpenAI()

# --------------------------- Clinical extraction (unchanged) ---------------------------

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

VERIFY_SYSTEM_PROMPT = """You verify that a neutralized clinical prompt corresponds to an extracted clinical representation.

You will be given:
- extracted_state: JSON with keys demographics, S, O (lists of atomic facts)
- neutralized_prompt: a clinical case summary (may be first-person or third-person) followed by a diagnostic question

Your job:
1) Ensure every clinical fact in neutralized_prompt appears in extracted_state (no new facts).
2) Ensure all clinically relevant facts in extracted_state are represented in neutralized_prompt (no omissions),
   except that stylistic rephrasing and summarization is allowed if facts are preserved.
3) Allow rewording, tense changes, order changes, and perspective changes (first-person vs third-person).
4) If the neutralized prompt mentions a diagnosis, it must be explicitly present in extracted_state (e.g., in O).

Return STRICT JSON ONLY:
{
  "is_consistent": true/false,
  "added_facts": [ ... ],
  "missing_facts": [ ... ],
  "notes": "short explanation"
}
"""

# --------------------------- Paper factors (only) ---------------------------

PAPER_FACTOR_KEYS = [
    "mentions_specific_guess",          # F1
    "contains_irrelevant_details",      # F2
    "lack_of_objective_data",           # F3
    "lack_of_symptom_history",          # F4
    "unstructured_question_format",     # F5
    "emotional_or_urgent_tone",         # F6 (merged: worry/anxiety + urgency/severity)
    "first_person_perspective",         # F7
]

CATEGORY_TO_PAPER_FACTORS: Dict[str, Set[str]] = {
    "content": {
        "mentions_specific_guess",
        "contains_irrelevant_details",
        "lack_of_objective_data",
        "lack_of_symptom_history",
    },
    "format": {
        "unstructured_question_format",
    },
    "tone": {
        "emotional_or_urgent_tone",
        "first_person_perspective",
    },
}

# --------------------------- Prompt builder ---------------------------

def build_neutralizer_system_prompt(remove_categories: Set[str]) -> str:
    """
    Build a system prompt that selectively neutralizes the chosen categories.

    remove_categories: subset of {"content", "format", "tone"}.
    """
    remove_content = "content" in remove_categories
    remove_format = "format" in remove_categories
    remove_tone = "tone" in remove_categories

    # Pre-compute what perspective and style the output should use
    if remove_tone:
        perspective_instruction = (
            "Write the neutralized prompt in THIRD-PERSON clinical narrative "
            "(e.g., \"A patient presents with...\", \"The patient reports...\")."
        )
    else:
        perspective_instruction = (
            "Write the neutralized prompt in FIRST-PERSON from the patient's perspective "
            "(e.g., \"I have been experiencing...\", \"My symptoms include...\"). "
            "Maintain the patient's own voice. Do NOT convert to third-person."
        )

    if remove_format:
        format_instruction = (
            "Consolidate all questions into a single clean diagnosis-only query. "
            "End with exactly: \"What is the most likely diagnosis?\""
        )
    else:
        format_instruction = (
            "Preserve the original question structure. If the user asked multiple things "
            "(diagnosis + treatment + diet + reassurance), keep ALL of those asks. "
            "Do NOT consolidate into a single diagnosis-only question. "
            "Do NOT append \"What is the most likely diagnosis?\" at the end."
        )

    return f"""
You are a medical expert and a reliable annotator. Your job is to rewrite an unstructured
raw user prompt into a neutralized version AND annotate 7 binary factors.

CRITICAL OUTPUT RULES (read these FIRST):
- Perspective: {perspective_instruction}
- Question format: {format_instruction}

Primary goals (always):
1. Maintain clinical accuracy. Never invent clinical facts.
2. Preserve all factual symptom descriptions, timelines, and any user-mentioned prior diagnoses.
3. Clean up grammar and spelling for readability, but respect the perspective and format rules above.
4. Produce output exclusively as a strict JSON object.

Selected categories to neutralize (REMOVE):
- content = {remove_content}  (if False: preserve speculations, guesses, irrelevant details)
- format  = {remove_format}  (if False: preserve original question structure and mixed asks)
- tone    = {remove_tone}  (if False: keep first-person, preserve worry/urgency language)

How to apply selective neutralization:

A) CONTENT (specific guess, irrelevant details, missingness cues)
- If content = True:
  - Remove non-medical irrelevant life details.
  - If the user mentions a specific guess, do NOT include that guess in the neutralized prompt.
  - Do NOT include belief/speculation language beyond factual symptoms.
- If content = False:
  - Preserve these cues in a standardized way without endorsing them.
    {"Third-person example: 'The patient wonders if this could be X.'" if remove_tone else "First-person example: 'I wonder if this could be X.'"}

Important: lack_of_objective_data and lack_of_symptom_history reflect missing information in the raw input.
You MUST NOT add tests, vitals, or symptom history that are not present, regardless of settings.

B) FORMAT (question structure and ending)
- If format = True:
  - Consolidate into a single clean diagnosis-only query.
  - End with exactly: "What is the most likely diagnosis?"
- If format = False:
  - Preserve the original question structure, even if it mixes multiple asks (diagnosis + treatment + 
    reassurance) or is unstructured. Do NOT consolidate into a single clean question.
  - Preserve the original ending question(s) as closely as possible. Do NOT append a clean 
    diagnosis-only question at the end.

C) TONE (worry, urgency, AND perspective)
- If tone = True:
  - Remove emotional language and urgency framing.
  - Rewrite in THIRD-PERSON clinical narrative. Clean grammar, clinical style.
- If tone = False:
  - Keep in FIRST-PERSON perspective. This is mandatory — do NOT use third-person.
  - Preserve worried/anxious language and urgency framing from the original.
  - Clean up grammar for readability, but maintain the patient's own voice and emotional language.

-------------------------------------------------------------------------------
DETAILED FACTOR DEFINITIONS (for annotation)
-------------------------------------------------------------------------------

F1) mentions_specific_guess
   - True if the user mentions a specific diagnosis guess or asks if the diagnosis could be a 
     specific outcome, even implicitly.
   - Example: "could this be diabetes?", "I think it might be appendicitis"

F2) contains_irrelevant_details
   - True if the prompt includes personal life details clearly unrelated to the medical scenario 
     (e.g., job/finance/relationship background) that do not meaningfully contribute to diagnosis.
   - Be conservative; only mark true if clearly tangential.

F3) lack_of_objective_data
   - True if NO vitals, exam findings, or test results are provided.
   - If ANY objective data appears (e.g., "blood pressure 140/90", "CT scan showed...", 
     "temperature 101°F"), mark false.

F4) lack_of_symptom_history
   - True if NO clear onset, duration, or symptom evolution is described.
   - If ANY timing information exists (e.g., "started yesterday", "for 3 months", "getting worse"), 
     mark false.
   - Be precise: vague statements like "recently" or "a while" still count as timing information.

F5) unstructured_question_format
   - True ONLY if the user explicitly asks for MULTIPLE DISTINCT things in the same message, such as:
     diagnosis + treatment + diet advice, or diagnosis + reassurance + triage guidance.
   - Example (True): "what could be the reason for this elevated SGPT and will I get clearance for 
     joining new company? what should I do to lower my SGPT levels? and what diet should I follow?"
     (asks for diagnosis + employment advice + treatment + diet = multiple distinct asks)
   - Example (False): "would you possibly know what it could be or might be?" 
     (single ask for diagnosis, even if phrased informally)
   - Example (False): "what could this be?" (single diagnostic ask)
   - Informal or casual phrasing does NOT make it unstructured. The key criterion is whether 
     the user asks for multiple distinct types of information.

F6) emotional_or_urgent_tone
   - True if the user expresses ANY of the following:
     * Strong fear, anxiety, panic, or emotionally urgent distress 
       ("I'm really scared", "I'm freaking out", "this is terrifying", "I'm panicking")
     * Urgent or severe danger notions ("emergency", "urgent", "life-threatening", 
       "could I die?", "is this very serious?")
     * Extreme severity emphasis beyond factual description ("unbearable pain", 
       "worst pain of my life", "can't function", "the pain brings tears to my eyes")
   - False for mild concern, polite worry, or factual severity descriptions without emotional emphasis.
   - Be strict: the language must go beyond simple reporting of symptoms.

F7) first_person_perspective
   - True if the raw input is written from the patient's own perspective using first-person 
     pronouns ("I", "my", "me").
   - False if the raw input is already in third-person or written by someone else about the patient
     (e.g., "my daughter has..." is first-person from a caregiver, still mark true).
   - This factor simply records the perspective of the raw input, regardless of neutralization.

-------------------------------------------------------------------------------
JSON OUTPUT FORMAT (STRICT)
-------------------------------------------------------------------------------
{{
  "neutralized_prompt": "the neutralized diagnostic query (first-person or third-person depending on tone setting)",
  "paper_factors": {{
      "mentions_specific_guess": true/false,
      "contains_irrelevant_details": true/false,
      "lack_of_objective_data": true/false,
      "lack_of_symptom_history": true/false,
      "unstructured_question_format": true/false,
      "emotional_or_urgent_tone": true/false,
      "first_person_perspective": true/false
  }},
  "removed_categories": ["content"/"format"/"tone"...]
}}

Rules:
- paper_factors must be filled using ONLY evidence from the raw input (not your rewritten version).
- Apply the detailed factor definitions strictly. For example:
  * F4 (lack_of_symptom_history): mark false even if minimal timing exists (e.g., "recently")
  * F5 (unstructured_question_format): mark true ONLY for multiple distinct asks, NOT for informal phrasing
  * F6 (emotional_or_urgent_tone): only mark true for explicit emotional/urgent language beyond factual reporting
  * F3 (lack_of_objective_data): mark false if ANY objective data exists
  * F7 (first_person_perspective): always reflects the raw input, not the rewritten version
- removed_categories must exactly match the selected removal categories passed to you.
- Output MUST be valid JSON, no commentary, no extra keys.

MANDATORY CHECKS (your output will be rejected if these fail):
- tone={remove_tone}: {"neutralized_prompt MUST use third-person (he/she/the patient). NO first-person pronouns (I/my/me)." if remove_tone else "neutralized_prompt MUST use first-person (I/my/me). NO third-person (the patient/he/she). This is NON-NEGOTIABLE."}
- format={remove_format}: {"neutralized_prompt MUST end with exactly 'What is the most likely diagnosis?'" if remove_format else "neutralized_prompt MUST preserve the original question(s). Do NOT end with 'What is the most likely diagnosis?' unless the raw input itself asked exactly that."}
"""


# --------------------------- Helpers ---------------------------

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

def _json_load_best_effort(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

def _validate_paper_factors(obj: Dict[str, Any]) -> None:
    pf = obj.get("paper_factors")
    if not isinstance(pf, dict):
        raise ValueError("paper_factors must be a dict.")
    missing = set(PAPER_FACTOR_KEYS) - set(pf.keys())
    extra = set(pf.keys()) - set(PAPER_FACTOR_KEYS)
    if missing:
        raise ValueError(f"paper_factors missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"paper_factors has unexpected keys: {sorted(extra)}")
    for k in PAPER_FACTOR_KEYS:
        if not isinstance(pf[k], bool):
            raise ValueError(f"paper_factors[{k}] must be boolean.")

# --------------------------- LLM calls ---------------------------

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
            text = _strip_code_fences(_get_output_text(resp))
            obj = _json_load_best_effort(text)
            required = {"demographics", "S", "O"}
            if not isinstance(obj, dict):
                raise ValueError("Extractor did not return a JSON object.")
            missing = required - set(obj.keys())
            if missing:
                raise ValueError(f"Extractor missing keys: {sorted(missing)}")
            extra = set(obj.keys()) - required
            if extra:
                raise ValueError(f"Extractor unexpected top-level keys: {sorted(extra)}")
            if not all(isinstance(obj[k], list) for k in ["demographics", "S", "O"]):
                raise ValueError("Extractor: demographics, S, O must all be lists.")
            for k in ["demographics", "S", "O"]:
                for i, item in enumerate(obj[k]):
                    if not isinstance(item, str):
                        raise ValueError(f"Extractor: {k}[{i}] is not a string.")
            return obj
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** attempt))
    raise RuntimeError(f"Extractor failed after {max_retries} attempts: {last_err}")

def neutralize_prompt_selective(
    raw_text: str,
    remove_categories: Set[str],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    Selectively neutralize the raw prompt according to remove_categories.
    Returns a dict with keys: neutralized_prompt, paper_factors, removed_categories.
    """
    sys_prompt = build_neutralizer_system_prompt(remove_categories)
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": sys_prompt},
            {"role": "user", "content": raw_text},
        ],
    )
    text = _strip_code_fences(_get_output_text(resp))
    try:
        data = _json_load_best_effort(text)
    except Exception:
        return {"error": "invalid JSON", "raw_output": text}

    if not isinstance(data, dict):
        return {"error": "output not a JSON object", "raw_output": text}

    # Ensure required keys exist
    data.setdefault("removed_categories", sorted(list(remove_categories)))

    # Best-effort validation; if missing keys, return debug-friendly error
    required_keys = {"neutralized_prompt", "paper_factors", "removed_categories"}
    if required_keys - set(data.keys()):
        return {
            "error": f"missing required keys: {sorted(list(required_keys - set(data.keys())))}",
            "raw_output": text,
        }

    try:
        _validate_paper_factors(data)
    except Exception as e:
        return {"error": f"invalid paper_factors: {e}", "raw_output": text}

    return data

def verify_neutralized_prompt(
    neutralized_prompt: str,
    extracted_state: Dict[str, Any],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    payload = {
        "neutralized_prompt": neutralized_prompt,
        "extracted_state": extracted_state,
    }
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    text = _strip_code_fences(_get_output_text(resp))
    try:
        return _json_load_best_effort(text)
    except Exception:
        return {
            "is_consistent": False,
            "added_facts": [],
            "missing_facts": [],
            "notes": "verification returned invalid JSON",
        }

# --------------------------- I/O ---------------------------

def load_json(path: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Selective neutralization by category: content/format/tone.")
    p.add_argument("--input_path", type=str, required=True, help="Path to input JSON (list or single object).")
    p.add_argument("--output_path", type=str, required=True, help="Path to output JSON.")
    p.add_argument(
        "--remove",
        type=str,
        required=True,
        help="Comma-separated categories to remove: content,format,tone (choose 1-3). Example: content,tone",
    )
    p.add_argument("--neutralize_model", type=str, default="gpt-4.1-mini", help="Model for selective neutralization.")
    p.add_argument("--extract_model", type=str, default="gpt-4.1-mini", help="Model for clinical extraction.")
    p.add_argument("--verify_model", type=str, default="gpt-4.1-mini", help="Model for verifier.")
    p.add_argument("--max_verify_retries", type=int, default=2, help="Retries to re-neutralize if verification fails.")
    p.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all samples).")
    return p.parse_args()

def parse_remove_categories(arg: str) -> Set[str]:
    cats = {c.strip().lower() for c in arg.split(",") if c.strip()}
    allowed = {"content", "format", "tone"}
    bad = cats - allowed
    if bad:
        raise ValueError(f"Unknown categories in --remove: {sorted(bad)}. Allowed: {sorted(allowed)}")
    if not (1 <= len(cats) <= 3):
        raise ValueError("--remove must include 1 to 3 categories among: content, format, tone.")
    return cats

# --------------------------- Main ---------------------------

def main():
    args = parse_args()
    remove_categories = parse_remove_categories(args.remove)

    data = load_json(args.input_path)
    if isinstance(data, dict):
        samples = [data]
        single_input = True
    else:
        samples = data
        single_input = False

    # Limit number of samples if specified
    if args.num_samples is not None and args.num_samples > 0:
        samples = samples[:args.num_samples]
        print(f"Processing {len(samples)} samples (limited by --num_samples={args.num_samples})")
    else:
        print(f"Processing all {len(samples)} samples")

    outputs: List[Dict[str, Any]] = []

    for _, sample in tqdm(list(enumerate(samples))):
        # Extract raw_input - this is the only field we neutralize
        raw_text = sample.get("raw_input", "") or sample.get("input", "")
        if not raw_text:
            print(f"Warning: Sample {_} has no raw_input or input field, skipping.")
            continue
        
        # Preserve original_output for the output
        original_output = sample.get("original_output") or sample.get("output")

        # Extract clinical state if not already present
        extracted_state = sample.get("extracted") or sample.get("extracted_state")
        if extracted_state is None:
            extracted_state = call_extractor(model=args.extract_model, raw_input=raw_text)

        # Perform selective neutralization
        out = neutralize_prompt_selective(
            raw_text,
            remove_categories=remove_categories,
            model=args.neutralize_model,
        )

        if out.get("error"):
            # Simplified error output
            outputs.append({
                "raw_input": raw_text,
                "neutralized_prompt": None,
                "original_output": original_output,
                "paper_factors": {},
                "neutralize_error": out.get("error"),
            })
            continue

        # Verify the neutralized prompt
        verification = verify_neutralized_prompt(
            neutralized_prompt=str(out.get("neutralized_prompt", "")),
            extracted_state=extracted_state,
            model=args.verify_model,
        )

        # Retry if verification fails
        retries = 0
        while not verification.get("is_consistent", False) and retries < args.max_verify_retries:
            retries += 1
            out = neutralize_prompt_selective(
                raw_text,
                remove_categories=remove_categories,
                model=args.neutralize_model,
            )
            if out.get("error"):
                break
            verification = verify_neutralized_prompt(
                neutralized_prompt=str(out.get("neutralized_prompt", "")),
                extracted_state=extracted_state,
                model=args.verify_model,
            )

        # Build simplified output with only essential fields
        outputs.append({
            "raw_input": raw_text,
            "neutralized_prompt": out.get("neutralized_prompt"),
            "original_output": original_output,
            "paper_factors": out.get("paper_factors", {}),
        })

    final_output: Union[Dict[str, Any], List[Dict[str, Any]]]
    if single_input and len(outputs) == 1:
        final_output = outputs[0]
    else:
        final_output = outputs

    save_json(final_output, args.output_path)
    print(f"Saved selective-neutralization outputs to {args.output_path}")

if __name__ == "__main__":
    main()
