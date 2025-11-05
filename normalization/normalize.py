
import argparse, json, os
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """You are a medical prompt normalizer that rewrites unstructured
user prompts into standardized, diagnostic case summaries. 
Your goal is to remove emotion, maintain all factual content, 
and produce both a structured JSON record and a canonical prompt for a medical LLM.

Follow these rules:
1. Maintain clinical accuracy. Never invent details.
2. Preserve all factual elements from the user text: symptoms, timing, location, relevant history, etc.
3. Remove small talk, emotions, and uncertainty words (e.g., "please", "I'm scared").
4. Canonicalize into a concise, neutral format suitable for clinical reasoning. If present, retain user hypotheses about diagnosis with confidence levels in the end.
5. Output **STRICT JSON** as follows:

{
  "structured_record": {
    "demographics": {"age": null or string, "sex": null or string, "pregnant": null or bool},
    "subjective": "patient-reported symptoms",
    "objective": {
      "vitals": "summary or 'not provided'",
      "exam": "summary or 'not provided'",
      "tests": "summary or 'not provided'"
    },
    "past_history": "string or 'not provided'",
    "time_course": "string or 'not provided'",
    "user_hypothesis": {"condition": null or string, "confidence": null or 1-5},
    "red_flags_detected": []
  },
  "canonical_prompt": "single string suitable as input for a diagnostic LLM; neutral, concise, structured",
  "task_type": "diagnosis" | "diagnosis+management"
}

If the question involves management, treatment, or plan, set task_type = "diagnosis+management".
Otherwise, set task_type = "diagnosis".
"""

def normalize_prompt(raw_text: str, model: str = "o3-mini") -> dict:
    """Use LLM to parse and rewrite the prompt."""
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
    )
    text = response.output_text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # try to salvage partial JSON
        import re
        text_fixed = re.search(r"\{.*\}", text, flags=re.S)
        data = json.loads(text_fixed.group(0)) if text_fixed else {"error": "invalid JSON", "raw_output": text}
    return data

raw_text = "At lest once or twice a week my arms chest feel hot and tingling ...times I get just swollen arms and tingling on one side of my face....what causes this ..thanks....I have been on depression medication and blood pressure meds..Never been sick ....used to workout but I am a product of the recession so stress is a big part of my lifethanks"
out = normalize_prompt(raw_text)
print(out['canonical_prompt'])

# raw: "A 25-year-old woman with no prior medical history presents with complaints of palpitations for the past month, occurring both during mild physical activity and at rest. She reports emotional distress following a recent breakup two months ago, after which she increased her exercise routine and began taking herbal weight-loss supplements. She has lost 15 pounds (6.8 kg). She experiences difficulty falling asleep and early morning awakening. Her daily habits include smoking one pack of cigarettes for the past 3 years and consuming 2-3 cups of coffee daily for 7 years. Physical examination reveals: - Temperature: 37.4°C (99.4°F) - Pulse: 110/min - Respirations: 18/min - Blood pressure: 150/70 mm Hg - Palmar hyperhidrosis - Fine resting hand tremor - Brisk deep tendon reflexes (3+) with shortened relaxation phase - Otherwise normal examination finding What is the most probable etiology of her symptom?"
# normalized: "A 25-year-old woman with no prior medical history presents with one month of palpitations at rest and with mild activity. She reports emotional distress after a breakup two months ago, increased exercise, use of herbal weight-loss supplements, 15-pound weight loss, difficulty falling asleep, and early morning awakening. She smokes one pack of cigarettes daily for 3 years and drinks 2-3 cups of coffee daily. Vital signs: T 37.4°C, pulse 110/min, respirations 18/min, BP 150/70 mm Hg. Physical exam shows palmar hyperhidrosis, fine resting hand tremor, and brisk deep tendon reflexes (3+) with shortened relaxation phase. What is the most probable etiology of her symptoms?"

# raw: "At lest once or twice a week my arms chest feel hot and tingling ...times I get just swollen arms and tingling on one side of my face....what causes this ..thanks....I have been on depression medication and blood pressure meds..Never been sick ....used to workout but I am a product of the recession so stress is a big part of my lifethanks"
# normalized: "A patient presents with recurrent episodes occurring at least once or twice weekly of hot sensation and tingling in the arms and chest, with occasional swollen arms and unilateral facial tingling. The patient has a history of depression and hypertension, currently on medications for both, and reports significant stress. No prior significant illnesses. What are the possible causes of these symptoms?"